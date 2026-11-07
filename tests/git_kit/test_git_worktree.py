"""The worktree pool's command line, permission check, hook choice and shared directories."""

import subprocess
from types import SimpleNamespace

import pytest

from kit_support import git, run_kit
from llm_agents.git import worktree

SHARED = worktree.SharedDir("assets/gen/models", what="models", items="local models")
HOSTED = worktree.PoolHost(shared=(SHARED,), share_command="share-models")


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setattr(worktree, "HOST", HOSTED)


def _parse(argv):
    return worktree.build_parser().parse_args(argv)


@pytest.mark.parametrize("command", ["land", "next", "release", "drop", "equalize", "share-models"])
def test_a_slot_is_accepted_positionally_and_as_dir(hosted, command):
    """`land --dir wt1` was refused with "unrecognized arguments" while
    `equalize --dir wt1` worked and `equalize wt1` did not (2026-09-13): one
    spelling per command, and never the one the caller reached for."""
    assert worktree.slot_named(_parse([command, "wt1"])) == "wt1"
    assert worktree.slot_named(_parse([command, "--dir", "wt1"])) == "wt1"
    assert worktree.slot_named(_parse([command, "wt1", "--dir", "wt1"])) == "wt1"


def test_equalize_defaults_to_the_current_worktree_and_land_refuses_to_guess():
    assert worktree.slot_named(_parse(["equalize"]), default=".") == "."
    with pytest.raises(worktree.Fail, match="name the slot"):
        worktree.slot_named(_parse(["land"]))


def test_two_different_slots_are_an_error_not_a_pick():
    with pytest.raises(worktree.Fail, match="two different slots"):
        worktree.slot_named(_parse(["land", "wt1", "--dir", "wt2"]))


def test_a_share_command_exists_only_for_a_host_that_shares(hosted, monkeypatch):
    commands = ["status", "new", "equalize", "share-models", "land", "push", "next",
                "release", "drop", "hook-create", "hook-remove"]
    assert list(worktree.build_parser()._subparsers._group_actions[0].choices) == commands
    monkeypatch.setattr(worktree, "HOST", worktree.EMPTY)
    choices = worktree.build_parser()._subparsers._group_actions[0].choices
    assert list(choices) == [c for c in commands if c != "share-models"]


def test_a_host_s_help_texts_replace_the_defaults(monkeypatch):
    def new_help():
        choices = worktree.build_parser()._subparsers._group_actions[0].choices
        return " ".join(choices["new"].format_help().split())

    assert "--no-install skip the host's install step" in new_help()
    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(help={"--no-install": "skip npm"}))
    assert "--no-install skip npm" in new_help()


def test_the_bare_script_says_it_runs_without_host_rules(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "Initial")
    done = run_kit("worktree", "status", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "no host rules given" in done.stderr
    assert "no slot (python -m llm_agents.git.worktree new)" in done.stdout


@pytest.fixture
def settings_tree(tmp_path, monkeypatch):
    main = tmp_path / "main"
    (main / ".claude").mkdir(parents=True)
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr(worktree, "main_dir", lambda: main)
    monkeypatch.setattr(worktree.Path, "home", classmethod(lambda cls: home))

    def write(where, dirs):
        (where / ".claude" / "settings.json").write_text(
            worktree.json.dumps({"permissions": {"additionalDirectories": [str(d) for d in dirs]}}),
            encoding="utf-8")
    return tmp_path, main, home, write


def test_a_slot_counts_as_permitted_only_when_a_settings_file_lists_it(settings_tree):
    """An unlisted slot parks its agent on a permission prompt (2026-10-09)."""
    root, main, home, write = settings_tree
    wt1, wt4 = root / "main-wt1", root / "main-wt4"
    write(main, [wt1])
    assert worktree.is_permitted(wt1)
    assert not worktree.is_permitted(wt4)
    # A sibling whose name merely starts with a listed one is not inside it.
    assert not worktree.is_permitted(root / "main-wt10")
    write(home, [root])
    assert worktree.is_permitted(wt4)
    assert "additionalDirectories" in worktree.permission_warning("wt4", wt4)


@pytest.mark.parametrize("text", ["{not json", '{"permissions": null}',
                                  '{"permissions": {"additionalDirectories": null}}', "[]"])
def test_unreadable_settings_permit_nothing(settings_tree, text):
    root, main, _, _ = settings_tree
    (main / ".claude" / "settings.json").write_text(text, encoding="utf-8")
    assert not worktree.is_permitted(root / "main-wt1")


def _free_pool(monkeypatch, root, names):
    slots = [worktree.Slot(int(n[2:]), root / f"main-{n}", n, "", False) for n in names]
    monkeypatch.setattr(worktree, "list_slots", lambda: slots)
    monkeypatch.setattr(worktree, "take_next_assignment", lambda: "")
    monkeypatch.setattr(worktree, "slot_state", lambda s: worktree.SlotState(
        exists=True, detached=False, locked=False, lock_reason="", clean=True, unlanded=[], behind=0, head=""))
    monkeypatch.setattr(worktree, "claim", lambda s, reason: True)
    return slots


def test_the_hook_never_hands_out_an_unlisted_slot_as_first_free(settings_tree, monkeypatch):
    root, main, _, write = settings_tree
    _free_pool(monkeypatch, root, ["wt1", "wt2"])
    write(main, [root / "main-wt2"])
    slot, _, how = worktree.choose_slot("agent-x", "r")
    assert (slot.name, how) == ("wt2", "first free")
    write(main, [])
    with pytest.raises(worktree.Fail, match="additionalDirectories"):
        worktree.choose_slot("agent-x", "r")
    # Asked for by name, the unlisted slot is handed out (with a warning on stderr).
    slot, _, how = worktree.choose_slot("wt1", "r")
    assert (slot.name, how) == ("wt1", "by name")


# ------------------------------------------------------------ shared dirs

@pytest.fixture
def shared_tree(tmp_path, monkeypatch, hosted):
    main = tmp_path / "main"
    slot = tmp_path / "main-wt2"
    source = main / SHARED.path
    source.mkdir(parents=True)
    slot.mkdir()
    monkeypatch.setattr(worktree, "main_dir", lambda: main)
    return main, slot, source, slot / SHARED.path


def test_shared_files_and_later_additions_are_visible_without_copying(shared_tree):
    _, slot, source, target = shared_tree
    (source / "chair.glb").write_bytes(b"chair")
    ignore = target.parent / ".gitignore"
    ignore.parent.mkdir(parents=True)
    ignore.write_text("*.glb\n", encoding="utf-8")

    worktree.share_dirs(slot)
    assert target.resolve() == source.resolve()
    assert (target / "chair.glb").read_bytes() == b"chair"
    # A file added after setup must need no slot refresh.
    (source / "bed.glb").write_bytes(b"bed")
    assert (target / "bed.glb").read_bytes() == b"bed"
    worktree.share_dirs(slot)
    assert ignore.read_text(encoding="utf-8") == "*.glb\n"
    assert not ignore.is_symlink()


@pytest.mark.skipif(worktree.os.name != "nt", reason="directory junctions are Windows-only")
def test_on_windows_the_link_is_a_junction(shared_tree):
    _, slot, _, target = shared_tree
    worktree.share_dirs(slot)
    assert worktree.is_directory_junction(target) and not target.is_symlink()


@pytest.mark.skipif(worktree.os.name == "nt", reason="Windows links with a junction")
def test_elsewhere_the_link_is_a_symlink(shared_tree):
    _, slot, _, target = shared_tree
    worktree.share_dirs(slot)
    assert target.is_symlink() and not worktree.is_directory_junction(target)


def test_local_files_are_preserved_and_explicit_sharing_refuses(shared_tree):
    _, slot, _, target = shared_tree
    target.mkdir(parents=True)
    model = target / "local.glb"
    model.write_bytes(b"local model")
    worktree.share_dirs(slot, keep_local=True)
    with pytest.raises(worktree.Fail, match="already exists"):
        worktree.share_dirs(slot)
    assert model.read_bytes() == b"local model"


def test_a_link_elsewhere_is_not_replaced(shared_tree, monkeypatch):
    main, slot, _, target = shared_tree
    worktree.share_dirs(slot)
    elsewhere = main.parent / "other-main"
    (elsewhere / SHARED.path).mkdir(parents=True)
    original = target.resolve()
    monkeypatch.setattr(worktree, "main_dir", lambda: elsewhere)
    with pytest.raises(worktree.Fail, match="already links elsewhere"):
        worktree.share_dirs(slot, keep_local=True)
    assert target.resolve() == original


def test_empty_source_is_prepared_before_the_main_checkout_fills_it(shared_tree):
    _, slot, source, target = shared_tree
    source.rmdir()
    worktree.share_dirs(slot)
    assert source.is_dir()
    assert target.resolve() == source.resolve()
    (source / "first.glb").write_bytes(b"first file")
    assert (target / "first.glb").read_bytes() == b"first file"


def test_main_checkout_cannot_be_linked_to_itself(shared_tree):
    main, _, source, _ = shared_tree
    with pytest.raises(worktree.Fail, match="to itself"):
        worktree.share_dirs(main)
    assert source.is_dir()


def test_equalize_links_missing_shared_dirs_without_an_install(shared_tree, monkeypatch):
    _, slot_path, source, target = shared_tree
    source.rmdir()
    installed = []
    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(
        shared=(SHARED,), install=lambda *a: installed.append(a)))
    monkeypatch.setattr(worktree, "git", lambda *args, **kwargs: SimpleNamespace(stdout="head"))
    monkeypatch.setattr(worktree, "all_submodules", lambda: [])
    slot = worktree.Slot(2, slot_path, "wt2", "head", False)
    worktree.reset_to_master(slot, install=False)
    assert target.resolve() == source.resolve()
    assert installed == []


def test_removing_a_worktree_keeps_shared_files(tmp_path, monkeypatch, hosted):
    main = tmp_path / "main"
    slot = tmp_path / "main-wt2"
    main.mkdir()
    monkeypatch.setattr(worktree, "main_dir", lambda: main)

    def git(*args):
        subprocess.run(["git", "-C", str(main), *args], check=True, capture_output=True)
    git("init")
    git("-c", "user.name=Shared test", "-c", "user.email=test@example.com",
        "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Initial")
    git("worktree", "add", "-b", "wt2", str(slot))
    source = main / SHARED.path
    source.mkdir(parents=True)
    model = source / "chair.glb"
    model.write_bytes(b"shared chair")
    worktree.share_dirs(slot)
    # Exercise the real drop command: Git alone follows a Windows junction.
    pool_slot = worktree.Slot(2, slot, "wt2", "head", False)
    monkeypatch.setattr(worktree, "find_slot", lambda spec: pool_slot)
    monkeypatch.setattr(worktree, "refuse_unless_idle", lambda slot: None)
    monkeypatch.setattr(worktree, "all_submodules", lambda: [])
    assert worktree.cmd_drop(_parse(["drop", "wt2"])) == 0
    assert not slot.exists()
    assert model.read_bytes() == b"shared chair"
