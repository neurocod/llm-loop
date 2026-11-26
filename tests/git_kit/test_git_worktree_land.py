"""Pin how the pool carries submodules along a gitlink bump: `land` into the
main checkout, and `equalize` / `next` / `new` into a slot.

Each test builds throwaway repositories: an upstream for the submodule, a main
checkout that records it at `sub`, and a pool slot `main-wt2` on branch `wt2`
whose own submodule clone makes the bumped commit (never pushed upstream, the
way a slot's submodule commit sits before its push).
"""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from llm_agents.git import live_workers, worktree

from kit_support import kit_env

_CONFIG = {
    "protocol.file.allow": "always",  # `submodule add/update` from a local path
    "user.name": "Land test",
    "user.email": "test@example.com",
    "commit.gpgsign": "false",
    # The empty host's main branch (`PoolHost.main_branch`).
    "init.defaultBranch": "main",
}


def run(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def commit_file(repo, name, text):
    (repo / name).write_text(text, encoding="utf-8")
    run(repo, "add", name)
    run(repo, "commit", "-q", "-m", f"{name}: {text}")
    return run(repo, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", str(len(_CONFIG)))
    for i, (key, value) in enumerate(_CONFIG.items()):
        monkeypatch.setenv(f"GIT_CONFIG_KEY_{i}", key)
        monkeypatch.setenv(f"GIT_CONFIG_VALUE_{i}", value)
    upstream, main, slot = tmp_path / "upstream", tmp_path / "main", tmp_path / "main-wt2"
    for repo in (upstream, main):
        repo.mkdir()
        run(repo, "init", "-q")
    base = commit_file(upstream, "a.txt", "one")
    commit_file(main, "readme.txt", "main")
    run(main, "submodule", "add", "-q", str(upstream), "sub")
    run(main, "commit", "-q", "-m", "add sub")
    run(main, "worktree", "add", "-q", "-b", "wt2", str(slot))
    run(slot, "submodule", "update", "-q", "--init", "--", "sub")
    monkeypatch.setattr(worktree, "HOST", worktree.EMPTY)
    monkeypatch.setattr(worktree, "main_dir", lambda: main)
    pool_slot = worktree.Slot(2, slot, "wt2", "", False)
    monkeypatch.setattr(worktree, "find_slot", lambda spec: pool_slot)
    return main, slot, base


def bump_in_slot(slot):
    """A new submodule commit made in the slot's clone, recorded by the slot's branch."""
    new = commit_file(slot / "sub", "a.txt", "two")
    run(slot, "add", "sub")
    run(slot, "commit", "-q", "-m", "bump sub")
    return new


def bump_in_main(main):
    """A submodule commit made in the MAIN checkout's clone and recorded on
    master, pushed nowhere: what master holds after a land, before the push."""
    new = commit_file(main / "sub", "a.txt", "landed")
    run(main, "add", "sub")
    run(main, "commit", "-q", "-m", "bump sub on master")
    return new


def command(*argv):
    args = worktree.build_parser().parse_args(list(argv))
    return args.func(args)


def land():
    assert command("land", "wt2", "--no-push") == 0


def test_a_fast_forward_gitlink_bump_moves_the_main_submodule(repos, capsys):
    main, slot, base = repos
    new = bump_in_slot(slot)
    land()
    assert run(main / "sub", "rev-parse", "HEAD") == new
    assert run(main, "status", "--porcelain") == ""
    assert "warning" not in capsys.readouterr().err


def test_a_diverged_main_submodule_is_left_alone_with_the_command(repos, capsys):
    main, slot, base = repos
    # Another session's unrecorded work in the main checkout's submodule.
    theirs = commit_file(main / "sub", "b.txt", "theirs")
    new = bump_in_slot(slot)
    land()
    assert run(main / "sub", "rev-parse", "HEAD") == theirs
    err = capsys.readouterr().err
    assert "warning: sub" in err and "HEAD has diverged from the new gitlink" in err
    # `submodule add` leaves the clone on its branch, so the command keeps it there.
    assert f'git -C "{main / "sub"}" merge --ff-only {new}' in err


def test_a_head_on_an_intermediate_ancestor_moves_to_the_new_gitlink(repos, capsys):
    main, slot, base = repos
    mid = commit_file(slot / "sub", "m.txt", "mid")
    new = bump_in_slot(slot)
    run(main / "sub", "fetch", "-q", str(slot / "sub"), mid)
    run(main / "sub", "checkout", "-q", "--detach", mid)
    land()
    assert run(main / "sub", "rev-parse", "HEAD") == new
    assert subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=main / "sub").returncode != 0
    assert "warning" not in capsys.readouterr().err


def test_a_head_ahead_of_the_new_gitlink_stays(repos, capsys):
    main, slot, base = repos
    new = bump_in_slot(slot)
    run(main / "sub", "fetch", "-q", str(slot / "sub"), new)
    run(main / "sub", "checkout", "-q", "--detach", new)
    ahead = commit_file(main / "sub", "c.txt", "ahead")
    land()
    assert run(main / "sub", "rev-parse", "HEAD") == ahead
    assert "ahead of the new gitlink" in capsys.readouterr().err


def test_a_gitlink_moved_sideways_is_not_followed_from_the_old_gitlink(repos, capsys):
    main, slot, base = repos
    # An unrelated history: following it would strand `base` in the reflog.
    run(slot / "sub", "checkout", "-q", "--orphan", "side")
    new = commit_file(slot / "sub", "s.txt", "side")
    run(slot, "add", "sub")
    run(slot, "commit", "-q", "-m", "move sub sideways")
    land()
    assert run(main / "sub", "rev-parse", "HEAD") == base
    err = capsys.readouterr().err
    assert "back or sideways" in err and new[:9] in err


def test_a_head_on_a_branch_moves_with_the_branch(repos, capsys):
    main, slot, base = repos
    run(main / "sub", "checkout", "-q", "-B", "main")
    new = bump_in_slot(slot)
    land()
    assert run(main / "sub", "symbolic-ref", "HEAD") == "refs/heads/main"
    assert run(main / "sub", "rev-parse", "main") == new
    assert "warning" not in capsys.readouterr().err


def test_a_moved_nested_gitlink_is_named(repos, tmp_path, capsys):
    main, slot, base = repos
    inner = tmp_path / "inner"
    inner.mkdir()
    run(inner, "init", "-q")
    commit_file(inner, "i.txt", "one")
    run(slot / "sub", "submodule", "add", "-q", str(inner), "inner")
    run(slot / "sub", "commit", "-q", "-m", "add inner")
    with_inner = run(slot / "sub", "rev-parse", "HEAD")
    run(main / "sub", "fetch", "-q", str(slot / "sub"), with_inner)
    run(main / "sub", "checkout", "-q", "--detach", with_inner)
    run(main / "sub", "submodule", "update", "-q", "--init", "--", "inner")
    commit_file(inner, "i.txt", "two")
    run(slot / "sub" / "inner", "pull", "-q", "origin", "main")
    run(slot / "sub", "add", "inner")
    run(slot / "sub", "commit", "-q", "-m", "bump inner")
    new = run(slot / "sub", "rev-parse", "HEAD")
    run(slot, "add", "sub")
    run(slot, "commit", "-q", "-m", "bump sub")
    land()
    assert run(main / "sub", "rev-parse", "HEAD") == new
    assert "sub/inner (a nested submodule)" in capsys.readouterr().err


def test_a_failing_sync_still_releases_the_slot_and_pushes(repos, monkeypatch, capsys):
    main, slot, base = repos
    bump_in_slot(slot)
    run(main, "worktree", "lock", "--reason", "claimed", str(slot))
    pushed = []
    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(push_by_default=True))
    monkeypatch.setattr(worktree, "push_main_branch", lambda: pushed.append(True) or True)

    def broken(old_main, pool_slot):
        raise worktree.Fail("sync exploded")

    monkeypatch.setattr(worktree, "sync_main_submodules", broken)
    assert command("land", "wt2") == 0
    assert pushed == [True]
    assert "locked" not in run(main, "worktree", "list", "--porcelain")
    assert "sync exploded" in capsys.readouterr().err


def test_the_empty_host_lands_into_main_and_pushes_only_when_asked(repos, monkeypatch, capsys):
    """A stranger's first `land` must not publish anything: the empty host
    lands into `main` and pushes only on `--push`, to `origin`."""
    main, slot, base = repos
    assert (worktree.EMPTY.main_branch, worktree.EMPTY.remote) == ("main", "origin")
    pushed = []
    monkeypatch.setattr(worktree, "push_main_branch", lambda: pushed.append(True) or True)
    commit_file(slot, "one.txt", "first")
    assert command("land", "wt2") == 0
    assert pushed == []
    assert run(main, "rev-parse", "main") == run(slot, "rev-parse", "HEAD")
    out = capsys.readouterr().out
    assert "main -> " in out
    assert "push skipped (this pool does not push by default; --push pushes main to origin)" in out
    commit_file(slot, "two.txt", "second")
    assert command("land", "wt2", "--push") == 0
    assert pushed == [True]
    with pytest.raises(worktree.Fail, match="both --push and --no-push"):
        command("land", "wt2", "--push", "--no-push")


def test_a_pushing_host_pushes_its_branch_to_its_remote(repos, monkeypatch, capsys):
    main, slot, base = repos
    calls = []
    real = worktree.git

    def recording(args, **kwargs):
        if args[0] == "push":
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")
        return real(args, **kwargs)

    monkeypatch.setattr(worktree, "git", recording)
    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(
        main_branch="main", remote="upstream", push_by_default=True))
    commit_file(slot, "one.txt", "first")
    assert command("land", "wt2") == 0
    assert calls == [["push", "upstream", "main"]]
    assert "pushed main to upstream" in capsys.readouterr().out
    commit_file(slot, "two.txt", "second")
    assert command("land", "wt2", "--no-push") == 0
    assert len(calls) == 1
    assert "push skipped (--no-push)" in capsys.readouterr().out


def test_main_refuses_both_push_flags_before_landing(repos):
    main, slot, _ = repos
    before = run(main, "rev-parse", "main")
    pending = commit_file(slot, "pending.txt", "unlanded")
    assert pending != before
    run(main, "worktree", "lock", "--reason", "claimed", str(slot))
    shown = subprocess.run(
        [sys.executable, "-c",
         "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
         "from llm_agents.git import worktree; root = Path(sys.argv[2]); "
         "worktree.main_dir = lambda: root; "
         "sys.argv = ['worktree', 'land', 'wt2', '--push', '--no-push']; "
         "sys.exit(worktree.main())",
         str(Path(worktree.__file__).resolve().parents[2]), str(main)],
        cwd=main, capture_output=True, text=True)
    assert shown.returncode == 1, shown.stderr
    assert "both --push and --no-push" in shown.stderr
    assert run(main, "rev-parse", "main") == before
    assert run(slot, "rev-parse", "HEAD") == pending
    assert "locked claimed" in run(main, "worktree", "list", "--porcelain")


@pytest.mark.parametrize("push_by_default", [False, True])
@pytest.mark.parametrize("flag", [None, "--push", "--no-push"])
@pytest.mark.parametrize("operation", ["equalize", "new"])
def test_slot_preparation_obeys_the_push_policy(repos, monkeypatch, push_by_default, flag, operation):
    main, slot, _ = repos
    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(push_by_default=push_by_default))
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    pushed = []
    monkeypatch.setattr(worktree, "push_main_branch", lambda: pushed.append(True) or True)
    argv = [operation, "wt2" if operation == "equalize" else "3", "--no-install"]
    if flag:
        argv.append(flag)
    assert command(*argv) == 0
    expected = flag == "--push" or (push_by_default and flag != "--no-push")
    assert pushed == ([True] if expected else [])


@pytest.mark.parametrize("push_by_default", [False, True])
def test_create_hook_obeys_the_host_push_default(repos, monkeypatch, push_by_default):
    main, slot, _ = repos
    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(push_by_default=push_by_default))
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree.sys, "stdin", io.StringIO('{"name": "wt2"}'))
    printed = io.StringIO()
    monkeypatch.setattr(worktree.sys, "__stdout__", printed)
    pushed = []
    monkeypatch.setattr(worktree, "push_main_branch", lambda: pushed.append(True) or True)
    assert command("hook-create") == 0
    assert printed.getvalue().strip() == str(slot)
    assert pushed == ([True] if push_by_default else [])


def test_the_help_lists_the_push_flag_that_changes_something(monkeypatch):
    def land_help():
        choices = worktree.build_parser()._subparsers._group_actions[0].choices
        return " ".join(choices["land"].format_help().split())

    monkeypatch.setattr(worktree, "HOST", worktree.EMPTY)
    assert "--push push main to origin afterwards" in land_help()
    assert "--no-push" not in land_help()
    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(main_branch="trunk", push_by_default=True))
    assert "--no-push do not push trunk afterwards" in land_help()
    assert "--push " not in land_help()


def test_all_submodules_keeps_a_path_with_spaces_whole(repos, tmp_path):
    main, slot, base = repos
    run(main, "submodule", "add", "-q", str(tmp_path / "upstream"), "my sub")
    assert worktree.all_submodules() == ["sub", "my sub"]


def test_all_submodules_without_submodules_is_empty(tmp_path, monkeypatch):
    bare = tmp_path / "bare"
    bare.mkdir()
    run(bare, "init", "-q")
    monkeypatch.setattr(worktree, "main_dir", lambda: bare)
    assert worktree.all_submodules() == []
    (bare / ".gitmodules").write_text("", encoding="utf-8")
    assert worktree.all_submodules() == []


def test_a_dirty_main_submodule_is_left_alone(repos, capsys):
    main, slot, base = repos
    (main / "sub" / "a.txt").write_text("edited", encoding="utf-8")
    bump_in_slot(slot)
    land()
    assert run(main / "sub", "rev-parse", "HEAD") == base
    assert (main / "sub" / "a.txt").read_text(encoding="utf-8") == "edited"
    assert "uncommitted changes" in capsys.readouterr().err


def test_a_land_that_does_not_move_the_gitlink_is_silent(repos, capsys):
    main, slot, base = repos
    theirs = commit_file(main / "sub", "b.txt", "theirs")
    commit_file(slot, "other.txt", "unrelated")
    land()
    assert run(main / "sub", "rev-parse", "HEAD") == theirs
    captured = capsys.readouterr()
    assert "warning" not in captured.err
    assert "sub:" not in captured.out


# ------------------------------------------- an unpushed gitlink in a slot

def test_equalize_fetches_an_unpushed_gitlink_from_the_main_clone(repos, capsys):
    """Master records a submodule commit that only the main checkout's clone
    has: `submodule update` in the slot fetches from the upstream and fails
    with "not our ref" (measured 2026-11-07), so `equalize` fetches it from
    the main clone first."""
    main, slot, base = repos
    landed = bump_in_main(main)
    assert command("equalize", "wt2", "--no-push", "--no-install") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == landed
    assert run(slot, "status", "--porcelain") == ""
    assert f"fetched {landed[:9]} from the main checkout's clone" in capsys.readouterr().out


def test_next_moves_a_rebased_slot_s_clean_submodule_forward(repos, capsys):
    """A slot rebased onto such a master keeps its work: `next` fetches the
    commit and moves the clean submodule forward to it."""
    main, slot, base = repos
    landed = bump_in_main(main)
    commit_file(slot, "work.txt", "task")
    run(slot, "rebase", "-q", "main")
    assert command("next", "wt2") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == landed
    assert run(slot, "status", "--porcelain") == ""
    assert "follows the slot's gitlink" in capsys.readouterr().out


def test_next_keeps_a_slot_s_unrecorded_submodule_work(repos, capsys):
    main, slot, base = repos
    bump_in_main(main)
    run(slot, "rebase", "-q", "main")
    mine = commit_file(slot / "sub", "mine.txt", "unrecorded")
    assert command("next", "wt2") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == mine
    assert "not an ancestor of the gitlink" in capsys.readouterr().err


def test_a_failed_fetch_from_the_main_clone_is_one_warning_naming_the_fix(repos, monkeypatch, capsys):
    """A fetch that failed printed nothing, and the slot silently stayed on the
    old commit showing `M sub`."""
    main, slot, base = repos
    landed = bump_in_main(main)
    run(slot, "rebase", "-q", "main")
    real = worktree.git

    def no_fetch(args, **kwargs):
        if args[0] == "fetch":
            return subprocess.CompletedProcess(args, 128, "", "fatal: simulated refusal\n")
        return real(args, **kwargs)

    monkeypatch.setattr(worktree, "git", no_fetch)
    assert command("next", "wt2") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == base
    err = capsys.readouterr().err
    fetch_line = next(line for line in err.splitlines() if "cannot fetch" in line)
    assert fetch_line.startswith(f"warning: sub: cannot fetch {landed[:9]} from the main checkout's clone")
    assert "fatal: simulated refusal" in fetch_line
    assert f'git -C "{slot / "sub"}" fetch "{main / "sub"}" {landed}' in fetch_line
    # And the follow step names the commit it could not reach.
    assert f"{landed[:9]} is not in its clone, and origin does not give it either" in err


def test_a_gitlink_only_origin_has_is_fetched_from_origin(repos, tmp_path, capsys):
    """Skipping a commit the slot's clone lacks also kept `submodule update`
    from fetching one that is on the submodule's origin."""
    main, slot, base = repos
    upstream = tmp_path / "upstream"
    pushed = commit_file(upstream, "a.txt", "pushed")
    run(main, "update-index", "--cacheinfo", f"160000,{pushed},sub")
    run(main, "commit", "-q", "-m", "record a pushed commit the main clone never fetched")
    commit_file(slot, "work.txt", "task")
    run(slot, "rebase", "-q", "main")
    assert command("next", "wt2") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == pushed
    assert "warning" not in capsys.readouterr().err


def test_a_follow_git_refuses_warns_and_the_hook_still_starts_the_agent(repos, monkeypatch, capsys):
    """`next` records its assignment and the create hook claims the slot
    before the submodule follow: an abort there stranded both and the agent
    never started. Here the gitlink's commit adds a file the slot's clone has
    untracked, so the checkout refuses to overwrite it."""
    main, slot, base = repos
    landed = commit_file(main / "sub", "new.txt", "landed")
    run(main, "add", "sub")
    run(main, "commit", "-q", "-m", "bump sub with a new file")
    commit_file(slot, "work.txt", "task")
    run(slot, "rebase", "-q", "main")
    (slot / "sub" / "new.txt").write_text("mine, untracked\n", encoding="utf-8")

    assert command("next", "wt2") == 0
    err = capsys.readouterr().err
    assert "the checkout of the gitlink failed" in err and landed[:9] in err
    assert (main / ".git" / worktree.NEXT_FILE).read_text() == "wt2"

    pool_slot = worktree.Slot(2, slot, "wt2", "", False)
    monkeypatch.setattr(worktree, "list_slots", lambda: [pool_slot])
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree, "push_main_branch", lambda: True)
    monkeypatch.setattr(worktree.sys, "stdin", io.StringIO('{"name": "agent", "session_id": "s"}'))
    printed = io.StringIO()
    monkeypatch.setattr(worktree.sys, "__stdout__", printed)
    assert command("hook-create") == 0
    assert printed.getvalue().strip() == str(slot)
    assert "the checkout of the gitlink failed" in capsys.readouterr().err
    assert (slot / "sub" / "new.txt").read_text(encoding="utf-8") == "mine, untracked\n"
    run(main, "worktree", "unlock", str(slot))


# ------------------------ a slot whose submodule only lags its gitlink (2699)

def lagging_slot(main, slot):
    """What a land plus `git -C <slot> rebase <main>` left (measured
    2026-11-26): the slot's branch records the landed gitlink, its clone
    stands on the old commit, the slot's status shows ` M sub`."""
    landed = bump_in_main(main)
    run(slot, "rebase", "-q", "main")
    assert run(slot, "status", "--porcelain") == "M sub"
    return landed


def first_free_hook(monkeypatch, slot):
    """The create hook with no assignment and no slot asked for by name;
    returns what it printed to the harness."""
    monkeypatch.setattr(worktree, "list_slots", lambda: [worktree.Slot(2, slot, "wt2", "", False)])
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree, "push_main_branch", lambda: True)
    monkeypatch.setattr(worktree.sys, "stdin", io.StringIO('{"name": "agent", "session_id": "s"}'))
    printed = io.StringIO()
    monkeypatch.setattr(worktree.sys, "__stdout__", printed)
    assert command("hook-create") == 0
    return printed.getvalue().strip()


def test_a_submodule_lagging_its_gitlink_is_updated_and_the_slot_handed_out(repos, monkeypatch, capsys):
    """The hook refused such a slot: "no free slot (wt2: dirty ...)"."""
    main, slot, base = repos
    landed = lagging_slot(main, slot)
    assert command("status") == 0
    assert f"sub at {base[:9]}, gitlink {landed[:9]}: updated on hand-out" in capsys.readouterr().out
    assert first_free_hook(monkeypatch, slot) == str(slot)
    assert run(slot / "sub", "rev-parse", "HEAD") == landed
    assert run(slot, "status", "--porcelain") == ""
    run(main, "worktree", "unlock", str(slot))


def test_equalize_takes_a_slot_whose_submodule_only_lags(repos):
    main, slot, base = repos
    landed = lagging_slot(main, slot)
    assert command("equalize", "wt2", "--no-push", "--no-install") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == landed


def test_land_updates_the_slot_s_lagging_submodule_without_a_warning(repos, capsys):
    main, slot, base = repos
    landed = bump_in_main(main)
    commit_file(slot, "work.txt", "task")
    run(slot, "rebase", "-q", "main")
    land()
    assert run(slot / "sub", "rev-parse", "HEAD") == landed
    assert run(slot, "status", "--porcelain") == ""
    assert "uncommitted changes" not in capsys.readouterr().err


@pytest.mark.parametrize("work", [True, False], ids=["landing", "nothing-to-land"])
def test_land_leaves_a_claimed_slot_s_lagging_submodule_alone(repos, monkeypatch, work):
    """A claim is an agent that may still run in the slot: its checkout must
    not move under it; the next hand-out updates it."""
    main, slot, base = repos
    bump_in_main(main)
    if work:
        commit_file(slot, "work.txt", "task")
    run(slot, "rebase", "-q", "main")
    # A real claim: `land` re-reads the slot under its lock (`reloaded`).
    run(main, "worktree", "lock", "--reason", f"{worktree.LOCK_REASON_PREFIX}agent session s",
        str(slot))
    land()
    assert run(slot / "sub", "rev-parse", "HEAD") == base


def held_by_the_hook(monkeypatch, slot, reason):
    with pytest.raises(worktree.Fail, match="no free slot") as refused:
        first_free_hook(monkeypatch, slot)
    assert reason in str(refused.value)


def test_a_dirty_lagging_submodule_holds_the_slot(repos, monkeypatch):
    main, slot, base = repos
    lagging_slot(main, slot)
    (slot / "sub" / "a.txt").write_text("edited", encoding="utf-8")
    held_by_the_hook(monkeypatch, slot, "wt2: dirty (sub: it has uncommitted changes or untracked files)")
    assert run(slot / "sub", "rev-parse", "HEAD") == base
    assert (slot / "sub" / "a.txt").read_text(encoding="utf-8") == "edited"


def test_unpushed_submodule_commits_on_a_branch_hold_the_slot(repos, monkeypatch):
    """A worker's submodule work on its own branch (2689 used `wt1-2689`),
    not recorded by the gitlink and on no remote."""
    main, slot, base = repos
    run(slot / "sub", "checkout", "-q", "-b", "wt2-2699")
    mine = commit_file(slot / "sub", "mine.txt", "unpushed")
    held_by_the_hook(monkeypatch, slot, f"its HEAD {mine[:9]} (branch wt2-2699) carries 1 commit(s) past "
                                        f"the gitlink {base[:9]}")
    assert run(slot / "sub", "rev-parse", "HEAD") == mine
    with pytest.raises(worktree.Fail, match="past the gitlink"):
        command("equalize", "wt2", "--no-push", "--no-install")


def test_a_submodule_diverged_from_its_gitlink_holds_the_slot(repos, monkeypatch):
    main, slot, base = repos
    landed = lagging_slot(main, slot)
    mine = commit_file(slot / "sub", "mine.txt", "side")
    held_by_the_hook(monkeypatch, slot, f"has diverged from the gitlink {landed[:9]}")
    assert run(slot / "sub", "rev-parse", "HEAD") == mine


def test_commits_past_the_gitlink_hold_the_slot_even_on_a_remote_tracking_branch(repos, monkeypatch):
    """A remote-tracking ref proves the commit reachable here, not published:
    a fetch from a local path or a stale ref makes one too."""
    main, slot, base = repos
    mine = commit_file(slot / "sub", "mine.txt", "pushed")
    run(slot / "sub", "push", "-q", "origin", "HEAD:refs/heads/pushed")
    run(slot / "sub", "fetch", "-q", "origin")
    assert run(slot / "sub", "branch", "-r", "--contains", mine)
    held_by_the_hook(monkeypatch, slot, f"carries 1 commit(s) past the gitlink {base[:9]}")
    assert run(slot / "sub", "rev-parse", "HEAD") == mine


def test_an_ignored_file_the_gitlink_tracks_is_not_overwritten_and_holds_the_slot(repos, monkeypatch):
    """An ordinary checkout replaced it silently."""
    main, slot, base = repos
    admin = Path(run(slot / "sub", "rev-parse", "--absolute-git-dir"))
    (admin / "info").mkdir(exist_ok=True)
    (admin / "info" / "exclude").write_text("cfg.local\n", encoding="utf-8")
    (slot / "sub" / "cfg.local").write_text("mine", encoding="utf-8")
    commit_file(main / "sub", "cfg.local", "theirs")
    run(main, "add", "sub")
    run(main, "commit", "-q", "-m", "bump sub with cfg.local")
    run(slot, "rebase", "-q", "main")
    assert run(slot / "sub", "status", "--porcelain") == ""
    held_by_the_hook(monkeypatch, slot, "wt2: dirty (wt2's sub was not updated: the checkout of the gitlink failed")
    assert run(slot / "sub", "rev-parse", "HEAD") == base
    assert (slot / "sub" / "cfg.local").read_text(encoding="utf-8") == "mine"
    assert "locked" not in run(main, "worktree", "list", "--porcelain")
    with pytest.raises(worktree.Fail, match="the checkout of the gitlink failed"):
        command("equalize", "wt2", "--no-push", "--no-install")
    assert (slot / "sub" / "cfg.local").read_text(encoding="utf-8") == "mine"


def test_a_submodule_status_that_fails_holds_the_slot(repos):
    """An error prints nothing on stdout, which read as a clean clone."""
    main, slot, base = repos
    lagging_slot(main, slot)
    admin = Path(run(slot / "sub", "rev-parse", "--absolute-git-dir"))
    (admin / "index").write_bytes(b"not an index")
    state = worktree.slot_state(worktree.find_slot("wt2"))
    assert not state.clean and not state.stale
    assert any(why.startswith("sub: its status cannot be read") for why in state.held), state.held


def test_the_update_keeps_other_branches_and_the_stash_of_the_submodule(repos, monkeypatch):
    """HEAD is an ancestor of the gitlink, so the slot is handed out; work
    that is not checked out -- a local branch with unpushed commits, a stash
    -- must come through the move."""
    main, slot, base = repos
    sub = slot / "sub"
    run(sub, "checkout", "-q", "-b", "side")
    kept = commit_file(sub, "side.txt", "unpushed")
    run(sub, "checkout", "-q", "--detach", base)
    (sub / "a.txt").write_text("stashed", encoding="utf-8")
    run(sub, "stash", "push", "-q", "-m", "wip")
    stash = run(sub, "rev-parse", "refs/stash")
    landed = lagging_slot(main, slot)
    assert first_free_hook(monkeypatch, slot) == str(slot)
    assert run(sub, "rev-parse", "HEAD") == landed
    assert run(sub, "rev-parse", "refs/heads/side") == kept
    assert run(sub, "rev-parse", "refs/stash") == stash
    run(sub, "cat-file", "-e", f"{kept}^{{commit}}")
    run(sub, "cat-file", "-e", f"{stash}^{{commit}}")
    assert "wip" in run(sub, "stash", "list")
    run(main, "worktree", "unlock", str(slot))


# ------------------------------- the worker's own git in a slot's submodule

def plant(slot, tmp_path):
    """What a sandboxed worker may do to its slot's submodule (item 2442): point
    `<slot>/sub/.git` at a repository of its own whose config names a program
    (`marker`'s) as fsmonitor and as clean filter for every file. Returns the
    marker file the program appends to."""
    marker = tmp_path / "marker.txt"
    script = tmp_path / "planted.sh"
    script.write_bytes(f"#!/bin/sh\necho ran >> '{marker.as_posix()}'\ncat\n".encode("utf-8"))
    script.chmod(0o755)  # POSIX git runs a hook or filter only when executable
    planted = tmp_path / "planted.git"
    run(tmp_path, "init", "-q", "--bare", str(planted))
    for key, value in (("core.bare", "false"), ("core.fsmonitor", script.as_posix()),
                       ("filter.evil.clean", script.as_posix())):
        run(tmp_path, "config", "--file", str(planted / "config"), key, value)
    (slot / "sub" / ".git").unlink()  # git for Windows makes it hidden: no overwrite in place
    (slot / "sub" / ".git").write_text(f"gitdir: {planted}\n", encoding="utf-8")
    (slot / "sub" / ".gitattributes").write_text("* filter=evil\n", encoding="utf-8")
    # The control: git by discovery there runs the worker's program.
    subprocess.run(["git", "status", "--porcelain"], cwd=slot / "sub", capture_output=True)
    assert marker.exists()
    marker.unlink()
    return marker


def test_next_and_the_create_hook_refuse_a_repointed_submodule(repos, tmp_path, monkeypatch):
    """Neither `next` nor the create hook it arms hands out a slot whose
    submodule git cannot be bound: an agent started there would work beside
    a `.git` nobody may touch (review of 2442)."""
    main, slot, base = repos
    bump_in_main(main)
    commit_file(slot, "work.txt", "task")
    run(slot, "rebase", "-q", "main")
    marker = plant(slot, tmp_path)
    with pytest.raises(worktree.Fail, match="sub: its .git names"):
        command("next", "wt2")
    assert not (main / ".git" / worktree.NEXT_FILE).exists()
    # Armed before the slot went bad: the hook refuses it all the same.
    (main / ".git" / worktree.NEXT_FILE).write_text("wt2")
    monkeypatch.setattr(worktree, "list_slots", lambda: [worktree.Slot(2, slot, "wt2", "", False)])
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree.sys, "stdin", io.StringIO('{"name": "agent", "session_id": "s"}'))
    printed = io.StringIO()
    monkeypatch.setattr(worktree.sys, "__stdout__", printed)
    with pytest.raises(worktree.Fail, match="a git in it is not the one the slot registers"):
        command("hook-create")
    assert printed.getvalue() == ""
    assert "locked" not in run(main, "worktree", "list", "--porcelain")
    assert not marker.exists()


def test_equalize_drop_and_the_pool_listing_refuse_a_repointed_submodule(repos, tmp_path, capsys,
                                                                          monkeypatch):
    main, slot, base = repos
    bump_in_main(main)
    marker = plant(slot, tmp_path)
    monkeypatch.setattr(worktree, "list_slots", lambda: [worktree.Slot(2, slot, "wt2", "", False)])
    for operation in ("equalize", "drop"):
        with pytest.raises(worktree.Fail, match="a git in it is not the one the slot registers"):
            command(operation, "wt2", *(["--no-push", "--no-install"] if operation == "equalize" else []))
    assert command("status") == 0
    assert "git not the slot's (sub: its .git names" in capsys.readouterr().out
    assert not marker.exists()
    assert slot.is_dir()


def test_land_does_not_fetch_through_a_repointed_slot_submodule(repos, tmp_path, capsys):
    main, slot, base = repos
    bump_in_slot(slot)
    plant(slot, tmp_path)
    land()
    assert run(main / "sub", "rev-parse", "HEAD") == base
    assert "wt2: not asked, its .git names" in capsys.readouterr().err


def test_every_git_in_a_slot_s_submodule_is_bound_to_its_registered_admin(repos, monkeypatch, capsys):
    """Each git `next` and `equalize`'s reset run in the slot's submodule
    names the admin directory the main clone registers (`--git-dir`), not
    what its `.git` says, with the forced settings and no hooks."""
    main, slot, base = repos
    admin = (main / ".git" / "worktrees" / "main-wt2" / "modules" / "sub").resolve()
    real = worktree.git
    seen = []

    def recording(args, **kwargs):
        cwd = kwargs.get("cwd")
        if cwd is not None and worktree.is_inside(cwd, slot / "sub"):
            seen.append((args, kwargs.get("pinned")))
        return real(args, **kwargs)

    monkeypatch.setattr(worktree, "git", recording)
    landed = bump_in_main(main)
    commit_file(slot, "work.txt", "task")
    run(slot, "rebase", "-q", "main")
    assert command("next", "wt2") == 0
    run(slot / "sub", "checkout", "-q", "--detach", base)
    worktree.reset_to_main_branch(worktree.Slot(2, slot, "wt2", "", False), install=False)
    assert run(slot / "sub", "rev-parse", "HEAD") == landed
    assert {args[0] for args, _ in seen} >= {"rev-parse", "status", "checkout", "cat-file"}
    for args, pinned in seen:
        assert pinned is not None, args
        argv = pinned.argv(args)
        assert argv[argv.index("--git-dir") + 1] == str(admin), args
        assert "core.fsmonitor=false" in argv and f"core.hooksPath={main / '.git' / worktree.NO_HOOKS_DIR}" in argv


def plant_hooks(main, slot, tmp_path):
    """The main clone's `core.hooksPath` is the first host's relative `.githooks`,
    so git by discovery in a slot runs `<slot>/.githooks/*`; the worker plants
    two there, ignored by a `.gitignore` of their own so the slot stays clean.
    Returns the marker file they append to."""
    run(main, "config", "core.hooksPath", ".githooks")
    marker = tmp_path / "hooks-ran.txt"
    hooks = slot / ".githooks"
    hooks.mkdir()
    (hooks / ".gitignore").write_text("*\n", encoding="utf-8")
    for name in ("post-index-change", "reference-transaction"):
        (hooks / name).write_bytes(f"#!/bin/sh\necho {name} >> '{marker.as_posix()}'\n"
                                   .encode("utf-8"))
        (hooks / name).chmod(0o755)  # POSIX git runs a hook only when executable
    # The control: git by discovery there runs both.
    os.utime(slot / "readme.txt")
    subprocess.run(["git", "status", "--porcelain"], cwd=slot, capture_output=True)
    subprocess.run(["git", "reset", "-q", "--hard", "--no-recurse-submodules", "HEAD"], cwd=slot,
                   capture_output=True)
    assert set(marker.read_text(encoding="utf-8").split()) == {"post-index-change",
                                                               "reference-transaction"}
    marker.unlink()
    return marker


def test_status_and_equalize_run_no_hook_of_the_slot(repos, tmp_path, capsys):
    main, slot, base = repos
    landed = bump_in_main(main)
    marker = plant_hooks(main, slot, tmp_path)
    os.utime(slot / "readme.txt")
    assert command("status") == 0
    assert "wt2" in capsys.readouterr().out
    os.utime(slot / "readme.txt")
    assert command("equalize", "wt2", "--no-push", "--no-install") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == landed
    assert not marker.exists()


def test_a_new_slot_s_submodules_are_cloned_by_the_bound_git(repos, tmp_path, monkeypatch):
    main, slot, base = repos
    # The bound git drops the fixture's GIT_CONFIG_* (`Pinned.environment`),
    # and the submodule's clone reads no config of the superproject.
    strip = worktree.Pinned.environment
    monkeypatch.setattr(worktree.Pinned, "environment", staticmethod(lambda base=None: {
        **strip(base), "GIT_CONFIG_PARAMETERS": "'protocol.file.allow'='always'"}))
    run(main, "config", "core.hooksPath", ".githooks")
    monkeypatch.setattr(worktree, "slot_path", lambda number: tmp_path / f"main-wt{number}")
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    seen = []
    real = worktree.git

    def recording(args, **kwargs):
        if args[0] == "submodule":
            seen.append(kwargs.get("pinned"))
        return real(args, **kwargs)

    monkeypatch.setattr(worktree, "git", recording)
    assert command("new", "3", "--submodules", "all", "--no-install", "--no-push") == 0
    assert (tmp_path / "main-wt3" / "sub" / "a.txt").read_text(encoding="utf-8") == "one"
    assert seen and all(p is not None and "--git-dir" in p.argv([]) for p in seen)


def test_a_bound_git_takes_the_main_checkout_s_lfs_endpoint(repos, tmp_path):
    """A worker's `.lfsconfig` names the URL git-lfs downloads from; the
    bound git's `lfs.url` (the main checkout's endpoint) beats it."""
    main, slot, base = repos
    if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        pytest.skip("git-lfs is not installed")
    run(main, "remote", "add", "origin", "https://main.invalid/project.git")
    (slot / ".lfsconfig").write_text("[lfs]\n\turl = https://worker.invalid/lfs\n", encoding="utf-8")
    pinned, why = worktree.slot_top(slot)
    assert pinned is not None, why
    shown = subprocess.run(pinned.argv(["lfs", "env"]), cwd=slot, capture_output=True, text=True,
                           env=worktree.Pinned.environment()).stdout
    endpoint = next(line for line in shown.splitlines() if line.startswith("Endpoint="))
    assert endpoint.startswith("Endpoint=https://main.invalid/project.git/info/lfs"), shown


def test_a_submodule_reached_through_a_link_is_not_bound(repos, tmp_path):
    """A bound checkout writes the work tree it names: a link along
    `<slot>/sub` would carry it out of the slot."""
    main, slot, base = repos
    assert worktree.slot_submodule(slot, "sub")[0] is not None
    moved = tmp_path / "elsewhere"
    (slot / "sub").rename(moved)
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(moved), str(slot / "sub"))
    else:
        (slot / "sub").symlink_to(moved, target_is_directory=True)
    pinned, why = worktree.slot_submodule(slot, "sub")
    assert pinned is None and why.endswith("is a link")
    os.rmdir(slot / "sub") if sys.platform == "win32" else (slot / "sub").unlink()


def test_find_slot_by_directory_asks_no_git(repos, tmp_path, monkeypatch):
    main, slot, base = repos
    monkeypatch.undo()
    monkeypatch.setattr(worktree, "HOST", worktree.EMPTY)
    monkeypatch.setattr(worktree, "main_dir", lambda: main)
    pool_slot = worktree.Slot(2, slot, "wt2", "", False)
    monkeypatch.setattr(worktree, "list_slots", lambda: [pool_slot])
    monkeypatch.setattr(worktree, "describe", lambda s: "free")
    monkeypatch.setattr(worktree, "git", lambda *a, **k: pytest.fail(f"git asked: {a}"))
    assert worktree.find_slot(str(slot / "sub")) is pool_slot
    with pytest.raises(worktree.Fail, match="in the main checkout"):
        worktree.find_slot(str(main / "sub"))
    with pytest.raises(worktree.Fail, match="in no pool slot"):
        worktree.find_slot(str(tmp_path))


# ------------------------------------------------------------ live workers

LIVE_REFUSAL = "a worker is (running|still alive) there"
# Each command that acts on a slot's tree, as `command` runs it.
TREE_COMMANDS = {
    "land": ("land", "wt2", "--no-push"),
    "next": ("next", "wt2"),
    "release": ("release", "wt2"),
    "equalize": ("equalize", "wt2", "--no-push", "--no-install"),
    "drop": ("drop", "wt2"),
}


@pytest.fixture
def locked_slot(repos, monkeypatch):
    """`repos` with the slot locked, as a task holds it, and work to land."""
    main, slot, base = repos
    commit_file(slot, "work.txt", "task")
    run(main, "worktree", "lock", "--reason", "worktree.py:test", str(slot))
    pool_slot = worktree.Slot(2, slot, "wt2", "", True, "worktree.py:test")
    monkeypatch.setattr(worktree, "find_slot", lambda spec: pool_slot)
    monkeypatch.setattr(worktree, "list_slots", lambda: [pool_slot])
    return main, slot, pool_slot


def ended_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = live_workers.process_identity(proc.pid)
    proc.wait()
    return proc.pid, identity


def write_record(roots, processes, job=""):
    """A record file as a wrapper whose guard is gone would have left it."""
    folder = live_workers.live_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"left-{len(list(folder.iterdir()))}{live_workers.RECORD_SUFFIX}"
    path.write_text(json.dumps({"name": "left", "roots": [str(r) for r in roots], "job": job,
                                "processes": [{"pid": p, "identity": i} for p, i in processes]}),
                    encoding="utf-8")
    return path


@pytest.mark.parametrize("held_by", ["guard", "record"])
@pytest.mark.parametrize("operation", sorted(TREE_COMMANDS))
def test_a_live_worker_refuses_every_tree_command(locked_slot, operation, held_by):
    """The pool's checks assume the slot's worker has ended (review of 2442):
    while a turn holds the slot's guard, or a record whose process lives
    names it, nothing acts on it."""
    main, slot, pool_slot = locked_slot
    head = run(main, "rev-parse", "main")
    with contextlib.ExitStack() as stack:
        if held_by == "guard":
            stack.enter_context(live_workers.recorded("worker", [slot]))
        else:
            write_record([slot], [(os.getpid(), live_workers.process_identity(os.getpid()))])
        with pytest.raises(worktree.Fail, match=LIVE_REFUSAL):
            command(*TREE_COMMANDS[operation])
    assert run(main, "rev-parse", "main") == head
    assert "locked" in run(main, "worktree", "list", "--porcelain")
    assert not (main / ".git" / worktree.NEXT_FILE).exists()
    assert slot.is_dir()


@pytest.mark.parametrize("operation", ["land", "next", "release"])
def test_the_override_and_a_stale_record_let_a_command_through(locked_slot, operation):
    main, slot, pool_slot = locked_slot
    with live_workers.recorded("worker", [slot]):
        assert command(*TREE_COMMANDS[operation], worktree.IGNORE_LIVE_FLAG) == 0
    stale = write_record([slot], [ended_pid()])
    # `land` and `release` unlocked it; `next` did not (git refuses a second lock).
    subprocess.run(["git", "worktree", "lock", "--reason", "worktree.py:test", str(slot)],
                   cwd=main, capture_output=True)
    assert command(*TREE_COMMANDS[operation]) == 0
    assert not stale.exists()


def test_a_live_worker_is_shown_and_its_slot_is_never_first_free(locked_slot):
    main, slot, pool_slot = locked_slot
    run(main, "worktree", "unlock", str(slot))
    free = worktree.Slot(2, slot, "wt2", "", False)
    with live_workers.recorded("worker", [slot]):
        state = worktree.slot_state(free)
        assert not state.free and "worker alive (worker (pid" in state.describe()


def test_a_worker_starting_during_a_pool_command_is_refused(repos, monkeypatch):
    """Check-then-act closed: a turn that starts while `equalize` resets and
    installs (the host's npm reads files a worker could plant) is refused,
    because the pool holds its guard for the whole command."""
    main, slot, base = repos
    started = []

    def install(path, admin):
        with pytest.raises(SystemExit, match="a worktree pool command is acting"):
            with live_workers.recorded("late", [slot / "src"]):
                started.append("late")
        started.append("install")

    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(main_branch="main", install=install))
    monkeypatch.setattr(worktree, "list_slots", lambda: [worktree.Slot(2, slot, "wt2", "", False)])
    assert command("equalize", "wt2", "--no-push") == 0
    assert started == ["install"]
    # And the guard went with the command: a turn starts now.
    with live_workers.recorded("now", [slot]):
        pass


OPERATION_REFUSAL = r"another pool command is acting on .*: pid \d+ \(worktree "


@pytest.mark.parametrize("second", [*sorted(TREE_COMMANDS), "hook-create", "hook-remove",
                                    "equalize-overriding"])
def test_a_second_pool_command_on_a_slot_another_is_preparing_is_refused(repos, monkeypatch,
                                                                         capsys, second):
    """The pool's guard is shared among pool commands, so a `land`
    could act while `equalize` (or the create hook) stood between its look at
    the slot and its reset and install, on a verdict the other had made
    stale. Each command holds the slot's operation lock from its first look
    to its last act; `--ignore-live-worker` is about workers and does not
    lift it."""
    main, slot, base = repos
    monkeypatch.setattr(live_workers, "OPERATION_WAIT_S", 0.2)
    free = worktree.Slot(2, slot, "wt2", "", False)
    monkeypatch.setattr(worktree, "list_slots", lambda: [free])
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree, "take_next_assignment", lambda: "")
    monkeypatch.setattr(worktree.sys, "stdin", io.StringIO(json.dumps({"path": str(slot)})))
    argv = {"hook-create": ("hook-create",), "hook-remove": ("hook-remove",),
            "equalize-overriding": (*TREE_COMMANDS["equalize"], worktree.IGNORE_LIVE_FLAG)}
    refused = []

    def install(path, admin):
        try:
            command(*argv.get(second, TREE_COMMANDS.get(second, ())))
        except worktree.Fail as error:
            refused.append(str(error))
        if second == "hook-remove":     # it refuses by keeping the slot, exit 0
            refused.extend(line for line in capsys.readouterr().err.splitlines()
                           if "keeping it as is" in line)

    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(main_branch="main", install=install))
    head = run(main, "rev-parse", "main")
    assert command("equalize", "wt2", "--no-push") == 0
    assert len(refused) == 1, f"{second} ran inside equalize's install"
    assert re.search(OPERATION_REFUSAL, refused[0]), refused[0]
    assert run(main, "rev-parse", "main") == head and slot.is_dir()
    assert "locked" not in run(main, "worktree", "list", "--porcelain")
    assert not (main / ".git" / worktree.NEXT_FILE).exists()
    # And the lock went with the command.
    assert command("release", "wt2") == 0


def test_the_create_hook_takes_the_next_slot_while_another_command_holds_one(repos,
                                                                             monkeypatch):
    """Not waited for: a held slot is not free now, and the next one is."""
    main, slot, base = repos
    other = main.parent / "main-wt3"
    run(main, "worktree", "add", "-q", "-b", "wt3", str(other))
    run(other, "submodule", "update", "-q", "--init", "--", "sub")
    monkeypatch.setattr(live_workers, "OPERATION_WAIT_S", 30)
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree, "take_next_assignment", lambda: "")
    with live_workers.slot_operation(slot, "worktree land wt2"):
        started = time.monotonic()
        taken, reset, how = worktree.choose_slot("agent", "r")
        assert time.monotonic() - started < 10
    assert (taken.name, reset, how) == ("wt3", True, "first free")
    assert worktree.norm(taken.path) == worktree.norm(other)
    locks = {s.name: s.locked for s in worktree.list_slots()}
    assert locks == {"wt2": False, "wt3": True}


# A pool command run as its own process, with two barriers in `barrier` (its
# own folder): `waiting` is touched right before it asks for the slot's
# operation lock -- after it has resolved the slot -- and, if `pause` names a
# `worktree` function, the command stops on entering it (`<pause>-reached`)
# until `<pause>-go` appears. Its wait for the lock is long (120 s): only a
# lock never let go may turn into a red.
CONTENDER = """
import sys, time
from pathlib import Path
from llm_agents.git import live_workers, worktree

barrier, pause, argv = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]

def wait_for(name):
    deadline = time.monotonic() + 120
    while not (barrier / name).exists():
        if time.monotonic() > deadline:
            sys.exit(f"contender: barrier {name} never came")
        time.sleep(0.02)

if pause:
    real_pause = getattr(worktree, pause)
    def paused(*args, **kwargs):
        (barrier / f"{pause}-reached").touch()
        wait_for(f"{pause}-go")
        return real_pause(*args, **kwargs)
    setattr(worktree, pause, paused)

real_operation = live_workers.slot_operation
def announced(*args, **kwargs):
    (barrier / "waiting").touch()
    return real_operation(*args, **kwargs)
live_workers.slot_operation = announced
live_workers.OPERATION_WAIT_S = 120
sys.argv = ["worktree.py", *argv]
sys.exit(worktree.main())
"""


class Contender:
    def __init__(self, main, tmp_path, name, pause, *argv, stdin=""):
        self.barrier = tmp_path / f"barrier-{name}"
        self.barrier.mkdir()
        self.proc = subprocess.Popen(
            [sys.executable, "-c", CONTENDER, str(self.barrier), pause, *argv], cwd=main,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", env=kit_env())
        self.proc.stdin.write(stdin)
        self.proc.stdin.close()

    def reached(self, name):
        # 120 s: a contender starts a Python and a few gits before it gets
        # there (about a second each); the bound only turns one that never
        # arrives into a red.
        deadline = time.monotonic() + 120
        while not (self.barrier / name).exists():
            assert self.proc.poll() is None, f"contender ended early: {self.finish()}"
            assert time.monotonic() < deadline, f"contender never reached {name}"
            time.sleep(0.02)

    def go(self, pause):
        (self.barrier / f"{pause}-go").touch()

    def finish(self):
        # 180 s: as `reached`, plus the reset of a slot (seconds).
        out, err = self.proc.communicate(timeout=180)
        return self.proc.returncode, out, err

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.communicate()


@pytest.fixture
def contenders(repos):
    main, slot, base = repos
    # The real `is_permitted` in the child: the slot is listed.
    (main / ".claude").mkdir()
    (main / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"additionalDirectories": [str(slot)]}}), encoding="utf-8")
    started = []
    yield started
    for contender in started:
        contender.kill()


def test_a_command_that_waited_for_the_hook_sees_the_slot_it_claimed(repos, tmp_path,
                                                                      contenders):
    """`equalize` resolved the slot while the create hook held its lock and
    had not yet claimed it; the hook claimed it and handed it out. Once
    `equalize` gets the lock it must see that claim and refuse, not reset an
    assigned slot on what it read before waiting."""
    main, slot, base = repos
    work = commit_file(main, "later.txt", "main moved on")
    hook = Contender(main, tmp_path, "hook", "claim", "hook-create",
                     stdin=json.dumps({"name": "", "session_id": "hook"}))
    contenders.append(hook)
    hook.reached("claim-reached")
    equalize = Contender(main, tmp_path, "equalize", "", "equalize", "wt2", "--no-push",
                         "--no-install")
    contenders.append(equalize)
    equalize.reached("waiting")
    hook.go("claim")
    code, out, err = hook.finish()
    assert code == 0, err
    assert worktree.norm(out.strip()) == worktree.norm(slot)
    code, out, err = equalize.finish()
    assert code != 0 and "wt2 is locked (worktree.py:" in err, err
    porcelain = run(main, "worktree", "list", "--porcelain")
    assert "locked worktree.py:unnamed session hook" in porcelain, porcelain
    # The hook's own reset brought it to main; nothing reset it after.
    assert run(slot, "rev-parse", "HEAD") == work


def test_an_assignment_that_waited_for_a_land_claims_the_slot_again(repos, tmp_path,
                                                                     contenders):
    """`next wt2` queued the claimed slot for its reviewer; the create hook
    read the claim while `land` held the lock, and `land` unclaimed it. Once
    the hook gets the lock it must see the slot unclaimed and claim it, not
    hand out an unclaimed slot on the claim it read before waiting."""
    main, slot, base = repos
    run(main, "worktree", "lock", "--reason", f"{worktree.LOCK_REASON_PREFIX}task", str(slot))
    command("next", "wt2")
    land = Contender(main, tmp_path, "land", "unclaim", "land", "wt2", "--no-push")
    contenders.append(land)
    land.reached("unclaim-reached")
    hook = Contender(main, tmp_path, "hook", "", "hook-create",
                     stdin=json.dumps({"name": "", "session_id": "review"}))
    contenders.append(hook)
    hook.reached("waiting")
    land.go("unclaim")
    code, out, err = land.finish()
    assert code == 0 and "wt2 has nothing to land" in out, err
    code, out, err = hook.finish()
    assert code == 0 and "-> wt2 (assigned" in err, err
    porcelain = run(main, "worktree", "list", "--porcelain")
    assert "locked worktree.py:unnamed session review" in porcelain, porcelain


def in_process_hook(monkeypatch, session):
    """The create hook run here: (exit code or the `Fail` it raised, what it
    printed to the harness)."""
    monkeypatch.setattr(worktree.sys, "stdin",
                        io.StringIO(json.dumps({"name": "", "session_id": session})))
    printed = io.StringIO()
    monkeypatch.setattr(worktree.sys, "__stdout__", printed)
    try:
        result = command("hook-create")
    except worktree.Fail as refused:
        result = refused
    return result, printed.getvalue().strip()


def test_an_assignment_survives_a_hook_that_could_not_take_the_slot_s_lock(repos, tmp_path,
                                                                            monkeypatch, capsys,
                                                                            contenders):
    """`next wt2` queued the claimed slot (its task's commits unlanded) for
    the task's reviewer; another pool command held wt2's lock longer than the
    hook waits, so the hook failed and no agent started. The assignment had
    been taken from the queue before the lock was asked for, so the retry
    found it empty and handed out the first free slot: the reviewer in
    another task's tree. The failed hook puts it back."""
    main, slot, base = repos
    commit_file(slot, "work.txt", "task")
    run(main, "worktree", "lock", "--reason", f"{worktree.LOCK_REASON_PREFIX}task", str(slot))
    other = main.parent / "main-wt3"
    run(main, "worktree", "add", "-q", "-b", "wt3", str(other))
    run(other, "submodule", "update", "-q", "--init", "--", "sub")
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree, "push_main_branch", lambda: True)
    assert command("next", "wt2") == 0
    queued = main / ".git" / worktree.NEXT_FILE
    # Holds wt2's lock from before its first look to the go below.
    holder = Contender(main, tmp_path, "equalize", "refuse_unless_idle", "equalize", "wt2",
                       "--no-push", "--no-install")
    contenders.append(holder)
    holder.reached("refuse_unless_idle-reached")
    monkeypatch.setattr(live_workers, "OPERATION_WAIT_S", 0.3)
    capsys.readouterr()
    refused, printed = in_process_hook(monkeypatch, "review")
    assert isinstance(refused, worktree.Fail) and re.search(OPERATION_REFUSAL, str(refused)), refused
    assert printed == ""
    assert queued.read_text() == "wt2"
    assert "`next wt2` stays queued" in capsys.readouterr().err
    holder.go("refuse_unless_idle")
    code, out, err = holder.finish()
    assert code != 0 and "wt2 is locked (worktree.py:" in err, err
    code, printed = in_process_hook(monkeypatch, "retry")
    assert code == 0 and "-> wt2 (assigned" in capsys.readouterr().err
    assert worktree.norm(printed) == worktree.norm(slot)
    assert not queued.exists()
    locks = {s.name: s.locked for s in worktree.list_slots()}
    assert locks == {"wt2": True, "wt3": False}


def test_a_requeued_assignment_never_replaces_a_newer_next(repos, capsys):
    main, slot, base = repos
    queued = main / ".git" / worktree.NEXT_FILE
    queued.write_text("wt3")
    worktree.requeue_assignment("wt2 --ignore-live-worker")
    assert queued.read_text() == "wt3"
    assert "a newer `next` was queued meanwhile" in capsys.readouterr().err
    queued.unlink()
    worktree.requeue_assignment("wt2 --ignore-live-worker")
    assert queued.read_text() == "wt2 --ignore-live-worker"
    assert [p.name for p in queued.parent.glob(f"{worktree.NEXT_FILE}*")] == [worktree.NEXT_FILE]


def test_a_free_slot_whose_reset_cannot_move_a_submodule_is_not_handed_out(repos, monkeypatch):
    """The create hook once reset the picked slot non-strictly, warned,
    and handed out a slot showing ` M sub`. Here the slot is behind main,
    whose gitlink tracks a file the slot's clone ignores."""
    main, slot, base = repos
    admin = Path(run(slot / "sub", "rev-parse", "--absolute-git-dir"))
    (admin / "info").mkdir(exist_ok=True)
    (admin / "info" / "exclude").write_text("cfg.local\n", encoding="utf-8")
    (slot / "sub" / "cfg.local").write_text("mine", encoding="utf-8")
    commit_file(main / "sub", "cfg.local", "theirs")
    run(main, "add", "sub")
    run(main, "commit", "-q", "-m", "bump sub with cfg.local")
    assert run(slot, "status", "--porcelain") == ""
    held_by_the_hook(monkeypatch, slot, "wt2: dirty (wt2's sub was not updated: the checkout of the gitlink failed")
    assert (slot / "sub" / "cfg.local").read_text(encoding="utf-8") == "mine"
    assert "locked" not in run(main, "worktree", "list", "--porcelain")


@pytest.mark.parametrize("install", [True, False])
@pytest.mark.parametrize("root", ["slot", "parent"])
def test_new_creates_nothing_under_a_worker(repos, monkeypatch, install, root):
    """Review 4 of 2470: `new` took the guard only around npm, so a worker
    allowed to write the new slot (or a parent of it) saw the tree created,
    linked and given submodules; `--no-install` skipped the guard entirely."""
    main, slot, base = repos
    installed = []
    monkeypatch.setattr(worktree, "HOST", worktree.PoolHost(
        main_branch="main", install=lambda path, admin: installed.append(path)))
    new = main.parent / "main-wt3"
    with live_workers.recorded("early", [new if root == "slot" else main.parent]):
        with pytest.raises(worktree.Fail, match=LIVE_REFUSAL):
            command("new", "3", "--no-push", *([] if install else ["--no-install"]))
    assert installed == [] and not new.exists()


def test_a_turn_cannot_start_in_a_slot_new_is_preparing(repos, monkeypatch):
    main, slot, base = repos
    late = []

    def init_submodules(new_slot, which):
        with pytest.raises(SystemExit, match="a worktree pool command is acting"):
            with live_workers.recorded("late", [new_slot.path]):
                late.append("started")

    monkeypatch.setattr(worktree, "init_submodules", init_submodules)
    assert command("new", "3", "--no-push", "--no-install") == 0
    assert late == []


@pytest.mark.parametrize("override", [False, True])
def test_a_queued_assignment_is_asked_again_unless_next_overrode(locked_slot, monkeypatch,
                                                                 override):
    main, slot, pool_slot = locked_slot
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    command("next", "wt2", *([worktree.IGNORE_LIVE_FLAG] if override else []))
    with live_workers.recorded("worker", [slot]):
        if override:
            assert worktree.choose_slot("agent", "r")[0] is pool_slot
        else:
            with pytest.raises(worktree.Fail, match=LIVE_REFUSAL):
                worktree.choose_slot("agent", "r")


def worker_finishing_before_the_guard(monkeypatch, change):
    """`worker_excluded` that first lets a worker do `change` and end: what a
    turn completing between an inspection and the guard leaves behind."""
    real = worktree.worker_excluded

    def late(slot, ignore=False, **kwargs):
        change()
        return real(slot, ignore, **kwargs)

    monkeypatch.setattr(worktree, "worker_excluded", late)


def test_the_create_hook_reads_the_free_verdict_under_the_guard(repos, monkeypatch):
    """Review of c390e64: a verdict read before the guard described a tree a
    worker had changed since, and the hook reset it as free."""
    main, slot, base = repos
    free = worktree.Slot(2, slot, "wt2", "", False)
    monkeypatch.setattr(worktree, "list_slots", lambda: [free])
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree, "take_next_assignment", lambda: "")
    worker_finishing_before_the_guard(
        monkeypatch, lambda: (slot / "readme.txt").write_text("worker's edit", encoding="utf-8"))
    with pytest.raises(worktree.Fail, match="no free slot.*dirty"):
        worktree.choose_slot("agent", "r")
    assert "locked" not in run(main, "worktree", "list", "--porcelain")


def test_the_create_hook_checks_an_assignment_s_git_under_the_guard(repos, tmp_path,
                                                                    monkeypatch):
    main, slot, base = repos
    monkeypatch.setattr(worktree, "list_slots", lambda: [worktree.Slot(2, slot, "wt2", "", False)])
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree, "take_next_assignment", lambda: "wt2")
    worker_finishing_before_the_guard(monkeypatch, lambda: plant(slot, tmp_path))
    with pytest.raises(worktree.Fail, match="a git in it is not the one the slot registers"):
        worktree.choose_slot("agent", "r")


def test_the_remove_hook_reads_the_slot_under_the_guard(repos, monkeypatch, capsys):
    main, slot, base = repos
    monkeypatch.setattr(worktree, "list_slots", lambda: [worktree.Slot(2, slot, "wt2", "", False)])
    monkeypatch.setattr(worktree.sys, "stdin", io.StringIO(json.dumps({"path": str(slot)})))
    worker_finishing_before_the_guard(
        monkeypatch, lambda: (slot / "readme.txt").write_text("worker's edit", encoding="utf-8"))
    assert command("hook-remove") == 0
    assert "keeping it as is" in capsys.readouterr().err
    assert (slot / "readme.txt").read_text(encoding="utf-8") == "worker's edit"


@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions are Windows'")
def test_a_worker_rooted_at_an_alias_of_the_slot_refuses_it(locked_slot, tmp_path):
    import _winapi

    main, slot, pool_slot = locked_slot
    alias = tmp_path / "alias"
    _winapi.CreateJunction(str(slot), str(alias))
    try:
        with live_workers.recorded("worker", [alias / "src"]):
            with pytest.raises(worktree.Fail, match=LIVE_REFUSAL):
                command("land", "wt2", "--no-push")
    finally:
        os.rmdir(alias)
