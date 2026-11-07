"""Pin how the pool carries submodules along a gitlink bump: `land` into the
main checkout, and `equalize` / `next` / `new` into a slot.

Each test builds throwaway repositories: an upstream for the submodule, a main
checkout that records it at `sub`, and a pool slot `main-wt2` on branch `wt2`
whose own submodule clone makes the bumped commit (never pushed upstream, the
way a slot's submodule commit sits before its push).
"""

import io
import subprocess

import pytest

from llm_agents.git import worktree

_CONFIG = {
    "protocol.file.allow": "always",  # `submodule add/update` from a local path
    "user.name": "Land test",
    "user.email": "test@example.com",
    "commit.gpgsign": "false",
    "init.defaultBranch": "master",
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
    run(slot / "sub" / "inner", "pull", "-q", "origin", "master")
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
    monkeypatch.setattr(worktree, "push_master", lambda: pushed.append(True) or True)

    def broken(old_master, pool_slot):
        raise worktree.Fail("sync exploded")

    monkeypatch.setattr(worktree, "sync_main_submodules", broken)
    assert command("land", "wt2") == 0
    assert pushed == [True]
    assert "locked" not in run(main, "worktree", "list", "--porcelain")
    assert "sync exploded" in capsys.readouterr().err


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
    run(slot, "rebase", "-q", "master")
    assert command("next", "wt2") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == landed
    assert run(slot, "status", "--porcelain") == ""
    assert "follows the slot's gitlink" in capsys.readouterr().out


def test_next_keeps_a_slot_s_unrecorded_submodule_work(repos, capsys):
    main, slot, base = repos
    bump_in_main(main)
    run(slot, "rebase", "-q", "master")
    mine = commit_file(slot / "sub", "mine.txt", "unrecorded")
    assert command("next", "wt2") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == mine
    assert "not an ancestor of the gitlink" in capsys.readouterr().err


def test_a_failed_fetch_from_the_main_clone_is_one_warning_naming_the_fix(repos, monkeypatch, capsys):
    """A fetch that failed printed nothing, and the slot silently stayed on the
    old commit showing `M sub`."""
    main, slot, base = repos
    landed = bump_in_main(main)
    run(slot, "rebase", "-q", "master")
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
    run(slot, "rebase", "-q", "master")
    assert command("next", "wt2") == 0
    assert run(slot / "sub", "rev-parse", "HEAD") == pushed
    assert "warning" not in capsys.readouterr().err


def test_a_follow_git_refuses_warns_and_the_hook_still_starts_the_agent(repos, monkeypatch, capsys):
    """`next` records its assignment and the create hook claims the slot
    before the submodule follow: an abort there stranded both and the agent
    never started. Here the gitlink's commit adds a file the slot's clone has
    untracked, so `submodule update` refuses to overwrite it."""
    main, slot, base = repos
    landed = commit_file(main / "sub", "new.txt", "landed")
    run(main, "add", "sub")
    run(main, "commit", "-q", "-m", "bump sub with a new file")
    commit_file(slot, "work.txt", "task")
    run(slot, "rebase", "-q", "master")
    (slot / "sub" / "new.txt").write_text("mine, untracked\n", encoding="utf-8")

    assert command("next", "wt2") == 0
    err = capsys.readouterr().err
    assert "submodule update failed" in err and landed[:9] in err
    assert (main / ".git" / worktree.NEXT_FILE).read_text() == "wt2"

    pool_slot = worktree.Slot(2, slot, "wt2", "", False)
    monkeypatch.setattr(worktree, "list_slots", lambda: [pool_slot])
    monkeypatch.setattr(worktree, "is_permitted", lambda path: True)
    monkeypatch.setattr(worktree, "push_master", lambda: True)
    monkeypatch.setattr(worktree.sys, "stdin", io.StringIO('{"name": "agent", "session_id": "s"}'))
    printed = io.StringIO()
    monkeypatch.setattr(worktree.sys, "__stdout__", printed)
    assert command("hook-create") == 0
    assert printed.getvalue().strip() == str(slot)
    assert "submodule update failed" in capsys.readouterr().err
    assert (slot / "sub" / "new.txt").read_text(encoding="utf-8") == "mine, untracked\n"
    run(main, "worktree", "unlock", str(slot))
