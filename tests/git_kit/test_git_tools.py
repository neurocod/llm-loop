"""The kit's small pieces: the slot naming rule, the union resolver, the
gitlink check and the overlap report."""

import json

import pytest

from kit_support import git, out, repo as make_repo, run_kit
from llm_agents.git import check_gitlinks, slots, union_conflicts


def test_slot_names_round_trip_and_fold_case():
    assert slots.slot_dir_name("Repo", 3) == "Repo-wt3"
    assert slots.slot_number("Repo", "repo-WT3") == 3
    assert slots.slot_number("Repo", "Repo-wt") is None
    assert slots.slot_number("Repo", "Other-wt3") is None
    assert slots.SLOT_NAME_RE.match(slots.slot_name(12))


def test_pool_slot_is_a_linked_worktree_named_after_its_main(tmp_path):
    main = make_repo(tmp_path, "main")
    git(main, "commit", "-q", "--allow-empty", "-m", "Initial")
    git(main, "worktree", "add", "-q", "--detach", str(tmp_path / "main-wt1"))
    git(main, "worktree", "add", "-q", "--detach", str(tmp_path / "review"))
    assert slots.pool_slot(tmp_path / "main-wt1")
    assert not slots.pool_slot(tmp_path / "review")
    assert not slots.pool_slot(main)


TEXT = "a\n<<<<<<< ours\nx\n=======\ny\n>>>>>>> theirs\nb\n"


@pytest.mark.parametrize(("spec", "body"), [("ours,theirs", "x\ny\n"), ("theirs,ours", "y\nx\n"),
                                            ("ours", "x\n"), ("theirs", "y\n")])
def test_a_union_keeps_the_sides_in_the_order_asked(spec, body):
    assert union_conflicts.resolve(TEXT, {}, spec) == (f"a\n{body}b\n", 1)


def test_a_diff3_hunk_needs_a_human():
    with pytest.raises(SystemExit, match="diff3"):
        union_conflicts.resolve("<<<<<<< o\nx\n||||||| base\nz\n=======\ny\n>>>>>>> t\n", {}, "ours")


def test_the_union_cli_writes_lf_on_every_platform(tmp_path):
    """Read with universal newlines, written untranslated: no CRLF appears on
    Windows, and a CRLF file comes back LF."""
    path = tmp_path / "c.txt"
    path.write_bytes(TEXT.replace("\n", "\r\n").encode())
    done = run_kit("union_conflicts", str(path), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert path.read_bytes() == b"a\nx\ny\nb\n"


def test_an_undeclared_gitlink_is_named(tmp_path):
    repo = make_repo(tmp_path)
    nested = make_repo(repo, "vendor")
    git(nested, "commit", "-q", "--allow-empty", "-m", "Nested")
    git(repo, "add", "vendor")
    problems = check_gitlinks.check([], cwd=str(repo))
    assert len(problems) == 1 and problems[0].startswith("vendor: recorded as a submodule")


def test_the_overlap_report_names_files_both_sides_touched(tmp_path):
    repo = make_repo(tmp_path)
    for name in ("both.txt", "moved.txt"):
        (repo / name).write_text("base\n" * 20, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "Base")
    ours = out(repo, "symbolic-ref", "--short", "HEAD")
    git(repo, "branch", "theirs")
    (repo / "both.txt").write_text("ours\n", encoding="utf-8")
    (repo / "moved.txt").write_text("base\n" * 20 + "ours\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "Ours")
    git(repo, "checkout", "-q", "theirs")
    (repo / "both.txt").write_text("theirs\n", encoding="utf-8")
    git(repo, "mv", "moved.txt", "renamed.txt")
    git(repo, "commit", "-q", "-am", "Theirs")
    done = run_kit("branch_overlap", ours, "theirs", "--json", cwd=repo)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["both"] == ["both.txt", "moved.txt"]
    assert report["moved_by_theirs"] == [["moved.txt", "renamed.txt"]]
