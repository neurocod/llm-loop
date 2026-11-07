"""agent-tools/find_refs.py and rename_gate.py on a throwaway repository."""

import subprocess
import sys
from pathlib import Path

import pytest

import find_refs

TOOLS = Path(__file__).resolve().parents[1] / "agent-tools"
# Each run is two git greps over a three-file repository, well under a second;
# the cap only stops a hung git from stalling the suite.
CAP_S = 60


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    (root / "live.txt").write_text("staleName\n", encoding="utf-8")
    (root / "notes").mkdir()
    (root / "notes" / "past.md").write_text("staleName once\n", encoding="utf-8")
    (root / "copies").mkdir()
    (root / "copies" / "entry.json").write_text('{"original": "staleName"}\n', encoding="utf-8")
    return root


def _gate(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOLS / "rename_gate.py"), *args],
                          cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          check=False, timeout=CAP_S)


def test_an_excluded_glob_is_never_a_hit(repo, capsys):
    status = find_refs.run(["staleName"], substring=False, list_all=False, globs=[],
                           gate=True, repo=str(repo), excludes=["**/copies/**"])
    out = capsys.readouterr().out
    assert status == 1, out  # live.txt: the search itself saw the tree
    assert "live.txt:1:staleName" in out
    assert "entry.json" not in out


def test_an_aside_hit_is_listed_but_never_blocks(repo, capsys):
    (repo / "live.txt").write_text("fresh\n", encoding="utf-8")
    status = find_refs.run(["staleName"], substring=False, list_all=False, globs=[],
                           gate=True, repo=str(repo), asides=[("notes/", "the notes")],
                           excludes=["**/copies/**"])
    out = capsys.readouterr().out
    assert status == 0, out
    assert "1 hit(s) in notes/ (the notes) -- not blocking:" in out
    assert "clean: nothing outside the notes" in out


def test_the_gate_runs_from_anywhere_with_repo_and_a_later_repo_wins(repo, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = _gate("--repo", str(elsewhere), "--repo", str(repo), "staleName", cwd=tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "live.txt:1:staleName" in done.stdout
    assert "3 live hit(s)" in done.stdout


def test_without_a_repository_the_gate_judges_nothing(tmp_path):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    done = _gate("--repo", str(lonely), "staleName", cwd=lonely)
    assert done.returncode == find_refs.NO_REPO_EXIT, done.stdout + done.stderr
    assert "not inside a git work tree" in done.stderr


def test_a_malformed_aside_is_refused():
    with pytest.raises(SystemExit):
        find_refs.build_parser("x", with_glob=False).parse_args(["--aside", "notes/", "n"])
