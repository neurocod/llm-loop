"""agent-tools/move_lines.py: the refusals that keep a line move from losing code."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "agent-tools" / "move_lines.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, check=False)


def _source(tmp_path: Path, data: bytes) -> Path:
    src = tmp_path / "src.txt"
    src.write_bytes(data)
    return src


SOURCE = b"one\ntwo\nthree\nfour\n"


@pytest.mark.parametrize("spelling", ["same", "alias"])
def test_refuses_the_source_as_its_own_destination(tmp_path, spelling):
    src = _source(tmp_path, SOURCE)
    sub = tmp_path / "sub"
    sub.mkdir()
    dst = src if spelling == "same" else sub / ".." / "src.txt"
    result = _run(str(src), str(dst), "2-3")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "source itself" in result.stdout
    assert src.read_bytes() == SOURCE


def test_refuses_a_hard_link_to_the_source(tmp_path):
    src = _source(tmp_path, SOURCE)
    link = tmp_path / "link.txt"
    try:
        os.link(src, link)
    except OSError as exc:
        pytest.skip(f"no hard links here: {exc}")
    result = _run(str(src), str(link), "2-3")
    assert result.returncode == 1, result.stdout + result.stderr
    assert src.read_bytes() == SOURCE


def test_refuses_overlapping_ranges_and_writes_nothing(tmp_path):
    src = _source(tmp_path, SOURCE)
    dst = tmp_path / "dst.txt"
    result = _run(str(src), str(dst), "1-2", "2-3")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "overlap" in result.stdout
    assert src.read_bytes() == SOURCE
    assert not dst.exists()


def test_refuses_a_range_past_the_end(tmp_path):
    src = _source(tmp_path, SOURCE)
    result = _run(str(src), str(tmp_path / "dst.txt"), "3-9")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "past the source" in result.stdout
    assert src.read_bytes() == SOURCE


def test_a_last_line_without_newline_does_not_join_the_next_range(tmp_path):
    src = _source(tmp_path, b"one\ntwo\nthree")
    dst = tmp_path / "dst.txt"
    result = _run(str(src), str(dst), "3-3", "1-1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert dst.read_bytes() == b"three\none\n"
    assert src.read_bytes() == b"two\n"


def test_the_added_newline_follows_the_source_line_endings(tmp_path):
    src = _source(tmp_path, b"one\r\ntwo\r\nthree")
    dst = tmp_path / "dst.txt"
    result = _run(str(src), str(dst), "3-3", "1-1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert dst.read_bytes() == b"three\r\none\r\n"


def test_a_plain_move_appends_in_the_order_given(tmp_path):
    src = _source(tmp_path, SOURCE)
    dst = tmp_path / "dst.txt"
    dst.write_bytes(b"head")
    result = _run(str(src), str(dst), "4-4", "2-2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert dst.read_bytes() == b"head\nfour\ntwo\n"
    assert src.read_bytes() == b"one\nthree\n"
