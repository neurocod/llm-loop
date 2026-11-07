"""agent-tools/agent_context_watch.py: the reading is the LAST usage record's sum."""

import json
import subprocess
import sys
from pathlib import Path

import agent_context_watch as watch

SCRIPT = Path(__file__).resolve().parents[1] / "agent-tools" / "agent_context_watch.py"
# A one-shot read of a two-line file ends in well under a second; the cap only
# stops a hang from stalling the suite.
CAP_S = 60


def _record(tokens: int) -> str:
    usage = {"input_tokens": tokens, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "output_tokens": 0,
             "output_tokens_details": {"reasoning": 7}}
    return json.dumps({"message": {"usage": usage}}, separators=(",", ":")) + "\n"


def test_fields_are_matched_whole_and_a_missing_one_reads_nothing():
    assert watch.tokens_of(_record(100).encode()) == 100
    assert watch.tokens_of(b'"usage":{"input_tokens":5}') is None


def test_the_reader_keeps_the_last_record_across_reads(tmp_path):
    path = tmp_path / "agent-x.jsonl"
    path.write_text(_record(100), encoding="utf-8")
    reader = watch.Reader(str(path))
    assert reader.poll() == 100
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_record(250))
    assert reader.poll() == 250


def test_once_prints_the_percentage_of_the_window(tmp_path):
    path = tmp_path / "agent-x.jsonl"
    path.write_text(_record(100) + _record(400_000), encoding="utf-8")
    done = subprocess.run([sys.executable, str(SCRIPT), str(path), "--once"],
                          capture_output=True, text=True, check=False, timeout=CAP_S)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "now: 40.0 % (400000 of 1000000 tokens)"
