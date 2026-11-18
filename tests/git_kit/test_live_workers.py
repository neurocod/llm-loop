"""Pin the live-worker record (`llm_agents.git.live_workers`): a record whose
process is alive names its roots, a record whose process is gone -- or whose
pid now belongs to another process -- is stale and never blocks, and a record
that cannot be read blocks rather than reads as absent."""

import json
import os
import subprocess
import sys

from llm_agents.git import live_workers


def write_record(name, roots, processes):
    folder = live_workers.live_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}{live_workers.RECORD_SUFFIX}"
    path.write_text(json.dumps({
        "name": name, "roots": [str(root) for root in roots], "since": "now",
        "processes": [{"pid": pid, "identity": identity} for pid, identity in processes],
    }), encoding="utf-8")
    return path


def ended_process():
    """(pid, identity) of a process that has exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = live_workers.process_identity(proc.pid)
    proc.wait()
    return proc.pid, identity


def test_a_record_of_a_live_process_names_its_roots_and_their_folders(tmp_path):
    slot, other = tmp_path / "repo-wt1", tmp_path / "repo-wt2"
    with live_workers.recorded("w", [slot / "src"]):
        found = live_workers.live_workers(slot)
        assert [w.name for w in found] == ["w"]
        assert found[0].processes == [(os.getpid(), live_workers.process_identity(os.getpid()))]
        assert live_workers.live_workers(other) == []
    assert live_workers.live_workers(slot) == []
    assert list(live_workers.live_dir().iterdir()) == []


def test_a_writable_root_above_the_slot_names_it(tmp_path):
    with live_workers.recorded("w", [tmp_path / "elsewhere", tmp_path]):
        assert len(live_workers.live_workers(tmp_path / "repo-wt1")) == 1


def test_a_server_the_wrapper_started_keeps_the_record_live(tmp_path):
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        dead = ended_process()
        path = write_record("w", [tmp_path], [dead, (server.pid, live_workers.process_identity(server.pid))])
        assert len(live_workers.live_workers(tmp_path)) == 1
    finally:
        server.kill()
        server.wait()
    assert live_workers.live_workers(tmp_path) == []
    assert not path.exists()


def test_a_dead_process_s_record_is_stale_and_pruned(tmp_path):
    path = write_record("w", [tmp_path], [ended_process()])
    assert live_workers.live_workers(tmp_path) == []
    assert not path.exists()


def test_a_reused_pid_with_another_creation_time_is_stale(tmp_path):
    """Windows hands a pid out again within seconds: the pid alone would read
    a stranger as the worker."""
    own = live_workers.process_identity(os.getpid())
    assert own is not None
    write_record("w", [tmp_path], [(os.getpid(), own + 1)])
    assert live_workers.live_workers(tmp_path) == []


def test_an_unreadable_record_blocks_whatever_it_named(tmp_path):
    folder = live_workers.live_dir()
    folder.mkdir(parents=True)
    (folder / f"torn{live_workers.RECORD_SUFFIX}").write_text("{", encoding="utf-8")
    found = live_workers.live_workers(tmp_path / "anything")
    assert len(found) == 1 and "unreadable record" in found[0].describe()
