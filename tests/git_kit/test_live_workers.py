"""Pin the live-worker exclusion (`llm_agents.git.live_workers`): the two
guards exclude each other and nothing else, a record whose process -- or job
-- is alive names its roots, a stale one never blocks and is pruned without
touching a newer one, an unreadable record or folder blocks, aliases are the
path they name, and a worker that could write the folder is refused."""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from llm_agents.git import live_workers

from kit_support import kit_env


def write_record(name, roots, processes, job=""):
    folder = live_workers.live_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}{live_workers.RECORD_SUFFIX}"
    path.write_text(json.dumps({
        "name": name, "roots": [str(root) for root in roots], "since": "now", "job": job,
        "processes": [{"pid": pid, "identity": identity} for pid, identity in processes],
    }), encoding="utf-8")
    return path


def ended_process():
    """(pid, identity) of a process that has exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = live_workers.process_identity(proc.pid)
    proc.wait()
    return proc.pid, identity


# --------------------------------------------------------------------- guards

def test_a_worker_and_a_pool_command_on_one_slot_exclude_each_other(tmp_path):
    slot = tmp_path / "repo-wt1"
    with live_workers.worker_guard([slot / "src"]):
        with pytest.raises(live_workers.Busy):
            live_workers.pool_guard(slot)
    with live_workers.pool_guard(slot):
        with pytest.raises(live_workers.Busy):
            live_workers.worker_guard([slot])
    live_workers.worker_guard([slot]).release()


def test_two_pool_commands_on_one_slot_exclude_each_other_and_name_the_holder(tmp_path):
    """`pool_guard` is shared among pool commands; `slot_operation` is not."""
    slot, other = tmp_path / "repo-wt1", tmp_path / "repo-wt2"
    with live_workers.pool_guard(slot), live_workers.pool_guard(slot):
        pass
    with live_workers.slot_operation(slot, "worktree land wt1"):
        with pytest.raises(live_workers.Busy,
                           match=rf"pid {os.getpid()} \(worktree land wt1\) since "):
            live_workers.slot_operation(slot, "worktree next wt1", wait_s=0)
        live_workers.slot_operation(other, "worktree next wt2", wait_s=0).release()
    live_workers.slot_operation(slot, "worktree next wt1", wait_s=0).release()


def test_a_holder_line_naming_an_ended_process_reads_as_holder_unknown(tmp_path):
    """The line is advisory: a holder killed before it cleared it leaves it
    for the next one, which takes the lock before it writes its own."""
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = live_workers.process_identity(ended.pid)
    ended.wait()
    with live_workers.slot_operation(tmp_path, "worktree land wt1") as held:
        left = json.dumps({"pid": ended.pid, "identity": identity, "what": "worktree drop wt1",
                           "since": "earlier"}).encode("utf-8")
        os.ftruncate(held._lock._fd, 0)
        os.lseek(held._lock._fd, 0, os.SEEK_SET)
        os.write(held._lock._fd, b"\n" + left)
        with pytest.raises(live_workers.Busy,
                           match=rf"holder unknown \(the line names pid {ended.pid}, which has ended"):
            live_workers.slot_operation(tmp_path, "worktree next wt1", wait_s=0)
        os.ftruncate(held._lock._fd, 1)
        with pytest.raises(live_workers.Busy, match=r"holder unknown \(it has not named itself"):
            live_workers.slot_operation(tmp_path, "worktree next wt1", wait_s=0)


def test_a_slot_operation_waits_for_a_short_holder(tmp_path):
    first = live_workers.slot_operation(tmp_path, "first")
    timer = threading.Timer(0.3, first.release)
    timer.start()
    try:
        # 30 s: only turns a lock never let go into a red; the wait itself is
        # the timer's 0.3 s.
        live_workers.slot_operation(tmp_path, "second", wait_s=30).release()
    finally:
        timer.join()


HOLDER = textwrap.dedent("""
    import sys, time
    from llm_agents.git import live_workers
    live_workers.slot_operation(sys.argv[1], "worktree hook-create")
    print("held", flush=True)
    time.sleep(120)
""")


def test_a_killed_holder_leaves_no_slot_operation_behind(tmp_path):
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(tmp_path)],
                              stdout=subprocess.PIPE, text=True, env=kit_env())
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(live_workers.Busy, match=rf"pid {holder.pid} \(worktree hook-create\)"):
            live_workers.slot_operation(tmp_path, "worktree land", wait_s=0)
    finally:
        holder.kill()
        holder.wait()
    # 30 s: the OS releases a dead process's lock asynchronously; the bound
    # only turns a lock that outlives its process into a red.
    live_workers.slot_operation(tmp_path, "worktree land", wait_s=30).release()


def test_a_worker_allowed_to_write_a_parent_holds_the_slot(tmp_path):
    with live_workers.worker_guard([tmp_path]):
        with pytest.raises(live_workers.Busy):
            live_workers.pool_guard(tmp_path / "repo-wt1")


def test_parties_of_one_side_and_other_slots_do_not_exclude(tmp_path):
    one, two = tmp_path / "repo-wt1", tmp_path / "repo-wt2"
    with live_workers.worker_guard([one]), live_workers.worker_guard([one / "src"]), \
            live_workers.pool_guard(two), live_workers.pool_guard(tmp_path / "repo-wt3"):
        with pytest.raises(live_workers.Busy):
            live_workers.pool_guard(one)


# -------------------------------------------------------------------- records

def test_a_record_of_a_live_process_names_its_roots_and_their_folders(tmp_path):
    slot, other = tmp_path / "repo-wt1", tmp_path / "repo-wt2"
    with live_workers.recorded("w", [slot / "src"]):
        found = live_workers.live_workers(slot)
        assert [w.name for w in found] == ["w"]
        assert found[0].processes == [(os.getpid(), live_workers.process_identity(os.getpid()))]
        assert live_workers.live_workers(other) == []
    assert live_workers.live_workers(slot) == []
    assert not any(p.name.endswith(live_workers.RECORD_SUFFIX)
                   for p in live_workers.live_dir().iterdir())


def test_a_shutdown_not_confirmed_keeps_the_record(tmp_path):
    with live_workers.recorded("w", [tmp_path], ended=lambda: False):
        pass
    assert [w.name for w in live_workers.live_workers(tmp_path)] == ["w"]


def test_a_dead_process_s_record_is_stale_and_pruned(tmp_path):
    path = write_record("w", [tmp_path], [ended_process()])
    assert live_workers.live_workers(tmp_path) == []
    assert not path.exists()


@pytest.mark.skipif(os.name != "nt" and not Path("/proc/self/stat").exists(),
                    reason="no process creation time here (macOS): `process_identity` "
                           "is None by contract and the pid alone is trusted")
def test_a_reused_pid_with_another_creation_time_is_stale(tmp_path):
    """Windows hands a pid out again within seconds: the pid alone would read
    a stranger as the worker."""
    own = live_workers.process_identity(os.getpid())
    assert own is not None
    write_record("w", [tmp_path], [(os.getpid(), own + 1)])
    assert live_workers.live_workers(tmp_path) == []


def test_adding_a_process_never_rewrites_a_record_a_reader_may_prune(tmp_path):
    """A reader that loaded the wrapper's file, saw its pid dead and unlinked
    it must not take the server's entry with it: each lives in its own file."""
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with live_workers.recorded("w", [tmp_path]) as record:
            first = record.paths[0]
            before = first.read_bytes()
            record.add_process(server.pid)
            assert first.read_bytes() == before and len(record.paths) == 2
            first.unlink()     # what pruning the wrapper's (dead) file does
            assert len(live_workers.live_workers(tmp_path)) == 1
    finally:
        server.kill()
        server.wait()


def test_an_unreadable_record_blocks_whatever_it_named(tmp_path):
    folder = live_workers.live_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"torn{live_workers.RECORD_SUFFIX}").write_text("{", encoding="utf-8")
    found = live_workers.live_workers(tmp_path / "anything")
    assert len(found) == 1 and "unreadable record" in found[0].describe()


def test_a_folder_that_cannot_be_listed_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv(live_workers.LIVE_DIR_ENV, str(tmp_path / "absent"))
    assert live_workers.live_workers(tmp_path) == []
    (tmp_path / "file").write_text("not a folder")
    monkeypatch.setenv(live_workers.LIVE_DIR_ENV, str(tmp_path / "file"))
    with pytest.raises(OSError):
        live_workers.live_workers(tmp_path)


# ---------------------------------------------------------------- paths

@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions are Windows'")
def test_a_junction_is_the_folder_it_names(tmp_path):
    import _winapi

    slot = tmp_path / "repo-wt1"
    slot.mkdir()
    alias = tmp_path / "alias"
    _winapi.CreateJunction(str(slot), str(alias))
    try:
        assert live_workers.canonical(alias / "src") == live_workers.canonical(slot / "src")
        with live_workers.worker_guard([alias]):
            with pytest.raises(live_workers.Busy):
                live_workers.pool_guard(slot)
    finally:
        os.rmdir(alias)


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 names are Windows'")
def test_a_short_name_is_the_folder_it_names(tmp_path):
    import ctypes

    slot = tmp_path / "a long slot name"
    slot.mkdir()
    buffer = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetShortPathNameW(str(slot), buffer, 1024)
    if buffer.value.lower() == str(slot).lower():
        pytest.skip("8.3 names are off on this volume")
    assert live_workers.canonical(buffer.value) == live_workers.canonical(slot)


EXTENDED = "\\\\?\\"


@pytest.mark.skipif(sys.platform != "win32", reason="extended-length paths are Windows'")
def test_an_extended_length_spelling_is_the_same_folder(tmp_path, monkeypatch):
    """`realpath` keeps a `\\\\?\\` it is given, and one checkout was read back
    both ways (review of c390e64): exclusion and overlap must not miss it."""
    slot = tmp_path / "repo-wt1"
    slot.mkdir()
    assert live_workers.canonical(EXTENDED + str(slot)) == live_workers.canonical(slot)
    assert (live_workers._plain_dos("\\\\?\\UNC\\host\\share\\x")
            == live_workers._plain_dos("\\\\host\\share\\x") == "\\\\host\\share\\x")
    with live_workers.worker_guard([EXTENDED + str(slot)]):
        with pytest.raises(live_workers.Busy):
            live_workers.pool_guard(slot)
    with live_workers.worker_guard([slot]):
        with pytest.raises(live_workers.Busy):
            live_workers.pool_guard(Path(EXTENDED + str(slot)))
    monkeypatch.setenv(live_workers.LIVE_DIR_ENV, EXTENDED + str(tmp_path / "work" / "registry"))
    with pytest.raises(SystemExit, match="overlaps"):
        live_workers.refuse_overlap([tmp_path / "work"])
    monkeypatch.setenv(live_workers.LIVE_DIR_ENV, str(tmp_path / "work" / "registry"))
    with pytest.raises(SystemExit, match="overlaps"):
        live_workers.refuse_overlap([Path(EXTENDED + str(tmp_path / "work"))])


def test_a_worker_that_could_write_the_folder_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(live_workers.LIVE_DIR_ENV, str(tmp_path / "work" / "registry"))
    with pytest.raises(SystemExit, match="overlaps the live-worker folder"):
        live_workers.refuse_overlap([tmp_path / "elsewhere", tmp_path / "work"])
    with pytest.raises(SystemExit, match="overlaps"):
        live_workers.refuse_overlap([tmp_path / "work" / "registry" / "x"])
    live_workers.refuse_overlap([tmp_path / "elsewhere"])


# ----------------------------------------------------------------------- jobs

GRANDCHILD = "import time; time.sleep(120)"
PARENT = textwrap.dedent(f"""
    import subprocess, sys
    child = subprocess.Popen([sys.executable, "-c", {GRANDCHILD!r}])
    print(child.pid, flush=True)
""")


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows'")
def test_a_descendant_that_outlives_its_parent_keeps_the_record_live(tmp_path):
    """The shim's pid is all the record names; when the shim is gone, the
    job still counts the Codex process it left, and the record stays live."""
    job = live_workers.WorkerJob.create()
    try:
        parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE,
                                  text=True, creationflags=job.creationflags)
        job.adopt(parent)
        grandchild = int(parent.stdout.readline())
        parent.wait()
        write_record("w", [tmp_path], [(parent.pid, None)], job=job.name)
        assert live_workers.same_process(grandchild, None)
        assert len(live_workers.live_workers(tmp_path)) == 1
        assert job.end(10.0)
        assert live_workers.live_workers(tmp_path) == []
    finally:
        job.close()


WRAPPER = textwrap.dedent(f"""
    import subprocess, sys, time
    from pathlib import Path
    from llm_agents.git import live_workers
    job = live_workers.WorkerJob.create()
    with live_workers.recorded("w", [Path(sys.argv[1])], job=job):
        shim = subprocess.Popen([sys.executable, "-c", {PARENT!r}], stdout=subprocess.PIPE,
                                text=True, creationflags=job.creationflags)
        job.adopt(shim)
        print(shim.stdout.readline().strip(), flush=True)
        time.sleep(120)
""")


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows'")
def test_a_wrapper_killed_outright_takes_its_whole_tree_with_it(tmp_path):
    """`taskkill /F` of the wrapper: no `finally` runs, yet the job closes
    with the process, every descendant ends, and the guard and the record
    stop holding the slot."""
    wrapper = subprocess.Popen([sys.executable, "-c", WRAPPER, str(tmp_path)],
                               stdout=subprocess.PIPE, text=True, env=kit_env())
    grandchild = int(wrapper.stdout.readline())
    with pytest.raises(live_workers.Busy):
        live_workers.pool_guard(tmp_path)
    wrapper.kill()
    wrapper.wait()
    # 10 s: termination is asynchronous; 0.2 ms after `wait` returned,
    # measured 2026-11-18 -- the bound only turns a hang into a red.
    deadline = time.monotonic() + 10
    while live_workers.same_process(grandchild, None):
        assert time.monotonic() < deadline, "the grandchild outlived the wrapper"
    assert live_workers.live_workers(tmp_path) == []
    live_workers.pool_guard(tmp_path).release()
