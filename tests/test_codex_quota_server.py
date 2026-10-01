"""`CodexUsageSource` keeps one private quota server and never orphans it."""

import gc
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from llm_loop import codex_usage, runlifecycle

FAKE = os.path.join(os.path.dirname(__file__), "_fake_codex_app_server.py")


@pytest.fixture
def pid_log(tmp_path):
    return tmp_path / "pids.txt"


@pytest.fixture
def sent(monkeypatch):
    """Every message written to any quota server, in order."""
    messages = []
    real_write = codex_usage._QuotaServer.write

    def recording_write(self, message):
        messages.append(message)
        real_write(self, message)

    monkeypatch.setattr(codex_usage._QuotaServer, "write", recording_write)
    return messages


def _source(pid_log, mode="ok", **kwargs):
    return codex_usage.CodexUsageSource(
        argv=lambda: [sys.executable, FAKE, mode, str(pid_log)], **kwargs)


def _starts(pid_log) -> int:
    return len(pid_log.read_text().split()) if pid_log.exists() else 0


def _percent(source) -> float:
    return source.get_usage(cache_value=False).week_all.percent


def test_cache_misses_reuse_one_server_with_growing_ids(pid_log):
    source = _source(pid_log)
    try:
        assert [_percent(source) for _ in range(3)] == [1, 2, 3]
        assert _starts(pid_log) == 1
        # initialize took id 0, the three reads 1..3
        assert source._next_id == 4
    finally:
        source.close()


def test_cached_read_does_not_touch_the_server(pid_log):
    source = _source(pid_log)
    try:
        assert source.get_usage().week_all.percent == 1
        assert source.get_usage().week_all.percent == 1
        source.invalidate()
        assert source.get_usage().week_all.percent == 2
    finally:
        source.close()


def test_gate_and_refresher_reads_are_serialized(pid_log, sent):
    source = _source(pid_log)
    results = []
    barrier = threading.Barrier(8)

    def reader():
        barrier.wait()
        results.append(_percent(source))

    threads = [threading.Thread(target=reader) for _ in range(8)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        # Every reader got its own reply, none another's: one server, eight
        # distinct answers, ids issued strictly in order.
        assert sorted(results) == list(range(1, 9))
        ids = [m["id"] for m in sent if "id" in m]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
        assert _starts(pid_log) == 1
    finally:
        source.close()


def test_eof_on_an_idle_server_restarts_it_once(pid_log, capsys):
    source = _source(pid_log, mode="exit-after-read")
    try:
        assert _percent(source) == 1
        # That server exited after its reply; the next read finds EOF (or a
        # broken pipe) on it and is retried on a fresh one, not lost.
        assert _percent(source) == 1
        assert _starts(pid_log) == 2
        assert "no Codex usage figures" not in capsys.readouterr().out
    finally:
        source.close()


def _alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x00100000, False, pid)   # SYNCHRONIZE
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def test_timeout_is_bounded_and_kills_the_server(pid_log, capsys):
    # The fake outlives stdin EOF, so the kill escalation in `stop` is what
    # ends it.
    source = _source(pid_log, mode="hang-ignore-eof", timeout=1.0)
    try:
        started = time.monotonic()
        assert source.get_usage(cache_value=False).week_all.percent is None
        # 1.0 s timeout + 2 s grace on stdin EOF + the kill; the fake would
        # otherwise live 60 s.
        assert time.monotonic() - started < 10
        assert "timed out" in capsys.readouterr().out
        assert source._server is None
        assert not _alive(int(pid_log.read_text().split()[0]))
        source.get_usage(cache_value=False)
        assert _starts(pid_log) == 2           # restarted, not reused
    finally:
        source.close()


@pytest.mark.skipif(os.name != "nt", reason="npm .cmd shims are Windows-only")
def test_timeout_behind_an_npm_shim_ends_the_grandchild(pid_log, tmp_path):
    # Real `codex` resolves to npm's `codex.cmd`: the handle is cmd.exe and
    # the CLI a grandchild holding stdout. Killing only the shim would leave
    # it running and block `stop` on the stdout close for its whole life.
    shim = tmp_path / "codex.cmd"
    shim.write_text(f'@"{sys.executable}" "{FAKE}" %*\r\n', encoding="utf-8")
    source = codex_usage.CodexUsageSource(
        argv=lambda: [str(shim), "hang-ignore-eof", str(pid_log)], timeout=1.0)
    try:
        started = time.monotonic()
        assert source.get_usage(cache_value=False).week_all.percent is None
        assert time.monotonic() - started < 15
        assert not _alive(int(pid_log.read_text().split()[0]))
    finally:
        source.close()


def test_eof_on_a_server_started_by_this_query_is_not_retried(pid_log, capsys):
    modes = iter(["ok", "exit-on-read", "exit-on-read"])
    source = codex_usage.CodexUsageSource(
        argv=lambda: [sys.executable, FAKE, next(modes), str(pid_log)],
        max_age=0.0)
    try:
        assert _percent(source) == 1
        # The over-age server is replaced inside this query; the replacement
        # dying is not "a reused server that died while idle".
        assert source.get_usage(cache_value=False).week_all.percent is None
        assert "no Codex usage figures" in capsys.readouterr().out
        assert _starts(pid_log) == 2
    finally:
        source.close()


def test_max_age_replaces_the_server(pid_log):
    source = _source(pid_log, max_age=0.0)
    try:
        assert _percent(source) == 1
        assert _percent(source) == 1
        assert _starts(pid_log) == 2
    finally:
        source.close()


def test_close_reaps_the_server_and_refuses_to_start_another(pid_log):
    source = _source(pid_log)
    _percent(source)
    proc = source._server.proc
    source.close()
    assert proc.poll() is not None
    # A status-line poll racing the epilogue keeps the last figures and
    # starts nothing.
    assert source.get_usage(cache_value=False).week_all.percent == 1
    assert _starts(pid_log) == 1
    source.close()                              # idempotent


def test_atexit_hook_reaps_a_source_nobody_closed(pid_log):
    source = _source(pid_log)
    _percent(source)
    proc = source._server.proc
    codex_usage._close_open_sources()
    assert proc.poll() is not None
    assert source not in codex_usage._open_sources


def test_atexit_hook_reaps_a_source_dropped_without_close(pid_log):
    source = _source(pid_log)
    _percent(source)
    proc = source._server.proc
    del source                       # an exception unwound the frame holding it
    gc.collect()
    codex_usage._close_open_sources()
    assert proc.poll() is not None


def test_close_run_closes_every_source_dry_run_included(pid_log):
    sources = [_source(pid_log), _source(pid_log)]
    procs = []
    for source in sources:
        _percent(source)
        procs.append(source._server.proc)
    usages = (SimpleNamespace(source=source, name=f"acct{i}")
              for i, source in enumerate(sources))
    ctx = SimpleNamespace(dry_run=True)
    runlifecycle.close_run(ctx, usages=[None, *usages], mailbox=None)
    assert all(proc.poll() is not None for proc in procs)


def test_start_failure_degrades_to_no_figures(capsys):
    source = codex_usage.CodexUsageSource(
        argv=lambda: [os.path.join(os.getcwd(), "no-such-codex-binary")])
    try:
        assert source.get_usage().week_all.percent is None
        assert "could not start 'codex app-server'" in capsys.readouterr().out
    finally:
        source.close()


def test_wire_messages_of_one_query(pid_log, sent):
    source = _source(pid_log)
    try:
        _percent(source)
    finally:
        source.close()
    assert [m["method"] for m in sent] == [
        "initialize", "initialized", "account/rateLimits/read"]
