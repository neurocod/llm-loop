"""Parallel startup waits precede usage/provider work and do not affect previews."""

import pytest

from llm_loop import cyclecore, parallel, runlifecycle, stopchannel

from _runfixtures import MemListDriver, isolated_run, par_args


@pytest.mark.parametrize("status_enabled", [False, True])
def test_parallel_start_delay_runs_once_after_stop_wait_before_claims(
        tmp_path, monkeypatch, status_enabled):
    order = []
    driver = MemListDriver(["a.md", "b.md"])
    monkeypatch.setattr(stopchannel, "wait_for_stop_file_clear",
                        lambda: order.append("stop"))

    def wait(spec, *, interactive):
        assert spec == "29m"
        assert interactive is status_enabled
        assert driver.pending_lines() == ["a.md", "b.md"]
        order.append("delay")

    def usage(provider):
        order.append("usage")
        return None

    def provider(*args):
        order.append("provider")
        return 0, None, None

    monkeypatch.setattr(cyclecore, "wait_before_start", wait)
    monkeypatch.setattr(runlifecycle, "usage_source_for", usage)
    monkeypatch.setattr(parallel, "run_job", provider)
    with isolated_run(monkeypatch, tmp_path):
        parallel.run_parallel(
            driver, par_args(tmp_path, jobs=1, start_in="29m", ignore_usage=False,
                             no_statusline=not status_enabled),
            app_name="pytest-parallel-start")
    assert order == ["stop", "delay", "usage", "provider", "provider"]


@pytest.mark.parametrize("dry_run,start_in", [(True, "29m"), (False, None)])
def test_parallel_preview_or_consumed_delay_does_not_wait(
        tmp_path, monkeypatch, dry_run, start_in):
    def refuse_wait(*args, **kwargs):
        pytest.fail("a preview or a consumed delay must not wait")

    monkeypatch.setattr(cyclecore, "wait_before_start", refuse_wait)
    monkeypatch.setattr(parallel, "run_job", lambda *args: (0, None, None))
    with isolated_run(monkeypatch, tmp_path):
        parallel.run_parallel(
            MemListDriver(["a.md"]),
            par_args(tmp_path, jobs=1, start_in=start_in, dry_run=dry_run,
                     no_statusline=True),
            app_name="pytest-parallel-start-skip", wait_on_start=False)
