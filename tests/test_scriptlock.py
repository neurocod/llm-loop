"""Exercise contention across real processes, including a forcibly killed owner."""

import atexit
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from llm_loop import cyclecore, parallel, runlifecycle, scriptlock


# Import + acquire + process exit: 2.903 s measured on Windows 2026-09-15.
# Leave room for multiple concurrent launches on slower hosts (> 2x measured).
PROCESS_TIMEOUT = 30
CHILD = """\
from pathlib import Path
import sys
from llm_loop import scriptlock
scriptlock.LOCK_DIR = Path(sys.argv[1])
scriptlock.ensure_script_lock()
scriptlock.ensure_script_lock()
print('READY', flush=True)
sys.stdin.readline()
"""


class Child:
    def __init__(self, path, lock_dir, cwd):
        env = dict(os.environ, PYTHONPATH=str(
            Path(scriptlock.__file__).resolve().parents[1]))
        self.proc = subprocess.Popen(
            [sys.executable, '-u', str(path), str(lock_dir)], cwd=cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def until(self, text):
        output = []
        while True:
            try:
                line = self.lines.get(timeout=PROCESS_TIMEOUT)
            except queue.Empty:
                pytest.fail(f"Timed out waiting for {text!r}; output: {''.join(output)!r}")
            assert line is not None, ''.join(output)
            output.append(line)
            if text in line:
                return ''.join(output)

    def send(self, text):
        self.proc.stdin.write(text + '\n')
        self.proc.stdin.flush()

    def wait(self, code=0):
        assert self.proc.wait(timeout=PROCESS_TIMEOUT) == code

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=PROCESS_TIMEOUT)
        self.reader.join(timeout=PROCESS_TIMEOUT)
        self.proc.stdin.close()
        self.proc.stdout.close()


@pytest.fixture(autouse=True)
def isolate_launch_decision(monkeypatch):
    launches = {}
    monkeypatch.setattr(scriptlock, '_launches', launches)
    yield
    for lock in launches.values():
        if lock is not None:
            atexit.unregister(lock.close)
            lock.close()


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.setattr(scriptlock, 'LOCK_DIR', tmp_path / 'locks')
    children = []

    def start(name='runCycle.py', *, cwd=None, path=None):
        script = tmp_path / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(CHILD, encoding='utf-8')
        child = Child(path or script, scriptlock.LOCK_DIR, cwd or tmp_path)
        children.append(child)
        return child

    yield start
    for child in children:
        child.close()


def test_same_script_offers_exit_without_starting_work(launch):
    owner = launch()
    owner.until('READY')
    other = launch()
    other.until('Another instance')
    other.send('invalid')
    assert '[e] Exit, [w] Wait, [i] Run independently' in other.until('Choose e, w or i')
    other.send('e')
    other.wait()
    assert owner.proc.poll() is None


@pytest.mark.parametrize('kill_owner', [False, True])
def test_wait_acquires_after_normal_or_forced_exit(launch, kill_owner):
    owner = launch()
    owner.until('READY')
    waiter = launch()
    waiter.until('Another instance')
    waiter.send('w')
    waiter.until('checking every 0.5 s')
    if kill_owner:
        owner.proc.kill()
        owner.proc.wait(timeout=PROCESS_TIMEOUT)
    else:
        owner.send('')
        owner.wait()
    waiter.until('READY')
    # The waiter owns the lock, rather than merely observing its release.
    third = launch()
    third.until('Another instance')
    third.send('e')
    third.wait()


def test_independent_launch_does_not_release_the_owner(launch):
    owner = launch()
    owner.until('READY')
    independent = launch()
    independent.until('Another instance')
    independent.send('i')
    independent.until('READY')
    third = launch()
    third.until('Another instance')
    third.send('e')
    third.wait()
    assert owner.proc.poll() is None
    assert independent.proc.poll() is None


def test_same_filename_in_different_directories_is_independent(launch):
    launch().until('READY')
    output = launch('refactor/runCycle.py').until('READY')
    assert 'Another instance' not in output


def test_relative_and_absolute_launch_paths_share_a_lock(launch, tmp_path):
    launch().until('READY')
    other = launch(cwd=tmp_path.parent, path=Path(tmp_path.name) / 'runCycle.py')
    other.until('Another instance')
    other.send('e')
    other.wait()


@pytest.mark.skipif(os.name != 'nt', reason='Windows paths ignore letter case')
def test_windows_case_alias_shares_a_lock(launch, tmp_path):
    launch().until('READY')
    other = launch(path=Path(str(tmp_path / 'runCycle.py').swapcase()))
    other.until('Another instance')
    other.send('e')
    other.wait()


def test_symlink_alias_shares_a_lock(launch, tmp_path):
    alias = tmp_path / 'alias.py'
    try:
        alias.symlink_to(tmp_path / 'runCycle.py')
    except OSError as exc:
        pytest.skip(f'Symlinks unavailable on this host: {exc}')
    launch().until('READY')
    other = launch(path=alias)
    other.until('Another instance')
    other.send('e')
    other.wait()


def test_batches_keep_the_initial_lock_after_changing_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['runCycle.py'])
    monkeypatch.setattr(scriptlock, 'LOCK_DIR', tmp_path / 'locks')
    scriptlock.ensure_script_lock()
    initial = dict(scriptlock._launches)
    other_directory = tmp_path / 'other'
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)
    scriptlock.ensure_script_lock()
    assert scriptlock._launches == initial
    assert len(list(scriptlock.LOCK_DIR.glob('*.lock'))) == 1


def test_eof_cannot_silently_start_a_duplicate(launch):
    launch().until('READY')
    other = launch()
    other.until('Another instance')
    other.proc.stdin.close()
    other.until('No choice received')
    other.wait(1)


def test_wait_uses_half_second_timer_and_cancel_is_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [str(tmp_path / 'runCycle.py')])
    monkeypatch.setattr(scriptlock, 'LOCK_DIR', tmp_path / 'locks')
    owner = scriptlock.ScriptLock(sys.argv[0])
    assert owner.acquire()
    monkeypatch.setattr('builtins.input', lambda _: 'w')
    sleeps = []

    def cancel(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(scriptlock.time, 'sleep', cancel)
    try:
        with pytest.raises(SystemExit) as stopped:
            scriptlock.ensure_script_lock()
        assert stopped.value.code == 130
        assert sleeps == [0.5]
        assert owner.script not in scriptlock._launches
    finally:
        owner.close()


@pytest.mark.parametrize('runner', [cyclecore.run_loop, parallel.run_parallel])
def test_both_runners_lock_before_any_startup_side_effect(runner, monkeypatch):
    class ReachedGuard(Exception):
        pass

    def guard():
        raise ReachedGuard

    def unexpected(*args, **kwargs):
        pytest.fail('Startup logging ran before acquiring the script lock')

    monkeypatch.setattr(runlifecycle, 'ensure_script_lock', guard)
    monkeypatch.setattr(runlifecycle.console, 'setup_file_logging', unexpected)
    monkeypatch.setattr(runlifecycle.exitlog, 'begin', unexpected)
    # Incomplete inputs deliberately fail if anything reads them before locking.
    with pytest.raises(ReachedGuard):
        runner(object(), SimpleNamespace(dry_run=False))


def test_dry_run_skips_the_guard(monkeypatch):
    def unexpected():
        pytest.fail('A preview must not contend with a real run')

    monkeypatch.setattr(runlifecycle, 'ensure_script_lock', unexpected)
    with pytest.raises(AttributeError, match='provider'):
        runlifecycle.begin_run(object(), SimpleNamespace(dry_run=True), 'preview')
