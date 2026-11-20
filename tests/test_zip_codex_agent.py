"""Exercise archival and deletion on disposable agent state only."""

from pathlib import Path
import runpy
from zipfile import ZipFile

import pytest


@pytest.fixture
def archiver(monkeypatch):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "zip-codex-agent.py"))
    monkeypatch.setattr("builtins.input", lambda _: "ARCHIVE")
    return module["cli"], module["main"].__globals__


def test_archive_progress_registry_and_previous_zip(archiver, tmp_path, capsys):
    cli, _ = archiver
    (tmp_path / "one" / "nested").mkdir(parents=True)
    (tmp_path / "two").mkdir()
    (tmp_path / "one" / "nested" / "events.jsonl").write_bytes(b"event\n")
    (tmp_path / "registry.json").write_bytes(b'{"one": {}}')
    previous = tmp_path / "previous.zip"
    previous.write_bytes(b"previous archive")
    (tmp_path / "zip.py").write_bytes(b"launcher")

    assert cli(tmp_path) == 0
    assert "Archiving 2 folders: ..\n" in capsys.readouterr().out
    archive, = [path for path in tmp_path.glob("*.zip") if path != previous]
    assert archive.name.endswith("-2.zip")
    with ZipFile(archive) as saved:
        assert set(saved.namelist()) == {"one/", "one/nested/", "two/", "one/nested/events.jsonl", "registry.json"}
        assert saved.read("registry.json") == b'{"one": {}}'
        assert saved.read("one/nested/events.jsonl") == b"event\n"
    assert previous.read_bytes() == b"previous archive"
    assert (tmp_path / "zip.py").read_bytes() == b"launcher"
    assert not (tmp_path / "registry.json").exists()
    assert not (tmp_path / "one").exists()
    assert not (tmp_path / "two").exists()
    assert cli(tmp_path) == 0
    assert len(list(tmp_path.glob("*.zip"))) == 2


def test_registry_without_folders(archiver, tmp_path):
    cli, _ = archiver
    (tmp_path / "registry.json").write_bytes(b"{}")
    assert cli(tmp_path) == 0
    archive, = tmp_path.glob("*-0.zip")
    with ZipFile(archive) as saved:
        assert saved.read("registry.json") == b"{}"
    assert not (tmp_path / "registry.json").exists()


def test_cancel_keeps_state(archiver, tmp_path, monkeypatch):
    cli, _ = archiver
    monkeypatch.setattr("builtins.input", lambda _: "NO")
    (tmp_path / "one").mkdir()
    (tmp_path / "registry.json").write_bytes(b"{}")
    assert cli(tmp_path) == 0
    assert {path.name for path in tmp_path.iterdir()} == {"one", "registry.json"}


def test_write_failure_keeps_sources(archiver, tmp_path, monkeypatch):
    cli, _ = archiver
    (tmp_path / "one").mkdir()
    (tmp_path / "registry.json").write_bytes(b"{}")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ZipFile, "write", fail)
    assert cli(tmp_path) == 1
    assert (tmp_path / "one").is_dir()
    assert (tmp_path / "registry.json").read_bytes() == b"{}"


def test_delete_failure_keeps_registry(archiver, tmp_path, monkeypatch):
    cli, namespace = archiver
    (tmp_path / "one").mkdir()
    (tmp_path / "registry.json").write_bytes(b"{}")

    def fail(*args, **kwargs):
        raise PermissionError("folder locked")

    monkeypatch.setattr(namespace["shutil"], "rmtree", fail)
    assert cli(tmp_path) == 1
    assert (tmp_path / "registry.json").read_bytes() == b"{}"
