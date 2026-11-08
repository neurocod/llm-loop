"""Stale file renewal scenarios.

The owners are stubbed (`_renewing_judge`), so the copy, the concurrent-write
checks and the stale-directory rule run on any platform; only the real owner
reader and the hidden attribute need Windows, and say so where they do.
"""

from llm_agents.codex.testing.fixtures import no_badge  # noqa: F401 — see no_badge's docstring
import os
import subprocess
from pathlib import Path
import pytest
from llm_agents.codex import file_renewal, sandbox
from llm_agents.codex.registry import AgentRecord


ADMINISTRATORS = "S-1-5-32-544"


def _renewing_judge(tmp_path, monkeypatch, stale=True, stale_dir=False,
                    own_owner="S-1-5-21-user"):
    """A file judge over `project/old.ts` whose owner reads as the sandbox
    account's (`stale`) or `own_owner`, one of this process's own SIDs; the
    directory is the user's unless `stale_dir`."""
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    project = tmp_path / "project"
    project.mkdir()
    old = project / "old.ts"
    old.write_bytes(b"one\r\ntwo\xff\n")
    # An elevated administrator's own SIDs: its user, and Administrators as
    # the default owner of its new files.
    monkeypatch.setattr(file_renewal, "_own_sids",
                        lambda: frozenset({"S-1-5-21-user", ADMINISTRATORS}))
    monkeypatch.setattr(file_renewal, "_owner_sid", lambda path: (
        "S-1-5-21-codexsandbox" if (stale_dir if path.is_dir() else stale)
        else own_owner))
    judge = sandbox.file_change_judge(AgentRecord(
        name="w", cwd=str(project), sandbox="workspace-write"))
    return project, old, judge


def test_a_granted_edit_renews_its_stale_files(tmp_path, monkeypatch):
    """Measured 2026-11-03: the agent's own new files turned unwritable because they
    are owned by the sandbox account and kept capability SIDs Codex had since
    regenerated. A granted retry replaces each STALE file (owned by someone
    else) by a byte-identical NEW file — owned by the real user, inheriting
    the directory's current ACEs — keeping its mtime and hidden attribute;
    a file the user owns is left alone, and no temporary file stays."""
    project, old, judge = _renewing_judge(tmp_path, monkeypatch)
    stamp = os.stat(old).st_mtime_ns
    before = os.stat(old).st_ino
    if os.name == "nt":
        subprocess.run(["attrib", "+h", str(old)], check=True)
    verdict = judge([str(old), str(project / "new.ts")])
    assert verdict and "renewed 1 file(s)" in verdict.reason
    assert old.read_bytes() == b"one\r\ntwo\xff\n"
    assert os.stat(old).st_ino != before
    assert os.stat(old).st_mtime_ns == stamp
    if os.name == "nt":
        assert os.stat(old).st_file_attributes & 0x2  # hidden
    assert not (project / "new.ts").exists()
    assert [p.name for p in project.iterdir()] == ["old.ts"]  # no temp left
    assert "stale access list" in sandbox._edit_escalation_fact([project])


@pytest.mark.parametrize("own_owner", ["S-1-5-21-user", ADMINISTRATORS])
def test_a_file_the_process_owns_is_not_renewed(tmp_path, monkeypatch, own_owner):
    """windows-latest CI 2026-11-08: an elevated run's own files are owned by
    Administrators, not its user; they are its own all the same."""
    project, old, judge = _renewing_judge(tmp_path, monkeypatch, stale=False,
                                          own_owner=own_owner)
    before = os.stat(old).st_ino
    # Nothing to do, not a kept-with-a-reason: an Administrators-owned
    # directory once read as "stale too" for it.
    assert file_renewal.renew_file(old) == (False, "")
    verdict = judge([str(old)])
    assert verdict and "renewed" not in verdict.reason
    assert os.stat(old).st_ino == before


def test_own_sids_are_the_user_and_the_default_owner(monkeypatch):
    """Off an elevated machine both token SIDs are one, so only this stub
    tells a set missing the default owner from the right one."""
    sids = {1: "S-1-5-21-user", 4: ADMINISTRATORS}
    monkeypatch.setattr(file_renewal, "_token_sid", sids.get)
    assert file_renewal._own_sids() == {"S-1-5-21-user", ADMINISTRATORS}
    monkeypatch.setattr(file_renewal, "_token_sid", lambda info_class: None)
    assert file_renewal._own_sids() == frozenset()


@pytest.mark.skipif(os.name != "nt", reason=(
    "Windows-only: the owner SIDs are read through advapi32; elsewhere "
    "`_owner_sid` is None and `_own_sids` empty by design"))
def test_the_real_owner_reader_tells_the_user_from_the_system(tmp_path):
    """A file this process just wrote is its own, and a system file is not —
    elevated or not (an elevated one's file is owned by Administrators)."""
    fresh = tmp_path / "fresh.txt"
    fresh.write_text("x")
    own = file_renewal._own_sids()
    # TokenOwner = 4: the default owner is what a new file gets.
    assert file_renewal._owner_sid(fresh) == file_renewal._token_sid(4) in own
    system = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "notepad.exe"
    assert file_renewal._owner_sid(system) not in (None, *own)


def test_a_write_during_the_renewal_is_kept_not_lost(tmp_path, monkeypatch):
    """Review 2026-11-03: a write between reading the file and replacing it
    was silently lost. Now the file is kept, the write stands, the reason is
    named, and the temporary file is gone."""
    project, old, judge = _renewing_judge(tmp_path, monkeypatch)
    real_utime = os.utime

    def write_meanwhile(*args, **kwargs):
        with open(old, "ab") as late:
            late.write(b"late\n")
        return real_utime(*args, **kwargs)

    monkeypatch.setattr(file_renewal.os, "utime", write_meanwhile)
    verdict = judge([str(old)])
    assert verdict and "kept stale: old.ts (changed while being renewed)" \
        in verdict.reason
    assert old.read_bytes() == b"one\r\ntwo\xff\nlate\n"
    assert [p.name for p in project.iterdir()] == ["old.ts"]


def test_a_write_during_the_final_hash_is_kept_not_lost(tmp_path, monkeypatch):
    """Second review (2026-11-03): the size/mtime check ran BEFORE the
    final hash, so a write into a part the hash had already read left the
    digest equal to the copy's and the replace discarded the write. The
    check now follows the hash."""
    project, old, judge = _renewing_judge(tmp_path, monkeypatch)
    real_digest = file_renewal._digest

    def write_behind_the_hash(path):
        digest = real_digest(path)  # the hash has read the old bytes...
        old.write_bytes(b"WXYZ\r\ntwo\xff\n")  # ...then a same-size write lands
        stamp = os.stat(old).st_mtime_ns + 10_000_000
        os.utime(old, ns=(stamp, stamp))
        return digest

    monkeypatch.setattr(file_renewal, "_digest", write_behind_the_hash)
    verdict = judge([str(old)])
    assert verdict and "kept stale: old.ts (changed while being renewed)" \
        in verdict.reason
    assert old.read_bytes() == b"WXYZ\r\ntwo\xff\n"
    assert [p.name for p in project.iterdir()] == ["old.ts"]


def test_a_file_in_a_stale_directory_is_reported_not_renewed(tmp_path, monkeypatch):
    """Second review (2026-11-03): in a directory the sandbox account owns,
    the copy inherits that directory's stale access list, so "renewed" was
    a claim the file did not bear out. The file is kept and the reason
    names the directory."""
    project, old, judge = _renewing_judge(tmp_path, monkeypatch, stale_dir=True)
    before = os.stat(old).st_ino
    verdict = judge([str(old)])
    assert verdict and "renewed 1" not in verdict.reason
    assert "its directory is stale too" in verdict.reason
    assert os.stat(old).st_ino == before


def test_a_directory_someone_else_owns_is_not_stale(tmp_path, monkeypatch):
    """Third review (2026-11-03): "stale directory" meant any owner but the
    user, so a tree owned by Administrators never renewed and blamed the
    sandbox. Stale is owned by the file's own owner, the sandbox account."""
    project, old, judge = _renewing_judge(tmp_path, monkeypatch)
    monkeypatch.setattr(file_renewal, "_owner_sid", lambda path: (
        "S-1-5-32-544" if path.is_dir() else "S-1-5-21-codexsandbox"))
    verdict = judge([str(old)])
    assert verdict and "renewed 1 file(s)" in verdict.reason, verdict.reason
    assert old.read_bytes() == b"one\r\ntwo\xff\n"


def test_a_failed_renewal_leaves_the_file_and_no_temp(tmp_path, monkeypatch):
    project, old, judge = _renewing_judge(tmp_path, monkeypatch)

    def refuse(*_args):
        raise PermissionError("access denied")

    monkeypatch.setattr(file_renewal.os, "replace", refuse)
    kept = judge([str(old)])
    assert kept and "renewed 1" not in kept.reason
    assert "PermissionError" in kept.reason
    assert old.read_bytes() == b"one\r\ntwo\xff\n"
    assert [p.name for p in project.iterdir()] == ["old.ts"]
