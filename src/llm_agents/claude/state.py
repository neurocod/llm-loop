"""Durable named Claude sessions and atomically claimed turn queues."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid


def validate_name(name: str) -> str:
    """Names also become log directories, including on Windows."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
        raise ValueError("Agent names must start with a letter, digit, '_' or '-' and contain only letters, digits, '_', '-' or '.'")
    devices = {"CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"}
    devices.update(f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10))
    if name.endswith(".") or name.split(".")[0].upper() in devices:
        raise ValueError(f"Invalid Windows directory name: {name!r}")
    return name


class Store:
    """completed_turn identifies the latest successful answer, not any ended turn."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "agents.sqlite3"
        with self._transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS agents (
                name TEXT PRIMARY KEY COLLATE NOCASE, cwd TEXT NOT NULL,
                options TEXT NOT NULL, session_id TEXT NOT NULL,
                session_started INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL, owner TEXT, cancelled INTEGER NOT NULL DEFAULT 0,
                turn INTEGER NOT NULL, completed_turn INTEGER NOT NULL DEFAULT 0,
                answer TEXT, error TEXT, updated REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS turns (
                name TEXT NOT NULL COLLATE NOCASE, turn INTEGER NOT NULL,
                prompt TEXT NOT NULL, status TEXT NOT NULL, outcome TEXT,
                expect TEXT, PRIMARY KEY (name, turn))""")
            # A registry created before `--expect` existed lacks the column;
            # its old turns read back as NULL, i.e. no answer contract.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(turns)")}
            if "expect" not in columns:
                db.execute("ALTER TABLE turns ADD COLUMN expect TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_context (
                name TEXT PRIMARY KEY COLLATE NOCASE, snapshot TEXT NOT NULL)""")

    @contextmanager
    def _transaction(self, *, write=True):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _get(db, name):
        validate_name(name)
        row = db.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown agent: {name}")
        result = dict(row)
        result["options"] = json.loads(result["options"])
        result["session_started"] = bool(result["session_started"])
        result["cancelled"] = bool(result["cancelled"])
        context = db.execute("SELECT snapshot FROM agent_context WHERE name = ?", (name,)).fetchone()
        result["context"] = json.loads(context["snapshot"]) if context else None
        return result

    def create(self, name, cwd, options: dict, prompt: str, expect: str = "") -> dict:
        validate_name(name)
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM agents WHERE name = ?", (name,)).fetchone():
                raise ValueError(f"Agent {name!r} already exists; choose a new name")
            db.execute("""INSERT INTO agents
                (name, cwd, options, session_id, status, turn, updated)
                VALUES (?, ?, ?, ?, 'queued', 1, ?)""",
                (name, str(cwd), json.dumps(options), str(uuid.uuid4()), time.time()))
            db.execute("""INSERT INTO turns (name, turn, prompt, status, expect)
                VALUES (?, 1, ?, 'queued', ?)""", (name, prompt, expect))
            return self._get(db, name)

    def get(self, name) -> dict:
        with self._transaction(write=False) as db:
            return self._get(db, name)

    def list(self) -> list[dict]:
        with self._transaction(write=False) as db:
            names = [row[0] for row in db.execute("SELECT name FROM agents ORDER BY name")]
            return [self._get(db, name) for name in names]

    def enqueue(self, name, prompt, expect: str = "") -> int:
        with self._transaction() as db:
            agent = self._get(db, name)
            turn = agent["turn"] + 1
            db.execute("""INSERT INTO turns (name, turn, prompt, status, expect)
                VALUES (?, ?, ?, 'queued', ?)""", (name, turn, prompt, expect))
            db.execute("""UPDATE agents SET turn = ?, status = ?, error = NULL,
                cancelled = CASE WHEN owner IS NULL THEN 0 ELSE cancelled END,
                updated = ? WHERE name = ?""",
                (turn, "running" if agent["owner"] else "queued", time.time(), name))
            return turn

    def claim(self, name, owner: str) -> bool:
        if not owner:
            raise ValueError("Worker owner must not be empty")
        with self._transaction() as db:
            self._get(db, name)
            return bool(db.execute("""UPDATE agents SET owner = ?, cancelled = 0,
                status = 'running', updated = ? WHERE name = ? AND owner IS NULL
                AND EXISTS (SELECT 1 FROM turns WHERE name = ? AND status = 'queued')""",
                (owner, time.time(), name, name)).rowcount)

    def next_turn(self, name, owner) -> dict | None:
        with self._transaction() as db:
            agent = self._get(db, name)
            if agent["owner"] != owner:
                return None
            if db.execute("SELECT 1 FROM turns WHERE name = ? AND status = 'running'", (name,)).fetchone():
                raise RuntimeError("Finish the active turn before claiming another")
            row = db.execute("SELECT * FROM turns WHERE name = ? AND status = 'queued' ORDER BY turn LIMIT 1", (name,)).fetchone()
            if agent["cancelled"] or row is None:
                status = "queued" if row else agent["status"]
                db.execute("UPDATE agents SET owner = NULL, cancelled = 0, status = ?, updated = ? WHERE name = ?",
                           (status, time.time(), name))
                return None
            db.execute("UPDATE turns SET status = 'running' WHERE name = ? AND turn = ?", (name, row["turn"]))
            db.execute("UPDATE agents SET status = 'running', updated = ? WHERE name = ?", (time.time(), name))
            return {"turn": row["turn"], "prompt": row["prompt"], "resume": agent["session_started"],
                    "expect": row["expect"] or ""}

    def finish(self, name, owner, turn: int, outcome: dict) -> None:
        with self._transaction() as db:
            agent = self._get(db, name)
            if agent["owner"] != owner:
                raise RuntimeError("Worker no longer owns this agent")
            row = db.execute("SELECT status FROM turns WHERE name = ? AND turn = ?", (name, turn)).fetchone()
            if row is None:
                raise KeyError(f"Unknown turn {turn} for agent {agent['name']}")
            if row["status"] != "running":
                raise RuntimeError("Turn is not running")
            outcome = dict(outcome)
            status = outcome.get("status", "completed")
            if status not in {"completed", "failed", "interrupted"}:
                raise ValueError(f"Invalid turn outcome status: {status}")
            # The provider may finish before the interrupt reaches its process.
            # A late request cannot invalidate an already successful result.
            if agent["cancelled"]:
                outcome["cancel_requested"] = True
                if status != "completed":
                    status = "interrupted"
            outcome["status"] = status
            db.execute("UPDATE turns SET status = ?, outcome = ? WHERE name = ? AND turn = ?",
                       (status, json.dumps(outcome), name, turn))
            session_id = outcome.get("session_id")
            session_started = bool(session_id) or agent["session_started"]
            session_id = session_id or agent["session_id"]
            # An unannounced process may still have saved this ID. A fresh
            # attempt must not reuse it as --session-id and collide with it.
            if status != "completed" and not session_started:
                session_id = str(uuid.uuid4())
                db.execute("DELETE FROM agent_context WHERE name = ?", (name,))
            aggregate_status = status
            if db.execute("SELECT 1 FROM turns WHERE name = ? AND status = 'queued'", (name,)).fetchone():
                aggregate_status = "running"
            db.execute("""UPDATE agents SET status = ?, completed_turn = ?,
                answer = CASE WHEN ? = 'completed' THEN ? ELSE answer END,
                error = ?, session_id = ?, session_started = ?, updated = ? WHERE name = ?""",
                (aggregate_status, turn if status == "completed" else agent["completed_turn"],
                 status, outcome.get("answer"), outcome.get("error"),
                 session_id, session_started, time.time(), name))

    def interrupt(self, name) -> None:
        with self._transaction() as db:
            agent = self._get(db, name)
            if not db.execute("SELECT 1 FROM turns WHERE name = ? AND status IN ('queued', 'running')", (name,)).fetchone():
                return
            db.execute("UPDATE turns SET status = 'interrupted', outcome = ? WHERE name = ? AND status = 'queued'",
                       (json.dumps({"status": "interrupted", "error": "Cancelled before execution"}), name))
            db.execute("UPDATE agents SET cancelled = ?, status = 'interrupted', updated = ? WHERE name = ?",
                       (bool(agent["owner"]), time.time(), name))

    def cancelled(self, name, owner) -> bool:
        agent = self.get(name)
        return agent["owner"] != owner or agent["cancelled"]

    def session_ready(self, name, owner, session_id: str) -> None:
        """Persist provider session creation before the first answer arrives."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Session ID must not be empty")
        with self._transaction() as db:
            agent = self._get(db, name)
            if not owner or agent["owner"] != owner:
                raise RuntimeError("Worker no longer owns this agent")
            db.execute("UPDATE agents SET session_id = ?, session_started = 1, updated = ? WHERE name = ?",
                       (session_id, time.time(), name))

    def fail_worker(self, name, owner, error: str) -> None:
        with self._transaction() as db:
            agent = self._get(db, name)
            if agent["owner"] != owner:
                return
            status = "interrupted" if agent["cancelled"] else "failed"
            outcome = json.dumps({"status": status, "error": error})
            failed = db.execute("UPDATE turns SET status = ?, outcome = ? WHERE name = ? AND status = 'running'",
                                (status, outcome, name)).rowcount
            session_id = str(uuid.uuid4()) if failed and not agent["session_started"] else agent["session_id"]
            if session_id != agent["session_id"]:
                db.execute("DELETE FROM agent_context WHERE name = ?", (name,))
            if db.execute("SELECT 1 FROM turns WHERE name = ? AND status = 'queued'", (name,)).fetchone():
                status = "queued"
            db.execute("UPDATE agents SET owner = NULL, cancelled = 0, status = ?, error = ?, session_id = ?, updated = ? WHERE name = ?",
                       (status, error, session_id, time.time(), name))

    def recover(self, name) -> None:
        """Recover a dead worker only while holding its exclusive OS lock.

        Ownership is never expired by elapsed time: a slow live worker owns the
        session until the OS releases its lock. The caller must acquire that
        lock before using this operation.
        """
        with self._transaction() as db:
            agent = self._get(db, name)
            if agent["owner"] is None:
                return
            error = "Worker exited before reporting a result"
            status = "interrupted" if agent["cancelled"] else "failed"
            failed = db.execute("UPDATE turns SET status = ?, outcome = ? WHERE name = ? AND status = 'running'",
                                (status, json.dumps({"status": status, "error": error}), name)).rowcount
            session_id = str(uuid.uuid4()) if failed and not agent["session_started"] else agent["session_id"]
            if session_id != agent["session_id"]:
                db.execute("DELETE FROM agent_context WHERE name = ?", (name,))
            if not failed:
                status = agent["status"]
                error = agent["error"]
            if db.execute("SELECT 1 FROM turns WHERE name = ? AND status = 'queued'", (name,)).fetchone():
                status = "queued"
            db.execute("UPDATE agents SET owner = NULL, cancelled = 0, status = ?, error = ?, session_id = ?, updated = ? WHERE name = ?",
                       (status, error, session_id, time.time(), name))

    def context_ready(self, name, owner, snapshot: dict) -> None:
        """Persist live usage so status and a resumed worker share one snapshot."""
        with self._transaction() as db:
            agent = self._get(db, name)
            if not owner or agent["owner"] != owner:
                raise RuntimeError("Worker no longer owns this agent")
            db.execute("""INSERT INTO agent_context (name, snapshot) VALUES (?, ?)
                ON CONFLICT (name) DO UPDATE SET snapshot = excluded.snapshot""",
                       (agent["name"], json.dumps(snapshot)))
            db.execute("UPDATE agents SET updated = ? WHERE name = ?", (time.time(), name))

    def result(self, name, turn: int | None = None) -> dict:
        with self._transaction(write=False) as db:
            agent = self._get(db, name)
            turn = agent["turn"] if turn is None else turn
            row = db.execute("SELECT * FROM turns WHERE name = ? AND turn = ?", (name, turn)).fetchone()
            if row is None:
                raise KeyError(f"Unknown turn {turn} for agent {agent['name']}")
            result = {"name": agent["name"], "turn": turn, "prompt": row["prompt"], "status": row["status"]}
            if row["outcome"]:
                result.update(json.loads(row["outcome"]))
            return result
