"""Exercise durable queue ordering, cancellation and competing workers."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from llm_agents.claude.state import Store


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root)
        self.store.create("review", self.root, {"model": "test"}, "first")

    def complete(self, owner="worker", turn=1, **outcome):
        self.store.finish("review", owner, turn, {
            "status": "completed", "answer": "first answer", "session_id": "real-session", **outcome})

    def test_names_and_case_collisions(self):
        for name in ("../escape", "bad/name", "bad\\name", "CON", "nul.txt", "LPT9.log", "review.", "", ".", "..", "a:b", "a b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.create(name, self.root, {}, "prompt")
        with self.assertRaises(ValueError):
            self.store.create("REVIEW", self.root, {}, "other")
        self.assertEqual(self.store.get("REVIEW")["name"], "review")
        self.assertEqual(len(self.store.list()), 1)

    def test_exclusive_claim_across_independent_connections(self):
        def claim(number):
            return Store(self.root).claim("review", f"worker-{number}")
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(claim, range(24))), 1)

    def test_durable_queue_and_resume(self):
        self.assertTrue(self.store.claim("review", "worker"))
        self.assertEqual(self.store.next_turn("review", "worker"),
                         {"turn": 1, "prompt": "first", "resume": False, "expect": ""})
        self.store.enqueue("review", "second", "^## Coverage")
        self.complete()
        self.assertEqual(self.store.get("review")["status"], "running")
        self.assertEqual(self.store.result("review", 1)["status"], "completed")
        reopened = Store(self.root)
        self.assertEqual(reopened.next_turn("review", "worker"),
                         {"turn": 2, "prompt": "second", "resume": True, "expect": "^## Coverage"})
        self.assertEqual(reopened.get("review")["status"], "running")
        self.assertEqual(reopened.get("review")["session_id"], "real-session")
        self.complete(turn=2, answer="second answer")
        self.assertIsNone(self.store.next_turn("review", "worker"))
        self.assertIsNone(self.store.get("review")["owner"])
        self.assertEqual(self.store.result("review", 1)["answer"], "first answer")
        self.assertEqual(self.store.result("review")["answer"], "second answer")

    def test_interruption_preserves_previous_results_and_later_followup(self):
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        self.complete()
        self.store.enqueue("review", "second")
        self.store.next_turn("review", "worker")
        self.store.enqueue("review", "cancel this queued prompt")
        self.store.interrupt("review")
        self.assertTrue(self.store.cancelled("review", "worker"))
        self.store.enqueue("review", "follow up after interruption")
        self.assertFalse(self.store.claim("review", "replacement"))
        self.complete(turn=2, status="interrupted", answer="partial answer")
        self.assertIsNone(self.store.next_turn("review", "worker"))
        self.assertEqual(self.store.result("review", 2)["status"], "interrupted")
        self.assertEqual(self.store.result("review", 3)["status"], "interrupted")
        self.assertEqual(self.store.get("review")["answer"], "first answer")
        self.assertEqual(self.store.get("review")["completed_turn"], 1)
        self.assertTrue(self.store.claim("review", "replacement"))
        self.assertEqual(self.store.next_turn("review", "replacement")["turn"], 4)

    def test_release_and_enqueue_race_never_loses_work(self):
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        self.complete()
        with ThreadPoolExecutor(max_workers=2) as pool:
            release = pool.submit(self.store.next_turn, "review", "worker")
            enqueue = pool.submit(Store(self.root).enqueue, "review", "second")
            claimed_turn = release.result()
            self.assertEqual(enqueue.result(), 2)
        if claimed_turn is None:
            self.assertTrue(self.store.claim("review", "replacement"))
            claimed_turn = self.store.next_turn("review", "replacement")
        self.assertEqual(claimed_turn["turn"], 2)

    def test_crash_recovery_keeps_queued_work(self):
        original_session = self.store.get("review")["session_id"]
        self.store.claim("review", "dead-worker")
        self.store.next_turn("review", "dead-worker")
        self.store.enqueue("review", "second")
        self.store.recover("review")
        self.assertEqual(self.store.result("review", 1)["status"], "failed")
        self.assertEqual(self.store.result("review", 2)["status"], "queued")
        self.assertTrue(self.store.claim("review", "replacement"))
        self.assertFalse(self.store.next_turn("review", "replacement")["resume"])
        self.assertNotEqual(self.store.get("review")["session_id"], original_session)
        self.assertEqual(self.store.get("review")["completed_turn"], 0)
        with self.assertRaises(RuntimeError):
            self.complete(owner="dead-worker")

    def test_worker_failure_retains_prior_answer(self):
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        self.complete()
        self.store.enqueue("review", "second")
        self.store.next_turn("review", "worker")
        self.store.fail_worker("review", "worker", "process failed")
        self.assertEqual(self.store.result("review")["status"], "failed")
        self.assertEqual(self.store.get("review")["answer"], "first answer")
        self.assertEqual(self.store.result("review", 1)["answer"], "first answer")
        self.assertEqual(self.store.get("review")["completed_turn"], 1)
        self.assertTrue(self.store.cancelled("review", "worker"))

    def test_session_announced_before_crash_is_resumed(self):
        self.store.claim("review", "dead-worker")
        self.store.next_turn("review", "dead-worker")
        self.store.session_ready("review", "dead-worker", "provider-session")
        self.store.enqueue("review", "after crash")
        self.store.recover("review")
        self.assertTrue(self.store.claim("review", "replacement"))
        self.assertTrue(self.store.next_turn("review", "replacement")["resume"])
        self.assertEqual(self.store.get("review")["session_id"], "provider-session")
        with self.assertRaises(RuntimeError):
            self.store.session_ready("review", "dead-worker", "stale-session")
        with self.assertRaises(ValueError):
            self.store.session_ready("review", "replacement", "")

    def test_recovery_after_persisting_answer_does_not_report_failure(self):
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        self.complete()
        self.store.recover("review")
        self.assertEqual(self.store.get("review")["status"], "completed")
        self.assertIsNone(self.store.get("review")["error"])
        self.assertIsNone(self.store.get("review")["owner"])

    def test_failed_first_turn_without_session_does_not_resume(self):
        original_session = self.store.get("review")["session_id"]
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        self.store.finish("review", "worker", 1, {"status": "failed", "error": "No process"})
        self.store.enqueue("review", "retry")
        self.assertFalse(self.store.next_turn("review", "worker")["resume"])
        self.assertNotEqual(self.store.get("review")["session_id"], original_session)
        self.assertEqual(self.store.get("review")["completed_turn"], 0)

    def test_successful_answer_survives_late_interrupt(self):
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        self.store.interrupt("review")
        self.complete()
        self.assertEqual(self.store.result("review")["status"], "completed")
        self.assertTrue(self.store.result("review")["cancel_requested"])
        self.assertEqual(self.store.get("review")["completed_turn"], 1)
        self.assertEqual(self.store.get("review")["answer"], "first answer")
        self.store.next_turn("review", "worker")
        self.assertEqual(self.store.get("review")["status"], "completed")

    def test_noncompleted_result_observes_interrupt_without_advancing_success(self):
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        self.complete()
        self.store.enqueue("review", "second")
        self.store.next_turn("review", "worker")
        self.store.interrupt("review")
        self.store.finish("review", "worker", 2, {"status": "failed", "error": "killed"})
        self.assertEqual(self.store.result("review", 2)["status"], "interrupted")
        self.assertEqual(self.store.get("review")["completed_turn"], 1)
        self.assertEqual(self.store.get("review")["session_id"], "real-session")

    def test_idle_interrupt_preserves_finished_agent_and_result(self):
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        self.complete()
        self.store.next_turn("review", "worker")
        original_agent = self.store.get("review")
        original_result = self.store.result("review")
        self.store.interrupt("review")
        self.assertEqual(self.store.get("review"), original_agent)
        self.assertEqual(self.store.result("review"), original_result)

    def test_worker_failure_rotates_only_unannounced_session(self):
        for announced in (False, True):
            with self.subTest(announced=announced):
                name = f"worker-{announced}"
                self.store.create(name, self.root, {}, "prompt")
                self.store.claim(name, "worker")
                self.store.next_turn(name, "worker")
                original_session = self.store.get(name)["session_id"]
                if announced:
                    self.store.session_ready(name, "worker", original_session)
                self.store.fail_worker(name, "worker", "crashed")
                self.assertEqual(self.store.get(name)["session_id"] == original_session, announced)

    def test_recovery_preserves_latest_successful_turn(self):
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        self.complete()
        self.store.enqueue("review", "second")
        self.store.next_turn("review", "worker")
        self.store.recover("review")
        self.assertEqual(self.store.result("review", 2)["status"], "failed")
        self.assertEqual(self.store.get("review")["completed_turn"], 1)
        self.assertEqual(self.store.get("review")["answer"], "first answer")
        self.assertEqual(self.store.get("review")["session_id"], "real-session")

    def test_reads_do_not_compete_for_reserved_writer_lock(self):
        # A reserved writer permits readers, but rejects another BEGIN IMMEDIATE.
        # Holding it in this thread makes an accidental writer poll fail rather
        # than relying on scheduler timing to expose the contention.
        db = sqlite3.connect(self.store.path)
        try:
            db.execute("BEGIN IMMEDIATE")
            self.assertEqual(self.store.get("review")["name"], "review")
            self.assertEqual(len(self.store.list()), 1)
            self.assertEqual(self.store.result("review")["status"], "queued")
            self.assertTrue(self.store.cancelled("review", "absent-owner"))
        finally:
            db.rollback()
            db.close()

    def test_missing_names_and_turns(self):
        with self.assertRaisesRegex(KeyError, "Unknown agent: missing"):
            self.store.get("missing")
        with self.assertRaisesRegex(KeyError, "Unknown turn 9 for agent review"):
            self.store.result("review", 9)

    def test_context_snapshot_is_live_durable_and_owner_checked(self):
        self.assertIsNone(self.store.get("review")["context"])
        with self.assertRaises(RuntimeError):
            self.store.context_ready("review", None, {})
        self.store.claim("review", "worker")
        self.store.next_turn("review", "worker")
        snapshot = {"model": "test", "input_tokens": None, "context_window": 200000,
                    "used_percent": None, "warning": None}
        self.store.context_ready("REVIEW", "worker", snapshot)
        self.assertEqual(Store(self.root).get("review")["context"], snapshot)
        self.assertEqual(self.store.list()[0]["context"], snapshot)
        with self.assertRaises(RuntimeError):
            self.store.context_ready("review", "stale-owner", {"input_tokens": 1})
        self.assertEqual(self.store.get("review")["context"], snapshot)
        snapshot.update(input_tokens=100000, used_percent=50, warning="Context occupancy is 50%")
        self.store.context_ready("review", "worker", snapshot)
        self.assertEqual(self.store.get("review")["context"], snapshot)
        self.assertEqual(self.store.result("review")["status"], "running")

    def test_existing_registry_adds_context_table_without_changing_agents(self):
        with closing(sqlite3.connect(self.store.path)) as db:
            original_columns = db.execute("PRAGMA table_info(agents)").fetchall()
            db.execute("DROP TABLE agent_context")
        reopened = Store(self.root)
        self.assertIsNone(reopened.get("review")["context"])
        self.assertEqual(reopened.result("review")["prompt"], "first")
        with closing(sqlite3.connect(self.store.path)) as db:
            self.assertEqual(db.execute("PRAGMA table_info(agents)").fetchall(), original_columns)

    def test_registry_without_expect_column_keeps_its_queued_turns(self):
        """A registry from before `--expect` has a five-column `turns` table;
        opening it adds the column, and its queued turn keeps no contract."""
        path = self.root / "old"
        path.mkdir()
        with closing(sqlite3.connect(path / "agents.sqlite3")) as db, db:
            db.execute("""CREATE TABLE turns (
                name TEXT NOT NULL COLLATE NOCASE, turn INTEGER NOT NULL,
                prompt TEXT NOT NULL, status TEXT NOT NULL, outcome TEXT,
                PRIMARY KEY (name, turn))""")
        old = Store(path)
        old.create("legacy", self.root, {}, "queued before the upgrade")
        with closing(sqlite3.connect(old.path)) as db, db:
            db.execute("UPDATE turns SET expect = NULL")
        old.enqueue("legacy", "after", "^## Verdict")
        self.assertTrue(old.claim("legacy", "worker"))
        self.assertEqual(old.next_turn("legacy", "worker")["expect"], "")
        old.finish("legacy", "worker", 1, {"status": "completed", "answer": "a"})
        self.assertEqual(Store(path).next_turn("legacy", "worker")["expect"], "^## Verdict")

    def test_context_cleared_only_when_failed_attempt_rotates_session(self):
        for method in ("finish", "fail_worker", "recover"):
            for announced in (False, True):
                with self.subTest(method=method, announced=announced):
                    name = f"{method}-{announced}"
                    self.store.create(name, self.root, {}, "prompt")
                    self.store.claim(name, "worker")
                    self.store.next_turn(name, "worker")
                    if announced:
                        self.store.session_ready(name, "worker", "announced-session")
                    snapshot = {"model": "test", "input_tokens": 100000,
                                "context_window": 200000, "used_percent": 50, "warning": "50%"}
                    self.store.context_ready(name, "worker", snapshot)
                    if method == "finish":
                        self.store.finish(name, "worker", 1, {"status": "failed"})
                    elif method == "fail_worker":
                        self.store.fail_worker(name, "worker", "crash")
                    else:
                        self.store.recover(name)
                    self.assertEqual(self.store.get(name)["context"], snapshot if announced else None)


if __name__ == "__main__":
    unittest.main()
