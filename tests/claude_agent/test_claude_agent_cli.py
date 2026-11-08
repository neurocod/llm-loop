"""Offline CLI/worker integration; never invoke a real Claude executable."""

from contextlib import ExitStack, closing, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from llm_agents import agent_counter, host_policy
from llm_agents.answer_contract import EXPECT_FOLLOWUP_PROMPT, coverage_expect

from llm_agents.claude import cli, follow_up, protocol, worker
from llm_agents.claude.state import Store

# A host of its own, so `main` has no empty-policy notice to print: the
# stderr these tests read is the CLI's own.
TEST_HOST = host_policy.HostPolicy()


class CliWorkerTests(unittest.TestCase):
    def enter(self, manager):
        """`TestCase.enterContext`, which exists only from 3.11: the package
        supports 3.9, and so do its tests."""
        result = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        return result

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.enter(patch.dict("os.environ", {agent_counter.DISABLE_ENV: "0"}))
        self.addCleanup(self.tmp.cleanup)
        # Resolved: the CLI prints resolved paths, and on macOS the temp
        # directory under /var is /private/var once resolved.
        self.cwd = Path(self.tmp.name).resolve()
        self.root = self.cwd / "state"
        self.store = Store(self.root)
        self.launch = self.enter(patch.object(cli, "launch"))
        self.worker_launch = self.enter(patch.object(worker, "launch"))
        self.run_turn = self.enter(patch.object(follow_up, "run_turn", side_effect=AssertionError("Unconfigured fake turn")))

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["--state-dir", str(self.root), *args],
                            policy=TEST_HOST, root=self.cwd)
        return code, stdout.getvalue(), stderr.getvalue()

    def spawn(self, name="review", command="spawn", message="private prompt"):
        return self.invoke(command, name, "--cwd", str(self.cwd), "--message", message, "--json")

    @staticmethod
    def answer(turn: int) -> str:
        # Complete under the review profile's default contract, so tests about
        # queues and locks see one provider call per turn, not a follow-up.
        return f"Final answer {turn}\n## Coverage\nnone"

    @classmethod
    def successful(cls, agent, turn, directory, cancelled, **kwargs):
        return {"status": "completed", "answer": cls.answer(turn["turn"]),
                "session_id": agent["session_id"]}

    def test_spawn_alias_launches_and_returns_before_provider_runs(self):
        code, output, _ = self.spawn(command="spawn_agent")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["turn"], 1)
        self.launch.assert_called_once()
        self.run_turn.assert_not_called()
        self.assertEqual(self.store.result("review")["status"], "queued")
        self.assertEqual(self.invoke("result", "review", "--json")[0], 2)

    @unittest.skipUnless(os.name == "nt", "Windows screen counter transport")
    def test_only_active_claude_turns_hold_one_counter_connection(self):
        connections = []
        sent = []

        class Handle:
            def __init__(self):
                connections.append(self)

            def write(self, data):
                sent.append(json.loads(data))

            def close(self):
                connections.remove(self)

        self.spawn()
        self.store.enqueue("review", "queued followup")
        with patch.dict(os.environ, {agent_counter.DISABLE_ENV: "1", "CLAUDE_AGENTS_COUNTER": "1"}), \
                patch.object(agent_counter, "_open_or_start", side_effect=Handle):
            def running(agent, turn, directory, cancelled, **kwargs):
                self.assertEqual(len(connections), 1)
                # A duplicate worker cannot claim the OS lock or count again.
                worker.work(Store(self.root), "review")
                self.assertEqual(len(connections), 1)
                self.assertEqual(sent[-1], {"op": "hello", "label": f"claude:review@{self.cwd.name}",
                                            "agent_type": "claude"})
                return self.successful(agent, turn, directory, cancelled)

            self.run_turn.side_effect = running
            worker.work(self.store, "review")
            self.assertEqual(connections, [])
            self.assertEqual(len(sent), 2)
            worker.work(self.store, "review")
            self.assertEqual(len(sent), 2)  # Idle sessions do not hold a badge.

    @unittest.skipUnless(os.name == "nt", "Windows screen counter transport")
    def test_missing_or_broken_counter_does_not_fail_claude_turn(self):
        self.spawn()
        self.run_turn.side_effect = self.successful
        with patch.dict(os.environ, {agent_counter.DISABLE_ENV: "1", "CLAUDE_AGENTS_COUNTER": "1"}), \
                patch.object(agent_counter, "_connect", return_value=None), \
                patch.object(agent_counter, "counter_exe", return_value=None), \
                patch.object(agent_counter, "_spawn") as spawn_counter, \
                redirect_stderr(io.StringIO()):
            worker.work(self.store, "review")
            spawn_counter.assert_not_called()
        self.assertEqual(self.store.result("review")["status"], "completed")
        self.store.enqueue("review", "followup")
        with patch.dict(os.environ, {agent_counter.DISABLE_ENV: "1", "CLAUDE_AGENTS_COUNTER": "1"}), \
                patch.object(agent_counter, "_open_or_start", side_effect=OSError("broken pipe")), \
                redirect_stderr(io.StringIO()):
            worker.work(self.store, "review")
        self.assertEqual(self.store.result("review")["status"], "completed")

    @unittest.skipUnless(os.name == "nt", "Windows screen counter transport")
    def test_worker_failure_and_interruption_release_the_counter(self):
        for outcome in ("failed", "interrupted"):
            with self.subTest(outcome=outcome):
                name = f"review-{outcome}"
                self.spawn(name=name)
                handle = unittest.mock.Mock()

                def running(agent, turn, directory, cancelled, **kwargs):
                    handle.close.assert_not_called()
                    if outcome == "failed":
                        raise RuntimeError("fake transport failure")
                    self.store.interrupt(name)
                    return {"status": "interrupted", "error": "cancelled"}

                self.run_turn.side_effect = running
                with patch.dict(os.environ, {agent_counter.DISABLE_ENV: "1", "CLAUDE_AGENTS_COUNTER": "1"}), \
                        patch.object(agent_counter, "_open_or_start", return_value=handle), \
                        redirect_stderr(io.StringIO()):
                    worker.work(self.store, name)
                handle.close.assert_called_once()
                self.assertEqual(self.store.result(name)["status"], outcome)

    def test_claude_counter_can_be_disabled_independently(self):
        self.spawn()
        self.run_turn.side_effect = self.successful
        with patch.dict(os.environ, {agent_counter.DISABLE_ENV: "1", "CLAUDE_AGENTS_COUNTER": "0"}), \
                patch.object(agent_counter, "_open_or_start") as open_counter:
            worker.work(self.store, "review")
        open_counter.assert_not_called()
        self.assertEqual(self.store.result("review")["status"], "completed")

    @unittest.skipUnless(os.name == "nt", "Windows screen counter transport")
    def test_failed_counter_write_or_close_does_not_replace_turn_result(self):
        for broken_write in (False, True):
            with self.subTest(broken_write=broken_write):
                name = f"review-pipe-{broken_write}"
                self.spawn(name=name)
                self.run_turn.side_effect = self.successful
                handle = unittest.mock.Mock()
                handle.close.side_effect = OSError("counter disconnected")
                if broken_write:
                    handle.write.side_effect = BrokenPipeError("counter disconnected")
                with patch.dict(os.environ, {agent_counter.DISABLE_ENV: "1", "CLAUDE_AGENTS_COUNTER": "1"}), \
                        patch.object(agent_counter, "_open_or_start", return_value=handle), \
                        redirect_stderr(io.StringIO()):
                    worker.work(self.store, name)
                handle.close.assert_called_once()
                self.assertEqual(self.store.result(name)["status"], "completed")

    def test_empty_prompt_bad_cwd_and_duplicates_do_not_launch(self):
        for args in (
            ("spawn", "empty", "--message", "  "),
            ("spawn", "missing", "--message", "task", "--cwd", str(self.cwd / "absent")),
        ):
            with self.subTest(args=args):
                self.assertEqual(self.invoke(*args)[0], 1)
        self.assertEqual(self.store.list(), [])
        self.launch.assert_not_called()
        self.spawn()
        self.launch.reset_mock()
        self.assertEqual(self.spawn(name="REVIEW")[0], 1)
        self.assertEqual(self.store.result("review")["prompt"], "private prompt")
        self.launch.assert_not_called()

    def test_send_aliases_return_their_turn_even_if_another_sender_races(self):
        self.spawn()
        self.launch.side_effect = lambda store, name: store.enqueue(name, "concurrent sender")
        for alias, expected_turn in (("send_message", 2), ("followup_task", 4)):
            code, output, _ = self.invoke(alias, "review", "--message", "followup", "--json")
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["turn"], expected_turn)
            self.assertEqual(self.store.result("review", expected_turn)["prompt"], "followup")

    def test_wait_expiry_preserves_pending_and_final_output_hides_prompt(self):
        self.spawn()
        code, output, _ = self.invoke("wait_agent", "review", "--timeout", "0", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["status"], "queued")
        self.assertEqual(json.loads(output)["answer"], "")
        self.assertEqual(json.loads(output)["error"], "")
        self.assertNotIn("private prompt", output)
        self.assertEqual(self.store.result("review")["status"], "queued")
        self.run_turn.side_effect = self.successful
        worker.work(self.store, "review")
        code, output, _ = self.invoke("wait", "review", "--timeout", "0")
        self.assertEqual(code, 0)
        self.assertIn("Final answer 1", output)
        self.assertNotIn("private prompt", output)

    def test_pending_text_suggests_one_wait_but_status_listing_stays_brief(self):
        code, output, _ = self.invoke("spawn", "review", "--cwd", str(self.cwd), "--message", "task")
        self.assertEqual(code, 0)
        self.assertIn("wait_agent --turn 1 --until-done -- review", output)
        self.assertIn(str(self.root.resolve()), output)
        self.assertIn(str(Path(cli.__file__).resolve().parent), output)
        code, output, _ = self.invoke("wait", "review", "--timeout", "0")
        self.assertEqual(code, 0)
        self.assertIn("resume that session instead of starting another wait", output)
        self.assertNotIn("Wait once:", self.invoke("status", "review")[1])

    def test_until_done_waits_for_selected_turn_without_chasing_a_followup(self):
        self.spawn()
        self.run_turn.side_effect = self.successful

        def finish_then_enqueue(_delay):
            worker.work(self.store, "review")
            self.store.enqueue("review", "later followup")

        with patch.object(cli.time, "sleep", side_effect=finish_then_enqueue) as sleep:
            code, output, _ = self.invoke("wait_agent", "review", "--until-done")
        self.assertEqual(code, 0)
        sleep.assert_called_once()
        self.assertIn("Final answer 1", output)
        self.assertNotIn("Wait once:", output)
        self.assertEqual(self.store.result("review", 2)["status"], "queued")

    def test_wait_modes_are_exclusive_and_default_window_is_preserved(self):
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["wait", "review"]).timeout, 90)
        self.assertIsNone(parser.parse_args(["wait", "review", "--until-done"]).timeout)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parser.parse_args(["wait", "review", "--timeout", "0", "--until-done"])
        self.assertEqual(error.exception.code, 2)

    def test_spawn_runs_on_opus_unless_a_model_is_named(self):
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["spawn", "r", "--message", "x"]).model, "opus")
        self.assertEqual(parser.parse_args(
            ["spawn", "r", "--message", "x", "--model", "fable"]).model, "fable")

    def test_until_done_reminds_every_thirty_minutes_without_polluting_json(self):
        self.spawn()
        pending = cli.turn_result(self.store, "review")
        completed = pending | {"status": "completed", "answer": "Done"}
        clock = [0.0]
        jumps = iter([1799, 1, 1800, 1800])

        def advance(_delay):
            clock[0] += next(jumps)

        with patch.object(cli.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(cli.time, "sleep", side_effect=advance), \
                patch.object(cli, "turn_result", side_effect=[pending, pending, pending, pending, completed]):
            code, output, stderr = self.invoke("wait", "review", "--until-done", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["answer"], "Done")
        self.assertEqual(stderr.count("Progress reminder:"), 2)
        self.assertIn("after 30 minutes", stderr)
        self.assertIn("after 60 minutes", stderr)
        self.assertNotIn("after 90 minutes", stderr)
        self.assertIn("Read-only reviews may leave the tree unchanged", stderr)

    def test_numeric_wait_does_not_emit_until_done_reminders(self):
        self.spawn()
        clock = [0.0]

        def advance(_delay):
            clock[0] += 1800

        with patch.object(cli.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(cli.time, "sleep", side_effect=advance):
            code, output, stderr = self.invoke("wait", "review", "--timeout", "3600", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["status"], "queued")
        self.assertEqual(stderr, "")

    def test_until_done_recovers_worker_that_dies_after_initial_launch(self):
        self.spawn()
        clock = [0.0]
        launches = []

        def launch_or_recover(store, name):
            launches.append(clock[0])
            if len(launches) == 1:
                store.claim(name, "crashing-owner")
                store.next_turn(name, "crashing-owner")
            else:
                worker.work(store, name)

        with ExitStack() as locks:
            self.assertTrue(locks.enter_context(worker.worker_lock(self.root / "review" / "worker.lock")))

            def advance(delay):
                self.assertEqual(delay, 1)
                clock[0] += delay
                self.assertLessEqual(clock[0], 61, "Unlimited wait failed to recover the dead worker")
                if clock[0] == 10:
                    # A hard process exit releases the OS lock without updating SQLite.
                    locks.close()

            self.launch.side_effect = launch_or_recover
            with patch.object(cli.time, "monotonic", side_effect=lambda: clock[0]), \
                    patch.object(cli.time, "sleep", side_effect=advance):
                code, output, stderr = self.invoke("wait", "review", "--until-done", "--json")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output)["status"], "failed")
        self.assertEqual(launches, [0, 60])
        self.assertEqual(stderr, "")
        self.run_turn.assert_not_called()

    def test_until_done_recovery_cadence_does_not_duplicate_live_provider(self):
        self.spawn()
        self.store.claim("review", "live-owner")
        self.store.next_turn("review", "live-owner")
        clock = [0.0]
        launches = []

        def recover(store, name):
            launches.append(clock[0])
            worker.work(store, name)
            self.assertEqual(store.get(name)["owner"], "live-owner")

        def advance(delay):
            self.assertEqual(delay, 1)
            clock[0] += delay
            self.assertLessEqual(clock[0], 125)
            if clock[0] == 125:
                self.store.finish("review", "live-owner", 1,
                                  {"status": "completed", "answer": "Original provider finished"})

        self.launch.side_effect = recover
        with worker.worker_lock(self.root / "review" / "worker.lock") as held:
            self.assertTrue(held)
            with patch.object(cli.time, "monotonic", side_effect=lambda: clock[0]), \
                    patch.object(cli.time, "sleep", side_effect=advance):
                code, output, stderr = self.invoke("wait", "review", "--until-done", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["answer"], "Original provider finished")
        self.assertEqual(launches, [0, 60, 120])
        self.assertEqual(stderr, "")
        self.run_turn.assert_not_called()

    def test_optional_recovery_failure_preserves_wait_and_retries(self):
        self.spawn()
        self.store.claim("review", "live-owner")
        self.store.next_turn("review", "live-owner")
        self.launch.reset_mock()
        self.launch.side_effect = [None, OSError("temporary launch failure"), None]
        clock = [0.0]

        def advance(delay):
            clock[0] += delay
            self.assertLessEqual(clock[0], 125)
            if clock[0] == 125:
                self.store.finish("review", "live-owner", 1,
                                  {"status": "completed", "answer": "Original provider finished"})

        with patch.object(cli.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(cli.time, "sleep", side_effect=advance):
            code, output, stderr = self.invoke("wait", "review", "--until-done", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["answer"], "Original provider finished")
        self.assertEqual(self.launch.call_count, 3)
        self.assertIn("temporary launch failure; waiting continues", stderr)

    def test_resume_uses_same_session_and_retrieves_earlier_turn(self):
        self.spawn()
        original_session = self.store.get("review")["session_id"]
        calls = []

        def record(agent, turn, directory, cancelled, **kwargs):
            calls.append((agent["session_id"], turn["resume"], turn["prompt"]))
            return self.successful(agent, turn, directory, cancelled)

        self.run_turn.side_effect = record
        worker.work(self.store, "review")
        self.invoke("send", "review", "--message", "second private prompt")
        worker.work(self.store, "review")
        self.assertEqual(calls, [(original_session, False, "private prompt"),
                                 (original_session, True, "second private prompt")])
        code, output, _ = self.invoke("result", "review", "--turn", "1", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["answer"], self.answer(1))
        self.assertNotIn("prompt", json.loads(output))
        code, output, _ = self.invoke("list_agents", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)[0]["status"], "completed")

    def test_interrupt_active_turn_and_followup_relaunches_after_unlock(self):
        self.spawn()

        def interrupted(agent, turn, directory, cancelled, **kwargs):
            self.assertFalse(cancelled())
            self.assertEqual(self.invoke("interrupt_agent", "review")[0], 0)
            self.assertTrue(cancelled())
            self.invoke("followup_task", "review", "--message", "after interruption")
            outcome = self.successful(agent, turn, directory, cancelled)
            outcome["status"] = "interrupted"
            return outcome

        self.run_turn.side_effect = interrupted
        worker.work(self.store, "review")
        self.assertEqual(self.store.result("review", 1)["status"], "interrupted")
        self.assertEqual(self.store.result("review", 2)["status"], "queued")
        self.worker_launch.assert_called_once()
        self.run_turn.side_effect = self.successful
        worker.work(self.store, "review")
        self.assertEqual(self.store.result("review", 2)["status"], "completed")

    def test_live_os_lock_prevents_recovery_and_duplicate_provider(self):
        self.spawn()
        self.store.claim("review", "live-owner")
        self.store.next_turn("review", "live-owner")
        with worker.worker_lock(self.root / "review" / "worker.lock") as held:
            self.assertTrue(held)
            worker.work(self.store, "review")
            self.assertEqual(self.store.get("review")["owner"], "live-owner")
            self.assertEqual(self.store.result("review", 1)["status"], "running")
        self.run_turn.assert_not_called()

    def test_released_crashed_worker_lock_recovers_then_runs_followup(self):
        self.spawn()
        self.store.claim("review", "dead-owner")
        self.store.next_turn("review", "dead-owner")
        self.store.enqueue("review", "after crash")
        self.run_turn.side_effect = self.successful
        worker.work(self.store, "review")
        self.assertEqual(self.store.result("review", 1)["status"], "failed")
        self.assertEqual(self.store.result("review", 2)["status"], "completed")
        self.assertFalse(self.run_turn.call_args.args[1]["resume"])

    def test_enqueue_after_empty_claim_relaunches_after_unlock(self):
        self.spawn()
        self.store.interrupt("review")
        original_claim = self.store.claim

        def raced_claim(name, owner):
            claimed = original_claim(name, owner)
            self.assertFalse(claimed)
            self.store.enqueue(name, "racing followup")
            # The sender's launched worker encounters the still-held lock.
            worker.work(Store(self.root), name)
            return claimed

        with patch.object(self.store, "claim", side_effect=raced_claim):
            worker.work(self.store, "review")
        self.worker_launch.assert_called_once()

    def test_unexpected_worker_failure_relaunches_preserved_followup(self):
        self.spawn()
        self.store.enqueue("review", "second")
        self.run_turn.side_effect = RuntimeError("fake transport failure")
        try:
            worker.work(self.store, "review")
        except RuntimeError:
            pass
        self.assertEqual(self.store.result("review", 1)["status"], "failed")
        self.assertEqual(self.store.result("review", 2)["status"], "queued")
        self.worker_launch.assert_called_once()

    def test_fake_stream_process_preserves_logs_but_exposes_only_final_result(self):
        self.spawn(message="secret prompt")
        script = (
            "import json,sys; request=json.loads(sys.stdin.readline()); "
            "print(json.dumps({'type':'assistant','message':{'content':'intermediate chatter'}}),flush=True); "
            "print(json.dumps({'type':'result','subtype':'success','is_error':False,"
            "'result':'Reviewed successfully\\n## Coverage\\nnone','session_id':request['session_id']}),flush=True)"
        )
        self.run_turn.side_effect = protocol.run_turn
        # The subject is the stream, not the review hook: the fake CLI never
        # loads it, and a tree without the built ask_user_gate.exe (a slot) would
        # fail the turn before it starts. The hook check is
        # test_claude_agent_protocol's (`verify_review_gate`).
        with patch.object(protocol, "command", return_value=[sys.executable, "-u", "-c", script]), \
                patch.object(protocol, "verify_review_gate") as verify:
            worker.work(self.store, "review")
        verify.assert_called_once()  # the review turn still asks for its gate
        code, output, _ = self.invoke("result", "review", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["answer"], "Reviewed successfully\n## Coverage\nnone")
        self.assertNotIn("secret prompt", output)
        self.assertNotIn("intermediate chatter", output)
        self.assertIn("intermediate chatter", (self.root / "review" / "turn-0001" / "events.jsonl").read_text())
        code, output, _ = self.invoke("logs", "review", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(Path(json.loads(output)["log_dir"]), self.root / "review" / "turn-0001")

    def test_database_error_preserves_json_contract(self):
        self.store.path.write_bytes(b"invalid sqlite database")
        code, output, stderr = self.invoke("status", "--json")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output)["status"], "error")
        self.assertIn("database", json.loads(output)["error"])
        self.assertEqual(stderr, "")

    def test_submit_and_wait_returns_the_submitted_turn(self):
        self.run_turn.side_effect = self.successful
        self.launch.side_effect = lambda store, name: worker.work(store, name)
        code, output, _ = self.invoke("spawn", "review", "--cwd", str(self.cwd),
                                      "--message", "one", "--wait", "0", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["answer"], self.answer(1))
        code, output, _ = self.invoke("send", "review", "--message", "two",
                                      "--wait", "0", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["turn"], 2)
        self.assertEqual(json.loads(output)["answer"], self.answer(2))

    def stored_expect(self, name, turn):
        with closing(sqlite3.connect(self.store.path)) as db:
            return db.execute("SELECT expect FROM turns WHERE name = ? AND turn = ?",
                              (name, turn)).fetchone()[0]

    def test_expect_defaults_to_the_review_contract_of_each_turns_message(self):
        """The review profile's default is `coverage_expect` of THAT turn's
        message, as `codex_agent --role reviewer` resolves it; `configured`
        has no answer contract; an explicit flag wins, empty included."""
        questions = "Review abc123.\n1. first\n2. second\n"
        self.assertEqual(self.spawn(message=questions)[0], 0)
        self.assertEqual(self.stored_expect("review", 1), coverage_expect(questions))
        self.invoke("send", "review", "--message", "continue")
        self.assertEqual(self.stored_expect("review", 2), "^## Coverage")
        self.invoke("send", "review", "--message", "1. again", "--expect=")
        self.assertEqual(self.stored_expect("review", 3), "")
        self.invoke("spawn", "edit", "--cwd", str(self.cwd), "--profile", "configured",
                    "--message", questions)
        self.assertEqual(self.stored_expect("edit", 1), "")
        self.invoke("send", "edit", "--message", "more", "--expect", "^## Verdict")
        self.assertEqual(self.stored_expect("edit", 2), "^## Verdict")
        self.launch.reset_mock()
        code, _, stderr = self.invoke("spawn", "broken", "--cwd", str(self.cwd),
                                      "--message", "task", "--expect", "(")
        self.assertEqual(code, 1)
        self.assertIn("Invalid --expect regex", stderr)
        self.assertNotIn("broken", [agent["name"] for agent in self.store.list()])
        self.launch.assert_not_called()

    def test_an_incomplete_review_is_followed_up_once_in_the_same_session(self):
        questions = "Review abc123.\n1. first\n2. second\n"
        self.spawn(message=questions)
        session = self.store.get("review")["session_id"]
        calls = []

        def provider(agent, turn, directory, cancelled, **kwargs):
            calls.append((agent["session_id"], turn, directory))
            answer = ("## Findings\nnone\n## Coverage\n2: answered" if len(calls) == 1
                      else "## Coverage\n1: nothing found\n2: answered")
            # Usage per invocation, cost the session's running total — the
            # shape the CLI reports on a resumed session (`follow_up.add_usage`).
            return {"status": "completed", "answer": answer, "session_id": agent["session_id"],
                    "duration_s": 2.0, "total_cost_usd": 0.25 * len(calls),
                    "usage": {"output_tokens": 100 // len(calls), "service_tier": "standard",
                              "output_tokens_details": {"thinking_tokens": 10},
                              "iterations": [{"n": len(calls)}]},
                    "log_dir": str(directory)}

        self.run_turn.side_effect = provider
        worker.work(self.store, "review")
        pattern = coverage_expect(questions)
        joined = ("## Findings\nnone\n## Coverage\n2: answered\n\n"
                  "## Coverage\n1: nothing found\n2: answered")
        self.assertEqual(len(calls), 2)
        self.assertEqual([call[0] for call in calls], [session, session])
        self.assertEqual(calls[1][1], {"turn": 1, "resume": True,
                                       "prompt": EXPECT_FOLLOWUP_PROMPT.format(pattern=pattern)})
        follow_up_dir = self.root / "review" / "turn-0001-follow-up-1"
        self.assertEqual(calls[1][2], follow_up_dir)
        result = self.store.result("review", 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["answer"], joined)
        self.assertEqual((result["follow_ups"], result["expect"], result["expect_met"]),
                         (1, pattern, True))
        self.assertEqual((result["duration_s"], result["total_cost_usd"]), (4.0, 0.5))
        self.assertEqual(result["usage"], {"output_tokens": 150, "service_tier": "standard",
                                           "output_tokens_details": {"thinking_tokens": 20},
                                           "iterations": [{"n": 1}, {"n": 2}]})
        turn_dir = self.root / "review" / "turn-0001"
        self.assertEqual(result["log_dir"], str(turn_dir))
        self.assertEqual(result["follow_up_log_dirs"], [str(follow_up_dir)])
        # The directory `Logs:` names holds the merged answer, not the first.
        self.assertEqual((turn_dir / "answer.md").read_text(encoding="utf-8"), joined)
        self.assertEqual(json.loads((turn_dir / "result.json").read_text(encoding="utf-8"))["answer"],
                         joined)
        code, output, _ = self.invoke("result", "review")
        self.assertEqual(code, 0)
        # Literal, not `followed_up_note(...)`: the oracle must not share the
        # wording it checks (the codex_agent twin pins the same line).
        self.assertIn(f"1 follow-up(s): the first answer lacked /{pattern}/", output)
        self.assertIn(f"Follow-up logs: {follow_up_dir}", output)
        self.assertNotIn("Incomplete", output)
        code, output, _ = self.invoke("logs", "review", "--json")
        self.assertEqual(json.loads(output)["follow_up_log_dirs"], [str(follow_up_dir)])

    def test_one_follow_up_only_and_a_failed_one_keeps_the_first_answer(self):
        for second in ("still incomplete", "failed"):
            with self.subTest(second=second):
                name = f"review-{second.split()[0]}"
                self.spawn(name=name, message="Review.\n1. only\n")
                calls = []

                def provider(agent, turn, directory, cancelled, **kwargs):
                    calls.append(turn)
                    if len(calls) == 2 and second == "failed":
                        # No result event: no usage, no cost, stderr only.
                        return {"status": "failed", "log_dir": str(directory),
                                "error": "Claude exited 1 without a result event; see stderr.log"}
                    return {"status": "completed", "answer": f"answer {len(calls)}",
                            "usage": {"output_tokens": 40}, "total_cost_usd": 0.5}

                self.run_turn.side_effect = provider
                worker.work(self.store, name)
                self.assertEqual(len(calls), 2)
                result = self.store.result(name, 1)
                follow_up_dir = self.root / name / "turn-0001-follow-up-1"
                code, output, _ = self.invoke("result", name)
                self.assertIn(f"Follow-up logs: {follow_up_dir}", output)
                if second == "failed":
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["answer"], "answer 1")
                    self.assertIn(f"see stderr.log (logs: {follow_up_dir}); the first answer is kept",
                                  result["error"])
                    self.assertNotIn("expect_met", result)
                    # A missing measurement stays missing, not one attempt's figure.
                    self.assertEqual((result["usage"], result["total_cost_usd"]), ({}, None))
                    self.assertEqual(code, 1)
                else:
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(result["answer"], "answer 1\n\nanswer 2")
                    self.assertIs(result["expect_met"], False)
                    self.assertEqual(code, 0)
                    self.assertIn("Incomplete: the answer still lacks /", output)
                    self.assertIn("after 1 follow-up(s)", output)

    def test_one_deadline_bounds_the_turn_and_its_follow_up(self):
        """`--turn-timeout` bounds the logical turn: the follow-up gets the
        first call's absolute deadline, and none at all once it is spent."""
        for timeout, first_takes, follows in ((100, 40, True), (100, 100, False), (0, 500, True)):
            with self.subTest(timeout=timeout, first_takes=first_takes):
                name = f"deadline-{timeout}-{first_takes}"
                self.invoke("spawn", name, "--cwd", str(self.cwd), "--message", "Review.\n1. only\n",
                            "--turn-timeout", str(timeout))
                clock = [1000.0]
                deadlines = []

                def provider(agent, turn, directory, cancelled, **kwargs):
                    deadlines.append(kwargs["deadline"])
                    clock[0] += first_takes
                    return {"status": "completed", "answer": f"answer {len(deadlines)}"}

                self.run_turn.side_effect = provider
                with patch.object(follow_up, "monotonic", lambda: clock[0]):
                    worker.work(self.store, name)
                expected = 1000.0 + timeout if timeout else None
                self.assertEqual(deadlines, [expected, expected] if follows else [expected])
                result = self.store.result(name, 1)
                self.assertEqual(result["follow_ups"], 1 if follows else 0)
                self.assertIs(result["expect_met"], False)

    def test_no_follow_up_after_a_cancel_request_or_under_configured(self):
        self.spawn(message="Review.\n1. only\n")
        self.invoke("spawn", "edit", "--cwd", str(self.cwd), "--profile", "configured",
                    "--message", "Edit.\n1. only\n")
        calls = []

        def cancelling(agent, turn, directory, cancelled, **kwargs):
            calls.append(agent["name"])
            if agent["name"] == "review":
                self.store.interrupt("review")
            return {"status": "completed", "answer": "no contract here"}

        self.run_turn.side_effect = cancelling
        worker.work(self.store, "review")
        worker.work(self.store, "edit")
        self.assertEqual(calls, ["review", "edit"])
        self.assertEqual(self.store.result("review", 1)["answer"], "no contract here")
        self.assertEqual(self.store.result("edit", 1)["status"], "completed")

    def test_status_reports_stream_liveness_without_content(self):
        self.spawn()
        directory = self.root / "review" / "turn-0001"
        directory.mkdir(parents=True)
        (directory / "events.jsonl").write_text("private stream", encoding="utf-8")
        code, output, _ = self.invoke("status", "review", "--json")
        self.assertEqual(code, 0)
        result = json.loads(output)[0]
        self.assertIsInstance(result["last_activity_s"], (int, float))
        self.assertIn("updated", result)
        self.assertNotIn("private stream", output)

    def test_interrupt_idle_completed_agent_preserves_status(self):
        self.spawn()
        self.run_turn.side_effect = self.successful
        worker.work(self.store, "review")
        code, output, _ = self.invoke("interrupt", "review", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["status"], "completed")

    def test_context_pressure_is_visible_on_result_status_and_send_receipt(self):
        self.spawn()
        context = {"model": "primary", "input_tokens": 450000,
                   "context_window": 1000000, "used_percent": 45.0, "warning": True}

        def report(agent, turn, directory, cancelled, **kwargs):
            kwargs["on_context"](context)
            return self.successful(agent, turn, directory, cancelled) | {"context": context}

        self.run_turn.side_effect = report
        worker.work(self.store, "review")
        for args in (("result", "review"), ("status", "review"),
                     ("send", "review", "--message", "same task followup")):
            code, output, _ = self.invoke(*args)
            self.assertEqual(code, 0)
            self.assertIn("45.0%", output)
            self.assertIn("Give new large tasks to a fresh spawn_agent", output)
        code, output, _ = self.invoke("status", "review", "--json")
        self.assertEqual(json.loads(output)[0]["context"], context)


if __name__ == "__main__":
    unittest.main()
