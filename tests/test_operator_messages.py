"""Notes typed at the console while the loop runs.

Two deliveries and one hazard, and every test here is about one of the three:

  * live - the note goes into the turn already running, over the provider's
    stdin. What makes that possible is a transport change (the prompt moves out
    of argv into a stdin that stays open), and what makes it SAFE is closing that
    stdin when the turn reports its result: a streaming-input CLI keeps its
    session alive until stdin reaches EOF, so a channel left open is a run that
    never returns. That is the hazard, and it is pinned from both sides.
  * queued - no turn in flight, so the note rides the next iteration's prompt.
  * the console - a Mode that consumes every key while a sentence is being
    typed, because `s` is the stop key one layer down.
"""

import io
import json
import os
import threading

import pytest

from llm_loop import cyclecore, operator, parallel, providers, textwidth
from llm_loop import statusline as sl
from llm_loop import termio as tio
from llm_loop.cyclecore import AgentCommand, Driver
from llm_loop.providers import build_agent_argv, start_agent_process


@pytest.fixture(autouse=True)
def _live_messages_on():
    """The default transport, whatever the environment of the test machine says.

    Through the setter, not the private global: a runner flips it the same way,
    so a test that leaves it wrong leaves it wrong for the rest of the session.
    """
    previous = providers.live_messages_enabled("claude")
    providers.set_live_messages(True)
    yield
    providers.set_live_messages(previous)


# --- the transport ------------------------------------------------------------


def test_claude_argv_moves_the_prompt_onto_the_stream():
    """`--input-format stream-json` ignores an argv prompt, so it must not be there."""
    argv = build_agent_argv(AgentCommand("work", "opus"), "claude", "/repo")

    assert "work" not in argv
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert "--replay-user-messages" in argv          # the delivery receipt
    assert argv[argv.index("--output-format") + 1] == "stream-json"


def test_codex_is_untouched_by_the_live_transport():
    argv = build_agent_argv(AgentCommand("work", "gpt-test"), "codex", "/repo")

    assert argv[-1] == "-"
    assert "--input-format" not in argv
    assert not providers.live_messages_enabled("codex")


def _capture_status_app(monkeypatch):
    """Make both runners build a headless StatusApp, and report what it was given.

    Returns a dict that gains `messages` — the mailbox the runner handed to the
    console — once the runner has built its status line. Both the sequential and
    the parallel test need exactly this, and they are asserting the same thing
    about two runners, so they share the patch rather than each carrying a copy.
    """
    seen = {}
    real_app_class = sl.StatusApp

    def _build(**kwargs):
        seen["messages"] = kwargs.get("messages")
        return real_app_class(terminal=tio.NullTerminal(),
                              input_source=tio.NullInputSource(), refresh=60,
                              **{k: v for k, v in kwargs.items()
                                 if k != "enabled"})

    monkeypatch.setattr(sl, "StatusApp", _build)
    return seen


class _Sink:
    """A stdin pipe that records what was written and whether it was closed."""

    def __init__(self):
        self.text = ""
        self.closed = False

    def write(self, value):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        self.text += value
        return len(value)

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def messages(self):
        """The user messages written so far, decoded."""
        return [json.loads(line)["message"]["content"][0]["text"]
                for line in self.text.splitlines() if line.strip()]


class _FakeProc:
    def __init__(self, stdout=(), stdin=None, returncode=0):
        self.stdin = stdin
        self.stdout = iter(stdout)
        self._returncode = returncode

    def wait(self):
        return self._returncode

    def terminate(self):
        pass


def test_the_prompt_is_written_as_a_user_message_and_the_pipe_stays_open(monkeypatch):
    sink = _Sink()
    monkeypatch.setattr(providers.subprocess, "Popen",
                        lambda argv, **kwargs: _FakeProc(stdin=sink))

    proc = start_agent_process(["claude", "-p"], "claude", "do the thing", "/repo")

    assert proc.stdin is sink
    assert sink.messages() == ["do the thing"]
    assert not sink.closed, "the console needs this pipe for the whole turn"


# --- the mailbox --------------------------------------------------------------


def test_a_note_with_no_turn_in_flight_waits_for_the_next_prompt():
    mailbox = operator.Mailbox()

    first = mailbox.submit("check the kit README")
    second = mailbox.submit("  and the export contract  ")

    assert (first.live, first.queued) == (False, 1)
    assert second.queued == 2
    assert mailbox.queued_count == 2
    assert mailbox.take_queued() == ["check the kit README",
                                     "and the export contract"]
    assert mailbox.queued_count == 0, "take_queued drains"


def test_an_empty_note_is_not_a_note():
    mailbox = operator.Mailbox()

    assert mailbox.submit("   ") is None
    assert mailbox.queued_count == 0


def test_a_live_note_is_framed_before_the_agent_sees_it():
    """A bare imperative arriving mid-turn reads like a prompt injection."""
    sink = _Sink()
    mailbox = operator.Mailbox()
    operator.AgentChannel(sink, mailbox)      # attaches itself

    delivery = mailbox.submit("use the kit's lathe")

    assert delivery.live
    written = sink.messages()[0]
    assert written.endswith("use the kit's lathe")
    assert operator.LIVE_NOTE_HEADER in written


def test_a_closed_channel_takes_the_mailbox_with_it():
    """Closing detaches, so a note typed afterwards is simply a queued note."""
    sink = _Sink()
    mailbox = operator.Mailbox()
    operator.AgentChannel(sink, mailbox).close()   # the result event won

    delivery = mailbox.submit("too late")

    assert not delivery.live and delivery.queued == 1
    assert delivery.error == "", "there was no live attempt to report"
    assert mailbox.take_queued() == ["too late"], "a lost note is never dropped"


def test_a_note_that_loses_the_race_with_the_pipe_is_queued():
    """The window the detach cannot close: the channel is still attached and
    the pipe under it is already gone."""
    sink = _Sink()
    mailbox = operator.Mailbox()
    operator.AgentChannel(sink, mailbox)
    sink.close()                    # the process died mid-turn

    delivery = mailbox.submit("too late")

    assert not delivery.live and delivery.queued == 1
    assert delivery.error, "a failed live attempt has to say so"
    assert mailbox.take_queued() == ["too late"]


def test_only_this_mailboxs_own_notes_are_claimed_from_the_replay():
    """--replay-user-messages echoes the ITERATION PROMPT back too."""
    sink = _Sink()
    mailbox = operator.Mailbox()
    operator.AgentChannel(sink, mailbox)      # attaches itself
    mailbox.submit("mind the bevels")
    framed = sink.messages()[0]

    assert mailbox.claim_echo("Generate Blender build scripts for …") is None
    assert mailbox.claim_echo(framed) == "mind the bevels"
    assert mailbox.claim_echo(framed) is None, "a receipt is claimed once"


def test_queued_notes_are_appended_to_the_prompt_as_a_named_section():
    prompt = operator.append_notes("Do the task.\n", ["one", "two"])

    assert prompt.startswith("Do the task.")
    assert operator.QUEUED_NOTES_HEADER in prompt
    assert prompt.rstrip().endswith("- two")
    assert operator.append_notes("Do the task.", []) == "Do the task."


# --- the runner: the channel's lifetime ----------------------------------------


def _events(*events):
    return [json.dumps(ev) for ev in events]


_RESULT = {"type": "result", "subtype": "success", "total_cost_usd": 0.1}


def _run_streaming(monkeypatch, lines, sink, mailbox):
    monkeypatch.setattr(
        cyclecore, "start_agent_process",
        lambda *args: _FakeProc(stdout=lines, stdin=sink))
    return cyclecore.run_agent_streaming(["claude", "-p"], "claude", False,
                                         prompt="task", mailbox=mailbox)


def test_the_result_event_closes_stdin_before_the_stream_ends(monkeypatch):
    """Closing stdin is how a streaming-input session ends — nothing else does.

    Observed from INSIDE the stream: the close after the loop would satisfy a
    test that only looks at the sink afterwards, so this one submits a note
    between the result event and the last line and requires it to be refused.
    """
    sink, mailbox = _Sink(), operator.Mailbox()
    afterwards = []

    def stream():
        yield json.dumps(_RESULT)
        afterwards.append(mailbox.submit("a moment too late"))
        yield json.dumps({"type": "system", "subtype": "anything"})

    assert _run_streaming(monkeypatch, stream(), sink, mailbox) == 0
    assert sink.closed
    assert not afterwards[0].live and afterwards[0].queued == 1, \
        "the turn had reported, so the note belongs to the next iteration"


def test_stdin_is_closed_even_when_no_result_ever_arrives(monkeypatch):
    """A crash that swallows `result` must not be able to hang the loop."""
    sink, mailbox = _Sink(), operator.Mailbox()

    _run_streaming(monkeypatch, _events({"type": "system", "subtype": "init"}),
                   sink, mailbox)

    assert sink.closed


def test_stdin_is_closed_when_the_render_loop_raises(monkeypatch):
    """The `finally`: an exception is the one path neither other close covers."""
    sink, mailbox = _Sink(), operator.Mailbox()

    def stream():
        yield json.dumps({"type": "system", "subtype": "init"})
        raise RuntimeError("the renderer blew up")

    with pytest.raises(RuntimeError):
        _run_streaming(monkeypatch, stream(), sink, mailbox)

    assert sink.closed


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_the_pipe_is_closed_even_with_nobody_listening(monkeypatch, runner):
    """`-j 2+` hands out no mailbox — and must still not leave the CLI alive.

    Measured before the fix: the turn's result came back in 2.7 s and the call
    returned only when the process was killed at 90 s, because the close was
    wired to the mailbox rather than to the transport.
    """
    sink = _Sink()
    proc = _FakeProc(stdout=_events(_RESULT), stdin=sink)

    if runner == "sequential":
        monkeypatch.setattr(cyclecore, "start_agent_process",
                            lambda *args: proc)
        cyclecore.run_agent_streaming(["claude", "-p"], "claude", False,
                                      prompt="task", mailbox=None)
    else:
        monkeypatch.setattr(parallel, "start_agent_process",
                            lambda *args: proc)
        parallel.run_job(1, AgentCommand("task", "opus", "item", "claude"))

    assert sink.closed


def test_a_note_typed_during_the_turn_reaches_the_agents_stdin(monkeypatch, capsys):
    """The whole point: advice about the task that is running, not the next one."""
    sink, mailbox = _Sink(), operator.Mailbox()
    delivered = []

    def stream():
        yield json.dumps({"type": "system", "subtype": "init", "model": "opus"})
        # Typed at the console while the turn is in flight. Called from here
        # rather than from a second thread so the ordering is a fact of the
        # test, not a race it has to win.
        delivered.append(mailbox.submit("skip the fasteners"))
        yield json.dumps({"type": "user", "message": {"role": "user", "content": [
            {"type": "text", "text": sink.messages()[-1]}]}})   # the CLI's replay
        yield json.dumps(_RESULT)

    _run_streaming(monkeypatch, stream(), sink, mailbox)

    assert delivered[0].live
    # start_agent_process is stubbed here, so the pipe holds the note alone;
    # that the PROMPT goes through the same pipe is pinned above.
    assert sink.messages() == [operator.frame_live("skip the fasteners")]
    # The replay is what puts the note in the console log, at the point in the
    # stream where the agent actually got it.
    assert "operator note: skip the fasteners" in capsys.readouterr().out


def test_the_replayed_iteration_prompt_is_not_printed_as_a_note(monkeypatch, capsys):
    sink, mailbox = _Sink(), operator.Mailbox()
    replayed_prompt = {"type": "user", "message": {"role": "user", "content": [
        {"type": "text", "text": "task"}]}}

    _run_streaming(monkeypatch, _events(replayed_prompt, _RESULT), sink, mailbox)

    assert "operator note" not in capsys.readouterr().out


# --- the runner: a queued note reaches the next prompt --------------------------


class _TwoIterationDriver(Driver):
    def __init__(self):
        self.served = 0

    def next_command(self):
        if self.served >= 2:
            return None
        self.served += 1
        return AgentCommand("Do the task.", "", f"item-{self.served}")


def _seq_args(project_dir):
    ns = type("NS", (), {})()
    # A bounded run: the loop then skips the usage machinery entirely, which is
    # what keeps this test off the network and away from the quota policy.
    ns.max = 3
    ns.dry_run = False
    ns.raw = False
    ns.start_in = None
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.cost = False
    ns.no_statusline = True
    ns.provider = None
    return ns


def test_a_queued_note_rides_the_next_iterations_prompt(tmp_path, monkeypatch):
    prompts = []
    mailboxes = []
    real_mailbox = operator.Mailbox

    def _mailbox():
        box = real_mailbox()
        mailboxes.append(box)
        return box

    monkeypatch.setattr(operator, "Mailbox", _mailbox)

    def fake_run(cmd, raw, partial, prompt="", mailbox=None):
        prompts.append(prompt)
        if len(prompts) == 1:
            mailbox.submit("prefer the kit's lathe")   # typed between iterations
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", fake_run)
    # Bounded runs skip the quota GATE but still bookend themselves with a usage
    # snapshot; a unit test has no business on the network for either.
    monkeypatch.setattr(cyclecore, "usage_source_for", lambda provider: None)

    cyclecore.run_loop(_TwoIterationDriver(), _seq_args(str(tmp_path)),
                       app_name="pytest-operator", setup_logging=False,
                       wait_on_start=False)

    assert prompts[0] == "Do the task."
    assert operator.QUEUED_NOTES_HEADER in prompts[1]
    assert prompts[1].rstrip().endswith("- prefer the kit's lathe")
    assert mailboxes[0].queued_count == 0


def test_no_live_messages_puts_the_prompt_back_in_argv(tmp_path, monkeypatch):
    """The rescue hatch, driven the way a user reaches it: through the runner."""
    seen = {}

    def fake_run(cmd, raw, partial, prompt="", mailbox=None):
        seen["argv"] = cmd
        seen["mailbox"] = mailbox
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", fake_run)
    monkeypatch.setattr(cyclecore, "usage_source_for", lambda provider: None)
    args = _seq_args(str(tmp_path))
    args.no_live_messages = True

    cyclecore.run_loop(_TwoIterationDriver(), args, app_name="pytest-operator",
                       setup_logging=False, wait_on_start=False)

    assert "Do the task." in seen["argv"], "the prompt has nowhere else to go"
    assert "--input-format" not in seen["argv"]
    assert "--replay-user-messages" not in seen["argv"]
    # The key still works; only the delivery is different, so the mailbox stays.
    assert seen["mailbox"] is not None


def test_a_note_the_run_never_delivered_is_reported(tmp_path, monkeypatch,
                                                    capsys):
    """A queued note promises "the next iteration" — there is not always one."""
    monkeypatch.setattr(cyclecore, "usage_source_for", lambda provider: None)

    def fake_run(cmd, raw, partial, prompt="", mailbox=None):
        mailbox.submit("typed while the last item was running")
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", fake_run)

    class _OneItem(_TwoIterationDriver):
        def next_command(self):
            return super().next_command() if self.served < 1 else None

    cyclecore.run_loop(_OneItem(), _seq_args(str(tmp_path)),
                       app_name="pytest-operator", setup_logging=False,
                       wait_on_start=False)

    out = capsys.readouterr().out
    assert "1 undelivered operator note" in out
    assert "typed while the last item was running" in out


def test_the_sequential_status_line_is_given_the_runs_mailbox(tmp_path,
                                                              monkeypatch):
    """The console's end of the wire: without this the `m` key is not offered."""
    seen = _capture_status_app(monkeypatch)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda cmd, raw, partial, prompt="", mailbox=None: 0)
    monkeypatch.setattr(cyclecore, "usage_source_for", lambda provider: None)

    cyclecore.run_loop(_TwoIterationDriver(), _seq_args(str(tmp_path)),
                       app_name="pytest-operator", setup_logging=False,
                       wait_on_start=False)

    assert isinstance(seen["messages"], operator.Mailbox)


# --- the console --------------------------------------------------------------


def _app(mailbox=None, stop_file=None):
    return sl.StatusApp(messages=mailbox, terminal=tio.NullTerminal(),
                        input_source=tio.NullInputSource(), refresh=60,
                        stop_file=stop_file)


def _type(app, text):
    for char in text:
        app.handle_event(tio.Key(char))


def test_the_message_key_is_absent_when_there_is_nobody_to_address():
    """-j 3 hands out no mailbox: several workers, one keyboard, no recipient."""
    keys = dict(_app().legend_entries())

    assert "m" not in keys
    assert "s" in keys, "the other keys are unaffected"


def test_typing_a_note_and_sending_it(tmp_path):
    mailbox = operator.Mailbox()
    app = _app(mailbox, stop_file=str(tmp_path / "stop"))

    app.handle_event(tio.Key("m"))
    _type(app, "use the studs")
    app.handle_event(tio.Key("\r"))

    assert mailbox.take_queued() == ["use the studs"]
    assert "queued for the next iteration" in app.status.note
    assert app.mode.name == "message" and app.mode.buffer == "", \
        "Enter sends and stays: leaving here would hand the rest of a paste " \
        "to the normal keys"


def test_a_pasted_newline_cannot_reach_the_stop_key(tmp_path):
    """One burst, keys delivered one by one — the tail of a paste is still text."""
    mailbox = operator.Mailbox()
    stop = tmp_path / "stop"
    app = _app(mailbox, stop_file=str(stop))

    app.handle_event(tio.Key("m"))
    _type(app, "check the sizes\r")
    _type(app, "stop after this one")

    assert not stop.exists(), "a pasted note requested a stop"
    assert mailbox.take_queued() == ["check the sizes"]
    assert app.mode.buffer == "stop after this one"


def test_escape_clears_before_it_leaves(tmp_path):
    """Alt+key arrives as ESC + key, so a one-step Esc would leak that key."""
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    _type(app, "half a thought")

    app.handle_event(tio.Key("\x1b"))
    assert app.mode.name == "message" and app.mode.buffer == ""

    app.handle_event(tio.Key("\x1b"))
    assert app.mode.name == "normal"


def test_the_legend_stops_offering_keys_the_editor_has_taken(tmp_path):
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    assert "s" in dict(app.legend_entries())

    app.handle_event(tio.Key("m"))

    keys = dict(app.legend_entries())
    assert "s" not in keys, "`s` is a letter in here — the legend must not lie"
    assert keys == {"Enter": "send", "Esc": "clear / leave",
                    "←/→": "move", "^W/^U/^K": "erase"}


def test_the_stop_key_inside_a_note_is_a_letter_not_a_stop(tmp_path):
    """The reason this is a Mode: `s` one layer down halts the run."""
    stop = tmp_path / "stop"
    app = _app(operator.Mailbox(), stop_file=str(stop))

    app.handle_event(tio.Key("m"))
    _type(app, "stop after this one")

    assert not stop.exists()
    assert app.mode.buffer == "stop after this one"


def test_backspace_in_both_spellings(tmp_path):
    mailbox = operator.Mailbox()
    app = _app(mailbox, stop_file=str(tmp_path / "stop"))

    app.handle_event(tio.Key("m"))
    _type(app, "abc")
    app.handle_event(tio.Key("\x08"))      # msvcrt spelling
    app.handle_event(tio.Key("\x7f"))      # POSIX spelling

    assert app.mode.buffer == "a"
    assert mailbox.queued_count == 0, "nothing is sent until Enter"


def test_the_prompt_row_shows_the_end_of_a_long_line(tmp_path):
    """Measured in COLUMNS: a note in a double-width script is the normal case
    for this project, and a row counted in code points would wrap — which pushes
    the reserved region up a line and desynchronises every later repaint."""
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    _type(app, "码" * 100)

    row = app.render(width=40)[-1]

    assert textwidth.cell_width(row) <= 40
    assert row.startswith(" ✉ …")
    assert row.endswith("码" + sl.MessagePromptRow.caret)


def test_an_unbound_key_is_swallowed_rather_than_dispatched(tmp_path):
    """Whatever the terminal sends, nothing may fall through to the normal keys.

    Discriminating on the NOTE: a key reaching NormalMode is answered with
    "unknown key", so the hint being replaced is the visible symptom of a key
    that fell through.
    """
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    hint = app.status.note

    app.handle_event(tio.Key("pgup"))
    app.handle_event(tio.Key("\x00"))

    assert app.mode.name == "message" and app.mode.buffer == ""
    assert app.status.note == hint


# --- editing the line ----------------------------------------------------------


def test_the_cursor_moves_and_typing_lands_where_it_points(tmp_path):
    """The report this fixes: the line was append-only, so a typo three words
    back could only be reached by erasing everything after it."""
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    _type(app, "use the studs")

    for _ in range(len("studs")):
        app.handle_event(tio.Key("left"))
    _type(app, "big ")

    assert app.mode.buffer == "use the big studs"

    app.handle_event(tio.Key("home"))
    _type(app, "please ")
    app.handle_event(tio.Key("end"))
    _type(app, "!")

    assert app.mode.buffer == "please use the big studs!"


def test_backspace_and_delete_work_off_the_cursor(tmp_path):
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    _type(app, "abXcd")
    for _ in range(3):
        app.handle_event(tio.Key("left"))

    app.handle_event(tio.Key("delete"))          # forward: eats the X
    assert app.mode.buffer == "abcd"

    app.handle_event(tio.Key("\x7f"))            # backward: eats the b
    assert app.mode.buffer == "acd"


def test_word_moves_and_word_erase(tmp_path):
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    _type(app, "measure the second shelf")

    app.handle_event(tio.Key("wordleft"))        # Ctrl+Left, both platforms
    app.handle_event(tio.Key("\x17"))            # Ctrl+W: erase "second "
    assert app.mode.buffer == "measure the shelf"

    app.handle_event(tio.Key("wordright"))       # to the end of "shelf"
    _type(app, "!")
    assert app.mode.buffer == "measure the shelf!"


def test_the_kill_keys_cut_from_the_cursor(tmp_path):
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    _type(app, "keep this drop that")
    for _ in range(len("drop that")):
        app.handle_event(tio.Key("left"))

    app.handle_event(tio.Key("\x0b"))            # Ctrl+K
    assert app.mode.buffer == "keep this "

    app.handle_event(tio.Key("\x15"))            # Ctrl+U
    assert app.mode.buffer == ""


def test_the_cursor_stays_visible_in_a_line_wider_than_the_row(tmp_path):
    """A window that only ever showed the tail would scroll the caret off the
    moment the typist went back to fix something."""
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    _type(app, "码" * 100)
    for _ in range(60):
        app.handle_event(tio.Key("left"))

    row = app.render(width=40)[-1]

    assert textwidth.cell_width(row) <= 40
    assert sl.MessagePromptRow.caret in row
    assert row.startswith(" ✉ …") and row.endswith("…")


def test_sending_resets_the_cursor_with_the_buffer(tmp_path):
    """A cursor left behind would put the next note's first character in the
    middle of nothing."""
    mailbox = operator.Mailbox()
    app = _app(mailbox, stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    _type(app, "first note")
    app.handle_event(tio.Key("home"))
    app.handle_event(tio.Key("\r"))

    _type(app, "second")

    assert app.mode.editor.cursor == len("second")
    assert app.mode.buffer == "second"


def test_a_pasted_line_lands_intact_at_the_cursor(tmp_path):
    """Ctrl+V is the terminal's own key: the clipboard arrives as an ordinary
    burst of characters, so paste is insertion — including into the middle."""
    app = _app(operator.Mailbox(), stop_file=str(tmp_path / "stop"))
    app.handle_event(tio.Key("m"))
    _type(app, "измерь полку")
    for _ in range(len("полку")):
        app.handle_event(tio.Key("left"))

    _type(app, "вторую\tсправа ")        # a tab inside the pasted fragment

    assert app.mode.buffer == "измерь вторую справа полку", \
        "a tab dropped outright would join the two words it separated"


def test_the_editor_refuses_what_it_does_not_own():
    """MessageMode gives Enter and Esc their own meaning, so the editor must not
    quietly consume them — nor any other key a later Mode may want."""
    editor = sl.LineEditor("ab")

    assert editor.handle("pgup") is False
    assert editor.handle("\r") is False
    assert editor.handle("\x1b") is False
    assert editor.buffer == "ab"


def test_the_legend_counts_what_is_still_waiting(tmp_path):
    mailbox = operator.Mailbox()
    app = _app(mailbox, stop_file=str(tmp_path / "stop"))
    mailbox.submit("one")
    mailbox.submit("two")

    assert dict(app.legend_entries())["m"] == "message (2 queued)"


# --- the parallel runner: one worker, one recipient -----------------------------


class _MemDriver(parallel.ListFileDriver):
    target_suffix = ".out.md"
    pick_order = "list"

    def __init__(self, items):
        super().__init__()
        self._items = list(items)
        self._lock = threading.Lock()

    def prompt(self, source, target):
        return f"do {os.path.basename(source)}"

    def pending_lines(self):
        with self._lock:
            return list(self._items)

    def strike(self, line):
        with self._lock:
            if line in self._items:
                self._items.remove(line)
                return True
            return False


def _par_args(project_dir, jobs):
    ns = type("NS", (), {})()
    ns.jobs = jobs
    ns.max = None
    ns.dry_run = False
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.ignore_usage = True
    ns.no_statusline = True
    return ns


@pytest.mark.parametrize("jobs, addressable", [(1, True), (2, False)])
def test_only_a_single_worker_run_has_a_mailbox(tmp_path, monkeypatch, jobs,
                                                addressable):
    seen = _capture_status_app(monkeypatch)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, cmd, mailbox=None: (0, 0.0, 0.01))

    parallel.run_parallel(_MemDriver(["a.md", "b.md"]),
                          _par_args(str(tmp_path), jobs),
                          app_name="pytest-operator-parallel")

    assert (seen["messages"] is not None) is addressable
