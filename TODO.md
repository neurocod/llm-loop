# TODO — llm-loop

A backlog of features worth borrowing from the wider "autonomous Claude-CLI
loop" ecosystem, scored against what this engine already does. Implement the
*behaviour* in Python — the reference tools are mostly Bash+jq, so their code
doesn't port directly, but the ideas do (we already parse stream-json and
`total_cost_usd`, so most of these are 10–30 lines).

Reference projects this is mined from:
- frankbria/ralph-claude-code (Bash) — richest feature set
- AnandChowdhary/continuous-claude (Bash + PowerShell) — PR/CI lifecycle
- anthropics/claude-code → plugin `ralph-wiggum` — completion promise, stop hook

Legend: priority is rough — `[P1]` do first, `[P3]` nice to have.

## Already done (for reference — don't re-add)

- Fresh context per iteration, state in files (the canonical Ralph pattern).
- Pause on the account's real quota %, via a per-Driver `limit_policy` (usage.py +
  limits.py): SessionLimit / DayNightLimit / WeeklyLimit rules, composable,
  day/night dynamic ceiling on the session rule. Read straight from the usage
  endpoint (no tokens), with the run's own `rate_limit_event` as the backstop.
- Idempotent list draining (strike each done path out of the list file).
- Git-push policy (none | after_new_commits | each_hour), final push on exit.
- Rotating mirror log, `--dry-run`, `--max-runs`, `--start-in`, `--raw`.
- Sequential `run_loop` + parallel `run_parallel` (N concurrent workers).
- Project-root decoupling (`--project-dir` / cwd), stop file.
- Opt-in per-user completion sound, configured through the platform config
  directory (or `LLM_LOOP_SETTINGS`).

## Worth borrowing

### [P1] Completion signal ("completion promise")
Source: all three reference tools.
StateFileDriver currently stops only on an `error` state or runs forever; there
is no positive "done" detection. Add an opt-in stop when `claude` outputs an
exact phrase (e.g. `<promise>ALL_DONE</promise>`) or the state file's first line
becomes a configured token (e.g. `done`).
- Require N consecutive signals before stopping (continuous-claude default 3) to
  avoid a false "done".
- Plumb through as a Driver hook / `--completion-signal` + `--completion-threshold`.
- The exact-string matching is brittle (Ralph plugin warns about this) — keep
  `--max-runs` as the primary safety net regardless.

### [P1] Circuit breaker — stuck-loop detection
Source: frankbria (its killer feature).
Today we retry transient errors (5 in a row → stop) but never catch "claude
exits 0 yet does nothing". For a forever state machine this is the key guard
against silently burning budget.
- Simplest fit: if `git status`/`git diff` shows no change for N iterations in a
  row (no commit, state file unchanged), stop (or pause).
- Also stop on M identical non-zero errors in a row.
- Expose thresholds: `--no-progress-limit N` (default 3), `--same-error-limit M`.

### [P2] Cost & duration stop conditions
Source: continuous-claude (`--max-cost`, `--max-duration`).
We already extract `total_cost_usd` per iteration from the result event.
- `--max-cost USD` — stop once cumulative spend crosses the ceiling.
- `--max-duration 2h` — wall-clock cap for the whole run (total runtime, unlike
  the usage-limit gate, which is about the session window).

### [P2] Structured per-iteration metrics (JSONL)
Source: frankbria (`metrics.jsonl`).
Append one line per iteration next to the mirror log:
`{loop, label, duration_s, cost_usd, returncode, status}`. We already compute all
of this for the console output — just persist it. Would let `--cost` read
numbers instead of re-parsing its own prose, and gives per-run analytics.

### [P2] Configurable allowed-tools
Source: continuous-claude / frankbria.
`build_claude_argv` hardcodes the tool list. Make it a Driver field /
`--allowed-tools` option (support granular patterns like `Bash(git *)`). Small
change, removes a magic string, tightens safety for unattended runs.

### [P3] Calls-per-hour throttle
Source: frankbria (`MAX_CALLS_PER_HOUR`), continuous-claude (`--max-calls-per-hour`).
A simple call-rate cap, complementary to the existing usage-% gate.

### [P3] Optional reviewer / verify pass
Source: continuous-claude (`-r/--review-prompt`).
After each iteration, optionally run a second `claude` pass that reviews the
diff, runs tests/lint, and verifies the change before moving on. Model it as an
optional follow-up command the Driver can supply.

### [P3] Stall self-correction — write diagnostics back into state
Source: continuous-claude (`--stall-threshold` appends diagnostics to notes).
On repeated failure, append a short diagnostic note into the state file
(`currentState.md`) so the next fresh-context iteration sees what went wrong.

### [P3] Git backup / rollback per iteration
Source: frankbria (`--backup` / `--rollback`).
Optionally create a backup branch before each iteration so a bad iteration can
be reverted. Lower value given push-forward workflow, but cheap insurance.

## Known defects

### [P2] A worker that dies mid-item hangs the whole parallel run
`parallel.worker` has no `try` around its per-item body. An exception between
`claim()` and `finish()` (or in `run_job`) ends that thread with the line still
in `shared.in_progress`, so it is never struck and never returned to the queue —
and every other worker then loops forever in the claim back-off, because
`claim()` keeps returning None while neither `stop` nor `claims_closed` is set.
The run never terminates and its `join()` never returns. Found while changing
`run_job`'s signature: the stale test stubs raised TypeError inside the workers
and the suite hung instead of failing. Fix: wrap the body, `shared.release(line)`
on an unexpected exception, and treat it as a failed attempt.

### [P3] A `p` pause does not survive a wrapper's batch boundary

Each runner call builds its own `StatusApp`, so `_paused` dies with it — the
same trap `s` has (and works around by returning a `RunResult.reason` the
wrapper acts on, see `REQUESTED_STOP_REASONS`). A wrapper that slices one
invocation into several runner calls — the host project's
`runGenerateModels.py -p`, which calls `run_parallel` once per batch — therefore
starts the next batch unpaused, silently, after the user held the run. Nothing
carries the flag across: `InvocationProgress` is the seam for invocation-wide
state, but it holds figures, not requests. Not reachable from `runCycle.py`
(one call per process), which is why it is here rather than fixed.

### [P3] Nothing ties `FLAG_ALIASES` to the parsers it mirrors
`cmdline.FLAG_ALIASES` is hand-kept in step with `cyclecore.parse_args`,
`parallel.parse_args` and the host wrapper's own flags — its comment says so and
says a missing spelling is not cosmetic. But no test asserts the coverage, so
forgetting an entry leaves the suite fully green; the symptom appears later, in
the status line's "reproduce this run" command, where the forgotten flag's value
token is read as a flag. A parametrized case needs the parser objects, which
`parse_args` builds locally and never returns — so the fix starts with splitting
out a `build_parser()`. Found by review of the `--cost-log` commit (that entry
was added, and would have been just as green if it had not been).

## Structure worth doing when something takes you there anyway

Three findings from the structural pass over the operator-note commits. None is
worth its churn on its own; each has a trigger that makes it cheap.

- **`statusline.py`, lines ~894–1290 are a separable front end.** `Terminal`,
  `NullTerminal`, `terminal_for`, the `InputEvent` family, `decode_escape`,
  `InputSource`, `TerminalInput`, `_EscapeDecoder` — ~400 lines that reference
  `LoopStatus`, `Row` and `Segment` exactly zero times. Trigger: a second front
  end, or a platform that needs its own reader. Not before — the operator-note
  change touched three regions of this file at once, and under any split that
  would have been a three-file change with two new import edges.
- **`operator.py` vs `providers.py` could be split by ownership** — providers
  owns the pipe (open, wire format, channel, close), operator owns the policy
  (framing, queue, receipts). Today `user_message_line` (a stream-json fact)
  lives in operator because `AgentChannel` needs it. Trigger: a second transport
  with a different wire format.
- **Test fixtures: three `_MemDriver` subclasses and seven args namespaces**
  across `test_parallel_statusline`, `test_parallel_termination` and
  `test_operator_messages`, with no `conftest.py` anywhere. The cost is three
  copies of a driver that must stay behaviourally identical for the parallel
  tests to mean the same thing. Trigger: the next test file that needs a fourth.

## Known gaps

- **Two tests read a rich-wrapped capture, so a narrow terminal fails them.**
  `test_operator_messages.py::test_a_note_the_run_never_delivered_is_reported`
  and `test_usage_limits.py::test_a_reading_without_a_reset_time_still_prints`
  assert on a whole sentence in `capsys` output, but `print_error` /
  `print_percents` hand their line to a rich Console, which word-wraps it at the
  console width — and rich reads `$COLUMNS`. Both pass at 80+ columns and fail
  under `COLUMNS=40`. Pre-existing, unrelated to what those tests are about; the
  fix is to assert on the plain copy handed to `print_markup` (as the width
  tests in `test_providers.py` do) rather than on the rendered capture.

## Deferred (revisit later — not now, but worth keeping on the radar)

- **Setup wizard** (frankbria `ralph-enable`) — interactive bootstrap that
  detects project type/framework and generates the loop's config + prompt/task
  files. Overkill for two modes today; useful if the tool gets reused across
  many projects.
- **Queue system** (frankbria `ralph-queue`) — a persistent task queue
  (`add`/`status`/`reorder`/`remove`, priorities, dependencies,
  `--halt-on-failure`). Our `list.md` is a flat queue already; a richer queue
  pays off only once tasks need ordering/dependencies.
- **PRD import** (frankbria `ralph-import`) — turn a free-form requirements doc
  (Markdown/txt/JSON/DOCX/PDF) into the loop's task/prompt files, with a
  completeness score. We fill `TODO.md`/`currentState.md` by hand for now; if
  wanted, a lightweight version is just a one-shot claude prompt, no DOCX/PDF
  parser.
- **`--worktree` isolation for parallel** (continuous-claude) — our parallel
  workers write distinct `*.ru.md` files, so there is no conflict today. Needed
  only if we ever parallelise edits to *code* (overlapping writes).

## Deliberately NOT doing (out of scope / wrong fit)

- `--continue` session continuity — we deliberately use fresh context + files
  (the recommended Ralph pattern); don't switch to resuming sessions.
