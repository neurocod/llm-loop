# TODO — claude-loop

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
- Rotating mirror log, `--dry-run`, `--max-runs`, `--start-in`, `--max-strike`, `--raw`.
- Sequential `run_loop` + parallel `run_parallel` (N concurrent workers).
- Project-root decoupling (`--project-dir` / cwd), stop file.
- Opt-in per-user completion sound, configured through the platform config
  directory (or `CLAUDE_LOOP_SETTINGS`).

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
- `--max-duration 2h` — wall-clock cap for the whole run (complements
  `--max-strike`, which is about the session window, not total runtime).

### [P2] Structured per-iteration metrics (JSONL)
Source: frankbria (`metrics.jsonl`).
Append one line per iteration next to the mirror log:
`{loop, label, duration_s, cost_usd, returncode, status}`. We already compute all
of this for the console output — just persist it. Makes `sum_session_costs.py`
trivial and gives per-run analytics.

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
