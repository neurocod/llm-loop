# llm-loop

**Run a coding agent unattended for hours.** `llm-loop` calls a headless LLM CLI
(Claude Code or Codex) over and over, each time with a *fresh* context window,
while the actual progress lives where it survives a context reset: in your files
and in git. The loop stops itself when the account's quota runs low, when the
agent reports a dead end, or when you ask it to — and it is a Python package
with zero third-party dependencies, so it behaves identically on Windows, Linux
and macOS.

## llm-loop vs. the Ralph loop

The technique this builds on is Geoffrey Huntley's
[Ralph](https://ghuntley.com/ralph/) — *"Ralph is a bash loop"*:

```bash
while :; do cat PROMPT.md | claude-code ; done
```

That one line is the whole idea, and it is a good one. Everything below is the
scaffolding you end up writing yourself the moment you leave such a loop running
overnight.

| | Ralph (the canonical loop) | llm-loop |
|---|---|---|
| Implementation | a bash one-liner; forks exist in bash/PowerShell | a Python package (3.9+, **stdlib only**), one code path on Windows/Linux/macOS |
| Iteration body | the same prompt file, every time | a pluggable `Driver`: state machine, work queue, or your own |
| Multi-step work | one undifferentiated "do the next thing" step | explicit phases; **the state is read back by the script**, not only by the agent |
| Prompt | fixed file | `prompt()` is computed per iteration from the current state / work item |
| Model choice | one model for the whole run | `model()` per iteration — a cheap model for `cleanup`, a smart one for `implementation` |
| Quota / rate limits | run until the CLI dies | the account's real figures are read over the provider's API **before** each iteration, and the loop waits under configurable ceilings (session / day-night / weekly), with a reactive `rate_limit_event` backstop |
| Parallelism | sequential (fan-out only *inside* one agent) | `-j N` concurrent CLI workers draining a work-queue file, one item per job |
| Stopping | Ctrl+C | `s` key (this run only), `stop` sentinel file (every run in the root), `--max-runs`, and an `error` state that halts for a human |
| Steering | stop it, edit the prompt, start again | `m` types a note into the turn already running (or queues it for the next one); `p` holds the loop at an iteration boundary so the files it reads can be edited, `p` again lets it go |
| Providers | whatever the pipe points at | Claude and Codex adapters (argv vs. stdin prompt transport, both stream-json rendered) |
| Observability | terminal scrollback | pinned status line (iteration, model, elapsed, live quota, per-job rows), rotating mirror log, optional Markdown rendering |
| git | your problem | `--git-push none\|after_new_commits\|each_hour` |
| Setup cost | none | one submodule + a ~15-line wrapper |

Nothing here replaces the idea; it makes the idea survivable unattended.

## Why the extra machinery pays off

You write only *what work to do each iteration* — a `Driver` subclass, usually
some fifteen lines. Two are ready-made, and the shape decides the rest.

**A state machine (`StateFileDriver`) gives each step a clean context.** The
loop is, in pseudocode:

```
while not error and not limits_reached:
    claude|codex -p "Follow the instructions in currentState.md"
    state = first_line_of("currentState.md")
```

and `currentState.md` is a playbook whose first line names the current step:

```
Current state: plan mode

If the state is "plan mode": read TODO.md, choose ONE task, write it to
currentTask.md, set the state to "implementation", exit.

If the state is "implementation": implement currentTask.md,
set the state to "cleanup", exit.

If the state is "cleanup": review the changes, commit, delete currentTask.md,
set the state to "plan mode", exit.
```

That sketch is boiled down to the mechanism. A playbook that survives real
unattended nights is considerably more detailed — where the task comes from when
the TODO list runs dry, when a test is written, what `cleanup` is allowed to
revert, what counts as a dead end — see
[`examples/currentState.md`](examples/currentState.md), the file this project's
own loop runs on.

Because every step is its own process, the implementation step is not reading a
context window already half-full of the *analysis* that chose the task —
planning can be as heavy as it likes and still costs the implementation nothing.
Add as many steps as your project deserves: human review, refactor pass,
benchmark, docs.

**The script sees the step, so it can specialise it.** The state line comes back
to the Python side, which is what makes per-step model selection, per-step
prompts and per-step limits possible at all — `cleanup` on a cheap model,
`implementation` on the expensive one.

**The other shape is a work queue (`ListFileDriver`)** — a list file that the
loop drains, striking each item out once its command succeeds:

```
while not error and not limits_reached:
    line = read_line_from(list_file)
    run("claude|codex", f"Translate {line} to Portuguese into Port-{line}, "
                        f"then strike {line} out of {list_file}")
```

Idempotent by construction: kill it any time, relaunch, it picks up the
remainder. And since the items are independent, they can run **N at a time**:

```
[job 3] 💻 pdfinfo 'D:\g\3d-research\musical-instruments\acoustic-upright-piano\kawai-k-series-…
[job 3] 📤 exit 0
[job 4] 💻 curl.exe -L --fail --output 'D:\g\3d-research\food-and-beverages\eggs-dozen\thirty-…

──────────────────────────────
 ⟳ iter 4/12 | codex | 4 jobs | 1m37s | session n/a | week 26% (6d8h) / ceil 95%
 job 1 ▶ gpt-5.6-terra | iter 1    | 1m37s  | calculator-desktop.md
 job 2 ▶ gpt-5.6-terra | iter 1    | 1m36s  | terminal-block-connector.md
 job 3 ▶ gpt-5.6-terra | iter 1    | 1m36s  | acoustic-upright-piano.md
 job 4 ▶ gpt-5.6-terra | iter 1    | 1m36s  | eggs-dozen.md
 keys: s stop | p pause | h help
```

## A playbook worth stealing

The state machine is the headline shape, and a playbook for it is where the
project-specific thinking goes. The shipped
[`examples/currentState.md`](examples/currentState.md) cycles like this:

```
         ┌──────────────────────────────────────────────┐
         ▼  (loop back once the task is wrapped up)     │
  ┌─────────────┐                                       │
  │  plan mode  │  look around, pick a task,            │
  │             │  fill currentTask.md                  │
  └──────┬──────┘                                       │
         ▼ task ready                                   │
  ┌─────────────┐                                       │
  │  implement  │ build it, write a self-test,          │
  │             │  commit intermediate work             │
  └──────┬──────┘                                       │
         ▼ code written                                 │
  ┌─────────────┐                                       │
  │   cleanup   │  commit or revert, tidy comments,     │
  │             │  update TODO / README                 │
  └──────┬──────┘                                       │
         │ task done                                    │
         └──────────────────────────────────────────────┘

  · · · from ANY state, on an unrecoverable problem · · ·
                          │
                          ▼
                  ┌─────────────┐
                  │    error    │  loop halts until a human
                  │   (halt)    │  resets the state line
                  └─────────────┘
```

## Requirements

**Python 3.9+ and nothing else.** The engine has zero third-party
dependencies — every import across its modules comes from the standard
library — so vendoring it adds no transitive dependency to your project and
cannot conflict with the versions you already pin.

**Plus the CLI you intend to drive, already signed in.** The loop authenticates
nothing itself: it shells out to `claude` or `codex`, and reads the quota
figures through that CLI's own login (see *Usage limits*). So an unauthenticated
CLI fails twice over — every iteration exits without doing any work, and the
limit gate, having no figures, cannot tell that it should have paused. Check
before a long run:

```bash
claude auth status     # JSON; "loggedIn": true is the field that matters
codex login status     # e.g. "Logged in using ChatGPT"
```

Sign in with `claude auth login` / `codex login`. Both keep the session on disk,
so this is a one-off per machine, not per run. For Claude the loop also accepts
`CLAUDE_CODE_OAUTH_TOKEN`, and honours `CLAUDE_CONFIG_DIR` when looking for the
credentials — useful on a CI box with no interactive login.

## Layout

```
src/
  llm_loop/
    cyclecore.py     engine: parse_args and the sequential run_loop
    streamrender.py  one provider run rendered live, and the single-stream state
                     that only a runner with one in flight can hold
    wire.py          the words the provider stream is made of, so both renderers
                     read the same event by the same name
    runlifecycle.py  the prologue and epilogue every run has, and the live
                     knobs (RunSettings) both runners read
    agentwork.py     what one unit of work is, and the Driver protocol a
                     wrapper subclasses to produce them
    console.py       what a run prints, and the rotating mirror log that is the
                     second copy of every line of it
    stopchannel.py   how a run is asked to stop or hold, and what it reports
    projectroot.py   where the project being driven is, and the one place that
                     answers it
    gitpush.py       when a run pushes what it has committed, and where
    providers.py     Claude/Codex executable flags, argv construction, and the
                     ending owed to a provider child
    drivers.py       StateFileDriver (state machine) and ListFileDriver (work queue)
    parallel.py      run_parallel: N concurrent LLM workers over a list file
    operator.py      notes typed at the console, on their way to the running agent
    usage.py         what is known about a quota: the queried figures
                     (UsageSource) and the verdict the wire streams back
    codex_usage.py   the same figures for Codex, over its own app-server protocol
    limits.py        LimitPolicy + SessionLimit / DayNightLimit / WeeklyLimit rules
    statusline.py    the status area pinned under a run: its rows, and the keys
    termio.py        the terminal underneath: reserved region, window title, keys
    textwidth.py     how wide terminal text is, and how much a line may hold
    compactline.py   the one-event-one-line shapes both runners print
    clispec.py       every option the family's command lines carry, declared
                     once: both parsers and cmdline's alias table come from it
    cmdline.py       the command line that would reproduce this run
    notifications.py opt-in settings and completion sounds for unattended runs
    exitlog.py       why a run ended — including the endings it cannot report itself
examples/
  runCycle.py            state-machine wrapper
  runFileList.py         per-file work-queue wrapper
  runFileListParallel.py parallel work-queue wrapper
ideas/                   study material, never shipped — see ideas/README.md
```

One line per module, and every module has one: the map is worth its upkeep only
while it is complete — a module missing from it is a module nobody browsing the
package knows to open. A new module gets its line here, saying what it owns.

The package sits under `src/` for one reason, and it is the reason to keep it
there: a host project puts that directory on `sys.path`, so whatever sits
directly inside it becomes a top-level importable name for them — at position
0, outranking their own modules. With `src/` holding nothing but the package,
vendoring costs an adopter exactly one name. Flat at the repository root it
would also hand them `tests`, which nearly every project already has.
`tests/test_vendoring_footprint.py` keeps that honest.

## Use it from a host project

Add it as a submodule, then copy one of the [`examples/`](examples/) wrappers
into your project root and adjust the paths, prompt and model:

```bash
git submodule add https://github.com/neurocod/llm-loop.git tools/llm-loop
cp tools/llm-loop/examples/runFileList.py .   # then edit the constants
```

Vendoring is the intended route: it pins the engine to a commit your project
controls, and a wrapper needs no install step. Where the code sits and where
your project sits are kept separate — the engine anchors every project-relative
operation (git/agent cwd, the stop file, the relative paths a Driver is handed)
to the project root: the current working directory by default, or
`--project-dir`/`-C`. If you would rather have it on
the import path proper, it is a normal PEP 621 project:

```bash
pip install git+https://github.com/neurocod/llm-loop.git
```

then drop the `sys.path.insert` line from the wrapper. Note that `pip install
llm-loop` does **not** get you this project — that name on PyPI belongs to an
unrelated 2023 package.

A wrapper is tiny — subclass a Driver, set the project-specific bits as class
attributes / an overridden `prompt()`, and call `.main()`:

```python
# runFileList.py  (in your project root)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools", "llm-loop", "src"))

from llm_loop import ListFileDriver

class FileListDriver(ListFileDriver):
    list_file     = "files.md"
    target_suffix = ".summary.md"
    pick_order    = "random"   # or "list" to walk the list top to bottom

    def model(self):
        return "sonnet"   # or "" to use the CLI's own configured model

    def prompt(self, source, target):
        return (f"Read {source} and write a Markdown summary to {target}. "
                f"Do not modify {source}.")

if __name__ == "__main__":
    FileListDriver.main()
```

Run it from the project root so the working directory is the project root (or
pass `--project-dir <path>` from anywhere):

```
python runFileList.py               # drain the list, one file per iteration
python runFileList.py --codex
python runFileList.py --max-runs 5  # at most 5 iterations
python runFileList.py --dry-run     # print the commands, run nothing
```

Items go out in random order by default, which keeps an interrupted run from
draining one section of a grouped list before touching the rest. Set
`pick_order = "list"` (or override `pick(pending)`) to hand them out top to
bottom instead — the setting governs the parallel runner too.

Claude remains the default for compatibility. Select Codex per invocation with
`--codex`, or set `provider = "codex"` on the Driver subclass. An empty
`model()` result lets the selected CLI use its configured default; a non-empty
result is forwarded to that provider. Codex normally runs through its
app-server protocol, so `m` can steer the active turn; the same sequential and
parallel renderers show agent messages, commands, file changes, failures, and
final token counts. `--no-live-messages` falls back to `codex exec --json` and
sends the prompt through a closed stdin stream (`codex exec ... -`) instead of
argv. Both transports avoid Windows command-line length limits without entering
the interactive UI. Claude keeps its existing provider-specific transport.
The adapter grants writes only to the workspace and routes any approval through
Codex's automatic reviewer; it does not bypass the sandbox.

### State-machine driver

Drives the state machine shown in the diagram above — the state file's first
line names the current mode, and an overridden `model()` picks the model per mode
(drop it to let the CLI use its own configured model):

```python
from llm_loop import StateFileDriver

class CycleDriver(StateFileDriver):
    state_file = "currentState.md"

    def model(self):
        return "opus"   # vary by self.first_line(), or "" for the CLI default

if __name__ == "__main__":
    CycleDriver.main()
```

### Parallel work queue

Any `ListFileDriver` subclass also runs concurrently via `.main_parallel()` — no
extra code, just a different entry point. Put it in its own wrapper file and the
derived `app_name` / `prog` give it a separate log file and `--help` name for
free:

```python
class FileListParallelDriver(FileListDriver):
    pass

if __name__ == "__main__":
    FileListParallelDriver.main_parallel()
```

### Wrapper options in `--help`

A wrapper that offers a mode of its own has to read it out of `argv` itself —
the two parsers have disjoint option sets, so the flag choosing between them
cannot be an option of either. `add_cli_options` is where such a flag gets
documented, and both entry points call it, so one override covers both `--help`
texts:

```python
class FileListDriver(ListFileDriver):
    @classmethod
    def add_cli_options(cls, parser):
        group = parser.add_argument_group("modes", "read before the options above")
        group.add_argument(MY_FLAG, action=ConsumedByWrapperAction,
                           help="what it switches on")
```

`ConsumedByWrapperAction` documents without parsing: it errors if the option
ever reaches the parser, which means the wrapper's own scan missed a spelling
(an abbreviation argparse resolves and a plain `argv` scan does not) and the run
would otherwise have gone ahead in the default mode. Anything the engine *should*
parse is an ordinary `add_argument` here instead.

## Usage limits

Before each iteration (and after any failed one) the loop reads the account's
quota figures and pauses if a watched one is at/over its ceiling. Which
quota, and at what ceiling, is a *specialisation* you pick by setting the
`limit_policy` class attribute on your Driver — a `LimitPolicy` holding one or
more rules; the loop pauses while **any** of them is exceeded:

```python
from llm_loop import (LimitPolicy, SessionLimit, DayNightLimit, WeeklyLimit)

class MyDriver(StateFileDriver):
    # pick ONE of these:
    limit_policy = LimitPolicy([SessionLimit(80)])                  # flat session cap
    limit_policy = LimitPolicy([DayNightLimit()])                   # smart session (default)
    limit_policy = LimitPolicy([WeeklyLimit(90)])                   # weekly cap
    limit_policy = LimitPolicy([DayNightLimit(), WeeklyLimit(90)])  # composite
```

| Rule | Watches | Ceiling behaviour |
|---|---|---|
| `SessionLimit(limit)` | the ~5-hour session | flat `limit`%, wait out the window |
| `DayNightLimit(day=, night=, deadline_hour=)` | the session | day/night base + a climb toward 100% as the window nears its reset |
| `WeeklyLimit(limit, sonnet_only=)` | the weekly quota | flat `limit`%, wait out the week |

Leaving `limit_policy` unset keeps Claude's historical
`LimitPolicy([DayNightLimit()])`. Codex defaults to the composite session and
weekly policy because plans may expose both windows or only a weekly window.
The bookend usage snapshots and the per-check log lines report exactly the quotas
the active policy watches — those are records of the gate. The pinned status line
is not: it lists every window the provider meters (see below). When `--max-runs N`
is given (a short bounded run) the limit gate is skipped entirely.

The hold itself watches both stop channels, so a run parked on the wall — which
in a parallel run means every worker at once, with nothing else left to notice a
keypress — still answers `s` and the `stop` file within a quarter-second instead
of only when the window resets.

**Where the figures come from.** For Claude, `usage.py` reads them over HTTP from the same
endpoint the CLI's own `/usage` panel and status line are built from,
authenticated with the OAuth token the CLI keeps in `~/.claude/.credentials.json`
(`CLAUDE_CONFIG_DIR` / `CLAUDE_CODE_OAUTH_TOKEN` / `ANTHROPIC_BASE_URL` are
honoured). No tokens, no model turn — the earlier `claude -p "/usage"` round-trip
that this replaced cost $0.33 and 17 s per check, i.e. it spent a slice of the
very budget it was measuring.

For Codex, `codex_usage.py` starts `codex app-server` and calls the official
`account/rateLimits/read` JSONL/JSON-RPC method using the CLI's existing login.
It does not start a thread or model turn. The returned window durations decide
whether a reading is the short session quota or the long weekly quota, including
weekly-only plans where Codex reports the seven-day window as `primary`.

The Claude endpoint is undocumented, and either provider query can fail because
of a temporary network, CLI, or authentication problem. A failure is therefore
treated as "no figures" rather than a fatal error. Claude also has a reactive
backstop: every run streams its own `rate_limit_event` verdict, and a `rejected`
makes the loop wait out exactly the quota that refused it
(`usage.RateLimitEvent`). A failed Codex turn forces a fresh app-server
reading before the bounded retry path continues.

## Common options

| Option | Meaning |
|---|---|
| `--codex` | run Codex CLI instead of the Driver's default provider |
| `-m, --max-runs N` | stop after N iterations (sequential) / N files total (parallel); `--max` is a deprecated alias |
| `-d, --dry-run` | print the commands, run nothing |
| `-g, --git-push none\|after_new_commits\|each_hour` | when to `git push` |
| `-C, --project-dir DIR` | project root (default: cwd) |
| `-s, --start-in 29m` | wait before starting (sequential only); alias `--startIn` |
| `-j, --jobs N` | concurrent workers (parallel only; else the Driver's `jobs`, else 10) |
| `-c, --cost` | print per-session cost totals from the mirror log and exit (sequential only) |
| `--cost-log LOG` | report on this log instead of this entry point's own — a rotated backup or a copy; implies `--cost` (sequential only) |
| `--ignore-usage` | don't pause on the session budget (parallel only) |
| `--raw` | print raw JSON events, for debugging (sequential only) |
| `--no-statusline` | do not pin the status rows (same as `LLM_LOOP_STATUSLINE=0`) |
| `--no-live-messages` | notes typed with `m` wait for the next prompt instead of going into the running turn (same as `LLM_LOOP_LIVE_MESSAGES=0`) |

## Why the run ended

Every run closes the mirror log with one line naming its ending:

```
=== run ended: no more work in the queue · 21 iteration(s), 21 completed · 4h12m ===
```

A stop file, the `s` key, `--max-runs`, a driver stop, five provider errors in a row, an
unhandled exception, Ctrl+C, a signal — each gets its own phrase there. What
cannot get one is a kill from outside (`Stop-Process`, `taskkill /F`, an OOM
kill, a power cut): the process is gone, so it writes nothing at all. For those,
each run keeps a small `<app>-<project>.<pid>.run.json` record beside its log
and removes it on the way out — a record still on disk whose owner is gone is
the report, and the next run prints it, naming the pid, the moment it was last
alive and the item it was working on. `llm_loop.exitlog` is the whole of it.

## Interactive status line

On a terminal, a run pins a few rows at the bottom — iteration, provider/model,
elapsed time, the provider's live quota figures and the script's own limits, one
row per job, and a legend of the keys it answers to. Piped output, CI and
`--no-statusline` get the plain scrolling output of before.

The same run also names the window (and the tab, and the taskbar button):
`⟳ iter 12/40 · bmx-bike.md` — the row's first field, then what each still-busy
job is on, so a run left in the background stays readable with the terminal
itself out of sight. It shares the rows' kill switch — piped output, CI and
`--no-statusline` get no title either — and the name is handed back when the run
ends, whichever way it ends.

The iteration counter carries the whole of "how much work is this run": its
denominator is `--max-runs`, whatever the driver reports from
`Driver.pending_total()`, or the smaller of the two. `--max-runs` therefore
gets no field of its own — a knob
whose value is already on the row registers with `show_in_status=False` and
stays editable and reproducible without spending row width twice.

**The quota fields have two halves**, split by ` / `:

```
 session 43% (2h11m) / ceil 95% | week 63% (17h27m)
 └──────── the provider ──────┘ └── the policy ──┘
```

Left of the slash is the account's own report — how much of the window is spent
and how long it still has to run. Every window the provider meters gets a field,
whether or not the run is gated on it, so the reader sees the whole budget and
not just the slice that happens to be watched: the 5-hour session and the week
are always there (`n/a` when the provider reports no figure), and a window a plan
does not have is dropped unless a rule watches it. Right of the slash is what the
`LimitPolicy` rule for that window makes of it — its live ceiling by default,
absent entirely when no rule watches it. A custom rule can say something else by
overriding `LimitRule.status(reading, now)`.

Keys: `s` requests a graceful stop of **this** run (pressing `s` again during
the countdown cancels it), `p` holds it at the next iteration boundary and `p`
again lets it go (see below), `m` sends the agent a note (see below), `h` or `?`
shows the full key list.

`s` sets an in-process flag and writes nothing to disk, so several loops
launched in one project root are stopped one at a time — the terminal you type
it into is the run that halts. The `stop` file below is the other channel: it
stops every run watching the root, survives the process, and is therefore what
scripts and run-chaining use. A stop file this run merely obeys is not the key's
to withdraw, so `s` neither writes nor removes it.

Where the two meet, the file wins: a run that stops while a sentinel is on disk
consumes it, whether or not `s` was also pressed. Otherwise the file would
outlive the loop it was written for and the launch queued behind it would wait
forever.

## Holding the run (`p`)

`p` pauses the loop and `p` again resumes it. The iteration in flight is never
interrupted — it finishes its one state transition, and the next one does not
begin. That gap is the point: nothing the loop reads (a state file, a task list,
the tree itself) is being written while it holds, so it can be edited, and the
next iteration is what reads the edit. In a parallel run the same key stops new
files being claimed, and whatever is already running finishes.

**Read the row, not the keypress**, because the two are not the same moment:

```
 ⟳ iter 12/40 | … | PAUSING — the iteration in flight finishes first
 ⏸ iter 12/40 | … | PAUSED — press p to resume
```

Only the second line means the run is standing still and the files are yours.
(`pause_state` in `statusline.py` is the whole rule.)

Held, a run still answers everything else: `s` ends the hold and stops the run
(with its usual cancel countdown), the `stop` file does the same, and `m` queues
a note that rides the next iteration's prompt. Like `s`, `p` is in-process and
writes nothing to disk — it holds the run whose terminal it was typed into, and
the ones next door work on.

What a hold does **not** do is keep a finished run alive: a sequential run that
has reached `--max-runs`, and a parallel one whose queue drained or hit its item
cap, end instead of waiting for a boundary that will never come.

## Talking to the running agent (`m`)

The loop is autonomous, not unattended. `m` opens a one-line editor in the
pinned area: Enter sends and leaves the editor open for the next note, Esc
clears a half-typed line, and Esc on an empty line leaves. (Both are two-step on
purpose — keys arrive one at a time from a single burst, so an editor that
closed on Enter would hand the rest of a pasted line to the normal keys, `s`
included.) While the editor is open the legend shows its keys and not the run's,
because in there `s` is a letter.

The line is properly editable, not append-only: ←/→ move the cursor and typing
lands where it points, Ctrl+←/→ move by word, Home/End (Ctrl+A / Ctrl+E) jump to
either end, Delete cuts forward, and Ctrl+W / Ctrl+U / Ctrl+K erase the word
before the cursor / everything before it / everything after it. The view scrolls
to keep the caret visible when the note outgrows the row. Both spellings of
every key are decoded, so the same bindings work on Windows and POSIX.

What you type reaches the agent one of two ways:

* **into the turn already running**, over its stdin, if one is in flight. It
  lands at the model's next turn boundary (a tool call in progress is not
  interrupted), so advice about the CURRENT task arrives while it can still
  change the outcome. The note may also be answered right after the turn it was
  typed in, as a continuation of the same session — either way it is answered.
* **queued for the next iteration** otherwise (between iterations, or with
  `--no-live-messages`), appended to that prompt as a named section. The legend
  counts what is waiting: `m message (2 queued)`.

Notes are framed before the agent sees them: who is speaking, and that a note is
guidance rather than a replacement for the task. That matters — a bare imperative
arriving mid-turn reads like a prompt injection, and is treated as one. It also
means an emphatic note ("stop reading files, do X instead") is obeyed literally:
say what you mean.

Every note is printed into the scrolling output — and therefore into the mirror
log — at the point where the agent received it, so an iteration that changed
course mid-flight is explainable afterwards. The receipt is the CLI's own replay
of the message, not this end assuming the pipe was read.

With one agent, `m` opens the editor directly. With several workers, `m` first
asks for a job number; Enter selects it and opens an editor labelled with that
job. Each job has its own queue, so a note typed between that worker's turns
rides its next prompt rather than being picked up by whichever worker starts
next.

A note still queued when the run ends is reported, with its text, rather than
dropped in silence: "the next iteration" is a promise the last iteration cannot
keep. A live send still lacking the provider's replay receipt at shutdown is
reported separately as unconfirmed, so an interrupted write cannot disappear.

`m` works with both providers. Claude receives a streaming-input user message;
Codex receives an app-server `turn/steer` request for the active turn. With
`--no-live-messages`, either provider queues the note for the next iteration
instead. It is not a paste target — every newline in a paste sends what
precedes it as its own note.

## Per-user settings

Wrappers can opt into a completion sound through the per-user JSON settings
file. It is disabled when the file or key is absent:

```json
{
  "completion_sound": true
}
```

On Windows the file is `%APPDATA%\llm-loop\settings.json`. On Linux and
macOS it is `${XDG_CONFIG_HOME:-~/.config}/llm-loop/settings.json`. Set
`LLM_LOOP_SETTINGS` to use a different absolute path. Windows wrappers use
the native system notification sound; other platforms emit a terminal bell.

Create a file named `stop` in the project root to halt **every** loop rooted
there at its next iteration boundary; it is removed on stop so the next launch
starts clean. A `--dry-run` never removes it — previewing commands while a real
run has a stop pending must not cancel that stop — it only reports that the file
is there. To stop just one of several concurrent loops, press `s` in its
terminal instead (see above): that request is in-process and leaves no file.

A launch that finds the sentinel already in place does not start and does not
consume it: it waits (sequential and parallel alike, before `--start-in`) until
the file goes away — cleared by the run it was meant for, or removed by hand —
and only then begins. So queueing the next run behind a stop you just requested
works, and a leftover `stop` never costs a run its first iteration.

`pip install rich` enables live Markdown rendering of the assistant's output
(the loop works without it, just plainer).
