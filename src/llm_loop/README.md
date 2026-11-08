# `llm_loop` — the engine

The package behind the loop: a headless LLM CLI (Claude Code or Codex) run
over and over with a fresh context window each time, the progress kept in
files and git, the account's quota read before every iteration. The
repository [README](../../README.md) is the overview; this file is the
loop's own manual.

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
| Implementation | a bash one-liner; forks exist in bash/PowerShell | a Python package (3.9+, **no required third-party package**), one code path on Windows/Linux/macOS |
| Iteration body | the same prompt file, every time | a pluggable `Driver`: state machine, work queue, or your own |
| Multi-step work | one undifferentiated "do the next thing" step | explicit phases; **the state is read back by the script**, not only by the agent |
| Prompt | fixed file | `prompt()` is computed per iteration from the current state / work item |
| Model choice | one model for the whole run | `model()` per iteration — a cheap model for `cleanup`, a smart one for `implementation` |
| Quota / rate limits | run until the CLI dies | the account's real figures are read over the provider's API **before** each iteration, and the loop waits under configurable ceilings (session / day-night / weekly), with a reactive `rate_limit_event` backstop |
| Parallelism | sequential (fan-out only *inside* one agent) | `-j N` concurrent CLI workers draining a work-queue file, one item per job |
| Stopping | Ctrl+C | `s` key (this run only), `stop` sentinel file (every run in the root), `--max-runs`, an `error` state that halts for a human, and a `done` / `complete` state that ends the run cleanly |
| Steering | stop it, edit the prompt, start again | `m` sends a note into the current iteration at the model's next turn boundary (or queues it for the next iteration when none is running); `p` holds the loop at an iteration boundary so the files it reads can be edited, `p` again lets it go |
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
[`examples/currentState.md`](../../examples/currentState.md), the file this project's
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
[job 3] 💻 pdfinfo 'D:\work\research\musical-instruments\acoustic-upright-piano\kawai-k-series-…
[job 3] 📤 exit 0
[job 4] 💻 curl.exe -L --fail --output 'D:\work\research\food-and-beverages\eggs-dozen\thirty-…

──────────────────────────────
 ⟳ iter 4/12 | codex | 4 jobs | 1m37s | session n/a | week 26% (6d8h) / ceil 95%
 job 1 ▶ gpt-5.6-terra | iter 1    | 1m37s  | calculator-desktop.md
 job 2 ▶ gpt-5.6-terra | iter 1    | 1m36s  | terminal-block-connector.md
 job 3 ▶ gpt-5.6-terra | iter 1    | 1m36s  | acoustic-upright-piano.md
 job 4 ▶ gpt-5.6-terra | iter 1    | 1m36s  | eggs-dozen.md
 keys: s stop | p pause | m message | +/- add/remove workers | h help
```

## A playbook worth stealing

The state machine is the headline shape, and a playbook for it is where the
project-specific thinking goes. The shipped
[`examples/currentState.md`](../../examples/currentState.md) cycles like this:

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

**Python 3.9+ and no required third-party package.** Every import across the
engine's modules comes from the standard library, except `rich`: installed, it
renders the assistant's Markdown live; absent, the output is plainer and
nothing fails. So vendoring the engine adds no transitive dependency to your
project and cannot conflict with the versions you already pin.

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
    scriptlock.py    ScriptLock / ensure_script_lock
    agentwork.py     what one unit of work is, and the Driver protocol a
                     wrapper subclasses to produce them
    console.py       what a run prints, and the rotating mirror log that is the
                     second copy of every line of it
    costlog.py       --cost: the two logged lines a run's spend is read back from
    stopchannel.py   how a run is asked to stop or hold, and what it reports
    breakpoints.py   run-local state breakpoint names and matching
    projectroot.py   where the project being driven is, and the one place that
                     answers it
    gitpush.py       when a run pushes what it has committed, and where
    providers.py     Claude/Codex executable flags, argv construction, and the
                     ending owed to a provider child
    procend.py       ending a child process tree: ask, wait, kill, wait
    appserver.py     JSON-RPC over a `codex app-server` child's stdio, for
                     the provider process, `codex_usage` and `llm_agents.codex`
    drivers.py       StateFileDriver (state machine) and ListFileDriver (work queue)
    parallel.py      run_parallel: N concurrent LLM workers over a list file
    ownership.py     one thread owns a shared resource, the rest post to it
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
    modeswitch.py    a wrapper's mode switches, declared once for both parsers
    ctrlc.py         the operator's Ctrl+C as an event a run polls, not an exception
    notifications.py opt-in settings and completion sounds for unattended runs
    exitlog.py       why a run ended — including the endings it cannot report itself
    diaglog.py       the status line's quiet failures, written where somebody can read them
    statlog.py       --stat: state timings read back from the mirror records
examples/
  runCycle.py            state-machine wrapper
  runFileList.py         per-file work-queue wrapper
  runFileListParallel.py parallel work-queue wrapper
scripts/
  fake_claude/           a stand-in `claude` and a runner against it, for the
                         status line's failure branches without a paid turn
  collect_codex_thread.py one Codex rollout JSONL as a chronological view
```

The rest of the repository (the agent bridges, the git kit, the shell gate,
`agent-tools/`, `extras/`) is mapped in the top-level [README](../../README.md).

## Use it from a host project

Add it as a submodule, then copy one of the [`examples/`](../../examples/) wrappers
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
        # Codex: "" lets the CLI use its own configured model; a Claude run
        # names one. The same test as in examples/runFileList.py.
        return "" if self.provider == "codex" else "sonnet"

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
`--codex`, or set `provider = "codex"` on the Driver subclass. Codex normally runs
through its app-server protocol, so `m` can steer the active turn; the same sequential and
parallel renderers show agent messages, commands, file changes, failures, and
final token counts. `--no-live-messages` falls back to `codex exec --json` and
sends the prompt through a closed stdin stream (`codex exec ... -`) instead of
argv. Both transports avoid Windows command-line length limits without entering
the interactive UI. Claude keeps its existing provider-specific transport.
The adapter grants writes only to the workspace and routes any approval through
Codex's automatic reviewer; it does not bypass the sandbox.

### State-machine driver

Drives the state machine shown in the diagram above — the state file's first
line names the current mode, `prompt()` supplies the required project playbook,
and an overridden `model()` picks the model per mode (drop `model()` to let the
CLI use its own configured model):

```python
from llm_loop import StateFileDriver

class CycleDriver(StateFileDriver):
    state_file = "currentState.md"

    def prompt(self):
        return f"Follow the instructions in {self.state_file}"

    def model(self):
        if self.state_name() == "implementation":
            return "claude/opus"
        return "codex"

if __name__ == "__main__":
    CycleDriver.main()
```

Two states end the run. `error` (anywhere in the line) halts with exit code 1 for
a human. A state that **begins with** `done` or `complete` — `Current state: done`,
`Current state: done — nothing before Phase 5 is left` — is a clean stop with exit
code 0 and the final push, for a playbook that has a finish line. The word has to
be the state itself: `review complete` or `done-list` are ordinary steps and keep
the loop going. Both words are the `done_tokens` class attribute if your playbook
wants a different one.

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

A wrapper that offers a mode of its own has to scan `argv` for it — the two
parsers have disjoint option sets, so the flag choosing between them decides
which parser runs. Declare such flags once, as a
[`modeswitch`](modeswitch.py) table: `register` (called from each
driver's `add_cli_options`) puts the switches into that driver's parser, and
`modeswitch.parse` is the dispatch — the scan, your choice of driver and parser,
the parse, and exit 2 when an abbreviation made the scan and the parser read a
switch differently. `Driver.main()` never acts on a switch, so it refuses a
line that gives one:

```python
SWITCHES = (ModeSwitch(("-p", "--parallel"), "parallel", False,
                       "run N workers", listed_in=frozenset({"sequential"})),)

class FileListDriver(ListFileDriver):
    @classmethod
    def add_cli_options(cls, parser):
        modeswitch.register(parser, SWITCHES, "sequential")

class FileListParallelDriver(FileListDriver):
    @classmethod
    def add_cli_options(cls, parser):      # parsed, not listed, in its --help
        modeswitch.register(parser, SWITCHES, "parallel")

def choose(scanned):
    if scanned.parallel:
        return modeswitch.Choice(FileListParallelDriver, parse_parallel_args)
    return modeswitch.Choice(FileListDriver, parse_args)

def main(argv=None):
    choice, scanned, args = modeswitch.parse(argv, SWITCHES, choose)
    run = run_parallel if scanned.parallel else run_loop
    run(choice.driver(), args, app_name=choice.driver.resolved_app_name())
```

The switches stay in `argv`: the chosen parser reads the whole line, so a value
flag followed by a switch is a usage error instead of the switch's neighbour
quietly becoming the value.

Hold the hook to [`clispec.unstrippable_flags`](clispec.py) in a
test of your own, over every parser your entry points build. `register` tells
the status line's argv rewriter about your switches; a value-taking option of
your own that is not a switch needs the same, through `clispec.record_flags`:

```python
parser.add_argument("--size", type=int)
clispec.record_flags(parser, {"size": clispec.Flag(("--size",), True)})
```

### Reacting to the world around a run

`Driver.item_started(command)` and `Driver.item_finished(command, returncode)`
are called at the two boundaries of one unit of work, in both runners (the
parallel one calls them from its worker threads, so keep them thread-safe).
Override the one you need to watch something the run does not own — a folder the
agents themselves write into, a lock, a budget.

Return a short reason from either and the runner stops handing out new work and
returns `RunStopReason.DRIVER_PAUSE` once everything in flight has finished: no
turn is cancelled and nothing is lost from the queue. It is the same close
`--max-runs` uses (new claims refused, work in flight untouched) rather than the
`s` key's — a pause cannot be taken back, so unlike `s` a claim that has not
started yet goes back to the queue instead of being held. A run whose LAST item
asks for the pause may report either `DRIVER_PAUSE` or `NO_WORK`, depending on
which worker got there first; a caller has to treat both as "come back and ask
again". It is the ending for a wrapper that alternates two kinds of run — "this runner call is over, ask me again" — as opposed to
`next_command() -> None` (the queue is empty) and `LoopStop` (abort, a human is
wanted). A host wrapper that slices its work into fixed-size batches to get an
inter-batch boundary usually wants this instead: the run then ends when the
condition actually appears rather than N items later.

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

## Command-line options

The option list is your wrapper's `--help`. Both runners build their parsers
from one table, [`llm_loop.clispec.OPTIONS`](clispec.py), and the two accept different
subsets of it (`clispec.OPTION_ORDER`), so ask the entry point you run:

```
python runFileList.py --help           # .main(): the sequential runner
python runFileListParallel.py --help   # .main_parallel(): the parallel one
python runCycle.py --stat              # time spent per state in the mirror log
```

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

## Ctrl+C

Inside a runner Ctrl+C is an event the run polls, not a `KeyboardInterrupt`:
the first press ends the turn in flight and every wait, and the run leaves with
exit code 130 after its usual ending — exit push, usage snapshot, the report of
undelivered notes; a further press gives up what that ending still waits for.
Which public calls return early for an embedder, and how a wrapper hears the
press once the runner has left, is `llm_loop.ctrlc` (the module docstring and
`ctrlc.captured`).

## Interactive status line

On a terminal a run pins a few rows at the bottom — iteration, provider/model,
elapsed time, the provider's live quota figures and the policy's ceilings, one
row per job, and a legend of the keys — and names the window after the row's
first field (`statusline.title_text`). Piped output, CI and `--no-statusline`
get the plain scrolling output of before, and no title.

The iteration counter's denominator is `--max-runs`, `Driver.pending_total()`,
or the smaller of the two. A quota field has two halves split by ` / `: the
account's own report (how much of the window is spent, how long it has to run)
and what the `LimitPolicy` rule for that window makes of it — every window the
provider meters is shown, the policy half only where a rule watches
(`statusline.quota_rows`, `LimitRule.status` for a custom rule):

```
 session 43% (2h11m) / ceil 95% | week 63% (17h27m)
 └── the provider ──┘└─ policy ─┘└─ the provider ─┘
```

The keys (`h` or `?` lists them), each with the symbol that holds its rule:

| Key | Does | Rule |
|---|---|---|
| `s` | stops **this** run; `s` again during the 5 s countdown cancels | `statusline.StopAction`, `stopchannel.confirm_stop_request` |
| `p` | holds the run at the next iteration boundary; `p` again lets it go | `statusline.PauseAction`, `stopchannel.wait_while_paused` |
| `m` | a note to the running agent (in a fleet, to one job of it) | `statusline.MessageAction`, `llm_loop.operator` |
| `+` / `-` | one more / one fewer worker in a parallel run, for the rest of the invocation | `parallel.ResizeWorkerPoolAction` |
| `w` | edits the weekly ceiling of the live policy | `statusline.WeeklyLimitAction` |
| `b` | a state breakpoint, in state-driven loops | `statusline.BreakpointAction` |

## Stopping: the `s` key and the `stop` file

Two channels, not interchangeable (`stopchannel.StopSource`). `s` is
in-process: it stops the run whose terminal it was typed into and writes
nothing to disk. A file named `stop` in the project root halts **every** run
rooted there at its next iteration boundary, survives the process, and is
therefore what scripts and run-chaining use: the run that obeys it removes it
on exit, and a launch that finds one already there waits until it is gone
(`stopchannel.wait_for_stop_file_clear`; `--dry-run` only reports it). When
both are up the file wins — `stopchannel.latched_stop` says why.

## Holding the run (`p`)

The iteration in flight finishes and the next does not begin, so the files the
loop reads are yours to edit until `p` again. Read the row, not the keypress:
`PAUSING` means an iteration is still running, `PAUSED` that the run stands
still (`statusline.pause_state`). A held run still answers `s`, the stop file
and `m`; one with nothing left to do ends instead of holding.

## Talking to the running agent (`m`)

`m` opens a one-line editor in the pinned area (with several workers, a job
number first): Enter sends and stays in the editor, Esc clears the line, Esc on
an empty line leaves; the cursor and erase keys are `statusline.LineEditor.ACTIONS`.
A note reaches the agent mid-turn while a turn is running (Claude over
streaming input, Codex over `turn/steer`) and rides the next prompt otherwise,
`--no-live-messages` included. Either way it is framed as a human's guidance
rather than a replacement for the task — an emphatic note is obeyed literally,
so say what you mean — and printed into the scrolling output, hence the mirror
log, where the agent received it. Notes the run ends still holding are
reported, not dropped (`operator.report_undelivered_notes`). Not a paste
target: every newline sends.

### When the status line freezes

Its quiet failure branches (a key reader that died, a frame that raised, keys
dropped behind a stuck write) are written to `<app>-<project>.diag.log` beside
the mirror log; the run's banner prints the path, and `llm_loop.diaglog` has
the `LLM_LOOP_KEYTRACE=1` trace. To reproduce without a paid turn,
`python scripts/fake_claude/fake_cycle.py` runs a real sequential runner
against a fake `claude`, and `py-spy dump --pid <pid> --native` shows where
each thread is.

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

`pip install rich` enables live Markdown rendering of the assistant's output
(the loop works without it, just plainer).
