# claude-loop

A reusable engine for **autonomous Claude-CLI loops**: it repeatedly invokes the
`claude` CLI to grind through a unit of work, handling all the scaffolding —
command-line parsing, a rotating mirror log, a git-push policy, the
token-usage / session-window limit machinery, live stream-json rendering, and a
graceful stop file — so a host project only has to say *what work to do each
iteration*.

It ships two ready-made task shapes (and you can write your own `Driver`) = **State machine** and **Work queue**.

- **State machine** (`StateFileDriver`) — read the first line of a state file
  each iteration and run a fixed prompt against it; stop on an `error` state.
  Good for "follow this playbook until done" loops where progress lives in files.
- **Work queue** (`ListFileDriver`) — process the files listed in a list file
  one at a time (or N at a time, see `run_parallel`), striking each out of the
  list once its command succeeds. Idempotent: stop any time and relaunch.

The state machine is the headline shape. In essence the loop is just:

```
while (state != error)
    claude -p "Follow instructions in currentState.md"
	state = readStateFrom("currentState.md")
```

Its state file's first line names the current mode; each iteration runs that
mode, then rewrites the line to point at the next one. The
[`examples/currentState.md`](examples/currentState.md) playbook cycles like this:

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

Designed to be vendored as a **git submodule** under a host project. The code
location and the project root are kept separate: the engine anchors every
project-relative operation (git/claude cwd, the stop file, the relative paths a
Driver is handed) to the project root — the current working directory by
default, or `--project-dir`/`-C`.

## Layout

```
claude_loop/
  cyclecore.py   engine: parse_args, run_loop, the Driver protocol,
                 git-push policy, mirror log, stream-json rendering
  usage.py       UsageSource: query / cache / parse `claude -p "/usage"`
  limits.py      LimitPolicy + SessionLimit / DayNightLimit / WeeklyLimit rules
  drivers.py     StateFileDriver (state machine) and ListFileDriver (work queue)
  parallel.py    run_parallel: N concurrent claude workers over a list file
examples/
  runCycle.py            state-machine wrapper
  runFileList.py         per-file work-queue wrapper
  runFileListParallel.py parallel work-queue wrapper
```

## Use it from a host project

Add it as a submodule, then copy one of the [`examples/`](examples/) wrappers
into your project root and adjust the paths, prompt and model:

```bash
git submodule add <repo-url> tools/claude-loop
cp tools/claude-loop/examples/runFileList.py .   # then edit the constants
```

A wrapper is tiny — subclass a Driver, set the project-specific bits as class
attributes / an overridden `prompt()`, and call `.main()`:

```python
# runFileList.py  (in your project root)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools", "claude-loop"))

from claude_loop import ListFileDriver

class FileListDriver(ListFileDriver):
    list_file     = "files.md"
    target_suffix = ".summary.md"

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
python runFileList.py --max-runs 5  # at most 5 iterations
python runFileList.py --dry-run     # print the commands, run nothing
```

### State-machine driver

Drives the state machine shown in the diagram above — the state file's first
line names the current mode, and an overridden `model()` picks the model per mode
(drop it to let the CLI use its own configured model):

```python
from claude_loop import StateFileDriver

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

## Usage limits

Before each iteration (and after any failed one) the loop consults
`claude -p "/usage"` and pauses if a watched quota is at/over its ceiling. Which
quota, and at what ceiling, is a *specialisation* you pick by setting the
`limit_policy` class attribute on your Driver — a `LimitPolicy` holding one or
more rules; the loop pauses while **any** of them is exceeded:

```python
from claude_loop import (LimitPolicy, SessionLimit, DayNightLimit, WeeklyLimit)

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

Leaving `limit_policy` unset uses `LimitPolicy([DayNightLimit()])`. The bookend
usage snapshots and the per-check status lines report exactly the quotas the
active policy watches. When `--max-runs N` is given (a short bounded run) the
limit gate is skipped entirely.

## Common options

| Option | Meaning |
|---|---|
| `-m, --max-runs N` | stop after N iterations (sequential) / N files total (parallel); `--max` is a deprecated alias |
| `-d, --dry-run` | print the commands, run nothing |
| `-g, --git-push none\|after_new_commits\|each_hour` | when to `git push` |
| `-C, --project-dir DIR` | project root (default: cwd) |
| `-s, --start-in 29m` | wait before starting (sequential only); alias `--startIn` |
| `-S, --max-strike 3h` | per-session work budget before a pre-emptive pause; alias `--maxStrike` |
| `-j, --jobs N` | concurrent workers (parallel only) |
| `--raw` | print raw JSON events, for debugging (sequential only) |

Create a file named `stop` in the project root to halt the loop at the next
iteration boundary; it is removed on stop so the next launch starts clean.

`pip install rich` enables live Markdown rendering of the assistant's output
(the loop works without it, just plainer).
