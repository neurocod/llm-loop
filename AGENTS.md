# Notes for agents working on llm-loop

## Checking CI after a push

A push to `main` runs one workflow, `tests` (`.github/workflows/tests.yml`):
a 15-job matrix (3 OS x 5 Python versions) with `fail-fast: false`, so a red
run still waits for every job and does not finish sooner than a green one.

- **5 min after the push:** an early look. A failed job is usually visible by
  then, while the rest are still running.
- **20 min after the push:** the full verdict. A run that is still going is
  most likely held up on a runner, so re-check every 4-5 min.

Measured 2026-11-14 over 139 push runs (2026-08-17 .. 2026-11-12):
push -> conclusion median 13.5 min, p90 18.4, max 135.5 (one stalled run);
on the current 15-job matrix (15 runs) median 10.6, p90 15.4, max 17.9.
Queue before the first job starts: median 0 s, max 1.8 min. First failed job
visible: median 2.8 min, p90 3.7, max 5.0 (29 failed runs).

Commands:

```
gh run list --branch main --limit 3
python agent-tools/ci_failures.py            # failed jobs + pytest failures of HEAD's runs
python agent-tools/ci_failures.py --wait     # same, after the runs finish
```

A long wait goes in chunks of at most ~4.5 min, so the waiting session wakes
at least that often: `timeout 270 gh run watch <id> --exit-status -i 60`.
Exit 124 means it is still running, so re-arm the same command. The `-i 60`
poll interval is separate and does not wake the session.
