# ideas

Working material kept for mining, not part of the library.

Nothing here is imported by `llm_loop`, and nothing here is shipped: the
package lives in `src/`, which is the only directory a host project puts on
`sys.path`, so these files are invisible to anyone who vendors or installs the
engine. They stay in the repository because reading a complete working
implementation beats rediscovering its decisions later.

## continuous_claude.py

The upstream project,
[`continuous_claude.sh`](https://github.com/AnandChowdhary/continuous-claude)
v0.24.7 by Anand Chowdhary, is written in **Bash**. This file is its **port to
Python** — a rewrite in another language rather than a transcription, so it
deviates deliberately: no external `jq` (native JSON parsing instead), no
self-update, and the rate/error/cost sliding windows live in memory instead of
temp files.

The upstream is MIT-licensed and this port is a derivative work, so it carries
the upstream notice at the top of the file.

It is here for its *shape*, which `llm_loop` does not currently have: the
commit / pull-request / wait-for-checks / merge cycle, and the prompts that
drive it.

`tests/test_continuous_claude.py` covers it, so the file keeps working rather
than quietly rotting into a snapshot that no longer runs.
