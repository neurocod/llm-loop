# ideas

Working material kept for mining, not part of the library.

Nothing here is imported by `llm_loop`, and nothing here is shipped: the
package lives in `src/`, which is the only directory a host project puts on
`sys.path`, so these files are invisible to anyone who vendors or installs the
engine. They stay in the repository because reading a complete working
implementation beats rediscovering its decisions later.

## continuous_claude.py

A Python port of [`continuous_claude.sh`](https://github.com/AnandChowdhary/continuous-claude)
v0.24.7 by Anand Chowdhary, MIT-licensed — see the notice at the top of the
file. The original is a Bash script; the port is a rewrite in another language,
not a transcription, and it deviates deliberately: no external `jq`, no
self-update, and the rate/error/cost sliding windows live in memory instead of
temp files.

It is here for its *shape*, which `llm_loop` does not currently have: the
commit / pull-request / wait-for-checks / merge cycle, and the prompts that
drive it.

`tests/test_continuous_claude.py` covers it, so the file keeps working rather
than quietly rotting into a snapshot that no longer runs.
