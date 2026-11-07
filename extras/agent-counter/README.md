# agent-counter

An optional on-screen badge that shows how many background agents (Claude,
Codex) are running right now, with the provider's weekly-limit remainder when
the provider reports one. The client half is `src/llm_agents/agent_counter.py`:
each running agent holds one named-pipe connection for the length of its turn,
and the badge counts the connections.

- **Windows only.** It speaks a Windows named pipe; `main.cpp` has never been
  built or run on another OS.
- **Optional.** The bridges work without it. When the executable is missing the
  client shows nothing, never fails a turn, and says so once on stderr.
- **Not part of the pip package.** Qt is not a dependency of `llm-loop`.

Build with CMake and Qt 6.5 or newer (Core, Gui, Network, Widgets):

```
cmake -S extras/agent-counter -B build-agent-counter -DCMAKE_PREFIX_PATH=<Qt>/msvc2022_64
cmake --build build-agent-counter --config Release
```

The exe needs the Qt DLLs at run time: either run `windeployqt AgentCounter.exe`
next to it, or put Qt's `bin` directory on `PATH`.

Point the client at the result with `AGENT_COUNTER_EXE=<path>\AgentCounter.exe`,
or name it in a host policy (`HostPolicy.counter_exes`). `AgentCounter.exe
--selftest` runs the registry checks without a window; `--status` prints the
current count. Set `AGENT_COUNTER_ENABLED=0` to keep the badge off the screen.

**Icons.** No provider logo ships here: by default the badge paints neutral
marks of its own (a disc with a letter). To show real icons, configure with
`-DAGENT_COUNTER_ICON_DIR=<dir>` naming a directory with `claude.png` and
`codex.png`; a missing or non-PNG file is a configure warning and the neutral
marks stay.
