"""Pin that reading the wire does not load the renderer.

`tools/codex_agent` (a host tool, not a runner) imports `llm_loop.wire` for the
app-server stream's classifiers and paints nothing. While the front door was
eager, that one import ran `__init__` and with it the whole engine — rich's
Markdown renderer and markdown_it among it — 0.64 s of import measured
2026-10-27, paid by every command of that tool. Two places had to give for it:
`__init__` resolves its names on demand, and `compactline`/`textwidth` (which
`wire` imports for `command_tool`) look rich up on first use.

Each check runs in a fresh interpreter: inside the suite every module is
already loaded by some earlier test, so `sys.modules` here says nothing.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import llm_loop

SRC = Path(__file__).resolve().parents[1] / "src"


def _fresh(code: str):
    """Run `code` in a new interpreter with only this package's src on the
    path, and return what it printed as JSON."""
    env = dict(os.environ, PYTHONPATH=str(SRC))
    done = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_importing_the_wire_loads_neither_rich_nor_the_engine():
    loaded = _fresh(
        "import json, sys, llm_loop.wire\n"
        "print(json.dumps(sorted(m for m in sys.modules\n"
        "    if m.split('.')[0] in ('llm_loop', 'rich', 'markdown_it'))))")

    assert loaded == ["llm_loop", "llm_loop.compactline", "llm_loop.textwidth",
                      "llm_loop.wire"]


def test_the_lazy_rich_lookups_still_find_rich_on_first_use():
    """The other half of deferring them: the first call must land on rich's
    escaper and measure, not on the no-rich fallbacks."""
    seen = _fresh(
        "import json, sys\n"
        "from llm_loop import compactline, textwidth\n"
        "print(json.dumps([compactline.esc('[b]x'),\n"
        "                  textwidth.cell_width('\\U0001F4BB'),\n"
        "                  'rich.markup' in sys.modules]))")

    assert seen == ["\\[b]x", 2, True]


def test_a_front_door_name_resolves_to_its_home_module_s_object():
    from llm_loop import agentwork, parallel

    assert llm_loop.Driver is agentwork.Driver
    assert llm_loop.parse_parallel_args is parallel.parse_args


def test_any_submodule_is_still_an_attribute_of_the_package():
    """`llm_loop.cyclecore` after a bare `import llm_loop` resolved while the
    front door was eager; host code written then must keep working."""
    assert _fresh("import json, llm_loop\n"
                  "print(json.dumps(llm_loop.cyclecore.__name__))"
                  ) == "llm_loop.cyclecore"


def test_an_unknown_name_is_an_attribute_error():
    assert not hasattr(llm_loop, "no_such_name")
