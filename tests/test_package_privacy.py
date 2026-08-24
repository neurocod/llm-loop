"""Pin what a leading underscore means inside this package: module-private.

It used to mean nothing checkable. Twelve underscored symbols owned by `console`
and `usage` were read from six other modules through 21 code sites — `_fmt_left`
by three importers, `_TeeToLog` by both runners — and the newest of those sites
had been added weeks after the rest without anybody reading it as a signal.
A marker every reader has to look past is worse than no marker: the next
extraction cannot tell "nothing outside this file may touch it" from "everything
in the package already does".

So the twelve were given public names and this gate keeps the meaning true.
`_` now says the one thing it can be trusted to say: *this file's business*.
The package's outward surface is a different question, answered by
`__init__.__all__` and pinned by test_front_door.py — a name being public HERE
does not put it on the front door, which is why making these public costs the
adopter nothing.

Deliberately AST, not import: the check is about what the SOURCE says, so it
sees a violation even in a branch no test executes, and it needs no module of
the package to import cleanly to run.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "llm_loop"
MODULES = {path.stem for path in PACKAGE.glob("*.py")}


def _is_private(name: str) -> bool:
    """`_x` yes, `__x__` no: dunders are the language's, not ours."""
    return name.startswith("_") and not (name.startswith("__")
                                         and name.endswith("__"))


def _sibling_aliases(tree: ast.AST) -> dict:
    """Local name -> sibling module, for every `from . import x [as y]`."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
            for alias in node.names:
                if alias.name in MODULES:
                    aliases[alias.asname or alias.name] = alias.name
    return aliases


def _reaches_into_a_sibling(path: Path):
    """Every place this file reads an underscored name out of another module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _sibling_aliases(tree)
    found = []
    for node in ast.walk(tree):
        # `from .console import _fmt_left`
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in MODULES:
            found += [(node.lineno, f"from .{node.module} import {alias.name}")
                      for alias in node.names if _is_private(alias.name)]
        # `console._fmt_left`
        elif (isinstance(node, ast.Attribute) and _is_private(node.attr)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            found.append((node.lineno, f"{node.value.id}.{node.attr}"))
    return found


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_no_module_reads_another_module_s_private_name(path):
    reaches = _reaches_into_a_sibling(path)

    assert reaches == [], (
        f"{path.name} reads underscored names out of sibling modules: "
        + ", ".join(f"line {line}: {what}" for line, what in reaches)
        + ". Either the name is this package's vocabulary — then drop the "
          "underscore where it is defined — or the caller wants something the "
          "owner has not offered yet.")


def test_the_gate_can_see_both_spellings(tmp_path):
    """The gate is only as good as the two forms it recognises.

    Both reach a sibling's private, and code here uses both: `limits` imports
    names it calls on nearly every path, while `termio` goes through the module
    so a test can replace the target. A gate blind to either would call the
    package clean while half of the 21 sites were still there — and a clean
    package is exactly what the check above reports either way, so blindness in
    it is invisible unless something feeds it a file that IS dirty.

    Which is why this drives `_reaches_into_a_sibling` itself on a written file
    rather than repeating its walk over a parsed string. Repeating it was the
    first version of this test, and deleting either branch of the real function
    left all 23 cases green: it was pinning a copy.
    """
    sample = tmp_path / "sample.py"
    sample.write_text("from . import console\n"
                      "from .console import _fmt_left\n"
                      "from .console import print_done\n"   # public import: fine
                      "console._real_stream()\n"
                      "console.print_done()\n"              # public attr: fine
                      "self._own_field\n",                  # own private: fine
                      encoding="utf-8")

    found = [what for _line, what in _reaches_into_a_sibling(sample)]

    assert sorted(found) == ["console._real_stream",
                             "from .console import _fmt_left"]
