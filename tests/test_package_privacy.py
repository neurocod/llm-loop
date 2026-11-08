"""Pin two kinds of privacy: what a leading underscore means inside this
package (module-private), and that the public tree keeps its host project's
names to itself (the last test).

The underscore used to mean nothing checkable. Twelve underscored symbols owned by `console`
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
import os
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


# --- the other privacy: the host project's, kept out of the public tree --------

REPO = PACKAGE.parents[1]

# Names of the project this engine was extracted from, and spellings that
# belong to its machines. Each was found in a comment, a docstring or a test
# fixture AFTER the tree went public (2026-11-08: a wrapper's file name in six
# modules and four tests, a drive path in a docstring, a backlog number in a
# test): a reader of this repository has none of them, so a sentence built on
# one explains nothing to them. Case-insensitive, matched as plain text.
HOST_PROJECT_NAMES = ("runGenerateModels", "kit-promotion", "webgame",
                      "MultiSkill", "C:\\g\\")

# What a host name could be written in: the tracked text kinds. Binaries and
# the generated trees are skipped by not being named.
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt",
                 ".cpp", ".h", ".hpp", ".inc", ".cmd", ".ps1", ".sh", ".cfg",
                 ".ini", ".gitignore", ".typed"}
SKIPPED_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules",
                "build", "dist", ".venv", "venv"}


def _text_files(root: Path):
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames
                             if name not in SKIPPED_DIRS
                             and not name.endswith(".egg-info"))
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix in TEXT_SUFFIXES or filename == "LICENSE":
                yield path


def test_no_tracked_text_names_the_host_project():
    """The sweep `git grep -i -E 'runGenerateModels|…'` as a test, so the next
    comment that reaches for a host name fails here rather than in a reader's
    eyes. This file is the one place the names are allowed to appear."""
    this = Path(__file__).resolve()
    lowered = tuple(name.lower() for name in HOST_PROJECT_NAMES)
    hits = []
    for path in _text_files(REPO):
        if path == this:
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            low = line.lower()
            hits += [f"{path.relative_to(REPO).as_posix()}:{number}: {name}"
                     for name, needle in zip(HOST_PROJECT_NAMES, lowered)
                     if needle in low]

    assert hits == [], (
        "the public tree names its host project: " + ", ".join(hits)
        + ". Say the thing itself (a batching wrapper, a drive path such as "
          "C:\\work) instead of the host's name for it.")
