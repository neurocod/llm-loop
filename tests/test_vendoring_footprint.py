"""Pin how much of a host project's namespace vendoring this engine costs.

Most adopters do not pip-install the engine: they add it as a git submodule and
put its `src/` on sys.path. Whatever sits directly inside the inserted
directory becomes a top-level importable name for them, and the insert goes to
position 0, so those names outrank the host's own modules of the same name.

Before the src layout the inserted directory was the repository root, which
handed an adopter four names - `llm_loop`, `tests`, `examples` and
`continuous_claude`. The `tests` collision is the one that hurt: nearly every
project has that package, ours won, and the host's suite would import modules
it never wrote. The failure is silent and lands far from its cause.

So the invariant is not "the package imports" but "src/ contains the package
and nothing else". That is a property of the directory, which is why this test
reads the filesystem instead of importing: under pytest `ideas/` is on the path
too (test_continuous_claude.py needs it), so an import-based check would be
measuring the test environment rather than what an adopter gets.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _top_level_names():
    """Every name `import X` could resolve to via the inserted directory.

    Filtered by `isidentifier`, which is the actual rule the import system
    applies, not a denylist of things seen so far: an editable install drops
    `llm_loop.egg-info` in here and `__pycache__` appears on any run, and
    neither is reachable by an import statement. A denylist would have to grow
    every time packaging invents another sidecar; this cannot fall behind.
    """
    names = []
    for p in SRC.iterdir():
        if p.is_dir() and p.name.isidentifier():
            names.append(p.name)            # package
        elif p.suffix == ".py" and p.stem.isidentifier():
            names.append(p.stem)            # module
    return sorted(names)


def test_src_supplies_only_the_package():
    assert _top_level_names() == ["llm_loop"]


def test_the_package_is_importable_from_there():
    assert (SRC / "llm_loop" / "__init__.py").is_file()


def test_the_type_marker_ships():
    """PEP 561: without it a consumer's type checker ignores every annotation."""
    assert (SRC / "llm_loop" / "py.typed").is_file()
