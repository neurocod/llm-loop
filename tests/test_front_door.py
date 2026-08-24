"""Pin that the package's front door is whole: every name `__all__` promises
resolves.

`__init__.__all__` is the vendored engine's public surface — a host wrapper
spells `from llm_loop import Driver, LoopStop, run_parallel` and never names an
internal module. That list is hand-kept beside the imports above it, so the two
can drift in either direction, and both drifts are silent:

  * a name listed but never imported raises AttributeError only when someone
    writes `from llm_loop import *` — which no test and no wrapper here does, so
    nothing fails until an adopter tries it;
  * a symbol MOVED between internal modules keeps the list true only while the
    import line above follows it. The moves this file was written for
    (`projectroot`, `console`, `gitpush`, `stopchannel`, and the three that
    emptied `cyclecore` of what it never owned) each rewrote those import lines,
    and the surface is exactly what such a move must not change.

So the assertion is deliberately about the LIST, not about a sample of names a
test happened to think of: a name added to `__all__` is covered the moment it is
added, and one whose home moved is checked at its new address for free.
"""
import llm_loop


def test_every_promised_name_resolves():
    missing = [name for name in llm_loop.__all__
               if not hasattr(llm_loop, name)]
    assert missing == [], (
        f"__all__ promises names the package does not supply: {missing}")


def test_the_promise_has_no_duplicates():
    """A name listed twice is a merge that kept both halves of a move."""
    seen = sorted(llm_loop.__all__)
    dupes = sorted({n for n in seen if seen.count(n) > 1})
    assert dupes == []


def test_star_import_is_what_the_list_says():
    """`from llm_loop import *` is the one spelling that reads `__all__` itself,
    so it is also the one that fails on a broken entry — here rather than in an
    adopter's project."""
    namespace = {}
    exec("from llm_loop import *", namespace)      # noqa: S102 - that is the test
    exported = {k for k in namespace if not k.startswith("__")}
    assert exported == set(llm_loop.__all__)
