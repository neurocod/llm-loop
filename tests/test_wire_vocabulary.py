"""Pin that the provider stream's vocabulary is written in exactly one module.

Two renderers parse the same stream — `streamrender` for the one run a
sequential loop has in flight, `parallel.run_job` for N at once — and before
`wire` existed they spelled it twice: 32 of the stream's literals appeared in
`cyclecore` and in `parallel` both (`agent_message`, `total_cost_usd`,
`cached_input_tokens`, `item.completed`, …). Nothing said so. A field added to
one renderer simply never reached the other, and the second went on printing
what it had always printed — the failure mode this gate exists to make loud.

Deliberately AST over the SOURCE, like `test_package_privacy`: a literal in a
branch no test executes is still a second spelling, and this needs no module to
import cleanly to run.

The guarded set is not typed out here. It is `wire.VOCABULARY`, which `wire`
derives from its own upper-case constants — so a word added there joins this
gate by existing, and a gate that restated the list would go green on exactly
the addition it is for.
"""

import ast
from pathlib import Path

import pytest

from llm_loop import wire

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "llm_loop"
HOME = "wire"


def _string_literals(path: Path):
    """Every string constant in a module, with the line it sits on."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def _spellings_outside_wire(path: Path):
    """The wire words this module spells itself, minus the ones it may."""
    allowed = wire.SPELLED_ELSEWHERE.get(path.stem, frozenset())
    return [(line, value) for line, value in _string_literals(path)
            if value in wire.VOCABULARY and value not in allowed]


@pytest.mark.parametrize("path",
                         sorted(p for p in PACKAGE.glob("*.py")
                                if p.stem != HOME),
                         ids=lambda p: p.name)
def test_no_module_spells_the_wire_vocabulary_itself(path):
    spelled = _spellings_outside_wire(path)

    assert spelled == [], (
        f"{path.name} spells the provider stream's vocabulary itself: "
        + ", ".join(f"line {line}: {value!r}" for line, value in spelled)
        + f". Every word of that stream lives in {HOME}.py — take the constant "
          "from there, or read the field through the accessor that already "
          "knows its key. A word genuinely belonging to another protocol goes "
          f"in {HOME}.SPELLED_ELSEWHERE, with the reason.")


def test_the_gate_can_see_a_second_spelling(tmp_path):
    """The gate is only as good as what it would report on a DIRTY file.

    The package is clean, so the check above passes whether it looks at anything
    or not — the same blindness `test_package_privacy` was caught by. This feeds
    the real function a file that is dirty, and drives that function rather than
    repeating its walk.
    """
    sample = tmp_path / "sample.py"
    sample.write_text('et = ev.get("type")\n'
                      'if et == "turn.completed":\n'
                      '    cost = ev.get("total_cost_usd")\n'
                      'name = "some ordinary string"\n',
                      encoding="utf-8")

    found = sorted(value for _line, value in _spellings_outside_wire(sample))

    assert found == ["total_cost_usd", "turn.completed"]


def test_every_exemption_is_still_needed():
    """`SPELLED_ELSEWHERE` may not outlive the spelling it excuses.

    An exemption whose module has stopped saying the word is an amnesty granted
    to whoever writes it next — the list would keep growing and never shrink,
    which is how allow-lists stop meaning anything.
    """
    stale = []
    for module, words in wire.SPELLED_ELSEWHERE.items():
        path = PACKAGE / f"{module}.py"
        assert path.exists(), f"{module} is exempted but does not exist"
        present = {value for _line, value in _string_literals(path)}
        stale += [f"{module}: {word!r}" for word in sorted(words)
                  if word not in present]

    assert stale == [], (
        "wire.SPELLED_ELSEWHERE excuses words nobody spells any more: "
        + ", ".join(stale) + ". Drop the entry — the gate covers them again.")


def test_the_vocabulary_is_derived_and_not_empty():
    """A guarded set built from `globals()` can silently become empty.

    If it did, every assertion above would pass on a package spelling the wire
    in ten places. Two known words and a floor are enough to catch that: the
    exact size is not the point and would only need editing.
    """
    assert wire.ITEM_COMPLETED in wire.VOCABULARY
    assert wire.CACHED_INPUT_TOKENS in wire.VOCABULARY
    assert len(wire.VOCABULARY) >= 30
