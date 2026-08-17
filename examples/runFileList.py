"""
Example wrapper: process every file listed in a files.md, one per iteration.

This is the ListFileDriver pattern. The list file (here `files.md`) holds one
source path per line; each iteration picks a still-pending path at random, runs
the selected provider with your prompt, and on success strikes that path out of
the list — so
the run is idempotent (stop any time and relaunch to pick up the rest).

This particular example asks the agent to write a short summary of each source file
into a sibling `<name>.summary.md`. Swap `prompt()` / `target_suffix` / `model()`
for your own per-file task (generate docs, add license headers, refactor, lint…).

Copy this into your host project root (next to the `tools/llm-loop`
submodule), adjust the class attributes, and run `python runFileList.py`.
"""

import os
import sys

# Make the vendored submodule importable. Adjust the path if you put the
# submodule somewhere other than tools/llm-loop.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools", "llm-loop", "src"))

from llm_loop import ListFileDriver

LIST_FILE_REL = "files.md"        # one source path per line


class FileListDriver(ListFileDriver):
    list_file = LIST_FILE_REL
    target_suffix = ".summary.md"  # output sibling: foo.py -> foo.summary.md
    source_ext = ""                # "" = append suffix; or e.g. ".py" to replace it
    description = f"Process every file listed in {LIST_FILE_REL}, one per iteration."

    def model(self) -> str:
        """A cheaper/faster model is enough for this mechanical task. Return "" to
        let the CLI use its own configured model instead."""
        return "" if self.provider == "codex" else "sonnet"

    def prompt(self, source: str, target: str) -> str:
        """Instructions for a single file (receives absolute paths)."""
        return (
            f"Read the file {source} and write a concise Markdown summary of what "
            f"it does to a NEW file {target}. Do not modify {source}. "
            f"If {target} already exists, overwrite it."
        )


# Another prompt example — swap it in by overriding prompt() with this body:
#
#     def prompt(self, source, target):
#         """Translate the prose to Ukrainian, keep technical tokens verbatim."""
#         return (
#             f"Translate the product configuration file from English to Ukrainian.\n\n"
#             f"Source file (English, DO NOT modify it): {source}\n"
#             f"Write the Ukrainian translation to a NEW file: {target}\n\n"
#             f"Rules:\n"
#             f"- Read the source, then create the target file with the translated content.\n"
#             f"- Translate prose, headings and table cell descriptions into natural Ukrainian.\n"
#             f"- Preserve the Markdown structure exactly: same headings, tables, lists, "
#             f"bold/italic, and blank-line layout.\n"
#             f"- Do NOT translate or alter: numbers, units, currency/prices, model and "
#             f"part numbers, material grades, and the marketplace search strings "
#             f'(e.g. `search: \"...\"`) — keep those verbatim.\n'
#             f"- Do NOT modify, rename or delete the source file; only write {target}.\n"
#             f"- If {target} already exists, overwrite it with a fresh translation."
#         )


if __name__ == "__main__":
    FileListDriver.main()
