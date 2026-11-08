"""npm, node and npx judges.

What `npm run` may start is the host's table (`WorkerCommands.npm_scripts`),
and what a node/tsx script may do beyond its arguments' paths is the host's
judge (`WorkerCommands.script_refusal`); both are read off
`CommandContext.policy.commands`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname
from .command_arguments import (
    CommandContext,
    _ArgvJudge,
    _arg_path_refusal,
    _args_refusal,
    _existing_file,
    _script_args_refusal,
)
from .command_shell import (
    Unparsed,
    Word,
    _segments,
    _words,
)
from .command_paths import (
    resolve_path,
    within,
)

# Words a `.cmd` shim hands to cmd.exe (npm and npx are `.cmd` shims on
# Windows). cmd acts on `% !` and a newline anywhere, and on `& | < > ^` in a
# word it does not see quoted (`npm test a&b` runs `b`). PowerShell 5.1 and
# try_patch (`subprocess.list2cmdline`) both double-quote an argument that
# holds whitespace, so `-t "queues a|answers b"` reaches cmd as one quoted
# word and is safe (a logged worker filter the blanket rule declined,
# review 2026-11-03), while `-t 'a|b'` reaches it bare — unless it ends in a
# backslash, which escapes the closing quote.
_SHIM_ALWAYS = re.compile(r"[%!\n\r]")
_SHIM_BARE = re.compile(r"[&|<>^]")


def _shim_refusal(args: list[str]) -> str:
    for arg in args:
        quoted = bool(re.search(r"\s", arg)) and not arg.endswith("\\")
        if _SHIM_ALWAYS.search(arg) or (_SHIM_BARE.search(arg) and not quoted):
            return (f"{arg!r}: npm/npx are cmd shims, and cmd acts on % ! "
                    f"anywhere and on & | < > ^ in a word with no space "
                    f"(which nothing quotes for it)")
    return ""


def _package_json(base: Path, ctx: CommandContext) -> Optional[Path]:
    """The package.json npm uses from `base`: the nearest one up the tree,
    inside the writable roots."""
    for folder in (base, *base.parents):
        if not within(folder, ctx.roots):
            return None
        candidate = folder / "package.json"
        if candidate.is_file():
            return candidate
    return None


def _package_scripts(package: Path) -> Optional[dict]:
    """The `scripts` of a package.json, or None when it has none readable."""
    try:
        scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts")
    except (OSError, ValueError, AttributeError):
        return None
    return scripts if isinstance(scripts, dict) else None


_NPM_OWN_SWITCHES = frozenset({"--silent", "-s", "--if-present", "--no-color"})


@dataclass(frozen=True)
class NpmScript:
    """One package.json script `npm test` / `npm run` may start outside the
    sandbox: what it is (the reason it is granted)."""
    may: str


# Listing a script name (`WorkerCommands.npm_scripts`) is not trusting its body:
# every body — the script's, its `pre`/`post` hooks', the scripts its runner
# starts and every nested `npm run` — is parsed and judged as worker commands
# like any other (`_npm_body_refusal`), so a body that is not plain worker
# commands, or a hook nobody listed, is a "no" (second review of the
# allowlist, 2026-11-03: `npm run review` with `git reset --hard` in its body and `npm
# test` with a `pretest: git reset --hard` were granted while bodies were read
# for bad patterns and hooks not at all).
# Nested `npm run` levels judged before the judge gives up (and declines).
_NPM_DEPTH = 4
# Programs npm finds on the PATH it gives a script (node_modules/.bin): read
# as `npx <tool>`, which needs the tool installed (`_npx_tool_refusal`).
_NPM_BINS = frozenset({"vitest", "tsc", "tsx"})


def npm_script(name: str, scripts: Mapping[str, NpmScript]) -> Optional[NpmScript]:
    """The `scripts` entry of a script name (a key ending in `:` is a
    family: `gate:` is every `gate:<x>`), or None."""
    for key, entry in scripts.items():
        if name == key or (key.endswith(":") and name.startswith(key)
                           and len(name) > len(key)):
            return entry
    return None


def _npm_body_refusal(name: str, scripts: dict, forwarded: list[str],
                      folder: Path, ctx: CommandContext,
                      *, judge_argv: _ArgvJudge) -> str:
    """Why the body of script `name` may not run, or "": cut at `&&`/`;`
    as cmd.exe (npm's script shell on Windows) reads it, each command a
    worker command judged in the package folder — `forwarded` appended to
    the last, as npm does — with `vitest`/`tsc`/`tsx` read as the installed
    tools (`_NPM_BINS`)."""
    body = str(scripts.get(name) or "")
    try:
        tokens = _words(body, "cmd")
    except Unparsed as unparsed:
        return f"npm run {name}: {unparsed} in its body"
    segments = _segments(tokens)
    if not segments:
        return f"npm run {name} has an empty body"
    for at, segment in enumerate(segments):
        if any(not isinstance(token, Word) for token in segment):
            return f"npm run {name}: a pipe or redirect in its body"
        argv = [token.value for token in segment]
        if at == len(segments) - 1:
            argv += forwarded
        program = argv[0].lower()
        if program in _NPM_BINS:
            why = _npx_tool_refusal(program, argv[1:], folder, ctx, judge_argv=judge_argv)
        else:
            why = judge_argv(argv, folder, ctx)
        if why:
            return f"npm run {name}: {why}"
    return ""


def _npm_script_refusal(name: str, scripts: dict, forwarded: list[str],
                        package: Path, ctx: CommandContext,
                        *, judge_argv: _ArgvJudge) -> str:
    """Why `npm run <name>` may not run outside, or "": the script must be
    on the host's `npm_scripts`; its `pre<name>`/`post<name>` hooks, which
    npm runs around it unasked, must be listed too; and every body judged
    (`_npm_body_refusal`)."""
    listed, table = ctx.policy.commands.npm_scripts, ctx.policy.commands.npm_name
    if npm_script(name, listed) is None:
        return f"npm run {name} is not on {table}"
    for hook in (f"pre{name}", f"post{name}"):
        if hook in scripts and npm_script(hook, listed) is None:
            return (f"{package} has {hook}, which npm runs around {name}, and "
                    f"it is not on {table}")
    for script in (f"pre{name}", name, f"post{name}"):
        if script not in scripts:
            continue
        why = _npm_body_refusal(script, scripts,
                                forwarded if script == name else [],
                                package.parent, ctx, judge_argv=judge_argv)
        if why:
            return why
    return ""


def _npm(argv: list[str], cwd: Path, ctx: CommandContext,
         *, judge_argv: _ArgvJudge) -> str:
    words = argv[1:]
    why = _shim_refusal(words)
    if why:
        return why
    prefix = None
    at = 0
    while at < len(words) and words[at].startswith("-"):
        flag, sep, value = words[at].partition("=")
        if flag == "--prefix":
            if sep:
                prefix = value
            elif at + 1 < len(words):
                prefix = words[at + 1]
                at += 1
            else:
                return "npm --prefix with no value"
        elif words[at] not in _NPM_OWN_SWITCHES:
            return f"npm option {words[at]}"
        at += 1
    if at >= len(words):
        return "npm with no command"
    verb, rest = words[at], words[at + 1:]
    if verb in ("test", "t", "tst"):
        script = "test"
    elif verb in ("run", "run-script"):
        if not rest:
            return "npm run with no script"
        script, rest = rest[0], rest[1:]
    else:
        return f"npm {verb} (only `npm test` and `npm run <script>`)"
    base = cwd
    if prefix is not None:
        why = _arg_path_refusal(prefix, cwd, ctx)
        if why:
            return why
        base = resolve_path(prefix, cwd)[1]
    before = rest[:rest.index("--")] if "--" in rest else rest
    for word in before:
        # npm reads options before `--` itself (`--script-shell=<exe>`).
        if word.startswith("-") and word not in _NPM_OWN_SWITCHES:
            return f"npm option {word} (put the script's own after `--`)"
    package = _package_json(base, ctx)
    if package is None:
        return f"no package.json for npm between {base} and your writable roots"
    scripts = _package_scripts(package)
    if scripts is None or script not in scripts:
        return f"{package} has no npm script {script!r}"
    if ctx.npm_depth >= _NPM_DEPTH:
        return f"npm run {script}: nested deeper than {_NPM_DEPTH} levels"
    # npm drops the FIRST `--` only: in `npm run x -- --all -- --y` the
    # script sees `--all -- --y`, and a runner may split its own options at
    # the second (third review 2026-11-03: dropping both granted `--all`).
    forwarded = list(rest)
    if "--" in forwarded:
        forwarded.remove("--")
    why = _npm_script_refusal(script, scripts, forwarded, package,
                              replace(ctx, npm_depth=ctx.npm_depth + 1), judge_argv=judge_argv)
    return why or _args_refusal(forwarded, package.parent, ctx)


def _local_prefix(cwd: Path, ctx: CommandContext) -> Optional[Path]:
    """The folder npm takes as the local project from `cwd` (its
    `localPrefix`): the nearest one holding a package.json or a
    node_modules, inside the worker's tree."""
    for folder in (cwd, *cwd.parents):
        if not within(folder, (ctx.workspace,)):
            return None
        if (folder / "package.json").is_file() or (folder / "node_modules").is_dir():
            return folder
    return None


def _installed(tool: str, cwd: Path, ctx: CommandContext) -> bool:
    """Whether npx finds `tool` in the local project's node_modules/.bin, so
    it runs that and downloads nothing."""
    prefix = _local_prefix(cwd, ctx)
    if prefix is None:
        return False
    bins = prefix / "node_modules" / ".bin"
    return any((bins / name).is_file() for name in (tool, f"{tool}.cmd"))


# Switches that turn a watching or serving tool into one that never exits.
_WATCH_SWITCHES = frozenset({"-w", "--watch", "--ui", "--open"})


def _checked_script_refusal(script: Path, args: list[str], cwd: Path,
                            ctx: CommandContext, *, judge_argv: _ArgvJudge) -> str:
    """Check a repository script and its arguments for both node and tsx:
    the host's judge of what the script does (`WorkerCommands.script_refusal`),
    then the paths its arguments name."""
    host = ctx.policy.commands.script_refusal
    return ((host(script, args, ctx, judge_argv) if host is not None else "")
            or _script_args_refusal(script, args, cwd, ctx))


def _npx_tool_refusal(tool: str, rest: list[str], cwd: Path,
                      ctx: CommandContext,
                      *, judge_argv: _ArgvJudge) -> str:
    """vitest in run mode, tsc without watch, tsx on a script
    `_checked_script_refusal` grants; then the arguments' paths. No `@file`: tsc
    reads more arguments from it (`--outDir C:/outside`, third review
    2026-11-03), and for the others it is refused as cheaply."""
    response = next((word for word in rest if word.startswith("@")), None)
    if response is not None:
        return f"{tool} {response}: a response file the judge does not read"
    watching = next((word for word in rest
                     if word.split("=", 1)[0] in _WATCH_SWITCHES), None)
    # tsx takes its script first: every word after it is the script's own.
    if watching and tool != "tsx":
        return f"{tool} {watching} never exits"
    if tool == "vitest":
        if not rest or (rest[0] != "run" and "--run" not in rest):
            return "vitest without `run` watches and never exits"
    elif tool == "tsx":
        if not rest or rest[0].startswith("-"):
            return "tsx takes the script first"
        script, why = _existing_file(rest[0], cwd, ctx, "script")
        if why:
            return why
        return _checked_script_refusal(script, rest[1:], cwd, ctx,
                                       judge_argv=judge_argv)
    elif tool != "tsc":
        return f"{tool} (only vitest run, tsc and tsx)"
    return _args_refusal(rest, cwd, ctx)


def _npx(argv: list[str], cwd: Path, ctx: CommandContext,
         *, judge_argv: _ArgvJudge) -> str:
    """npx runs a tool, and DOWNLOADS one it does not find installed — into
    npm's cache outside the tree, then runs it; non-interactively it does
    not ask, and `npx tsc` without TypeScript installed fetches the
    unrelated package named `tsc` (review 2026-11-03). Granted only with
    `--no-install`/`--no` or when the tool is in the local project's
    node_modules/.bin (`_installed`)."""
    words = argv[1:]
    why = _shim_refusal(words)
    if why:
        return why
    base = cwd
    no_install = False
    at = 0
    while at < len(words) and words[at].startswith("-"):
        flag, sep, value = words[at].partition("=")
        if flag == "--prefix":
            target = value if sep else (words[at + 1] if at + 1 < len(words) else "")
            at += 0 if sep else 1
            why = _arg_path_refusal(target, cwd, ctx) if target else \
                "npx --prefix with no value"
            if why:
                return why
            base = resolve_path(target, cwd)[1]
        elif words[at] in ("--no-install", "--no"):
            no_install = True
        else:
            return f"npx option {words[at]} (it may install a package)"
        at += 1
    if at >= len(words):
        return "npx with no tool"
    tool, rest = words[at], words[at + 1:]
    if tool not in _NPM_BINS:
        return f"npx {tool} (only vitest run, tsc and tsx)"
    if not (no_install or _installed(tool, base, ctx)):
        return (f"npx {tool}: not installed in the project's node_modules/.bin, "
                f"so npx would download it (add --no-install)")
    why = _npx_tool_refusal(tool, rest, cwd, ctx, judge_argv=judge_argv)
    return f"npx {why}" if why else ""


_NODE_SWITCHES = frozenset({"--expose-gc", "--no-warnings",
                            "--enable-source-maps", "--check"})
_TSX_LOADERS = frozenset({"tsx", "tsx/esm"})
# Packages whose entry `node node_modules/<package>/...` may start, as the
# tool `_npx_tool_refusal` judges.
_NODE_PACKAGES = {"vitest": "vitest", "typescript": "tsc"}


def _loader_refusal(value: str, cwd: Path, ctx: CommandContext) -> str:
    """`--import` takes tsx and nothing else: the package name, or its
    loader file inside the writable roots (a `file:///` URL is read as the
    path it names)."""
    if value in _TSX_LOADERS:
        return ""
    raw = value
    if value.lower().startswith("file:"):
        # url2pathname, not `.path.lstrip("/")`: stripping the slash made
        # Windows' `/C:/x` a path but POSIX's `/tmp/x` a relative `tmp/x`.
        raw = url2pathname(urlparse(value).path)
    loader, why = _existing_file(raw, cwd, ctx, "--import")
    if why:
        return why
    parts = [part.casefold() for part in loader.parts]
    if "node_modules" not in parts or "tsx" not in parts:
        return f"--import {value} is not the tsx loader"
    return ""


def _node(argv: list[str], cwd: Path, ctx: CommandContext,
          *, judge_argv: _ArgvJudge) -> str:
    words = argv[1:]
    at = 0
    while at < len(words) and words[at].startswith("-"):
        flag, sep, value = words[at].partition("=")
        if flag == "--import":
            if not sep:
                at += 1
                value = words[at] if at < len(words) else ""
            why = _loader_refusal(value, cwd, ctx)
            if why:
                return why
        elif not (words[at] in _NODE_SWITCHES
                  or flag == "--max-old-space-size"):
            return f"node option {words[at]} (-e/-p/--require run code " \
                   f"the judge cannot read)"
        at += 1
    if at >= len(words):
        return "node with no script"
    script, why = _existing_file(words[at], cwd, ctx, "script")
    if why:
        return why
    if "--check" in words[:at]:
        # Syntax only: node parses the script and runs none of it, whatever
        # it is (third review 2026-11-03: `node --check scripts/drive.mjs`
        # was declined as Vite's bake).
        return ""
    rest = words[at + 1:]
    parts = [part.casefold() for part in script.parts]
    if "node_modules" in parts:
        # A package's own entry is tooling, not one of the repo's scripts:
        # vitest's and TypeScript's, judged as those tools (a reviewer runs
        # a slot's vitest this way); any other package — vite's server
        # among them — is a "no".
        package = parts[len(parts) - 1 - parts[::-1].index("node_modules") + 1:]
        tool = _NODE_PACKAGES.get(package[0] if package else "")
        if tool is None:
            return (f"node {words[at]}: only vitest's and TypeScript's entries "
                    f"run from node_modules")
        return _npx_tool_refusal(tool, rest, cwd, ctx, judge_argv=judge_argv)
    return _checked_script_refusal(script, rest, cwd, ctx,
                                   judge_argv=judge_argv)
