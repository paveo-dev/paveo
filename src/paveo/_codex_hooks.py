"""``paveo test codex``: the same test as ``_guard_test``, with Codex's rules (D111).

Read on 25 Oct 2026 from Codex's hooks page (learn.chatgpt.com/docs/hooks) and
its source (openai/codex ``970111f6cb``: ``codex-rs/hooks``, ``protocol/src/hooks.rs``,
``config/src/hook_config.rs``), where the page is silent:

- **Where.** ``hooks.json`` and the ``[hooks]`` table of ``config.toml``, in
  ``$CODEX_HOME`` (``~/.codex``) and in the ``.codex`` folder of every folder
  from the project's root down to where Codex starts. **All load**: a higher
  layer adds hooks, never replaces them. A file that fails Codex's schema is
  skipped whole, with a warning, so none of its hooks runs.
- **How it runs one.** ``$SHELL -lc "<command>"``, the person's own login
  shell (``/bin/sh`` when unset), for at most ``timeout`` seconds (600 unset).
- **How it reads the answer.** Exit 2 blocks, but only with a reason on stderr.
  Exit 0 with nothing printed lets the call on; with JSON, ``permissionDecision:
  deny`` and the older ``decision: block`` block, each only with a reason.
  Everything else is the hook **failing**, which lets the call on: another exit,
  a timeout, JSON that fails the schema, and every answer Codex parses but does
  not support yet, ``ask`` among them. So **an ask is never a stop in Codex.**
- **What it cannot see, said in the run.** Codex runs a hook only once a person
  has trusted it, and a project's only in a trusted project; this tests what
  each hook does once trusted. Hooks a plugin or a managed ``requirements.toml``
  adds are not run.

Opens no socket; the hooks it runs are the project's own.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

from ._bypasses import Spelling
from ._doctor import _not_json, _read, _reason, parse_settings
from ._guard_test import (
    _DEFAULT_TIMEOUT_S,
    _TIMEOUT_S,
    Agent,
    Decision,
    Found,
    Handler,
    Run,
    _untestable,
    by_tool,
)

# Every event Codex reads from a hooks file. A group under any of them that
# fails the schema makes Codex skip the whole file, so all are checked.
_EVENTS = (
    "PreModelRequest",
    "PostModelRequest",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "Interrupt",
)
# The fields of each hook type Codex reads, and the JSON types each takes;
# None is null, which Codex reads as "not set". Other fields are ignored.
_TEXT_OR_NULL = (str, type(None))
_COUNT_OR_NULL = (int, type(None))
_FIELDS: dict[str, dict[str, tuple[type, ...]]] = {
    "command": {
        "command": (str,),
        "commandWindows": _TEXT_OR_NULL,
        "command_windows": _TEXT_OR_NULL,
        "timeout": _COUNT_OR_NULL,
        "async": (bool,),
        "statusMessage": _TEXT_OR_NULL,
        "additionalContextLimit": _COUNT_OR_NULL,
    },
    "mcp_tool": {
        "server": (str,),
        "tool": (str,),
        "input": (dict,),
        "timeout": _COUNT_OR_NULL,
        "statusMessage": _TEXT_OR_NULL,
    },
    "prompt": {},
    "agent": {},
}
_REQUIRED = {"command": ("command",), "mcp_tool": ("server", "tool")}
# A matcher of only these characters is a list of exact names (matcher.rs).
_NAMES = re.compile(r"[A-Za-z0-9_|]+")
# Codex compiles any other matcher with Rust's regex crate, which refuses
# look-around, back-references, comments and possessive quantifiers, and
# reads some escapes otherwise than Python. Such a matcher is not run rather
# than read Python's way; one flagged here that Rust would take only makes
# the run unsure.
_NOT_RUST = re.compile(
    r"\(\?(?:<?[=!]|P=|#)|\\[1-9gZaNU]|[*+?}]\+|\[[^\]]*(?:&&|--|~~)"
)

# What Codex reads in a PreToolUse answer printed on exit 0, and each type.
_TOP: dict[str, tuple[type, ...]] = {
    "continue": (bool,),
    "stopReason": _TEXT_OR_NULL,
    "suppressOutput": (bool,),
    "systemMessage": _TEXT_OR_NULL,
    "decision": _TEXT_OR_NULL,
    "reason": _TEXT_OR_NULL,
    "hookSpecificOutput": (dict, type(None)),
}
_SPECIFIC: dict[str, tuple[type, ...]] = {
    "hookEventName": (str,),
    "permissionDecision": _TEXT_OR_NULL,
    "permissionDecisionReason": _TEXT_OR_NULL,
    "updatedInput": (dict, list, str, int, float, bool, type(None)),
    "additionalContext": _TEXT_OR_NULL,
}
# The event names Codex accepts in ``hookEventName`` (schema.rs).
_NAMED_EVENTS = frozenset(_EVENTS) - {
    "PreModelRequest",
    "PostModelRequest",
    "SessionEnd",
}
_UNSUPPORTED = "printed an answer Codex does not support yet"


def agent(shell: str) -> Agent:
    """Codex, running each hook in ``shell``, as Codex runs it in ``$SHELL``."""
    return Agent(
        title="Codex",
        nothing_found="no Codex hooks file found",
        tools=frozenset({"Bash"}),
        find=lambda files, _project, named: _find(files, named=named, shell=shell),
        call=_call,
        decide=lambda run, _handler: _decide(run),
        hook_names="PreToolUse",
        # Codex reads an ask but does not support it: _decide reads it as the
        # hook failing, so no verdict here is ever an ask.
        asks_stop=False,
        without_read="Codex has no such tool, and reads files through its shell.",
        ask_note="Codex does not support ask yet, so an ask is never a stop.",
        failed_closed="Codex has no way for a hook to fail closed.",
    )


def files(project: Path, home: Path) -> list[Path]:
    """Where Codex loads hooks for a session started in ``project``, lowest
    layer first: ``home``, then each ``.codex`` from the project's root (the
    nearest folder above holding ``.git``, Codex's default marker) down."""
    root = next(
        (
            folder
            for folder in (project, *project.parents)
            if (folder / ".git").exists()
        ),
        project,
    )
    folders = [project, *project.parents]
    layers = [home] + [
        folder / ".codex" for folder in reversed(folders[: folders.index(root) + 1])
    ]
    return [layer / name for layer in layers for name in ("hooks.json", "config.toml")]


# --------------------------------------------------------------------------
# Reading the files
# --------------------------------------------------------------------------


def _find(files: Sequence[Path], *, named: bool, shell: str) -> Found:
    found = Found()
    # A hook switched off in /hooks, by its key: the file, the event, two indexes.
    off: set[str] = set()
    loaded: list[tuple[Path, list[object]]] = []
    # The home folder being the project reads its files once.
    for path in dict.fromkeys(Path(os.path.abspath(path)) for path in files):
        events = _load(path, named=named, found=found, off=off)
        if events is not None:
            found.read += 1
            groups = events.get("PreToolUse", [])
            loaded.append((path, groups if isinstance(groups, list) else []))
    for path, groups in loaded:
        _read_groups(path, groups, found=found, off=off, shell=shell)
    if found.handlers:
        found.notes.append(
            "Codex runs a hook only once someone has trusted it in /hooks, and a "
            "project's hooks only in a trusted project: this tests what each hook "
            "does once trusted."
        )
    return found


def _parse(path: Path, *, named: bool, found: Found) -> dict[str, object] | None:
    """The whole file, parsed, or None with the reason in ``found.unread``
    unless it is simply absent and was not asked for by name."""
    try:
        if path.suffix == ".toml":
            # Bounded, and only a regular file: a cloned repository's
            # config.toml linked to a FIFO would otherwise hang the run.
            document: object = tomllib.loads(_read(path).decode("utf-8"))
        else:
            document = parse_settings(path)
    except FileNotFoundError:
        if named:
            found.unread.append(f"{path}: no such file")
        return None
    except (OSError, ValueError, RecursionError) as e:  # TOMLDecodeError: ValueError
        if isinstance(e, RecursionError):
            why = "it is nested too deeply"
        elif isinstance(e, tomllib.TOMLDecodeError):
            why = "not valid TOML"
        else:
            why = _reason(e)
        found.unread.append(f"{path}: could not read it ({why})")
        return None
    if not isinstance(document, dict):
        found.unread.append(f"{path}: not a JSON object")
        return None
    return document


def _load(
    path: Path, *, named: bool, found: Found, off: set[str]
) -> dict[str, object] | None:
    """The events in one file, as Codex would load them, or None."""
    document = _parse(path, named=named, found=found)
    if document is None:
        return None
    hooks: object = None
    if path.suffix == ".toml":
        features = document.get("features")
        if isinstance(features, dict) and isinstance(features.get("hooks"), bool):
            # The last file read is the highest layer, and its value stands.
            found.disabled = (
                None
                if features["hooks"]
                else f"{path} sets features.hooks = false, so Codex runs no hook."
            )
        hooks = document.get("hooks", {})
        events = hooks
        if isinstance(hooks, dict):
            events = {k: v for k, v in hooks.items() if k != "state"}
    else:
        extra = set(document) - {"description", "hooks"}
        if extra or not isinstance(document.get("description"), _TEXT_OR_NULL):
            odd = f"top-level {sorted(extra)[0]!r}" if extra else "description not text"
            found.unread.append(
                f"{path}: Codex refuses a hooks.json with a {odd}, so none of its "
                "hooks would run"
            )
            return None
        events = document.get("hooks", {})
    wrong = _misshapen(events)
    if wrong is not None:
        found.unread.append(
            f"{path}: {wrong}, and Codex skips such a file whole, so none of its "
            "hooks would run"
        )
        return None
    if path.suffix == ".toml" and isinstance(hooks, dict):
        # Only from a [hooks] table Codex loads: one it skips switches nothing.
        off.update(_switched_off(hooks))
    return events if isinstance(events, dict) else None  # _misshapen: always a dict


def _switched_off(hooks: dict[str, object]) -> set[str]:
    """The keys of hooks a person switched off in /hooks (``[hooks.state]``)."""
    states = hooks.get("state")
    if not isinstance(states, dict):
        return set()
    return {
        key
        for key, state in states.items()
        if isinstance(state, dict) and state.get("enabled") is False
    }


def _misshapen(events: object) -> str | None:  # noqa: PLR0911 - one per rule
    """Why Codex's schema refuses these events, or None."""
    if not isinstance(events, dict):
        return "its hooks are not a table of events"
    for event in _EVENTS:
        groups = events.get(event, [])
        if not isinstance(groups, list):
            return f"its {event} is not a list"
        for g, group in enumerate(groups):
            if not isinstance(group, dict):
                return f"its {event}[{g}] is not an object"
            if not isinstance(group.get("matcher"), _TEXT_OR_NULL):
                return f"the matcher of {event}[{g}] is not text"
            entries = group.get("hooks", [])
            if not isinstance(entries, list):
                return f"the hooks of {event}[{g}] are not a list"
            for h, entry in enumerate(entries):
                why = _bad_hook(entry)
                if why is not None:
                    return f"{event}[{g}].hooks[{h}] {why}"
    return None


def _bad_hook(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return "is not an object"
    kind = entry.get("type")
    if not isinstance(kind, str) or kind not in _FIELDS:
        return "has a type Codex does not know"
    for name in _REQUIRED.get(kind, ()):
        if name not in entry:
            return f"has no {name}"
    for name, types in _FIELDS[kind].items():
        value = entry.get(name)
        if name not in entry:
            continue
        # bool is an int in Python; Codex's counts are never booleans.
        counted = types is _COUNT_OR_NULL and isinstance(value, bool)
        if counted or not isinstance(value, types):
            return f"has a {name} of the wrong type"
        # Codex reads each count as a u64, so one past its range fails too.
        if (
            types is _COUNT_OR_NULL
            and isinstance(value, int)
            and not 0 <= value < 2**64
        ):
            return f"has a {name} out of range"
    return None


def _read_groups(
    path: Path, groups: list[object], *, found: Found, off: set[str], shell: str
) -> None:
    # _misshapen has checked every type below; the isinstance checks are for
    # the type checker, and skip nothing.
    for g, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        tool_reaches, unreadable = _reaches(group.get("matcher"))
        reaches = by_tool(tool_reaches)
        entries = group.get("hooks", [])
        for h, entry in enumerate(entries if isinstance(entries, list) else []):
            if not isinstance(entry, dict):
                continue
            where = f"{path}: PreToolUse[{g}].hooks[{h}]"
            keys = {
                f"{place}:pre_tool_use:{g}:{h}"
                for place in (str(path), os.path.realpath(path))
            }
            kind = entry["type"]
            if keys & off:
                found.notes.append(f"{where} is switched off in /hooks: not run")
            elif kind in {"prompt", "agent"}:
                found.notes.append(
                    f"{where} is a hook of type {kind}, which Codex skips"
                )
            elif kind == "mcp_tool":
                handler = Handler(where, reaches, (), 0, blocks_on_failure=False)
                found.handlers.append(
                    _untestable(handler, "it is an MCP tool hook, which is not run")
                )
            elif not str(entry["command"]).strip():
                found.notes.append(f"{where} has an empty command, which Codex skips")
            elif entry.get("async") is True:
                found.notes.append(f"{where} runs in the background: it stops nothing")
            else:
                handler = _handler(where, entry, reaches, shell=shell)
                found.handlers.append(
                    handler if unreadable is None else _untestable(handler, unreadable)
                )


def _handler(
    where: str,
    entry: dict[str, object],
    reaches: Callable[[Spelling], bool],
    *,
    shell: str,
) -> Handler:
    timeout = entry.get("timeout")
    # Codex: unset is 600 seconds, and 0 is raised to 1. Compared, never
    # converted: a 400-digit count is valid JSON and too large for a float.
    seconds = timeout if isinstance(timeout, int) and timeout >= 1 else 1
    if timeout is None:
        seconds = int(_DEFAULT_TIMEOUT_S)
    return Handler(
        where,
        reaches,
        (shell, "-lc", str(entry["command"])),
        float(min(seconds, _TIMEOUT_S)),
        blocks_on_failure=False,  # Codex has no way to fail closed
        capped=seconds > _TIMEOUT_S,
    )


def _reaches(matcher: object) -> tuple[Callable[[str], bool], str | None]:
    """Which tool names a matcher reaches, as Codex's ``HookMatcher`` reads it:
    empty or ``*`` is every tool; only letters, digits, ``_`` and ``|`` is a
    list of exact names; anything else is a regex that may match anywhere."""
    if not isinstance(matcher, str) or matcher in ("", "*"):  # None: no matcher
        return (lambda _tool: True), None
    if _NAMES.fullmatch(matcher):
        names = set(matcher.split("|"))
        return names.__contains__, None
    unread = (lambda _tool: True), "its matcher is not a pattern paveo test can read"
    if _NOT_RUST.search(matcher):
        return unread
    try:
        pattern = re.compile(matcher)
    except (re.error, RecursionError, OverflowError):
        return unread
    return (lambda tool: pattern.search(tool) is not None), None


# --------------------------------------------------------------------------
# The call, and the answer
# --------------------------------------------------------------------------


def _call(spelling: Spelling, project: Path, _mode: str, _event: str) -> bytes:
    """The PreToolUse input Codex sends for a shell command (schema.rs)."""
    return json.dumps(
        {
            "session_id": "paveo-test",
            "turn_id": "paveo-test-turn",
            "transcript_path": None,
            "cwd": str(project),
            "hook_event_name": "PreToolUse",
            "model": "paveo-test",
            "permission_mode": "default",
            "tool_name": spelling.tool,
            "tool_input": {"command": spelling.text},
            "tool_use_id": "call_paveo_test",
        }
    ).encode()


def _decide(run: Run) -> tuple[Decision, str | None]:  # noqa: PLR0911 - one per rule
    """One hook's answer to one call, as ``pre_tool_use.rs`` reads it, and why
    it failed if it did. A failure always lets the call on."""
    if run.failure is not None:
        return "none", run.failure
    if run.exit == 2:  # noqa: PLR2004 - Codex's block code
        if run.wrote_stderr:
            return "block", None
        return "none", "exit 2 with no reason on stderr"
    if run.exit != 0:
        return "none", f"exit {run.exit}"
    if run.stdout is None:
        return "unknown", None
    text = run.stdout.decode("utf-8", "replace").strip()
    if not text:
        return "none", None
    try:
        value = json.loads(text, parse_constant=_not_json)
    except (ValueError, RecursionError):
        value = None
    if not isinstance(value, dict):
        # Plain text decides nothing; text that starts like JSON is a failure.
        if text.startswith(("{", "[")):
            return "none", "printed JSON Codex cannot read"
        return "none", None
    return _decision(value)


def _decision(output: dict[str, object]) -> tuple[Decision, str | None]:
    schema = "printed JSON that fails Codex's schema"
    specific = output.get("hookSpecificOutput")
    if not _typed(output, _TOP) or output.get("decision") not in (
        None,
        "block",
        "approve",
    ):
        return "none", schema
    if isinstance(specific, dict) and (
        not _typed(specific, _SPECIFIC)
        or specific.get("hookEventName") not in _NAMED_EVENTS
        or specific.get("permissionDecision") not in (None, "allow", "deny", "ask")
    ):
        return "none", schema
    if (
        output.get("continue") is False
        or output.get("stopReason") is not None
        or output.get("suppressOutput") is True
    ):
        return "none", _UNSUPPORTED
    if isinstance(specific, dict) and any(
        specific.get(name) is not None
        for name in ("permissionDecision", "permissionDecisionReason", "updatedInput")
    ):
        return _specific(specific)
    return _legacy(output.get("decision"), output.get("reason"))


def _specific(specific: dict[str, object]) -> tuple[Decision, str | None]:
    chosen = specific.get("permissionDecision")
    rewrites = specific.get("updatedInput") is not None
    if chosen == "allow":
        return ("none", None) if rewrites else ("none", _UNSUPPORTED)
    if rewrites or chosen == "ask" or chosen is None:
        return "none", _UNSUPPORTED
    reason = specific.get("permissionDecisionReason")
    if isinstance(reason, str) and reason.strip():
        return "block", None
    return "none", "denied with no reason, which Codex treats as a failure"


def _legacy(decision: object, reason: object) -> tuple[Decision, str | None]:
    if decision == "approve" or (decision is None and reason is not None):
        return "none", _UNSUPPORTED
    if decision is None:
        return "none", None
    if isinstance(reason, str) and reason.strip():
        return "block", None
    return "none", "blocked with no reason, which Codex treats as a failure"


def _typed(fields: dict[str, object], types: dict[str, tuple[type, ...]]) -> bool:
    """Every field Codex reads has its type, and there is no other: Codex's
    answer schema refuses unknown fields, which ``_guard_test._typed`` does
    not check, since Claude Code's schema allows them."""
    if set(fields) - set(types):
        return False
    return all(
        isinstance(fields[name], kinds)
        for name, kinds in types.items()
        if name in fields
    )
