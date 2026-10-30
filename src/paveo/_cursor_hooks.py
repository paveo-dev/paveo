"""``paveo test cursor``: the same test as ``_guard_test``, with Cursor's rules (D116).

Read on 30 Oct 2026 from Cursor's hooks page (cursor.com/docs/agent/hooks),
and checked against a hook author's account of the same rules (r/ClaudeCode,
30 Oct):

- **Where.** ``hooks.json`` in an organisation's folder, ``~/.cursor`` and
  ``<project>/.cursor``. **All matching hooks from every source run**, and
  their answers merge: ``deny`` wins over ``ask``, ``ask`` over ``allow``,
  whatever the source. Hooks set for a team in Cursor's dashboard, and
  plugins' hooks, are not on disk here; by that merge they can only stop more.
- **Which hooks.** ``beforeShellExecution``, whose matcher is tested against
  the whole command, and ``preToolUse``, whose matcher is tested against the
  tool's name (``Shell`` for a command). Both fire for a shell command.
- **How it runs one.** The command line in ``/bin/sh``; the page names no
  shell on macOS and Linux (D116). A project's hooks start in the project, a
  person's in ``~/.cursor``, an organisation's in its file's folder.
- **How it reads the answer.** Exit 2 blocks. Exit 0 with JSON: ``permission``
  ``deny`` blocks; ``ask`` asks a person from ``beforeShellExecution`` and is
  not enforced from ``preToolUse``. Whether auto-run skips that question the
  page does not say, so an ask is never counted as a stop. **JSON that cannot
  be read, or that does not match the schema, blocks**, even without
  ``failClosed``. Any other
  failure (another exit, a crash, a timeout, nothing printed) lets the call
  through, unless the hook sets ``failClosed: true``.
- **What it cannot see, said in the run.** Cursor runs a project's hooks only
  in a trusted workspace; this tests what each hook does once trusted. An
  answer with no ``permission``, about which the page says nothing, is
  counted as could-not-tell.

Opens no socket; the hooks it runs are the project's own.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from . import _matcher_trial
from ._bypasses import ACTIONS, ORDINARY, Spelling
from ._doctor import _not_json, _reason, parse_settings
from ._guard_test import (
    _DEFAULT_TIMEOUT_S,
    _TIMEOUT_S,
    Agent,
    Decision,
    Found,
    Handler,
    Run,
    _untestable,
)

_SHELL_EVENT = "beforeShellExecution"
_TOOL_EVENT = "preToolUse"
# Cursor's name for each tool of the list's spellings that it runs.
_TOOLS = {"Bash": "Shell"}
# Cursor compiles a matcher as a JavaScript regex. Python reads these
# otherwise, or JavaScript refuses them; such a matcher is not run rather than
# read Python's way. One flagged here that JavaScript would take only makes
# the run unsure.
_NOT_JS = re.compile(r"\(\?(?:P|#|>|[aiLmsux-]+[):])|\\[AZ]|[*+?}]\+|\{,|\[\^?\]")
# Until it is tried, a matcher reaches every call and cannot be told, so a
# hook whose matcher is never tried says so, never "not guarded".
_NOT_TRIED = "its matcher was not tried"
_TYPES = ("command", "prompt")
_PERMISSION: dict[str, dict[str, Decision]] = {
    _SHELL_EVENT: {"deny": "block", "ask": "ask", "allow": "none"},
    # "ask is accepted by the schema but not enforced for preToolUse today".
    _TOOL_EVENT: {"deny": "block", "ask": "none", "allow": "none"},
}
_MESSAGES = ("user_message", "agent_message")


def agent() -> Agent:
    """Cursor, running each hook as its hooks page describes."""
    return Agent(
        title="Cursor",
        nothing_found="no Cursor hooks file found",
        tools=frozenset(_TOOLS),
        find=_find,
        call=_call,
        decide=_decide,
        hook_names="beforeShellExecution or preToolUse",
        asks_stop=False,
        without_read=(
            "Cursor's hooks page does not give its Read tool's input, so they "
            "cannot be sent as Cursor would send them."
        ),
        ask_note=(
            "an ask from beforeShellExecution is not counted as a stop: Cursor "
            "asks the person, and its hooks page does not say whether auto-run "
            "skips the question."
        ),
        failed_closed="it sets failClosed, so each one was stopped",
    )


def files(project: Path, home: Path, organisation: Path) -> list[Path]:
    """Where Cursor loads hooks for ``project``: the organisation's file, the
    person's, the project's. Their order decides nothing, since every
    matching hook runs and the strictest answer wins."""
    return [
        organisation,
        home / ".cursor" / "hooks.json",
        project / ".cursor" / "hooks.json",
    ]


# --------------------------------------------------------------------------
# Reading the files
# --------------------------------------------------------------------------


def _find(files: Sequence[Path], project: Path, named: bool) -> Found:
    found = Found()
    own = Path(os.path.abspath(project / ".cursor" / "hooks.json"))
    # Matchers still to try, by the hook's place.
    pending: dict[str, _Pending] = {}
    project_hooks = False
    # A file named twice, the home folder being the project, is read once. One
    # reached by two names, a project file linked to the person's, is two
    # sources to Cursor, each started from its own folder, so it is read twice.
    named_files = dict.fromkeys(Path(os.path.abspath(path)) for path in files)
    # The project's own file last, so its matchers are tried after the
    # organisation's and the person's, however the files were named.
    for path in sorted(named_files, key=lambda path: path == own):
        events = _load(path, named=named, found=found)
        if events is None:
            continue
        found.read += 1
        cwd = project if path == own else path.parent
        before = len(found.handlers)
        for event in (_SHELL_EVENT, _TOOL_EVENT):
            _read_event(
                path, event, events.get(event), found=found, cwd=cwd, pending=pending
            )
        project_hooks = project_hooks or (path == own and len(found.handlers) > before)
    _try_matchers(found, pending)
    if found.read:
        found.notes.append(
            "hooks set for a team in Cursor's dashboard, and plugins' hooks, are "
            "not on disk here and are not run: they can only stop more."
        )
    if project_hooks:
        found.notes.append(
            "Cursor runs a project's hooks only in a trusted workspace: this "
            "tests what each hook does once trusted."
        )
    return found


def _load(path: Path, *, named: bool, found: Found) -> dict[str, object] | None:
    """The hooks in one file, or None with the reason in ``found.unread``
    unless it is simply absent and was not asked for by name."""
    try:
        document = parse_settings(path)
    except FileNotFoundError:
        if named:
            found.unread.append(f"{path}: no such file")
        return None
    except (OSError, ValueError) as e:
        found.unread.append(f"{path}: could not read it ({_reason(e)})")
        return None
    if not isinstance(document, dict):
        found.unread.append(f"{path}: not a JSON object")
        return None
    version = document.get("version")
    # True == 1 in Python, and 1.0 too; Cursor's version is the integer 1.
    if type(version) is not int or version != 1:
        found.unread.append(
            f"{path}: Cursor requires version 1, and its hooks page does not say "
            "whether it runs a file with another or none"
        )
        return None
    hooks = document.get("hooks", {})
    if not isinstance(hooks, dict):
        found.unread.append(f"{path}: its hooks are not an object of events")
        return None
    return hooks


@dataclass(frozen=True)
class _Pending:
    """A hook whose matcher is still to be tried."""

    matcher: str
    event: str
    # Whether "not tried" is its only reason for not being run, so a try
    # that answers runs it.
    placeholder: bool


def _read_event(  # noqa: PLR0913 - each is a separate fact about the file
    path: Path,
    event: str,
    entries: object,
    *,
    found: Found,
    cwd: Path,
    pending: dict[str, _Pending],
) -> None:
    if entries is None:
        return
    if not isinstance(entries, list):
        found.unread.append(f"{path}: its {event} is not a list")
        return
    for h, entry in enumerate(entries):
        where = f"{path}: {event}[{h}]"
        if not isinstance(entry, dict):
            found.unread.append(f"{where} is not an object")
            continue
        kind, command = entry.get("type", "command"), entry.get("command")
        if kind not in _TYPES:
            found.unread.append(f"{where} has a type Cursor does not know")
            continue
        if kind == "command" and (not isinstance(command, str) or not command.strip()):
            found.unread.append(f"{where} has no command")
            continue
        # Every call, until a matcher says otherwise.
        if isinstance(command, str) and kind == "command":
            handler = _handler(where, entry, command, _every, event=event, cwd=cwd)
        else:
            # A prompt hook has a prompt, not a command, and may still deny.
            handler = _untestable(
                Handler(where, _every, (), 0, blocks_on_failure=False, event=event),
                "it is a prompt hook, which is not run",
            )
        found.handlers.append(_matched(handler, entry.get("matcher"), pending=pending))


def _matched(
    handler: Handler, matcher: object, *, pending: dict[str, _Pending]
) -> Handler:
    """The hook as far as its matcher can be read here: absent, empty or ``*``
    reaches every call; syntax JavaScript and Python read differently cannot
    be read; any other is a regex, could-not-tell until ``_try_matchers``
    tries it, against the command (beforeShellExecution) or the tool's name
    (preToolUse)."""
    if matcher is None or matcher in ("", "*"):
        return handler
    if (
        not isinstance(matcher, str)
        or len(matcher) > _matcher_trial.MAX_LENGTH
        or _NOT_JS.search(matcher)
    ):
        return (
            handler
            if handler.untestable
            else _untestable(handler, _matcher_trial.UNREAD)
        )
    placeholder = handler.untestable is None
    pending[handler.where] = _Pending(matcher, handler.event, placeholder)
    return _untestable(handler, _NOT_TRIED) if placeholder else handler


def _every(_spelling: Spelling) -> bool:
    return True


def _among(hits: frozenset[str], event: str) -> Callable[[Spelling], bool]:
    """A matcher's reach, from the texts it was found in: the command for
    beforeShellExecution, the tool's name for preToolUse."""
    if event == _SHELL_EVENT:
        return lambda spelling: spelling.text in hits
    return lambda spelling: _TOOLS[spelling.tool] in hits


def _try_matchers(
    found: Found,
    pending: dict[str, _Pending],
    *,
    try_all: Callable[
        [Sequence[tuple[str, str]], Mapping[str, Sequence[str]]],
        list[frozenset[str] | str],
    ]
    | None = None,
) -> None:
    """Give each hook in ``pending`` the calls its matcher reaches. A try
    that does not answer leaves the hook could-not-tell (Rule 14: the child
    is injected)."""
    if not pending:
        return
    texts = {
        _SHELL_EVENT: sorted(
            {s.text for a in ACTIONS for s in a.spellings if s.tool in _TOOLS}
            | {s.text for s in ORDINARY if s.tool in _TOOLS}
        ),
        _TOOL_EVENT: sorted(_TOOLS.values()),
    }
    # In the order the files were read: the project's own last.
    keys = list(dict.fromkeys((each.matcher, each.event) for each in pending.values()))
    # Resolved here, not as a default, so one patch reaches all three agents.
    tried = (try_all or _matcher_trial.try_all)(keys, texts)
    answers = dict(zip(keys, tried, strict=True))
    for i, handler in enumerate(found.handlers):
        wanted = pending.get(handler.where)
        if wanted is None:
            continue
        hits = answers[wanted.matcher, wanted.event]
        if isinstance(hits, str):
            if wanted.placeholder:
                found.handlers[i] = replace(handler, untestable=hits)
            continue
        found.handlers[i] = replace(
            handler,
            reaches=_among(hits, wanted.event),
            untestable=None if wanted.placeholder else handler.untestable,
        )


def _handler(  # noqa: PLR0913 - each is a separate fact about the hook
    where: str,
    entry: dict[str, object],
    command: str,
    reaches: Callable[[Spelling], bool],
    *,
    event: str,
    cwd: Path,
) -> Handler:
    handler = Handler(
        where,
        reaches,
        ("/bin/sh", "-c", command),
        _TIMEOUT_S,
        blocks_on_failure=entry.get("failClosed") is True,
        event=event,
        cwd=cwd,
    )
    # Unset, Cursor uses a "platform default" its page does not give, so a
    # hook cut short here may not have been cut short by Cursor.
    seconds = entry.get("timeout", _DEFAULT_TIMEOUT_S)
    # Compared, never converted: a 400-digit number is valid JSON and too
    # large for a float.
    if not (
        isinstance(seconds, int | float)
        and not isinstance(seconds, bool)
        and seconds > 0
    ):
        return _untestable(handler, "its timeout is not a positive number")
    return replace(
        handler, timeout=float(min(seconds, _TIMEOUT_S)), capped=seconds > _TIMEOUT_S
    )


# --------------------------------------------------------------------------
# The call, and the answer
# --------------------------------------------------------------------------


def _call(spelling: Spelling, project: Path, _mode: str, event: str) -> bytes:
    """What Cursor sends ``event`` for a shell command: the fields every hook
    gets, then the event's own."""
    call: dict[str, object] = {
        "conversation_id": "paveo-test",
        "generation_id": "paveo-test-generation",
        "model": "paveo-test",
        "hook_event_name": event,
        "cursor_version": "paveo-test",
        "workspace_roots": [str(project)],
        "user_email": None,
        "transcript_path": None,
    }
    if event == _SHELL_EVENT:
        call |= {"command": spelling.text, "cwd": str(project), "sandbox": False}
    else:
        call |= {
            "tool_name": _TOOLS[spelling.tool],
            "tool_input": {"command": spelling.text, "working_directory": str(project)},
            "tool_use_id": "paveo-test-tool-use",
            "cwd": str(project),
        }
    return json.dumps(call).encode()


def _decide(run: Run, handler: Handler) -> tuple[Decision, str | None]:  # noqa: PLR0911 - one per rule
    """One hook's answer to one call, as Cursor's page reads it, and why it
    failed if it did."""
    failed: Decision = "block" if handler.blocks_on_failure else "none"
    if run.failure is not None:
        return failed, run.failure
    if run.exit == 2:  # noqa: PLR2004 - Cursor's block code
        return "block", None
    if run.exit != 0:
        return failed, f"exit {run.exit}"
    if run.stdout is None:
        return "unknown", None
    text = run.stdout.decode("utf-8", "replace").strip()
    if not text:
        return failed, "printed nothing"
    try:
        answer = json.loads(text, parse_constant=_not_json)
    except (ValueError, RecursionError):
        return "block", None  # JSON Cursor cannot read blocks the call
    if not isinstance(answer, dict) or any(
        name in answer and not isinstance(answer[name], str) for name in _MESSAGES
    ):
        return "block", None  # an answer that fails the schema blocks it too
    if "permission" not in answer:
        return "unknown", None
    chosen = answer["permission"]
    if not isinstance(chosen, str) or chosen not in _PERMISSION[handler.event]:
        return "block", None
    return _PERMISSION[handler.event][chosen], None
