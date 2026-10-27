"""``paveo test``: whether a project's own Claude Code hooks stop the dangerous
actions, however they are spelled (D102). Codex's hooks are tested the same
way with Codex's rules (``_codex_hooks``, D111); this module describes Claude
Code's, and the run both share.

::

    paveo test [codex] [--expect ACTION ...] [--settings FILE ...] [--json FILE]

Each spelling in ``_bypasses`` is sent to every ``PreToolUse`` hook whose matcher
reaches it, as the JSON Claude Code sends, and each answer is read the way Claude
Code's hooks reference reads it (read 19 Oct 2026, code.claude.com/docs/en/hooks):
exit 2 blocks; a printed ``permissionDecision`` of ``deny`` blocks and ``ask``
hands the call to a person (counted as stopped), in a ``hookSpecificOutput`` that
names ``PreToolUse``; the deprecated top-level ``decision: block`` is a deny. A
hook that fails, by exiting with another code, printing JSON that fails the
schema, not starting or timing out, lets the call through unless it sets
``onFailure: "block"``.

**It fails** (exit 1) when an action is stopped in some spellings and not others,
the guard that can be walked around, or when an action named with ``--expect``
passes in any spelling, the guard that is off. An action no hook stops at all is
reported but fails only when expected: a team that guards pushes alone is not
failed for ``rm``. When no hook stops anything, the run fails whatever is
expected: the settings moved, or every hook was switched off. **It cannot
tell** (exit 2) when a settings file is unreadable,
when a hook it does not run reaches a spelling no other hook stops (or one only an
ask stops, since the hook not run might defer), when a hook stops an ordinary call
too, so a stop may not mean it read the command, or when managed settings may be
on the machine. Otherwise 0. A pass means these spellings were stopped, never
that the guard holds.

**It runs the project's hooks**, which ``doctor`` never does: like ``make test``,
run it on code you would run anyway. Every spelling is harmless if a hook ran it
(``_bypasses``). Each hook gets at most 10 seconds; one still running then is
ended with everything it started. **It never prints a
hook's command or output**: either can carry a token. It names where.

**Not read yet, and said so when present:** a timeout that is not a positive
number, whose effect Claude Code's schema decides; an ``if`` filter, which takes
permission-rule syntax; hooks that are not commands (``http`` would open a
socket); managed settings, which can switch every hook off, so their presence
makes the run unsure (server-managed settings leave nothing on disk to see);
Claude Code's own permission rules, which can only stop more; plugins' hooks,
which can stop more, and could outrank an ask with a defer, so beside them an
ask is not counted; ``continue: false``, whose effect on the call the reference
leaves unclear, so it is not counted as a stop.

Paveo opens no socket here; the hooks it runs are the project's own.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, TextIO

from ._bypasses import ACTIONS, ORDINARY, VERSION, Spelling
from ._doctor import (
    PERMISSION_MODES,
    _not_json,
    _settings,
    _shown,
    hook_groups,
    reaches_tools,
    runs_in_background,
)

# Claude Code waits up to 600 s for a hook. A test that waited that long per
# spelling would never be run, so no hook is given longer than this. A hook whose
# own limit is shorter times out as it would in Claude Code; one stopped here
# before its own limit cannot be told, since Claude Code would have waited.
_TIMEOUT_S = 10.0
_DEFAULT_TIMEOUT_S = 600.0
# Claude Code finishes reading "shortly after the hook's own process exits".
_EXIT_GRACE_S = 0.5
_MAX_OUTPUT = 1 << 20
# The fields of PreToolUse's own output besides the decision, and their types.
_SPECIFIC: dict[str, type] = {
    "permissionDecision": str,
    "permissionDecisionReason": str,
    "additionalContext": str,
    "updatedInput": dict,
}
# Top-level fields every event accepts, and the type the schema gives each.
_UNIVERSAL: dict[str, type] = {
    "continue": bool,
    "suppressOutput": bool,
    "stopReason": str,
    "systemMessage": str,
    "reason": str,
    "terminalSequence": str,
}
Decision = Literal["block", "defer", "ask", "none", "unknown"]

_PERMISSION: dict[str, Decision] = {
    "deny": "block",
    "defer": "defer",
    "ask": "ask",
    "allow": "none",
}
# The deprecated top-level field: approve is allow, block is deny.
_LEGACY: dict[str, Decision] = {"block": "block", "approve": "none"}
_TIMED_OUT = "did not answer in time"
_LEFT_RUNNING = "could not be stopped (the system refused), so it was left running"
_ASK_STOPS = {mode for mode, stops in PERMISSION_MODES.items() if stops}


@dataclass(frozen=True)
class Handler:
    """One command hook, as Claude Code would run it."""

    where: str
    reaches: Callable[[str], bool]
    argv: tuple[str, ...]
    timeout: float
    blocks_on_failure: bool
    # Why it cannot be run as Claude Code would run it, or None.
    untestable: str | None = None
    # Its own time limit is longer than the one paveo test gives it.
    capped: bool = False


@dataclass(frozen=True)
class Run:
    exit: int | None  # None: it timed out or did not start, said in ``failure``
    stdout: bytes | None  # None: more than _MAX_OUTPUT
    failure: str | None = None
    timed_out: bool = False
    # Whether it wrote anything but white space to stderr, which Codex needs
    # before exit 2 blocks. What it wrote is never kept: it can carry a token.
    wrote_stderr: bool = False


@dataclass
class Found:
    """What an agent's files say about its hooks, read before any is run."""

    handlers: list[Handler] = field(default_factory=list)
    unread: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    read: int = 0
    # Why the agent runs none of these hooks, said as a sentence, or None.
    disabled: str | None = None
    permission_rules: bool = False
    # Each plugin's latest enabled value, files read lowest precedence first.
    plugins: dict[str, object] = field(default_factory=dict)
    # The latest permissions.defaultMode, which decides whether an ask reaches
    # a person at all.
    default_mode: str = "default"
    # Claude Code runs a handler defined in several settings files once.
    seen: set[str] = field(default_factory=set)

    @property
    def plugins_on(self) -> bool:
        """Whether any plugin is left enabled, whose hooks are not run."""
        return any(value is not False for value in self.plugins.values())


@dataclass
class _Tally:
    """What happened to each handler across the run, for the notes."""

    calls: Counter[str] = field(default_factory=Counter)
    failures: dict[str, Counter[str]] = field(default_factory=dict)
    # A hook that timed out once is not run again: a hook that hangs would
    # otherwise cost ten seconds for every spelling.
    timed_out: set[str] = field(default_factory=set)
    # Of those, the ones stopped by paveo test's limit before their own: what
    # they would have answered cannot be told, so it is no failure either.
    cut_short: set[str] = field(default_factory=set)
    # Of those, the ones the system would not let us stop (D103); an uncapped
    # one says so in its failure line instead.
    left_running: set[str] = field(default_factory=set)


Runner = Callable[[Handler, bytes], Run]


@dataclass(frozen=True)
class Agent:
    """What differs between the agents ``paveo test`` reads: where their hooks
    are, the call each sends a hook, and how each reads the answer."""

    title: str
    nothing_found: str
    # The tools it has, of those the list's spellings use.
    tools: frozenset[str]
    find: Callable[[Sequence[Path], Path, bool], Found]  # files, project, named
    call: Callable[[Spelling, Path, str], bytes]  # spelling, project, mode
    decide: Callable[[Run, bool], tuple[Decision, str | None]]  # blocks on failure


@dataclass(frozen=True)
class _Probe:
    """What every call in one run is sent with."""

    agent: Agent
    project: Path
    run: Runner
    tally: _Tally
    # Hooks from plugins, which are not run, may also reach the call.
    unseen: bool
    # The permission mode the settings start a session in.
    mode: str


def command(  # noqa: PLR0913 - each is an input the command line decides
    files: Sequence[Path],
    *,
    project: Path,
    expect: Sequence[str],
    out: TextIO,
    run: Runner,
    managed: Sequence[Path] = (),
    named: bool = False,
    record: TextIO | None = None,
    agent: Agent | None = None,
) -> int:
    """Run every spelling ``agent`` has the tools for against the hooks in
    ``files``, lowest precedence first, and report; see the module for what
    each exit means. ``managed``: where managed settings may be, which are not
    read; one that is there makes the run unsure, since it can switch every
    hook here off. ``record``: where ``--json`` writes the same results for a
    machine, the list's version with them."""
    agent = agent or CLAUDE_CODE
    found = agent.find(files, project, named)
    present = [path for path in managed if _maybe_there(path)]
    for line in found.unread:
        out.write(f"paveo: {_shown(line)}. Not read.\n")
    if found.read == 0 and not found.unread:
        out.write(f"paveo: {agent.nothing_found}. Looked for:\n")
        out.writelines(f"  {_shown(str(path))}\n" for path in files)
    handlers = found.handlers
    if found.disabled is not None:
        out.write(f"paveo: {_shown(found.disabled)}\n")
        handlers = []
    elif not handlers and found.read:
        out.write("paveo: no PreToolUse command hook found.\n")
    tally = _Tally()
    probe = _Probe(
        agent, project, run, tally, unseen=found.plugins_on, mode=found.default_mode
    )
    tested = {
        action.name: [s for s in action.spellings if s.tool in agent.tools]
        for action in ACTIONS
    }
    results = {
        name: [_verdict(s, handlers, probe) for s in spellings]
        for name, spellings in tested.items()
    }
    ordinary_stopped = [
        _named(spelling)
        for spelling in ORDINARY
        if spelling.tool in agent.tools
        and _verdict(spelling, handlers, replace(probe, unseen=False))
        in {"block", "ask"}
    ]
    spellings = sum(len(each) for each in tested.values())
    out.write(
        f"paveo test: {len(handlers)} hook{'s' * (len(handlers) != 1)}, "
        f"{spellings} spellings of {len(ACTIONS)} dangerous actions.\n"
    )
    notes = _notes(
        found,
        handlers,
        tally,
        title=agent.title,
        present=present,
        ordinary=ordinary_stopped,
    )
    left_out = sum(len(action.spellings) for action in ACTIONS) - spellings
    if left_out:
        notes.append(
            f"{left_out} spellings that use Claude Code's Read tool are left out: "
            f"{agent.title} has no such tool, and reads files through its shell."
        )
    if any("ask" in verdicts for verdicts in results.values()):
        notes.append(
            "an ask is counted as a stop, which holds in a session that starts "
            "in default (manual), plan, acceptEdits or dontAsk mode; one started "
            "with --dangerously-skip-permissions, --permission-mode "
            "bypassPermissions or in auto mode asks no one."
        )
    out.writelines(f"  note: {_shown(line)}\n" for line in notes)
    out.write("\n")
    outcomes = [
        _outcome(name, tested[name], results[name], expected=name in expect)
        for name in tested
    ]
    walked = [outcome.name for outcome in outcomes if _report(outcome, out)]
    unsure = bool(found.unread or ordinary_stopped or present)
    unsure = unsure or any(outcome.unknown for outcome in outcomes)
    stopped_any = any(outcome.stopped for outcome in outcomes)
    out.write("\n")
    code, ending = _ending(walked, unsure=unsure, stopped_any=stopped_any)
    out.write(f"paveo: {ending}\n")
    if record is not None:
        document = {
            "bypass_list": VERSION,
            "exit": code,
            "ending": ending,
            "hooks": len(handlers),
            "notes": [_shown(line) for line in notes],
            "actions": [asdict(outcome) for outcome in outcomes],
        }
        record.write(json.dumps(document, indent=2) + "\n")
    return code


def _ending(
    walked: Sequence[str], *, unsure: bool, stopped_any: bool
) -> tuple[int, str]:
    if walked:
        return 1, f"these get past the hooks: {', '.join(walked)}."
    if not unsure and not stopped_any:
        # A test of guards that finds none is no pass, --expect or not: this is
        # the day the settings moved or every hook was switched off.
        return 1, "no hook stops any of these actions."
    if unsure:
        return 2, "could not tell for everything, so this is not a pass."
    return 0, "nothing expected got past the hooks."


def _notes(  # noqa: PLR0913 - each is a separate thing the run could not see
    found: Found,
    handlers: Sequence[Handler],
    tally: _Tally,
    *,
    title: str,
    present: Sequence[Path],
    ordinary: Sequence[str],
) -> list[str]:
    """Everything the run could not see or could not trust, said before the
    results it qualifies."""
    notes = [*found.notes, *_failure_notes(handlers, tally, title=title)]
    if found.permission_rules:
        notes.append(
            "Claude Code's own permission rules are not tested yet: a spelling "
            "that passes every hook may still be refused by them."
        )
    if found.default_mode not in _ASK_STOPS:
        notes.append(
            f"permissions.defaultMode is {found.default_mode}, where no one is "
            "asked, so an ask is not counted as a stop."
        )
    if found.plugins_on:
        notes.append(
            "hooks from plugins are not run yet: a spelling that passes every "
            "hook here may still be stopped by one, and an ask is not counted as "
            "a stop, since a plugin's defer would outrank it."
        )
    notes.extend(
        f"{path} holds managed settings, which are not read yet: they can "
        "switch these hooks off or add their own."
        for path in present
    )
    notes.extend(
        f"a hook also stops `{spelling}`, an ordinary call, so a stop below may "
        "not mean a hook read the command."
        for spelling in ordinary
    )
    return notes


@dataclass(frozen=True)
class _Outcome:
    """One action's result, as the report and ``--json`` both give it."""

    name: str
    expected: bool
    spellings: int
    stopped: int  # asked included
    asked: int
    unknown: int
    passing: tuple[str, ...]


def _outcome(
    name: str,
    spellings: Sequence[Spelling],
    verdicts: list[Decision],
    *,
    expected: bool,
) -> _Outcome:
    return _Outcome(
        name=name,
        expected=expected,
        spellings=len(verdicts),
        stopped=sum(v in {"block", "ask"} for v in verdicts),
        asked=verdicts.count("ask"),
        unknown=verdicts.count("unknown"),
        passing=tuple(
            _named(s) for s, v in zip(spellings, verdicts, strict=True) if v == "none"
        ),
    )


def _report(outcome: _Outcome, out: TextIO) -> bool:
    """Write one action's line; True when it fails the run."""
    total, stopped, unknown = outcome.spellings, outcome.stopped, outcome.unknown
    head = f"  {outcome.name:<11} "
    by_asking = f" ({outcome.asked} by asking a person)" if outcome.asked else ""
    if stopped == 0 and not unknown:
        if not outcome.expected:
            out.write(f"{head}not guarded: no hook stops any of {total} spellings\n")
            return False
        out.write(f"{head}EXPECTED, but no hook stops any of {total} spellings\n")
        return True
    out.write(f"{head}stopped {stopped} of {total}{by_asking}")
    if unknown:
        out.write(f", could not tell for {unknown}")
    if not outcome.passing:
        out.write("\n")
        return False
    out.write(f". {len(outcome.passing)} pass every hook:\n")
    out.writelines(f"{' ' * 16}{text}\n" for text in outcome.passing)
    return bool(stopped) or outcome.expected


def _verdict(
    spelling: Spelling, handlers: Sequence[Handler], probe: _Probe
) -> Decision:
    """What Claude Code would do with this call, of every hook that reaches it:
    precedence is ``deny`` > ``defer`` > ``ask`` > ``allow``, and ``defer`` is
    ignored outside ``claude -p``, so what an ask means beside a defer cannot
    be told, nor beside a hook that was not run (plugins' hooks, which reach
    every call for all this can tell), which might defer."""
    payload = probe.agent.call(spelling, probe.project, probe.mode)
    tally = probe.tally
    decisions: set[Decision] = set()
    for handler in handlers:
        if not handler.reaches(spelling.tool):
            continue
        if handler.untestable is not None or handler.where in tally.timed_out:
            decisions.add("unknown")
            continue
        tally.calls[handler.where] += 1
        answer = probe.run(handler, payload)
        if answer.timed_out:
            tally.timed_out.add(handler.where)
        if answer.timed_out and handler.capped:
            tally.cut_short.add(handler.where)
            if answer.failure == _LEFT_RUNNING:
                tally.left_running.add(handler.where)
            decision: Decision = "unknown"
            failure: str | None = None
        else:
            decision, failure = probe.agent.decide(answer, handler.blocks_on_failure)
        if failure is not None:
            tally.failures.setdefault(handler.where, Counter())[failure] += 1
        decisions.add(decision)
    if "block" in decisions:
        return "block"
    if "ask" in decisions:
        unsure = {"defer", "unknown"} & decisions or probe.unseen
        return "unknown" if unsure or probe.mode not in _ASK_STOPS else "ask"
    return "unknown" if "unknown" in decisions else "none"


def _decide(run: Run, blocks_on_failure: bool) -> tuple[Decision, str | None]:
    """One hook's answer to one call, and why it failed if it did; the table in
    the reference's "Exit code output" section."""
    failure = run.failure
    if failure is None and run.exit == 2:  # noqa: PLR2004 - Claude Code's block code
        return "block", None
    if run.stdout is None:
        return "unknown", None
    if failure is None:
        output = _output(run.stdout)
        if output is None:
            if run.exit == 0:
                return "none", None
            failure = f"exit {run.exit}"
        elif isinstance(output, str):
            failure = output
        elif run.exit != 0 and blocks_on_failure:
            failure = f"exit {run.exit}"
        else:
            decision = _decision(output)
            if decision is not None:
                return decision, None
            failure = "printed JSON that fails Claude Code's schema"
    return ("block" if blocks_on_failure else "none"), failure


def _output(stdout: bytes) -> dict[str, object] | str | None:
    """The JSON object Claude Code reads from ``stdout``; None for plain text,
    which decides nothing; a reason when it is a parse failure."""
    text = stdout.decode("utf-8", "replace").strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    invalid = "printed JSON Claude Code cannot read"
    try:
        value = json.loads(text, parse_constant=_not_json)
    except (ValueError, RecursionError):
        lines = [line for line in text.splitlines() if line.strip()]
        try:
            each = [json.loads(line, parse_constant=_not_json) for line in lines]
        except (ValueError, RecursionError):
            return invalid
        # Several lines of JSON are plain text unless one sets a field.
        sets_a_field = any(isinstance(v, dict) and v for v in each)
        return invalid if len(lines) < 2 or sets_a_field else None  # noqa: PLR2004
    return value if isinstance(value, dict) else invalid


def _decision(output: dict[str, object]) -> Decision | None:
    """The decision in a parsed object, or None when it fails the schema."""
    if not _fits_schema(output):
        return None
    specific = output.get("hookSpecificOutput")
    chosen = specific.get("permissionDecision") if isinstance(specific, dict) else None
    if isinstance(chosen, str):  # _fits_schema refused any other type
        return _PERMISSION.get(chosen)  # None: a decision Claude Code does not know
    legacy = output.get("decision")
    return _LEGACY[legacy] if isinstance(legacy, str) else "none"


def _fits_schema(output: dict[str, object]) -> bool:
    """As far as the reference states the schema: ``hookSpecificOutput`` names
    its event, a reason is a string, and each field every event accepts has
    its type. The whole object is checked before any decision in it is read."""
    specific = output.get("hookSpecificOutput")
    if "hookSpecificOutput" in output and (
        not isinstance(specific, dict)
        or specific.get("hookEventName") != "PreToolUse"
        or not _typed(specific, _SPECIFIC)
    ):
        return False
    legacy = output.get("decision")
    if "decision" in output and not (isinstance(legacy, str) and legacy in _LEGACY):
        return False
    return _typed(output, _UNIVERSAL)


def _typed(fields: dict[str, object], types: dict[str, type]) -> bool:
    return all(
        isinstance(fields[name], kind) for name, kind in types.items() if name in fields
    )


def _call(spelling: Spelling, project: Path, mode: str) -> bytes:
    """The PreToolUse input Claude Code sends for this spelling."""
    if spelling.tool == "Read":
        tool_input: dict[str, object] = {"file_path": str(project / spelling.text)}
    else:
        tool_input = {"command": spelling.text}
    return json.dumps(
        {
            "session_id": "paveo-test",
            "transcript_path": os.devnull,
            "cwd": str(project),
            "permission_mode": mode,
            "hook_event_name": "PreToolUse",
            "tool_name": spelling.tool,
            "tool_input": tool_input,
            "tool_use_id": "toolu_paveo_test",
        }
    ).encode()


def _failure_notes(
    handlers: Sequence[Handler], tally: _Tally, *, title: str
) -> list[str]:
    notes = [
        f"{handler.where} is not run: {handler.untestable}"
        for handler in handlers
        if handler.untestable is not None
    ]
    notes.extend(
        f"{handler.where} did not answer within paveo test's 10 s, before its own "
        "time limit, so what it decides cannot be told; it was not run again"
        + (
            ", and it could not be stopped (the system refused), so it was left running"
            if handler.where in tally.left_running
            else ""
        )
        for handler in handlers
        if handler.where in tally.cut_short
    )
    for handler in handlers:
        failures = tally.failures.get(handler.where)
        if not failures:
            continue
        failed = sum(failures.values())
        why = ", ".join(f"{reason} ({n})" for reason, n in failures.most_common())
        effect = (
            "it sets onFailure block, so each one was stopped (Claude Code "
            "v2.1.295 or later; an earlier one lets the call through)"
            if handler.blocks_on_failure
            else f"{title} lets a call through when its hook fails"
        )
        skipped = (
            "; it timed out, so it was not run for the calls after that"
            if handler.where in tally.timed_out - tally.cut_short
            else ""
        )
        notes.append(
            f"{handler.where} failed on {failed} of {tally.calls[handler.where]} "
            f"calls: {why}; {effect}{skipped}"
        )
    return notes


# --------------------------------------------------------------------------
# Reading the settings
# --------------------------------------------------------------------------


def _find(files: Sequence[Path], project: Path, named: bool) -> Found:
    found = Found()
    # A file named twice, the home folder being the project, is read once.
    for path in dict.fromkeys(Path(os.path.realpath(path)) for path in files):
        settings = _settings(path, named=named, unread=found.unread)
        if settings is None:
            continue
        found.read += 1
        disabled = settings.get("disableAllHooks")
        if isinstance(disabled, bool):  # the last file read is the highest
            found.disabled = (
                f"{path} sets disableAllHooks, so Claude Code runs none of its hooks."
                if disabled
                else None
            )
        permissions = settings.get("permissions")
        if isinstance(permissions, dict) and (
            permissions.get("deny") or permissions.get("ask")
        ):
            found.permission_rules = True
        mode = permissions.get("defaultMode") if isinstance(permissions, dict) else None
        if isinstance(mode, str):  # one of the seven: _settings refused others
            found.default_mode = "default" if mode == "manual" else mode
        enabled = settings.get("enabledPlugins")
        if isinstance(enabled, dict):
            found.plugins.update(enabled)
        _read_hooks(path, settings.get("hooks"), project=project, found=found)
    return found


def _read_hooks(path: Path, hooks: object, *, project: Path, found: Found) -> None:
    if hooks is None:
        return
    for g, group in enumerate(hook_groups(hooks).get("PreToolUse", [])):
        entries = group.get("hooks") if isinstance(group, dict) else None
        if not isinstance(group, dict) or not isinstance(entries, list):
            found.unread.append(f"{path}: PreToolUse[{g}] has no list of hooks")
            continue
        reaches, unreadable = reaches_tools(group.get("matcher"))
        for h, entry in enumerate(entries):
            where = f"{path}: PreToolUse[{g}].hooks[{h}]"
            if not isinstance(entry, dict):
                found.unread.append(f"{where} is not an object")
            elif runs_in_background(entry):
                found.notes.append(f"{where} runs in the background: it stops nothing")
            elif (
                same := json.dumps([group.get("matcher"), entry], sort_keys=True)
            ) in found.seen:
                continue
            else:
                found.seen.add(same)
                handler = _handler(where, entry, reaches, project=project)
                if handler is None:
                    found.unread.append(f"{where} has no command")
                elif unreadable:
                    found.handlers.append(_untestable(handler, unreadable))
                else:
                    found.handlers.append(handler)


def _handler(
    where: str,
    entry: dict[str, object],
    reaches: Callable[[str], bool],
    *,
    project: Path,
) -> Handler | None:
    """The hook ``entry`` as Claude Code would run it, or None when it has no
    command to run."""
    kind, program, args = entry.get("type"), entry.get("command"), entry.get("args")
    if kind != "command":
        why = f"it is a {_shown(str(kind))} hook, which is not run"
        return Handler(where, reaches, (), 0, blocks_on_failure=False, untestable=why)
    if not isinstance(program, str) or not program:
        return None
    if args is None:
        # The reference: shell form runs `sh -c` on macOS and Linux, and the
        # `shell` field chooses bash or PowerShell explicitly.
        shell = "bash" if entry.get("shell") == "bash" else "/bin/sh"
        argv: tuple[str, ...] = (shell, "-c", program)
    elif isinstance(args, list) and all(isinstance(a, str) for a in args):
        # Exec form: placeholders are replaced as plain strings, with no shell.
        argv = tuple(
            word.replace("${CLAUDE_PROJECT_DIR}", str(project))
            for word in (program, *args)
        )
    else:
        return None
    timeout = entry.get("timeout", _DEFAULT_TIMEOUT_S)
    # Compared, never converted: a 400-digit integer is valid JSON and too
    # large for a float.
    if not (
        isinstance(timeout, int | float)
        and not isinstance(timeout, bool)
        and timeout > 0
    ):
        # Claude Code's schema decides what a timeout of the wrong type does.
        handler = Handler(where, reaches, argv, _TIMEOUT_S, blocks_on_failure=False)
        return _untestable(handler, "its timeout is not a positive number")
    handler = Handler(
        where,
        reaches,
        argv,
        float(min(timeout, _TIMEOUT_S)),
        blocks_on_failure=entry.get("onFailure") == "block",
        capped=timeout > _TIMEOUT_S,
    )
    reason = _not_run(entry)
    return handler if reason is None else _untestable(handler, reason)


def _not_run(entry: dict[str, object]) -> str | None:
    """Why a command hook cannot be run as Claude Code would run it."""
    if "if" in entry:
        return "it has an `if` filter, which is not read yet"
    if entry.get("args") is None and entry.get("shell") == "powershell":
        return "it asks for PowerShell"
    return None


def _maybe_there(path: Path) -> bool:
    """Whether ``path`` may exist: a folder that cannot be read hides it, and
    Rule 3 says that is not the same as nothing there."""
    try:
        os.stat(path)
    except OSError as error:
        # Only "not there" is absence; one that cannot be looked at may be there.
        return not isinstance(error, FileNotFoundError | NotADirectoryError)
    return True


def user_profiles(folder: Path, name: str) -> list[Path]:
    """``name`` in each user's folder under ``folder``, for ``_maybe_there``
    to judge. A folder that cannot be listed is returned itself, so that what
    it may hide is not taken for nothing (``Path.glob`` would skip it)."""
    try:
        users = [entry.path for entry in os.scandir(folder) if entry.is_dir()]
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError:
        return [folder]
    return [Path(user) / name for user in users]


def _untestable(handler: Handler, why: str) -> Handler:
    return replace(handler, untestable=why)


def _named(spelling: Spelling) -> str:
    """A spelling as the report shows it: a command as typed, a file read by
    the Read tool as such."""
    return (
        spelling.text if spelling.tool == "Bash" else f"{spelling.tool} {spelling.text}"
    )


CLAUDE_CODE = Agent(
    title="Claude Code",
    nothing_found="no Claude Code settings file found",
    tools=frozenset({"Bash", "Read"}),
    find=_find,
    call=_call,
    decide=_decide,
)


# --------------------------------------------------------------------------
# Running a hook
# --------------------------------------------------------------------------


def run_handler(  # noqa: PLR0913 - the clock and pause are injected (Rule 14)
    handler: Handler,
    payload: bytes,
    *,
    project: Path,
    env: Mapping[str, str],
    read_stderr: bool = False,
    clock: Callable[[], float] = time.monotonic,
    pause: Callable[[float], None] = time.sleep,
) -> Run:
    """Run one hook with ``payload`` on stdin, as Claude Code would.
    ``read_stderr``: learn whether it wrote to stderr, which Codex needs;
    otherwise stderr goes nowhere, as before, so a child the hook leaves
    holding stderr open does not make every call wait out the grace.

    The call goes into a pipe before the hook starts and the pipe is closed, so
    a hook that never reads stdin cannot leave the write waiting: every payload
    here is far below the smallest pipe buffer. The hook leads its own process
    group, so one still running at its limit is ended with everything it
    started. One that has exited is never signalled: once it is reaped, its
    group id can belong to someone else."""
    call_in, call_out = os.pipe()
    answer_in, answer_out = os.pipe()
    said_in, said_out = os.pipe() if read_stderr else (-1, subprocess.DEVNULL)
    try:
        os.write(call_out, payload)
    finally:
        os.close(call_out)
    try:
        process = subprocess.Popen(  # noqa: S603 - the project's own hook, the point of paveo test
            handler.argv,
            stdin=call_in,
            stdout=answer_out,
            stderr=said_out,
            cwd=project,
            env=dict(env),
            start_new_session=True,
        )
    except (OSError, ValueError):  # ValueError: a NUL byte in the command
        os.close(answer_in)
        if read_stderr:
            os.close(said_in)
        return Run(None, b"", "could not start")
    finally:
        os.close(call_in)
        os.close(answer_out)
        if read_stderr:
            os.close(said_out)
    try:
        answer = _answer(
            process, answer_in, said_in, handler.timeout, clock=clock, pause=pause
        )
    finally:
        os.close(answer_in)
        if read_stderr:
            os.close(said_in)
        stopped = _stop(process)
    if not stopped:
        return Run(None, b"", _LEFT_RUNNING, timed_out=True)
    return answer


def _stop(process: subprocess.Popen[bytes]) -> bool:
    """End a hook still running, with everything it started; ``False`` when the
    system refuses. ``killpg`` fails with EPERM only when no process in the
    group may be signalled (the hook itself runs as another user, setuid), and
    waiting for it could take forever, so it is left running and said, never a
    traceback (D102). One such process among others that die is not seen.

    Left unreaped on purpose: reaping is waiting."""
    if process.poll() is not None:
        return True
    try:
        # Not reaped, so the group id is still the hook's own. It can hold
        # nothing but processes that have exited and wait to be reaped.
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        return False
    process.wait()
    return True


def _answer(  # noqa: PLR0913 - the clock and pause are injected (Rule 14)
    process: subprocess.Popen[bytes],
    answer: int,
    said: int,  # -1: stderr is not read
    timeout: float,
    *,
    clock: Callable[[], float],
    pause: Callable[[float], None],
) -> Run:
    """The hook's exit code, what it printed, and whether it wrote to stderr,
    read until it exits and both end, or ``_EXIT_GRACE_S`` after it exits if
    something it left running holds them open. Stderr is read only to keep
    its pipe from filling, and dropped. The clock is injected (Rule 14), so a
    test can run past the deadline without waiting for it."""
    deadline = clock() + timeout
    output = bytearray()
    too_long = wrote_stderr = False
    exited_at: float | None = None
    with selectors.DefaultSelector() as selector:
        selector.register(answer, selectors.EVENT_READ)
        if said >= 0:
            selector.register(said, selectors.EVENT_READ)
        while True:
            now = clock()
            reading = bool(selector.get_map())
            if exited_at is None and process.poll() is not None:
                exited_at = now
            if exited_at is not None and (
                not reading or now >= exited_at + _EXIT_GRACE_S
            ):
                break
            if now >= deadline:
                if exited_at is None:
                    return Run(None, b"", _TIMED_OUT, timed_out=True)
                break
            if not reading:
                pause(0.01)
                continue
            for key, _events in selector.select(min(deadline - now, 0.05)):
                chunk = os.read(key.fd, 1 << 16)
                if not chunk:
                    selector.unregister(key.fd)
                elif key.fd == said:
                    # Codex trims stderr before it reads a reason.
                    wrote_stderr = wrote_stderr or bool(
                        chunk.decode("utf-8", "replace").strip()
                    )
                elif len(output) + len(chunk) > _MAX_OUTPUT:
                    too_long = True
                else:
                    output += chunk
    return Run(
        process.returncode,
        None if too_long else bytes(output),
        wrote_stderr=wrote_stderr,
    )
