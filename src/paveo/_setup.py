"""Wiring the guard into a coding agent, and checking the wiring (D49, D50, D57).

``paveo init <agent>`` sets a project up in one command, and ``paveo guard
<agent> --selftest`` checks that every configured hook really refuses. Both exist
for the same reason: every agent treats a hook it cannot run as one with no
objection, so a seatbelt that is set up wrong looks exactly like one that works.
What differs between the agents is in ``_harnesses``, plus the few branches here
named for the agent they serve.

**Only a plain paveo command is ever run** (``_plain_command``). Hook files
ship with repositories, and running whatever one says would skip the trust
prompt the agent shows before it runs a cloned repository's hooks.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from importlib import resources
from pathlib import Path
from typing import TextIO

from ._doctor import misshapen, parse_settings, reaches_tools, runs_in_background
from ._harnesses import (
    CLAUDE_CODE,
    CODEX,
    CURSOR,
    Event,
    Harness,
    for_policy,
    real_paths,
)
from ._licence import DEVELOPER, LICENCE_FILE, Plan, apply, plan_in
from ._policy_document import load_document, load_file
from .audit import utc_now
from .errors import ConfigError
from .policy import Policy, Recall

# What the first call of any session remembers.
_FRESH = Recall(now=0.0, footprints=frozenset(), calls=())

# Every agent's contract: exit 2 blocks. Anything else may not.
BLOCK = 2
# The guard refuses a call it has not decided by then. Well under any hook
# timeout a person would set, and far over a normal decision (about 60 ms, most
# of it starting Python and importing; tests/overhead.py).
GUARD_DEADLINE_S = 4.0
# A hook timeout shorter than this can end before the guard answers: the 4 s
# timer starts only once Python has started and imported (D103).
_MIN_HOOK_TIMEOUT_S = 10.0
POLICY = "policy.json"
# `--selftest` sends a call the guard cannot read and expects this refusal back:
# only a guard that started, loaded its policy and opened its log gets this far.
NOT_A_CALL = "is not a tool call this guard reads"
_SELFTEST_CALL = b'{"paveo_selftest": true}'
_SELFTEST_TIMEOUT_S = 30
_PROJECT_VARIABLE = re.compile(r"\$\{CLAUDE_PROJECT_DIR\}|\$CLAUDE_PROJECT_DIR\b")
_OPERATORS = ";&|<>()"
_GUARD_OPTIONS = frozenset({"--dir", "--agent"})
# Every option `paveo guard` takes, for reading an abbreviation as argparse does.
_GUARD_ALL_OPTIONS = (
    "--dir",
    "--agent",
    "--policy",
    "--selftest",
    "--settings",
    "--help",
)
# The log, the stop file and a licence key are this machine's, never the
# repository's: a key committed to a public repository is anyone's (D62).
# At most this many agent names from a policy file go into a refusal.
_LISTED = 5
_IGNORED = "audit.jsonl\naudit.jsonl.anchor\nstop\nlicence.key\n"


@dataclass(frozen=True)
class _Hook:
    """One configured hook that runs the paveo guard, and where it was found."""

    source: Path
    event: str
    group: int
    handler: int
    matcher: object
    command: str
    fail_closed: bool
    # What in the entry keeps Claude Code from asking it about every call, read
    # whatever the agent but judged for Claude Code only (``_skipped``).
    wiring: tuple[str, ...]


def init(  # noqa: PLR0913 - keyword-only; the files and the process runner are injected
    harness: Harness,
    *,
    project: Path,
    program: Path,
    hook_files: Sequence[Path],
    run: Callable[[Sequence[str], Path], tuple[int, str]],
    out: TextIO,
    global_files: Sequence[Path] = (),
    now: Callable[[], datetime] = utc_now,
) -> int:
    """Set ``project`` up for the guard, then prove it works. Safe to run twice.

    Writes ``.paveo/policy.json`` (the agent's starter policy, unless one is
    there), ``.paveo/.gitignore``, and the hook in the agent's folder, naming
    this machine's ``paveo`` by its full path. ``hook_files`` are every file the
    agent reads hooks from, so a paveo hook in any of them is found.

    **One policy for every agent in the project** (D88). A policy already there
    that does not name this agent gets the agent's starter rules added, so
    Claude Code, Codex and Cursor in one project follow one file. Before, the
    second agent's init kept a policy that named only the first, and every call
    from the second was refused as an unknown agent. If the plan's number of
    agents would leave the new one refused, init says so and writes nothing.

    **It never says the seatbelt is on without having checked** (/code-review,
    D50): the hook must reach every tool the policy governs, the self-test must
    pass, and the policy itself must refuse ``rm -rf`` and an edit to its own
    folder. **It writes nothing until every check that can run first has
    passed**, and never through a symbolic link: a cloned repository can ship
    one where a file is expected (/security-review). What it cannot check, it
    says: Codex's trust in the hook is given in Codex, after init.

    **Never where the hook would be global** (D75): not when the project's
    agent folder is the one holding ``global_files``, the hook files the agent
    reads in every project. That is the home folder for every agent, and a
    ``CODEX_HOME`` inside the project for Codex. A person meant to guard one
    project, not all of them.
    """
    home, folder = project / ".paveo", project / harness.folder
    target = folder / harness.hook_file
    refusal = _cannot_start(
        harness,
        program=program,
        folder=folder,
        global_files=global_files,
        touched=(
            home,
            home / POLICY,
            home / ".gitignore",
            home / LICENCE_FILE,
            target,
            folder / ".gitignore",
        ),
    )
    if refusal is not None:
        out.write(f"FAIL {refusal}\n")
        return 1
    plan = _plan_hook(harness, target, hook_files, out)
    if plan is None:
        return 1
    today = now().date()
    try:
        # A key is read only when a policy is already there to judge against:
        # the starter names one agent, which every plan covers.
        licence = plan_in(home, today=today) if (home / POLICY).exists() else DEVELOPER
        addition = _agent_addition(
            harness, home / POLICY, plan.agents, licence, today=today
        )
    except ConfigError as e:
        addition = _Refusal(f"{e} Nothing was changed.")
    if isinstance(addition, _Refusal):
        out.write(f"FAIL {addition.why}\n")
        return 1
    try:
        _write_files(
            harness,
            project=project,
            home=home,
            target=target,
            plan=plan,
            program=program,
            out=out,
            addition=addition,
        )
    except OSError as e:
        out.write(f"FAIL could not write the setup ({type(e).__name__}: {e})\n")
        return 1
    out.write("\nchecking it:\n")
    checked = selftest(
        harness,
        hook_files=[path for path in hook_files if path.exists()],
        project=project,
        run=run,
        # The one program it will run is the one that wrote the hook.
        confirm=lambda named: Path(os.path.abspath(named)) == program,
        out=out,
        global_files=global_files,
        program=str(program),
    )
    weakness = _policy_weakness(
        harness, home / POLICY, plan.agents, licence, today=today
    )
    if weakness is not None:
        out.write(f"FAIL {home / POLICY} {weakness}\n")
        return 1
    if checked == 0:
        out.write(
            f"\n{harness.last_step}\nPanic button, from this folder: "
            f"{shlex.quote(str(program))} stop   (undo: ... resume)\n"
        )
    return checked


def _cannot_start(
    harness: Harness,
    *,
    program: Path,
    folder: Path,
    global_files: Sequence[Path],
    touched: Sequence[Path],
) -> str | None:
    """Why init must not start here or with this program, or ``None``.
    ``touched`` are the files and folders it would write or read: none may be a
    link, so a cloned repository cannot point one at another file."""
    if any(_same_place(folder, each.parent) for each in global_files):
        return (
            f"{folder} is where {harness.name} keeps the settings it reads in "
            f"every project (is this your home folder?), so a hook set up here "
            f"would guard all of them. cd into the project you want guarded and "
            f"run init there. Nothing was changed."
        )
    if program.name != "paveo" or not program.is_file():
        return (
            f"run this through the installed `paveo` command, so the hook can name "
            f"it: e.g. /path/to/venv/bin/paveo init {harness.name}"
        )
    links = [path for path in (folder, *touched) if path.is_symlink()]
    if links:
        return (
            f"{links[0]} is a symbolic link. init writes only real files inside "
            f"the project, so nothing was changed."
        )
    return None


class _HookPlan:
    """What ``init`` will do to the hook file, decided before any write: add a
    hook for each of ``events`` to ``document``, or leave every file as it is."""

    def __init__(
        self,
        document: dict[str, object] | None,
        *,
        existed: bool,
        agents: frozenset[str],
        events: frozenset[str] = frozenset(),
    ) -> None:
        self.document = document  # None: every event already has a paveo hook
        self.existed = existed
        self.agents = agents  # every agent id a hook will judge calls as
        self.events = events


def _plan_hook(
    harness: Harness, target: Path, hook_files: Sequence[Path], out: TextIO
) -> _HookPlan | None:
    """Read every hook file first. ``None`` means refuse, having said why."""
    documents = _read_hook_files(harness, target, hook_files, out)
    if documents is None:
        return None
    for path, document in documents.items():
        if harness is CLAUDE_CODE and document.get("disableAllHooks") is True:
            out.write(
                f"FAIL {path} sets disableAllHooks, so no hook runs at all. Remove "
                f"it, then run init again.\n"
            )
            return None
    found = _hooks_for(harness, documents, target.parent.parent / ".paveo", out)
    unguarded = _unguarded(harness, found)
    for reason in unguarded:
        out.write(f"FAIL {reason}\n")
    if unguarded:
        return None
    # Only the events with no paveo hook yet get one: Cursor has two, and a
    # person who removed one must get it back by running init again.
    missing = frozenset(
        event.name
        for event in harness.events
        if not any(hook.event == event.name for hook in found)
    )
    # A kept hook may name its own --agent: the policy is asked as that agent,
    # or a repository could pair a strict `codex` with a lax one the hook really
    # uses (/security-review, D57).
    agents = frozenset(_agent(harness, hook) for hook in found)
    if found:
        out.write("kept      the paveo hook already configured\n")
    if not missing:
        return _HookPlan(None, existed=True, agents=agents)
    document = documents[target]
    hooks = document.setdefault("hooks", {})
    # Cursor's file carries a version; one this does not know is not edited.
    if (
        (harness.flat and document.setdefault("version", 1) != 1)
        or not isinstance(hooks, dict)
        or not all(
            isinstance(hooks.setdefault(event.name, []), list)
            for event in harness.events
        )
    ):
        out.write(
            f"FAIL {target} is not a hook file this can safely edit, so nothing "
            f"was changed. Add the hook by hand (README).\n"
        )
        return None
    return _HookPlan(
        document,
        existed=target.exists(),
        agents=agents | {harness.name},
        events=missing,
    )


def _hooks_for(
    harness: Harness, documents: dict[Path, dict[str, object]], home: Path, out: TextIO
) -> list[_Hook]:
    """The paveo hooks in ``documents`` that read the policy in ``home``."""
    found = []
    for path, document in documents.items():
        for hook in _paveo_hooks_in(harness, path, document):
            if _reads(hook, home):
                found.append(hook)
            else:
                out.write(
                    f"note      the paveo hook in {path} guards another folder "
                    f"than {home}; one is added for this project\n"
                )
    return found


@dataclass(frozen=True)
class _Refusal:
    """Why init must stop before writing anything."""

    why: str


def _agent_addition(
    harness: Harness,
    policy: Path,
    agents: frozenset[str],
    licence: Plan,
    *,
    today: date,
) -> str | _Refusal | None:
    """The policy's new text with this agent's starter rules added, ``None`` if
    nothing is to be added, or why init must stop. **Decided before anything is
    written** (/code-review, D88), on the policy as the guard will load it.

    A policy already there is loaded with ``load_file``, so its size limit and
    validation apply before a cloned repository's file is parsed here; one that
    does not load, fails open, or would leave an agent a hook judges as unnamed
    or past the plan, stops init with nothing changed. Only an agent a hook will
    judge calls as by the agent's own name is added.
    """
    if not policy.exists():
        return None
    loaded = _judged_before_writing(harness, policy, agents, licence, today=today)
    if isinstance(loaded, _Refusal):
        return loaded
    if harness.name not in agents or harness.name in loaded.agents:
        return None
    return _merged(harness, policy, loaded, licence)


def _judged_before_writing(
    harness: Harness,
    policy: Path,
    agents: frozenset[str],
    licence: Plan,
    *,
    today: date,
) -> Policy | _Refusal:
    """The policy as the guard will load it, or why init must not go on."""
    try:
        loaded = apply(load_file(policy), licence, today=today)
    except ConfigError as e:
        return _Refusal(
            f"{policy} does not load ({e}). Nothing was changed: fix it, or remove "
            f"it and run init again for the starter policy."
        )
    if loaded.fail_open:
        return _Refusal(
            f"{policy} sets defaults.fail_open, so a call whose decision cannot be "
            f"recorded is let through. init will not call that a seatbelt; "
            f"nothing was changed."
        )
    for agent in sorted(agents):
        if agent not in loaded.agents and agent != harness.name:
            return _Refusal(
                f"{policy} names no agent {agent!r}, which a paveo hook here judges "
                f"calls as, so every such call would be refused. Nothing was changed."
            )
        if agent in loaded.over_plan:
            return _Refusal(_past_plan(policy, loaded.agents, licence, harness))
    return loaded


def _merged(
    harness: Harness, policy: Path, loaded: Policy, licence: Plan
) -> str | _Refusal:
    """``policy`` with the agent's starter rules appended, validated whole."""
    starter = json.loads(
        resources.files("paveo")
        .joinpath("starters", f"{harness.name}.json")
        .read_text(encoding="utf-8")
    )
    added = starter["agents"]
    if licence.agents is not None and len(loaded.agents) + len(added) > licence.agents:
        return _Refusal(_past_plan(policy, loaded.agents, licence, harness))
    # load_file has refused anything over its size limit or not valid, so this
    # second read parses the same small document; the merged text is validated
    # whole, so what is written is a policy that loads.
    document = json.loads(policy.read_text(encoding="utf-8"))
    document["agents"] = [*document["agents"], *added]
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    try:
        load_document(json.loads(text), source=str(policy))
    except ConfigError as e:
        return _Refusal(
            f"{policy} with {harness.title}'s starter rules added does not load "
            f"({e}). Nothing was changed."
        )
    return text


def _past_plan(
    policy: Path, names: Mapping[str, object], licence: Plan, harness: Harness
) -> str:
    """The refusal for an agent the plan would leave refused every call. Names
    come from the file, so they are quoted and only the first few listed."""
    listed = ", ".join(repr(name) for name in list(names)[:_LISTED])
    more = "" if len(names) <= _LISTED else f" and {len(names) - _LISTED} more"
    return (
        f"{policy} names {len(names)} agent(s) ({listed}{more}), and the "
        f"{licence.name} plan covers {licence.agents} per policy, so "
        f"{harness.title} would have every call refused. Nothing was changed. A "
        f"paid plan covers more agents (https://paveo.dev/#pricing)."
    )


def _read_hook_files(
    harness: Harness, target: Path, hook_files: Sequence[Path], out: TextIO
) -> dict[Path, dict[str, object]] | None:
    """Every hook file the agent loads, by path, or ``None`` to refuse, having
    said why. A Claude Code file it rejects whole is left out, said in a note;
    if it is the target, init refuses."""
    documents: dict[Path, dict[str, object]] = {}
    for path in dict.fromkeys([*hook_files, target]):
        document = _read_settings(path, out)
        if document is None:
            return None
        wrong = misshapen(document) if harness is CLAUDE_CODE else None
        if wrong is not None and _same_place(path, target):
            # A guard written into a file Claude Code rejects whole would never
            # run, while init said it was on (locked decision #4).
            out.write(
                f"FAIL {path}: its {wrong} has the wrong type, and Claude Code "
                "rejects such a file whole, so a guard written there would never "
                "run, and nothing was changed: fix it, then run init again.\n"
            )
            return None
        if wrong is not None and document.get("disableAllHooks") is True:
            # Whether Claude Code drops the file or applies it, read the safe
            # way both times: its hooks do not count, and its switch does.
            out.write(
                f"FAIL {path} sets disableAllHooks, so no hook may run at all. "
                f"Remove it, then run init again.\n"
            )
            return None
        # Claude Code drops only that file: its hooks, a paveo one included,
        # do not run, so none of them counts as already there. A file that
        # cannot be parsed at all still refuses, above: Paveo and Claude Code
        # may disagree on it, and then its disableAllHooks could apply unseen.
        if not _rejected_whole(harness, path, document, out):
            documents[path] = document
    return documents


def _read_settings(path: Path, out: TextIO) -> dict[str, object] | None:
    """A hook file as a dict, ``{}`` if absent, ``None`` (said why) if it
    cannot be read: rewriting what could not be read would drop settings."""
    try:
        document = parse_settings(path)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        out.write(
            f"FAIL {path} could not be read ({type(e).__name__}), so nothing was "
            f"changed.\n"
        )
        return None
    if not isinstance(document, dict):
        out.write(f"FAIL {path} is not a settings object, so nothing was changed.\n")
        return None
    return document


def _write_files(  # noqa: PLR0913 - one call, each path a different file it writes
    harness: Harness,
    *,
    project: Path,
    home: Path,
    target: Path,
    plan: _HookPlan,
    program: Path,
    out: TextIO,
    addition: str | None = None,
) -> None:
    home.mkdir(mode=0o700, exist_ok=True)
    policy = home / POLICY
    if addition is not None:
        _replace(policy, addition)
        out.write(
            f"added     {harness.name!r} to {policy}, with its starter rules: one "
            f"policy now covers every agent in this project\n"
        )
    elif policy.exists():
        out.write(f"kept      {policy} (already there)\n")
    else:
        starter = resources.files("paveo").joinpath("starters", f"{harness.name}.json")
        _create(policy, starter.read_text(encoding="utf-8"))
        out.write(f"wrote     {policy} (the starter policy: edit it to fit)\n")
    if not (home / ".gitignore").exists():
        _create(home / ".gitignore", _IGNORED)
    # A project set up before keys existed gets the line on its next init (D62).
    keep_out_of_git(home / ".gitignore", "licence.key")
    target.parent.mkdir(exist_ok=True)
    # The hook names this machine's paths, so it is never the repository's.
    # Claude Code's personal settings file is personal by definition; the others'
    # hooks.json may be a file the team shares, so it is ignored only if new.
    if harness is CLAUDE_CODE or not plan.existed:
        keep_out_of_git(target.parent / ".gitignore", target.name)
    if plan.document is None:
        return
    if plan.existed and harness is not CLAUDE_CODE:
        out.write(
            f"note      {target.relative_to(project)} was already there; the line "
            f"added names this machine's paveo, so keep it out of any commit.\n"
        )
    hooks = plan.document["hooks"]
    assert isinstance(hooks, dict)  # checked by _plan_hook  # noqa: S101
    # Claude Code names the project in a variable. Codex has none, and runs a
    # hook from wherever the session started, so the others get the full path.
    directory = (
        '"$CLAUDE_PROJECT_DIR/.paveo"'
        if harness is CLAUDE_CODE
        else shlex.quote(os.path.abspath(home))
    )
    command = f"{shlex.quote(str(program))} guard {harness.name} --dir {directory}"
    for event in (event for event in harness.events if event.name in plan.events):
        entry: dict[str, object]
        if harness.flat:
            entry = {"command": command, "failClosed": True}
        else:
            entry = {"hooks": [{"type": "command", "command": command}]}
        if event.matcher is not None:
            entry["matcher"] = event.matcher
        hooks[event.name].append(entry)
    _replace(target, json.dumps(plan.document, indent=2) + "\n")
    out.write(f"added     the hook in {target.relative_to(project)}\n")


def _create(path: Path, text: str) -> None:
    """Create a file that must not exist yet, never through a link."""
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def keep_out_of_git(ignore: Path, name: str) -> None:
    """Keep ``name`` out of git, as Claude Code does when it creates its
    personal settings file itself: the hook in it names this machine's paths."""
    try:
        present = ignore.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        _create(ignore, f"{name}\n")
        return
    if name not in present:
        descriptor = os.open(ignore, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(f"\n{name}\n")


def _replace(target: Path, text: str) -> None:
    """Write ``target`` whole, or not at all.

    The draft gets a fresh, unpredictable name from ``mkstemp``, which creates it
    exclusively: a fixed draft name was a place a cloned repository could put a
    link to the user's global settings and have them overwritten
    (/security-review, D50). Renamed over the target, so a crash never leaves a
    half-written settings file. The target's permissions are kept.
    """
    try:
        mode = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        mode = 0o600
    descriptor, draft = tempfile.mkstemp(dir=target.parent, prefix=".paveo-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(draft, mode)
        os.replace(draft, target)
    except BaseException:
        Path(draft).unlink(missing_ok=True)
        raise


def _policy_weakness(
    harness: Harness,
    policy: Path,
    agents: frozenset[str],
    licence: Plan,
    *,
    today: date,
) -> str | None:
    """Why this policy would not do what init is about to promise, or ``None``.

    A policy already in the project was kept unread, and a cloned repository can
    ship a permissive one (/code-review, D50), so it is asked, not trusted: as
    every agent a hook will judge calls as. **An agent it does not name, or one
    past the plan's number, fails too**: either is refused every call, which the
    probes alone would read as a policy that refuses what it should (D88).
    """
    try:
        loaded = apply(load_file(policy), licence, today=today)
    except ConfigError as e:
        return f"does not load: {e}"
    for agent in sorted(agents):
        if agent not in loaded.agents:
            return (
                f"names no agent {agent!r}, so {harness.title} would have every "
                f"call refused. Add it, or remove this file and run init again."
            )
        if agent in loaded.over_plan:
            return (
                f"names {agent!r} past the {licence.name} plan's {licence.agents} "
                f"agents, so {harness.title} would have every call refused."
            )
        if loaded.shadows(agent):
            return (
                f"is in shadow mode for {agent!r}, so it records refusals and "
                f"enforces none."
            )
        for tool, arguments, what in harness.probes:
            judged = for_policy(harness, tool, arguments)
            # An agent mostly writes a path in full, as the guard's reading of
            # the file it really reaches does (D79), so the edit is asked about
            # both ways and must be refused both ways.
            real = real_paths(
                harness, tool, judged, os.path.abspath(policy.parent.parent)
            )
            for reading in (judged,) if real is None else (judged, real):
                # As the first call of a session sees it: remembering nothing,
                # not unable to remember, which would refuse for the wrong reason
                # (D59). Nor does `requires_unmet` count: one earlier call lifts
                # it, so it does not refuse the command (/security-review, D59).
                denial = loaded.evaluate_tool(agent, tool, reading, recall=_FRESH)
                if denial is not None and denial.reason != "requires_unmet":
                    continue
                return (
                    f"does not refuse {what} for {agent!r}. If that is deliberate, "
                    f"the seatbelt is off by your choice; to start from the starter "
                    f"policy, remove this file and run init again."
                )
    return None


def _unguarded(harness: Harness, hooks: Sequence[_Hook]) -> list[str]:
    """For each event with a paveo hook, why the hooks the agent waits for do not
    reach every call the policy governs; empty when they do. Running a hook by
    hand cannot show this, since it refuses the same however it is wired, so
    ``init`` and ``--selftest`` both read it here. The hooks are judged
    together: a skipped extra beside a sound hook costs nothing, and two whose
    matchers split the tools between them guard every call. Every fault of every
    hook is named at once, so one run shows every fix. An event with no paveo
    hook at all is ``_agent_problems``'s to name."""
    reasons: list[str] = []
    for event in harness.events:
        mine = [hook for hook in hooks if hook.event == event.name]
        sound = [hook for hook in mine if not hook.wiring]
        if not mine or _reached(harness, sound, event):
            continue
        reasons.extend(
            f"the paveo {hook.event} hook in {hook.source} {fault}"
            for hook in mine
            for fault in hook.wiring
        )
        missed = [
            tool
            for tool in event.reaches or ()
            if not any(_covers(harness, hook.matcher, (tool,)) for hook in sound)
        ]
        where = ", ".join(sorted({str(hook.source) for hook in mine}))
        reasons.append(
            f"no paveo {event.name} hook {harness.title} waits for in {where} "
            + (
                f"sees {', '.join(missed)}: add them to its matcher, as in "
                f"{event.matcher!r}."
                if missed
                else "sees every call: set its matcher to "
                f"{event.matcher or 'nothing'!r}."
            )
        )
    return reasons


def _reached(harness: Harness, hooks: Sequence[_Hook], event: Event) -> bool:
    """Whether every call the policy governs reaches at least one of ``hooks``."""
    if event.reaches is None:
        return any(_covers(harness, hook.matcher, None) for hook in hooks)
    return all(
        any(_covers(harness, hook.matcher, (tool,)) for hook in hooks)
        for tool in event.reaches
    )


def _wiring(entry: dict[str, object]) -> tuple[str, ...]:
    """Every fault in a Claude Code hook entry that keeps its command from
    deciding every call, from the hooks reference, each with its fix; empty when
    there is none. Codex's and Cursor's fields are unread, so they are not
    judged (Rule 3).

    The same rules as ``paveo test``'s ``_handler`` where both read a field;
    where they differ it is on purpose: ``paveo test`` reports a hook it cannot
    run as untestable, this fails one Paveo itself must run."""
    kind = entry.get("type")
    timeout = entry.get("timeout")
    too_short = timeout is not None and not (
        isinstance(timeout, int | float)
        and not isinstance(timeout, bool)
        and timeout >= _MIN_HOOK_TIMEOUT_S
    )
    background = "async" if entry.get("async") is True else "asyncRewake"
    checks = (
        (
            kind != "command",
            f"has {f'the type {kind!r}' if 'type' in entry else 'no type'}, so "
            f'Claude Code never runs its command. Set "type": "command".',
        ),
        (
            entry.get("args") is not None,
            'has "args", so Claude Code runs the command as one program name. '
            'Remove "args".',
        ),
        (
            entry.get("args") is None and entry.get("shell") == "powershell",
            'asks for PowerShell, which the guard does not run on. Remove "shell".',
        ),
        (
            runs_in_background(entry),
            f'runs in the background ("{background}"), so Claude Code does not '
            f'wait for it. Remove "{background}" from the hook.',
        ),
        (
            "if" in entry,
            'has an "if" filter, so Claude Code runs it only for some calls. '
            'Remove "if" from the hook.',
        ),
        (
            too_short,
            f"has the timeout {timeout!r}: Claude Code lets the call through if the "
            f"guard has not answered by then, and with Python starting it can need "
            f"more than its own {GUARD_DEADLINE_S:g} s. Set it to "
            f"{_MIN_HOOK_TIMEOUT_S:g} or more, or remove it.",
        ),
    )
    return tuple(why for wrong, why in checks if wrong)


def _covers(harness: Harness, matcher: object, reaches: Sequence[str] | None) -> bool:
    """Whether a hook's matcher reaches every tool in ``reaches`` (``None``:
    every call). Claude Code's is read as ``paveo test`` reads it (names, else a
    regular expression); Codex's and Cursor's are a regular expression,
    searched. Absent, empty and ``*`` match everything."""
    if matcher is None or matcher in (
        "",
        "*",
    ):  # a tuple: a list matcher is not hashable
        return True
    if reaches is None or not isinstance(matcher, str):
        return False
    if harness.names_matcher:
        reached, unread = reaches_tools(matcher)
        return unread is None and all(map(reached, reaches))
    try:
        pattern = re.compile(matcher)
    except re.error:
        return False
    return all(pattern.search(name) for name in reaches)


def _reads(hook: _Hook, home: Path) -> bool:
    """Whether a hook reads the policy in ``home``. A global hook file can hold
    a paveo hook with another project's folder in it: that one guards there, not
    here (/code-review, D57). ``--dir`` and ``--policy`` are read as argparse
    reads them (``_option``), with ``$CLAUDE_PROJECT_DIR`` filled in, so
    ``--dir=/elsewhere``, ``--di`` and ``$CLAUDE_PROJECT_DIR/sub/.paveo`` are
    judged by the folder they name (the third review of 0.1.10). Judged on the
    words, never by running anything; a hook with shell syntax is never run.
    """
    words = _words(hook.command, home.parent)
    folder = _option(words, "--dir")
    policy = _option(words, "--policy")
    if folder is not None and (home.parent / folder).resolve() != home.resolve():
        return False
    return (
        policy is None or (home.parent / policy).resolve() == (home / POLICY).resolve()
    )


def _agent(harness: Harness, hook: _Hook) -> str:
    """The agent id a hook judges calls as: its ``--agent``, else the agent's
    name. Read from the words; a hook with shell syntax is never run anyway."""
    return _option(_words(hook.command, Path(".")), "--agent") or harness.name


def _option(words: Sequence[str], name: str) -> str | None:
    """The value argparse gives the guard's option ``name``: the last one given,
    whole, after ``=``, or as an abbreviation no other guard option shares."""
    value = None
    for index, word in enumerate(words):
        if word == "--":  # what follows is no option
            break
        flag, equals, rest = word.partition("=")
        named = (
            flag.startswith("--")
            and flag != "--"
            and name.startswith(flag)
            and sum(option.startswith(flag) for option in _GUARD_ALL_OPTIONS) == 1
        )
        if named and equals:
            value = rest
        elif named and index + 1 < len(words):
            value = words[index + 1]
    return value


def _paveo_hooks_in(
    harness: Harness, source: Path, document: dict[str, object]
) -> list[_Hook]:
    """Every hook in one hook file that runs ``paveo guard <agent>``."""
    hooks = document.get("hooks")
    found: list[_Hook] = []
    for event in harness.events:
        groups = hooks.get(event.name) if isinstance(hooks, dict) else None
        for index, group in enumerate(groups if isinstance(groups, list) else []):
            if not isinstance(group, dict):
                continue
            entries = [group] if harness.flat else group.get("hooks")
            for handler, hook in enumerate(
                entries if isinstance(entries, list) else []
            ):
                command = hook.get("command") if isinstance(hook, dict) else None
                if isinstance(command, str) and f"guard {harness.name}" in command:
                    found.append(
                        _Hook(
                            source=source,
                            event=event.name,
                            group=index,
                            handler=handler,
                            matcher=group.get("matcher"),
                            command=command,
                            fail_closed=group.get("failClosed") is True,
                            wiring=_wiring(hook) if harness is CLAUDE_CODE else (),
                        )
                    )
    return found


def selftest(  # noqa: PLR0913 - keyword-only; the files and the process runner are injected
    harness: Harness,
    *,
    hook_files: Sequence[Path],
    project: Path,
    run: Callable[[Sequence[str], Path], tuple[int, str]],
    confirm: Callable[[str], bool],
    out: TextIO,
    codex_config: Path | None = None,
    global_files: Sequence[Path] = (),
    program: str = "paveo",
) -> int:
    """Run each configured paveo hook exactly as the agent would, and check it
    refuses. Returns 0 only if at least one was found and every one refused.

    Every agent lets a call through when it cannot run the hook, so a deleted
    venv or a typo in the path turns the seatbelt off without a word. This is the
    check for that, and it is run by a person, because the guard cannot report
    its own absence. ``codex_config`` also checks Codex has been told to trust
    the hook, which ``init`` leaves for the person to do.
    """
    found = [_paveo_hooks(harness, path, out) for path in hook_files]
    # A hook file that cannot be read may be one the agent drops too, with its
    # hooks: that is a failure, whatever the other files say (/code-review).
    unreadable = sum(hooks is None for hooks in found)
    hooks = [hook for each in found if each for hook in each]
    if not hooks:
        out.write(
            f"FAIL no hook runs `paveo guard {harness.name}` in: "
            + ", ".join(str(path) for path in hook_files)
            + "\n"
        )
        return 1
    for source in dict.fromkeys(hook.source for hook in hooks):
        if any(_same_place(source, each) for each in global_files):
            # A note, not a failure: a company rolling the guard out to every
            # machine puts it there on purpose (D72). A person who meant to guard
            # one project needs to hear it (a tester's report, D75).
            out.write(
                f"note the paveo hook in {source} runs in every project on this "
                f"machine, not only this one. To guard only chosen projects, "
                f"remove it there and run {shlex.quote(program)} init "
                f"{harness.name} in each.\n"
            )
    # A hook the agent skips refuses perfectly when run by hand, so its wiring is
    # read first: in 0.1.9 a Bash-only matcher passed while edits went unguarded.
    # Only hooks that read this project's policy count, as in init.
    home = project / ".paveo"
    unguarded = _unguarded(harness, [hook for hook in hooks if _reads(hook, home)])
    for reason in unguarded:
        out.write(f"FAIL {reason}\n")
    failed = len(unguarded)
    for command in dict.fromkeys(hook.command for hook in hooks):
        verdict = _check_hook(
            harness, command, project=project, run=run, confirm=confirm
        )
        failed += verdict is not None
        out.write(f"{'FAIL' if verdict else 'ok  '} {command}\n")
        if verdict:
            out.write(f"     {verdict}\n")
    for problem in _agent_problems(harness, hooks, project, codex_config, out):
        failed += 1
        out.write(f"FAIL {problem}\n")
    return 1 if failed or unreadable else 0


def _same_place(first: Path, second: Path) -> bool:
    """Whether two paths name the same file or folder. Asked of the filesystem
    when both exist, so links, hard links and letter case count; then by folder
    and name, for one not made yet; then by resolved path, where neither folder
    exists and letter case cannot be told apart."""
    try:
        return os.path.samefile(first, second)
    except OSError:
        pass
    try:
        return os.path.samefile(first.parent, second.parent) and (
            first.name == second.name
        )
    except OSError:
        return os.path.realpath(first) == os.path.realpath(second)


def _paveo_hooks(harness: Harness, path: Path, out: TextIO) -> list[_Hook] | None:
    """The paveo hooks in one hook file, or ``None`` if it exists and cannot be
    read."""
    try:
        document = parse_settings(path)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        out.write(f"FAIL {path} could not be read ({type(e).__name__})\n")
        return None
    if not isinstance(document, dict):
        out.write(f"note      {path} is not a settings object, so it holds no hook\n")
        return []
    if harness is CLAUDE_CODE and document.get("disableAllHooks") is True:
        # init refuses this; a file edited after init must fail the check too.
        out.write(f"FAIL {path} sets disableAllHooks, so no hook runs at all\n")
        return None
    if _rejected_whole(harness, path, document, out):
        # None of its hooks counts, and a guard found nowhere else then fails
        # as missing.
        return []
    return _paveo_hooks_in(harness, path, document)


def _rejected_whole(
    harness: Harness, path: Path, document: dict[str, object], out: TextIO
) -> bool:
    """Whether Claude Code rejects this file whole, said in a note when it
    does. The shape is Claude Code's; Codex's and Cursor's rules are unread."""
    wrong = misshapen(document) if harness is CLAUDE_CODE else None
    if wrong is not None:
        out.write(
            f"note      {path}: its {wrong} has the wrong type, so Claude Code "
            "ignores that file and its hooks; fix it.\n"
        )
    return wrong is not None


def _agent_problems(
    harness: Harness,
    hooks: Sequence[_Hook],
    project: Path,
    codex_config: Path | None,
    out: TextIO,
) -> list[str]:
    """What stops a hook that refuses from ever being asked, where it can be seen."""
    home = project / ".paveo"
    problems = [
        f"no paveo hook for {harness.title}'s {event.name} reads {home}, so this "
        f"project's policy and `paveo stop` never apply. Run `paveo init "
        f"{harness.name}` here."
        for event in harness.events
        if not any(hook.event == event.name and _reads(hook, home) for hook in hooks)
    ]
    if harness is CURSOR:
        return problems + [
            f'the {hook.event} hook in {hook.source} has no "failClosed": true, '
            f"so Cursor lets the call through if the guard crashes or times out."
            for hook in hooks
            if not hook.fail_closed
        ]
    if harness is CODEX and codex_config is not None:
        return problems + _codex_trust(hooks, project, codex_config, out)
    return problems


def _codex_trust(
    hooks: Sequence[_Hook], project: Path, config: Path, out: TextIO
) -> list[str]:
    """Whether Codex will run each hook: hooks on, the project trusted (for a
    project's own hooks), and the hook trusted in ``/hooks``.

    Codex records trust in its config by the hook's place: the file, the event
    and two indexes. **A missing record is certain: the hook never runs.** A
    present one is not proof, because Codex also records a hash of the hook as
    trusted and skips it if the hook has changed since; paveo does not
    reproduce that hash, and says so rather than call it checked (Rule 3).
    """
    try:
        document = tomllib.loads(config.read_text(encoding="utf-8"))
    except FileNotFoundError:
        document = {}
    except (OSError, ValueError) as e:  # TOMLDecodeError is a ValueError
        return [
            f"{config} could not be read ({type(e).__name__}), so paveo cannot "
            f"tell whether Codex runs the hook."
        ]
    problems: list[str] = []
    features = document.get("features")
    if isinstance(features, dict) and features.get("hooks") is False:
        problems.append(f"{config} sets features.hooks = false: Codex runs no hook.")
    trusted = _trust_level(document.get("projects"), project) == "trusted"
    hooks_table = document.get("hooks")
    states = hooks_table.get("state") if isinstance(hooks_table, dict) else None
    for hook in hooks:
        own = Path(os.path.abspath(hook.source)).is_relative_to(
            os.path.abspath(project)
        )
        if own and not trusted:
            problems.append(
                f"paveo found no record that Codex trusts {project}, and Codex "
                f"loads no hook from a project it does not trust. Start Codex here "
                f"and trust the folder."
            )
        keys = {
            f"{place}:pre_tool_use:{hook.group}:{hook.handler}"
            for place in {os.path.abspath(hook.source), os.path.realpath(hook.source)}
        }
        state = next(
            (
                states[key]
                for key in keys
                if isinstance(states, dict) and isinstance(states.get(key), dict)
            ),
            None,
        )
        if state is None or not isinstance(state.get("trusted_hash"), str):
            problems.append(
                f"Codex has no record of you trusting the hook in {hook.source}, so "
                f"it never runs. Start Codex here, type /hooks and trust it."
            )
        elif state.get("enabled") is False:
            problems.append(f"the hook in {hook.source} is switched off in /hooks.")
        else:
            out.write(
                f"note Codex trusts a hook at this place in {hook.source}. paveo "
                f"cannot check it is this exact hook: if /hooks shows it as "
                f"modified, Codex skips it until you trust it again.\n"
            )
    return list(dict.fromkeys(problems))


def _trust_level(projects: object, project: Path) -> object:
    """The trust Codex gives ``project``, found as Codex finds it
    (``config/src/project_trust.rs``): the folder, then its git repository's
    root, each resolved spelling before the literal one, and **the first entry
    present decides**, even one with no trust level (/code-review, D57)."""
    if not isinstance(projects, dict):
        return None
    root = next(
        (
            folder
            for folder in (project, *project.parents)
            if (folder / ".git").exists()
        ),
        None,
    )
    keys = [
        key
        for folder in (project, root)
        if folder is not None
        for key in dict.fromkeys((os.path.realpath(folder), os.path.abspath(folder)))
    ]
    for key in keys:
        if key in projects:
            entry = projects[key]
            return entry.get("trust_level") if isinstance(entry, dict) else None
    return None


def _check_hook(
    harness: Harness,
    command: str,
    *,
    project: Path,
    run: Callable[[Sequence[str], Path], tuple[int, str]],
    confirm: Callable[[str], bool],
) -> str | None:
    """Why this hook would not protect anything, or ``None`` if it refuses."""
    argv = _plain_command(harness, command, project, confirm)
    if isinstance(argv, str):
        return argv
    code, stderr = run(argv, project)
    if code == BLOCK and NOT_A_CALL in stderr:
        return None
    if code == BLOCK:
        first = stderr.strip().splitlines()[0] if stderr.strip() else "no message"
        return f"it refuses every call, before reading it: {first}"
    return (
        f"it exited {code}. {harness.title} blocks a call only on exit 2, so this "
        f"seatbelt is not refusing anything."
    )


def _plain_command(
    harness: Harness, command: str, project: Path, confirm: Callable[[str], bool]
) -> list[str] | str:
    """The argv to run, or why this command will not be run.

    **Only a plain paveo command is run.** Hook files come with repositories,
    and a cloned one could put any command there with "guard <agent>" in it;
    running it here would skip the trust prompt the agent shows first. So: no
    shell syntax, no variable but the project directory, a program named
    ``paveo``, and the guard's own arguments. A program inside the project runs
    only if the person confirms they put it there: a virtualenv in the project is
    the ordinary case, and a cloned repository's own ``paveo`` the one to refuse.
    """
    words = _words(command, project)
    if not words or Path(words[0]).name != "paveo":
        return f"not run: only a plain `paveo guard {harness.name}` command is tested"
    options = words[3::2]
    if words[1:3] != ["guard", harness.name] or not (
        len(words) % 2 == 1 and set(options) <= _GUARD_OPTIONS
    ):
        return "not run: arguments other than --dir and --agent"
    # A relative path is the project's, where the agent runs the hook and where
    # this runs it, so it is checked where it will run.
    program = (
        shutil.which(words[0]) if os.sep not in words[0] else str(project / words[0])
    )
    if program is None or not Path(program).is_file():
        return (
            f"`{words[0]}` was not found. {harness.title} treats that as a hook "
            f"that failed without blocking, so this seatbelt is off."
        )
    written, target, inside = (
        Path(os.path.abspath(program)),
        Path(program).resolve(),
        project.resolve(),
    )
    if target.name != "paveo":
        # A link named paveo to something else: /bin/sh would read the next word,
        # `guard`, as a script to run (/security-review, D49).
        return f"not run: `{words[0]}` is a link to {target.name}, not to paveo"
    if (
        written.is_relative_to(inside) or target.is_relative_to(inside)
    ) and not confirm(program):
        return (
            "not run: the program is inside this project and was not confirmed. "
            "Run the self-test in a terminal and answer yes if you installed it."
        )
    return [program, *words[1:]]


def _words(command: str, project: Path) -> list[str]:
    """The command's words with the project directory filled in, or ``[]`` if it
    uses any shell syntax: an operator, a substitution, or a variable other than
    ``$CLAUDE_PROJECT_DIR``.

    Judged on the command as written, not after the directory is filled in, so a
    project at ``~/Code/app (old)`` or ``~/R&D`` is not mistaken for shell syntax
    (/code-review, D50). Quoted characters are literal to the shell and here.
    """
    if "`" in command or "\n" in command:
        return []
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:  # an unclosed quote
        return []
    words: list[str] = []
    for token in tokens:
        if token and set(token) <= set(_OPERATORS):
            return []
        if "$" in _PROJECT_VARIABLE.sub("", token):
            return []
        words.append(_PROJECT_VARIABLE.sub(lambda _: str(project), token))
    return words


def ask(program: str) -> bool:
    answer = input(
        f"The hook runs {program}, which is inside this project. Run it to test "
        f"it? Say yes only if you installed it yourself. [y/N] "
    )
    return answer.strip().lower() in {"y", "yes"}


def run_hook(argv: Sequence[str], project: Path) -> tuple[int, str]:
    """Run one hook with the self-test call on stdin, as the agent would."""
    import subprocess  # noqa: PLC0415 - only --selftest starts a process

    try:
        done = subprocess.run(  # noqa: S603 - a plain paveo command, vetted by _check_hook
            list(argv),
            input=_SELFTEST_CALL,
            capture_output=True,
            timeout=_SELFTEST_TIMEOUT_S,
            env={**os.environ, "CLAUDE_PROJECT_DIR": str(project)},
            cwd=project,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return -1, "timed out: the agent lets a timed-out call through"
    return done.returncode, done.stderr.decode("utf-8", "replace")
