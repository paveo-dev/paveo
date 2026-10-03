"""The coding agents the guard sits in front of, and how each one runs a hook.

Three agents, one guard. What differs is only the contract around it, each read
from the agent's own documentation (and Codex's source) on 2026-09-26, D57:

- **Claude Code**: a ``PreToolUse`` hook in a settings file. Exit 2 blocks and
  hands stderr to the model; every other exit, a crash, a timeout and a missing
  binary let the call through.
- **Codex**: the same ``PreToolUse`` shape, in ``.codex/hooks.json``. Exit 2
  blocks only when stderr has text; the rest is Claude Code's. **Codex runs a
  hook only after the person trusts it**, and a project's hooks only in a
  trusted project, so a hook written and never trusted does nothing.
- **Cursor**: ``beforeShellExecution`` and ``preToolUse`` in
  ``.cursor/hooks.json``. Exit 2 blocks. Crashes, timeouts and other exits let
  the call through **unless the hook sets ``failClosed``**, which ``init`` does.

This module is what ``init`` writes, what ``--selftest`` checks, the calls
``init`` asks the policy about before it says the seatbelt is on, and the one
change the guard makes to a call before the policy sees it (``for_policy``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import ConfigError

# The file headers of Codex's patch format (codex-rs/apply-patch).
_PATCH_HEADER = re.compile(r"\*\*\* (?:Add File|Update File|Delete File|Move to): (.*)")


@dataclass(frozen=True)
class Event:
    """One hook event the guard is registered for."""

    name: str
    # The matcher init writes; None writes none, which reaches every call.
    matcher: str | None
    # Tool names the matcher must reach, or None: it must reach every call.
    reaches: tuple[str, ...] | None


@dataclass(frozen=True)
class Harness:
    name: str  # the word after `paveo init` and `paveo guard`
    title: str
    folder: str  # the agent's folder in a project, and in the home directory
    hook_file: str  # inside ``folder``, where init writes the hook
    events: tuple[Event, ...]
    # Claude Code's matcher is tool names joined by | or ,; the others', a regex.
    names_matcher: bool
    # Cursor's hooks.json is flat, one command per entry, with a version.
    flat: bool
    # What init asks the policy: each must be refused before it says "on".
    probes: tuple[tuple[str, dict[str, object], str], ...]
    # Said once init has checked everything it can: what is left, if anything.
    last_step: str


_RM = "rm -rf ./paveo-init-probe"
_OWN_POLICY = ".paveo/policy.json"

CLAUDE_CODE = Harness(
    name="claude-code",
    title="Claude Code",
    folder=".claude",
    # The personal settings file: the hook names this machine's paveo by path.
    hook_file="settings.local.json",
    events=(
        Event(
            "PreToolUse",
            "Bash|Write|Edit|NotebookEdit",
            ("Bash", "Write", "Edit", "NotebookEdit"),
        ),
    ),
    names_matcher=True,
    flat=False,
    probes=(
        ("Bash", {"command": _RM}, "`rm -rf`"),
        ("Write", {"file_path": _OWN_POLICY, "content": "{}"}, "an edit to it"),
    ),
    last_step="Start Claude Code in this folder: `rm -rf` and edits to the guard "
    "are now refused.",
)

CODEX = Harness(
    name="codex",
    title="Codex",
    folder=".codex",
    hook_file="hooks.json",
    # Codex reports every shell command as Bash and every file edit as
    # apply_patch, whose input is the patch text (source, e72da2b).
    events=(Event("PreToolUse", "^(?:Bash|apply_patch)$", ("Bash", "apply_patch")),),
    names_matcher=False,
    flat=False,
    probes=(
        ("Bash", {"command": _RM}, "`rm -rf`"),
        (
            "apply_patch",
            {"command": f"*** Begin Patch\n*** Update File: {_OWN_POLICY}\n"},
            "an edit to it",
        ),
    ),
    last_step="One step is left, and only you can take it: Codex runs no hook "
    "until you trust it. Start Codex in this folder, trust the folder if it "
    "asks, type /hooks and trust the paveo hook. Then check it:\n"
    "  paveo guard codex --selftest",
)

CURSOR = Harness(
    name="cursor",
    title="Cursor",
    folder=".cursor",
    hook_file="hooks.json",
    events=(
        Event("beforeShellExecution", None, None),
        # The file tools Cursor documents. A file tool under another name never
        # reaches the guard; one whose name merely contains these is refused,
        # as a tool the policy does not name. Their input is undocumented (D57).
        Event("preToolUse", "Write|Delete", ("Write", "Delete")),
    ),
    names_matcher=False,
    flat=True,
    probes=(
        ("Shell", {"command": _RM}, "`rm -rf`"),
        ("Write", {"file_path": _OWN_POLICY, "content": "{}"}, "an edit to it"),
    ),
    last_step="Open this folder in Cursor and trust the workspace if it asks: "
    "Cursor runs a project's hooks only in a trusted workspace. `rm -rf` is then "
    "refused, and so are Write and Delete on the guard's files; a file tool "
    "Cursor names otherwise is not seen.",
)

HARNESSES = {harness.name: harness for harness in (CLAUDE_CODE, CODEX, CURSOR)}


def for_policy(
    harness: Harness, tool: str, arguments: dict[str, object]
) -> dict[str, object]:
    """The arguments the policy judges. Only Codex's ``apply_patch`` changes: it
    gets ``paths``, every file its patch names, one a line, always replacing
    whatever the call held (/code-review, D57). The guard and init's probes both
    come through here, so init tests the reading it relies on.
    """
    if harness is not CODEX or tool != "apply_patch":
        return arguments
    patch = arguments.get("command")
    if not isinstance(patch, str):
        raise ConfigError(
            "the apply_patch call has no patch text.",
            remedy="nothing else is judged, so it is refused.",
        )
    return {**arguments, "paths": _patch_paths(patch)}


def _patch_paths(patch: str) -> str:
    """Read from the header lines alone, in one pass: a pattern searching the
    whole patch for a header could be made to take quadratic time, and would
    refuse a patch whose *content* mentions a guarded folder. Split on ``\\n``
    only, as Codex does: ``splitlines`` also breaks at ``\\x0b`` and others, and
    so read a shorter path than Codex writes (/code-review, D57). Space around
    a line is dropped, as Rust's ``trim`` drops it.
    """
    return "\n".join(
        header.group(1)
        for line in patch.split("\n")
        if (header := _PATCH_HEADER.match(line.strip())) is not None
    )
