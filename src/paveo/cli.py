"""The ``paveo`` command: a seatbelt for coding agents, and a panic button (D49).

::

    paveo init {claude-code,codex,cursor}
    paveo guard {claude-code,codex,cursor} [--dir .paveo] [--agent NAME]
    paveo guard [AGENT] --selftest [--settings FILE ...]
    paveo trial [--dir .paveo]
    paveo stop [--dir .paveo]
    paveo resume [--dir .paveo]
    paveo replay claude-code [PATH ...] [--dir .paveo] [--agent claude-code]
    paveo learn claude-code --from-history [PATH ...] [--dir .paveo]
    paveo evidence [--dir .paveo] [--since DATE] [--until DATE] [--out DIR]

``--dir`` holds three things, so the hook and the panic button cannot disagree
about where they are: ``policy.json``, ``audit.jsonl`` and, while stopped,
``stop``.

**The guard is a pre-tool hook** for Claude Code, Codex and Cursor, and their
shared contract decides its shape. Exit 2 blocks the call and hands stderr to
the model. *Every other exit may let the call through*: 1, a crash, a timeout, a
command that cannot be found. So every failure inside the guard answers 2, and
the one it cannot answer, a missing binary, is what ``--selftest`` is for. The
differences between the agents are in ``_harnesses`` (D57).

**Allowing prints nothing.** An explicit ``"allow"`` would skip the agent's own
permission prompt, and a seatbelt must never widen what is permitted. Silence
leaves the call to the agent's normal flow.

Nothing here opens a socket: stdin, files and, for ``--selftest``, a child
process. The standalone no-egress run drives the guard, ``stop``, ``resume``,
``trial`` and ``evidence``.
Setting the guard up, ``init`` and ``--selftest``, lives in ``_setup``; reading
past sessions, ``replay`` and ``learn``, in ``_replay``; audit evidence, a paid
feature, in ``_evidence``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import BinaryIO, TextIO

from . import _evidence, enforce
from ._harnesses import CLAUDE_CODE, CODEX, CURSOR, HARNESSES, Harness, for_policy
from ._licence import (
    LICENCE_FILE,
    TRIAL_PREFIX,
    apply,
    plan_in,
    read_key,
    trial,
)
from ._memory import SessionMemory
from ._policy_document import load_file
from ._replay import from_history
from ._setup import BLOCK as _BLOCK
from ._setup import NOT_A_CALL as _NOT_A_CALL
from ._setup import POLICY as _POLICY
from ._setup import ask, init, keep_out_of_git, run_hook, selftest
from .audit import AuditLog, utc_now
from .errors import ConfigError, PaveoError, PolicyDenied
from .session import Identity

_LOG, _STOP = "audit.jsonl", "stop"
# Well under any hook timeout a person would set, and far over a normal decision
# (about 60 ms, most of it starting Python and importing; tests/overhead.py).
_DEADLINE_S = 4.0
# A Write can carry a whole file. Over this, the call is refused unread.
_MAX_CALL_BYTES = 64 << 20
_MAX_PRINCIPAL = 128
# The file headers of Codex's patch format (codex-rs/apply-patch).


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "init":
        project = Path.cwd()
        harness = HARNESSES[arguments.harness]
        return init(
            harness,
            project=project,
            # The program the person ran, which is what the hook must run too.
            program=Path(os.path.abspath(sys.argv[0])),
            hook_files=_hook_files(harness, project),
            run=run_hook,
            out=sys.stdout,
            global_files=_global_hook_files(harness),
        )
    directory = Path(arguments.dir)
    if arguments.command in {"replay", "learn"}:
        # Claude Code keeps every project's sessions here. Unless told otherwise,
        # only what was done in the project --dir belongs to is read (D53).
        paths = [Path(path) for path in arguments.paths]
        return from_history(
            arguments.command,
            paths=paths or [Path.home() / ".claude" / "projects"],
            within=None if paths else Path(os.path.abspath(directory)).parent,
            directory=directory,
            agent=arguments.agent,
            now=datetime.now(UTC),
            out=sys.stdout,
        )
    if arguments.command in {"trial", "stop", "resume"}:
        return _manage(arguments.command, directory)
    if arguments.command == "evidence":
        now = datetime.now(UTC)
        return _evidence.command(
            directory=directory,
            log=arguments.log,
            policy=arguments.policy,
            out=arguments.out or Path(f"paveo-evidence-{now.date()}"),
            since=arguments.since,
            until=arguments.until,
            now=now,
            stdout=sys.stdout,
        )
    if arguments.selftest:
        project = Path.cwd()
        harness = HARNESSES[arguments.harness or CLAUDE_CODE.name]
        return selftest(
            harness,
            hook_files=[Path(path) for path in arguments.settings]
            if arguments.settings
            else _hook_files(harness, project),
            project=project,
            run=run_hook,
            confirm=ask if sys.stdin.isatty() else lambda _: False,
            out=sys.stdout,
            codex_config=_codex_home() / "config.toml",
            global_files=_global_hook_files(harness),
            program=sys.argv[0],
        )
    if arguments.harness is None:
        _parser().error("say which agent to guard: paveo guard claude-code")
    _refuse_after(_DEADLINE_S)
    return guard(
        HARNESSES[arguments.harness],
        directory=directory,
        agent=arguments.agent or arguments.harness,
        stdin=sys.stdin.buffer,
        stdout=sys.stdout,
        stderr=sys.stderr,
        stopped=lambda: _stop_file_present(directory / _STOP),
        # The raw word the hook ran paveo by, a full path when init wrote it.
        # Nothing that can raise runs before the guard's own try (D75).
        program=sys.argv[0],
    )


def _manage(command: str, directory: Path) -> int:
    """The commands a person runs on ``--dir`` itself."""
    if command == "trial":
        return start_trial(directory, today=datetime.now(UTC).date(), out=sys.stdout)
    if command == "stop":
        return stop(directory, out=sys.stdout)
    return resume(directory, out=sys.stdout)


def _hook_files(harness: Harness, project: Path) -> list[Path]:
    """Every file the agent reads a project's hooks from, managed ones aside."""
    if harness is CODEX:
        own = [project / ".codex" / "hooks.json"]
    elif harness is CURSOR:
        own = [project / ".cursor" / "hooks.json"]
    else:
        own = [
            project / ".claude" / "settings.json",
            project / ".claude" / "settings.local.json",
        ]
    return list(dict.fromkeys(own + _global_hook_files(harness)))


def _global_hook_files(harness: Harness) -> list[Path]:
    """The hook files the agent reads in every project (D75)."""
    if harness is CODEX:
        return [_codex_home() / "hooks.json"]
    if harness is CURSOR:
        return [Path.home() / ".cursor" / "hooks.json"]
    return [Path.home() / ".claude" / "settings.json"]


def _codex_home() -> Path:
    """Where Codex keeps its own configuration: ``$CODEX_HOME``, or ~/.codex."""
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _refuse_after(seconds: float) -> None:
    """Exit 2 the moment ``seconds`` pass, whatever the guard is doing.

    Claude Code and Codex let a call through when a hook times out, so a guard
    that can be made slow can be switched off: a pattern made to search a
    crafted megabyte, or an agent holding the audit log's lock (/code-review,
    D49). ``os._exit`` from
    the signal handler rather than an exception, because an exception can be
    caught, or arrive after the answer was chosen and turn it into exit 1.
    """

    def refuse(_signal: int, _frame: object) -> None:
        os.write(2, b"paveo refused this call: it could not decide in time.\n")
        os._exit(_BLOCK)

    signal.signal(signal.SIGALRM, refuse)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def guard(  # noqa: PLR0913 - keyword-only; the streams and clock are injected (Rule 14)
    harness: Harness,
    *,
    directory: Path,
    agent: str,
    stdin: BinaryIO,
    stdout: TextIO,
    stderr: TextIO,
    stopped: Callable[[], bool],
    now: Callable[[], datetime] | None = None,
    salt: Callable[[int], bytes] = os.urandom,
    program: str = "paveo",
) -> int:
    """Decide one tool call from ``harness``. Returns the exit code: 0 or 2.

    The policy and the log are opened before the call is read, so a guard that
    refuses an unreadable call has proved both work, which ``--selftest`` relies
    on. Refusals carry ``PolicyDenied.for_model``; a failure carries its type and,
    for Paveo's own errors, their message, which never holds payload (§8).
    """
    clock = utc_now if now is None else now
    try:
        today = clock().date()
        policy = apply(
            load_file(directory / _POLICY), plan_in(directory, today=today), today=today
        )
        with AuditLog(directory / _LOG, now=now) as audit:
            tool, arguments, principal = _read_call(harness, stdin)
            decide = partial(
                enforce.check_tool,
                policy=policy,
                audit=audit,
                identity=Identity(agent_id=agent, principal=principal),
                tool=tool,
                arguments=arguments,
                stopped=stopped(),
            )
            if not policy.needs_memory(agent, tool) or principal is None:
                # Nothing to read or leave, or no session to keep it for: a rule
                # that needs memory then refuses as memory_unavailable (D59).
                decide(recall=None)
            else:
                with ExitStack() as held:
                    try:
                        memory = held.enter_context(
                            SessionMemory(directory, agent, principal, salt=salt)
                        )
                        recall = memory.recall(clock().timestamp())
                    except ConfigError:
                        # Memory that cannot be opened or read is judged as none,
                        # so the refusal is recorded like any other rather than
                        # leaving no trace (/code-review, D59).
                        decide(recall=None)
                    else:
                        admitting, admitted = memory.steps(
                            policy, agent, tool, arguments, recall
                        )
                        decide(recall=recall, admitting=admitting, admitted=admitted)
    except PolicyDenied as refused:
        return _refuse(harness, refused.for_model, stdout, stderr)
    except PaveoError as e:
        expected, named = directory / _POLICY, shlex.quote(program)
        if _absent(expected):
            # Still refused (locked decision #4); only the words differ. A hook
            # in settings a project does not own reaches projects nobody set up,
            # and "check the path" told a tester nothing (D75).
            return _refuse(
                harness,
                f"paveo refused this call: there is no paveo policy at {expected}, "
                f"so nothing can be judged. To guard the project that folder is "
                f"in, run `{named} init {harness.name}` there. If this project "
                f"was never meant to be guarded, a paveo hook in settings it does "
                f"not own is reaching it: `{named} guard {harness.name} "
                f"--selftest` lists every paveo hook and where it is.",
                stdout,
                stderr,
            )
        return _refuse(
            harness,
            f"paveo refused this call because it could not decide it: {e}",
            stdout,
            stderr,
        )
    except BaseException as e:  # any other exit code may let the call through
        # The type only: an arbitrary exception's message may quote the call.
        return _refuse(
            harness,
            f"paveo refused this call because it failed ({type(e).__name__}). "
            f"Run `{shlex.quote(program)} guard {harness.name} --selftest`.",
            stdout,
            stderr,
        )
    return 0


def _absent(path: Path) -> bool:
    """Whether ``path`` is certainly not there. Only "no such file" counts: a
    folder that cannot be read, or a broken link, is not absent, and saying so
    would send a person to write a second policy (/code-review, D75). Never
    raises, because it runs inside the guard's error handling."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _refuse(harness: Harness, message: str, stdout: TextIO, stderr: TextIO) -> int:
    """Exit 2 with the reason on stderr, which every agent reads (Codex blocks on
    exit 2 only when stderr has text). Cursor takes its message for the model
    from a JSON answer on stdout, so it gets one too; exit 2 blocks either way."""
    if harness is CURSOR:
        answer = {
            "permission": "deny",
            "user_message": message,
            "agent_message": message,
        }
        stdout.write(json.dumps(answer) + "\n")
    stderr.write(message + "\n")
    return _BLOCK


def _read_call(
    harness: Harness, stdin: BinaryIO
) -> tuple[str, dict[str, object], str | None]:
    """The tool name, its input and the session id, or ``ConfigError``.

    Claude Code and Codex send one shape. Codex's ``apply_patch`` also gets
    ``paths``, the files its patch names, so the policy judges the paths and not
    the patch text. Cursor sends a shell command to ``beforeShellExecution`` as a
    bare command, judged as its ``Shell`` tool, and every other tool to
    ``preToolUse``.
    """
    raw = stdin.read(_MAX_CALL_BYTES + 1)
    if len(raw) > _MAX_CALL_BYTES:
        raise ConfigError(
            f"the call on stdin is over {_MAX_CALL_BYTES >> 20} MiB.",
            remedy="nothing that large is read, so it is refused.",
        )
    remedy = f"run paveo guard {harness.name} only as the hook init writes."
    try:
        call = json.loads(raw)
    except ValueError as e:  # JSONDecodeError and UnicodeDecodeError both
        raise ConfigError(
            f"the input {_NOT_A_CALL}: it is not JSON.", remedy=remedy
        ) from e
    if not isinstance(call, dict):
        call = {}
    tool, arguments = call.get("tool_name"), call.get("tool_input")
    session = call.get("session_id")
    if harness is CURSOR:
        session = call.get("conversation_id")
        if call.get("hook_event_name") == "beforeShellExecution":
            command = call.get("command")
            tool, arguments = "Shell", {"command": command}
            if not isinstance(command, str):
                tool = None
        elif call.get("hook_event_name") != "preToolUse":
            tool = None
    if not isinstance(tool, str) or not tool or not isinstance(arguments, dict):
        raise ConfigError(
            f"the input {_NOT_A_CALL}: it has no tool and no input.", remedy=remedy
        )
    arguments = for_policy(harness, tool, arguments)
    # An opaque id, recorded as the principal so one session's calls can be read
    # together. The agent chose it, not the model; bounded all the same.
    principal = (
        session if isinstance(session, str) and len(session) <= _MAX_PRINCIPAL else None
    )
    return tool, arguments, principal or None


def _stop_file_present(path: Path) -> bool:
    """``Path.exists`` answers False when it cannot look, which would be a panic
    button that fails open. Only "no such file" means not stopped."""
    try:
        os.stat(path)
    except FileNotFoundError:
        return False
    return True


def start_trial(directory: Path, *, today: date, out: TextIO) -> int:
    """Start 30 days of Team in ``--dir``: a trial token in its licence file.

    Never replaces a key someone paid for, and never restarts a trial already
    running: run again, it says how long is left (D65). Whatever is wrong with
    the file is said, not raised: this is a person at a terminal (/code-review).
    """
    if not _guarded(directory, out):
        return 1
    location = directory / LICENCE_FILE
    try:
        current = location.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        current = None
    except (OSError, UnicodeDecodeError) as e:
        out.write(f"paveo: {location} could not be read ({e}); nothing changed.\n")
        return 1
    if current and not current.startswith(TRIAL_PREFIX):
        out.write(
            f"paveo: {location} already holds a licence key; the trial would "
            f"replace it, so nothing changed.\n"
        )
        return 1
    if current:
        return _trial_state(current, today=today, out=out)
    key = trial(today)
    try:
        # A key is this machine's, never the repository's (D62), and a project
        # set up before this existed has no such line yet.
        keep_out_of_git(directory / ".gitignore", LICENCE_FILE)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        descriptor = os.open(location, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(key + "\n")
    except OSError as e:
        out.write(f"paveo: {location} could not be written ({e}); nothing changed.\n")
        return 1
    plan = read_key(key, today=today)
    out.write(
        f"paveo: Team trial started: up to {plan.agents} agents per policy "
        f"until {plan.expires}. "
        f"Nothing is charged, and when it ends the free plan applies; no agent "
        f"stops.\n"
    )
    return 0


def _trial_state(current: str, *, today: date, out: TextIO) -> int:
    """Say where a trial already started stands, rather than restart it."""
    try:
        plan = read_key(current, today=today)
    except ConfigError as e:
        out.write(f"paveo: {e} Nothing changed.\n")
        return 1
    state = "ended on" if plan.lapsed is not None else "is running until"
    out.write(f"paveo: the Team trial {state} {plan.expires}.\n")
    return 0


def stop(directory: Path, *, out: TextIO) -> int:
    """Refuse every check under ``directory`` until ``resume``. Idempotent.

    Refuses a directory with no policy in it: a stop file written where no guard
    reads it looks like a stop and is not one (/code-review, D49).
    """
    if not _guarded(directory, out):
        return 1
    descriptor = os.open(directory / _STOP, os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(descriptor)
    out.write(
        f"paveo: stopped. Every call guarded from {directory} is refused until "
        f"`paveo resume --dir {directory}`.\n"
    )
    return 0


def resume(directory: Path, *, out: TextIO) -> int:
    """Lift ``stop``. Idempotent."""
    if not _guarded(directory, out):
        return 1
    (directory / _STOP).unlink(missing_ok=True)
    out.write(f"paveo: resumed. Calls guarded from {directory} are checked again.\n")
    return 0


def _guarded(directory: Path, out: TextIO) -> bool:
    if (directory / _POLICY).is_file():
        return True
    out.write(
        f"paveo: {directory} holds no {_POLICY}, so no guard reads it. Pass the "
        f"--dir your hook uses, e.g. --dir /path/to/project/.paveo\n"
    )
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paveo", description="Admission control for the calls an agent makes."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    agents = sorted(HARNESSES)
    decide = commands.add_parser("guard", help="decide a tool call from stdin")
    decide.add_argument("harness", nargs="?", choices=agents)
    decide.add_argument("--dir", default=".paveo", help="policy, log and stop file")
    decide.add_argument("--agent", help="agent id in policy (default: the agent)")
    decide.add_argument(
        "--selftest", action="store_true", help="check the configured hooks refuse"
    )
    decide.add_argument("--settings", action="append", help="hook file to check")

    setup = commands.add_parser("init", help="set up the guard in this project")
    setup.add_argument("harness", choices=agents)

    for name, summary in (
        ("stop", "refuse every call"),
        ("resume", "lift a stop"),
        ("trial", "start 30 days of the Team plan here"),
    ):
        command = commands.add_parser(name, help=summary)
        command.add_argument("--dir", default=".paveo")

    replay = commands.add_parser("replay", help="judge past sessions by the policy")
    learn = commands.add_parser("learn", help="propose a policy from past sessions")
    learn.add_argument("--from-history", action="store_true", required=True)
    for command in (replay, learn):
        command.add_argument("harness", choices=["claude-code"])
        command.add_argument("paths", nargs="*", help="session files or folders")
        command.add_argument("--dir", default=".paveo", help="where the policy is")
        command.add_argument(
            "--agent", default="claude-code", help="agent id in policy"
        )
    evidence = commands.add_parser(
        "evidence", help="export a period of the audit log for an auditor"
    )
    evidence.add_argument("--dir", default=".paveo", help="licence, log and policy")
    evidence.add_argument("--log", type=Path, help="audit log (default: in --dir)")
    evidence.add_argument("--policy", type=Path, help="policy (default: in --dir)")
    evidence.add_argument("--since", type=date.fromisoformat, help="first day, UTC")
    evidence.add_argument("--until", type=date.fromisoformat, help="last day, UTC")
    evidence.add_argument("--out", type=Path, help="new folder to write")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
