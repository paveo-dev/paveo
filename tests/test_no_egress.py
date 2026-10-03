"""The claim that Paveo opens no sockets, made checkable.

Run this **standalone**, on purpose::

    python -m tests.test_no_egress

It is also collected by pytest, but pytest plugins open sockets of their own and
a guard installed inside a pytest process can be defeated or masked by them. The
standalone run is the one that counts, which is why this file exits non-zero by
itself and does not need pytest to be meaningful.

**What this proves, precisely.** It is a regression guard against *our own* code
growing a network call — including in a build we did not mean to ship. It is not
a sandbox: code that reaches past the ``socket`` module, shells out, or loads a
C extension of its own would not be caught here. That is why SECURITY.md asks
customers to verify a release themselves with a firewall rule or ``tcpdump``,
and why we would rather they did.

``_exercise_library`` and ``_exercise_command`` are the places to extend: every
new entry point gets a line there, so what the test covers grows with Paveo.
"""

from __future__ import annotations

import _socket
import socket
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NoReturn

_GUARDED: tuple[tuple[ModuleType, str], ...] = (
    (socket, "socket"),
    (socket, "socketpair"),
    (socket, "create_connection"),
    (socket, "create_server"),
    (_socket, "socket"),
)


class SocketConstructedError(AssertionError):
    """Raised the instant anything tries to create a socket."""


def _refuse(label: str, attempts: list[str]) -> Callable[..., NoReturn]:
    def guard(*_args: object, **_kwargs: object) -> NoReturn:
        attempts.append(label)
        raise SocketConstructedError(
            f"{label} was called. Paveo must open no sockets "
            f"(locked decision #2, SPEC_V1.md §9)."
        )

    return guard


@contextmanager
def _no_sockets(attempts: list[str]) -> Iterator[None]:
    """Replace every socket-creating entry point for the duration of the block."""
    originals = [(mod, name, getattr(mod, name)) for mod, name in _GUARDED]
    for mod, name in _GUARDED:
        setattr(mod, name, _refuse(f"{mod.__name__}.{name}", attempts))
    try:
        yield
    finally:
        for mod, name, original in originals:
            setattr(mod, name, original)


@contextmanager
def _unimported(prefix: str) -> Iterator[None]:
    """Force ``prefix`` to be imported afresh inside the block, then put it back.

    Without this, a module imported by an earlier test is already in
    ``sys.modules`` and its import-time code never runs under the guard. Restoring
    the originals afterwards matters just as much: two copies of the package in
    one process would give two copies of every exception class, and ``except
    PolicyDenied`` elsewhere would stop matching.
    """

    def belongs(name: str) -> bool:
        return name == prefix or name.startswith(f"{prefix}.")

    saved = {name: mod for name, mod in sys.modules.items() if belongs(name)}
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [n for n in list(sys.modules) if belongs(n)]:
            del sys.modules[name]
        sys.modules.update(saved)


def _exercise_library(workdir: Path) -> None:
    """Drive every entry point Paveo has, public or not.

    Extend this as the public surface grows. A no-egress test that exercises less
    than the library does is a test that says nothing (Rule 3).
    """
    # Imported under the guard, deliberately, so import-time code is covered too.
    import paveo  # noqa: PLC0415
    from paveo._policy_document import load_document  # noqa: PLC0415
    from paveo.audit import AuditLog  # noqa: PLC0415
    from paveo.budget import Reservation, _BudgetCore  # noqa: PLC0415
    from paveo.policy import BudgetPolicy  # noqa: PLC0415
    from paveo.prices import _LISTINGS, _PriceTable  # noqa: PLC0415
    from paveo.stores import InMemoryBudgetStore  # noqa: PLC0415

    policy = load_document(
        {
            "version": 1,
            "policy_id": "no-egress",
            "agents": [
                {
                    "id": "bot",
                    "budget": {"period": "day", "limit_usd": "1.00"},
                    "tools": {
                        "allow": [
                            {"name": "refund", "constraints": {"n": {"max": "5"}}}
                        ],
                        "deny": ["wire_transfer"],
                    },
                }
            ],
        }
    )

    log = AuditLog(workdir / "audit.jsonl")
    try:
        for tool, arguments in (("refund", {"n": 1}), ("refund", {"n": 9}), ("x", {})):
            denial = policy.evaluate_tool("bot", tool, arguments)
            log.append(
                {
                    "agent_id": "bot",
                    "action": {"kind": "tool", "name": tool},
                    "decision": "deny" if denial else "allow",
                    "reason": denial.reason if denial else None,
                    "policy_id": policy.policy_id,
                    "policy_hash": policy.policy_hash,
                }
            )
    finally:
        log.close()

    assert paveo.verify_chain(workdir / "audit.jsonl").ok

    # Not public yet, and exercised here anyway: the budget core mints its
    # reservation ids from `uuid4`, which reads `os.urandom`. That is not a
    # socket — and this is where we prove it rather than assert it in a docstring.
    ledger = _BudgetCore(limit=Decimal("1.00"), period="day")
    moment = datetime(2026, 9, 21, 23, 0, tzinfo=UTC)
    held = ledger.try_reserve(Decimal("0.10"), moment)
    assert isinstance(held, Reservation)
    ledger.settle(held.reservation_id, Decimal("0.02"), moment)
    assert ledger.remaining(moment) == Decimal("0.98")

    # The store serves coroutines without importing `asyncio` (D33), and this is
    # where that stays true: `asyncio` imports `ssl`, `ssl` subclasses
    # `socket.socket` at import time, and the guard above has replaced that name
    # with a function — so an import added anywhere in the library fails this
    # run outright rather than quietly pulling a TLS stack into a library that
    # opens no sockets. Verified by adding one: exit 1 (D29 #4, D33).
    ceiling = BudgetPolicy(period="day", limit_usd=Decimal("1.00"))
    keeper = InMemoryBudgetStore()
    reservation = keeper.reserve(
        agent_id="bot", budget=ceiling, worst_case=Decimal("0.10")
    )
    keeper.settle(
        agent_id="bot",
        reservation_id=reservation.reservation_id,
        actual=Decimal("0.02"),
    )
    keeper.release(
        agent_id="bot",
        reservation_id=keeper.reserve(
            agent_id="bot", budget=ceiling, worst_case=Decimal("0.10")
        ).reservation_id,
    )
    assert keeper.remaining(agent_id="bot", budget=ceiling) == Decimal("0.98")

    # The price table is built at import and takes a lock to mark itself stale.
    # A fresh table, so the process's own is not left refusing every call.
    table = _PriceTable(_LISTINGS)
    rates = table.resolve("claude-sonnet-5", {"inference_geo": "us"})
    assert table.worst_case(rates, 1000, 1000) > 0
    own = table.resolve("claude-sonnet-5", {})
    assert table.actual(own, {"input": 1, "new_class": 1}).price_table_stale

    # The whole model-call path, through the public surface: policy, both
    # adapters, the price table, the shared store and both audit records.
    llm_policy = {
        "version": 1,
        "policy_id": "no-egress-llm",
        "agents": [
            {
                "id": "bot",
                "budget": {"period": "day", "limit_usd": "1.00"},
                "models": {
                    "allow": [
                        "claude-sonnet-5",
                        "local-model",
                        "gpt-5.6-sol",
                        "gemini-2.5-flash",
                    ]
                },
            }
        ],
        "prices": {"local-model": {"input_per_mtok": "0", "output_per_mtok": "0"}},
    }
    with (
        paveo.Paveo.from_policy(llm_policy, audit_path=workdir / "llm.jsonl") as pf,
        pf.session(agent_id="bot") as s,
    ):
        request = {
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        s.check_llm(request, shape="anthropic").record(
            {"input_tokens": 3, "output_tokens": 4}
        )
        s.check_llm({**request, "model": "local-model"}, shape="generic").release()
        s.check_llm(
            {
                "model": "gpt-5.6-sol",
                "service_tier": "default",
                "max_completion_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
            shape="openai",
        ).record({"prompt_tokens": 3, "completion_tokens": 4})
        s.check_llm(
            {
                "model": "gemini-2.5-flash",
                "contents": "hi",
                "config": {"max_output_tokens": 10},
            },
            shape="gemini",
        ).record({"prompt_token_count": 3, "candidates_token_count": 4})
        response = SimpleNamespace(usage={"input_tokens": 3, "output_tokens": 4})
        client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response))
        s.wrap_anthropic(client).messages.create(**request)
        assert s.remaining() < Decimal("1.00")


def _exercise_command(workdir: Path) -> None:
    """The `paveo` command (D49): the Claude Code guard allowing, refusing and
    stopped, the panic button, the trial and audit evidence. Driven in-process
    so the guard sees it; `--selftest` starts a child process, which this guard
    could not see into, and is left to its own tests."""
    import io  # noqa: PLC0415
    import json  # noqa: PLC0415

    import paveo  # noqa: PLC0415
    from paveo._harnesses import CLAUDE_CODE, CODEX, CURSOR  # noqa: PLC0415
    from paveo.cli import guard, main  # noqa: PLC0415

    home = workdir / "seatbelt"
    home.mkdir()
    (home / "policy.json").write_text(
        json.dumps(
            {
                "version": 1,
                "policy_id": "no-egress-cli",
                "agents": [
                    {
                        "id": "claude-code",
                        "tools": {
                            "allow": [
                                {
                                    "name": "Bash",
                                    "constraints": {"command": {"not_matches": "rm"}},
                                    # Drives the guard's memory on disk (D59).
                                    "repeat": {"seconds": 60},
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for command, stopped, expected in (
        ("ls", False, 0),
        ("rm -rf /", False, 2),
        ("ls", True, 2),
        ("ls", False, 2),  # a repeat, remembered in .paveo/memory
        ("pwd", False, 0),
    ):
        hook_input = {
            "session_id": "no-egress",
            "tool_name": "Bash",
            "tool_input": {"command": command},
        }
        code = guard(
            CLAUDE_CODE,
            directory=home,
            agent="claude-code",
            stdin=io.BytesIO(json.dumps(hook_input).encode()),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            stopped=lambda stopped=stopped: stopped,
        )
        assert code == expected, (command, stopped, code)

    # Codex and Cursor (B2c, D57), each with its starter policy and its format.
    for harness, calls in (
        (
            CODEX,
            [
                {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
                {
                    "tool_name": "apply_patch",
                    "tool_input": {"command": "*** Add File: a"},
                },
            ],
        ),
        (
            CURSOR,
            [
                {"hook_event_name": "beforeShellExecution", "command": "rm -rf /"},
                {
                    "hook_event_name": "preToolUse",
                    "tool_name": "Write",
                    "tool_input": {"file_path": "a", "content": ""},
                },
            ],
        ),
    ):
        seat = workdir / harness.name
        seat.mkdir()
        starter = Path(paveo.__file__).parent / "starters" / f"{harness.name}.json"
        (seat / "policy.json").write_text(starter.read_text("utf-8"), "utf-8")
        for hook_input, expected in zip(calls, (2, 0), strict=True):
            code = guard(
                harness,
                directory=seat,
                agent=harness.name,
                stdin=io.BytesIO(json.dumps(hook_input).encode()),
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                stopped=lambda: False,
            )
            assert code == expected, (harness.name, hook_input, code)
    assert main(["stop", "--dir", str(home)]) == 0
    assert main(["resume", "--dir", str(home)]) == 0
    assert paveo.verify_chain(home / "audit.jsonl").ok

    # Audit evidence (D66), a paid feature, under a trial started here.
    assert main(["trial", "--dir", str(home)]) == 0
    evidence = workdir / "evidence"
    assert main(["evidence", "--dir", str(home), "--out", str(evidence)]) == 0
    assert (evidence / "report.html").is_file()

    # Replay and learn (B3, D53) read session files and write one policy.
    sessions = workdir / "sessions.jsonl"
    sessions.write_text(
        json.dumps(
            {
                "type": "assistant",
                "timestamp": "2026-09-26T10:00:00Z",
                "sessionId": "s",
                "cwd": str(workdir),
                "message": {
                    "id": "msg_1",
                    "model": "claude-sonnet-5",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [
                        {"type": "tool_use", "id": "t", "name": "Bash", "input": {}}
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for verb in (["replay"], ["learn", "--from-history"]):
        argv = [*verb, "claude-code", str(sessions), "--dir", str(home)]
        assert main(argv) == 0, verb

    # `init` writes the policy and the settings in-process; the self-test it ends
    # with starts a child, stood in for here, as `--selftest` is above.
    from paveo._setup import NOT_A_CALL, init  # noqa: PLC0415

    program = workdir / "bin" / "paveo"
    program.parent.mkdir()
    program.write_text("", encoding="utf-8")
    project = workdir / "project"
    project.mkdir()
    code = init(
        CLAUDE_CODE,
        project=project,
        program=program,
        hook_files=[project / ".claude" / "settings.local.json"],
        run=lambda _argv, _where: (2, NOT_A_CALL),
        out=io.StringIO(),
    )
    assert code == 0


def check_no_egress() -> list[str]:
    """Run the library under the guard. Returns the socket calls it attempted."""
    attempts: list[str] = []
    with (
        tempfile.TemporaryDirectory() as workdir,
        _unimported("paveo"),
        _no_sockets(attempts),
    ):
        _exercise_library(Path(workdir))
        _exercise_command(Path(workdir))
    return attempts


def test_no_socket_is_ever_constructed() -> None:
    assert check_no_egress() == []


if __name__ == "__main__":
    made = check_no_egress()
    if made:
        print(f"FAIL: Paveo attempted to open sockets: {made}", file=sys.stderr)
        sys.exit(1)
    print("ok: no socket was constructed during any library operation")
