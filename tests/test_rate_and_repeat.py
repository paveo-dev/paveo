"""B4, piece 2: ``rate`` and ``repeat``, and the guard's memory on disk (D59).

The claims:

1. A limit that is not a whole number in range does not load.
2. ``rate`` refuses the call past its count inside the window, and admits again
   once the window has moved on; ``repeat`` refuses the same call inside its
   window. Only calls the rules admitted count, and a clock that steps back
   refuses more, never less.
3. The ``paveo`` guard remembers across processes: a loop of identical calls
   stops at the limit, hooks run at once cannot both squeeze under it, and a
   memory file that is damaged or a link refuses rather than forgets.
4. Nothing kept or written carries a value.
5. A property: whatever the sequence, no window ever holds more admitted calls
   than the rate allows.
"""

from __future__ import annotations

import errno
import io
import json
import os
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import SENTINEL, records_in
from paveo import ConfigError, Paveo, PolicyDenied, _memory
from paveo._harnesses import CLAUDE_CODE
from paveo._policy_document import load_document
from paveo._replay import History, replay
from paveo.cli import guard

START = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.moment = START

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, seconds: float) -> None:
        self.moment += timedelta(seconds=seconds)


def document(agent: str = "mailer") -> dict[str, object]:
    return {
        "version": 1,
        "policy_id": "b4-rate",
        "agents": [
            {
                "id": agent,
                "tools": {
                    "allow": [
                        {
                            "name": "send_email",
                            "constraints": {"to": {"matches": "[a-z]+"}},
                            "rate": {"calls": 3, "seconds": 60},
                        },
                        {"name": "Bash", "repeat": {"seconds": 30}},
                        {"name": "lookup_order"},
                    ]
                },
            }
        ],
    }


def tool_rule(policy: dict[str, object], name: str) -> dict[str, object]:
    agents = cast("list[dict[str, object]]", policy["agents"])
    tools = cast("dict[str, list[dict[str, object]]]", agents[0]["tools"])
    return next(entry for entry in tools["allow"] if entry["name"] == name)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def pf(tmp_path: Path, clock: Clock) -> Iterator[Paveo]:
    with Paveo.from_policy(
        document(), audit_path=tmp_path / "audit.jsonl", now=clock
    ) as paveo:
        yield paveo


def outcome(call: Callable[[], None]) -> str:
    try:
        call()
    except PolicyDenied as e:
        return e.reason
    return "allow"


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rate", {"calls": 0, "seconds": 60}),
        ("rate", {"calls": 3, "seconds": 0}),
        ("rate", {"calls": 10_001, "seconds": 60}),
        ("rate", {"calls": 3, "seconds": 86_401}),
        ("rate", {"calls": "3", "seconds": 60}),
        ("rate", {"calls": True, "seconds": 60}),
        ("rate", {"calls": 3}),
        ("rate", {"calls": 3, "seconds": 60, "burst": 5}),
        ("rate", 3),
        ("repeat", {"seconds": -1}),
        ("repeat", {"seconds": "30"}),
        ("repeat", {"within": 30}),
        ("repeat", {}),
    ],
)
def test_a_limit_out_of_range_does_not_load(key: str, value: object) -> None:
    policy = document()
    tool_rule(policy, "lookup_order")[key] = value
    with pytest.raises(ConfigError):
        load_document(policy)


def test_a_limit_is_part_of_the_hash() -> None:
    changed = document()
    tool_rule(changed, "Bash")["repeat"] = {"seconds": 31}
    assert load_document(document()).policy_hash != load_document(changed).policy_hash


# --------------------------------------------------------------------------
# 2. In a library session
# --------------------------------------------------------------------------


def test_rate_refuses_past_its_count_and_admits_once_the_window_moves(
    pf: Paveo, clock: Clock
) -> None:
    with pf.session(agent_id="mailer") as s:
        send = lambda: s.check_tool("send_email", {"to": "alice"})  # noqa: E731
        assert [outcome(send) for _ in range(4)] == ["allow"] * 3 + ["rate_limited"]
        clock.advance(59)
        assert outcome(send) == "rate_limited"
        clock.advance(1.5)
        assert outcome(send) == "allow"


def test_refused_calls_do_not_use_up_the_rate(pf: Paveo) -> None:
    with pf.session(agent_id="mailer") as s:
        for _ in range(5):
            assert outcome(lambda: s.check_tool("send_email", {"to": "BAD"})) == (
                "constraint_violated"
            )
        for _ in range(3):
            s.check_tool("send_email", {"to": "alice"})


def test_repeat_refuses_the_same_call_inside_its_window(
    pf: Paveo, clock: Clock
) -> None:
    with pf.session(agent_id="mailer") as s:
        s.check_tool("Bash", {"command": "make test", "timeout": 5})
        # The same arguments in another order are the same call.
        again = lambda: s.check_tool("Bash", {"timeout": 5, "command": "make test"})  # noqa: E731
        assert outcome(again) == "repeated"
        s.check_tool("Bash", {"command": "make lint", "timeout": 5})
        clock.advance(31)
        assert outcome(again) == "allow"


def test_a_clock_that_steps_back_refuses_more_not_less(pf: Paveo, clock: Clock) -> None:
    with pf.session(agent_id="mailer") as s:
        send = lambda: s.check_tool("send_email", {"to": "alice"})  # noqa: E731
        for _ in range(3):
            send()
        clock.advance(-3600)
        assert outcome(send) == "rate_limited"
        # Forgotten calls do not fall back inside the window: moved far ahead,
        # then far back, the clock is held at its latest reading.
        clock.advance(3600 + 120)
        send()
        clock.advance(-7200)
        send()
        send()
        assert outcome(send) == "rate_limited"


def test_in_shadow_mode_a_let_through_call_is_not_counted(
    tmp_path: Path, clock: Clock
) -> None:
    policy = document()
    cast("list[dict[str, object]]", policy["agents"])[0]["mode"] = "shadow"
    log = tmp_path / "a.jsonl"
    with (
        Paveo.from_policy(policy, audit_path=log, now=clock) as pf,
        pf.session(agent_id="mailer") as s,
    ):
        for _ in range(5):
            s.check_tool("send_email", {"to": "alice"})
    decisions = [(r["decision"], r["reason"]) for r in records_in(log)]
    # Three admitted, then two that enforcing would refuse, each still refused
    # rather than counted as though it had been admitted.
    assert decisions[:3] == [("allow", None)] * 3
    assert decisions[3:] == [("would_deny", "rate_limited"), ("allow", None)] * 2


def test_a_session_entered_again_forgets_its_calls(pf: Paveo) -> None:
    s = pf.session(agent_id="mailer")
    with s:
        for _ in range(3):
            s.check_tool("send_email", {"to": "alice"})
    with s:
        s.check_tool("send_email", {"to": "alice"})


def test_arguments_that_cannot_be_compared_are_refused(pf: Paveo) -> None:
    with pf.session(agent_id="mailer") as s:
        strange = lambda: s.check_tool("Bash", {"command": object()})  # noqa: E731
        assert outcome(strange) == "not_comparable"


# --------------------------------------------------------------------------
# 3. The guard, across processes
# --------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / ".paveo"
    directory.mkdir()
    (directory / "policy.json").write_text(
        json.dumps(document("claude-code")), encoding="utf-8"
    )
    return directory


def hook(
    home: Path, tool: str, tool_input: dict[str, object], clock: Clock
) -> tuple[int, str]:
    stderr = io.StringIO()
    payload = {"session_id": "sess-1", "tool_name": tool, "tool_input": tool_input}
    code = guard(
        CLAUDE_CODE,
        directory=home,
        agent="claude-code",
        stdin=io.BytesIO(json.dumps(payload).encode()),
        stdout=io.StringIO(),
        stderr=stderr,
        stopped=lambda: False,
        now=clock,
        salt=lambda n: b"\x07" * n,
    )
    return code, stderr.getvalue()


def test_a_loop_of_identical_calls_stops_at_the_limit_through_the_hook(
    home: Path, clock: Clock
) -> None:
    command = {"command": "curl https://api.example.com/retry"}
    first, second = (
        hook(home, "Bash", command, clock),
        hook(home, "Bash", command, clock),
    )
    assert first == (0, "")
    assert second[0] == 2
    assert "(repeated)" in second[1]
    for _ in range(3):
        assert hook(home, "send_email", {"to": "alice"}, clock) == (0, "")
    code, said = hook(home, "send_email", {"to": "alice"}, clock)
    assert code == 2
    assert "(rate_limited)" in said


def test_hooks_run_at_once_cannot_both_squeeze_under_the_rate(
    home: Path, clock: Clock
) -> None:
    codes: list[int] = []
    lock = threading.Lock()

    def one() -> None:
        code, _ = hook(home, "send_email", {"to": "alice"}, clock)
        with lock:
            codes.append(code)

    threads = [threading.Thread(target=one) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(codes) == [0] * 3 + [2] * 9


def test_a_damaged_memory_file_refuses_rather_than_forgets(
    home: Path, clock: Clock
) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    (memory,) = (home / "memory").glob("*.json")
    memory.write_text('{"version": 1, "salt": "zz"}', encoding="utf-8")
    code, said = hook(home, "send_email", {"to": "alice"}, clock)
    assert code == 2
    assert "(memory_unavailable)" in said
    assert records_in(home / "audit.jsonl")[-1]["reason"] == "memory_unavailable"


def test_a_memory_file_that_is_a_link_is_refused(
    home: Path, clock: Clock, tmp_path: Path
) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    (memory,) = (home / "memory").glob("*.json")
    memory.unlink()
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}", encoding="utf-8")
    memory.symlink_to(elsewhere)
    assert hook(home, "send_email", {"to": "alice"}, clock)[0] == 2
    assert elsewhere.read_text(encoding="utf-8") == "{}"


def test_the_memory_folder_keeps_itself_out_of_git(home: Path, clock: Clock) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    assert (home / "memory" / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_a_policy_with_no_remembering_rule_writes_no_memory(tmp_path: Path) -> None:
    directory = tmp_path / ".paveo"
    directory.mkdir()
    plain = document("claude-code")
    tool_rule(plain, "Bash").pop("repeat")
    tool_rule(plain, "send_email").pop("rate")
    (directory / "policy.json").write_text(json.dumps(plain), encoding="utf-8")
    assert hook(directory, "Bash", {"command": "ls"}, Clock()) == (0, "")
    assert not (directory / "memory").exists()


def test_replay_counts_remembering_rules_apart_rather_than_as_refusals() -> None:
    history = History()
    history.tool_calls.extend(
        [("t1", "Bash", {"command": "ls"}), ("t2", "lookup_order", {})]
    )
    lines = replay(
        history,
        policy=load_document(document("claude-code")),
        agent="claude-code",
        now=START,
    )
    text = "\n".join(lines)
    assert "allowed 1, refused 0" in text
    assert "not judged: 1 calls to tools with a requires, rate or repeat rule" in text


# --------------------------------------------------------------------------
# 4. No value is kept or written
# --------------------------------------------------------------------------


def test_the_memory_file_and_every_refusal_hold_no_value(
    home: Path, clock: Clock
) -> None:
    secret = {"command": f"echo {SENTINEL}"}
    hook(home, "Bash", secret, clock)
    code, said = hook(home, "Bash", secret, clock)
    assert code == 2
    assert SENTINEL not in said
    (memory,) = (home / "memory").glob("*.json")
    assert SENTINEL not in memory.read_text(encoding="utf-8")
    assert SENTINEL not in (home / "audit.jsonl").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 5. The property
# --------------------------------------------------------------------------

STEPS = st.lists(
    st.tuples(st.floats(min_value=-5, max_value=40), st.sampled_from(["a", "BAD"])),
    max_size=40,
)


@settings(max_examples=150, deadline=None)
@given(steps=STEPS)
def test_no_window_ever_holds_more_admitted_calls_than_the_rate(
    tmp_path_factory: pytest.TempPathFactory, steps: list[tuple[float, str]]
) -> None:
    clock = Clock()
    admitted: list[datetime] = []
    latest = START
    log = tmp_path_factory.mktemp("prop") / "a.jsonl"
    with (
        Paveo.from_policy(document(), audit_path=log, now=clock) as pf,
        pf.session(agent_id="mailer") as s,
    ):
        for step, who in steps:
            clock.advance(step)
            # Time as the rules see it: moved on by the clock's forward steps,
            # never back (D59).
            latest += timedelta(seconds=max(0.0, step))
            to = {"to": f"{who}"}
            if outcome(lambda to=to: s.check_tool("send_email", to)) == "allow":
                admitted.append(latest)
    # Every fourth admitted call is at least the window after the first of the
    # three before it, however the clock moved: a call exactly the window old
    # has left it.
    for earlier, later in zip(admitted, admitted[3:], strict=False):
        assert later - earlier >= timedelta(seconds=60)


# --------------------------------------------------------------------------
# 6. What the reviews found (/code-review, /security-review, D59)
# --------------------------------------------------------------------------


def test_a_write_that_fails_keeps_the_old_memory_rather_than_none(
    home: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _ in range(3):
        hook(home, "send_email", {"to": "alice"}, clock)

    def full(*_: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patched:
        patched.setattr(os, "pwrite", full)
        # The rate is used up, so this is refused before anything is written.
        assert hook(home, "send_email", {"to": "alice"}, clock)[0] == 2
        # A call that would be admitted fails to be remembered: refused.
        assert hook(home, "Bash", {"command": "ls"}, clock)[0] == 2
    # The memory still holds the three: the rate is still used up.
    code, said = hook(home, "send_email", {"to": "alice"}, clock)
    assert code == 2
    assert "(rate_limited)" in said


def test_an_interrupted_write_refuses_rather_than_starting_afresh(
    home: Path, clock: Clock
) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    (memory,) = (home / "memory").glob("*.json")
    whole = memory.read_bytes()
    memory.write_bytes(whole[: len(whole) // 2])  # a write cut short
    code, said = hook(home, "send_email", {"to": "alice"}, clock)
    assert code == 2
    assert "(memory_unavailable)" in said
    assert records_in(home / "audit.jsonl")[-1]["reason"] == "memory_unavailable"


def test_repeat_with_same_ignores_what_the_model_rephrases(
    tmp_path: Path, clock: Clock
) -> None:
    policy = document()
    tool_rule(policy, "Bash")["constraints"] = {"command": {}, "description": {}}
    tool_rule(policy, "Bash")["repeat"] = {"seconds": 30, "same": ["command"]}
    with (
        Paveo.from_policy(policy, audit_path=tmp_path / "b.jsonl", now=clock) as pb,
        pb.session(agent_id="mailer") as s,
    ):
        s.check_tool("Bash", {"command": "curl -X POST /charge", "description": "try"})
        again = {"command": "curl -X POST /charge", "description": "try again"}
        assert outcome(lambda: s.check_tool("Bash", again)) == "repeated"


@pytest.mark.parametrize(
    "same", [[], ["command", "command"], ["undeclared"], "command"]
)
def test_a_repeat_same_that_could_not_work_does_not_load(same: object) -> None:
    policy = document()
    tool_rule(policy, "Bash")["constraints"] = {"command": {}}
    tool_rule(policy, "Bash")["repeat"] = {"seconds": 30, "same": same}
    with pytest.raises(ConfigError):
        load_document(policy)


def test_a_clock_that_jumps_ahead_and_back_does_not_freeze_the_window(
    pf: Paveo, clock: Clock
) -> None:
    with pf.session(agent_id="mailer") as s:
        send = lambda: s.check_tool("send_email", {"to": "alice"})  # noqa: E731
        clock.advance(6 * 3600)  # a clock six hours fast, seen by one call
        send()
        clock.advance(-6 * 3600)  # then corrected
        send()
        send()
        assert outcome(send) == "rate_limited"
        # Not frozen until the clock catches up six hours: one real minute.
        clock.advance(61)
        assert outcome(send) == "allow"


def test_a_file_holding_more_than_the_guard_keeps_is_refused(
    home: Path, clock: Clock
) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    (memory,) = (home / "memory").glob("*.json")
    document_ = json.loads(memory.read_text(encoding="utf-8"))
    document_["footprints"] = [
        ["lookup_order", [], f"sha256:{n:064x}"]
        for n in range(_memory._MAX_FOOTPRINTS + 1)
    ]
    memory.write_text(json.dumps(document_), encoding="utf-8")
    assert hook(home, "send_email", {"to": "alice"}, clock)[0] == 2


def test_a_memory_folder_others_can_write_to_is_refused(
    home: Path, clock: Clock
) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    (home / "memory").chmod(0o770)
    assert hook(home, "send_email", {"to": "alice"}, clock)[0] == 2


def test_a_memory_file_with_a_second_name_is_refused(
    home: Path, clock: Clock, tmp_path: Path
) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    (memory,) = (home / "memory").glob("*.json")
    os.link(memory, tmp_path / "second-name.json")
    assert hook(home, "send_email", {"to": "alice"}, clock)[0] == 2


def test_a_new_session_sweeps_memory_untouched_for_a_week(
    home: Path, clock: Clock
) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    (old,) = (home / "memory").glob("*.json")
    week = START.timestamp() - 8 * 86_400
    os.utime(old, (week, week))
    recent = home / "memory" / ("b" * 32 + ".json")
    recent.write_text("{}", encoding="utf-8")
    recent.chmod(0o600)

    stderr = io.StringIO()
    payload = {"session_id": "sess-2", "tool_name": "lookup_order", "tool_input": {}}
    guard(
        CLAUDE_CODE,
        directory=home,
        agent="claude-code",
        stdin=io.BytesIO(json.dumps(payload).encode()),
        stdout=io.StringIO(),
        stderr=stderr,
        stopped=lambda: False,
        now=clock,
    )
    # lookup_order has no remembering rule, so sess-2 made no file: sweep again
    # from one that does.
    payload["tool_name"], payload["tool_input"] = "send_email", {"to": "alice"}
    guard(
        CLAUDE_CODE,
        directory=home,
        agent="claude-code",
        stdin=io.BytesIO(json.dumps(payload).encode()),
        stdout=io.StringIO(),
        stderr=stderr,
        stopped=lambda: False,
        now=clock,
    )
    assert not old.exists()
    assert recent.exists()


# --------------------------------------------------------------------------
# 7. The second review (/code-review over the fixes, D59)
# --------------------------------------------------------------------------


def test_a_refused_lookup_whose_record_failed_leaves_no_footprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from paveo.audit import AuditLog  # noqa: PLC0415
    from paveo.errors import PolicyUnavailable  # noqa: PLC0415

    policy = document()
    tool_rule(policy, "lookup_order")["constraints"] = {"order_id": {}}
    agents = cast("list[dict[str, object]]", policy["agents"])
    tools = cast("dict[str, list[dict[str, object]]]", agents[0]["tools"])
    tools["allow"].append(
        {"name": "refund", "requires": {"tool": "lookup_order", "same": ["order_id"]}}
    )
    real = AuditLog.append
    with (
        Paveo.from_policy(policy, audit_path=tmp_path / "a.jsonl") as pf,
        pf.session(agent_id="mailer") as s,
    ):

        def fails(_self: AuditLog, _record: object) -> str:
            raise PolicyUnavailable("the log is full.", remedy="free space.")

        monkeypatch.setattr(AuditLog, "append", fails)
        with pytest.raises(PolicyUnavailable):
            s.check_tool("lookup_order", {"order_id": "A-1"})
        monkeypatch.setattr(AuditLog, "append", real)
        refused = outcome(lambda: s.check_tool("refund", {"order_id": "A-1"}))
    assert refused == "requires_unmet"


class _DiskFullForMemory:
    """``os`` as ``_memory`` sees it, with a full disk; the audit log's ``os``
    is untouched, so only the memory fails."""

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)

    @staticmethod
    def pwrite(*_: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")


def test_memory_that_cannot_be_written_is_a_recorded_refusal(
    home: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_memory, "os", _DiskFullForMemory())
    code, said = hook(home, "send_email", {"to": "alice"}, clock)
    assert code == 2
    assert "(memory_unavailable)" in said
    last = records_in(home / "audit.jsonl")[-1]
    assert (last["decision"], last["reason"]) == ("deny", "memory_unavailable")


def test_two_agents_in_one_session_count_apart(tmp_path: Path) -> None:
    home = tmp_path / ".paveo"
    home.mkdir()
    policy = document("a")
    other = document("b")
    cast("list[object]", policy["agents"]).extend(cast("list[object]", other["agents"]))
    (home / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    clock = Clock()

    def as_agent(agent: str) -> int:
        payload = {
            "session_id": "shared",
            "tool_name": "send_email",
            "tool_input": {"to": "alice"},
        }
        return guard(
            CLAUDE_CODE,
            directory=home,
            agent=agent,
            stdin=io.BytesIO(json.dumps(payload).encode()),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            stopped=lambda: False,
            now=clock,
        )

    assert [as_agent("a") for _ in range(4)] == [0, 0, 0, 2]
    assert [as_agent("b") for _ in range(3)] == [0, 0, 0]


def test_a_file_swept_between_open_and_lock_is_opened_again(
    home: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    (memory,) = (home / "memory").glob("*.json")
    real = _memory.fcntl.flock
    swept = []

    def sweep_first(fd: int, operation: int) -> None:
        if not swept and operation == _memory.fcntl.LOCK_EX:
            swept.append(True)
            memory.unlink()  # another guard's sweep, just before this lock
        real(fd, operation)

    monkeypatch.setattr(_memory.fcntl, "flock", sweep_first)
    assert hook(home, "send_email", {"to": "alice"}, clock) == (0, "")
    assert swept
    (again,) = (home / "memory").glob("*.json")
    assert again == memory  # the call was kept in the file at the path


def test_a_tool_no_rule_watches_does_not_touch_the_memory(
    home: Path, clock: Clock
) -> None:
    assert hook(home, "lookup_order", {}, clock) == (0, "")
    assert not (home / "memory").exists()


def test_a_corrected_clock_frees_the_limit_after_one_window_through_the_hook(
    home: Path, clock: Clock
) -> None:
    clock.advance(86_400)  # a day fast
    for _ in range(3):
        assert hook(home, "send_email", {"to": "alice"}, clock)[0] == 0
    clock.advance(-86_400)  # corrected
    assert hook(home, "send_email", {"to": "alice"}, clock)[0] == 2
    clock.advance(61)
    # One real window later, not a day later (/security-review, D59).
    assert hook(home, "send_email", {"to": "alice"}, clock) == (0, "")


@pytest.mark.parametrize(
    "broken",
    [
        {"footprints": "abc"},
        {"calls": [["send_email", 1.0]]},
        {"calls": [["send_email", 1.0, "d", "extra"]]},
        {"footprints": [["lookup_order", "same", "d"]]},
    ],
)
def test_a_memory_file_of_the_wrong_shape_is_refused(
    home: Path, clock: Clock, broken: dict[str, object]
) -> None:
    hook(home, "send_email", {"to": "alice"}, clock)
    (memory,) = (home / "memory").glob("*.json")
    document_ = json.loads(memory.read_text(encoding="utf-8"))
    document_.update(broken)
    memory.write_text(json.dumps(document_), encoding="utf-8")
    code, said = hook(home, "send_email", {"to": "alice"}, clock)
    assert code == 2
    assert "(memory_unavailable)" in said
    assert records_in(home / "audit.jsonl")[-1]["reason"] == "memory_unavailable"


def test_shadow_mode_previews_a_call_that_came_with_no_session_id(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".paveo"
    home.mkdir()
    policy = document("claude-code")
    cast("list[dict[str, object]]", policy["agents"])[0]["mode"] = "shadow"
    (home / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    payload = {"tool_name": "send_email", "tool_input": {"to": "alice"}}
    code = guard(
        CLAUDE_CODE,
        directory=home,
        agent="claude-code",
        stdin=io.BytesIO(json.dumps(payload).encode()),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        stopped=lambda: False,
        now=Clock(),
    )
    assert code == 0
    assert [(r["decision"], r["reason"]) for r in records_in(home / "audit.jsonl")] == [
        ("would_deny", "memory_unavailable"),
        ("allow", None),
    ]
