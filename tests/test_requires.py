"""B4, piece 1: ``requires``, a tool only after another on the same item (D58).

The claims, each tested through the public surface where there is one:

1. A policy whose ``requires`` could never be met does not load.
2. In a session, the tool is refused until an earlier *admitted* call to the other
   tool had the same values for the named arguments, and is permitted after.
3. Nothing the model chose is kept or written: the session remembers digests, and
   no refusal, record or footprint carries a value.
4. A checkpoint with no memory refuses the tool rather than skipping the rule:
   ``paveo guard``, which starts fresh on every call, is one.
5. A property test: across any sequence of calls, a refund is admitted exactly
   when an earlier admitted lookup named the same order.
"""

from __future__ import annotations

import copy
import io
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import SENTINEL, make_clock, records_in
from paveo import ConfigError, Paveo, PolicyDenied
from paveo._harnesses import CLAUDE_CODE
from paveo._policy_document import load_document
from paveo.cli import guard
from paveo.policy import Recall


def document() -> dict[str, object]:
    return {
        "version": 1,
        "policy_id": "b4-requires",
        "agents": [
            {
                "id": "refund-bot",
                "tools": {
                    "allow": [
                        {
                            "name": "lookup_order",
                            "constraints": {"order_id": {"matches": "[A-Z0-9-]{1,40}"}},
                        },
                        {
                            "name": "refund",
                            "constraints": {
                                "order_id": {},
                                "amount_usd": {"max": "100.00"},
                            },
                            "requires": {"tool": "lookup_order", "same": ["order_id"]},
                        },
                        {"name": "close_ticket", "requires": {"tool": "lookup_order"}},
                    ],
                    "deny": ["transfer_funds"],
                },
            }
        ],
    }


def tools_of(policy: dict[str, object]) -> dict[str, object]:
    agents = cast("list[dict[str, object]]", policy["agents"])
    return cast("dict[str, object]", agents[0]["tools"])


def rule(policy: dict[str, object], name: str) -> dict[str, object]:
    allow = cast("list[dict[str, object]]", tools_of(policy)["allow"])
    return next(entry for entry in allow if entry["name"] == name)


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


@pytest.fixture
def pf(log: Path) -> Iterator[Paveo]:
    with Paveo.from_policy(document(), audit_path=log, now=make_clock()) as paveo:
        yield paveo


def refusal(pf: Paveo, *calls: tuple[str, dict[str, object]]) -> PolicyDenied:
    """Make ``calls`` in one session; the last must be refused, and is returned."""
    *before, (tool, arguments) = calls
    with pf.session(agent_id="refund-bot") as s:
        for earlier_tool, earlier_arguments in before:
            s.check_tool(earlier_tool, earlier_arguments)
        with pytest.raises(PolicyDenied) as refused:
            s.check_tool(tool, arguments)
    return refused.value


# --------------------------------------------------------------------------
# 1. A requires that could never be met does not load
# --------------------------------------------------------------------------


def broken(edit: str) -> dict[str, object]:
    policy = document()
    refund = rule(policy, "refund")
    if edit == "itself":
        refund["requires"] = {"tool": "refund"}
    elif edit == "not allowed":
        refund["requires"] = {"tool": "issue_credit"}
    elif edit == "denied":
        refund["requires"] = {"tool": "transfer_funds"}
    elif edit == "denied and allowed":
        tools_of(policy)["deny"] = ["lookup_order"]
    elif edit == "undeclared here":
        refund["requires"] = {"tool": "lookup_order", "same": ["customer_id"]}
    elif edit == "undeclared there":
        cast("dict[str, object]", refund["constraints"])["customer_id"] = {}
        refund["requires"] = {"tool": "lookup_order", "same": ["customer_id"]}
    elif edit == "listed twice":
        refund["requires"] = {"tool": "lookup_order", "same": ["order_id", "order_id"]}
    elif edit == "unknown key":
        refund["requires"] = {"tool": "lookup_order", "sam": ["order_id"]}
    elif edit == "no tool":
        refund["requires"] = {"same": ["order_id"]}
    elif edit == "not an object":
        refund["requires"] = "lookup_order"
    elif edit == "a cycle":
        rule(policy, "lookup_order")["requires"] = {"tool": "close_ticket"}
    elif edit == "a longer cycle":
        allow = cast("list[dict[str, object]]", tools_of(policy)["allow"])
        allow.append({"name": "open_ticket", "requires": {"tool": "refund"}})
        rule(policy, "lookup_order")["requires"] = {"tool": "open_ticket"}
    return policy


@pytest.mark.parametrize(
    "edit",
    [
        "itself",
        "not allowed",
        "denied",
        "denied and allowed",
        "undeclared here",
        "undeclared there",
        "listed twice",
        "unknown key",
        "no tool",
        "not an object",
        "a cycle",
        "a longer cycle",
    ],
)
def test_a_requires_that_could_never_be_met_does_not_load(edit: str) -> None:
    with pytest.raises(ConfigError):
        load_document(broken(edit))


@pytest.mark.parametrize("edit", ["itself", "a cycle", "a longer cycle"])
def test_a_chain_that_comes_back_names_itself(edit: str) -> None:
    with pytest.raises(ConfigError, match="can never be met"):
        load_document(broken(edit))


def test_the_example_loads_and_requires_is_part_of_the_hash() -> None:
    plain = document()
    del rule(plain, "refund")["requires"]
    assert load_document(document()).policy_hash != load_document(plain).policy_hash


def test_a_denied_tool_is_not_held_to_its_requires() -> None:
    # The refund flow switched off: every rule still names a tool on deny.
    policy = document()
    tools_of(policy)["deny"] = ["refund", "lookup_order", "close_ticket"]
    load_document(policy)


def test_an_unconstrained_tool_may_compare_any_argument() -> None:
    policy = document()
    tools_of(policy)["allow"] = [
        {"name": "lookup_order"},
        {"name": "refund", "requires": {"tool": "lookup_order", "same": ["order_id"]}},
    ]
    load_document(policy)


# --------------------------------------------------------------------------
# 2. Refused until the earlier call, permitted after
# --------------------------------------------------------------------------


def test_a_refund_with_no_lookup_is_refused(pf: Paveo) -> None:
    refused = refusal(pf, ("refund", {"order_id": "A-1", "amount_usd": "10"}))
    assert refused.reason == "requires_unmet"
    assert refused.rule == "refund.requires.lookup_order"


def test_a_refund_after_a_lookup_of_the_same_order_is_permitted(
    pf: Paveo, log: Path
) -> None:
    with pf.session(agent_id="refund-bot") as s:
        s.check_tool("lookup_order", {"order_id": "A-1"})
        s.check_tool("refund", {"order_id": "A-1", "amount_usd": "10"})
        # Once looked up, the order may be refunded more than once: `requires` is
        # about order, and how many is B4's next piece.
        s.check_tool("refund", {"order_id": "A-1", "amount_usd": "5"})
    assert [r["decision"] for r in records_in(log)] == ["allow"] * 3


def test_a_lookup_of_another_order_does_not_count(pf: Paveo) -> None:
    refused = refusal(
        pf,
        ("lookup_order", {"order_id": "A-1"}),
        ("refund", {"order_id": "B-2", "amount_usd": "10"}),
    )
    assert refused.reason == "requires_unmet"


def test_a_refused_lookup_does_not_count(pf: Paveo) -> None:
    with pf.session(agent_id="refund-bot") as s:
        with pytest.raises(PolicyDenied):
            s.check_tool("lookup_order", {"order_id": "a lowercase id"})
        with pytest.raises(PolicyDenied) as refused:
            s.check_tool("refund", {"order_id": "a lowercase id", "amount_usd": "1"})
    assert refused.value.reason == "requires_unmet"


def test_a_refund_over_its_limit_is_refused_even_after_the_lookup(pf: Paveo) -> None:
    refused = refusal(
        pf,
        ("lookup_order", {"order_id": "A-1"}),
        ("refund", {"order_id": "A-1", "amount_usd": "500"}),
    )
    assert refused.reason == "constraint_violated"


def test_the_compared_argument_must_be_supplied(pf: Paveo) -> None:
    refused = refusal(
        pf, ("lookup_order", {"order_id": "A-1"}), ("refund", {"amount_usd": "1"})
    )
    assert (refused.reason, refused.rule) == ("argument_missing", "refund.order_id")


def test_with_nothing_compared_any_earlier_call_meets_it(pf: Paveo) -> None:
    refused = refusal(pf, ("close_ticket", {}))
    assert refused.reason == "requires_unmet"
    with pf.session(agent_id="refund-bot") as s:
        s.check_tool("lookup_order", {"order_id": "Z-9"})
        s.check_tool("close_ticket", {"ticket": "anything"})


def test_memory_belongs_to_one_session(pf: Paveo) -> None:
    with pf.session(agent_id="refund-bot") as s:
        s.check_tool("lookup_order", {"order_id": "A-1"})
    refused = refusal(pf, ("refund", {"order_id": "A-1", "amount_usd": "1"}))
    assert refused.reason == "requires_unmet"


def test_a_session_entered_again_starts_with_no_memory(pf: Paveo) -> None:
    s = pf.session(agent_id="refund-bot")
    with s:
        s.check_tool("lookup_order", {"order_id": "A-1"})
    with s, pytest.raises(PolicyDenied) as refused:
        s.check_tool("refund", {"order_id": "A-1", "amount_usd": "1"})
    assert refused.value.reason == "requires_unmet"


def test_values_are_compared_with_their_types(tmp_path: Path) -> None:
    policy = document()
    tools_of(policy)["allow"] = [
        {"name": "lookup_order"},
        {"name": "refund", "requires": {"tool": "lookup_order", "same": ["order_id"]}},
    ]
    with Paveo.from_policy(policy, audit_path=tmp_path / "a.jsonl") as pf:
        refused = refusal(
            pf, ("lookup_order", {"order_id": 1}), ("refund", {"order_id": "1"})
        )
        assert refused.reason == "requires_unmet"


class Shouting(str):
    """A str subclass: exact types only, since a subclass may serialise as anything."""


@pytest.mark.parametrize(
    ("looked_up", "refunded"),
    ids=[
        "object",
        "nan",
        "float",
        "list",
        "int-vs-str-keys",
        "tuple-vs-list",
        "str-subclass",
        "huge-int",
    ],
    argvalues=[
        (object(), None),
        (float("nan"), None),
        (1.5, 1.5),
        (["A-1"], ["A-1"]),
        # Containers serialise alike when they differ (/code-review, D58).
        ({1: "x"}, {"1": "x"}),
        (("A-1",), ["A-1"]),
        (Shouting("A-1"), Shouting("A-1")),
        (10**5000, 10**5000),
    ],
)
def test_only_a_plain_scalar_can_meet_it(
    tmp_path: Path, looked_up: object, refunded: object
) -> None:
    policy = document()
    tools_of(policy)["allow"] = [
        {"name": "lookup_order"},
        {"name": "refund", "requires": {"tool": "lookup_order", "same": ["order_id"]}},
    ]
    same = looked_up if refunded is None else refunded
    with Paveo.from_policy(policy, audit_path=tmp_path / "a.jsonl") as pf:
        refused = refusal(
            pf,
            ("lookup_order", {"order_id": looked_up}),
            ("refund", {"order_id": same}),
        )
    assert refused.reason == "requires_unmet"


@pytest.mark.parametrize("value", ["A-1", 7, True, None])
def test_each_plain_scalar_meets_it(tmp_path: Path, value: object) -> None:
    policy = document()
    tools_of(policy)["allow"] = [
        {"name": "lookup_order"},
        {"name": "refund", "requires": {"tool": "lookup_order", "same": ["order_id"]}},
    ]
    with (
        Paveo.from_policy(policy, audit_path=tmp_path / "a.jsonl") as pf,
        pf.session(agent_id="refund-bot") as s,
    ):
        s.check_tool("lookup_order", {"order_id": value})
        s.check_tool("refund", {"order_id": value})


def test_in_shadow_mode_the_refusal_is_recorded_and_the_call_goes_on(
    tmp_path: Path,
) -> None:
    policy = document()
    cast("list[dict[str, object]]", policy["agents"])[0]["mode"] = "shadow"
    log = tmp_path / "a.jsonl"
    with (
        Paveo.from_policy(policy, audit_path=log, now=make_clock()) as pf,
        pf.session(agent_id="refund-bot") as s,
    ):
        s.check_tool("refund", {"order_id": "A-1", "amount_usd": "1"})
        # A lookup shadow mode let through is not a lookup the rules admitted, so
        # the refund after it is still one enforcing would refuse: shadow mode
        # previews both refusals, not one (/code-review, D58).
        s.check_tool("lookup_order", {"order_id": "a lowercase id"})
        s.check_tool("refund", {"order_id": "a lowercase id", "amount_usd": "1"})
    assert [(r["decision"], r["reason"]) for r in records_in(log)] == [
        ("would_deny", "requires_unmet"),
        ("allow", None),
        ("would_deny", "constraint_violated"),
        ("allow", None),
        ("would_deny", "requires_unmet"),
        ("allow", None),
    ]


def test_sessions_on_threads_keep_their_own_memory(pf: Paveo) -> None:
    outcomes: dict[int, tuple[bool, str]] = {}

    def work(index: int) -> None:
        mine, theirs = f"T-{index}", f"T-{index + 100}"
        with pf.session(agent_id="refund-bot") as s:
            s.check_tool("lookup_order", {"order_id": mine})
            s.check_tool("refund", {"order_id": mine, "amount_usd": "1"})
            try:
                s.check_tool("refund", {"order_id": theirs, "amount_usd": "1"})
            except PolicyDenied as e:
                outcomes[index] = (True, e.reason)
            else:
                outcomes[index] = (False, "")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == dict.fromkeys(range(16), (True, "requires_unmet"))


# --------------------------------------------------------------------------
# 3. No value is kept or written
# --------------------------------------------------------------------------


def test_no_refusal_record_or_footprint_carries_a_value(pf: Paveo, log: Path) -> None:
    refused = refusal(
        pf,
        ("lookup_order", {"order_id": "SENTINEL-1"}),
        ("refund", {"order_id": SENTINEL, "amount_usd": "1"}),
    )
    for text in (str(refused), refused.remedy, refused.for_model):
        assert SENTINEL not in text
    assert SENTINEL not in log.read_text(encoding="utf-8")
    assert "SENTINEL-1" not in log.read_text(encoding="utf-8")

    fresh = Recall(now=0.0, footprints=frozenset(), calls=())
    footprints, _ = load_document(document()).remember(
        "refund-bot", "lookup_order", {"order_id": SENTINEL}, fresh
    )
    assert len(footprints) == 2  # one per rule that names lookup_order
    assert SENTINEL not in repr(footprints)


def test_the_model_is_told_what_to_do_without_the_policy(pf: Paveo) -> None:
    text = refusal(pf, ("refund", {"order_id": "A-1", "amount_usd": "1"})).for_model
    assert "(requires_unmet)" in text
    assert "not been done in this session" in text
    assert "tools.allow" not in text
    assert "agents[" not in text


# --------------------------------------------------------------------------
# 4. No memory: refused, never skipped
# --------------------------------------------------------------------------


def test_a_checkpoint_with_no_memory_refuses_the_tool() -> None:
    policy = load_document(document())
    denial = policy.evaluate_tool(
        "refund-bot", "refund", {"order_id": "A-1", "amount_usd": "1"}
    )
    assert denial is not None
    assert denial.reason == "memory_unavailable"
    assert policy.evaluate_tool("refund-bot", "lookup_order", {"order_id": "A"}) is None


def guarded(tmp_path: Path) -> Path:
    policy = document()
    cast("list[dict[str, object]]", policy["agents"])[0]["id"] = "claude-code"
    home = tmp_path / ".paveo"
    home.mkdir()
    (home / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return home


def through_guard(
    home: Path, tool: str, tool_input: dict[str, object], session: str | None = "s"
) -> tuple[int, str]:
    stderr = io.StringIO()
    payload: dict[str, object] = {"tool_name": tool, "tool_input": tool_input}
    if session is not None:
        payload["session_id"] = session
    code = guard(
        CLAUDE_CODE,
        directory=home,
        agent="claude-code",
        stdin=io.BytesIO(json.dumps(payload).encode()),
        stdout=io.StringIO(),
        stderr=stderr,
        stopped=lambda: False,
        now=make_clock(),
    )
    return code, stderr.getvalue()


def test_the_command_guard_remembers_within_a_session(tmp_path: Path) -> None:
    home = guarded(tmp_path)
    refund = {"order_id": "A-1", "amount_usd": "1"}
    code, said = through_guard(home, "refund", refund)
    assert code == 2
    assert "(requires_unmet)" in said
    assert through_guard(home, "lookup_order", {"order_id": "A-1"}) == (0, "")
    assert through_guard(home, "refund", refund) == (0, "")
    # Another session remembers nothing of this one.
    assert through_guard(home, "refund", refund, session="other")[0] == 2


def test_the_memory_file_holds_no_value(tmp_path: Path) -> None:
    home = guarded(tmp_path)
    through_guard(home, "lookup_order", {"order_id": "SENTINEL-9"})
    (memory,) = (home / "memory").glob("*.json")
    assert "SENTINEL-9" not in memory.read_text(encoding="utf-8")
    assert oct(memory.stat().st_mode & 0o777) == oct(0o600)


def test_with_no_session_id_the_guard_refuses_the_tool(tmp_path: Path) -> None:
    home = guarded(tmp_path)
    code, said = through_guard(home, "close_ticket", {}, session=None)
    assert code == 2
    assert "(memory_unavailable)" in said


# --------------------------------------------------------------------------
# 5. The property: admitted exactly when an earlier admitted lookup matched
# --------------------------------------------------------------------------

ORDERS = st.sampled_from(["A-1", "B-2", "C-3", "bad id"])
CALLS = st.lists(
    st.tuples(st.sampled_from(["lookup_order", "refund"]), ORDERS), max_size=25
)


@settings(max_examples=150, deadline=None)
@given(calls=CALLS)
def test_a_refund_is_admitted_exactly_when_its_order_was_looked_up(
    tmp_path_factory: pytest.TempPathFactory, calls: list[tuple[str, str]]
) -> None:
    log = tmp_path_factory.mktemp("prop") / "a.jsonl"
    looked_up: set[str] = set()
    with (
        Paveo.from_policy(copy.deepcopy(document()), audit_path=log) as pf,
        pf.session(agent_id="refund-bot") as s,
    ):
        for tool, order in calls:
            arguments: dict[str, object] = {"order_id": order}
            if tool == "refund":
                arguments["amount_usd"] = "1"
            try:
                s.check_tool(tool, arguments)
            except PolicyDenied:
                admitted = False
            else:
                admitted = True
            if tool == "lookup_order":
                assert admitted == (order != "bad id")
                if admitted:
                    looked_up.add(order)
            else:
                assert admitted == (order in looked_up)


def test_a_footprint_that_cannot_be_kept_does_not_refuse_an_allowed_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from paveo import _memory  # noqa: PLC0415

    home = guarded(tmp_path)

    def cannot(*_: object) -> bool:
        raise ConfigError("the disk is full.", remedy="free space.")

    monkeypatch.setattr(_memory.SessionMemory, "leave", cannot)
    # The record said allow, so the call goes on (/code-review, D59) ...
    assert through_guard(home, "lookup_order", {"order_id": "A-1"}) == (0, "")
    assert records_in(home / "audit.jsonl")[-1]["decision"] == "allow"
    monkeypatch.undo()
    # ... and with no footprint kept, the refund it would have unlocked is refused.
    code, said = through_guard(home, "refund", {"order_id": "A-1", "amount_usd": "1"})
    assert code == 2
    assert "(requires_unmet)" in said
