"""B1: refusals a model can act on, and shadow mode (D48).

Two claims, each tested through the public surface:

1. ``PolicyDenied.for_model`` is text a tool loop can hand back to the model. It
   names the rule and the reason, carries no payload, and **never carries the
   operator's remedy**, which says how to change the policy: handed to a model,
   that is an instruction to edit its own guard.
2. An agent with ``"mode": "shadow"`` has its rule refusals recorded as
   ``would_deny`` and let through, with a warning on every call. **Its budget is
   never shadowed**: the ceiling refuses exactly as it does for an enforced agent.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest

from conftest import SENTINEL, make_clock, records_in
from paveo import (
    BudgetExceeded,
    ConfigError,
    Paveo,
    PolicyDenied,
    PolicyUnavailable,
    PricingUnknown,
    verify_chain,
)


def document(*, mode: str | None = "shadow") -> dict[str, object]:
    shadowed: dict[str, object] = {
        "id": "trial",
        "budget": {"period": "day", "limit_usd": "1.00"},
        "models": {"allow": ["claude-sonnet-5"], "deny": ["claude-fable-5-1"]},
        "tools": {
            "allow": [
                {"name": "lookup_order"},
                {"name": "refund", "constraints": {"amount_usd": {"max": "500.00"}}},
            ],
            "deny": ["transfer_funds"],
        },
    }
    if mode is not None:
        shadowed["mode"] = mode
    return {
        "version": 1,
        "policy_id": "b1",
        "agents": [
            shadowed,
            {
                "id": "enforced",
                "budget": {"period": "day", "limit_usd": "1.00"},
                "models": {"allow": ["claude-sonnet-5"]},
                "tools": {"allow": [{"name": "lookup_order"}]},
            },
            {"id": "no-ceiling", "mode": "shadow", "models": {"allow": []}},
        ],
    }


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


@pytest.fixture
def pf(log: Path) -> Iterator[Paveo]:
    with Paveo.from_policy(document(), audit_path=log, now=make_clock()) as paveo:
        yield paveo


def ask(model: str = "claude-sonnet-5", max_tokens: int = 4_000) -> dict[str, object]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "Draft the reply."}],
    }


def usage(output: int) -> dict[str, object]:
    return {
        "input_tokens": 100,
        "output_tokens": output,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "service_tier": "standard",
    }


def decisions(log: Path) -> list[tuple[object, object]]:
    return [(r["decision"], r["reason"]) for r in records_in(log)]


# --------------------------------------------------------------------------
# 1. The refusal a model can act on
# --------------------------------------------------------------------------

# One call per reason a tool check can refuse with, each carrying the sentinel
# wherever the model chooses content: an argument value, an argument name, a
# tool name.
REFUSED: dict[str, tuple[str, dict[str, object]]] = {
    "tool_denied": ("transfer_funds", {"amount_usd": SENTINEL}),
    "tool_not_allowed": (f"exfil::{SENTINEL}", {}),
    "argument_not_permitted": ("refund", {"amount_usd": 1, SENTINEL: 1}),
    "argument_missing": ("refund", {}),
    "constraint_violated": ("refund", {"amount_usd": f"600{SENTINEL}"}),
}
REFUSED_TOOL_CALLS = [
    pytest.param(*call, id=reason) for reason, call in REFUSED.items()
]


@pytest.mark.parametrize(("tool", "arguments"), REFUSED_TOOL_CALLS)
def test_the_refusal_for_a_model_names_the_rule_and_nothing_else(
    tmp_path: Path, tool: str, arguments: dict[str, object]
) -> None:
    with (
        Paveo.from_policy(
            document(mode=None), audit_path=tmp_path / "a.jsonl", now=make_clock()
        ) as pf,
        pf.session(agent_id="trial") as s,
        pytest.raises(PolicyDenied) as refused,
    ):
        s.check_tool(tool, arguments)

    text = refused.value.for_model
    assert f"{refused.value.rule} ({refused.value.reason})" in text
    assert SENTINEL not in text
    # The operator's remedy says how to change the policy. None of it, and no
    # path into the policy document, reaches the model (D47 #7).
    assert refused.value.remedy not in text
    assert "agents[" not in text
    assert "tools.allow" not in text
    assert "policy" in text  # "refused by policy", "within the policy", no more


def test_each_tool_refusal_tells_the_model_something_different(tmp_path: Path) -> None:
    """A refusal a model can act on says what to do about *this* refusal: a
    missing argument is fixed by supplying it, a denied tool is not fixed at all."""
    texts: set[str] = set()
    with (
        Paveo.from_policy(
            document(mode=None), audit_path=tmp_path / "a.jsonl", now=make_clock()
        ) as pf,
        pf.session(agent_id="trial") as s,
    ):
        for reason, (tool, arguments) in REFUSED.items():
            with pytest.raises(PolicyDenied) as refused:
                s.check_tool(tool, arguments)
            assert refused.value.reason == reason
            texts.add(refused.value.for_model.split(f"({reason}). ", 1)[1])
    # A tool on the deny list and one on no list at all read the same to a model.
    assert len(texts) == 4


def test_a_model_call_refusal_also_has_text_for_the_model(pf: Paveo) -> None:
    with (
        pf.session(agent_id="enforced") as s,
        pytest.raises(PolicyDenied) as refused,
    ):
        s.check_llm(ask(f"claude-{SENTINEL}"), shape="anthropic")

    text = refused.value.for_model
    assert "Do not make this call again." in text
    assert SENTINEL not in text
    assert refused.value.remedy not in text


# --------------------------------------------------------------------------
# 2. Shadow mode: loading
# --------------------------------------------------------------------------


def test_an_agent_enforces_unless_it_says_shadow(log: Path) -> None:
    with Paveo.from_policy(document(mode="enforce"), audit_path=log) as pf:
        with pf.session(agent_id="trial") as s, pytest.raises(PolicyDenied):
            s.check_tool("transfer_funds")
    with Paveo.from_policy(document(mode=None), audit_path=log) as pf:
        with pf.session(agent_id="trial") as s, pytest.raises(PolicyDenied):
            s.check_tool("transfer_funds")


@pytest.mark.parametrize("mode", ["Shadow", "shadow ", "", "off", 1, True, None, []])
def test_a_mode_that_is_not_one_of_the_two_words_does_not_load(
    log: Path, mode: object
) -> None:
    """A typo must not quietly turn enforcement off, nor quietly leave it on."""
    policy = document()
    agents = policy["agents"]
    assert isinstance(agents, list)
    agents[0]["mode"] = mode
    with pytest.raises(ConfigError, match="mode"):
        Paveo.from_policy(policy, audit_path=log)


def test_shadow_mode_is_part_of_the_policy_hash(tmp_path: Path) -> None:
    hashes = set()
    for mode in ("shadow", "enforce"):
        with Paveo.from_policy(
            document(mode=mode), audit_path=tmp_path / f"{mode}.jsonl"
        ) as pf:
            with pf.session(agent_id="trial") as s:
                s.check_tool("lookup_order")
        hashes.add(records_in(tmp_path / f"{mode}.jsonl")[0]["policy_hash"])
    assert len(hashes) == 2


# --------------------------------------------------------------------------
# 3. Shadow mode: tools
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("tool", "arguments"), REFUSED_TOOL_CALLS)
def test_a_shadowed_refusal_is_recorded_and_the_call_goes_on(
    pf: Paveo, log: Path, tool: str, arguments: dict[str, object]
) -> None:
    with pf.session(agent_id="trial") as s:
        assert s.check_tool(tool, arguments) is None  # type: ignore[func-returns-value]  # asserting the "no result to check" contract

    records = records_in(log)
    assert [r["decision"] for r in records] == ["would_deny", "allow"]
    shadowed, allowed = records
    assert shadowed["reason"] is not None
    assert shadowed["rule"] is not None
    assert allowed["reason"] is None
    assert shadowed["action"] == allowed["action"]
    assert SENTINEL not in log.read_text(encoding="utf-8")
    assert verify_chain(log).ok


def test_a_permitted_call_in_shadow_mode_writes_one_record(
    pf: Paveo, log: Path
) -> None:
    with pf.session(agent_id="trial") as s:
        s.check_tool("refund", {"amount_usd": 10})
    assert decisions(log) == [("allow", None)]


def test_shadow_mode_warns_on_every_call(
    pf: Paveo, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="paveo"):
        with pf.session(agent_id="trial") as s:
            s.check_tool("lookup_order")
            s.check_tool("lookup_order")
            s.check_tool("transfer_funds")

    modes = [m for m in caplog.messages if "is in shadow mode" in m]
    let_through = [m for m in caplog.messages if "would refuse" in m]
    assert len(modes) == 3
    assert len(let_through) == 1
    assert "transfer_funds" in let_through[0]


def test_shadow_mode_belongs_to_one_agent(pf: Paveo, log: Path) -> None:
    """Another agent in the same policy, and an agent the policy does not
    declare, are refused as ever."""
    with pf.session(agent_id="enforced") as s, pytest.raises(PolicyDenied):
        s.check_tool("transfer_funds")
    with pf.session(agent_id="stranger") as s, pytest.raises(PolicyDenied) as unknown:
        s.check_tool("lookup_order")
    assert unknown.value.reason == "agent_unknown"
    assert decisions(log) == [("deny", "tool_not_allowed"), ("deny", "agent_unknown")]


def test_a_shadowed_refusal_that_cannot_be_recorded_is_a_refusal(
    pf: Paveo,
) -> None:
    """An unlogged decision did not happen (§7), and a would_deny that cannot be
    written is still a decision."""
    with pf.session(agent_id="trial") as s:
        pf.close()
        with pytest.raises(PolicyUnavailable):
            s.check_tool("transfer_funds")


# --------------------------------------------------------------------------
# 4. Shadow mode: model calls, and the budget it never covers
# --------------------------------------------------------------------------


def test_a_shadowed_model_refusal_is_recorded_and_the_call_is_charged(
    pf: Paveo, log: Path
) -> None:
    with pf.session(agent_id="trial") as s:
        with s.check_llm(ask("claude-opus-5"), shape="anthropic") as call:
            charged = call.record(usage(output=1_000))
        assert s.remaining() == Decimal("1.00") - charged

    records = records_in(log)
    assert decisions(log) == [
        ("would_deny", "model_not_allowed"),
        ("allow", None),
        ("settle", "recorded"),
    ]
    # Not declared by the policy, so not written down (D26), in any of the three.
    assert {str(r["action"]) for r in records} == {
        "{'kind': 'llm', 'model': '<undeclared>'}"
    }
    assert verify_chain(log).ok


def test_a_shadowed_agent_cannot_breach_its_ceiling(pf: Paveo, log: Path) -> None:
    """Locked decision #7. A model the rules refuse is let through; the ceiling
    still refuses the call that would breach it, and spend never exceeds it."""
    spent = Decimal(0)
    refusal: BudgetExceeded | None = None
    with pf.session(agent_id="trial") as s:
        for _ in range(20):
            try:
                call = s.check_llm(ask("claude-opus-5"), shape="anthropic")
            except BudgetExceeded as e:
                refusal = e
                break
            spent += call.record(usage(output=4_000))
        assert s.remaining() == Decimal("1.00") - spent
    assert refusal is not None
    assert Decimal(0) < spent <= Decimal("1.00")
    assert decisions(log)[-2:] == [
        ("would_deny", "model_not_allowed"),
        ("deny", "budget_exceeded"),
    ]


def test_a_call_dearer_than_the_whole_ceiling_is_refused_in_shadow_mode(
    pf: Paveo, log: Path
) -> None:
    """20,000 output tokens of an explicitly denied $50/M model: over $1 alone."""
    with pf.session(agent_id="trial") as s, pytest.raises(BudgetExceeded):
        s.check_llm(ask("claude-fable-5-1", max_tokens=20_000), shape="anthropic")
    assert decisions(log) == [
        ("would_deny", "model_denied"),
        ("deny", "budget_exceeded"),
    ]


def test_no_ceiling_is_not_shadowed(log: Path) -> None:
    """``no_budget`` is the ceiling, not a rule: a shadowed agent with no budget
    has every model call refused, like any other. Alone in its policy: as the
    third agent it would be past the free plan (D74)."""
    policy = document()
    agents = policy["agents"]
    assert isinstance(agents, list)
    policy["agents"] = [a for a in agents if a["id"] == "no-ceiling"]
    with (
        Paveo.from_policy(policy, audit_path=log, now=make_clock()) as pf,
        pf.session(agent_id="no-ceiling") as s,
        pytest.raises(PolicyDenied) as e,
    ):
        s.check_llm(ask("claude-sonnet-5"), shape="anthropic")
    assert e.value.reason == "no_budget"
    assert decisions(log) == [
        ("would_deny", "model_not_allowed"),
        ("deny", "no_budget"),
    ]


def test_a_shadowed_call_that_cannot_be_priced_is_refused(pf: Paveo, log: Path) -> None:
    with pf.session(agent_id="trial") as s, pytest.raises(PricingUnknown) as e:
        s.check_llm(ask(f"claude-{SENTINEL}"), shape="anthropic")
    # A name no rule stopped must not come back in the error either (/code-review).
    assert SENTINEL not in str(e.value)
    assert e.value.model == "<undeclared>"
    assert decisions(log) == [
        ("would_deny", "model_not_allowed"),
        ("deny", "pricing_unknown"),
    ]
    assert SENTINEL not in log.read_text(encoding="utf-8")


def test_many_threads_in_shadow_mode_cannot_overspend_one_ceiling(log: Path) -> None:
    """The concurrent path: sixteen threads, each reserving a model the rules
    refuse and holding it until every thread has asked. What is held never exceeds the
    ceiling, and the chain survives the interleaving."""
    start, holding = threading.Barrier(16), threading.Barrier(16)
    held: list[Decimal] = []
    refused: list[BudgetExceeded] = []
    lock = threading.Lock()

    with Paveo.from_policy(document(), audit_path=log, now=make_clock()) as pf:

        def agent() -> None:
            start.wait()
            with pf.session(agent_id="trial") as s:
                try:
                    call = s.check_llm(ask("claude-opus-5"), shape="anthropic")
                except BudgetExceeded as e:
                    with lock:
                        refused.append(e)
                    holding.wait()
                    return
                with lock:
                    held.append(call._admitted.reservation.worst_case)
                holding.wait()  # every admitted hold is outstanding at once here
                call.release()

        threads = [threading.Thread(target=agent) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert held
    assert sum(held, Decimal(0)) <= Decimal("1.00")
    assert len(held) + len(refused) == 16
    assert verify_chain(log).ok
