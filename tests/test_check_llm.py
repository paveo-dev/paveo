"""The decision point for model calls, end to end (S4, D23, D39).

Every test here drives the public surface: a ``Paveo`` from a policy, a session,
``check_llm``, and the handle it returns. The ledger, the price table and the
audit log are the real ones, so what these prove is what a customer gets.

S4's card is done when **a $1 ceiling refuses the call that would breach it, end
to end, with both audit records on disk and the chain verifying.** That is the
first test.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from datetime import datetime
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

MILLION = Decimal(1_000_000)

# Claude Sonnet 5 with inference_geo unset is reserved at the US rate (D38):
# output $10 x 1.1 per million. The dearest input class is the 1h cache write,
# $2 x 2 x 1.1.
OUTPUT_RATE = Decimal("11") / MILLION
INPUT_CEILING = Decimal("4.4") / MILLION


def document(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "version": 1,
        "policy_id": "s4",
        "agents": [
            {
                "id": "bot",
                "budget": {"period": "day", "limit_usd": "1.00"},
                "models": {
                    "allow": ["claude-sonnet-5", "mistral-large-3"],
                    "deny": ["claude-fable-5-1"],
                },
            },
            {
                "id": "per-session",
                "budget": {"period": "session", "limit_usd": "1.00"},
                "models": {"allow": ["claude-sonnet-5"]},
            },
            {"id": "tools-only", "models": {"allow": ["claude-sonnet-5"]}},
        ],
        "prices": {
            "mistral-large-3": {"input_per_mtok": "2.00", "output_per_mtok": "6.00"}
        },
    }
    base.update(extra)
    return base


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


@pytest.fixture
def pf(log: Path) -> Iterator[Paveo]:
    with Paveo.from_policy(document(), audit_path=log, now=make_clock()) as paveo:
        yield paveo


def ask(max_tokens: int = 20_000, **extra: object) -> dict[str, object]:
    request: dict[str, object] = {
        "model": "claude-sonnet-5",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "Draft the reply."}],
    }
    request.update(extra)
    return request


def usage(output: int, input_tokens: int = 100) -> dict[str, object]:
    """The shape `response.usage.model_dump()` has."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "service_tier": "standard",
    }


def test_a_one_dollar_ceiling_refuses_the_call_that_would_breach_it(
    pf: Paveo, log: Path
) -> None:
    """S4's done-when. Each call may cost up to about $0.22 (20,000 output tokens
    at the US rate). Four are admitted and spend what they spend; the fifth
    would take the day past $1.00 and is refused before it leaves."""
    with pf.session(agent_id="bot", principal="user_1") as s:
        for _ in range(4):
            call = s.check_llm(ask(), shape="anthropic")
            call.record(usage(output=20_000))
        before = s.remaining()

        with pytest.raises(BudgetExceeded) as refused:
            s.check_llm(ask(), shape="anthropic")

        assert s.remaining() == before
    assert refused.value.limit == Decimal("1.00")
    assert refused.value.spent + refused.value.requested > Decimal("1.00")

    records = records_in(log)
    decisions = [(r["decision"], r["reason"]) for r in records]
    assert decisions == [("allow", None), ("settle", "recorded")] * 4 + [
        ("deny", "budget_exceeded")
    ]
    for allow, settle in zip(records[0:8:2], records[1:8:2], strict=True):
        assert allow["reservation_id"] == settle["reservation_id"]
        assert allow["estimated_cost_usd"] is not None
        assert allow["actual_cost_usd"] is None
        assert Decimal(str(settle["actual_cost_usd"])) <= Decimal(
            str(allow["estimated_cost_usd"])
        )
    assert verify_chain(log).ok


def test_an_allowed_call_writes_a_decision_and_then_a_settlement(
    pf: Paveo, log: Path
) -> None:
    with pf.session(agent_id="bot", principal="user_1") as s:
        call = s.check_llm(ask(max_tokens=1000), shape="anthropic")
        charged = call.record(usage(output=500, input_tokens=2000))

    decision, settlement = records_in(log)
    assert charged == 2000 * Decimal("2.2") / MILLION + 500 * OUTPUT_RATE
    assert decision["action"] == {"kind": "llm", "model": "claude-sonnet-5"}
    assert decision["rate_key"] == "claude-sonnet-5|speed=standard|inference_geo=unset"
    assert decision["reservation_id"] == call.reservation_id
    assert decision["token_count_mode"] == "conservative"  # noqa: S105 - a mode, not a secret
    assert decision["prices_version"] == settlement["prices_version"]
    # Zero counts are written too: the record says what the response reported.
    assert settlement["usage_by_class"] == {
        "input": 2000,
        "output": 500,
        "cache_read": 0,
        "cache_write_1h": 0,
    }
    assert Decimal(str(settlement["actual_cost_usd"])) == charged
    assert settlement["principal"] == "user_1"


def test_every_session_of_one_paveo_spends_one_ceiling(pf: Paveo) -> None:
    """D33: one store for every session, or each would get the whole ceiling."""
    with pf.session(agent_id="bot") as first:
        first.check_llm(ask(), shape="anthropic").record(usage(output=20_000))
        left = first.remaining()
    with pf.session(agent_id="bot") as second:
        assert second.remaining() == left


def test_a_session_ceiling_lasts_one_session(pf: Paveo) -> None:
    """§4.7 and D31: its own store, built as the session opens, gone as it ends."""
    with pf.session(agent_id="per-session") as first:
        first.check_llm(ask(), shape="anthropic").record(usage(output=20_000))
        assert first.remaining() < Decimal("1.00")
    with pf.session(agent_id="per-session") as second:
        assert second.remaining() == Decimal("1.00")


def test_a_call_released_before_it_was_sent_costs_nothing(pf: Paveo, log: Path) -> None:
    with pf.session(agent_id="bot") as s:
        s.check_llm(ask(), shape="anthropic").release()
        assert s.remaining() == Decimal("1.00")

    settlement = records_in(log)[-1]
    assert (settlement["decision"], settlement["reason"]) == ("settle", "released")
    assert settlement["actual_cost_usd"] == "0"


def worst_case_of(log: Path) -> Decimal:
    return Decimal(str(records_in(log)[0]["estimated_cost_usd"]))


def test_a_call_never_settled_is_charged_its_worst_case_at_session_end(
    pf: Paveo, log: Path
) -> None:
    """It may have reached the provider and been billed; nothing said otherwise."""
    with pf.session(agent_id="bot") as s:
        s.check_llm(ask(), shape="anthropic")
    worst = worst_case_of(log)

    settlement = records_in(log)[-1]
    assert settlement["reason"] == "unsettled_at_session_end"
    assert Decimal(str(settlement["actual_cost_usd"])) == worst
    with pf.session(agent_id="bot") as s:
        assert s.remaining() == Decimal("1.00") - worst


def test_a_block_that_raises_before_recording_is_charged_its_worst_case(
    pf: Paveo, log: Path
) -> None:
    with pf.session(agent_id="bot") as s:
        with pytest.raises(TimeoutError), s.check_llm(ask(), shape="anthropic"):
            raise TimeoutError("the provider hung up mid-stream")
        assert s.remaining() == Decimal("1.00") - worst_case_of(log)

    assert records_in(log)[-1]["reason"] == "unsettled_at_block_exit"


def test_a_call_settles_exactly_once(pf: Paveo) -> None:
    with pf.session(agent_id="bot") as s:
        call = s.check_llm(ask(), shape="anthropic")
        call.record(usage(output=10))
        with pytest.raises(ConfigError, match="already recorded or released"):
            call.record(usage(output=10))
        with pytest.raises(ConfigError, match="already recorded or released"):
            call.release()


def test_usage_that_cannot_be_read_is_charged_its_worst_case_then_raises(
    pf: Paveo, log: Path
) -> None:
    with pf.session(agent_id="bot") as s:
        call = s.check_llm(ask(), shape="anthropic")
        with pytest.raises(ConfigError, match="full worst case"):
            call.record({"input_tokens": "lots"})
        assert s.remaining() == Decimal("1.00") - worst_case_of(log)

    assert records_in(log)[-1]["reason"] == "usage_unreadable"


@pytest.mark.parametrize(
    ("model", "shape", "usage"),
    [
        ("claude-sonnet-5", "anthropic", {}),
        ("claude-sonnet-5", "anthropic", {"input_tokens": 5}),
        ("claude-sonnet-5", "anthropic", {"output_tokens": 5, "input_tokens": None}),
        ("mistral-large-3", "generic", {}),
        ("mistral-large-3", "generic", {"input": 5}),
        ("mistral-large-3", "generic", {"output": 5, "cache_read": 5}),
    ],
)
def test_usage_without_both_counts_is_unreadable_not_free(
    pf: Paveo, log: Path, model: str, shape: str, usage: dict[str, object]
) -> None:
    """Both once settled an empty usage at $0; now every adapter's reading must
    carry an input and an output count (D44)."""
    with pf.session(agent_id="bot") as s:
        call = s.check_llm(ask(model=model), shape=shape)
        with pytest.raises(ConfigError, match="full worst case"):
            call.record(usage)
        assert s.remaining() == Decimal("1.00") - worst_case_of(log)

    assert records_in(log)[-1]["reason"] == "usage_unreadable"


def test_a_model_the_policy_does_not_allow_is_refused_and_not_named(
    pf: Paveo, log: Path
) -> None:
    """D26 for models: an unlisted name is not written down."""
    with pf.session(agent_id="bot") as s:
        with pytest.raises(PolicyDenied):
            s.check_llm(ask(model=SENTINEL), shape="anthropic")
        with pytest.raises(PolicyDenied, match="claude-fable-5-1"):
            s.check_llm(ask(model="claude-fable-5-1"), shape="anthropic")

    unlisted, denied = records_in(log)
    assert unlisted["action"] == {"kind": "llm", "model": "<undeclared>"}
    assert unlisted["reason"] == "model_not_allowed"
    assert denied["reason"] == "model_denied"
    assert SENTINEL not in log.read_text()


def test_an_agent_with_no_budget_may_call_no_model(log: Path) -> None:
    # Alone in its policy: as the third agent it would be past the free plan (D74).
    policy = document()
    agents = policy["agents"]
    assert isinstance(agents, list)
    policy["agents"] = [a for a in agents if a["id"] == "tools-only"]
    with (
        Paveo.from_policy(policy, audit_path=log, now=make_clock()) as pf,
        pf.session(agent_id="tools-only") as s,
    ):
        with pytest.raises(PolicyDenied, match="no_budget"):
            s.check_llm(ask(), shape="anthropic")
        with pytest.raises(ConfigError, match="no budget"):
            s.remaining()


def test_a_call_that_cannot_be_priced_is_refused_and_recorded(
    pf: Paveo, log: Path
) -> None:
    image = {"type": "image", "source": {"type": "url", "url": "https://x/a.png"}}
    with pf.session(agent_id="bot") as s:
        with pytest.raises(PricingUnknown):
            s.check_llm(
                ask(messages=[{"role": "user", "content": [image]}]),
                shape="anthropic",
            )
        assert s.remaining() == Decimal("1.00")

    (denial,) = records_in(log)
    assert (denial["decision"], denial["reason"]) == ("deny", "pricing_unknown")


def test_nothing_in_a_request_reaches_the_audit_log(pf: Paveo, log: Path) -> None:
    """Locked decision #5, on the path that reads the most payload."""
    request = ask(
        system=f"Customer is {SENTINEL}.",
        messages=[{"role": "user", "content": f"Refund {SENTINEL} today."}],
    )
    with pf.session(agent_id="bot") as s:
        s.check_llm(request, shape="anthropic").record(usage(output=5))

    assert SENTINEL not in log.read_text()


def test_a_model_priced_in_the_policy_is_admitted_through_the_door(
    pf: Paveo, log: Path
) -> None:
    """D39: any provider, through the door, in a shape paveo has no adapter for."""
    request = {
        "model": "mistral-large-3",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "Summarise."}],
    }
    with pf.session(agent_id="bot") as s:
        charged = s.check_llm(request, shape="generic").record(
            {"input": 1_000_000, "output": 1000}
        )

    assert charged == Decimal("2.00") + 1000 * Decimal("6") / MILLION
    assert records_in(log)[0]["rate_key"] == "mistral-large-3|declared-by-policy"


def test_a_shape_paveo_has_no_adapter_for_is_a_wiring_error(pf: Paveo) -> None:
    with pf.session(agent_id="bot") as s, pytest.raises(ConfigError, match="shape"):
        s.check_llm(ask(), shape="cobol")


def test_an_assumed_output_bound_is_written_into_the_record(log: Path) -> None:
    """§4.4: the ceiling is approximate, and the record says so."""
    policy = document(defaults={"assumed_max_output_tokens": 1000})
    request = ask()
    del request["max_tokens"]
    with (
        Paveo.from_policy(policy, audit_path=log, now=make_clock()) as pf,
        pf.session(agent_id="bot") as s,
    ):
        s.check_llm(request, shape="anthropic").release()

    assert records_in(log)[0]["reason"] == "assumed_max_output_tokens"


def test_a_response_with_a_class_nobody_priced_stops_the_next_call(
    pf: Paveo, log: Path
) -> None:
    """§4.8.3 through the public surface: charged at the dearest rate, marked
    stale, and every later reserve refused until the table is updated."""
    with pf.session(agent_id="bot") as s:
        call = s.check_llm(ask(max_tokens=10), shape="anthropic")
        call.record({"input_tokens": 1, "output_tokens": 1, "audio_tokens": 5})
        with pytest.raises(PricingUnknown, match="audio_tokens"):
            s.check_llm(ask(max_tokens=10), shape="anthropic")

    records = records_in(log)
    assert records[1]["price_table_stale"] is True
    assert records[2]["reason"] == "pricing_unknown"


def test_a_decision_that_cannot_be_recorded_gives_its_hold_back(pf: Paveo) -> None:
    """§7: an unlogged decision did not happen, so the call is refused and the
    ceiling is exactly as it was."""
    with pf.session(agent_id="bot") as s:
        pf._audit.close()
        with pytest.raises(PolicyUnavailable):
            s.check_llm(ask(), shape="anthropic")
        assert s.remaining() == Decimal("1.00")


def test_a_session_that_was_never_entered_checks_nothing(pf: Paveo) -> None:
    with pytest.raises(ConfigError, match="never entered"):
        pf.session(agent_id="bot").check_llm(ask(), shape="anthropic")


def test_many_threads_spending_one_ceiling_cannot_overspend_it(
    log: Path,
) -> None:
    """The concurrent path, through the public surface: sixteen threads, each with
    its own session, all reserving and recording against one $1.00 ceiling. The
    ceiling holds and nothing is left reserved."""
    clock: Callable[[], datetime] = make_clock()
    start = threading.Barrier(16)
    admitted: list[Decimal] = []
    refused: list[BudgetExceeded] = []
    lock = threading.Lock()

    with Paveo.from_policy(document(), audit_path=log, now=clock) as pf:

        def agent() -> None:
            start.wait()
            with pf.session(agent_id="bot") as s:
                for _ in range(3):
                    try:
                        call = s.check_llm(ask(), shape="anthropic")
                    except BudgetExceeded as e:
                        with lock:
                            refused.append(e)
                        continue
                    charged = call.record(usage(output=20_000))
                    with lock:
                        admitted.append(charged)

        threads = [threading.Thread(target=agent) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        with pf.session(agent_id="bot") as s:
            left = s.remaining()

    spent = sum(admitted, Decimal(0))
    assert spent <= Decimal("1.00")
    assert left == Decimal("1.00") - spent
    assert refused
    assert verify_chain(log).ok


# What /code-review found in S4 (D40), each pinned so it cannot come back.


def test_a_declared_price_does_not_switch_off_the_anthropic_reader(log: Path) -> None:
    """The shape decides how a request is read; the policy only what it costs.
    Otherwise declaring a price for an older Claude model would skip the tool
    allowances and let a server tool through unbounded."""
    policy = document(
        prices={
            "claude-sonnet-4-5": {"input_per_mtok": "3.00", "output_per_mtok": "15.00"}
        }
    )
    agents = policy["agents"]
    assert isinstance(agents, list)
    agents[0]["models"]["allow"].append("claude-sonnet-4-5")
    request = ask(
        model="claude-sonnet-4-5", tools=[{"type": "web_search_20250305", "name": "w"}]
    )
    with (
        Paveo.from_policy(policy, audit_path=log, now=make_clock()) as pf,
        pf.session(agent_id="bot") as s,
        pytest.raises(PricingUnknown, match=r"tools\[0\]"),
    ):
        s.check_llm(request, shape="anthropic")


def test_zero_of_a_class_the_policy_did_not_price_costs_nothing(pf: Paveo) -> None:
    """A declared model with no cached price, reporting `cache_read: 0`, must not
    stop every model in the process. A non-zero count still does."""
    request = {"model": "mistral-large-3", "max_tokens": 10, "messages": []}
    with pf.session(agent_id="bot") as s:
        s.check_llm(request, shape="generic").record(
            {"input": 5, "output": 5, "cache_read": 0}
        )
        s.check_llm(ask(max_tokens=10), shape="anthropic").release()


def test_any_failure_to_record_a_yes_gives_the_hold_back(pf: Paveo) -> None:
    """Not only PolicyUnavailable: whatever stops the record, nobody holds a
    handle to that reservation, so nothing else would ever give it back."""

    def broken(_record: object) -> str:
        raise RuntimeError("the disk went away in an unexpected way")

    with pf.session(agent_id="bot") as s:
        pf._audit.append = broken  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            s.check_llm(ask(), shape="anthropic")
        assert s.remaining() == Decimal("1.00")


@pytest.mark.parametrize(
    ("request_", "shape"),
    [
        pytest.param({"max_tokens": 1, "messages": []}, "anthropic", id="no-model"),
        pytest.param(ask(), "cobol", id="unknown-shape"),
        pytest.param(ask(metadata={"user_id": object()}), "anthropic", id="not-json"),
    ],
)
def test_a_request_wired_wrong_is_still_a_recorded_refusal(
    pf: Paveo, log: Path, request_: dict[str, object], shape: str
) -> None:
    with pf.session(agent_id="bot") as s, pytest.raises(ConfigError):
        s.check_llm(request_, shape=shape)

    (refusal,) = records_in(log)
    assert (refusal["decision"], refusal["reason"]) == ("deny", "invalid_request")


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(ConfigError("the clock went missing.", remedy="x"), id="ours"),
        pytest.param(ConnectionError("the ledger is down"), id="a-custom-store"),
    ],
)
def test_a_store_that_cannot_answer_is_still_a_recorded_refusal(
    pf: Paveo, log: Path, failure: Exception
) -> None:
    """Not only BudgetExceeded, and not only our own errors: a customer's store
    can raise anything, and an unlogged refusal did not happen (D77)."""

    def broken(**_kwargs: object) -> None:
        raise failure

    with pf.session(agent_id="bot") as s:
        s._gate.store.reserve = broken  # type: ignore[method-assign]
        with pytest.raises(type(failure)):
            s.check_llm(ask(), shape="anthropic")

    (refusal,) = records_in(log)
    assert (refusal["decision"], refusal["reason"]) == ("deny", "reserve_failed")


def test_a_ledger_that_refuses_a_settle_leaves_the_call_open(pf: Paveo) -> None:
    """The call is closed only once the ledger has taken the charge, so a settle
    that failed there is charged again, in full, when the session ends."""
    with pf.session(agent_id="bot") as s:
        call = s.check_llm(ask(), shape="anthropic")
        store = s._gate.store
        real = store.settle
        calls = {"n": 0}

        def flaky(**kwargs: object) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConfigError("the clock stopped.", remedy="restart it.")
            real(**kwargs)  # type: ignore[arg-type]

        store.settle = flaky  # type: ignore[method-assign]
        with pytest.raises(ConfigError, match="clock"):
            call.record(usage(output=10))
    assert calls["n"] == 2
