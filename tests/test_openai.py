"""OpenAI: the price rows and the Chat Completions adapter (S4b, D41).

The rates are pinned to the figures OpenAI's pricing page and model pages print,
read on 2026-09-24, not derived from our own multipliers, so a wrong multiplier
disagrees with this file instead of being copied into it.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from conftest import SENTINEL, make_clock, records_in
from paveo import BudgetExceeded, ConfigError, Paveo, PricingUnknown, verify_chain
from paveo._openai import _TOOLS_ALLOWANCE, bound, usage_classes
from paveo.prices import _ALL_LISTINGS, _OPENAI_LISTINGS, _PriceTable

MILLION = Decimal(1_000_000)

# Standard and Fast rows as the page prints them: input, cached input, output.
STANDARD = {
    "gpt-6-astra": ("10", "1", "50"),
    "gpt-6-sol": ("2", "0.2", "10"),
    "gpt-6-luna": ("0.10", "0.01", "0.50"),
    "gpt-5.6-sol": ("4", "0.4", "20"),
    "gpt-5.6-terra": ("2", "0.2", "12"),
    "gpt-5.6-luna": ("0.20", "0.02", "1.20"),
    "gpt-5.5": ("5", "0.5", "30"),
    "gpt-5.4": ("2.50", "0.25", "15"),
    "gpt-5.4-mini": ("0.75", "0.075", "4.50"),
    "gpt-5.4-nano": ("0.20", "0.02", "1.25"),
    "gpt-5.2": ("1.75", "0.175", "14"),
    "gpt-5.1": ("1.25", "0.125", "10"),
    "gpt-5": ("1.25", "0.125", "10"),
    "gpt-5-mini": ("0.25", "0.025", "2"),
    "gpt-5-nano": ("0.05", "0.005", "0.40"),
    "gpt-4.1": ("2", "0.5", "8"),
    "gpt-4.1-mini": ("0.40", "0.10", "1.60"),
    "gpt-4.1-nano": ("0.10", "0.025", "0.40"),
    "gpt-4o": ("2.50", "1.25", "10"),
    "gpt-4o-mini": ("0.15", "0.075", "0.60"),
    "o3": ("2", "0.5", "8"),
    "o4-mini": ("1.10", "0.275", "4.40"),
}
FAST = {
    "gpt-6-astra": ("20", "2", "100"),
    "gpt-6-sol": ("4", "0.4", "20"),
    "gpt-5.6-sol": ("8", "0.8", "40"),
    "gpt-5.5": ("12.50", "1.25", "75"),
    "gpt-4o": ("4.25", "2.125", "17"),
    "o4-mini": ("2", "0.5", "8"),
}
# The data-residency list, released on or after 2026-03-05.
REGIONAL = {
    "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-5.6-luna", "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
}  # fmt: skip
CACHE_WRITE_PREMIUM = {m for m in STANDARD if m.startswith(("gpt-6", "gpt-5.6"))}


def table() -> _PriceTable:
    return _PriceTable(_ALL_LISTINGS)


def per_million(rate: Decimal) -> Decimal:
    return rate * MILLION


def test_the_table_carries_exactly_the_models_the_page_was_read_for() -> None:
    assert set(_OPENAI_LISTINGS) == set(STANDARD)


@pytest.mark.parametrize("model", sorted(STANDARD))
def test_every_standard_rate_is_the_page_s_with_its_regional_uplift(model: str) -> None:
    """The uplift is chosen by endpoint or project region, which no request shows,
    so the ten eligible models are always charged it (D41)."""
    rates = table().resolve(model, {"service_tier": "default"})
    uplift = Decimal("1.1") if model in REGIONAL else Decimal(1)
    base, cached, output = map(Decimal, STANDARD[model])

    assert per_million(rates.per_token["input"]) == base * uplift
    assert per_million(rates.per_token["cache_read"]) == cached * uplift
    assert per_million(rates.per_token["output"]) == output * uplift
    premium = Decimal("1.25") if model in CACHE_WRITE_PREMIUM else Decimal(1)
    assert per_million(rates.per_token["cache_write"]) == base * uplift * premium


@pytest.mark.parametrize("model", sorted(FAST))
def test_fast_and_priority_select_the_page_s_fast_row(model: str) -> None:
    t = table()
    uplift = Decimal("1.1") if model in REGIONAL else Decimal(1)
    for tier in ("fast", "priority"):
        rates = t.resolve(model, {"service_tier": tier})
        assert (
            per_million(rates.per_token["output"]) == Decimal(FAST[model][2]) * uplift
        )


@pytest.mark.parametrize("tier", [None, "auto"])
def test_an_unset_or_auto_tier_is_priced_at_the_dearest_the_model_has(
    tier: str | None,
) -> None:
    """A project set to Fast makes every unset request Fast (the OpenAI guide),
    and we cannot see the project: D38's trap, priced the same way."""
    t = table()
    modifiers = {} if tier is None else {"service_tier": tier}

    unset = t.resolve("gpt-5.6-sol", modifiers)
    fast = t.resolve("gpt-5.6-sol", {"service_tier": "fast"})

    assert unset.per_token == fast.per_token
    assert t.resolve("gpt-5-nano", modifiers).per_token == (
        t.resolve("gpt-5-nano", {"service_tier": "default"}).per_token
    )


def test_flex_is_priced_at_standard_the_safe_side() -> None:
    t = table()
    assert t.resolve("o3", {"service_tier": "flex"}).per_token == (
        t.resolve("o3", {"service_tier": "default"}).per_token
    )


@pytest.mark.parametrize(
    ("model", "tier"), [("gpt-5", "scale"), ("gpt-5-nano", "fast"), ("gpt-5", 3)]
)
def test_a_tier_the_table_does_not_price_is_refused(model: str, tier: object) -> None:
    with pytest.raises(PricingUnknown, match="service_tier"):
        table().resolve(model, {"service_tier": tier})


def test_a_prompt_over_272k_bills_the_whole_request_at_the_long_rows() -> None:
    """GPT-6 Sol's page: "Beyond 272K tokens: input/cache priced at 2x, output at
    1.5x", for the entire request. Chosen from the response's own prompt size."""
    t = table()
    rates = t.resolve("gpt-6-sol", {"service_tier": "default"})
    short = {"input": 272_000, "output": 1_000_000}
    long = {"input": 272_001, "output": 1_000_000}

    short_cost = t.actual(rates, short).cost
    long_cost = t.actual(rates, long).cost

    uplift = Decimal("1.1")
    assert (
        short_cost
        == (272_000 * Decimal(2) + 1_000_000 * Decimal(10)) * uplift / MILLION
    )
    assert (
        long_cost == (272_001 * Decimal(4) + 1_000_000 * Decimal(15)) * uplift / MILLION
    )


def test_a_bound_near_272k_is_reserved_at_the_long_rows() -> None:
    """Reserved long from 13,600 tokens below the line: the bound carries a chosen
    tools allowance, and a small miss would land a short reservation on a long
    bill (/code-review, D41). Settling uses the real prompt size."""
    t = table()
    rates = t.resolve("gpt-5.4", {"service_tier": "default"})
    edge = 272_000 - 13_600

    short = t.worst_case(rates, edge, 0)
    long = t.worst_case(rates, edge + 1, 0)

    assert short == edge * Decimal("2.50") * Decimal("1.1") / MILLION
    assert long == (edge + 1) * Decimal("5.00") * Decimal("1.1") / MILLION


def ask(**extra: object) -> dict[str, object]:
    request: dict[str, object] = {
        "model": "gpt-5.6-sol",
        "service_tier": "default",
        "max_completion_tokens": 1000,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Summarise the ticket."},
        ],
    }
    request.update(extra)
    return request


def size(request: dict[str, object]) -> int:
    body = {k: v for k, v in request.items() if k != "timeout"}
    return len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())


def test_a_text_request_is_bounded_by_its_bytes() -> None:
    measured = bound(ask(), assumed_max_output_tokens=None)

    assert measured.input_upper_bound == size(ask())
    assert measured.max_output_tokens == 1000
    assert measured.modifiers == {"service_tier": "default"}


def test_function_tools_add_the_rendering_allowance() -> None:
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    request = ask(tools=tools)

    assert bound(request, assumed_max_output_tokens=None).input_upper_bound == (
        size(request) + _TOOLS_ALLOWANCE
    )


def test_n_answers_multiply_the_output_bound_and_the_larger_cap_wins() -> None:
    measured = bound(ask(n=4, max_tokens=1500), assumed_max_output_tokens=None)

    assert measured.max_output_tokens == 4 * 1500


@pytest.mark.parametrize(
    "extra",
    [
        {"audio": {"voice": "alloy"}},
        {"modalities": ["text", "audio"]},
        {"prediction": {"type": "content", "content": "x"}},
        {"web_search_options": {}},
        {"functions": []},
        {"extra_body": {}},
        {"tools": [{"type": "custom", "name": "c"}]},
        {"tools": [{"type": "web_search"}]},
        {"n": 0},
        {"max_completion_tokens": -1},
    ],
)
def test_what_cannot_be_bounded_is_refused(extra: dict[str, object]) -> None:
    with pytest.raises(PricingUnknown):
        bound(ask(**extra), assumed_max_output_tokens=None)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
        {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}},
        {"type": "file", "file": {"file_id": "f"}},
        {"type": ["text"]},
    ],
)
def test_content_that_is_not_text_is_refused_and_not_echoed(part: object) -> None:
    request = ask(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": SENTINEL}, part]}
        ]
    )

    with pytest.raises(PricingUnknown, match=r"messages\[0\]\.content\[1\]") as caught:
        bound(request, assumed_max_output_tokens=None)
    assert SENTINEL not in str(caught.value)


def test_no_output_cap_is_refused_unless_the_policy_assumes_one() -> None:
    request = ask()
    del request["max_completion_tokens"]

    with pytest.raises(PricingUnknown, match="max_completion_tokens"):
        bound(request, assumed_max_output_tokens=None)
    assert bound(request, assumed_max_output_tokens=500).assumed_output


def test_usage_splits_the_prompt_into_the_rates_it_was_billed_at() -> None:
    """Cached tokens and cache writes are inside prompt_tokens, each billed at one
    rate; reasoning is inside completion_tokens."""
    usage = {
        "prompt_tokens": 10_000,
        "completion_tokens": 900,
        "total_tokens": 10_900,
        "prompt_tokens_details": {
            "cached_tokens": 6_000,
            "cache_write_tokens": 3_000,
            "audio_tokens": 0,
        },
        "completion_tokens_details": {
            "reasoning_tokens": 700,
            "rejected_prediction_tokens": 0,
        },
    }

    assert usage_classes(usage) == {
        "input": 1_000,
        "cache_read": 6_000,
        "cache_write": 3_000,
        "output": 900,
    }


def test_image_or_audio_tokens_in_a_response_reach_the_table_as_unknown() -> None:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {"image_tokens": 50},
        "completion_tokens_details": {"audio_tokens": 5},
    }

    classes = usage_classes(usage)
    assert classes["prompt_image_tokens"] == 50
    assert classes["completion_audio_tokens"] == 5


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 10},
        {"prompt_tokens": "10", "completion_tokens": 1},
        {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 11},
        },
        "10 tokens",
    ],
)
def test_usage_that_does_not_add_up_is_a_wiring_error(usage: object) -> None:
    with pytest.raises(ConfigError):
        usage_classes(usage)  # type: ignore[arg-type]


def test_a_one_dollar_ceiling_refuses_the_openai_call_that_would_breach_it(
    tmp_path: Path,
) -> None:
    """S4b's done-when, through the public surface. Each call may cost up to
    20,000 output tokens at GPT-5.6 Sol's $20 x 1.1 = $0.44; two fit under $1.00
    and the third does not."""
    log = tmp_path / "audit.jsonl"
    policy = {
        "version": 1,
        "policy_id": "s4b",
        "agents": [
            {
                "id": "bot",
                "budget": {"period": "day", "limit_usd": "1.00"},
                "models": {"allow": ["gpt-5.6-sol"]},
            }
        ],
    }
    usage = {"prompt_tokens": 50, "completion_tokens": 20_000}
    with (
        Paveo.from_policy(policy, audit_path=log, now=make_clock()) as pf,
        pf.session(agent_id="bot") as s,
    ):
        for _ in range(2):
            s.check_llm(ask(max_completion_tokens=20_000), shape="openai").record(usage)
        with pytest.raises(BudgetExceeded):
            s.check_llm(ask(max_completion_tokens=20_000), shape="openai")

    decisions = [(r["decision"], r["reason"]) for r in records_in(log)]
    assert decisions == [("allow", None), ("settle", "recorded")] * 2 + [
        ("deny", "budget_exceeded")
    ]
    assert (
        records_in(log)[0]["rate_key"]
        == "gpt-5.6-sol|service_tier=default|regional_uplift"
    )
    assert verify_chain(log).ok


# What both reviews found in S4b (D41), each pinned.


@pytest.mark.parametrize("content", [None, "hi"])
def test_audio_beside_text_or_empty_content_is_refused(content: object) -> None:
    """The audio check sat after an early return for string or empty content."""
    request = ask(
        messages=[
            {"role": "assistant", "content": content, "audio": {"id": "audio_1"}},
            {"role": "user", "content": "go on"},
        ]
    )

    with pytest.raises(PricingUnknown, match=r"messages\[0\]"):
        bound(request, assumed_max_output_tokens=None)


def test_a_tool_call_that_is_not_a_function_is_refused() -> None:
    request = ask(
        messages=[{"role": "assistant", "tool_calls": [{"type": "custom", "id": "c"}]}]
    )

    with pytest.raises(PricingUnknown, match="tool_calls"):
        bound(request, assumed_max_output_tokens=None)


def test_a_top_level_usage_field_nobody_reads_reaches_the_table() -> None:
    """LiteLLM reports Claude's cache writes at the top level; dropping them
    charged them at plain input without a word."""
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "cache_creation_input_tokens": 90,
    }

    assert usage_classes(usage)["cache_creation_input_tokens"] == 90


def test_the_whole_response_can_be_passed_and_its_tier_settles_the_charge() -> None:
    """Reserved at Fast because the tier was unset; the response says default,
    so it is charged at default rather than twice the bill for ever."""
    t = table()
    reserved = t.resolve("gpt-5.6-sol", {})
    response = {
        "service_tier": "default",
        "usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000},
    }
    from paveo._openai import served  # noqa: PLC0415 - the reader under test

    charge = t.actual(reserved, usage_classes(response), served(response))

    assert charge.rate_key == "gpt-5.6-sol|service_tier=default|regional_uplift"
    assert (
        charge.cost
        == (1_000 * Decimal(4) + 1_000 * Decimal(20)) * Decimal("1.1") / MILLION
    )


def test_a_long_prompt_is_recorded_under_the_long_row() -> None:
    t = table()
    rates = t.resolve("gpt-6-sol", {"service_tier": "default"})

    charge = t.actual(rates, {"input": 300_000, "output": 10})

    assert charge.rate_key.endswith("|long_context")


def test_an_openai_cache_write_for_a_claude_row_is_charged_at_the_dearer_write() -> (
    None
):
    """A Claude model through LiteLLM reports OpenAI's single `cache_write`;
    unaliased, it stopped every model in the process."""
    t = table()
    rates = t.resolve("claude-sonnet-5", {"inference_geo": "global"})

    charge = t.actual(rates, {"input": 1, "cache_write": 1_000_000, "output": 0})

    assert not charge.price_table_stale
    # One input token at $2/M, a million writes at the 1-hour $4/M.
    assert charge.cost == Decimal("2") / MILLION + Decimal("4")


def test_no_two_providers_share_a_model_id() -> None:
    """The merged table is one dict; a shared id would silently replace a row."""
    from paveo.prices import _LISTINGS  # noqa: PLC0415

    assert not set(_LISTINGS) & set(_OPENAI_LISTINGS)


def test_every_openai_listing_says_what_an_unset_tier_means() -> None:
    from paveo.prices import _ABSENT, _modifier_values  # noqa: PLC0415

    for listing in _OPENAI_LISTINGS.values():
        assert set(_modifier_values(listing)) == {"service_tier"} <= set(_ABSENT)


def test_the_settle_record_names_the_row_actually_charged(tmp_path: Path) -> None:
    """Reserved at Fast (tier unset), served and charged at default: the audit
    line must say default, or an auditor rebuilding the cost gets twice the bill."""
    log = tmp_path / "audit.jsonl"
    policy = {
        "version": 1,
        "policy_id": "s4b-served",
        "agents": [
            {
                "id": "bot",
                "budget": {"period": "day", "limit_usd": "5.00"},
                "models": {"allow": ["gpt-5.6-sol"]},
            }
        ],
    }
    request = ask()
    del request["service_tier"]
    response = {
        "service_tier": "default",
        "usage": {"prompt_tokens": 50, "completion_tokens": 50},
    }
    with (
        Paveo.from_policy(policy, audit_path=log, now=make_clock()) as pf,
        pf.session(agent_id="bot") as s,
    ):
        s.check_llm(request, shape="openai").record(response)

    allow, settle = records_in(log)
    assert allow["rate_key"] == "gpt-5.6-sol|service_tier=unset|regional_uplift"
    assert settle["rate_key"] == "gpt-5.6-sol|service_tier=default|regional_uplift"
