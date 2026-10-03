"""Gemini: the price rows and the generate_content adapter (S4c, D42).

Rates are pinned to the figures Google's pricing page prints, read 2026-09-24.
Three Flash models change price on 2027-01-01, so the table is given a clock and
tested on both sides of midnight rather than waiting for it (Rule 14).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from conftest import SENTINEL, make_clock, records_in
from paveo import BudgetExceeded, ConfigError, Paveo, PricingUnknown, verify_chain
from paveo._gemini import _TOOLS_ALLOWANCE, bound, usage_classes
from paveo.prices import (
    _ALL_LISTINGS,
    _GEMINI_LISTINGS,
    _LISTINGS,
    _OPENAI_LISTINGS,
    _PriceTable,
)

MILLION = Decimal(1_000_000)

# (input, cached input or None, output), and Priority (input, output).
STANDARD = {
    "gemini-3.8-flash": ("0.75", "0.075", "3.75"),
    "gemini-3.7-flash": ("0.75", "0.075", "3.75"),
    "gemini-3.6-flash": ("0.75", "0.075", "3.75"),
    "gemini-3.5-flash": ("1.50", "0.15", "9.00"),
    "gemini-3.5-flash-lite": ("0.30", None, "2.50"),
    "gemini-3.1-flash-lite": ("0.25", "0.025", "1.50"),
    "gemini-3.1-pro-preview": ("2.00", "0.20", "12.00"),
    "gemini-3-flash-preview": ("0.50", "0.05", "3.00"),
    "gemini-2.5-pro": ("1.25", "0.125", "10.00"),
    "gemini-2.5-flash": ("0.30", "0.03", "2.50"),
    "gemini-2.5-flash-lite": ("0.10", "0.01", "0.40"),
}
PRIORITY = {
    "gemini-3.8-flash": ("1.35", "6.75"),
    "gemini-3.5-flash": ("2.70", "16.20"),
    "gemini-3.1-pro-preview": ("3.60", "21.60"),
    "gemini-2.5-pro": ("2.25", "18.00"),
    "gemini-2.5-flash-lite": ("0.18", "0.72"),
}
BEFORE_2027 = datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)
FROM_2027 = datetime(2027, 1, 1, 0, 0, tzinfo=UTC)


def table(at: datetime = BEFORE_2027) -> _PriceTable:
    return _PriceTable(_ALL_LISTINGS, now=lambda: at)


def per_million(rate: Decimal) -> Decimal:
    return rate * MILLION


def test_the_table_carries_exactly_the_models_the_page_was_read_for() -> None:
    assert set(_GEMINI_LISTINGS) == set(STANDARD)


def test_no_two_providers_share_a_model_id() -> None:
    ids = [set(_LISTINGS), set(_OPENAI_LISTINGS), set(_GEMINI_LISTINGS)]
    assert sum(len(i) for i in ids) == len(set().union(*ids))


@pytest.mark.parametrize("model", sorted(STANDARD))
def test_every_standard_rate_is_the_one_the_page_prints(model: str) -> None:
    rates = table().resolve(model, {})
    base, cached, output = STANDARD[model]

    assert per_million(rates.per_token["input"]) == Decimal(base)
    assert per_million(rates.per_token["output"]) == Decimal(output)
    if cached is None:
        # "Context caching: Not available": a cached count would stop the table.
        assert "cache_read" not in rates.per_token
    else:
        assert per_million(rates.per_token["cache_read"]) == Decimal(cached)


@pytest.mark.parametrize("model", sorted(PRIORITY))
def test_priority_selects_the_page_s_priority_row(model: str) -> None:
    rates = table().resolve(model, {"service_tier": "priority"})
    pri_in, pri_out = map(Decimal, PRIORITY[model])

    assert per_million(rates.per_token["input"]) == pri_in
    assert per_million(rates.per_token["output"]) == pri_out
    # Priority's cached rate is not published; charged at Priority input.
    assert per_million(rates.per_token["cache_read"]) == pri_in


@pytest.mark.parametrize("tier", [None, "standard", "STANDARD", "unspecified", "flex"])
def test_unset_standard_and_flex_all_price_at_standard(tier: str | None) -> None:
    """Unset is standard (the SDK enum); flex at standard is the safe side."""
    t = table()
    modifiers = {} if tier is None else {"service_tier": tier.lower()}

    assert t.resolve("gemini-2.5-flash", modifiers).per_token == (
        t.resolve("gemini-2.5-flash", {}).per_token
    )


def test_a_tier_the_table_does_not_price_is_refused() -> None:
    with pytest.raises(PricingUnknown, match="service_tier"):
        table().resolve("gemini-2.5-flash", {"service_tier": "provisioned"})


def test_three_flash_models_double_at_midnight_on_new_year_2027() -> None:
    """The page: "$0.75 through December 31, 2026. $1.50 starting January 1,
    2027". Priced by the day the call is admitted, from an injected clock."""
    before = table(BEFORE_2027).resolve("gemini-3.8-flash", {})
    after = table(FROM_2027).resolve("gemini-3.8-flash", {})

    assert per_million(before.per_token["input"]) == Decimal("0.75")
    assert per_million(after.per_token["input"]) == Decimal("1.50")
    assert per_million(after.per_token["output"]) == Decimal("7.50")
    assert after.rate_key == "gemini-3.8-flash|service_tier=unset|from=2027-01-01"
    priority = table(FROM_2027).resolve(
        "gemini-3.8-flash", {"service_tier": "priority"}
    )
    assert per_million(priority.per_token["output"]) == Decimal("13.50")


def test_a_prompt_over_200k_bills_the_whole_request_at_the_long_rows() -> None:
    """Every "> 200k" row on the page is 2x input and 1.5x output."""
    t = table()
    rates = t.resolve("gemini-2.5-pro", {})

    short = t.actual(rates, {"input": 200_000, "output": 1_000})
    long = t.actual(rates, {"input": 200_001, "output": 1_000})

    assert short.cost == (200_000 * Decimal("1.25") + 1_000 * Decimal(10)) / MILLION
    assert long.cost == (200_001 * Decimal("2.50") + 1_000 * Decimal(15)) / MILLION
    assert long.rate_key.endswith("|long_context")


def test_a_bound_near_200k_is_reserved_long() -> None:
    t = table()
    rates = t.resolve("gemini-3.1-pro-preview", {})
    edge = 200_000 - 10_000

    assert t.worst_case(rates, edge, 0) == edge * Decimal(2) / MILLION
    assert t.worst_case(rates, edge + 1, 0) == (edge + 1) * Decimal(4) / MILLION


def ask(**config: object) -> dict[str, object]:
    base: dict[str, object] = {"max_output_tokens": 1000}
    base.update(config)
    return {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "user", "parts": [{"text": "Summarise the ticket."}]}],
        "config": base,
    }


def size(request: dict[str, object]) -> int:
    return len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode())


def test_a_text_request_is_bounded_by_its_bytes() -> None:
    measured = bound(
        ask(system_instruction="Be brief."), assumed_max_output_tokens=None
    )

    assert measured.input_upper_bound == size(ask(system_instruction="Be brief."))
    assert measured.max_output_tokens == 1000
    assert measured.modifiers == {}


@pytest.mark.parametrize(
    "contents",
    [
        "a plain string",
        ["a", "list", "of", "strings"],
        [
            {
                "role": "model",
                "parts": [{"function_call": {"name": "f", "args": {"image": "x"}}}],
            }
        ],
        [{"parts": [{"thought": True, "text": "..."}, {"thoughtSignature": "sig"}]}],
    ],
)
def test_every_text_shape_contents_can_take_is_admitted(contents: object) -> None:
    request = ask()
    request["contents"] = contents

    assert bound(request, assumed_max_output_tokens=None).input_upper_bound > 0


def test_function_declarations_add_the_rendering_allowance() -> None:
    request = ask(tools=[{"function_declarations": [{"name": "lookup"}]}])

    assert bound(request, assumed_max_output_tokens=None).input_upper_bound == (
        size(request) + _TOOLS_ALLOWANCE
    )


def test_candidates_multiply_the_output_bound() -> None:
    assert (
        bound(ask(candidate_count=3), assumed_max_output_tokens=None).max_output_tokens
        == 3000
    )


@pytest.mark.parametrize(
    "config",
    [
        {"tools": [{"google_search": {}}]},
        {"tools": [{"code_execution": {}}]},
        {"tools": [{"url_context": {}}]},
        {"tools": [{"googleMaps": {}}]},
        {"cached_content": "cachedContents/1"},
        {"response_modalities": ["AUDIO"]},
        {"media_resolution": "MEDIA_RESOLUTION_HIGH"},
        {"candidate_count": 0},
        {"max_output_tokens": -1},
    ],
)
def test_what_cannot_be_bounded_is_refused(config: dict[str, object]) -> None:
    with pytest.raises(PricingUnknown):
        bound(ask(**config), assumed_max_output_tokens=None)


@pytest.mark.parametrize(
    "part",
    [
        {"inline_data": {"mime_type": "image/png", "data": "..."}},
        {"inlineData": {"mimeType": "audio/wav", "data": "..."}},
        {"file_data": {"file_uri": "gs://x"}},
        {"executable_code": {"code": "print(1)"}},
    ],
)
def test_a_part_that_is_not_text_is_refused_and_not_echoed(part: object) -> None:
    request = ask()
    request["contents"] = [{"role": "user", "parts": [{"text": SENTINEL}, part]}]

    with pytest.raises(PricingUnknown, match=r"contents\[0\]\.parts\[1\]") as caught:
        bound(request, assumed_max_output_tokens=None)
    assert SENTINEL not in str(caught.value)


def test_no_output_cap_is_refused_unless_the_policy_assumes_one() -> None:
    request = ask()
    request["config"] = {}

    with pytest.raises(PricingUnknown, match="max_output_tokens"):
        bound(request, assumed_max_output_tokens=None)
    assert bound(request, assumed_max_output_tokens=64).assumed_output


def test_usage_takes_cached_out_of_input_and_adds_thoughts_to_output() -> None:
    usage = {
        "prompt_token_count": 10_000,
        "cached_content_token_count": 6_000,
        "candidates_token_count": 300,
        "thoughts_token_count": 700,
        "total_token_count": 11_000,
        "prompt_tokens_details": [{"modality": "TEXT", "token_count": 10_000}],
        "traffic_type": "ON_DEMAND",
    }

    assert usage_classes(usage) == {
        "input": 4_000,
        "cache_read": 6_000,
        "output": 1_000,
    }


def test_the_rest_spelling_and_the_whole_response_both_read() -> None:
    response = {
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "thoughtsTokenCount": 3,
        }
    }

    assert usage_classes(response) == {"input": 10, "cache_read": 0, "output": 5}


def test_tool_results_or_non_text_modalities_reach_the_table_as_unknown() -> None:
    usage = {
        "prompt_token_count": 100,
        "candidates_token_count": 1,
        "tool_use_prompt_token_count": 40,
        "prompt_tokens_details": [{"modality": "AUDIO", "token_count": 60}],
        "a_new_billable_field": 9,
    }

    classes = usage_classes(usage)
    assert classes["tool_use_prompt_token_count"] == 40
    assert classes["prompt_audio"] == 60
    assert classes["a_new_billable_field"] == 9


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_token_count": "10"},
        {"prompt_token_count": 5, "cached_content_token_count": 6},
        "ten tokens",
    ],
)
def test_usage_that_does_not_add_up_is_a_wiring_error(usage: object) -> None:
    with pytest.raises(ConfigError):
        usage_classes(usage)  # type: ignore[arg-type]


def policy_for(model: str, limit: str = "1.00") -> dict[str, object]:
    return {
        "version": 1,
        "policy_id": "s4c",
        "agents": [
            {
                "id": "bot",
                "budget": {"period": "day", "limit_usd": limit},
                "models": {"allow": [model]},
            }
        ],
    }


def test_a_one_dollar_ceiling_refuses_the_gemini_call_that_would_breach_it(
    tmp_path: Path,
) -> None:
    """S4c's done-when. 2.5 Pro, 40,000 output tokens a call at $10/M: $0.40 each,
    two fit under $1.00 and the third does not."""
    log = tmp_path / "audit.jsonl"
    request = ask(max_output_tokens=40_000)
    request["model"] = "gemini-2.5-pro"
    usage = {
        "prompt_token_count": 50,
        "candidates_token_count": 30_000,
        "thoughts_token_count": 10_000,
    }
    with (
        Paveo.from_policy(
            policy_for("gemini-2.5-pro"), audit_path=log, now=make_clock()
        ) as pf,
        pf.session(agent_id="bot") as s,
    ):
        for _ in range(2):
            s.check_llm(request, shape="gemini").record(usage)
        with pytest.raises(BudgetExceeded):
            s.check_llm(request, shape="gemini")

    decisions = [(r["decision"], r["reason"]) for r in records_in(log)]
    assert decisions == [("allow", None), ("settle", "recorded")] * 2 + [
        ("deny", "budget_exceeded")
    ]
    assert verify_chain(log).ok


def test_a_call_that_costs_more_than_its_reservation_stops_the_table(
    tmp_path: Path,
) -> None:
    """D42's guard behind every chosen allowance and the thinking-cap inference:
    if a response ever costs more than the worst case reserved for it, the bound
    was wrong, and the next call is refused rather than trusted."""
    log = tmp_path / "audit.jsonl"
    request = ask(max_output_tokens=100)
    over_the_cap = {
        "prompt_token_count": 10,
        "candidates_token_count": 100,
        "thoughts_token_count": 5_000,
    }
    with (
        Paveo.from_policy(
            policy_for("gemini-2.5-flash"), audit_path=log, now=make_clock()
        ) as pf,
        pf.session(agent_id="bot") as s,
    ):
        s.check_llm(request, shape="gemini").record(over_the_cap)
        with pytest.raises(PricingUnknown, match="cost more than the worst case"):
            s.check_llm(request, shape="gemini")

    settle = records_in(log)[1]
    assert settle["price_table_stale"] is True


# What /code-review found in S4c (D42), each pinned.


@pytest.mark.parametrize(
    "http",
    [
        {"extra_body": {"generationConfig": {"maxOutputTokens": 65536}}},
        {"base_url": "https://us-central1-aiplatform.googleapis.com"},
        {"headers": {"x-goog-extra": "1"}},
    ],
)
def test_http_options_cannot_carry_anything_past_the_reader(http: object) -> None:
    """`extra_body` is merged into the request body the SDK sends, so it could
    smuggle an eightfold cap or a server tool past every check here."""
    with pytest.raises(PricingUnknown, match="http_options"):
        bound(ask(http_options=http), assumed_max_output_tokens=None)


def test_a_timeout_and_retries_are_fine() -> None:
    request = ask(http_options={"timeout": 30_000, "retry_options": {"attempts": 2}})

    assert bound(request, assumed_max_output_tokens=None).max_output_tokens == 1000


def test_a_model_dump_config_full_of_nones_is_read_as_unset_fields() -> None:
    """The documented route: GenerateContentConfig(...).model_dump() writes every
    field, most of them None."""
    config = {
        "max_output_tokens": 100,
        "cached_content": None,
        "response_modalities": None,
        "candidate_count": None,
        "labels": None,
        "http_options": None,
    }
    request = {"model": "gemini-2.5-flash", "contents": "hi", "config": config}

    assert bound(request, assumed_max_output_tokens=None).max_output_tokens == 100


def test_a_response_whose_usage_is_none_is_unreadable_not_free() -> None:
    with pytest.raises(ConfigError):
        usage_classes({"candidates": [], "usage_metadata": None, "model_version": "x"})


def test_a_call_on_an_assumed_output_may_exceed_it_without_stopping_the_table(
    tmp_path: Path,
) -> None:
    """§4.4: the assumed cap caps nothing, so going past it is the documented
    approximation, not a broken bound."""
    policy = policy_for("gemini-2.5-flash")
    policy["defaults"] = {"assumed_max_output_tokens": 100}
    request = ask()
    request["config"] = {}
    with (
        Paveo.from_policy(
            policy, audit_path=tmp_path / "a.jsonl", now=make_clock()
        ) as pf,
        pf.session(agent_id="bot") as s,
    ):
        s.check_llm(request, shape="gemini").record(
            {"prompt_token_count": 5, "candidates_token_count": 5_000}
        )
        s.check_llm(request, shape="gemini").release()


def test_the_breach_refusal_names_the_model_and_the_real_cause(tmp_path: Path) -> None:
    request = ask(max_output_tokens=10)
    with (
        Paveo.from_policy(
            policy_for("gemini-2.5-flash"),
            audit_path=tmp_path / "a.jsonl",
            now=make_clock(),
        ) as pf,
        pf.session(agent_id="bot") as s,
    ):
        s.check_llm(request, shape="gemini").record(
            {"prompt_token_count": 5, "candidates_token_count": 5_000}
        )
        with pytest.raises(PricingUnknown) as refused:
            s.check_llm(request, shape="gemini")

    assert "gemini-2.5-flash" in str(refused.value)
    assert "restarts" in refused.value.remedy
    assert "token class" not in str(refused.value)


def test_a_call_admitted_just_before_the_change_is_reserved_at_the_dearer_price() -> (
    None
):
    """It may be billed after midnight, at the new price."""
    t = table(datetime(2026, 12, 31, 12, 0, tzinfo=UTC))
    rates = t.resolve("gemini-3.8-flash", {})

    assert t.worst_case(rates, 0, 1_000_000) == Decimal("7.50")


def test_a_call_admitted_before_the_change_and_settled_after_pays_the_new_price() -> (
    None
):
    moment = {"now": BEFORE_2027}
    t = _PriceTable(_ALL_LISTINGS, now=lambda: moment["now"])
    rates = t.resolve("gemini-3.8-flash", {})
    moment["now"] = FROM_2027

    charge = t.actual(rates, {"input": 0, "output": 1_000_000})

    assert charge.cost == Decimal("7.50")
    assert charge.rate_key.endswith("|from=2027-01-01")


def test_a_second_dated_change_gets_rows_of_its_own() -> None:
    """Every link of a chain of changes is built, not only the first."""
    from paveo.prices import _GeminiListing  # noqa: PLC0415

    chained = _GeminiListing(
        ("1", "0.1", "2"),
        changes=(
            "2027-01-01",
            _GeminiListing(
                ("3", "0.3", "4"),
                changes=("2027-07-01", _GeminiListing(("5", "0.5", "6"))),
            ),
        ),
    )
    t = _PriceTable({"m": chained}, now=lambda: datetime(2027, 8, 1, tzinfo=UTC))

    assert per_million(t.resolve("m", {}).per_token["output"]) == Decimal("6")


# What /security-review found in S4c (D42).


def test_a_function_result_carrying_media_is_refused() -> None:
    """FunctionResponse.parts can hold a video by URL; tool output is what an
    agent's inputs reach."""
    request = ask()
    request["contents"] = [
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "f",
                        "response": {},
                        "parts": [{"file_data": {"file_uri": "https://x/v.mp4"}}],
                    }
                }
            ],
        }
    ]

    with pytest.raises(PricingUnknown, match=r"function_response\.parts"):
        bound(request, assumed_max_output_tokens=None)


@pytest.mark.parametrize("usage", [{}, {"candidates_token_count": 10}])
def test_usage_with_no_prompt_count_is_unreadable_not_free(usage: object) -> None:
    with pytest.raises(ConfigError, match="prompt_token_count"):
        usage_classes(usage)  # type: ignore[arg-type]
