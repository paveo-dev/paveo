"""The price table (``docs/SPEC_V1.md`` §4.8, §11 "Pricing").

The ledger's guarantee is only as good as the bound it is handed, so these tests
are about that bound: that every rate is the one the provider publishes, that the
reservation can never be smaller than the charge, and that a table which can no
longer promise either says so and refuses.
"""

from __future__ import annotations

import copy
import dataclasses
import decimal
import threading
from decimal import ROUND_FLOOR, Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from conftest import SENTINEL
from paveo import PricingUnknown
from paveo.prices import (
    _ABSENT,
    _LISTINGS,
    OUTPUT_CLASS,
    PRICES_VERSION,
    Rates,
    _modifier_values,
    _PriceTable,
)

MILLION = Decimal(1_000_000)

# Anthropic's own input classes. The table's INPUT_CLASSES also holds OpenAI's
# single `cache_write`, which no Anthropic row prices.
ANTHROPIC_INPUT = ("input", "cache_read", "cache_write_5m", "cache_write_1h")

# The row the page's headline prices are for. Said explicitly, because an unset
# `inference_geo` is deliberately priced as the dearer geo (D38).
GLOBAL = {"inference_geo": "global"}

# The pricing page's own absolute figures, USD per million tokens, read on
# 2026-09-23: base input, 5m cache write, 1h cache write, cache hit, output.
# Written as the page prints them rather than derived, so a wrong multiplier in
# prices.py disagrees with this rather than being copied into it.
PUBLISHED = {
    "claude-fable-5-1": ("10", "12.50", "20", "0.25", "50"),
    "claude-fable-5": ("10", "12.50", "20", "1", "50"),
    "claude-opus-5-5": ("4", "5", "8", "0.20", "20"),
    "claude-opus-5": ("5", "6.25", "10", "0.50", "25"),
    "claude-opus-4-8": ("5", "6.25", "10", "0.50", "25"),
    "claude-opus-4-7": ("5", "6.25", "10", "0.50", "25"),
    "claude-opus-4-6": ("5", "6.25", "10", "0.50", "25"),
    "claude-sonnet-5": ("2", "2.50", "4", "0.20", "10"),
    "claude-sonnet-4-6": ("3", "3.75", "6", "0.30", "15"),
    "claude-haiku-4-5": ("1", "1.25", "2", "0.10", "5"),
    "claude-haiku-4-5-20251001": ("1", "1.25", "2", "0.10", "5"),
}

# Fast mode's input/output pair from the same page. Opus 4.6 accepts the
# parameter and bills it at standard rates.
PUBLISHED_FAST = {
    "claude-opus-5-5": ("8", "40"),
    "claude-opus-5": ("10", "50"),
    "claude-opus-4-8": ("10", "50"),
    "claude-opus-4-6": ("5", "25"),
}

# The page applies the 1.1x US-inference multiplier to "Claude 4.6 and later".
NO_US_INFERENCE = {"claude-haiku-4-5", "claude-haiku-4-5-20251001"}

ALL_ROWS = [
    (model, speed, geo)
    for model, listing in _LISTINGS.items()
    for speed in (("standard", "fast") if listing.fast else ("standard",))
    for geo in (("global",) if model in NO_US_INFERENCE else ("global", "us"))
]


def fresh() -> _PriceTable:
    """A table of our own, so marking it stale cannot leak into another test."""
    return _PriceTable(_LISTINGS)


def per_million(rates: Rates, name: str) -> Decimal:
    return rates.per_token[name] * MILLION


def test_the_table_prices_exactly_the_models_the_page_was_read_for() -> None:
    assert set(_LISTINGS) == set(PUBLISHED)


@pytest.mark.parametrize("model", sorted(PUBLISHED))
def test_every_standard_rate_is_the_one_the_page_publishes(model: str) -> None:
    rates = fresh().resolve(model, GLOBAL)
    base, write_5m, write_1h, hit, output = map(Decimal, PUBLISHED[model])

    assert per_million(rates, "input") == base
    assert per_million(rates, "cache_write_5m") == write_5m
    assert per_million(rates, "cache_write_1h") == write_1h
    assert per_million(rates, "cache_read") == hit
    assert per_million(rates, OUTPUT_CLASS) == output
    assert set(rates.per_token) == {*ANTHROPIC_INPUT, OUTPUT_CLASS}


@pytest.mark.parametrize("model", sorted(PUBLISHED_FAST))
def test_fast_mode_selects_the_published_fast_pair(model: str) -> None:
    rates = fresh().resolve(model, {"speed": "fast", **GLOBAL})
    fast_in, fast_out = map(Decimal, PUBLISHED_FAST[model])

    assert per_million(rates, "input") == fast_in
    assert per_million(rates, OUTPUT_CLASS) == fast_out
    # "Prompt caching multipliers apply on top of fast mode pricing."
    assert per_million(rates, "cache_write_1h") == fast_in * 2


@pytest.mark.parametrize(("model", "speed", "geo"), ALL_ROWS)
def test_every_permitted_modifier_resolves_to_its_own_row(
    model: str, speed: str, geo: str
) -> None:
    """§11: for every model and every permitted value, the table's rate, never a
    default. And the more expensive row is never priced as the cheaper one."""
    table = fresh()
    rates = table.resolve(model, {"speed": speed, "inference_geo": geo})
    standard = table.resolve(model, GLOBAL)

    assert rates.rate_key == f"{model}|speed={speed}|inference_geo={geo}"
    multiplier = Decimal("1.1") if geo == "us" else Decimal(1)
    base_in = Decimal(
        (PUBLISHED_FAST[model] if speed == "fast" else PUBLISHED[model])[0]
    )
    assert per_million(rates, "input") == base_in * multiplier
    for name, rate in rates.per_token.items():
        assert rate >= standard.per_token[name]


def test_us_inference_raises_every_class_by_a_tenth() -> None:
    """The page: "a 1.1x multiplier on all token pricing categories, including
    input tokens, output tokens, cache writes, and cache reads"."""
    table = fresh()
    us = table.resolve("claude-opus-5", {"speed": "fast", "inference_geo": "us"})
    fast = table.resolve("claude-opus-5", {"speed": "fast", **GLOBAL})

    for name in (*ANTHROPIC_INPUT, OUTPUT_CLASS):
        assert us.per_token[name] == fast.per_token[name] * Decimal("1.1")


@pytest.mark.parametrize("model", sorted(PUBLISHED))
def test_an_unset_inference_geo_is_priced_as_the_dearest_geo_the_model_takes(
    model: str,
) -> None:
    """Found by /security-review (D38). A workspace's `default_inference_geo`
    decides an unset parameter, we cannot see it, and orgs that had opted out of
    global routing were migrated to "us" with no code change. Priced at the API's
    documented "global", every such call is under-reserved and under-settled by a
    tenth, silently."""
    table = fresh()
    dearest = "global" if model in NO_US_INFERENCE else "us"

    unset = table.resolve(model, {})

    assert unset.per_token == table.resolve(model, {"inference_geo": dearest}).per_token
    # The audit record says the request left it unset, not that it asked for US
    # inference: a disputed 10% has to be explainable a year later (/code-review).
    assert unset.rate_key == f"{model}|speed=standard|inference_geo=unset"


def test_unset_is_not_a_value_a_request_may_send() -> None:
    with pytest.raises(PricingUnknown):
        fresh().resolve("claude-opus-5", {"inference_geo": "unset"})


def test_every_price_affecting_parameter_says_what_unset_means() -> None:
    """One list of parameters, checked rather than kept in step by hand: a
    parameter with no unset meaning would crash resolve() instead of denying."""
    for listing in _LISTINGS.values():
        assert set(_modifier_values(listing)) <= set(_ABSENT)


def test_a_parameter_that_does_not_affect_price_is_ignored() -> None:
    table = fresh()
    assert table.resolve("claude-sonnet-5", {"max_tokens": 5, **GLOBAL}) == (
        table.resolve("claude-sonnet-5", GLOBAL)
    )


def test_one_response_carrying_three_input_classes_is_priced_class_by_class() -> None:
    """§11 "the shape": a table with a single input rate cannot pass this."""
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)
    usage = {"input": 512, "cache_read": 8192, "cache_write_5m": 2048, "output": 300}

    charge = table.actual(rates, usage)

    expected = 512 * Decimal("2") + 8192 * Decimal("0.20")
    expected += 2048 * Decimal("2.50") + 300 * Decimal("10")
    assert charge.cost == expected / MILLION
    assert not charge.price_table_stale


@pytest.mark.parametrize("model", ["claude-opus-6", "", SENTINEL])
def test_a_model_the_table_does_not_carry_is_refused(model: str) -> None:
    with pytest.raises(PricingUnknown) as caught:
        fresh().resolve(model, GLOBAL)

    assert caught.value.model == model
    assert PRICES_VERSION in str(caught.value)
    assert "claude-sonnet-5" in caught.value.remedy


@pytest.mark.parametrize(
    ("model", "modifiers"),
    [
        pytest.param("claude-sonnet-5", {"speed": "fast"}, id="no-fast-mode"),
        pytest.param("claude-opus-4-7", {"speed": "fast"}, id="fast-is-an-error"),
        pytest.param("claude-opus-5", {"speed": "warp"}, id="unknown-speed"),
        pytest.param("claude-opus-5", {"inference_geo": "eu"}, id="unknown-geo"),
        pytest.param("claude-haiku-4-5", {"inference_geo": "us"}, id="pre-4-6"),
        pytest.param("claude-opus-5", {"speed": None}, id="not-a-string"),
        pytest.param("claude-opus-5", {"speed": SENTINEL}, id="payload-shaped"),
    ],
)
def test_a_price_affecting_value_the_table_does_not_carry_is_refused(
    model: str, modifiers: dict[str, object]
) -> None:
    """§4.8.2: we do not guess. The value itself is not written into the error:
    only the parameter's name and what it may be set to."""
    with pytest.raises(PricingUnknown) as caught:
        fresh().resolve(model, modifiers)

    (name,) = modifiers
    assert name in str(caught.value)
    assert "warp" not in str(caught.value)
    assert SENTINEL not in str(caught.value)


def test_an_unknown_class_is_charged_dearest_and_stops_every_reserve() -> None:
    """§4.8.3, all three steps, including the one people forget: once a response
    has shown the table to be incomplete, the next reserve is refused too, for
    every model, not only the one that reported it."""
    table = fresh()
    rates = table.resolve("claude-opus-5", GLOBAL)

    charge = table.actual(rates, {"input": 10, "cache_write_24h": 1000})

    output_rate = Decimal("25") / MILLION
    assert charge.cost == 10 * Decimal("5") / MILLION + 1000 * output_rate
    assert charge.price_table_stale
    with pytest.raises(PricingUnknown, match="cache_write_24h"):
        table.worst_case(rates, 10, 10)  # Rates obtained before it went stale
    for model in ("claude-opus-5", "claude-haiku-4-5"):
        with pytest.raises(PricingUnknown) as caught:
            table.resolve(model, GLOBAL)
        assert "cache_write_24h" in caught.value.detail
        assert PRICES_VERSION in caught.value.detail


def test_a_call_already_in_flight_still_settles_after_the_table_goes_stale() -> None:
    """A settle that raised would lose the charge of a call that really happened."""
    table = fresh()
    in_flight = table.resolve("claude-sonnet-5", GLOBAL)
    table.actual(in_flight, {"surprise": 1})

    charge = table.actual(in_flight, {"input": 1000, "output": 10})

    assert charge.cost == (1000 * Decimal(2) + 10 * Decimal(10)) / MILLION
    assert not charge.price_table_stale


def test_an_unknown_class_marks_the_table_stale_despite_a_malformed_response() -> None:
    """Found by both reviews. A bad count beside an unknown class raises, as a
    count no correct caller produces should (D24), but the table has seen the
    class and must refuse every later reserve regardless."""
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)

    with pytest.raises(ValueError, match="token count"):
        table.actual(rates, {"cache_write_24h": 5000, "input": None})  # type: ignore[dict-item]

    with pytest.raises(PricingUnknown, match="cache_write_24h"):
        table.resolve("claude-sonnet-5", GLOBAL)


def test_the_table_prices_its_own_row_never_the_callers_copy() -> None:
    """Found by /code-review, twice. A hand-built Rates could carry any rate, so
    the caller's numbers are never read. But refusing everything that is not the
    issued object made a copy's settle raise and lose its charge, so a copy of a
    real row settles at the real row's price."""
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)
    forged = Rates(
        model="claude-sonnet-5",
        rate_key=rates.rate_key,
        per_token=dict.fromkeys(rates.per_token, Decimal(0)),
    )
    usage = {"input": 1000, "output": 10}
    expected = table.actual(rates, usage).cost

    for candidate in (forged, copy.copy(rates), dataclasses.replace(rates)):
        assert table.worst_case(candidate, 1000, 10) == table.worst_case(
            rates, 1000, 10
        )
        assert table.actual(candidate, usage).cost == expected


def test_a_key_the_table_never_issued_is_refused() -> None:
    table = fresh()
    invented = Rates(
        model="claude-sonnet-5", rate_key="claude-sonnet-5|free", per_token={}
    )

    with pytest.raises(ValueError, match="not resolved by this price table"):
        table.worst_case(invented, 10, 10)
    with pytest.raises(ValueError, match="not resolved by this price table"):
        table.actual(invented, {"input": 10})


def test_a_non_token_field_passed_as_usage_raises_without_stopping_the_process() -> (
    None
):
    """Found by /code-review. An adapter that passes `inference_geo: "us"`
    through as if it were a count is a bug to raise, not evidence of a new token
    class; marking the table stale for it would stop every agent in the process."""
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)

    with pytest.raises(ValueError, match=r"usage\['inference_geo'\] is str"):
        table.actual(rates, {"input": 10, "inference_geo": "us"})  # type: ignore[dict-item]

    assert table.resolve("claude-sonnet-5", GLOBAL) is rates


def test_a_bool_is_called_a_bool() -> None:
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)

    with pytest.raises(ValueError, match="is a bool"):
        table.worst_case(rates, True, 0)  # type: ignore[arg-type]


def test_a_rejected_token_count_is_not_echoed() -> None:
    """Locked decision #5: the value came off a request or a response."""
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)

    for call in (
        lambda: table.worst_case(rates, SENTINEL, 0),  # type: ignore[arg-type]
        lambda: table.actual(rates, {"input": SENTINEL}),  # type: ignore[dict-item]
    ):
        with pytest.raises(ValueError, match="token count") as caught:
            call()
        assert SENTINEL not in str(caught.value)


def test_an_unprintable_class_name_is_not_written_into_the_error() -> None:
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)
    table.actual(rates, {f"bad {SENTINEL}": 1})

    with pytest.raises(PricingUnknown) as caught:
        table.resolve("claude-sonnet-5", GLOBAL)

    assert SENTINEL not in str(caught.value)
    assert "unprintable" in caught.value.detail


def test_the_first_unknown_class_is_the_one_reported() -> None:
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)
    table.actual(rates, {"first_class": 1})
    table.actual(rates, {"second_class": 1})

    with pytest.raises(PricingUnknown, match="first_class"):
        table.resolve("claude-sonnet-5", GLOBAL)


@pytest.mark.parametrize("count", [-1, True, 1.5, "10", None])
def test_a_token_count_no_correct_caller_produces_raises(count: object) -> None:
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)

    with pytest.raises(ValueError, match="token count"):
        table.worst_case(rates, count, 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token count"):
        table.worst_case(rates, 10, count)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token count"):
        table.actual(rates, {"input": count})  # type: ignore[dict-item]


@given(
    row=st.sampled_from(ALL_ROWS),
    input_upper_bound=st.integers(min_value=0, max_value=2_000_000),
    max_output_tokens=st.integers(min_value=0, max_value=128_000),
    cuts=st.lists(st.floats(min_value=0, max_value=1), min_size=3, max_size=3),
    output_share=st.floats(min_value=0, max_value=1),
)
def test_no_split_of_the_tokens_can_cost_more_than_was_reserved(
    row: tuple[str, str, str],
    input_upper_bound: int,
    max_output_tokens: int,
    cuts: list[float],
    output_share: float,
) -> None:
    """§11 "the reserve bound", and the test S3 is done on: over random per-class
    splits of the same input, ``actual <= worst_case``, always. It is what proves
    §4.1's ``max()`` is doing its job. The floats only choose where to cut; every
    count is an int and every sum is exact."""
    model, speed, geo = row
    table = fresh()
    rates = table.resolve(model, {"speed": speed, "inference_geo": geo})

    marks = sorted(int(input_upper_bound * cut) for cut in cuts)
    edges = [0, *marks, input_upper_bound]
    usage = {name: edges[i + 1] - edges[i] for i, name in enumerate(ANTHROPIC_INPUT)}
    usage[OUTPUT_CLASS] = int(max_output_tokens * output_share)

    charge = table.actual(rates, usage)

    assert sum(usage[name] for name in ANTHROPIC_INPUT) == input_upper_bound
    assert charge.cost <= table.worst_case(rates, input_upper_bound, max_output_tokens)


@pytest.mark.parametrize(("model", "speed", "geo"), ALL_ROWS)
def test_the_bound_is_reached_when_every_input_token_is_the_dearest_class(
    model: str, speed: str, geo: str
) -> None:
    """The bound is tight: `max()` over-reserves only when the split is kinder
    than the worst case, never by construction."""
    table = fresh()
    rates = table.resolve(model, {"speed": speed, "inference_geo": geo})

    charge = table.actual(rates, {"cache_write_1h": 1000, "output": 500})

    assert charge.cost == table.worst_case(rates, 1000, 500)


# What prices.py computes with. A new public method fails the guard below until
# it is added to the hostile-context test beside it (D35).
# `breached` and `carries` are listed but do no arithmetic: one marks the table
# stale, the other looks a name up.
TABLE_OPERATIONS = {"resolve", "worst_case", "actual", "breached", "carries"}


def test_every_pricing_operation_is_in_the_hostile_context_test() -> None:
    public = {
        name
        for name, member in vars(_PriceTable).items()
        if not name.startswith("_") and callable(member)
    }
    assert public == TABLE_OPERATIONS, (
        "a public operation was added to or removed from _PriceTable: run it "
        "under the hostile contexts below, then update this set"
    )
    assert not [
        name
        for name, member in vars(Rates).items()
        if not name.startswith("_") and callable(member)
    ], "Rates prices nothing; only the table does, so staleness cannot be skipped"


@pytest.mark.parametrize(
    "callers",
    [
        pytest.param(decimal.Context(prec=3, traps=[]), id="traps-switched-off"),
        pytest.param(decimal.Context(prec=6), id="precision-lowered"),
        pytest.param(decimal.Context(prec=6, rounding=ROUND_FLOOR), id="rounding-down"),
    ],
)
def test_the_callers_decimal_context_cannot_lower_a_price(
    callers: decimal.Context,
) -> None:
    """D35 again, where S3 was told it would come back. Tokens times rates is the
    sum a lowered precision rounds, and a rounded-down worst case is an
    under-reservation. Every figure below has more digits than these contexts
    hold, so arithmetic that escaped the ledger's own context would fail an
    equality."""
    reference = fresh()
    rates = reference.resolve("claude-opus-5", {"inference_geo": "us"})
    usage = {"input": 123_457, "cache_read": 987_653, "output": 54_321}
    expected_worst = reference.worst_case(rates, 1_234_567, 76_543)
    expected_cost = reference.actual(rates, usage).cost

    table = fresh()
    with decimal.localcontext(callers):
        hostile_rates = table.resolve("claude-opus-5", {"inference_geo": "us"})
        worst = table.worst_case(hostile_rates, 1_234_567, 76_543)
        cost = table.actual(hostile_rates, usage).cost

    assert hostile_rates == rates
    assert (
        worst
        == expected_worst
        == Decimal("1234567") * Decimal("0.000011")
        + Decimal("76543") * Decimal("0.0000275")
    )
    assert cost == expected_cost


def test_many_threads_marking_the_table_stale_leave_it_refusing() -> None:
    """The concurrent path (Rule 18). Every thread reports an unknown class at
    once; afterwards the table refuses, naming one of them, and no settle lost
    its charge."""
    table = fresh()
    rates = table.resolve("claude-sonnet-5", GLOBAL)
    start = threading.Barrier(8)
    costs: list[Decimal] = []

    def settle(n: int) -> None:
        start.wait()
        costs.append(table.actual(rates, {f"class_{n}": 1_000_000}).cost)

    threads = [threading.Thread(target=settle, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert costs == [Decimal(10)] * 8
    with pytest.raises(PricingUnknown, match=r"class_[0-7]"):
        table.resolve("claude-sonnet-5", GLOBAL)
