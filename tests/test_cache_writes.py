"""OpenAI cache writes: never charged below the bill (D115).

The design was reviewed three times before this code (paveo-notes, DESIGN-0.1.13).
OpenAI's SDK, generated from its API spec (28 Oct 2026), reports a write count
and no lifetime split, and supports the 30-minute lifetime only. So: a missing
write count is not a zero, nothing the adapter does not read is dropped, names
that collide add, and the long-context row is chosen from every input count.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from paveo._openai import bound, usage_classes
from paveo.enforce import _counts
from paveo.errors import PricingUnknown
from paveo.prices import _ALL_LISTINGS, _PriceTable

MILLION = Decimal(1_000_000)
# gpt-5.6-sol as OpenAI's pricing page prints it (read 27 Oct): input, cached
# input and output per million, a 30-minute write at 1.25x input, the long
# rows above 272K at 2x input and 1.5x output, and 10% regional uplift.
SOL_IN, SOL_CACHED, SOL_OUT = Decimal(4), Decimal("0.4"), Decimal(20)
UPLIFT, LONG_AT = Decimal("1.1"), 272_000


def table() -> _PriceTable:
    return _PriceTable(_ALL_LISTINGS)


def charge(model: str, usage: dict[str, object]) -> tuple[Decimal, bool]:
    t = table()
    rates = t.resolve(model, {"service_tier": "default"} if "gpt" in model else {})
    result = t.actual(rates, usage_classes(usage))
    return result.cost, result.price_table_stale


def bill(uncached: int, cached: int, writes: int, output: int) -> Decimal:
    """OpenAI's bill for gpt-5.6-sol, computed from the page, not from Paveo."""
    prompt = uncached + cached + writes
    long = prompt > LONG_AT
    rate_in = SOL_IN * (2 if long else 1)
    rate_cached = SOL_CACHED * (2 if long else 1)
    rate_out = SOL_OUT * (Decimal("1.5") if long else 1)
    total = (
        uncached * rate_in
        + cached * rate_cached
        + writes * rate_in * Decimal("1.25")
        + output * rate_out
    )
    return total * UPLIFT / MILLION


def usage(prompt: int = 10_000, **details: object) -> dict[str, object]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": 10,
        "prompt_tokens_details": details,
    }


# --------------------------------------------------------------------------
# A missing write count is not a zero
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        usage(cached_tokens=0),  # field absent
        usage(cached_tokens=0, cache_write_tokens=None),  # dumped as None
        {"prompt_tokens": 10_000, "completion_tokens": 10},  # no details
        {
            "prompt_tokens": 10_000,
            "completion_tokens": 10,
            "prompt_tokens_details": None,
        },
    ],
)
def test_a_missing_write_count_is_charged_at_the_write_rate(
    response: dict[str, object],
) -> None:
    """A bridge that drops the field must not turn 8,000 writes into plain input."""
    cost, stale = charge("gpt-5.6-sol", response)
    assert not stale
    assert cost == bill(0, 0, 10_000, 10)  # every uncached token as a write
    assert cost >= bill(2_000, 0, 8_000, 10)


def test_a_present_zero_is_a_real_zero() -> None:
    cost, _ = charge("gpt-5.6-sol", usage(cached_tokens=0, cache_write_tokens=0))
    assert cost == bill(10_000, 0, 0, 10)


def test_a_reported_write_count_is_charged_exactly() -> None:
    cost, stale = charge(
        "gpt-5.6-sol", usage(cached_tokens=1_000, cache_write_tokens=3_000)
    )
    assert not stale
    assert cost == bill(6_000, 1_000, 3_000, 10)


@pytest.mark.parametrize("model", ["claude-opus-5-5", "gemini-2.5-flash"])
def test_a_missing_write_count_off_an_openai_row_is_plain_input(model: str) -> None:
    """Through LiteLLM a Claude or Gemini usage arrives in this shape and never
    carries OpenAI's write count: it costs what it did before, and stops nothing."""
    t = table()
    rates = t.resolve(model, {})
    unsplit = t.actual(rates, usage_classes(usage(cached_tokens=0)))
    plain = t.actual(rates, {"input": 10_000, "output": 10})
    assert not unsplit.price_table_stale
    assert unsplit.cost == plain.cost


def test_an_earlier_model_has_no_write_charge_so_missing_costs_nothing_more() -> None:
    t = table()
    rates = t.resolve("gpt-5.5", {"service_tier": "default"})
    unsplit = t.actual(rates, usage_classes(usage(cached_tokens=0)))
    plain = t.actual(rates, {"input": 10_000, "output": 10})
    assert unsplit.cost == plain.cost


# --------------------------------------------------------------------------
# Nothing the adapter does not read is dropped
# --------------------------------------------------------------------------


def test_a_lifetime_breakdown_stops_the_table_and_is_charged_highest() -> None:
    """The day OpenAI adds one, Paveo stops rather than under-charge."""
    response = usage(
        cached_tokens=0,
        cache_write_tokens=8_000,
        cache_write_tokens_details={"ttl_30m_tokens": 3_000, "ttl_12h_tokens": 5_000},
    )
    classes = usage_classes(response)
    assert classes["unpriced_prompt_cache_write_tokens_details"] == 8_000
    cost, stale = charge("gpt-5.6-sol", response)
    assert stale
    assert cost >= bill(2_000, 0, 8_000, 10)


@pytest.mark.parametrize(
    "unreadable", [{"x": "8000"}, {"a": {"b": 1}}, "8000", 7.5, True, ["x"]]
)
def test_an_unreadable_breakdown_is_charged_at_the_most_it_could_describe(
    unreadable: object,
) -> None:
    response = usage(cached_tokens=1_000, cache_write_tokens=0, surprise=unreadable)
    assert usage_classes(response)["unpriced_prompt_surprise"] == 9_000
    _, stale = charge("gpt-5.6-sol", response)
    assert stale


@pytest.mark.parametrize("nothing", [None, 0, {}, [], {"a": 0, "b": None}])
def test_nothing_in_a_detail_stops_nothing(nothing: object) -> None:
    """An SDK dumping optional fields must not stop every agent in the process."""
    response = usage(cached_tokens=0, cache_write_tokens=0, extra=nothing)
    _, stale = charge("gpt-5.6-sol", response)
    assert not stale


def test_the_fields_that_are_already_counted_stay_skipped() -> None:
    response = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "text_tokens": 100,
            "audio_tokens": 0,
        },
        "completion_tokens_details": {
            "reasoning_tokens": 40,
            "text_tokens": 10,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 3,
        },
    }
    assert not any(name.startswith("unpriced_") for name in usage_classes(response))


def test_a_cache_count_under_completion_details_is_not_skipped() -> None:
    response = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "completion_tokens_details": {"cache_write_tokens": 5},
    }
    assert usage_classes(response)["unpriced_completion_cache_write_tokens"] == 5


# --------------------------------------------------------------------------
# Names that collide add; no field can overwrite a priced class
# --------------------------------------------------------------------------


def test_a_top_level_field_named_like_a_priced_class_cannot_zero_it() -> None:
    response = {"prompt_tokens": 10_000, "completion_tokens": 5_000, "output": 0}
    classes = usage_classes(response)
    assert classes["output"] == 5_000
    assert classes["unpriced_output"] == 0


def test_a_top_level_none_is_nothing_and_an_unreadable_one_is_charged() -> None:
    assert "unpriced_cost" not in usage_classes(
        {"prompt_tokens": 10, "completion_tokens": 1, "cost": None}
    )
    classes = usage_classes({"prompt_tokens": 100, "completion_tokens": 7, "x": "?"})
    assert classes["unpriced_x"] == 107


def test_two_keys_that_print_alike_keep_both_counts() -> None:
    response = usage(cached_tokens=0, cache_write_tokens=0, **{"\x00a": 5, "\x00b": 7})
    names = [n for n in usage_classes(response) if n.startswith("unpriced_")]
    assert len(names) == 2  # distinct names, each its own count
    classes = usage_classes(response)
    assert sum(int(str(classes[n])) for n in names) == 12


def test_a_long_or_odd_key_still_prints_within_64_characters() -> None:
    key = "Cache Write Tokens Details For A Lifetime Nobody Has Priced Yet (beta)"
    response = usage(cached_tokens=0, cache_write_tokens=0, **{key: 1})
    (name,) = [n for n in usage_classes(response) if n.startswith("unpriced_")]
    assert len(name) <= 64
    assert name == name.lower()
    assert all(c.isalnum() or c == "_" for c in name)


def test_the_audit_record_adds_names_that_print_alike() -> None:
    assert _counts({"\x00a": 5, "\x00b": 7, "input": 3}) == {
        "<unprintable class name>": 12,
        "input": 3,
    }


# --------------------------------------------------------------------------
# The price table: aliases add, and sizing counts every input token
# --------------------------------------------------------------------------


def test_aliases_landing_on_one_name_add() -> None:
    t = table()
    rates = t.resolve("claude-opus-5-5", {})
    both = t.actual(rates, {"input": 3_000, "input_unsplit": 7_000, "output": 0})
    plain = t.actual(rates, {"input": 10_000, "output": 0})
    assert both.cost == plain.cost


def test_an_alias_with_a_bad_count_still_lets_the_unknown_class_stop_the_table() -> (
    None
):
    t = table()
    rates = t.resolve("claude-opus-5-5", {})
    with pytest.raises(ValueError, match="input_unsplit"):
        t.actual(rates, {"input": 5, "input_unsplit": -1, "new_class": 3, "output": 0})
    with pytest.raises(PricingUnknown):
        t.resolve("claude-opus-5-5", {})


def test_an_unpriced_class_moves_a_call_onto_the_long_context_row() -> None:
    t = table()
    rates = t.resolve("gpt-6-sol", {"service_tier": "default"})
    result = t.actual(
        rates, {"input": 200_000, "unpriced_prompt_x": 100_000, "output": 10}
    )
    assert result.rate_key.endswith("|long_context")


# --------------------------------------------------------------------------
# Request side: only the 30-minute lifetime can be asked for
# --------------------------------------------------------------------------


def _with_part(part: dict[str, object]) -> dict[str, object]:
    return {
        "model": "gpt-5.6-sol",
        "service_tier": "default",
        "max_completion_tokens": 100,
        "messages": [{"role": "user", "content": [part]}],
    }


@pytest.mark.parametrize(
    "breakpoint_", [{"ttl": "12h"}, {"ttl": "24h"}, "12h", None, {"ttl": None}]
)
def test_a_part_asking_for_another_cache_lifetime_is_refused(
    breakpoint_: object,
) -> None:
    part = {"type": "text", "text": "hi", "prompt_cache_breakpoint": breakpoint_}
    with pytest.raises(PricingUnknown, match="prompt_cache_breakpoint"):
        bound(_with_part(part), assumed_max_output_tokens=None)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "text", "text": "hi"},
        {"type": "text", "text": "hi", "prompt_cache_breakpoint": {}},
        {"type": "text", "text": "hi", "prompt_cache_breakpoint": {"ttl": "30m"}},
        # LiteLLM's Claude caching rides on parts; it must keep working.
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
    ],
)
def test_a_part_with_the_30_minute_lifetime_or_none_is_admitted(
    part: dict[str, object],
) -> None:
    bound(_with_part(part), assumed_max_output_tokens=None)


def test_prompt_cache_options_stays_refused() -> None:
    request = _with_part({"type": "text", "text": "hi"})
    request["prompt_cache_options"] = {"ttl": "30m"}
    with pytest.raises(PricingUnknown, match="prompt_cache_options"):
        bound(request, assumed_max_output_tokens=None)


# --------------------------------------------------------------------------
# The property: never below the bill
# --------------------------------------------------------------------------

_FIELD = st.sampled_from(["reported", "absent", "none", "no_details"])


@settings(max_examples=300, deadline=None)
@given(
    uncached=st.integers(0, 400_000),
    cached=st.integers(0, 100_000),
    writes=st.integers(0, 100_000),
    output=st.integers(0, 50_000),
    field=_FIELD,
    extra=st.dictionaries(
        st.text(min_size=1, max_size=12),
        st.one_of(st.none(), st.integers(0, 10_000), st.text(max_size=4)),
        max_size=3,
    ),
)
def test_the_charge_is_never_below_openai_s_bill(  # noqa: PLR0913 - one per generated fact
    *,
    uncached: int,
    cached: int,
    writes: int,
    output: int,
    field: str,
    extra: dict[str, object],
) -> None:
    """The true counts come first; the response then reports, drops or blanks
    the write count, and carries arbitrary extra fields. Paveo's charge must
    not be below OpenAI's bill for the true counts. (A bridge that writes a
    false 0 is the one case no reader can tell from no writes: the README says
    so, and it is not generated here.)"""
    prompt = uncached + cached + writes
    details: dict[str, object] = {"cached_tokens": cached}
    if field == "reported":
        details["cache_write_tokens"] = writes
    elif field == "none":
        details["cache_write_tokens"] = None
    response: dict[str, object] = {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        **{f"x{k}": v for k, v in extra.items()},
    }
    if field != "no_details":
        response["prompt_tokens_details"] = details
    elif cached:
        return  # with no details, the cached count is unknown: covered above
    cost, _ = charge("gpt-5.6-sol", response)
    assert cost >= bill(uncached, cached, writes, output)


@pytest.mark.parametrize(
    "usage_",
    [
        # The review's order: the alias lands first, the target's own name second.
        {"cache_write": "x", "cache_write_1h": 100, "input": 1, "output": 0},
        {"input_unsplit": -1, "input": 5, "output": 0},
        {"input": 5, "input_unsplit": -1, "output": 0},
    ],
)
def test_a_bad_count_landing_on_an_alias_is_never_lost(
    usage_: dict[str, object],
) -> None:
    t = table()
    rates = t.resolve("claude-opus-5-5", {})
    with pytest.raises(ValueError, match="usage"):
        t.actual(rates, usage_)


@pytest.mark.parametrize("value", ["12h", "30m", "", None, 24])
def test_prompt_cache_retention_takes_only_the_values_openai_lists(
    value: object,
) -> None:
    request = _with_part({"type": "text", "text": "hi"})
    request["prompt_cache_retention"] = value
    with pytest.raises(PricingUnknown, match="prompt_cache_retention"):
        bound(request, assumed_max_output_tokens=None)


@pytest.mark.parametrize("value", ["in_memory", "24h"])
def test_prompt_cache_retention_values_openai_lists_are_admitted(value: str) -> None:
    request = _with_part({"type": "text", "text": "hi"})
    request["prompt_cache_retention"] = value
    bound(request, assumed_max_output_tokens=None)
