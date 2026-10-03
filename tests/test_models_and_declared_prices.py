"""Which models an agent may call, and the door for providers without a table.

Model rules are deny-by-default like tool rules (§5.1). A declared price is the
way any provider without an adapter comes in (§4.8.4, D39), and the generic
reader bounds its calls by what holds for every tokenizer: text is never fewer
bytes than tokens. Anything else it refuses.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from conftest import SENTINEL
from paveo import ConfigError, PricingUnknown
from paveo._generic import bound, usage_classes
from paveo._policy_document import load_document
from paveo.policy import UNDECLARED_MODEL, Policy
from paveo.prices import _LISTINGS, _PriceTable


def policy(**extra: object) -> Policy:
    document: dict[str, object] = {
        "version": 1,
        "policy_id": "models",
        "agents": [
            {
                "id": "bot",
                "budget": {"period": "day", "limit_usd": "1.00"},
                "models": {
                    "allow": ["claude-sonnet-5", "mistral-large-3"],
                    "deny": ["claude-fable-5-1"],
                },
            },
            {"id": "tools-only", "models": {"allow": ["claude-sonnet-5"]}},
        ],
    }
    document.update(extra)
    return load_document(document)


def test_an_allowed_model_with_a_budget_is_permitted() -> None:
    assert policy().evaluate_model("bot", "claude-sonnet-5") is None


@pytest.mark.parametrize(
    ("agent", "model", "reason"),
    [
        ("nobody", "claude-sonnet-5", "agent_unknown"),
        ("bot", "claude-fable-5-1", "model_denied"),
        ("bot", "claude-opus-5", "model_not_allowed"),
    ],
)
def test_every_other_model_call_is_refused(agent: str, model: str, reason: str) -> None:
    """Deny by default, and no ceiling is not an open one (§5.1)."""
    denial = policy().evaluate_model(agent, model)

    assert denial is not None
    assert denial.reason == reason


def test_no_ceiling_is_not_an_open_one() -> None:
    """Its own check, not a model rule, so shadow mode never reaches it (D48)."""
    denial = policy().evaluate_budget("tools-only")

    assert denial is not None
    assert denial.reason == "no_budget"
    assert policy().evaluate_budget("bot") is None


def test_a_model_the_policy_never_names_is_not_repeated_back() -> None:
    """D26 for models: a name nobody wrote may have been chosen under injection."""
    denial = policy().evaluate_model("bot", SENTINEL)

    assert denial is not None
    assert denial.rule == UNDECLARED_MODEL
    assert SENTINEL not in denial.remedy
    assert not policy().declares_model("bot", SENTINEL)
    assert policy().declares_model("bot", "claude-fable-5-1")


def declared(**prices: object) -> Policy:
    return policy(prices=prices)


def test_a_declared_price_loads_as_an_exact_decimal() -> None:
    loaded = declared(
        **{
            "mistral-large-3": {
                "input_per_mtok": "2.00",
                "output_per_mtok": "6.00",
                "cached_input_per_mtok": "0.20",
            }
        }
    )

    price = loaded.prices["mistral-large-3"]
    assert price.input_per_mtok == Decimal("2.00")
    assert price.cached_input_per_mtok == Decimal("0.20")


@pytest.mark.parametrize(
    "entry",
    [
        {"input_per_mtok": "2.00"},
        {"output_per_mtok": "6.00"},
        {"input_per_mtok": "-1", "output_per_mtok": "6.00"},
        {"input_per_mtok": "2.00", "output_per_mtok": "6.00", "per_image": "0.01"},
        {"input_per_mtok": "NaN", "output_per_mtok": "6.00"},
        "2.00/6.00",
    ],
)
def test_a_declared_price_that_cannot_price_anything_is_refused(entry: object) -> None:
    with pytest.raises(ConfigError, match="prices"):
        declared(**{"mistral-large-3": entry})


def test_a_declared_price_may_not_overwrite_a_model_the_table_knows() -> None:
    """A typo there could only under-price a model whose real price we have."""
    loaded = declared(
        **{"claude-sonnet-5": {"input_per_mtok": "0.01", "output_per_mtok": "0.01"}}
    )

    with pytest.raises(ConfigError, match="already carries"):
        _PriceTable(_LISTINGS, declared=loaded.prices)


def test_a_declared_model_is_priced_at_exactly_what_the_policy_says() -> None:
    loaded = declared(
        **{
            "mistral-large-3": {
                "input_per_mtok": "2.00",
                "output_per_mtok": "6.00",
                "cached_input_per_mtok": "0.20",
            }
        }
    )
    table = _PriceTable(_LISTINGS, declared=loaded.prices)

    rates = table.resolve("mistral-large-3", {"service_tier": "default"})
    charge = table.actual(rates, {"input": 1_000_000, "cache_read": 1_000_000})

    assert rates.rate_key == "mistral-large-3|declared-by-policy"
    assert table.worst_case(rates, 1_000_000, 1_000_000) == Decimal("8.00")
    assert charge.cost == Decimal("2.20")
    assert not charge.price_table_stale


def test_a_class_the_operator_did_not_price_stops_the_table() -> None:
    """A declared model with no cached price, reporting cache reads: the policy
    never said what they cost (§4.8.3)."""
    loaded = declared(
        **{"mistral-large-3": {"input_per_mtok": "2.00", "output_per_mtok": "6.00"}}
    )
    table = _PriceTable(_LISTINGS, declared=loaded.prices)
    rates = table.resolve("mistral-large-3", {})

    assert table.actual(rates, {"cache_read": 10}).price_table_stale


def openai_shaped(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model": "mistral-large-3",
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Summarise the ticket."},
        ],
    }
    base.update(overrides)
    return base


def size(request: dict[str, object]) -> int:
    return len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode())


def test_a_text_request_of_any_shape_is_bounded_by_its_bytes() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}},
            },
        },
    }
    request = openai_shaped(tools=[tool])

    measured = bound(request, assumed_max_output_tokens=None)

    assert measured.input_upper_bound == size(request)
    assert measured.max_output_tokens == 512
    assert measured.modifiers == {}


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            [{"type": "image_url", "image_url": {"url": "https://x/a.png"}}],
            id="openai-image",
        ),
        pytest.param(
            [{"type": "input_image", "image_url": "data:..."}], id="openai-responses"
        ),
        pytest.param([{"inline_data": {"mime_type": "image/png"}}], id="gemini-inline"),
        pytest.param([{"type": "input_audio", "input_audio": {}}], id="audio"),
        pytest.param([{"type": "file", "file": {"file_id": "f"}}], id="file"),
        pytest.param([{"type": "image", "source": {}}], id="anthropic-image"),
    ],
)
def test_anything_that_is_not_text_is_refused(content: object) -> None:
    request = openai_shaped(messages=[{"role": "user", "content": content}])

    with pytest.raises(PricingUnknown, match="not text") as caught:
        bound(request, assumed_max_output_tokens=None)
    assert "https://x" not in str(caught.value)


@pytest.mark.parametrize(
    ("caps", "expected"),
    [
        ({"max_completion_tokens": 300}, 300),
        ({"max_output_tokens": 200}, 200),
        ({"max_tokens": 100, "max_completion_tokens": 900}, 900),
    ],
)
def test_the_largest_output_cap_the_request_names_is_the_bound(
    caps: dict[str, object], expected: int
) -> None:
    request = openai_shaped(**caps)
    if "max_tokens" not in caps:
        del request["max_tokens"]

    assert bound(request, assumed_max_output_tokens=None).max_output_tokens == expected


def test_no_output_cap_is_refused_unless_the_policy_assumes_one() -> None:
    request = openai_shaped()
    del request["max_tokens"]

    with pytest.raises(PricingUnknown, match="output cap"):
        bound(request, assumed_max_output_tokens=None)
    assert bound(request, assumed_max_output_tokens=64).assumed_output


def test_reported_usage_must_use_the_names_the_policy_prices() -> None:
    assert usage_classes({"input": 5, "output": 6}) == {"input": 5, "output": 6}
    with pytest.raises(ConfigError, match="input, output and cache_read"):
        usage_classes({"prompt_tokens": 5})
    with pytest.raises(ConfigError, match="not a mapping"):
        usage_classes([5, 6])  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["n", "candidateCount", "candidate_count"])
def test_asking_for_several_answers_multiplies_the_output_bound(name: str) -> None:
    """OpenAI bills "the number of generated tokens across all of the choices",
    and the cap is per choice. Found reading the Chat Completions reference for
    S4b; without it an `n: 8` request is reserved at an eighth of its worst case."""
    measured = bound(openai_shaped(**{name: 8}), assumed_max_output_tokens=None)

    assert measured.max_output_tokens == 8 * 512


@pytest.mark.parametrize("value", [0, -1, True, "8"])
def test_a_number_of_choices_that_is_not_a_count_is_refused(value: object) -> None:
    with pytest.raises(PricingUnknown, match="choices"):
        bound(openai_shaped(n=value), assumed_max_output_tokens=None)


def test_gemini_keeps_its_cap_and_its_choices_one_level_down() -> None:
    request = {
        "model": "mistral-large-3",
        "contents": [{"parts": [{"text": "hi"}]}],
        "generationConfig": {"maxOutputTokens": 300, "candidateCount": 2},
    }

    assert bound(request, assumed_max_output_tokens=10).max_output_tokens == 600


def test_best_of_multiplies_the_output_bound_too() -> None:
    assert (
        bound(
            openai_shaped(best_of=3), assumed_max_output_tokens=None
        ).max_output_tokens
        == 3 * 512
    )


def test_the_arguments_of_a_past_tool_call_are_not_content() -> None:
    """Found by /code-review: `{"file": ...}` in a tool call's arguments is an
    argument name, and refusing it would deny every read_file agent."""
    request = openai_shaped(
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t",
                        "name": "read",
                        "input": {"file": "a.txt"},
                    }
                ],
            },
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "see", "args": {"image": "id"}}}],
            },
        ]
    )

    assert bound(request, assumed_max_output_tokens=None).input_upper_bound > 0


def test_a_type_that_is_a_list_does_not_crash_the_scan() -> None:
    request = openai_shaped(response_format={"type": ["string", "null"]})

    assert bound(request, assumed_max_output_tokens=None).input_upper_bound > 0


@pytest.mark.parametrize(
    "extra",
    [
        {"web_search_options": {}},
        {"search_parameters": {"mode": "on"}},
        {"previous_response_id": "resp_1"},
        {"conversation": "conv_1"},
        {"prompt": {"id": "pmpt_1"}},
        {"cachedContent": "cachedContents/1"},
        {"mcp_servers": []},
        {"container": "c"},
        {"tools": [{"type": "web_search"}]},
        {"tools": [{"url_context": {}}]},
        {"tools": [{"type": "web_search_20250305", "name": "w"}]},
    ],
)
def test_a_server_tool_or_a_reference_is_refused_at_the_door(
    extra: dict[str, object],
) -> None:
    """Found by /security-review (D40, HIGH): the door refused only the non-text
    words it knew, so a server tool or a reference to earlier content was
    admitted at a reservation of its bytes and billed at whatever it pulled in."""
    with pytest.raises(PricingUnknown, match="cannot bound"):
        bound(openai_shaped(**extra), assumed_max_output_tokens=None)


def test_function_tools_in_either_spelling_are_admitted() -> None:
    tools = [
        {"type": "function", "function": {"name": "f", "parameters": {}}},
        {"functionDeclarations": [{"name": "g"}]},
    ]

    assert bound(openai_shaped(tools=tools), assumed_max_output_tokens=None)


def test_every_content_part_known_to_be_text_is_admitted() -> None:
    request = openai_shaped(
        system=[{"type": "text", "text": "Be brief."}],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Refund A-1."}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Check first.", "signature": "s"},
                    {"type": "text", "text": "Checked.", "citations": []},
                    {"type": "refusal", "refusal": "No."},
                ],
                "tool_calls": [
                    {"id": "c", "index": 0, "type": "function", "function": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "is_error": False,
                        "content": [{"type": "text", "text": "done"}],
                    }
                ],
            },
        ],
    )

    assert bound(request, assumed_max_output_tokens=None).input_upper_bound == size(
        request
    )


@pytest.mark.parametrize(
    "request_",
    [
        pytest.param(
            openai_shaped(
                system=[
                    {
                        "type": "text",
                        "text": "Long policy.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            id="cache-write",
        ),
        pytest.param(
            openai_shaped(
                messages=[{"role": "user", "content": [{"type": "reference"}]}]
            ),
            id="unknown-type",
        ),
        pytest.param(
            openai_shaped(messages=[{"role": "user", "content": [{"text": "hi"}]}]),
            id="no-type",
        ),
        pytest.param(
            openai_shaped(messages=[{"role": "user", "content": ["hi"]}]),
            id="not-a-part",
        ),
        pytest.param(
            openai_shaped(
                messages=[
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        },
                    }
                ]
            ),
            id="a-lone-part-not-in-a-list",
        ),
        pytest.param(
            openai_shaped(
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "x"}],
                    }
                ]
            ),
            id="redacted-thinking",
        ),
        pytest.param(
            openai_shaped(
                messages=[
                    {
                        "role": "assistant",
                        "tool_calls": [{"type": "custom", "custom": {}}],
                    }
                ]
            ),
            id="custom-tool-call",
        ),
        pytest.param(
            openai_shaped(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t",
                                "content": [{"type": "search_result"}],
                            }
                        ],
                    }
                ]
            ),
            id="inside-a-tool-result",
        ),
    ],
)
def test_a_content_part_not_known_to_be_text_is_refused(
    request_: dict[str, object],
) -> None:
    """D40's open item, closed in B7 (D73): the word list refuses only the
    spellings of "not text" it names, so a part is now admitted only by type and
    keys. A cache write on a text part was admitted before this, at the input
    price, and it bills at 1.25x or 2x. A lone part where a list belongs was
    admitted by the first version (/code-review)."""
    with pytest.raises(PricingUnknown, match="not a plain content part") as caught:
        bound(request_, assumed_max_output_tokens=None)
    assert "ephemeral" not in str(caught.value)
    assert "reference" not in str(caught.value)


def test_a_string_that_is_not_unicode_is_a_wiring_error() -> None:
    request = openai_shaped(messages=[{"role": "user", "content": "x\ud800"}])

    with pytest.raises(ConfigError, match="Unicode"):
        bound(request, assumed_max_output_tokens=None)


@pytest.mark.parametrize(
    "modifiers",
    [
        {"service_tier": "priority"},
        {"service_tier": "fast"},
        {"service_tier": "auto"},
        {"service_tier": "anything"},
        {"speed": "fast"},
        {"inference_geo": "us"},
    ],
)
def test_a_premium_setting_on_a_policy_priced_model_is_refused_not_dropped(
    modifiers: dict[str, object],
) -> None:
    """Found by /security-review (D41, HIGH): the declared row has no Fast row,
    so the setting was dropped and a Fast call charged at the declared standard
    rate, about half its bill."""
    loaded = declared(
        **{"mistral-large-3": {"input_per_mtok": "2.00", "output_per_mtok": "6.00"}}
    )
    table = _PriceTable(_LISTINGS, declared=loaded.prices)

    with pytest.raises(PricingUnknown, match="does not cover"):
        table.resolve("mistral-large-3", modifiers)
