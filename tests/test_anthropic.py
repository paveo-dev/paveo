"""The Anthropic adapter: bounding a request and reading a response (D38, D39).

The bound is what the ledger reserves against, so every test here is a version
of one question: can a request cost more than this says? Where the answer would
be "yes, and we cannot say by how much", the request must be refused.
"""

from __future__ import annotations

import json

import pytest

from conftest import SENTINEL
from paveo import ConfigError, PricingUnknown
from paveo._anthropic import (
    _IMAGE_TOKENS,
    _TOOL_SYSTEM_PROMPT,
    bound,
    model_of,
    usage_classes,
)
from paveo.prices import _ALL_LISTINGS, _PriceTable

# A one-pixel PNG, base64: 92 bytes that are billed as an image.
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
    "/pLvAAAAAElFTkSuQmCC"
)


def request(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model": "claude-sonnet-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Summarise the attached notes."}],
    }
    base.update(overrides)
    return base


def size(req: dict[str, object]) -> int:
    return len(json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode())


def measure(req: dict[str, object], assumed: int | None = None) -> int:
    return bound(req, assumed_max_output_tokens=assumed).input_upper_bound


def user(*blocks: object) -> list[object]:
    return [{"role": "user", "content": list(blocks)}]


def test_a_text_request_is_bounded_by_its_own_bytes() -> None:
    """§4.5: no token is under one byte, so the bytes are the bound."""
    req = request(system="You are a careful assistant. Ünïcödé counts in bytes.")

    measured = bound(req, assumed_max_output_tokens=None)

    assert measured.input_upper_bound == size(req)
    assert measured.max_output_tokens == 1024
    assert measured.model == "claude-sonnet-5"
    assert not measured.assumed_output


def test_any_tool_adds_the_system_prompt_the_api_adds() -> None:
    tool = {"name": "lookup", "input_schema": {"type": "object"}}
    req = request(tools=[tool])

    assert measure(req) == size(req) + _TOOL_SYSTEM_PROMPT
    assert measure(request(tools=[])) == size(request(tools=[]))


@pytest.mark.parametrize(
    ("kind", "allowance"),
    [
        ("bash_20250124", 325),
        ("text_editor_20250728", 700),
        ("computer_toolset_20260801", 5_000),
        ("computer_20251124", 1_500),
        ("browser_toolset_20260801", 8_000),
    ],
)
def test_an_anthropic_defined_tool_adds_what_its_definition_costs(
    kind: str, allowance: int
) -> None:
    """Named in a few bytes, billed in hundreds or thousands of tokens (D38)."""
    req = request(tools=[{"type": kind, "name": "t"}])

    assert measure(req) == size(req) + _TOOL_SYSTEM_PROMPT + allowance


@pytest.mark.parametrize(
    "kind",
    [
        "web_search_20260209",
        "web_fetch_20260209",
        "code_execution_20260521",
        "tool_search_tool_regex_20251119",
        "mcp_toolset",
        "memory_20250818",
        "something_new_20270101",
    ],
)
def test_a_tool_whose_cost_cannot_be_bounded_is_refused(kind: str) -> None:
    """Server tools bill searches, fetched pages or container time beyond the
    request. An unknown type is refused too: an allowlist, not a denylist."""
    with pytest.raises(PricingUnknown, match=r"tools\[0\]"):
        measure(request(tools=[{"type": kind, "name": "t"}]))


def test_every_image_adds_the_most_an_image_can_cost() -> None:
    """A one-pixel PNG is 92 bytes of base64 and can be billed as an image. An
    agent reading the web is handed images by strangers, so the allowance is the
    cap, not an estimate of this image."""
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
    }
    req = request(messages=user(image, image, {"type": "text", "text": "Compare."}))

    assert measure(req) == size(req) + 2 * _IMAGE_TOKENS


def test_an_image_inside_a_tool_result_counts_too() -> None:
    """Computer-use screenshots come back this way."""
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
    }
    result = {"type": "tool_result", "tool_use_id": "t1", "content": [image]}
    req = request(messages=user(result))

    assert measure(req) == size(req) + _IMAGE_TOKENS


def test_a_plain_text_document_is_bounded_by_its_bytes() -> None:
    doc = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "notes"},
    }
    req = request(messages=user(doc))

    assert measure(req) == size(req)


def test_images_inside_a_custom_content_document_are_found() -> None:
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
    }
    doc = {"type": "document", "source": {"type": "content", "content": [image]}}
    req = request(messages=user(doc))

    assert measure(req) == size(req) + _IMAGE_TOKENS


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/a.png"},
            },
            id="image-by-url",
        ),
        pytest.param(
            {"type": "image", "source": {"type": "file", "file_id": "file_1"}},
            id="image-by-file-id",
        ),
        pytest.param(
            {"type": "document", "source": {"type": "file", "file_id": "file_1"}},
            id="document-by-file-id",
        ),
        pytest.param(
            {"type": "document", "source": {"type": "url", "url": "https://x/a.pdf"}},
            id="document-by-url",
        ),
        pytest.param(
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBE",
                },
            },
            id="inline-pdf",
        ),
        pytest.param({"type": "image"}, id="image-without-source"),
        pytest.param({"type": "container_upload", "file_id": "f"}, id="unknown-block"),
        pytest.param(
            {"type": "image_url", "image_url": {"url": "x"}}, id="openai-shape"
        ),
        pytest.param("just a string", id="not-a-block"),
    ],
)
def test_content_whose_tokens_are_not_in_its_bytes_is_refused(block: object) -> None:
    """A reference costs its content, and a PDF's per-page cost is not verified.
    An OpenAI-shaped block is refused rather than read as if it were ours."""
    with pytest.raises(PricingUnknown, match=r"messages\[0\]\.content\[0\]"):
        measure(request(messages=user(block)))


def test_content_in_the_system_prompt_is_walked_too() -> None:
    image = {"type": "image", "source": {"type": "url", "url": "https://x/a.png"}}
    with pytest.raises(PricingUnknown, match=r"system\[0\]"):
        measure(request(system=[image]))


@pytest.mark.parametrize(
    "extra",
    [
        {"mcp_servers": []},
        {"container": "c"},
        {"context_management": {"edits": []}},
        {"fallbacks": "default"},
        {"extra_body": {}},
        {"output_config": {"format": {"type": "json_schema"}}},
        {"output_config": "high"},
    ],
)
def test_a_parameter_whose_cost_is_not_verified_is_refused(
    extra: dict[str, object],
) -> None:
    """Fallbacks re-run the request on another model, compaction runs a second
    model call, MCP and containers bring content in from elsewhere."""
    with pytest.raises(PricingUnknown):
        measure(request(**extra))


def test_effort_and_other_plain_parameters_are_accepted() -> None:
    req = request(
        output_config={"effort": "high"},
        thinking={"type": "adaptive"},
        temperature=1,
        stream=True,
        speed="fast",
        inference_geo="us",
        betas=["fast-mode-2026-02-01"],
    )

    measured = bound(req, assumed_max_output_tokens=None)

    # `betas` travels as a header, not in the body, so it is not measured.
    body = {k: v for k, v in req.items() if k != "betas"}
    assert measured.input_upper_bound == size(body)
    assert measured.modifiers == {"speed": "fast", "inference_geo": "us"}


def test_no_max_tokens_is_refused_unless_the_policy_assumes_one() -> None:
    """§4.4: never silently assume a small output."""
    req = request()
    del req["max_tokens"]

    with pytest.raises(PricingUnknown, match="max_tokens"):
        bound(req, assumed_max_output_tokens=None)

    assumed = bound(req, assumed_max_output_tokens=4096)
    assert assumed.max_output_tokens == 4096
    assert assumed.assumed_output


@pytest.mark.parametrize("value", [True, -1, 1.5, "1024"])
def test_a_max_tokens_that_is_not_a_count_is_refused(value: object) -> None:
    with pytest.raises(PricingUnknown, match="max_tokens"):
        measure(request(max_tokens=value))


@pytest.mark.parametrize("req", [{}, {"model": ""}, {"model": 5}, "claude-sonnet-5"])
def test_a_request_naming_no_model_is_a_wiring_error(req: object) -> None:
    with pytest.raises(ConfigError, match="no model"):
        model_of(req)  # type: ignore[arg-type]


def test_a_request_that_is_not_plain_json_is_a_wiring_error() -> None:
    with pytest.raises(ConfigError, match="plain JSON"):
        measure(request(metadata={"user_id": object()}))


def test_nothing_from_the_request_is_echoed_into_a_refusal() -> None:
    """Locked decision #5. The request is payload; the error names positions."""
    image = {"type": "image", "source": {"type": "url", "url": f"https://x/{SENTINEL}"}}
    cases = [
        request(messages=user({"type": "text", "text": SENTINEL}, image)),
        request(**{f"{SENTINEL} key": 1}),
        request(messages=user({"type": SENTINEL})),
    ]
    for req in cases:
        with pytest.raises(PricingUnknown) as caught:
            measure(req)
        assert SENTINEL not in str(caught.value)


def test_the_usage_of_a_real_response_maps_onto_the_tables_classes() -> None:
    """The shape `response.usage.model_dump()` returns, with the fields that are
    not token counts left out rather than read as counts."""
    usage = {
        "input_tokens": 512,
        "output_tokens": 300,
        "cache_read_input_tokens": 8192,
        "cache_creation_input_tokens": 2048,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 2000,
            "ephemeral_1h_input_tokens": 48,
        },
        "server_tool_use": None,
        "service_tier": "standard",
        "inference_geo": "us",
    }

    assert usage_classes(usage) == {
        "input": 512,
        "output": 300,
        "cache_read": 8192,
        "cache_write_5m": 2000,
        "cache_write_1h": 48,
    }


def attempt(**counts: int) -> dict[str, object]:
    return {"type": "message", **counts}


def recorded(**extra: object) -> dict[str, object]:
    """The usage Claude Code recorded from the API on 2026-09-26: every field it
    carried, counts invented."""
    counts = {
        "input_tokens": 6,
        "output_tokens": 420,
        "cache_read_input_tokens": 90_000,
        "cache_creation_input_tokens": 1_500,
    }
    return {
        **counts,
        "output_tokens_details": {"thinking_tokens": 300},
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        "service_tier": "standard",
        "cache_creation": {
            "ephemeral_1h_input_tokens": 1_500,
            "ephemeral_5m_input_tokens": 0,
        },
        "inference_geo": "not_available",
        "iterations": [attempt(**counts)],
        "speed": "standard",
        **extra,
    }


def test_a_response_as_the_api_now_sends_it_is_priced_without_stopping_the_table() -> (
    None
):
    """Before this, `iterations` and `output_tokens_details` passed through as
    unknown classes, so the first real response stopped every call after it."""
    table = _PriceTable(_ALL_LISTINGS)
    rates = table.resolve("claude-opus-5-5", {})

    charge = table.actual(rates, usage_classes(recorded()))

    assert usage_classes(recorded()) == {
        "input": 6,
        "output": 420,
        "cache_read": 90_000,
        "cache_write_1h": 1_500,
        "cache_write_5m": 0,
    }
    assert not charge.price_table_stale
    table.resolve("claude-opus-5-5", {})


def test_a_count_one_side_leaves_out_is_read_as_zero() -> None:
    """A zero the attempt omits is no difference in the bill (/code-review, D52)."""
    usage = recorded(
        cache_creation_input_tokens=0,
        cache_creation=None,
        iterations=[
            attempt(input_tokens=6, output_tokens=420, cache_read_input_tokens=90_000)
        ],
    )

    assert "iterations" not in usage_classes(usage)


@pytest.mark.parametrize(
    "iterations",
    [
        # A fallback ran: the refused attempt is billed, and the top level omits it.
        [attempt(input_tokens=6, output_tokens=0), attempt(input_tokens=6)],
        [{**attempt(input_tokens=6, output_tokens=420), "type": "fallback_message"}],
        # One attempt whose counts are not the top level's.
        [attempt(input_tokens=7, output_tokens=420)],
        ["not an attempt"],
        "not a list",
    ],
)
def test_iterations_that_record_more_than_the_top_level_reach_the_table(
    iterations: object,
) -> None:
    classes = usage_classes(recorded(iterations=iterations))

    assert "iterations" in classes


def test_cache_writes_of_unknown_lifetime_are_charged_at_the_dearer_one() -> None:
    usage = {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 900}

    assert usage_classes(usage)["cache_write_1h"] == 900


def test_a_server_tool_that_was_used_reaches_the_table_as_unknown() -> None:
    """Refused at reserve time, so if one billed anyway the table must see it and
    stop admitting calls it can no longer bound."""
    usage = {
        "input_tokens": 1,
        "output_tokens": 1,
        "server_tool_use": {"web_search_requests": 2},
    }

    assert usage_classes(usage)["server_tool_use"] == 2
    assert "server_tool_use" not in usage_classes(
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "server_tool_use": {"web_search_requests": 0},
        }
    )


def test_a_field_this_adapter_does_not_know_is_passed_through() -> None:
    """So the price table sees a class it does not know (§4.8.3)."""
    classes = usage_classes(
        {"input_tokens": 1, "output_tokens": 1, "audio_tokens": 40, "novel": {"x": 1}}
    )

    assert classes["audio_tokens"] == 40
    assert classes["novel"] == 0


def test_usage_that_is_not_a_mapping_is_a_wiring_error() -> None:
    with pytest.raises(ConfigError, match="model_dump"):
        usage_classes("512 tokens")  # type: ignore[arg-type]


def test_a_cache_write_of_a_lifetime_nobody_knows_reaches_the_table() -> None:
    """Found by /code-review: the breakdown's unknown keys were dropped, so a new
    lifetime was charged nothing and the table never noticed."""
    usage = {
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_creation_input_tokens": 50_000,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_24h_input_tokens": 50_000,
        },
    }

    classes = usage_classes(usage)

    assert classes["ephemeral_24h_input_tokens"] == 50_000
    assert classes["cache_write_5m"] == 0


def test_cache_writes_the_breakdown_does_not_account_for_are_charged() -> None:
    usage = {
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_creation_input_tokens": 900,
        "cache_creation": {"ephemeral_5m_input_tokens": 100},
    }

    assert usage_classes(usage)["cache_write_1h"] == 800


@pytest.mark.parametrize(
    "extra",
    [
        {"betas": ["context-1m-2025-08-07"]},
        {"betas": "fast-mode-2026-02-01"},
        {"extra_headers": {"anthropic-beta": "anything"}},
    ],
)
def test_a_beta_that_could_change_the_price_is_refused(
    extra: dict[str, object],
) -> None:
    with pytest.raises(PricingUnknown):
        measure(request(**extra))


def test_a_timeout_object_is_not_measured_as_part_of_the_request() -> None:
    """The SDK keeps `timeout` for itself; it is often not JSON at all."""
    assert measure(request(timeout=object())) == size(request())


def test_a_block_type_that_is_not_a_string_is_refused_not_crashed_on() -> None:
    with pytest.raises(PricingUnknown):
        measure(request(messages=user({"type": ["text"]})))
