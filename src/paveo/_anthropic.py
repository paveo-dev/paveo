"""The Anthropic adapter: a Messages request in, a bound out (§4.5, D38, D39).

**Provider-specific on purpose, and the only file that is.** The decision point
speaks a neutral language (a model, the most input and output a call can use,
and the per-class counts it did use), so a second provider is a second adapter
beside this one, not a change to the core. OpenAI and Gemini hide different
tokens in different places and report usage in different words; each needs its
own reader, written against its own provider's documentation, and a price list
of its own in ``prices.py``. None is guessed from this one.

What follows is how an Anthropic request is bounded.

A reservation is only a ceiling if its input count really is an upper bound, and
§4.5's bound, the request's own bytes, holds only for tokens that are *in* the
bytes. No BPE token is under one byte, so text is safe. Three things are not:

- **What the provider adds.** A system prompt whenever ``tools`` is present, and
  the definitions of Anthropic-defined tools, which a request names by ``type``
  in a few bytes and is billed for in thousands of tokens.
- **Images.** An image costs by its pixels, up to a per-image cap, not by its
  bytes: a tiny, highly compressed image can cost far more tokens than it has
  bytes, and an agent reading the web is handed images by strangers.
- **Anything referred to rather than sent.** An image by URL or a document by
  ``file_id`` costs its content; its bytes are a reference.

The first two are added at the largest figure the provider documents. The third,
and everything else this module cannot put a number on, is **refused**
(``PricingUnknown``) rather than guessed. Every allowance below cites the page it
came from, read on the date in ``prices.VERIFIED["anthropic"]``.

**Nothing here is stored, logged or echoed.** The request is payload: it is
walked for its shape and measured for its length, and every error names a
position in it (``messages[3].content[0]``) and never a value from it.

**Allowlists, not denylists, at every level.** A top-level parameter, a content
block type or a source type this module does not know is refused, because the
day the provider ships a new one that bills hidden tokens is the day a denylist
starts under-reserving in silence (§10.10).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn

from .errors import ConfigError, PricingUnknown
from .prices import _printable

# The tool-use system prompt the API adds whenever `tools` is present: the largest
# figure in the pricing page's per-model table (Opus 4.7 with `tool_choice` any or
# tool). One number for every model, so the bound never depends on the table
# being keyed right.
_TOOL_SYSTEM_PROMPT = 804

# Anthropic-defined tools, by `type` prefix: the input their definitions add, from
# the pricing page. Where the page says "about", the figure is rounded up past its
# largest per-model value, because "about" is not a bound.
_DEFINED_TOOLS: Mapping[str, int] = {
    "bash_": 325,
    "text_editor_": 700,
    # "about 4,500 ... about 4,590 on Claude Sonnet 5", default members.
    "computer_toolset_": 5_000,
    # Earlier versions: 466-499 system prompt tokens plus about 735 per definition.
    "computer_2": 1_500,
    # "about 6,600 ... about 6,670 on Claude Sonnet 5", plus about 880 when every
    # optional member is enabled.
    "browser_toolset_": 8_000,
}

# The most visual tokens one image can cost on any model the table carries: the
# high-resolution tier, Claude 4.7 and later (vision page). Added for every image
# on top of its bytes, since a small file can decode to a large image.
_IMAGE_TOKENS = 4_784

# Parameters that change nothing about the input or are bounded by `max_tokens`.
# `thinking` counts against `max_tokens`. Everything not named here or in the
# handled set below is refused, with a trigger in D39.
_PLAIN_PARAMETERS = frozenset(
    {
        "model",
        "max_tokens",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "metadata",
        "stream",
        "thinking",
        "speed",
        "inference_geo",
        "cache_control",
        "service_tier",
        "output_config",
        # SDK keyword arguments that shape the HTTP request, not the prompt. Only
        # these: `extra_headers` is refused because a header can carry a beta
        # that changes billing, and `betas` is checked value by value below.
        "betas",
        "timeout",
    }
)

# Beta flags whose only effect on price is a parameter this module already reads
# and prices. Any other beta can switch on a feature priced where we cannot see
# it (/code-review, D40), so it is refused until its cost is verified.
_KNOWN_BETAS = frozenset({"fast-mode-2026-02-01"})

# Keyword arguments the SDK sends as headers or keeps for itself: not in the body,
# so not measured, and one of them (`timeout`) is often not JSON at all.
_NOT_IN_BODY = frozenset({"betas", "timeout"})

# Inside `output_config`: effort and a task budget change how many output tokens
# are spent, which `max_tokens` already caps. `format` is structured output, whose
# input cost we have not verified, so it is refused until we have.
_OUTPUT_CONFIG_KEYS = frozenset({"effort", "task_budget"})

_BLOCK_TYPES = frozenset(
    {
        "text",
        "image",
        "document",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
    }
)

_PRICE_AFFECTING = ("speed", "inference_geo")


@dataclass(frozen=True, slots=True)
class RequestBound:
    """What a request can cost, in the terms the price table prices.

    ``assumed_output`` is True when ``max_output_tokens`` came from the policy's
    ``assumed_max_output_tokens`` rather than the request, which §4.4 says must be
    visible in the audit record, because the ceiling is then approximate.
    """

    model: str
    modifiers: Mapping[str, object]
    input_upper_bound: int
    max_output_tokens: int
    assumed_output: bool


def model_of(request: Mapping[str, object]) -> str:
    """The model a request names, or ``ConfigError``.

    Read on its own, first, because the policy is asked about the model before
    anything else looks at the request (§5.1): a model the policy never names is
    refused before its name can reach a pricing error, or, in shadow mode, unless
    the price table carries it (D48).
    """
    model = request.get("model") if isinstance(request, Mapping) else None
    if not isinstance(model, str) or not model:
        raise ConfigError(
            "the request names no model.",
            remedy=(
                "pass the same mapping you pass to messages.create, including `model`."
            ),
        )
    return model


def bound(
    request: Mapping[str, object], *, assumed_max_output_tokens: int | None
) -> RequestBound:
    """Measure a request, or refuse it if it cannot be measured."""
    model = model_of(request)
    refuse = _Refusal(model)

    for key in request:
        if key not in _PLAIN_PARAMETERS:
            refuse(
                f"the request sets {_printable(key)!r}, a parameter paveo cannot "
                f"bound the cost of",
                "leave it out, or tell us you need it: each one is added once its "
                "cost is verified (D39).",
            )

    output_config = request.get("output_config", {})
    if not isinstance(output_config, Mapping) or not set(output_config) <= (
        _OUTPUT_CONFIG_KEYS
    ):
        refuse(
            "output_config sets something other than effort or task_budget, and "
            "structured output's input cost is not yet verified",
            "leave output_config.format out for now (D39).",
        )

    betas = request.get("betas", [])
    if (
        not isinstance(betas, Sequence)
        or isinstance(betas, str | bytes)
        or not all(isinstance(b, str) and b in _KNOWN_BETAS for b in betas)
    ):
        refuse(
            "the request enables a beta whose effect on price paveo has not verified",
            "leave that beta out for now; each is added once its cost is known (D40).",
        )

    max_tokens = request.get("max_tokens")
    assumed = False
    if max_tokens is None:
        if assumed_max_output_tokens is None:
            refuse(
                "the request sets no max_tokens, so its output has no bound",
                "set max_tokens on the request, or set "
                "defaults.assumed_max_output_tokens in the policy (SPEC_V1.md §4.4).",
            )
        max_tokens = assumed_max_output_tokens
        assumed = True
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 0
    ):
        refuse(
            "max_tokens is not a non-negative whole number",
            "set max_tokens to the most output tokens this call may produce.",
        )

    allowance = _tools_allowance(request.get("tools"), refuse)
    allowance += _content_allowance(request, refuse)

    return RequestBound(
        model=model,
        modifiers={name: request[name] for name in _PRICE_AFFECTING if name in request},
        input_upper_bound=_byte_length(
            {k: v for k, v in request.items() if k not in _NOT_IN_BODY}
        )
        + allowance,
        max_output_tokens=max_tokens,
        assumed_output=assumed,
    )


class _Refusal:
    """Raises ``PricingUnknown`` for one model. A callable so the walkers stay flat."""

    def __init__(self, model: str) -> None:
        self._model = model

    def __call__(self, detail: str, remedy: str) -> NoReturn:
        raise PricingUnknown(model=self._model, detail=detail, remedy=remedy)


def _tools_allowance(tools: object, refuse: _Refusal) -> int:
    if tools is None:
        return 0
    if not isinstance(tools, Sequence) or isinstance(tools, str | bytes):
        refuse("tools is not a list", "pass tools as a list of tool definitions.")
    if not tools:
        return 0
    total = _TOOL_SYSTEM_PROMPT
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            refuse(f"tools[{index}] is not a mapping", "pass each tool as a dict.")
        kind = tool.get("type")
        if kind is None or kind == "custom":
            continue  # a tool you define: its definition is in the bytes
        allowance = next(
            (
                tokens
                for prefix, tokens in _DEFINED_TOOLS.items()
                if isinstance(kind, str) and kind.startswith(prefix)
            ),
            None,
        )
        if allowance is None:
            refuse(
                f"tools[{index}] is a server or Anthropic-defined tool whose cost "
                f"paveo cannot bound: its results, searches or execution time are "
                f"billed beyond the request",
                "leave it out of calls checked by paveo for now; each is added "
                "once its cost can be bounded (D39).",
            )
        total += allowance
    return total


def _content_allowance(request: Mapping[str, object], refuse: _Refusal) -> int:
    """Walk ``system`` and ``messages`` for images and anything referred to."""
    total = 0
    system = request.get("system")
    if system is not None and not isinstance(system, str):
        total += _blocks(system, "system", refuse)
    messages = request.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes):
        refuse("messages is not a list", "pass messages as a list of turns.")
    for index, message in enumerate(messages):
        where = f"messages[{index}]"
        if not isinstance(message, Mapping):
            refuse(f"{where} is not a mapping", "pass each turn as a dict.")
        content = message.get("content")
        if not isinstance(content, str):
            total += _blocks(content, f"{where}.content", refuse)
    return total


def _blocks(blocks: object, where: str, refuse: _Refusal) -> int:
    if not isinstance(blocks, Sequence) or isinstance(blocks, str | bytes):
        refuse(f"{where} is neither text nor a list of blocks", _SHAPE_REMEDY)
    total = 0
    for index, block in enumerate(blocks):
        here = f"{where}[{index}]"
        kind = block.get("type") if isinstance(block, Mapping) else None
        if not isinstance(kind, str) or kind not in _BLOCK_TYPES:
            refuse(
                f"{here} is a content block of a type paveo cannot bound",
                "send text, images and documents inline; other block types are "
                "added once their cost can be bounded (D39).",
            )
        kind = block["type"]
        if kind == "image":
            _require_inline(block, here, refuse, allowed=("base64",))
            total += _IMAGE_TOKENS
        elif kind == "document":
            source = _require_inline(block, here, refuse, allowed=("text", "content"))
            if source.get("type") == "content":
                inner = source.get("content")
                total += _blocks(inner, f"{here}.source.content", refuse)
        elif kind == "tool_result":
            content = block.get("content")
            if content is not None and not isinstance(content, str):
                total += _blocks(content, f"{here}.content", refuse)
    return total


def _require_inline(
    block: Mapping[str, object],
    where: str,
    refuse: _Refusal,
    *,
    allowed: tuple[str, ...],
) -> Mapping[str, object]:
    source = block.get("source")
    if not isinstance(source, Mapping) or source.get("type") not in allowed:
        refuse(
            f"{where} is sent by reference or in a form whose tokens are not "
            f"bounded by its bytes (a URL, a file_id, or a PDF)",
            "send images as base64 and documents as plain text; content by "
            "reference and PDFs are added once their cost can be bounded "
            "(D39).",
        )
    return source


_SHAPE_REMEDY = "pass the request in the shape messages.create takes."


def _byte_length(request: Mapping[str, object]) -> int:
    """The request's length in UTF-8 bytes, which bounds its text tokens (§4.5).

    Serialised compactly and without ASCII escaping, so the count is the text's
    own length plus its structure: never less than the tokens the text can be
    split into. The serialisation is measured and discarded.
    """
    try:
        text = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        raise ConfigError(
            "the request holds a value that is not plain JSON, so it cannot be "
            "measured.",
            remedy=(
                "pass plain dicts, lists, strings and numbers, as you would send "
                "over HTTP, rather than SDK objects."
            ),
        ) from e
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError as e:  # a lone surrogate: not text any provider takes
        raise ConfigError(
            "the request holds a string that is not valid Unicode.",
            remedy="pass text that can be encoded as UTF-8.",
        ) from e


# Fields of a response's `usage` that are not token counts. They describe the call
# rather than bill it, and passing them to the price table as classes would read
# a string as a count, or mark the table stale over a field that costs nothing.
_INFORMATIONAL = frozenset({"service_tier", "inference_geo", "speed"})

_USAGE_CLASSES: Mapping[str, str] = {
    "input_tokens": "input",
    "cache_read_input_tokens": "cache_read",
    "output_tokens": "output",
}


def usage_classes(usage: Mapping[str, object]) -> dict[str, object]:
    """A response's ``usage``, in the price table's class names (§4.2).

    Pass ``response.usage.model_dump()``, or the dict the HTTP API returned.
    ``None`` means a field the SDK left empty, and is skipped.

    **Cache writes are split by lifetime when the response says so**
    (``cache_creation``), and charged at the dearer 1-hour rate when it gives only
    the total: we do not know which it was, so we take the one that cannot
    under-charge.

    **A field this adapter does not know is passed through under its own name**,
    so the table sees a class it does not know, charges it at its highest rate,
    and refuses every later reserve (§4.8.3). That includes any server tool
    reporting a request: every such tool was refused at reserve time, so one here
    means something billed that paveo did not bound. A new field that costs
    nothing stops calls too, which is §10.12's published trade, and the reason
    this list of known fields is re-read with the price page.
    """
    if not isinstance(usage, Mapping):
        raise ConfigError(
            "usage is not a mapping.",
            remedy="pass response.usage.model_dump(), or the usage dict itself.",
        )
    classes: dict[str, object] = {}
    breakdown = usage.get("cache_creation")
    for key, value in usage.items():
        if value is None or key in _INFORMATIONAL:
            continue
        # How many of the billed output tokens were thinking: a breakdown of
        # output_tokens, not a charge beside it (release notes, 27 May 2026).
        if key == "output_tokens_details" and isinstance(value, Mapping):
            continue
        if key == "iterations" and _one_attempt(value, usage):
            continue
        if key in _USAGE_CLASSES:
            classes[_USAGE_CLASSES[key]] = value
        elif key == "cache_creation_input_tokens":
            if not isinstance(breakdown, Mapping):
                classes["cache_write_1h"] = value
        elif key == "cache_creation" and isinstance(value, Mapping):
            classes.update(
                _cache_writes(value, usage.get("cache_creation_input_tokens"))
            )
        elif key == "server_tool_use" and isinstance(value, Mapping):
            used = sum(v for v in value.values() if isinstance(v, int) and v > 0)
            if used:
                classes["server_tool_use"] = used
        else:
            classes[key] = value if isinstance(value, int) else 0
    return classes


_ATTEMPT_COUNTS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _one_attempt(iterations: object, usage: Mapping[str, object]) -> bool:
    """Whether ``usage.iterations`` records only the attempt the top level counts.

    It is "the per-attempt record of what you're billed", and the top-level
    counts "describe only the attempt that produced the returned message", so
    after a fallback the top level under-counts the bill, at another model's rate
    (refusals-and-fallback page, read 2026-09-26). One plain attempt whose counts
    are the top level's adds nothing. Anything else passes through, so the table
    sees a class it does not know and stops (§4.8.3). The door refuses
    ``fallbacks`` already; this is what holds if a second attempt appears anyway.
    """
    if not isinstance(iterations, list) or len(iterations) > 1:
        return False
    if not iterations:
        return True
    attempt = iterations[0]
    return (
        isinstance(attempt, Mapping)
        and attempt.get("type") == "message"
        # A count one side leaves out is zero on it, as the API reports zeros.
        and all(
            (attempt.get(key) or 0) == (usage.get(key) or 0) for key in _ATTEMPT_COUNTS
        )
    )


_LIFETIMES: Mapping[str, str] = {
    "ephemeral_5m_input_tokens": "cache_write_5m",
    "ephemeral_1h_input_tokens": "cache_write_1h",
}


def _cache_writes(breakdown: Mapping[str, object], total: object) -> dict[str, object]:
    """Cache writes split by lifetime, with nothing the total reports left out.

    A lifetime this adapter does not know passes through under its own name, so
    the table sees an unknown class and stops (§4.8.3). And if the known
    lifetimes add up to less than ``cache_creation_input_tokens``, the difference
    is charged at the dearer 1-hour rate: a write the breakdown did not account
    for is still a write (/code-review, D40).
    """
    classes: dict[str, object] = {}
    for key, value in breakdown.items():
        if value is None:
            continue
        name = _LIFETIMES.get(key, key)
        classes[name] = value if isinstance(value, int) else 0
    known = sum(
        v for k, v in classes.items() if k in _LIFETIMES.values() and isinstance(v, int)
    )
    if isinstance(total, int) and not isinstance(total, bool) and total > known:
        missing = total - known
        existing = classes.get("cache_write_1h", 0)
        classes["cache_write_1h"] = (
            existing if isinstance(existing, int) else 0
        ) + missing
    return classes


def served(usage: Mapping[str, object]) -> dict[str, str]:
    """Where the response says the call ran, so an unset geo settles at the truth.

    ``usage.inference_geo`` reports it (data residency page). An unset geo is
    reserved at the dearest, which is safe, and this lets the charge be exact.
    """
    geo = usage.get("inference_geo") if isinstance(usage, Mapping) else None
    return {"inference_geo": geo} if geo in {"us", "global"} else {}
