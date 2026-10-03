"""The OpenAI adapter: a Chat Completions request in, a bound out (D39, D41).

Beside ``_anthropic.py`` and built the same way: allowlists at every level, the
request's own bytes as the bound on its text, what the provider adds on its side
added on top, and anything this module cannot put a number on refused. It is
also the shape LiteLLM speaks for every provider it routes to, so a model whose
price the policy declares is read here too when its request comes in this shape.

What the provider's pages said, read on 2026-09-24 (D41), and what this does
about each:

- **`n` answers each get their own cap**, and all are billed, so the output bound
  is the cap times ``n``.
- **`max_completion_tokens` caps reasoning and visible output together**, which
  is what makes it an output bound for reasoning models. ``max_tokens`` is
  deprecated and read as a cap too; if both are set the larger bounds.
- **An unset or ``auto`` `service_tier` is whatever the project is set to**, and
  a project can be set to Fast. The table prices unset and ``auto`` at the dearest
  tier the model has; set ``service_tier: "default"`` for the exact rate.
- **Images, audio, files and predicted outputs** are refused: each bills tokens
  the request's bytes do not measure, and none is verified yet.
- **Tools**: only function tools, whose definitions are in the bytes. OpenAI
  renders them into a format of its own before the model reads them, and does
  not publish what that adds, so a fixed allowance is added whenever tools are
  present. It is chosen, not documented (D41), and generous: at GPT-6 Sol prices
  it is a fraction of a cent.

Like the other readers, nothing here is stored, logged or echoed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._anthropic import RequestBound, _byte_length, _Refusal, model_of
from .errors import ConfigError
from .prices import _printable

_PARAMETERS = frozenset(
    {
        "model",
        "messages",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "temperature",
        "top_p",
        "stop",
        "stream",
        "stream_options",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "user",
        "safety_identifier",
        "metadata",
        "store",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "reasoning_effort",
        "verbosity",
        "service_tier",
        "prompt_cache_key",
        "prompt_cache_retention",
        # The SDK's own keyword argument, not part of the body.
        "timeout",
    }
)
_NOT_IN_BODY = frozenset({"timeout"})

# The keys a message may carry. Anything else, `audio` above all (a replayed audio
# turn is billed as input the bytes do not hold), is refused (/code-review, D41).
_MESSAGE_KEYS = frozenset(
    {"role", "content", "name", "tool_calls", "tool_call_id", "refusal"}
)

# Top-level fields of `usage` this adapter reads. Any other one passes through
# under its own name, so a billable field OpenAI or LiteLLM adds is seen by the
# table and stops it, rather than being dropped (§4.8.3, /code-review, D41).
_USAGE_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    }
)

# Content parts that are text. `refusal` is the text of an assistant's refusal.
_TEXT_PARTS = frozenset({"text", "refusal"})

# Not from any page: OpenAI does not publish what rendering function definitions
# adds. Chosen to be generous, and recorded as chosen (D41).
_TOOLS_ALLOWANCE = 1_000

# Fields of `usage` that describe tokens already counted elsewhere, or nothing
# billable: reasoning and prediction tokens are inside `completion_tokens`, text
# tokens inside the totals they describe.
_INFORMATIONAL = frozenset(
    {"total_tokens", "reasoning_tokens", "text_tokens", "accepted_prediction_tokens"}
)


def bound(
    request: Mapping[str, object], *, assumed_max_output_tokens: int | None
) -> RequestBound:
    """Measure a Chat Completions request, or refuse it if it cannot be measured."""
    model = model_of(request)
    refuse = _Refusal(model)

    unknown = [key for key in request if key not in _PARAMETERS]
    if unknown:
        refuse(
            f"the request sets {_printable(unknown[0])!r}, a parameter paveo "
            f"cannot bound the cost of",
            "leave it out, or tell us you need it: each is added once its cost is "
            "verified (D41).",
        )

    tools = request.get("tools")
    if tools is not None and not _function_tools(tools):
        refuse(
            "the request carries a tool that is not a function you define",
            "use function tools only on calls checked by paveo (D41).",
        )
    allowance = _TOOLS_ALLOWANCE if tools else 0

    messages = request.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes):
        refuse("messages is not a list", "pass messages as a list of turns.")
    for index, message in enumerate(messages):
        where = _not_text_message(message)
        if where is not None:
            refuse(
                f"messages[{index}]{where} is not text: images, audio and files "
                f"cost tokens their bytes do not measure",
                "send text only on calls checked by paveo for now (D41).",
            )

    caps = [
        request[name]
        for name in ("max_completion_tokens", "max_tokens")
        if name in request
    ]
    assumed = False
    if not caps:
        if assumed_max_output_tokens is None:
            refuse(
                "the request sets no max_completion_tokens, so its output has no bound",
                "set max_completion_tokens, which caps reasoning and output together, "
                "or set defaults.assumed_max_output_tokens in the policy (§4.4).",
            )
        caps = [assumed_max_output_tokens]
        assumed = True
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in caps):
        refuse(
            "the output cap is not a non-negative whole number",
            "set max_completion_tokens to the most tokens this call may produce.",
        )
    answers = request.get("n", 1)
    if isinstance(answers, bool) or not isinstance(answers, int) or answers < 1:
        refuse(
            "n is not a positive whole number",
            "set n to how many answers you want, or leave it unset for one.",
        )

    return RequestBound(
        model=model,
        modifiers={"service_tier": request["service_tier"]}
        if "service_tier" in request
        else {},
        input_upper_bound=_byte_length(
            {k: v for k, v in request.items() if k not in _NOT_IN_BODY}
        )
        + allowance,
        max_output_tokens=answers * max(c for c in caps if isinstance(c, int)),
        assumed_output=assumed,
    )


def usage_classes(usage: Mapping[str, object]) -> dict[str, object]:
    """A Chat Completions ``usage``, in the price table's class names (§4.2).

    Pass ``response.usage.model_dump()``. ``prompt_tokens`` includes the cached
    tokens and, on GPT-5.6 and later, the cache writes: each token is billed at
    exactly one rate, so they are taken out of ``input`` rather than added to it.
    ``completion_tokens`` already includes reasoning and rejected prediction
    tokens, which are billed as output.

    Image or audio tokens in a response were refused at reserve time, so a
    non-zero count passes through under its own name: the table sees a class it
    does not know, charges it at its highest rate and stops (§4.8.3).
    """
    if isinstance(usage, Mapping) and isinstance(usage.get("usage"), Mapping):
        usage = usage["usage"]  # type: ignore[assignment]  # checked on the line above
    if not isinstance(usage, Mapping):
        raise ConfigError(
            "usage is not a mapping.",
            remedy="pass response.usage.model_dump(), or the usage dict itself.",
        )
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    if not (
        _count(prompt)
        and _count(completion)
        and isinstance(prompt_details, Mapping)
        and isinstance(completion_details, Mapping)
    ):
        raise ConfigError(
            "usage does not carry whole prompt and completion token counts.",
            remedy="pass response.usage.model_dump() from a Chat Completions response.",
        )
    assert isinstance(prompt, int)  # noqa: S101 - _count checked it
    cached = prompt_details.get("cached_tokens") or 0
    written = prompt_details.get("cache_write_tokens") or 0
    if not (_count(cached) and _count(written)) or cached + written > prompt:
        raise ConfigError(
            "usage reports more cached and cache-written tokens than prompt tokens.",
            remedy="pass the usage object exactly as the response returned it.",
        )
    classes: dict[str, object] = {
        "input": prompt - cached - written,
        "cache_read": cached,
        "cache_write": written,
        "output": completion,
    }
    for key, value in usage.items():
        if key not in _USAGE_FIELDS:
            classes[_printable(key)] = value if _count(value) else 0
    for details, side in (
        (prompt_details, "prompt"),
        (completion_details, "completion"),
    ):
        for key, value in details.items():
            if key in {"cached_tokens", "cache_write_tokens"} or key in _INFORMATIONAL:
                continue
            if key == "rejected_prediction_tokens":
                continue  # inside completion_tokens, billed as output
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                classes[f"{side}_{key}"] = value
    return classes


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _function_tools(tools: object) -> bool:
    if not isinstance(tools, Sequence) or isinstance(tools, str | bytes):
        return False
    return all(
        isinstance(tool, Mapping)
        and tool.get("type") == "function"
        and set(tool) <= {"type", "function"}
        for tool in tools
    )


def _not_text_message(message: object) -> str | None:
    """Where in one message there is something that is not text, or ``None``.

    The message's own keys are checked before its content, so a turn that is
    not text by any other route (`audio` beside a string or empty content) is
    refused too (/code-review and /security-review, D41).
    """
    if not isinstance(message, Mapping) or not set(message) <= _MESSAGE_KEYS:
        return ""
    calls = message.get("tool_calls")
    if calls is not None and not _function_calls(calls):
        return ".tool_calls"
    content = message.get("content")
    if content is None or isinstance(content, str):
        return None
    if not isinstance(content, Sequence) or isinstance(content, bytes):
        return ".content"
    for index, part in enumerate(content):
        kind = part.get("type") if isinstance(part, Mapping) else None
        if not isinstance(kind, str) or kind not in _TEXT_PARTS:
            return f".content[{index}]"
    return None


def _function_calls(calls: object) -> bool:
    if not isinstance(calls, Sequence) or isinstance(calls, str | bytes):
        return False
    return all(
        isinstance(call, Mapping)
        and call.get("type") == "function"
        and set(call) <= {"id", "type", "function"}
        for call in calls
    )


def served(usage: Mapping[str, object]) -> dict[str, str]:
    """The tier the response says it was served at, when given the response.

    Only ``response.model_dump()`` carries it; ``usage`` alone does not, and the
    call then settles at the tier it was reserved at.
    """
    tier = usage.get("service_tier") if isinstance(usage, Mapping) else None
    return {"service_tier": tier} if isinstance(tier, str) else {}
