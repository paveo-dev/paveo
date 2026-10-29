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

import hashlib
import re
from collections.abc import Mapping, Sequence

from ._anthropic import RequestBound, _byte_length, _Refusal, model_of
from .errors import ConfigError
from .prices import _PRINTABLE_CLASS, _printable

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
# Fields of `prompt_tokens_details` that `usage_classes` reads itself.
_READ_DETAILS = frozenset({"cached_tokens", "cache_write_tokens"})
_BREAKPOINT = ".prompt_cache_breakpoint"


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
    if request.get("prompt_cache_retention", "24h") not in {"in_memory", "24h"}:
        refuse(
            "prompt_cache_retention is not one of the values OpenAI lists",
            "use in_memory or 24h, the values in OpenAI's SDK (D115).",
        )
    for index, message in enumerate(messages):
        where = _not_text_message(message)
        if where is not None and where.endswith(_BREAKPOINT):
            refuse(
                f"messages[{index}]{where} asks for a cache lifetime other than 30m, "
                f"which paveo does not price",
                "leave the lifetime unset, or set it to 30m (D115).",
            )
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

    **A missing write count is not a zero** (D115). ``cache_write_tokens`` is
    optional in OpenAI's own types, and a bridge that rebuilds the details can
    drop it, so absent or ``None`` (or no prompt details at all) sends the
    uncached prompt to ``input_unsplit``: an OpenAI row charges it at its
    cache-write rate, every other row as input. A present 0 is a real zero.

    **Nothing this adapter does not read is dropped** (D115). Each such field
    reaches the table as ``unpriced_<name>`` with the tokens it holds: a count
    as it is, a breakdown by its counts one level down, and anything else at
    the most it could describe. The table charges it at the row's highest rate
    and stops (§4.8.3). So the day OpenAI reports cache writes some new way,
    Paveo stops rather than under-charge.
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
    assert isinstance(completion, int)  # noqa: S101 - _count checked it
    cached = prompt_details.get("cached_tokens") or 0
    written = prompt_details.get("cache_write_tokens")
    if not (_count(cached) and (written is None or _count(written))) or (
        cached + (written or 0) > prompt
    ):
        raise ConfigError(
            "usage reports more cached and cache-written tokens than prompt tokens.",
            remedy="pass the usage object exactly as the response returned it.",
        )
    classes: dict[str, int] = (
        {"input": 0, "input_unsplit": prompt - cached}
        if written is None
        else {"input": prompt - cached - written, "cache_write": written}
    )
    classes.update({"cache_read": cached, "output": completion})
    for key, value in usage.items():
        if key in _USAGE_FIELDS or value is None:
            continue
        # A top-level field passes at any count, 0 included: a name the table
        # does not know is news at any count (§4.8.3, D38).
        tokens = _tokens_in(value)
        most = prompt - cached + completion
        _add(classes, _unpriced(key), most if tokens is None else tokens)
    for details, side, read, most in (
        (prompt_details, "prompt", _READ_DETAILS, prompt - cached),
        (completion_details, "completion", frozenset(), completion),
    ):
        for key, value in details.items():
            if key in read or key in _INFORMATIONAL:
                continue
            if key == "rejected_prediction_tokens":
                continue  # inside completion_tokens, billed as output
            tokens = _tokens_in(value)
            if tokens != 0:
                _add(
                    classes,
                    _unpriced(f"{side}_{key}"),
                    most if tokens is None else tokens,
                )
    return dict(classes)


def _tokens_in(value: object) -> int | None:
    """The tokens a field reports: a count, or the counts of a breakdown one level
    down; 0 for nothing (``None``, 0, an empty or all-zero breakdown); ``None``
    for anything else, which is never taken for nothing. One definition of
    "nothing", for every field (D115)."""
    if value is None or (isinstance(value, Mapping | list) and not value):
        return 0
    if _count(value):
        assert isinstance(value, int)  # noqa: S101 - _count checked it
        return value
    if isinstance(value, Mapping):
        counts = [
            None if isinstance(v, Mapping) else _tokens_in(v) for v in value.values()
        ]
        whole = [c for c in counts if c is not None]
        if len(whole) == len(counts):
            return sum(whole)
    return None


def _unpriced(key: object) -> str:
    """The class a field this adapter does not read reaches the table under: a
    name no priced class has, printable, and at most 64 characters, so the
    stale error and the audit record can name it. A name that does not fit is
    cut, with a hash of the whole so two keys never share it (D115)."""
    name = f"unpriced_{key}"
    if isinstance(key, str) and _PRINTABLE_CLASS.fullmatch(name):
        return name
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:8]
    kept = re.sub(r"[^a-z0-9_]", "_", str(key).lower())[:44]
    return f"unpriced_{kept}_{digest}"


def _add(classes: dict[str, int], name: str, tokens: int) -> None:
    """Add to a class, never overwrite: two fields reaching the same name both
    count (D115)."""
    classes[name] = classes.get(name, 0) + tokens


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
    return _not_text_part(content)


def _not_text_part(content: Sequence[object]) -> str | None:
    """Where in a message's content there is a part that is not plain text, or
    one asking for a cache lifetime Paveo does not price, or ``None``."""
    for index, part in enumerate(content):
        kind = part.get("type") if isinstance(part, Mapping) else None
        if not isinstance(kind, str) or kind not in _TEXT_PARTS:
            return f".content[{index}]"
        assert isinstance(part, Mapping)  # noqa: S101 - kind was read from it
        if not _thirty_minute_breakpoint(part):
            return f".content[{index}]{_BREAKPOINT}"
    return None


def _thirty_minute_breakpoint(part: Mapping[str, object]) -> bool:
    """A cache breakpoint on a part may ask only for the 30-minute lifetime,
    the one OpenAI's API supports today and the one Paveo prices (its SDK,
    28 Oct 2026, D115). A dearer lifetime is refused, not guessed at."""
    if "prompt_cache_breakpoint" not in part:
        return True
    breakpoint_ = part["prompt_cache_breakpoint"]
    return isinstance(breakpoint_, Mapping) and breakpoint_.get("ttl", "30m") == "30m"


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
