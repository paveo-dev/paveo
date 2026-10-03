"""The door for every provider without an adapter: a policy-priced model (D39).

Mistral, DeepSeek, Groq, xAI, a model you host: the operator writes the price
into the policy (§4.8.4) and paveo bounds the call from what it can see without
knowing the provider. **That is less than an adapter can see, and it says so.**

What holds for any provider: a text token is never under one byte, whatever the
tokenizer, so a request's UTF-8 bytes bound the tokens of the text it carries.

What does not, and is therefore refused: anything that is not text. An image, a
file, audio or video costs by content the bytes do not measure, and each provider
spells them differently, so this looks for every spelling it knows (OpenAI's
``image_url`` and ``input_image``, Gemini's ``inline_data``, Anthropic's
``image`` and ``document``) and refuses the request if it finds one. Where a
provider types its content parts, a part is admitted only as a type whose bytes
bound it, so a spelling missing from that list is refused too (D73).

What it cannot see at all, and publishes instead (§10.15): tokens a provider adds
on its own side, for formatting tools for example, and any request parameter
that changes that provider's price. The price is the operator's, and so is
knowing what their provider adds. **For a guarantee that does not rest on the
operator, use a provider with an adapter.**

Like the Anthropic adapter, nothing here is stored, logged or echoed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._anthropic import RequestBound, _byte_length, model_of
from .errors import ConfigError, PricingUnknown

# The only top-level parameters a policy-priced request may carry: the model, the
# conversation in each provider's spelling, output caps, choice counts, sampling
# and plain function tools. **Everything else is refused**, because a parameter
# this door does not know can switch on a server tool or pull in content by
# reference (`web_search_options`, `previous_response_id`, `cachedContent`, ...),
# billed beyond the bytes it can measure. Found by /security-review (D40).
_PARAMETERS = frozenset(
    {
        "model",
        "messages",
        "contents",
        "system",
        "system_instruction",
        "systemInstruction",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "generationConfig",
        "generation_config",
        "n",
        "best_of",
        "candidateCount",
        "candidate_count",
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "stream",
        "stream_options",
        "user",
        "metadata",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
    }
)

# A tool entry is admitted only if it is a function the caller defines: its
# definition is in the bytes. Any other tool type may run on the provider's side.
_FUNCTION_TOOL_KEYS = frozenset(
    {"type", "function", "name", "description", "parameters"}
)
_GEMINI_TOOL_KEYS = frozenset({"functionDeclarations", "function_declarations"})

# The ways the providers we know spell "this is not text", as keys and as `type`
# values. Found anywhere in the request except inside a JSON schema.
_NOT_TEXT = frozenset(
    {
        "image",
        "image_url",
        "input_image",
        "input_audio",
        "audio",
        "video",
        "file",
        "file_id",
        "file_data",
        "file_uri",
        "input_file",
        "inline_data",
        "inlineData",
        "fileData",
        "document",
        "document_url",
        "images",
    }
)

# The content parts whose every byte is in the request, and every key each may
# carry. A part in a message's `content` or `tool_calls`, or in `system`, of any
# other type or with any other key is refused: the word list above names only the
# spellings of "not text" someone thought of. Found by it: `cache_control` on a
# text part is a cache write, which the table prices at 1.25x or 2x the input
# rate (D73). `redacted_thinking` stays out: what the provider decrypts it into is
# not in the bytes. `index` is what a streamed tool call is rebuilt with.
_BOUNDED_PARTS = {
    "text": frozenset({"type", "text", "citations"}),
    "refusal": frozenset({"type", "refusal"}),
    "thinking": frozenset({"type", "thinking", "signature"}),
    "function": frozenset({"type", "id", "index", "function"}),
    "tool_use": frozenset({"type", "id", "name", "input"}),
    "tool_result": frozenset({"type", "tool_use_id", "content", "is_error"}),
}

# A tool's parameter schema describes arguments, and a past tool call carries the
# arguments the model chose: neither is content. A `read_file` tool with a `file`
# property sends no file, and nor does the call that used it. Only these exact
# places are skipped: "input" at the top of an OpenAI Responses request is the
# conversation itself and is scanned.
_SCHEMA_KEYS = frozenset(
    {
        "parameters",
        "input_schema",
        "properties",
        "schema",
        "functionCall",
        "function_call",
    }
)

# The names providers give the output cap. If a request sets more than one, the
# largest is the bound.
_OUTPUT_CAPS = (
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "maxOutputTokens",
)

# Gemini keeps its cap and its choice count one level down.
_GENERATION_CONFIG = ("generationConfig", "generation_config")

# How many answers a request asks for, under the names providers use. Each answer
# gets its own output cap, and OpenAI bills "the number of generated tokens across
# all of the choices" (Chat Completions, `n`), so the output bound is the cap times
# this. Gemini calls it `candidateCount`.
_CHOICES = ("n", "best_of", "candidateCount", "candidate_count")

# The classes a caller reports for a declared-price call: what the policy can
# price (DeclaredPrice).
_CLASSES = frozenset({"input", "cache_read", "output"})


def bound(
    request: Mapping[str, object], *, assumed_max_output_tokens: int | None
) -> RequestBound:
    """Measure a request of any provider's shape, or refuse what is not text."""
    model = model_of(request)
    unknown = [key for key in request if key not in _PARAMETERS]
    if unknown or not _plain_tools(request.get("tools")):
        raise PricingUnknown(
            model=model,
            detail=(
                "the request sets a parameter or a tool this door cannot bound: "
                "on a model priced in the policy, only text, output caps, "
                "sampling and function tools are admitted"
            ),
            remedy=(
                "leave server tools, stored prompts and references to earlier "
                "responses out of calls to this model, or use a provider paveo "
                "has an adapter for (D40)."
            ),
        )
    where = _not_text(request, "request")
    if where is not None:
        raise PricingUnknown(
            model=model,
            detail=(
                f"{where} is not text, and on a model priced in the policy paveo "
                f"cannot tell what non-text content costs"
            ),
            remedy=(
                "send text only on this model, or use a provider paveo has an "
                "adapter for (D39)."
            ),
        )
    where = _unknown_part(request)
    if where is not None:
        raise PricingUnknown(
            model=model,
            detail=(
                f"{where} is not a plain content part: on a model priced in the "
                f"policy, only text, thinking and tool parts with their usual "
                f"fields are admitted, and a cache setting is refused"
            ),
            remedy=(
                "send plain parts to this model, with no cache settings, or use a "
                "provider paveo has an adapter for (D73)."
            ),
        )

    settings: list[Mapping[str, object]] = [request]
    for name in _GENERATION_CONFIG:
        nested = request.get(name)
        if isinstance(nested, Mapping):
            settings.append(nested)
    caps = [s[name] for s in settings for name in _OUTPUT_CAPS if name in s]
    assumed = False
    if not caps:
        if assumed_max_output_tokens is None:
            raise PricingUnknown(
                model=model,
                detail="the request sets no output cap, so its output has no bound",
                remedy=(
                    f"set one of {', '.join(_OUTPUT_CAPS)} on the request, or set "
                    f"defaults.assumed_max_output_tokens in the policy (§4.4)."
                ),
            )
        caps = [assumed_max_output_tokens]
        assumed = True
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in caps):
        raise PricingUnknown(
            model=model,
            detail="the output cap is not a non-negative whole number",
            remedy="set it to the most output tokens this call may produce.",
        )

    choices = [s[name] for s in settings for name in _CHOICES if name in s]
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in choices):
        raise PricingUnknown(
            model=model,
            detail="the number of choices requested is not a positive whole number",
            remedy="set n to how many answers you want, or leave it unset for one.",
        )
    answers = max((c for c in choices if isinstance(c, int)), default=1)

    return RequestBound(
        model=model,
        modifiers={},
        input_upper_bound=_byte_length(request),
        max_output_tokens=answers * max(c for c in caps if isinstance(c, int)),
        assumed_output=assumed,
    )


def usage_classes(usage: Mapping[str, object]) -> dict[str, object]:
    """The counts a caller reports for a declared-price call, checked.

    The caller does the mapping here, because only they know their provider's
    words: ``{"input": ..., "output": ..., "cache_read": ...}``. A key outside
    those is refused as a wiring error rather than passed to the table, where an
    unknown class would stop every model in the process.
    """
    if not isinstance(usage, Mapping):
        raise ConfigError(
            "usage is not a mapping.",
            remedy='pass {"input": n, "output": m}, with "cache_read" if priced.',
        )
    unknown = set(usage) - _CLASSES
    if unknown:
        raise ConfigError(
            "usage for a model priced in the policy has keys other than input, "
            "output and cache_read.",
            remedy=(
                "report the provider's counts under those three names. Anything "
                "it bills that is none of them is not priced by the policy."
            ),
        )
    return dict(usage)


def _not_text(value: object, where: str) -> str | None:
    """Where the first non-text content is, or ``None``. Never what it is."""
    if isinstance(value, Mapping):
        kind = value.get("type")
        if isinstance(kind, str) and kind in _NOT_TEXT:
            return where
        for key, inner in value.items():
            if key in _NOT_TEXT:
                return f"{where}.<content>"
            if key in _SCHEMA_KEYS or (kind == "tool_use" and key == "input"):
                continue
            found = _not_text(inner, f"{where}.<field>")
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            found = _not_text(item, f"{where}[{index}]")
            if found is not None:
                return found
    return None


def _unknown_part(request: Mapping[str, object]) -> str | None:
    """Where the first content part not known to be text is, or ``None``.

    Only where parts are typed: ``system`` and each message's ``content`` and
    ``tool_calls``. Gemini's parts carry no type, and its tool results are the
    caller's own JSON, so they are left to the word list.
    """
    lists = [("request.system", request.get("system"))]
    messages = request.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, str | bytes):
        for index, message in enumerate(messages):
            if isinstance(message, Mapping):
                where = f"request.messages[{index}]"
                lists.append((f"{where}.content", message.get("content")))
                lists.append((f"{where}.tool_calls", message.get("tool_calls")))
    for where, parts in lists:
        found = _unknown_in(parts, where)
        if found is not None:
            return found
    return None


def _unknown_in(parts: object, where: str) -> str | None:
    """A string or nothing is text; a list is walked; anything else, a lone part
    among them, is refused, as the OpenAI adapter refuses it (/code-review)."""
    if parts is None or isinstance(parts, str):
        return None
    if not isinstance(parts, Sequence) or isinstance(parts, bytes):
        return where
    for index, part in enumerate(parts):
        here = f"{where}[{index}]"
        if not isinstance(part, Mapping):
            return here
        kind = part.get("type")
        keys = _BOUNDED_PARTS.get(kind) if isinstance(kind, str) else None
        if keys is None or not set(part) <= keys:
            return here
        if kind == "tool_result":
            found = _unknown_in(part.get("content"), f"{here}.content")
            if found is not None:
                return found
    return None


def _plain_tools(tools: object) -> bool:
    """Whether every tool is a function the caller defines, and nothing else."""
    if tools is None:
        return True
    if not isinstance(tools, Sequence) or isinstance(tools, str | bytes):
        return False
    for tool in tools:
        if not isinstance(tool, Mapping):
            return False
        keys = set(tool)
        if keys <= _GEMINI_TOOL_KEYS and keys:
            continue
        if (
            not keys <= _FUNCTION_TOOL_KEYS
            or tool.get("type", "function") != "function"
        ):
            return False
    return True


def served(_usage: Mapping[str, object]) -> dict[str, str]:
    """A policy-priced model has no rows to move between."""
    return {}
