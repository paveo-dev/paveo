"""The Gemini adapter: a ``generate_content`` request in, a bound out (D39, D42).

The shape the google-genai SDK takes, ``{"model", "contents", "config"}``, with
``config`` as a dict (``GenerateContentConfig.model_dump()`` works). Built like
the other two readers: allowlists at every level, the bytes as the bound on
text, a stated allowance for what the provider adds, and anything else refused.

What Google's pages and SDK said, read on 2026-09-24 (D42), and what this does:

- **Thinking tokens are billed as output**, and are reported apart from
  ``candidates_token_count`` (``total`` is the sum of prompt, candidates, tool
  results and thoughts). Output is charged as candidates plus thoughts.
- **The output cap**: Google's thinking guide says ``max_output_tokens`` limits
  thinking and response together. The guide is written for the Interactions API;
  this reads ``generate_content`` the same way, **an inference, not a quotation**.
  It is backed by a check rather than trusted: a call that costs more than its
  reservation stops the table (``_PriceTable.breached``), so if the inference is
  ever wrong the first call to show it is the last one admitted.
- **``candidate_count`` answers are each capped and all billed**: bound = cap
  times the count.
- **Unset ``service_tier`` is standard** (the SDK's enum says so), so there is
  no hidden-default trap here, unlike OpenAI and Anthropic.
- **Refused**: images, audio, video and files (each priced by modality, and
  audio dearer), explicit ``cached_content`` (billed for storage by the hour,
  which is not a call), and every server tool (Google Search and Maps grounding
  is billed per request).
- **Function declarations** are rendered by Google before the model reads them,
  with no published overhead, so 1,000 tokens are added whenever tools are
  present: chosen, as for OpenAI, and backed by the same check.

Nothing here is stored, logged or echoed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._anthropic import RequestBound, _byte_length, _Refusal, model_of
from .errors import ConfigError
from .prices import _printable

_TOP = frozenset({"model", "contents", "config"})

_CONFIG = frozenset(
    {
        "system_instruction",
        "temperature",
        "top_p",
        "top_k",
        "candidate_count",
        "max_output_tokens",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "response_mime_type",
        "response_schema",
        "response_json_schema",
        "tools",
        "tool_config",
        "thinking_config",
        "service_tier",
        "safety_settings",
        "automatic_function_calling",
        # Transport, kept by the SDK. Only its timeout and retries are admitted:
        # `extra_body` is merged into the request body ("Extra parameters to add
        # to the request body", the SDK), so it could carry any refused parameter
        # past this reader, and `base_url` can point at an endpoint priced
        # differently (/code-review, D42).
        "http_options",
    }
)
_NOT_IN_BODY = frozenset({"http_options"})
_HTTP_OPTIONS = frozenset({"timeout", "retry_options"})

# A part that is text, a thought, or a function call and its result. Both the
# SDK's snake_case and the REST API's camelCase spellings.
_PART_KEYS = frozenset(
    {
        "text",
        "thought",
        "thought_signature",
        "thoughtSignature",
        "function_call",
        "functionCall",
        "function_response",
        "functionResponse",
    }
)
_TOOL_KEYS = frozenset({"function_declarations", "functionDeclarations"})

# Not from any page, as D41's for OpenAI: chosen, generous, and backed by the
# table's breach check.
_TOOLS_ALLOWANCE = 1_000

_USAGE_FIELDS = frozenset(
    {
        "prompt_token_count",
        "cached_content_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "tool_use_prompt_token_count",
        "total_token_count",
        "traffic_type",
        "prompt_tokens_details",
        "candidates_tokens_details",
        "cache_tokens_details",
        "tool_use_prompt_tokens_details",
    }
)
_CAMEL = {
    "promptTokenCount": "prompt_token_count",
    "cachedContentTokenCount": "cached_content_token_count",
    "candidatesTokenCount": "candidates_token_count",
    "thoughtsTokenCount": "thoughts_token_count",
    "toolUsePromptTokenCount": "tool_use_prompt_token_count",
    "totalTokenCount": "total_token_count",
    "trafficType": "traffic_type",
    "promptTokensDetails": "prompt_tokens_details",
    "candidatesTokensDetails": "candidates_tokens_details",
    "cacheTokensDetails": "cache_tokens_details",
    "toolUsePromptTokensDetails": "tool_use_prompt_tokens_details",
}


def bound(
    request: Mapping[str, object], *, assumed_max_output_tokens: int | None
) -> RequestBound:
    """Measure a ``generate_content`` request, or refuse it."""
    model = model_of(request)
    refuse = _Refusal(model)

    unknown = [key for key in request if key not in _TOP]
    raw = request.get("config")
    if raw is not None and not isinstance(raw, Mapping):
        refuse("config is not a mapping", "pass config as a dict (model_dump()).")
    # `GenerateContentConfig.model_dump()` writes every field, most of them None:
    # a None is a field left unset, not a request for it (/code-review, D42).
    config = {k: v for k, v in (raw or {}).items() if v is not None}
    unknown += [key for key in config if key not in _CONFIG]
    http = config.get("http_options", {})
    if not isinstance(http, Mapping) or any(
        value is not None and key not in _HTTP_OPTIONS for key, value in http.items()
    ):
        refuse(
            "config.http_options sets something other than a timeout or retries: "
            "extra_body is merged into the request, and headers and base_url can "
            "change what the call is billed",
            "keep http_options to timeout and retry_options on calls checked by "
            "paveo (D42).",
        )
    if unknown:
        refuse(
            f"the request sets {_printable(unknown[0])!r}, which paveo cannot "
            f"bound the cost of",
            "leave it out, or tell us you need it: each is added once its cost is "
            "verified (D42).",
        )

    tools = config.get("tools")
    if tools is not None and not _function_tools(tools):
        refuse(
            "the request carries a tool that is not a function you declare: "
            "Google Search and Maps grounding and the other server tools bill "
            "beyond the request",
            "use function declarations only on calls checked by paveo (D42).",
        )

    for where, content in (
        ("contents", request.get("contents")),
        ("config.system_instruction", config.get("system_instruction")),
    ):
        found = _not_text(content, where)
        if found is not None:
            refuse(
                f"{found} is not text: images, audio, video and files cost by "
                f"modality, which their bytes do not measure",
                "send text only on calls checked by paveo for now (D42).",
            )

    cap = config.get("max_output_tokens")
    assumed = False
    if cap is None:
        if assumed_max_output_tokens is None:
            refuse(
                "the request sets no max_output_tokens, so its output has no bound",
                "set config.max_output_tokens, which caps thinking and response "
                "together, or set defaults.assumed_max_output_tokens (§4.4).",
            )
        cap, assumed = assumed_max_output_tokens, True
    answers = config.get("candidate_count", 1)
    for value, name in ((cap, "max_output_tokens"), (answers, "candidate_count")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            refuse(
                f"{name} is not a whole number of tokens or answers",
                f"set config.{name} to a whole number.",
            )
    if answers == 0:
        refuse("candidate_count is 0", "set it to 1 or more, or leave it unset.")

    tier = config.get("service_tier")
    body = {
        **{k: v for k, v in request.items() if k != "config"},
        "config": {k: v for k, v in config.items() if k not in _NOT_IN_BODY},
    }
    return RequestBound(
        model=model,
        modifiers={"service_tier": tier.lower()} if isinstance(tier, str) else {},
        input_upper_bound=_byte_length(body) + (_TOOLS_ALLOWANCE if tools else 0),
        max_output_tokens=answers * cap,
        assumed_output=assumed,
    )


def usage_classes(usage: Mapping[str, object]) -> dict[str, object]:
    """A response's ``usage_metadata``, in the price table's class names (§4.2).

    Pass ``response.usage_metadata.model_dump()``, the REST ``usageMetadata``, or
    the whole response. ``prompt_token_count`` includes the cached tokens, so they
    are taken out of ``input``; thoughts are added to output, since they are
    reported apart from the candidates and billed as output.

    Tool-result tokens (``tool_use_prompt_token_count``) come from server tools,
    which were refused at reserve time, so a non-zero count reaches the table as a
    class it does not know and stops it. So does any modality but text in the
    prompt breakdown, and any top-level field this adapter does not read.
    """
    if isinstance(usage, Mapping):
        for key in ("usage_metadata", "usageMetadata"):
            if key in usage:
                # A whole response: its usage, or nothing. A response whose usage
                # is None is unreadable, not a call that cost nothing
                # (/code-review, D42).
                usage = usage[key]  # type: ignore[assignment]  # checked below
                break
    if not isinstance(usage, Mapping):
        raise ConfigError(
            "usage is not a mapping.",
            remedy="pass response.usage_metadata.model_dump().",
        )
    fields = {_CAMEL.get(key, key): value for key, value in usage.items()}
    if not isinstance(fields.get("prompt_token_count"), int):
        # No prompt count is not a call that cost nothing (/security-review, D42).
        raise ConfigError(
            "usage carries no prompt_token_count.",
            remedy="pass response.usage_metadata.model_dump() as returned.",
        )
    prompt = fields.get("prompt_token_count") or 0
    cached = fields.get("cached_content_token_count") or 0
    candidates = fields.get("candidates_token_count") or 0
    thoughts = fields.get("thoughts_token_count") or 0
    tool_use = fields.get("tool_use_prompt_token_count") or 0
    counts = (prompt, cached, candidates, thoughts, tool_use)
    if not all(
        isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in counts
    ):
        raise ConfigError(
            "usage does not carry whole token counts.",
            remedy="pass response.usage_metadata.model_dump() as returned.",
        )
    if cached > prompt:  # type: ignore[operator]  # all checked to be ints above
        raise ConfigError(
            "usage reports more cached tokens than prompt tokens.",
            remedy="pass the usage metadata exactly as the response returned it.",
        )
    classes: dict[str, object] = {
        "input": prompt - cached,  # type: ignore[operator]  # checked above
        "cache_read": cached,
        "output": candidates + thoughts,  # type: ignore[operator]  # checked above
    }
    if tool_use:
        classes["tool_use_prompt_token_count"] = tool_use
    for key, value in fields.items():
        if key not in _USAGE_FIELDS:
            classes[_printable(key)] = (
                value if isinstance(value, int) and not isinstance(value, bool) else 0
            )
    for entry in _as_list(fields.get("prompt_tokens_details")):
        modality = entry.get("modality") if isinstance(entry, Mapping) else None
        count = (
            entry.get("token_count", entry.get("tokenCount"))
            if isinstance(entry, Mapping)
            else None
        )
        if (
            isinstance(modality, str)
            and modality.upper() != "TEXT"
            and isinstance(count, int)
            and count > 0
        ):
            classes[_printable(f"prompt_{modality.lower()}")] = count
    return classes


def served(_usage: Mapping[str, object]) -> dict[str, str]:
    """Gemini's usage reports no tier, and an unset one is standard anyway."""
    return {}


def _as_list(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _function_tools(tools: object) -> bool:
    if not isinstance(tools, Sequence) or isinstance(tools, str | bytes):
        return False
    return all(isinstance(t, Mapping) and t and set(t) <= _TOOL_KEYS for t in tools)


def _not_text(value: object, where: str) -> str | None:
    """Where in contents something is not text, or ``None``. Never what it is.

    Contents may be a string, a list of strings and parts, or a list of
    ``{"role", "parts"}`` turns; a system instruction is a string or one turn.
    """
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, Mapping):
        return _not_text_mapping(value, where)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        for index, item in enumerate(value):
            found = _not_text(item, f"{where}[{index}]")
            if found is not None:
                return found
        return None
    return where


def _not_text_mapping(value: Mapping[str, object], where: str) -> str | None:
    """A turn is walked into its parts; a part must be text, a thought or a call.

    A function's result may carry parts of its own (``FunctionResponse.parts``),
    and those can be an image or a video by URL: tool output is the part of the
    conversation an agent's inputs reach, so a result carrying any is refused
    (/security-review, D42).
    """
    if "parts" in value or "role" in value:
        if not set(value) <= {"role", "parts"}:
            return where
        return _not_text(value.get("parts"), f"{where}.parts")
    if not value or not set(value) <= _PART_KEYS:
        return where
    for key in ("function_response", "functionResponse"):
        result = value.get(key)
        if isinstance(result, Mapping) and result.get("parts"):
            return f"{where}.{key}.parts"
    return None
