"""One serialisation, so that hashing a structure gives the same answer twice.

Two things in Paveo are hashed and both must be stable against formatting:
the policy document (``§5.1`` — reloading produces a new hash, so the hash has to
mean "different policy" and not "different whitespace"), and every audit record
(``§6`` — ``hash = sha256(canonical_json(record_without_hash) || prev_hash)``).

Shared rather than duplicated because two copies of a canonicaliser that drift
apart would break the audit chain silently, which is the worst way for it to
break.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal


def _encode_unsupported(value: object) -> object:
    """Render the one non-JSON type we allow a caller to hand us.

    ``Decimal`` appears because money is a ``Decimal`` everywhere in Paveo and
    a policy built in Python rather than loaded from a file may carry one. It
    serialises as the string form, so ``Decimal("50.00")`` and ``"50.00"`` hash
    identically — they mean the same policy.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        # `json` special-cases `dict` and nothing else, so a policy validated as
        # a Mapping — a MappingProxyType, or a config library's own type — would
        # otherwise reach here and raise a bare TypeError out of a public entry
        # point. Sorting still applies: json re-serialises what this returns.
        return dict(value)
    raise TypeError(f"cannot canonicalise {type(value).__name__}")


def canonical_json(value: object) -> bytes:
    """Serialise ``value`` to the one byte string that stands for it.

    Sorted keys, no insignificant whitespace, ASCII-escaped, and ``NaN`` and the
    infinities rejected outright rather than emitted as the non-standard JSON
    literals Python would otherwise produce.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_encode_unsupported,
    ).encode("ascii")
