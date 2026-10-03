"""Plans and licence keys (D46, D61, D62).

The free plan, Developer, covers up to two agents per policy (D74). A key sold for a
paid plan raises that and switches on the paid features. **Everything is checked
here, offline** (locked decision #2): a key is a signed statement of its plan and
its expiry date, and its signature is verified against ``PUBLIC_KEY`` with no
call to anyone. The signing key never leaves the maintainer's machine.

**Only a key needs a dependency.** Verifying an Ed25519 signature is done by
``cryptography``, which the free core never imports: it is loaded only when a key
is read, and installed with ``paveo[team]``. An Ed25519 written here would be the
riskiest code in the package.

**Over the limit, the extra agents are refused, never the policy** (D62). The
first agents in the policy file, up to the plan's number, keep working; each one
after them has every call refused as ``plan_limit``. A key that expires reverts
to Developer the same way, after warnings that start 14 days before, so running
agents are never all stopped at once and nothing is ever let through.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

from ._canonical import canonical_json
from .errors import ConfigError
from .policy import Policy

_logger = logging.getLogger("paveo")

# The public half of the signing key (tools/licence_keys.py keygen).
PUBLIC_KEY = bytes.fromhex(
    "0a69c4d60f645c57bc34d18c41c4aa47989150845054b244d5dcbb6febc23578"
)

_PREFIX = "paveo1"
# A trial is a start date, not a signed key: it is started on the customer's
# machine with no server to sign it (prime directive 3, D61, D63).
TRIAL_PREFIX = "paveo-trial."
TRIAL_DAYS = 30
_WARN_BEFORE = timedelta(days=14)
_MAX_KEY_LENGTH = 1024

# Agents per policy each plan covers; None is no limit.
PLANS: dict[str, int | None] = {
    "developer": 2,
    "team": 10,
    "business": 50,
    "enterprise": None,
}
# What each plan switches on beyond its agents. A feature is added here with the
# code that reads it, never before (Rule 12). A trial is Team.
FEATURES: dict[str, frozenset[str]] = {
    "developer": frozenset(),
    "team": frozenset({"evidence"}),
    "business": frozenset({"evidence"}),
    "enterprise": frozenset({"evidence"}),
}
LICENCE_FILE = "licence.key"


@dataclass(frozen=True, slots=True)
class Plan:
    """What the running copy is licensed for. ``expires`` is ``None`` for
    Developer, which never does."""

    name: str
    agents: int | None
    expires: date | None = None
    # Set when a key ran out: the plan it was, so the warning can say so.
    lapsed: str | None = None
    features: frozenset[str] = frozenset()


DEVELOPER = Plan(name="developer", agents=PLANS["developer"])


def encode(
    *, plan: str, expires: date, reference: str, sign: Callable[[bytes], bytes]
) -> str:
    """A key for ``plan`` until ``expires``, signed by ``sign``. Maintainer-only."""
    if plan not in PLANS or plan == "developer":
        raise ConfigError(f"{plan!r} is not a plan a key is sold for.", remedy="")
    payload = canonical_json(
        {"v": 1, "plan": plan, "expires": expires.isoformat(), "ref": reference}
    )
    return ".".join((_PREFIX, _b64(payload), _b64(sign(payload))))


def plan_in(directory: Path, *, today: date) -> Plan:
    """The plan ``directory/licence.key`` grants, or Developer when there is
    none. A key that is there and cannot be read raises, so every call is
    refused until it is fixed or removed, rather than guarded under a plan
    nobody chose. Shared by the guard and replay, so the two agree."""
    location = directory / LICENCE_FILE
    try:
        text = location.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEVELOPER
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(
            f"the licence key at {location} could not be read.",
            remedy="fix its permissions, or remove it to use the free plan.",
        ) from e
    return read_key(text, today=today)


def trial(started: date) -> str:
    """The token ``paveo trial`` writes: 30 days of Team from ``started``."""
    return f"{TRIAL_PREFIX}{started.isoformat()}"


def read_key(text: str, *, today: date) -> Plan:
    """The plan ``text`` grants today, or ``ConfigError`` if it is not a key we
    signed or a trial. An expired key or trial is not an error: it grants
    Developer, and says so."""
    text = text.strip()
    if text.startswith(TRIAL_PREFIX):
        return _read_trial(text.removeprefix(TRIAL_PREFIX), today=today)
    payload, signature = _split(text)
    _verify(payload, signature, PUBLIC_KEY)
    try:
        claims = json.loads(payload)
        if claims["v"] != 1:
            raise ValueError("a newer key format")  # noqa: TRY301 - one message for every unreadable key
        plan, expires = claims["plan"], date.fromisoformat(claims["expires"])
        reference = claims["ref"]
        agents = PLANS[plan]
    except (ValueError, KeyError, TypeError) as e:
        # Signed by us yet unreadable: a key from a newer version of Paveo.
        raise ConfigError(
            "the licence key is from a version of Paveo this one cannot read.",
            remedy="upgrade Paveo, or ask for a key for this version.",
        ) from e
    if not isinstance(reference, str) or plan == "developer":
        raise ConfigError(
            "the licence key names no plan this version sells.",
            remedy="ask for a new key.",
        )
    if today > expires:
        return replace(DEVELOPER, expires=expires, lapsed=plan)
    return Plan(name=plan, agents=agents, expires=expires, features=FEATURES[plan])


def _read_trial(started_on: str, *, today: date) -> Plan:
    """A trial needs no signature and so no ``paveo[team]``. Rewriting its date
    restarts it, which the licence forbids; one dated in the future was not
    started by ``paveo trial`` and is refused."""
    try:
        started = date.fromisoformat(started_on)
    except ValueError as e:
        raise _not_a_key() from e
    if started > today:
        raise ConfigError(
            "the trial in the licence key starts in the future.",
            remedy="run `paveo trial` again, or remove the key to use the free plan.",
        )
    # Thirty days means the start day and 29 more: over on the thirtieth day after.
    ends = started + timedelta(days=TRIAL_DAYS)
    if today >= ends:
        return replace(DEVELOPER, expires=ends, lapsed="team trial")
    return Plan(
        name="team", agents=PLANS["team"], expires=ends, features=FEATURES["team"]
    )


def apply(policy: Policy, plan: Plan, *, today: date) -> Policy:
    """``policy`` as ``plan`` allows it: agents past the plan's number, in the
    order the file lists them, are marked to be refused (``plan_limit``)."""
    _warn(plan, today)
    names = list(policy.agents)
    over = (
        frozenset()
        if plan.agents is None or len(names) <= plan.agents
        else frozenset(names[plan.agents :])
    )
    if over:
        _logger.warning(
            "paveo's %s plan covers %d agents per policy and this one declares "
            "%d: every call from %s is refused until the policy is trimmed or a "
            "plan that covers them is added.",
            plan.name,
            plan.agents,
            len(names),
            ", ".join(sorted(over)),
        )
    return replace(policy, plan=plan.name, over_plan=over)


def _warn(plan: Plan, today: date) -> None:
    if plan.lapsed is not None:
        _logger.warning(
            "paveo's %s licence ended on %s: the Developer plan applies, and the "
            "paid features are off. Every rule still enforces.",
            plan.lapsed,
            plan.expires,
        )
    elif plan.expires is not None and plan.expires - today <= _WARN_BEFORE:
        _logger.warning(
            "paveo's %s licence ends on %s. After that the Developer plan applies.",
            plan.name,
            plan.expires,
        )


def _split(text: str) -> tuple[bytes, bytes]:
    parts = text.split(".")
    if len(text) > _MAX_KEY_LENGTH or len(parts) != 3 or parts[0] != _PREFIX:  # noqa: PLR2004 - prefix, payload, signature
        raise _not_a_key()
    try:
        return _unb64(parts[1]), _unb64(parts[2])
    except (binascii.Error, ValueError) as e:
        raise _not_a_key() from e


def _verify(payload: bytes, signature: bytes, public_key: bytes) -> None:
    try:
        from cryptography.exceptions import InvalidSignature  # noqa: PLC0415
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: PLC0415
            Ed25519PublicKey,
        )
    except ImportError as e:
        raise ConfigError(
            "a licence key needs paveo[team], which verifies its signature.",
            remedy="pip install 'paveo[team]'. The free plan needs nothing extra.",
        ) from e
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except (InvalidSignature, ValueError) as e:
        raise _not_a_key() from e


def _not_a_key() -> ConfigError:
    return ConfigError(
        "the licence key is not one Paveo issued, or it was changed.",
        remedy=(
            "paste the key exactly as it was sent, or remove it to use the free plan."
        ),
    )


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
