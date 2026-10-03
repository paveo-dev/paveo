"""What a call costs (``docs/SPEC_V1.md`` §4.8, D9).

The ledger can only promise a ceiling if the number it reserves is really an
upper bound on what the call will cost. This module produces that number and,
after the call, the true one. It is where "cannot overspend" meets the
provider's price list.

**Price is a function of three things, not one** (§4.1):
the model, the price-affecting request parameters, and how the tokens divided
between classes. The first two are known before the call and select a row of
the table (``resolve``). The third is only known after it, so the reservation
takes the most expensive class that could apply (``_PriceTable.worst_case``) and the
response's own counts settle it (``_PriceTable.actual``).

**The table is data, dated and checked in.** Rates are written as the provider
publishes them, in USD per million tokens, and every combination a request can
select resolves to absolute per-token ``Decimal`` rates once, at import. No
arithmetic anywhere else touches a multiplier.

**Every number here was read from its provider's pricing page on the date in
``VERIFIED[provider]``**. Re-verify before every release, Rule 4:
a stale rate that is too low is a ceiling that quietly stops being true
(§10.4, §10.10, §12.5). ``tests/test_prices.py`` pins each resolved rate to the
page's own absolute figures, so a mistyped multiplier fails there.

**Arithmetic runs in the ledger's private decimal context** (D35). Token counts
times rates is exactly the sum a caller's lowered precision could round.

**Thread safety.** ``Rates`` is immutable and safe to share. ``_PriceTable`` is
safe to share across threads and tasks: its rows never change after
construction, and the one thing that does change, the stale mark of §4.8.3, is
written and read under its own lock, held for a single assignment and never
across an ``await``. Nothing here may import ``asyncio`` (D29, D33).
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from types import MappingProxyType

from .budget import _ARITHMETIC
from .errors import ConfigError, PricingUnknown
from .policy import DeclaredPrice

# The day each provider's rows were last read from its own pricing page. The
# only thing that keeps a price table true is someone re-reading those pages, so
# the dates are data a check can read rather than a comment someone must notice
# (§12.5, P1, D43): Paveo warns once a table is over _STALE_AFTER_DAYS old, and
# `make release-check` refuses to ship one over 30 days old. docs/PRICES.md is
# the procedure.
VERIFIED: Mapping[str, str] = {
    "anthropic": "2026-09-23",
    "openai": "2026-09-24",
    "gemini": "2026-09-24",
}
# The version names the table as a whole: the latest date any part was read.
PRICES_VERSION = max(VERIFIED.values())
_STALE_AFTER_DAYS = 60


# Providers already warned about in this process: a warning that repeats for every
# Paveo a service builds is one operators learn to ignore (/code-review, D43).
_WARNED: set[str] = set()
_WARNED_LOCK = threading.Lock()
_logger = logging.getLogger("paveo")


def _warn_if_old(moment: datetime) -> None:
    """Say once, per provider and process, that its prices are old.

    A warning and not a refusal: an old table is not a wrong one, and a calendar
    date must not become every customer's outage; the refusal belongs to the
    release (``make release-check``, §12.5, D43). A naive clock is left to the
    ledger and the log, which refuse it on their first use (D19); read here as
    local time it could only put the date a day out.
    """
    if moment.tzinfo is None:
        return
    for provider in stale_providers(
        moment.astimezone(UTC).date(), older_than_days=_STALE_AFTER_DAYS
    ):
        with _WARNED_LOCK:
            if provider in _WARNED:
                continue
            _WARNED.add(provider)
        _logger.warning(
            "paveo's %s prices were last verified on %s, over %d days ago. A price "
            "the provider raised since is under-charged until paveo is upgraded; "
            "the version is in every audit record as prices_version.",
            provider,
            VERIFIED[provider],
            _STALE_AFTER_DAYS,
        )


def stale_providers(today: date, *, older_than_days: int) -> list[str]:
    """Providers whose rows were last verified more than ``older_than_days`` ago."""
    return sorted(
        provider
        for provider, verified in VERIFIED.items()
        if (today - date.fromisoformat(verified)).days > older_than_days
    )


_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"

# The token classes a response can report (§4.8.1). Input classes are priced off
# the base input rate, `output` off the base output rate.
INPUT_CLASSES = (
    "input",
    "cache_read",
    "cache_write_5m",
    "cache_write_1h",
    "cache_write",
)
OUTPUT_CLASS = "output"
_KNOWN_CLASSES = frozenset({*INPUT_CLASSES, OUTPUT_CLASS})

# The same for every model on the page. `cache_read` is not, and sits on each row.
_CACHE_WRITE_5M = Decimal("1.25")
_CACHE_WRITE_1H = Decimal("2")

# §4.8.2's price-affecting parameters, and what each is priced as when the request
# leaves it unset. A request parameter not named here does not change the price as
# of PRICES_VERSION; a new one the provider ships would be invisible to us, which
# is §10.10's published gap and the reason this list is re-read on every update.
#
# `speed` unset is standard: it is chosen per request and nowhere else.
# `inference_geo` unset is **not** the "global" the API documents as its default,
# because a workspace's `default_inference_geo` decides it and we cannot see the
# workspace. Every org that had opted out of global routing was migrated to a US
# default "with no code changes required", so a request that never mentions the
# parameter can be billed at 1.1x; priced at 1.0x, every such call would be
# under-reserved and under-settled by a tenth, silently. Found by /security-review
# (D38). It gets a row of its own, priced at the dearest geo the model accepts, so
# the audit record says "unset" rather than claiming a geo nobody asked for.
_UNSET = "unset"
_ABSENT = {"speed": "standard", "inference_geo": _UNSET, "service_tier": _UNSET}
# The value of each that selects no premium, the only one a row without that
# parameter may be asked for.
_PLAIN = {"speed": "standard", "inference_geo": "global", "service_tier": "default"}

# `inference_geo: "us"` multiplies every class, input and output alike, on the
# models that accept it. It stacks with fast mode and the cache multipliers.
_US_INFERENCE = Decimal("1.1")

# A class name the table does not know ends up in an error message and, from S4,
# in the audit log. It comes from a provider response rather than from the model,
# but a string that reaches the deny path unchecked is how D26 happened, so only
# a plain identifier is written down.
_PRINTABLE_CLASS = re.compile(r"[a-z0-9_]{1,64}")

_PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class _Listing:
    """One model's row on the pricing page, in USD per million tokens.

    ``fast`` is the base pair ``speed: "fast"`` selects, or ``None`` where the
    page does not price it; a request asking for it is then refused, since we do
    not guess (§4.8.2). ``us_inference`` is whether ``inference_geo: "us"`` is
    priced for this model: the page applies it to "Claude 4.6 and later" and says
    earlier models reject the parameter outright.
    """

    standard: tuple[str, str]
    cache_read: str = "0.1"
    fast: tuple[str, str] | None = None
    us_inference: bool = True


# Read from _SOURCE on 2026-09-23. Deliberately absent: the Mythos models
# (limited availability), Opus 4.5 and Sonnet 4.5, and everything retired. A model
# not listed is refused with PricingUnknown rather than guessed at.
_LISTINGS: Mapping[str, _Listing] = {
    # Cache reads at 0.025x on Fable 5.1, footnote 1 on the page.
    "claude-fable-5-1": _Listing(standard=("10", "50"), cache_read="0.025"),
    "claude-fable-5": _Listing(standard=("10", "50")),
    # Cache reads at 0.05x, footnote 2.
    "claude-opus-5-5": _Listing(
        standard=("4", "20"), cache_read="0.05", fast=("8", "40")
    ),
    "claude-opus-5": _Listing(standard=("5", "25"), fast=("10", "50")),
    "claude-opus-4-8": _Listing(standard=("5", "25"), fast=("10", "50")),
    # Opus 4.7 returns an error for `speed: "fast"`, so it is not priced. Opus 4.6
    # accepts it, "runs at standard speed and is billed at standard rates".
    "claude-opus-4-7": _Listing(standard=("5", "25")),
    "claude-opus-4-6": _Listing(standard=("5", "25"), fast=("5", "25")),
    # $2/$10 is the standard price; the increase once scheduled for 1 Sep 2026
    # was cancelled, per the note on the page.
    "claude-sonnet-5": _Listing(standard=("2", "10")),
    "claude-sonnet-4-6": _Listing(standard=("3", "15")),
    # Earlier than 4.6, so `inference_geo` is not accepted on it at all. Listed
    # under both ids it is served by.
    "claude-haiku-4-5": _Listing(standard=("1", "5"), us_inference=False),
    "claude-haiku-4-5-20251001": _Listing(standard=("1", "5"), us_inference=False),
}


@dataclass(frozen=True, slots=True)
class _OpenAIListing:
    """One OpenAI model's rows, USD per million tokens: (input, cached input, output).

    Read from ``developers.openai.com/api/docs/pricing`` and the model pages on
    2026-09-24 (D41). ``fast`` is the Fast (formerly priority) tier, ``None``
    where the page lists none. ``cache_write`` is GPT-5.6 and later, where a
    cache write costs 1.25x uncached input and each token is billed at exactly one
    rate; on earlier models a write costs plain input. ``long_context``: a prompt
    over 272K input tokens bills the **whole request** at 2x input and 1.5x
    output. ``regional``: released on or after 2026-03-05 and on the data
    residency list, so a regional endpoint or a project's region adds 10%, which
    no request can show us, so it is always charged.
    """

    standard: tuple[str, str, str]
    fast: tuple[str, str, str] | None = None
    cache_write: bool = False
    long_context: bool = False
    regional: bool = False


_OPENAI_LISTINGS: Mapping[str, _OpenAIListing] = {
    "gpt-6-astra": _OpenAIListing(
        ("10", "1", "50"),
        ("20", "2", "100"),
        cache_write=True,
        long_context=True,
        regional=True,
    ),
    "gpt-6-sol": _OpenAIListing(
        ("2", "0.2", "10"),
        ("4", "0.4", "20"),
        cache_write=True,
        long_context=True,
        regional=True,
    ),
    "gpt-6-luna": _OpenAIListing(
        ("0.10", "0.01", "0.50"),
        ("0.20", "0.02", "1.00"),
        cache_write=True,
        long_context=True,
        regional=True,
    ),
    "gpt-5.6-sol": _OpenAIListing(
        ("4", "0.4", "20"),
        ("8", "0.8", "40"),
        cache_write=True,
        long_context=True,
        regional=True,
    ),
    "gpt-5.6-terra": _OpenAIListing(
        ("2", "0.2", "12"),
        ("4", "0.4", "24"),
        cache_write=True,
        long_context=True,
        regional=True,
    ),
    "gpt-5.6-luna": _OpenAIListing(
        ("0.20", "0.02", "1.20"),
        ("0.40", "0.04", "2.40"),
        cache_write=True,
        long_context=True,
        regional=True,
    ),
    # Fast is 2.5x standard here, not the 2x of the newer models.
    "gpt-5.5": _OpenAIListing(
        ("5", "0.5", "30"), ("12.50", "1.25", "75"), long_context=True, regional=True
    ),
    "gpt-5.4": _OpenAIListing(
        ("2.50", "0.25", "15"), ("5", "0.5", "30"), long_context=True, regional=True
    ),
    "gpt-5.4-mini": _OpenAIListing(
        ("0.75", "0.075", "4.50"), ("1.50", "0.15", "9"), regional=True
    ),
    "gpt-5.4-nano": _OpenAIListing(("0.20", "0.02", "1.25"), regional=True),
    "gpt-5.2": _OpenAIListing(("1.75", "0.175", "14"), ("3.50", "0.35", "28")),
    "gpt-5.1": _OpenAIListing(("1.25", "0.125", "10"), ("2.50", "0.25", "20")),
    "gpt-5": _OpenAIListing(("1.25", "0.125", "10"), ("2.50", "0.25", "20")),
    "gpt-5-mini": _OpenAIListing(("0.25", "0.025", "2"), ("0.45", "0.045", "3.60")),
    "gpt-5-nano": _OpenAIListing(("0.05", "0.005", "0.40")),
    "gpt-4.1": _OpenAIListing(("2", "0.5", "8"), ("3.50", "0.875", "14")),
    "gpt-4.1-mini": _OpenAIListing(("0.40", "0.10", "1.60"), ("0.70", "0.175", "2.80")),
    "gpt-4.1-nano": _OpenAIListing(("0.10", "0.025", "0.40"), ("0.20", "0.05", "0.80")),
    "gpt-4o": _OpenAIListing(("2.50", "1.25", "10"), ("4.25", "2.125", "17")),
    "gpt-4o-mini": _OpenAIListing(("0.15", "0.075", "0.60"), ("0.25", "0.125", "1.00")),
    "o3": _OpenAIListing(("2", "0.5", "8"), ("3.50", "0.875", "14")),
    "o4-mini": _OpenAIListing(("1.10", "0.275", "4.40"), ("2", "0.5", "8")),
}


@dataclass(frozen=True, slots=True)
class _GeminiListing:
    """One Gemini model's paid-tier rows, USD per million tokens.

    Read from ``ai.google.dev/gemini-api/docs/pricing`` on 2026-09-24 (D42).
    ``standard`` is (input, cached input, output); cached is ``None`` where the
    page says caching is not available. ``priority`` is (input, output); its
    cached rate is not published, so a cached token on Priority is charged at
    Priority input, the dearer of the two. ``long_context``: a prompt over 200K
    tokens bills the whole request at 2x input and 1.5x output (every "> 200k"
    row on the page fits). ``changes``: the date a new price takes effect, and the
    listing from then, since three Flash models double on 2027-01-01.
    """

    standard: tuple[str, str | None, str]
    priority: tuple[str, str] | None = None
    long_context: bool = False
    changes: tuple[str, _GeminiListing] | None = None


_FLASH_2026 = _GeminiListing(
    ("0.75", "0.075", "3.75"),
    ("1.35", "6.75"),
    changes=("2027-01-01", _GeminiListing(("1.50", "0.15", "7.50"), ("2.70", "13.50"))),
)

_GEMINI_LISTINGS: Mapping[str, _GeminiListing] = {
    "gemini-3.8-flash": _FLASH_2026,
    "gemini-3.7-flash": _FLASH_2026,
    "gemini-3.6-flash": _FLASH_2026,
    "gemini-3.5-flash": _GeminiListing(("1.50", "0.15", "9.00"), ("2.70", "16.20")),
    "gemini-3.5-flash-lite": _GeminiListing(("0.30", None, "2.50"), ("0.54", "4.50")),
    # Text, image and video input; audio is dearer and is refused by the adapter.
    "gemini-3.1-flash-lite": _GeminiListing(
        ("0.25", "0.025", "1.50"), ("0.45", "2.70")
    ),
    "gemini-3.1-pro-preview": _GeminiListing(
        ("2.00", "0.20", "12.00"), ("3.60", "21.60"), long_context=True
    ),
    "gemini-3-flash-preview": _GeminiListing(
        ("0.50", "0.05", "3.00"), ("0.90", "5.40")
    ),
    "gemini-2.5-pro": _GeminiListing(
        ("1.25", "0.125", "10.00"), ("2.25", "18.00"), long_context=True
    ),
    "gemini-2.5-flash": _GeminiListing(("0.30", "0.03", "2.50"), ("0.54", "4.50")),
    "gemini-2.5-flash-lite": _GeminiListing(("0.10", "0.01", "0.40"), ("0.18", "0.72")),
}

# Gemini's tiers. Unset is standard: the SDK's own enum says UNSPECIFIED is
# "Default service tier, which is standard". `flex` is priced at standard, the
# safe side of a cheaper tier (D42).
_GEMINI_TIERS = ("standard", "flex", "unspecified")
_GEMINI_LONG_CONTEXT_TOKENS = 200_000

_AnyListing = _Listing | _OpenAIListing | _GeminiListing

# Every model the table prices, all three providers. A test checks the ids do not
# collide, since one dict would silently keep only the last.
# Each provider's rows, by the name VERIFIED dates them under. A test checks the
# two name the same providers, so a fourth provider cannot be added without a
# date its age is judged by (/code-review, D43).
_PROVIDERS: Mapping[str, Mapping[str, _AnyListing]] = {
    "anthropic": _LISTINGS,
    "openai": _OPENAI_LISTINGS,
    "gemini": _GEMINI_LISTINGS,
}
_ALL_LISTINGS: Mapping[str, _AnyListing] = {
    model: listing for rows in _PROVIDERS.values() for model, listing in rows.items()
}

_LONG_CONTEXT_TOKENS = 272_000

# A class some adapters report under a name a row may not carry, and the row's
# class it is charged at instead: OpenAI's single `cache_write`, arriving for a
# Claude model through LiteLLM, is charged at Anthropic's dearer 1-hour write.
_ALIASES: Mapping[str, str] = {"cache_write": "cache_write_1h"}
_LONG_INPUT = Decimal(2)
_LONG_OUTPUT = Decimal("1.5")
_OPENAI_CACHE_WRITE = Decimal("1.25")
_REGIONAL_UPLIFT = Decimal("1.1")
# `flex` is priced at standard rates, the safe side of a cheaper tier we have not
# checked falls back to; `auto` is whatever the project is set to, so it is priced
# like an unset tier, at the dearest one the model has (D41).
_OPENAI_TIERS = ("default", "flex", "fast", "priority", "auto")


@dataclass(frozen=True, slots=True)
class Rates:
    """Per-token rates for one model under one set of price-affecting parameters.

    ``rate_key`` names which row was selected, for the audit record, so a
    disputed charge can be explained a year later (§6). ``per_token`` holds every
    class the row prices, in USD per token.

    Only the table that issued a ``Rates`` prices anything with it, and it refuses
    one it did not issue. A ``Rates`` held across calls would otherwise keep
    pricing after its table had gone stale, and "call ``resolve`` every time" is
    the kind of rule Rule 5 says will eventually be forgotten.
    """

    model: str
    rate_key: str
    per_token: Mapping[str, Decimal]


@dataclass(frozen=True, slots=True)
class Charge:
    """What a finished call cost, and whether its response broke the table.

    ``price_table_stale`` is True when the response carried a class the table
    does not know. That class was charged at the table's highest rate, which may
    still be too low, and every reserve after it is refused (§4.8.3). The audit
    record carries the flag (§4.2). ``rate_key`` names the row actually charged,
    which can differ from the one reserved (a long prompt, or the tier the
    response says it was served at), so the audit line explains the number.
    """

    cost: Decimal
    price_table_stale: bool
    rate_key: str


class _PriceTable:
    """Every row a request can select, and whether a response has outrun them.

    Built once from listings, which is where every multiplier is applied, and
    from any prices the policy declares for models the listings lack (§4.8.4,
    D39). See the module docstring for thread safety.
    """

    def __init__(
        self,
        listings: Mapping[str, _AnyListing],
        *,
        declared: Mapping[str, DeclaredPrice] | None = None,
        version: str = PRICES_VERSION,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        declared = {} if declared is None else declared
        collisions = sorted(set(declared) & set(listings))
        if collisions:
            raise ConfigError(
                f"the policy declares a price for {', '.join(collisions)}, which "
                f"price table version {version} already carries.",
                remedy=(
                    "remove it from the policy's prices: a declared price is for a "
                    "model paveo does not know, and one written over a known model "
                    "could only be a mistake or an under-price (D39)."
                ),
            )
        self.version = version
        # Injected (Rule 14): a price that changes on a date is chosen by it, and a
        # test cannot wait for 1 January 2027.
        self._now = now if now is not None else _utc_now
        # The dates a listing's price changes, earliest first (D42).
        self._changes: dict[str, list[str]] = {
            model: _change_dates(listing)
            for model, listing in listings.items()
            if isinstance(listing, _GeminiListing) and listing.changes is not None
        }
        self._permitted: dict[str, dict[str, tuple[str, ...]]] = {
            model: _modifier_values(listing) for model, listing in listings.items()
        }
        self._rows: dict[tuple[str, frozenset[tuple[str, str]]], Rates] = {}
        # A row with a long-context tier: the prompt size above which the whole
        # request is billed at the long row instead (D41).
        self._long: dict[str, tuple[int, Rates]] = {}
        # Which choice each row is, so a settle can move to the row the response
        # says it was served at (D41).
        self._choice: dict[str, tuple[str, dict[str, str]]] = {}
        for model, listing in listings.items():
            for choice, rates, long_rates in _rows_for(model, listing):
                self._rows[(model, frozenset(choice.items()))] = rates
                self._choice[rates.rate_key] = (model, choice)
                if long_rates is not None:
                    threshold = (
                        _GEMINI_LONG_CONTEXT_TOKENS
                        if isinstance(listing, _GeminiListing)
                        else _LONG_CONTEXT_TOKENS
                    )
                    self._long[rates.rate_key] = (threshold, long_rates)
        for model, price in declared.items():
            # A declared price has no price-affecting parameters we know of: the
            # operator wrote the price they pay, and it is only as true as that.
            self._permitted[model] = {}
            self._rows[(model, frozenset())] = _declared_row(model, price)
        self._by_key = {rates.rate_key: rates for rates in self._rows.values()}
        self._lock = threading.Lock()
        self._unrecognised: str | None = None
        # Set when a call cost more than its reservation (D42): a different
        # failure from an unknown class, with a different remedy.
        self._breach: str | None = None
        _warn_if_old(self._now())

    def carries(self, model: str) -> bool:
        """Whether this exact name is one the table prices: built in or declared."""
        return model in self._permitted

    def resolve(self, model: str, modifiers: Mapping[str, object]) -> Rates:
        """The rates this request will be charged at, or ``PricingUnknown``.

        ``modifiers`` is the request's parameters; only the price-affecting ones
        are read. An absent one is priced as ``_ABSENT`` says, which for
        ``inference_geo`` is deliberately not the API's documented default.

        Refused, never guessed: a model the table does not carry, a
        price-affecting parameter set to a value it does not carry, and, once any
        response has reported a class the table does not know, every request at
        all (§4.8.3). ``fail_open`` is the caller's to apply, not this table's.

        The model name is echoed in the error. That is safe because ``check_llm``
        asks the policy first, and the policy admits only models the operator
        wrote down (§5.1); a model a shadowed rule let through reaches here only
        if ``carries`` it (D48).
        """
        self._refuse_if_stale(model)

        permitted = self._permitted.get(model)
        if permitted is None:
            priced = ", ".join(sorted(self._permitted))
            raise PricingUnknown(
                model=model,
                detail=f"it is not in price table version {self.version}",
                remedy=(
                    f"use a model the table prices ({priced}), or upgrade paveo if "
                    f"the model is newer than the table."
                ),
            )

        # A price-affecting setting this row has no rows for is refused unless
        # it names the plain choice, rather than dropped: dropping it priced a
        # Fast call to a policy-priced model at the declared standard rate
        # (/security-review, D41).
        for name in _ABSENT:
            value = modifiers.get(name)
            if (
                name not in permitted
                and value is not None
                and value != _PLAIN.get(name)
            ):
                raise PricingUnknown(
                    model=model,
                    detail=(
                        f"{name} is set to a value this model's price does not cover"
                    ),
                    remedy=(
                        f"leave {name} unset or set it to {_PLAIN[name]!r}, or "
                        f"price the model through a provider paveo has a table for."
                    ),
                )

        chosen: dict[str, str] = {}
        for name, values in permitted.items():
            if name not in modifiers:
                chosen[name] = _ABSENT[name]
                continue
            value = modifiers[name]
            if not isinstance(value, str) or value not in values:
                # The value is not echoed: name the parameter and what would work.
                raise PricingUnknown(
                    model=model,
                    detail=(
                        f"{name} is set to a value price table version "
                        f"{self.version} does not price for this model"
                    ),
                    remedy=(
                        f"set {name} to one of: {', '.join(values)}, or leave it unset."
                    ),
                )
            chosen[name] = value

        if model in self._changes:
            chosen["from"] = self._period(model, self._now())
        return self._rows[(model, frozenset(chosen.items()))]

    def _period(self, model: str, moment: datetime) -> str:
        """Which dated price is in force at ``moment``, by its UTC day."""
        day = moment.astimezone(UTC).date().isoformat()
        return max((d for d in self._changes[model] if d <= day), default="current")

    def _dated_twins(self, row: Rates, *moments: datetime) -> list[Rates]:
        """The same row under the price in force at each moment, for a dated model.

        A call admitted at 23:59 on the last day of a price may be billed at the
        next day's, so reserving and settling near a change take the dearer
        (/code-review, D42).
        """
        known = self._choice.get(row.rate_key)
        if known is None or known[0] not in self._changes:
            return [row]
        model, choice = known
        twins = [row]
        for moment in moments:
            twin = self._rows.get(
                (
                    model,
                    frozenset({**choice, "from": self._period(model, moment)}.items()),
                )
            )
            if twin is not None and twin not in twins:
                twins.append(twin)
        return twins

    def worst_case(
        self, rates: Rates, input_upper_bound: int, max_output_tokens: int
    ) -> Decimal:
        """The most this call can cost, before anyone knows how its tokens divide.

        Every input token is priced at the most expensive input class, because a
        request that enables caching can come back as almost all cache writes and
        nothing at reserve time says otherwise (§4.1). This over-reserves on a
        cache-read-heavy call, by up to 80x its input cost on Fable 5.1, and it can
        never under-reserve, which is the trade §10.9 publishes.

        Refuses on a stale table exactly as ``resolve`` does, so the refusal holds
        however the ``Rates`` were obtained. Not rounded: the ledger rounds it up,
        once, where the ceiling is checked.
        """
        row = self._row(rates)
        self._refuse_if_stale(row.model)
        inputs = _tokens(input_upper_bound, "input_upper_bound")
        outputs = _tokens(max_output_tokens, "max_output_tokens")
        now = self._now()
        worst = Decimal(0)
        for twin in self._dated_twins(row, now, now + timedelta(days=1)):
            sized = self._sized(twin, input_upper_bound, margin=True)
            with localcontext(_ARITHMETIC):
                dearest_input = max(
                    rate
                    for name, rate in sized.per_token.items()
                    if name in INPUT_CLASSES
                )
                worst = max(
                    worst,
                    inputs * dearest_input + outputs * sized.per_token[OUTPUT_CLASS],
                )
        return worst

    def actual(
        self,
        rates: Rates,
        usage: Mapping[str, object],
        served: Mapping[str, str] | None = None,
    ) -> Charge:
        """Price what a response reported, class by class (§4.2).

        ``usage`` is the response's own per-class token counts, and nothing else
        feeds this: the reserve-time estimate is never an input to the actual.

        A class the table does not know is charged at the row's highest rate of
        either basis, since which basis it is priced off is exactly what we do not
        know, and the table is marked stale so that every later ``resolve``
        refuses. Deliberately not a refusal here: this call has already happened,
        and a settle that raised would lose its charge.

        **An unknown class marks the table stale even if the response is otherwise
        malformed**, so the refusal of every later reserve does not depend on the
        rest of it being well-formed. But only a key carrying a real token count
        is evidence of a token class: a string under ``inference_geo`` is an
        adapter passing a non-token field through, which raises as the bug it is
        rather than stopping every agent in the process.

        A count no correct caller produces raises ``ValueError`` (D24), naming the
        class but never the value.

        A row with a long-context tier is priced from the response's own prompt
        size, not from the bound it was reserved on (D41). And ``served``, what
        the response says it was served at (a ``service_tier``, an
        ``inference_geo``), moves the charge to that row when the table prices
        it: an unset tier is reserved at the dearest, and settled at the truth.
        """
        row = self._served(self._row(rates), served or {})
        usage = {
            (
                _ALIASES[name]
                if name in _ALIASES and name not in row.per_token
                else name
            ): count
            for name, count in usage.items()
        }
        prompt = sum(
            count
            for name, count in usage.items()
            if name in INPUT_CLASSES
            and _count_problem(count) is None
            and isinstance(count, int)
        )
        base = row
        row = self._sized(row, prompt)
        unknown = [
            name
            for name, count in usage.items()
            if name not in row.per_token
            and _count_problem(count) is None
            # A class we know by name, which this row does not price (a declared
            # price with no cached rate, say), is news only if it carries tokens:
            # zero of it costs nothing whatever its price (/code-review, D40). A
            # name we do not know stops the table at any count (§4.8.3, D38).
            and (name not in _KNOWN_CLASSES or count != 0)
        ]
        if unknown:
            with self._lock:
                if self._unrecognised is None:
                    self._unrecognised = _printable(unknown[0])

        for name, count in usage.items():
            problem = _count_problem(count)
            if problem is not None:
                raise ValueError(_not_a_count(f"usage[{_printable(name)!r}]", problem))

        charged: tuple[Decimal, Rates] | None = None
        for twin in self._dated_twins(base, self._now()):
            sized = self._sized(twin, prompt)
            with localcontext(_ARITHMETIC):
                highest = max(sized.per_token.values()) if unknown else Decimal(0)
                cost = Decimal(0)
                for name, count in usage.items():
                    cost += _tokens(count, "a usage count") * sized.per_token.get(
                        name, highest
                    )
            if charged is None or cost > charged[0]:
                charged = (cost, sized)
        assert charged is not None  # noqa: S101 - _dated_twins always returns the row itself
        cost, row = charged
        return Charge(cost=cost, price_table_stale=bool(unknown), rate_key=row.rate_key)

    def _served(self, row: Rates, served: Mapping[str, str]) -> Rates:
        """The row for what the response says it was served at, if the table has one."""
        known = self._choice.get(row.rate_key)
        if known is None or not served:
            return row
        model, choice = known
        moved = {**choice, **{k: v for k, v in served.items() if k in choice}}
        return self._rows.get((model, frozenset(moved.items())), row)

    def _sized(self, row: Rates, prompt_tokens: int, *, margin: bool = False) -> Rates:
        """The long-context row instead, if this prompt is over its threshold.

        With ``margin``, from a twentieth below the line, for a reservation whose
        bound carries a chosen allowance (/code-review, D41).
        """
        long = self._long.get(row.rate_key)
        if long is None:
            return row
        threshold, long_rates = long
        if prompt_tokens > threshold - (threshold // 20 if margin else 0):
            return long_rates
        return row

    def breached(self, rate_key: str) -> None:
        """A call cost more than its reserved worst case: the bound was wrong.

        Every reservation rests on the bound being an upper bound, so the table
        stops admitting calls, exactly as for a class it does not know (§4.8.3).
        This is what stands behind every allowance an adapter chose rather than
        read (D41's tools allowance, D42's thinking cap): if one is ever too
        small, the first call to show it stops the next (D42).
        """
        model = self._by_key[rate_key].model if rate_key in self._by_key else None
        with self._lock:
            if self._breach is None:
                # A table model id or a policy-declared one: names the operator or
                # this table wrote, safe to print.
                self._breach = model or "a model"

    def _row(self, rates: Rates) -> Rates:
        """This table's own row for ``rates``, which is what gets priced.

        Looked up by key, and the caller's copy is never read for a rate: one
        built by hand could carry any number at all, and a copy of a real one
        (``copy.copy``, ``dataclasses.replace``) must still settle, because a
        settle that raised would lose the charge of a call that happened. A key
        this table never issued is a bug above us and raises.
        """
        row = self._by_key.get(rates.rate_key)
        if row is None:
            raise ValueError(
                "these rates were not resolved by this price table. Pass the Rates "
                "that this table's resolve() returned for the call."
            )
        return row

    def _refuse_if_stale(self, model: str) -> None:
        with self._lock:
            unrecognised, breach = self._unrecognised, self._breach
        if breach is not None:
            raise PricingUnknown(
                model=model,
                detail=(
                    f"a call to {breach} cost more than the worst case paveo reserved "
                    f"for it, so an estimate under the ceiling is wrong"
                ),
                remedy=(
                    "report it: the audit log's settle record with "
                    "price_table_stale shows which call. Every call is refused until "
                    "this process restarts, deliberately: a guard that knows its "
                    "estimate is wrong should not keep guessing (D42)."
                ),
            )
        if unrecognised is None:
            return
        raise PricingUnknown(
            model=model,
            detail=(
                f"a response reported a token class this price table does not "
                f"know ({unrecognised}), so table version {self.version} can no "
                f"longer bound a call"
            ),
            remedy=(
                "upgrade paveo to a release whose price table carries that class. "
                "Every call is refused until then, deliberately: a ceiling enforced "
                "with a table known to be incomplete is not a ceiling "
                "(SPEC_V1.md §4.8.3)."
            ),
        )


def _speeds(listing: _Listing) -> dict[str, tuple[str, str]]:
    """The base pair each value of ``speed`` selects, where the page prices it."""
    if listing.fast is None:
        return {"standard": listing.standard}
    return {"standard": listing.standard, "fast": listing.fast}


def _geos(listing: _Listing) -> dict[str, Decimal]:
    """The multiplier each value of ``inference_geo`` applies, where it is priced."""
    if not listing.us_inference:
        return {"global": Decimal(1)}
    return {"global": Decimal(1), "us": _US_INFERENCE}


def _modifier_values(listing: _AnyListing) -> dict[str, tuple[str, ...]]:
    """The values of each price-affecting parameter this listing prices."""
    if isinstance(listing, _GeminiListing):
        priority = ("priority",) if listing.priority is not None else ()
        return {"service_tier": _GEMINI_TIERS + priority}
    if isinstance(listing, _OpenAIListing):
        tiers = (
            _OPENAI_TIERS if listing.fast is not None else ("default", "flex", "auto")
        )
        return {"service_tier": tiers}
    return {"speed": tuple(_speeds(listing)), "inference_geo": tuple(_geos(listing))}


def _rows_for(
    model: str, listing: _AnyListing
) -> list[tuple[dict[str, str], Rates, Rates | None]]:
    """Every (speed, inference_geo) a listing prices, resolved to per-token rates.

    A list, not a generator: a generator yielding inside ``localcontext`` would
    leave the ledger's context installed in its consumer between yields (D35).
    """
    if isinstance(listing, _OpenAIListing):
        return _openai_rows(model, listing)
    if isinstance(listing, _GeminiListing):
        dated = _gemini_rows(model, listing, "current")
        link = listing
        while link.changes is not None:
            when, link = link.changes
            dated += _gemini_rows(model, link, when, dated=True)
        return dated
    geos = _geos(listing)
    geos[_UNSET] = max(geos.values())
    rows: list[tuple[dict[str, str], Rates, Rates | None]] = []
    with localcontext(_ARITHMETIC):
        for speed, (per_million_in, per_million_out) in _speeds(listing).items():
            for geo, geo_multiplier in geos.items():
                base_in = Decimal(per_million_in) / _PER_MILLION * geo_multiplier
                base_out = Decimal(per_million_out) / _PER_MILLION * geo_multiplier
                per_token = {
                    "input": base_in,
                    "cache_read": base_in * Decimal(listing.cache_read),
                    "cache_write_5m": base_in * _CACHE_WRITE_5M,
                    "cache_write_1h": base_in * _CACHE_WRITE_1H,
                    OUTPUT_CLASS: base_out,
                }
                rates = Rates(
                    model=model,
                    rate_key=f"{model}|speed={speed}|inference_geo={geo}",
                    per_token=MappingProxyType(per_token),
                )
                rows.append(({"speed": speed, "inference_geo": geo}, rates, None))
    return rows


def _openai_rows(
    model: str, listing: _OpenAIListing
) -> list[tuple[dict[str, str], Rates, Rates | None]]:
    """Every ``service_tier`` a listing prices, each with its long-context row."""
    dearest = listing.fast or listing.standard
    triples = {
        "default": listing.standard,
        "flex": listing.standard,
        "auto": dearest,
        _UNSET: dearest,
    }
    if listing.fast is not None:
        triples["fast"] = triples["priority"] = listing.fast
    uplift = _REGIONAL_UPLIFT if listing.regional else Decimal(1)
    suffix = "|regional_uplift" if listing.regional else ""
    rows: list[tuple[dict[str, str], Rates, Rates | None]] = []
    with localcontext(_ARITHMETIC):
        for tier, (per_in, per_cached, per_out) in triples.items():
            base_in = Decimal(per_in) / _PER_MILLION * uplift
            per_token = {
                "input": base_in,
                "cache_read": Decimal(per_cached) / _PER_MILLION * uplift,
                "cache_write": base_in
                * (_OPENAI_CACHE_WRITE if listing.cache_write else 1),
                OUTPUT_CLASS: Decimal(per_out) / _PER_MILLION * uplift,
            }
            key = f"{model}|service_tier={tier}{suffix}"
            rates = Rates(
                model=model, rate_key=key, per_token=MappingProxyType(per_token)
            )
            long_rates = (
                _long_row(model, key, per_token) if listing.long_context else None
            )
            rows.append(({"service_tier": tier}, rates, long_rates))
    return rows


def _declared_row(model: str, price: DeclaredPrice) -> Rates:
    """The one row a policy-declared price makes (§4.8.4).

    Only the classes the operator priced: input, output, and cached input if
    they gave it. A response reporting any other class, a cache write for
    instance, is a class this row does not know and stops the table (§4.8.3),
    because the operator has not told us what it costs.
    """
    with localcontext(_ARITHMETIC):
        per_token = {
            "input": price.input_per_mtok / _PER_MILLION,
            OUTPUT_CLASS: price.output_per_mtok / _PER_MILLION,
        }
        if price.cached_input_per_mtok is not None:
            per_token["cache_read"] = price.cached_input_per_mtok / _PER_MILLION
    return Rates(
        model=model,
        rate_key=f"{model}|declared-by-policy",
        per_token=MappingProxyType(per_token),
    )


def _tokens(count: object, where: str) -> Decimal:
    """``count`` as a ``Decimal``, or ``ValueError`` if no correct caller made it."""
    problem = _count_problem(count)
    if problem is None and isinstance(count, int):
        return Decimal(count)
    raise ValueError(_not_a_count(where, problem or "not a count"))


def _count_problem(count: object) -> str | None:
    """What is wrong with a token count, or ``None`` if nothing is (D24).

    ``bool`` is refused although it is an ``int``: ``True`` tokens is a bug above
    us, not a count of one. Describes the value's shape and never the value: it
    came off a request or a response, and neither belongs in an error (locked
    decision #5).
    """
    if isinstance(count, bool):
        return "a bool"
    if not isinstance(count, int):
        return type(count).__name__
    if count < 0:
        return "a negative int"
    return None


def _not_a_count(where: str, problem: str) -> str:
    return (
        f"{where} is {problem}, which is not a token count: it must be a "
        f"non-negative int."
    )


def _printable(name: object) -> str:
    if isinstance(name, str) and _PRINTABLE_CLASS.fullmatch(name):
        return name
    return "<unprintable class name>"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _change_dates(listing: _GeminiListing) -> list[str]:
    dates = []
    while listing.changes is not None:
        when, listing = listing.changes
        date.fromisoformat(when)  # a malformed date fails at import, not at a call
        dates.append(when)
    return dates


def _gemini_rows(
    model: str, listing: _GeminiListing, period: str, *, dated: bool = False
) -> list[tuple[dict[str, str], Rates, Rates | None]]:
    """Every tier a Gemini listing prices for one price period, with its long row."""
    per_cached = listing.standard[1]
    tiers: dict[str, tuple[str, str | None, str]] = {
        "standard": listing.standard,
        "flex": listing.standard,
        "unspecified": listing.standard,
        _UNSET: listing.standard,
    }
    if listing.priority is not None:
        pri_in, pri_out = listing.priority
        tiers["priority"] = (
            pri_in,
            pri_in if per_cached is not None else None,
            pri_out,
        )
    suffix = "" if period == "current" else f"|from={period}"
    rows: list[tuple[dict[str, str], Rates, Rates | None]] = []
    with localcontext(_ARITHMETIC):
        for tier, (t_in, t_cached, t_out) in tiers.items():
            per_token = {
                "input": Decimal(t_in) / _PER_MILLION,
                OUTPUT_CLASS: Decimal(t_out) / _PER_MILLION,
            }
            if t_cached is not None:
                per_token["cache_read"] = Decimal(t_cached) / _PER_MILLION
            key = f"{model}|service_tier={tier}{suffix}"
            rates = Rates(
                model=model, rate_key=key, per_token=MappingProxyType(per_token)
            )
            long_rates = (
                _long_row(model, key, per_token) if listing.long_context else None
            )
            choice = {"service_tier": tier}
            if listing.changes is not None or dated:
                choice["from"] = period
            rows.append((choice, rates, long_rates))
    return rows


def _long_row(model: str, key: str, per_token: Mapping[str, Decimal]) -> Rates:
    """A row's long-context twin: 2x every input class, 1.5x output (D41, D42)."""
    return Rates(
        model=model,
        rate_key=f"{key}|long_context",
        per_token=MappingProxyType(
            {
                name: rate * (_LONG_OUTPUT if name == OUTPUT_CLASS else _LONG_INPUT)
                for name, rate in per_token.items()
            }
        ),
    )
