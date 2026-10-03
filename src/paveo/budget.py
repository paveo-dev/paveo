"""Reserve → reconcile accounting (``docs/SPEC_V1.md`` §4).

The decision must be made **before** the call and the cost is not known until
**after** it. That gap is the whole difficulty, and it is closed by holding
headroom for the worst case the call could cost, then giving back the difference
once the real number arrives.

**This module is the arithmetic and nothing else.** No lock, no file, no clock of
its own (§4.9.1, D10). The store wraps it in the lock, and nothing else gets its
own copy of the hard part. So the guarantee is proved once, against this file,
with no lock in the picture at all. It is also why one lock is enough for threads
and coroutines alike: nothing here can await, so nothing inside the lock ever
does (D33).

**Nothing here raises a denial.** ``try_reserve`` *returns* a ``BudgetDenial``
where the public surface would raise ``BudgetExceeded``. §3's "every denial
raises" governs the public surface; this core is private, and a property test
wants to inspect thousands of outcomes without wrapping each one in a ``try``.
Turning a value into an exception is the store's job and happens in exactly one
place — the same split ``policy.Denial`` already uses.

It does raise ``ValueError``, for inputs no correct caller produces: a NaN
amount, a naive datetime, a reservation settled twice. Those are bugs in the code
above, not decisions about a call, and a ledger that quietly absorbs them is a
ledger nobody can trust (Rule 15).

**What "cannot overspend" rests on.** Every reservation is admitted only if
``spent + reserved + worst_case`` fits under the ceiling, so the ceiling holds as
long as the call's actual cost really is bounded by the worst case it reserved.
That bound is the pricing table's job (§4.1's ``max()`` over the token classes),
and when the table can no longer promise it the table says so and every
subsequent reserve is denied (§4.8.3). This file assumes the bound and records
the truth if it is ever broken: an ``actual`` larger than its reservation is
charged in full rather than clamped, which pushes ``spent`` past the ceiling and
denies everything after it. A ledger that hides a breach is worse than the
breach.

**Thread safety: none, deliberately.** ``_BudgetCore`` is not safe to share
across threads or tasks and is not meant to be. Do not reach for it directly —
use the store, which owns the lock. Every symbol here is private to the package:
§3 names the entire public surface and none of this is on it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import (
    ROUND_CEILING,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from uuid import uuid4

# Money is quantised to 8 decimal places (§4.6). Eight and not the two a currency
# has, because a single cached input token costs on the order of 1e-8 USD: a
# ledger kept in cents would round every small call to nothing and a ceiling made
# of nothings is not a ceiling.
_QUANTUM = Decimal("0.00000001")

# The ledger never does its arithmetic in the caller's decimal context (D35).
# That context belongs to the calling thread or task, and anything in the
# customer's process can change it. Lower its precision and a sum is rounded
# before it meets the ceiling; switch its traps off and an amount that cannot be
# quantised becomes NaN, which every comparison admits. Both were reproduced
# here. Twenty-eight digits hold every amount `_round_up` accepts, and every sum
# of them below 1e20 USD, exactly.
#
# **Every field is spelled out, deliberately.** A field left out is copied from
# `decimal.DefaultContext`, which the customer's process can change as well:
# /code-review showed a lowered `Emax` there turning a settle into an `Overflow`
# after the hold was already given back, losing the charge. These are Python's
# own defaults, pinned, except `rounding`: `ROUND_CEILING`, so any rounding that
# ever did happen would err against admission.
_ARITHMETIC = Context(
    prec=28,
    rounding=ROUND_CEILING,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)


def _start_of_day(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_hour(moment: datetime) -> datetime:
    return moment.replace(minute=0, second=0, microsecond=0)


# §4.7. `day` resets at 00:00 UTC and `hour` on the hour — UTC rather than local,
# because a ceiling whose reset instant depends on where the server happens to be
# standing is not one anyone can reason about. `session` is the lifetime of the
# context manager, so it has no truncation at all rather than a different one.
_WINDOWS: Mapping[str, Callable[[datetime], datetime] | None] = {
    "day": _start_of_day,
    "hour": _start_of_hour,
    "session": None,
}

_PERIODS = frozenset(_WINDOWS)


@dataclass(frozen=True, slots=True)
class Reservation:
    """Headroom held against the ceiling for one call that is about to be made.

    ``worst_case`` is the amount actually held — already quantised — and not the
    amount that was asked for. ``release`` gives back this exact number.
    """

    reservation_id: str
    worst_case: Decimal


@dataclass(frozen=True, slots=True)
class BudgetDenial:
    """The ceiling would be breached, with the four numbers that explain it.

    Carries no prose. The store turns this into ``BudgetExceeded``, which is
    where the remedy belongs: the wording depends on the period and the agent,
    and the ledger knows about neither.
    """

    limit: Decimal
    spent: Decimal
    reserved: Decimal
    requested: Decimal


@dataclass(frozen=True, slots=True)
class _Outstanding:
    """A reservation the ledger is still holding, and the window it belongs to."""

    worst_case: Decimal
    window: datetime | None


class _BudgetCore:
    """One ceiling, and every reservation outstanding against it.

    Pure: no lock, no I/O, and the time is handed to it rather than read (Rule
    14). That is the only reason §4.7's 00:00 UTC reset is testable at all — a
    test suite cannot wait until midnight, so a ledger that called
    ``datetime.now()`` inline would have its period-boundary logic silently never
    exercised, which is exactly where money bugs live.

    **Not safe to share across threads or tasks.** See the module docstring.
    """

    def __init__(
        self,
        *,
        limit: Decimal,
        period: str,
        new_id: Callable[[], str] | None = None,
    ) -> None:
        if period not in _PERIODS:
            raise ValueError(
                f"period is {period!r}. Use one of: {', '.join(sorted(_PERIODS))}."
            )
        self._limit = _checked_money(limit, "limit")
        self._truncate = _WINDOWS[period]
        self._new_id = _new_reservation_id if new_id is None else new_id
        self._spent = Decimal(0)
        self._reserved = Decimal(0)
        self._window: datetime | None = None
        self._outstanding: dict[str, _Outstanding] = {}

    @property
    def spent(self) -> Decimal:
        """Settled cost in the current window.

        As of the last call that carried a time: a window that has rolled since is
        not reflected until the next operation notices (see ``_roll``).
        """
        return self._spent

    @property
    def reserved(self) -> Decimal:
        """Held by calls still in flight, across every window they were made in."""
        return self._reserved

    def try_reserve(
        self, worst_case: Decimal, now: datetime
    ) -> Reservation | BudgetDenial:
        """Hold ``worst_case`` against the ceiling, or explain why it does not fit.

        Rounds UP before checking (§4.6), so the fraction lost to rounding is held
        against the ceiling rather than given away.
        """
        with localcontext(_ARITHMETIC):
            requested = _round_up(_checked_money(worst_case, "worst_case"))
            self._roll(now)

            if self._spent + self._reserved + requested > self._limit:
                return BudgetDenial(
                    limit=self._limit,
                    spent=self._spent,
                    reserved=self._reserved,
                    requested=requested,
                )

            reservation_id = self._new_id()
            if reservation_id in self._outstanding:
                raise ValueError(
                    f"reservation id {reservation_id!r} is already outstanding. "
                    f"Two reservations sharing an id would settle as one and lose "
                    f"the other's hold on the ceiling. Supply a new_id that does not "
                    f"repeat."
                )
            self._outstanding[reservation_id] = _Outstanding(
                worst_case=requested, window=self._window
            )
            self._reserved += requested
            return Reservation(reservation_id=reservation_id, worst_case=requested)

    def settle(self, reservation_id: str, actual: Decimal, now: datetime) -> None:
        """Replace a reservation with what the call really cost (§4.2).

        ``actual`` is rounded UP, and never recomputed from the reserve-time
        estimate — the response's own per-class counts are the only input to it. An
        ``actual`` larger than the hold is charged in full rather than clamped; see
        the module docstring for why a breach is recorded rather than hidden.

        **A call that outlived its window is not charged to the next one — except
        for the part nobody authorised.** A reservation made at 23:59 was admitted
        against yesterday's ceiling, and charging it to today would spend a ceiling
        it was never checked against. But yesterday only authorised it *up to the
        hold*; anything past that was authorised by nobody, so the excess follows
        the money into whichever window is live. A breach that vanishes at midnight
        is worse than no ledger at all (D25).

        The hold is given back either way, so ``reserved`` still returns to zero.
        """
        with localcontext(_ARITHMETIC):
            charge = _round_up(_checked_money(actual, "actual"))
            self._roll(now)
            outstanding = self._take(reservation_id, "settle")
            if outstanding.window == self._window:
                self._spent += charge
            else:
                self._spent += max(charge - outstanding.worst_case, Decimal(0))

    def release(self, reservation_id: str) -> Decimal:
        """Give back a reservation whose call spent nothing, and say how much.

        Returns the stored amount **verbatim** (D23 #3). §4.6 rounds releases DOWN,
        which governs derived partial releases — v1 has none — and rounding down a
        value already rounded up would strand a sliver in ``reserved`` forever.

        A leaked reservation is a denial of service against the customer's own
        agent, which is a bug of the same severity as an overspend, so this runs
        in a ``finally`` on every exit path (§4.2, §3).
        """
        with localcontext(_ARITHMETIC):
            return self._take(reservation_id, "release").worst_case

    def remaining(self, now: datetime) -> Decimal:
        """``limit - spent - reserved`` (D23 #6).

        The only reading that cannot mislead: subtracting spend alone would report
        headroom that calls already in flight have claimed, and the next thing a
        caller does with that number is make a call that gets refused. Goes
        negative if a mispriced call has already breached the ceiling, because
        that is the true answer.

        Reads the ledger but **writes** to it, because only an operation carrying a
        time can notice the period has turned. It belongs inside the store's lock
        with the rest, not outside it as a read.
        """
        with localcontext(_ARITHMETIC):
            self._roll(now)
            return self._limit - self._spent - self._reserved

    def _take(self, reservation_id: str, action: str) -> _Outstanding:
        outstanding = self._outstanding.pop(reservation_id, None)
        if outstanding is None:
            raise ValueError(
                f"cannot {action} reservation {reservation_id!r}: it is not "
                f"outstanding. It was already settled or released, or it was never "
                f"made. Each reservation unwinds exactly once — settling one twice "
                f"would charge the ceiling twice for one call."
            )
        self._reserved -= outstanding.worst_case
        return outstanding

    def _roll(self, now: datetime) -> None:
        """Start a new window if ``now`` has crossed into one (§4.7).

        Lazily, on the operations that carry a time, because there is no thread to
        do it on the hour and there does not need to be.

        Validates the clock before anything else, and before the ``session``
        early return: a guard that holds on two periods out of three is the kind
        Rule 5 exists to prevent.

        **The window never moves backwards.** A clock stepped back — an NTP
        correction, a VM resumed — would otherwise reset ``spent`` and hand out a
        refilled ceiling, which is a way to overspend that has nothing to do with
        the agent. Refusing instead would make us the outage over a clock
        correction, so the later window simply stands.
        """
        moment = _as_utc(now)
        if self._truncate is None:
            return
        window = self._truncate(moment)
        if self._window is None:
            self._window = window
        elif window > self._window:
            self._window = window
            self._spent = Decimal(0)


def _new_reservation_id() -> str:
    """An id unique across processes and restarts.

    A counter would be shorter and would collide where it matters: D23 #2
    correlates a decision record with its settlement record through this id, and
    two runs appending to one audit log would both start counting at one. ``uuid4``
    reads ``os.urandom``, which is neither a socket nor a clock; Rule 14 is
    satisfied by the injection point above, which is how tests get determinism.
    """
    return uuid4().hex


def _as_utc(moment: datetime) -> datetime:
    """Refuse a naive datetime rather than guess which zone it is in.

    D19 made the same call for the audit log's clock. Reading a naive datetime as
    local time moves §4.7's reset instant by hours, and the ceiling it refills
    early is real money.
    """
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise ValueError(
            "a budget needs an aware datetime; a naive one would be read as local "
            "time and move the period reset. Pass datetime.now(datetime.UTC), or a "
            "clock that returns an aware value."
        )
    return moment.astimezone(UTC)


def _checked_money(value: object, where: str) -> Decimal:
    """Refuse an amount that cannot be accounted.

    Typed ``object`` and narrowed, the way the policy loader treats a document off
    disk. The callers are all typed ``Decimal``, but the public surface above them
    is not type-checked in anyone else's process, and a ``float`` that slipped
    through would otherwise fail with ``AttributeError`` from somewhere inside the
    ledger instead of being refused at its edge.

    NaN is the one that matters. Every comparison against it is False, so
    ``spent + reserved + worst_case > limit`` would be False and the reserve would
    be **admitted** — after which ``reserved`` is NaN forever and the ceiling is
    gone, silently. Caught here because here is the only place it is cheap to
    catch (Rule 5).
    """
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(
            f"{where} is {value!r}, which cannot be accounted: it must be a finite, "
            f"non-negative Decimal amount in USD."
        )
    return value


def _round_up(amount: Decimal) -> Decimal:
    """Quantise to 8 places, in the direction that protects the ceiling (§4.6).

    Safe as ``ROUND_CEILING`` only because ``_checked_money`` has already refused
    negatives, where ceiling rounds towards zero and would under-charge.

    An amount too large to hold eight decimal places is refused rather than
    allowed to escape as a bare ``decimal.InvalidOperation``: the adapter above
    translates a denial and a ``ValueError``, and would pass anything else
    straight through to the customer's agent.
    """
    try:
        return amount.quantize(_QUANTUM, rounding=ROUND_CEILING)
    except InvalidOperation as e:
        raise ValueError(
            f"{amount} is too large to account to eight decimal places. A cost "
            f"this size means a price table or a token count is wrong; no budget "
            f"is denominated in it."
        ) from e
