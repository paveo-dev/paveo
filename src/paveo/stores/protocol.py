"""The interface a second budget store would implement (§10.2, D23 #5).

**Not required today, which is why it is four signatures and not a package.**
v1 ships exactly one store and needs no interface to do it. This exists because
§10.2 publishes the gap that store leaves — two processes sharing a ceiling both
think they own it — and the v1 scope answers it in advance: *define the interface,
ship the in-memory implementation only*. Naming the seam costs four signatures
now; finding it later costs the shape of every caller.

It is a ``Protocol`` rather than a base class, so an implementation inherits
nothing and imports nothing from us to satisfy it, and it is **not public**: §3
names the entire public surface and this is not on it. *Trigger for exporting
it: the first user who needs one ceiling shared by two processes — the sidecar
D7 leaves open, not a v1 feature.*

**Not required: an async twin of this Protocol.** A store whose calls never wait
on anything is correct inside an event loop as it stands, which is why v1 has one
store for both worlds rather than two (D33). *Trigger: the first store whose calls
do I/O.* A coroutine must then be able to await that wait instead of blocking the
loop on it, and a coroutine is a different type — no one signature covers both.

**The contract, which is the part worth reading.** Every method is atomic with
respect to every other (§4.3): the check and the hold that follows it cannot be
split by another caller, or four sub-agents each see the same headroom and all
four are admitted. A denial is *raised* and never returned, because a caller
must not be able to forget to check it (§3) — the core returns values, and
converting one into an exception happens in the store and nowhere else (§4.9.1).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from paveo.budget import Reservation
from paveo.policy import BudgetPolicy


class BudgetStore(Protocol):
    """One ceiling per agent, guarded so concurrent callers cannot overspend it.

    **An implementation must say which concurrency world it is safe in, and be
    safe in it** (Rule 16). That is not a note about the implementation: the
    whole reason this interface exists rather than a bare ledger is that
    something has to make concurrent callers safe, and only the implementation
    knows what it used to do that.

    An implementation owns the lock, the clock and the storage. ``budget``
    travels with the calls that need it because the ceiling is declared per agent
    in the policy (§5) and a store does not read policy — which is also what lets
    a future store live somewhere that has never seen the policy file.
    """

    def reserve(
        self, *, agent_id: str, budget: BudgetPolicy, worst_case: Decimal
    ) -> Reservation:
        """Hold ``worst_case`` against the agent's ceiling, before the call.

        Raises ``BudgetExceeded`` if the ceiling would be breached, carrying the
        four numbers that explain it. The hold is the worst case and not an
        estimate: a ceiling that an unlucky long completion can exceed is not a
        ceiling (§4.1).
        """
        ...

    def settle(self, *, agent_id: str, reservation_id: str, actual: Decimal) -> None:
        """Replace the hold with what the call really cost (§4.2).

        Raises ``ValueError`` if that reservation is not outstanding — settling
        one twice would charge the ceiling twice for one call.
        """
        ...

    def release(self, *, agent_id: str, reservation_id: str) -> None:
        """Give back a hold whose call spent nothing (§4.2).

        Runs in a ``finally`` on every exit path, so it takes no clock and no
        amount: a leaked reservation is a denial of service against the
        customer's own agent, and the release must not be able to fail for a
        reason the failed call already caused.
        """
        ...

    def remaining(self, *, agent_id: str, budget: BudgetPolicy) -> Decimal:
        """``limit - spent - reserved`` (D23 #6).

        The only reading that cannot mislead a caller into making a call that is
        about to be refused. Negative if a mispriced call has already breached
        the ceiling, because that is the true answer.
        """
        ...
