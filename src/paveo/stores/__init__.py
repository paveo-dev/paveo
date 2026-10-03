"""Where a ceiling's numbers live, and what guards them.

``_BudgetCore`` (``paveo.budget``) is the arithmetic and holds no lock, so that
it can be proved once with no lock in the picture (§4.9.1). A store is the object
that makes the core safe to share: one lock around every call into it, correct in
threads and in an event loop alike because nothing inside it awaits (D33), one
ledger per agent, and the clock the core is not allowed to read for itself
(Rule 14).

v1 ships one store and it keeps its ledgers in this process. Two processes
sharing a ceiling still both think they own it — §10.2, published rather than
hidden.
"""

from typing import TYPE_CHECKING

from .memory import InMemoryBudgetStore
from .protocol import BudgetStore

if TYPE_CHECKING:
    # Structural typing checks nothing until someone assigns one to the other,
    # and nothing does until S4 hands a store to `Paveo`. So the check is made
    # here, where the verification gate already runs: an implementation that
    # drifts from the interface fails `mypy --strict` in the session that drifts
    # it, rather than in the session that finally wires it up (Rule 5).
    _conforms: type[BudgetStore] = InMemoryBudgetStore

__all__ = ["BudgetStore", "InMemoryBudgetStore"]
