"""The error taxonomy (``docs/SPEC_V1.md`` §8).

**Every denial raises.** There are no ``(ok, reason)`` tuples, no boolean
returns and no result objects a caller can forget to check. A guard you can
accidentally ignore is not a guard, so the decision is a security one and not a
matter of style.

Two invariants hold for every error here, and both are tested rather than
trusted:

1. **The message says how to fix it.** ``PaveoError`` takes ``remedy`` as a
   required keyword argument, so an error without one does not compile. Spec §8
   asks for this; making it a parameter is how it stops depending on anyone
   remembering (Rule 5).
2. **No message ever contains a payload.** Identities, rule names, model names,
   token counts and costs — never a prompt, a completion or a tool argument.
   Assume every traceback will one day be pasted into a bug report by someone
   who should not have seen customer data (locked decision #5).

Every class here is thread-safe in the only sense that matters: they are
immutable value objects carrying no shared state.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

from paveo.budget import _ARITHMETIC


class PaveoError(Exception):
    """Base class for everything Paveo raises.

    Catching this catches every denial and every configuration failure, which is
    the right granularity for "stop the agent". Catch a subclass to distinguish
    *why*.
    """

    def __init__(self, message: str, *, remedy: str) -> None:
        super().__init__(f"{message} Fix: {remedy}")
        self.remedy = remedy


class ConfigError(PaveoError):
    """The wiring is wrong.

    Covers a policy that cannot be loaded or does not validate, an audit log or a
    clock that cannot be trusted, and a budget store handed a ceiling it cannot
    honour. Widened deliberately to mean "wiring is wrong" rather than earning a
    separate error type, because nobody deliberately catches "I wired this up
    wrong" — it is a bug to fix, not a condition to handle (D10).
    """


class PolicyDenied(PaveoError):
    """The policy does not permit this call.

    ``rule`` names what denied it, in the dotted form the audit log records —
    ``refund.amount_usd.max``. ``reason`` is the stable machine-readable code.
    Neither ever carries the value that failed the check: recording the *rule*
    rather than the *value* is what keeps locked decision #5 true while still
    leaving a denial explainable a year later (§6).
    """

    def __init__(self, *, reason: str, rule: str, remedy: str) -> None:
        super().__init__(f"{rule} is not permitted ({reason}).", remedy=remedy)
        self.reason = reason
        self.rule = rule

    @property
    def for_model(self) -> str:
        """The refusal as text to hand back to the model in place of a tool result.

        ``str(e)`` is written for the operator, and its remedy says how to change
        the policy. Handed to a model, that is an instruction to edit its own
        guard (D47 #7), so this carries the rule and the reason and no remedy
        (D48). Like the error itself, it holds no payload.
        """
        guidance = _GUIDANCE.get(self.reason, "Do not make this call again.")
        return (
            f"Refused by policy: {self.rule} ({self.reason}). {guidance} If the "
            f"task cannot be done within the policy, stop and tell the user; do "
            f"not try to reach the same result another way."
        )


# What a model can do about each refusal a tool call can meet. Every other
# reason, a model call's included, gets the plain "do not make this call again".
_GUIDANCE = {
    "tool_denied": "This tool is not available to you. Do not call it again.",
    "tool_not_allowed": "This tool is not available to you. Do not call it again.",
    "argument_not_permitted": (
        "The call passed an argument this tool does not accept. Call it again "
        "only with the arguments it declares."
    ),
    "argument_missing": (
        "The call left out an argument this tool requires. Call it again with "
        "that argument supplied."
    ),
    "requires_unmet": (
        "This tool may be called only after another step on the same item, and "
        "that step has not been done in this session. Do it first only if the "
        "task calls for it; do not change the values to fit."
    ),
    "memory_unavailable": (
        "This check cannot see what ran before this call, which this tool's rule "
        "needs, so the tool is not available here. Do not call it again."
    ),
    "rate_limited": (
        "This tool has been called as often as it may be for now. Do not retry "
        "it in a loop; if the task needs more, stop and tell the user."
    ),
    "repeated": (
        "A call the policy counts as this one was just made, though some "
        "arguments may differ. Do not assume it succeeded or failed: check its "
        "result before doing anything else, and do not repeat it."
    ),
    "not_comparable": (
        "This call's arguments could not be compared with earlier calls. Call it "
        "again only with plain JSON values."
    ),
    "stopped": (
        "The operator has stopped this agent. Make no further calls; stop and "
        "tell the user."
    ),
    "constraint_violated": (
        "An argument is outside what this rule permits. Correct it only if it "
        "was a mistake; do not split the action into smaller calls to fit."
    ),
}


class BudgetExceeded(PaveoError):
    """This call would breach the ceiling, so it was not made.

    Carries the four numbers needed to explain the refusal without re-deriving
    them: what the ceiling is, what has been spent, what is currently reserved by
    calls still in flight, and what this call asked for.
    """

    def __init__(
        self,
        *,
        limit: Decimal,
        spent: Decimal,
        reserved: Decimal,
        requested: Decimal,
        remedy: str,
    ) -> None:
        # In the ledger's own context, not the caller's (D35): a hostile one
        # would otherwise turn this refusal into a bare `decimal.Inexact`. `:f`
        # writes plain decimals: a zero quantised to eight places would otherwise
        # print as `0E-8`, which reads as an error to a person.
        with localcontext(_ARITHMETIC):
            available = limit - spent - reserved
            message = (
                f"this call needs {requested:f} USD and {available:f} USD of the "
                f"{limit:f} USD ceiling remains "
                f"(spent {spent:f}, reserved by calls in flight {reserved:f})."
            )
        super().__init__(message, remedy=remedy)
        self.limit = limit
        self.spent = spent
        self.reserved = reserved
        self.requested = requested


class PolicyUnavailable(PaveoError):
    """The decision could not be made, so the call was denied.

    This is the fail-closed path (locked decision #4): a budget store that cannot
    be reached, or an audit log that cannot be written. An unlogged decision did
    not happen, so it is not allowed to have happened.
    """


class PricingUnknown(PaveoError):
    """This call cannot be priced, so its cost cannot be bounded.

    Raised for an unknown model, for a price-affecting request parameter set to a
    value the table does not carry, and for every reserve after a response
    reported a token class the table does not know (§4.8.3). Denying is the
    honest end of "cannot overspend": the alternative is claiming a ceiling we
    can no longer enforce.
    """

    def __init__(self, *, model: str, detail: str, remedy: str) -> None:
        super().__init__(f"cannot price a call to {model}: {detail}.", remedy=remedy)
        self.model = model
        self.detail = detail
