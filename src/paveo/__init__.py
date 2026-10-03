"""Paveo — admission control for the calls an AI agent makes.

Every tool in this space reports what an agent *already* did. Paveo decides
whether the next call is allowed to leave, using a policy and a budget, and
refuses it if not.

This module is the entire public surface (``docs/SPEC_V1.md`` §3). Anything not
exported here is private and may change without notice.

Paveo opens no network sockets. See ``tests/test_no_egress.py``.
"""

import sys

# The audit log and the guard's memory lock files with POSIX calls that Windows
# lacks. Said here, before any of them is imported, so a Windows user reads why
# instead of a missing-module traceback (found while preparing R1).
if sys.platform == "win32":
    raise ImportError(
        "Paveo runs on macOS and Linux: its audit log needs POSIX file locking, "
        "which Windows does not have. On Windows, run it inside WSL."
    )

from .audit import ChainStatus, verify_chain
from .errors import (
    BudgetExceeded,
    ConfigError,
    PaveoError,
    PolicyDenied,
    PolicyUnavailable,
    PricingUnknown,
)
from .session import Paveo, Session

__all__ = [
    "BudgetExceeded",
    "ChainStatus",
    "ConfigError",
    "Paveo",
    "PaveoError",
    "PolicyDenied",
    "PolicyUnavailable",
    "PricingUnknown",
    "Session",
    "verify_chain",
]
