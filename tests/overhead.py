"""The overhead gate (Rule 17): run ``python -m tests.overhead``.

Measures what Paveo adds to a call, excluding the provider call itself, and
exits non-zero if any figure is over its budget. ``make check`` runs it, so CI
prints the numbers on every push and fails on a regression past the budget.

Three figures, because the product sits in three places:

- ``check_tool``: one tool call judged against a constraint and recorded.
- ``check_llm`` then ``record``: one model call priced, reserved, settled and
  recorded twice. This is the budget arithmetic and the audit chain together.
- the Claude Code hook: a whole ``paveo guard claude-code`` process on the
  starter policy with a ``rate`` rule added to Bash, so the session memory is
  read and written too, the dearest path a call takes. Less a bare interpreter
  starting and stopping: Python's own start-up is not ours to cut; everything
  after it is (imports included).

Each figure is a median, which a noisy CI runner moves far less than a mean or
a tail. The tail is printed, not gated. Reads the real clock on purpose, like
``price_freshness``: it measures this machine today, and ``test_overhead.py``
covers the verdict with fixed figures (Rule 14). Budgets and the data they were
set from: D73.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import paveo
from paveo._mcp import Gate, judge_line

# Microseconds. About four times the median on an Apple M4, and over twice the
# slowest run measured, x86 under emulation (D73): room for a shared CI runner,
# none for a regression that doubles the cost twice. The library's are inside
# Rule 17's 1 ms; the hook's is far inside the 4 s it must decide in.
BUDGET_US = {
    "check_tool": 250.0,
    "check_llm + record": 600.0,
    "hook, less the interpreter": 200_000.0,
    "MCP tool call, judged": 250.0,
}

_LIBRARY_RUNS, _HOOK_RUNS, _WARMUP = 2000, 30, 5

_POLICY = {
    "version": 1,
    "policy_id": "overhead",
    "agents": [
        {
            "id": "bot",
            "budget": {"period": "day", "limit_usd": "1000000.00"},
            "models": {"allow": ["claude-sonnet-5"]},
            "tools": {
                "allow": [
                    {
                        "name": "lookup_order",
                        "constraints": {"order_id": {"matches": "^A-[0-9]+$"}},
                    }
                ]
            },
        }
    ],
}
_REQUEST = {
    "model": "claude-sonnet-5",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "Draft the reply."}],
}
_USAGE = {"input_tokens": 20, "output_tokens": 300}
_HOOK_CALL = json.dumps(
    {
        "session_id": "overhead",
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la", "description": "List files"},
    }
).encode()
# What the console script `init` installs runs, less the script file.
_HOOK = "import sys; from paveo.cli import main; sys.exit(main())"


def over_budget(medians_us: dict[str, float]) -> list[str]:
    """One line per figure over its budget; empty when all are inside."""
    return [
        f"{name}: median {medians_us[name]:,.0f} us, budget {budget:,.0f} us"
        for name, budget in BUDGET_US.items()
        if medians_us[name] > budget
    ]


def _timed(call: Callable[[], object], runs: int) -> list[float]:
    for _ in range(_WARMUP):
        call()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1e6)
    return samples


def _library(directory: Path) -> dict[str, list[float]]:
    audit = directory / "audit.jsonl"
    with (
        paveo.Paveo.from_policy(_POLICY, audit_path=audit) as pf,
        pf.session(agent_id="bot", principal="overhead") as s,
    ):
        return {
            "check_tool": _timed(
                lambda: s.check_tool("lookup_order", {"order_id": "A-1"}),
                _LIBRARY_RUNS,
            ),
            "check_llm + record": _timed(
                lambda: s.check_llm(_REQUEST, shape="anthropic").record(_USAGE),
                _LIBRARY_RUNS,
            ),
            "MCP tool call, judged": _mcp(directory),
        }


def _mcp(directory: Path) -> list[float]:
    """One tools/call line parsed, judged and re-serialized, as `paveo mcp` does."""
    home = directory / "mcp"
    home.mkdir()
    (home / "policy.json").write_text(json.dumps(_POLICY), encoding="utf-8")
    line = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "lookup_order", "arguments": {"order_id": "A-1"}},
            }
        ).encode()
        + b"\n"
    )
    with Gate(
        home, agent="bot", principal="overhead", salt=b"s", stopped=lambda: False
    ) as decide:
        return _timed(lambda: judge_line(line, decide), _LIBRARY_RUNS)


def _run(argv: list[str], stdin: bytes) -> float:
    """Microseconds for one process. Every run must exit 0: a refusal, a crash
    or the deadline is a different path, and timing it would be timing that."""
    start = time.perf_counter()
    done = subprocess.run(  # noqa: S603 - our own interpreter and our own entry point
        argv, input=stdin, capture_output=True, check=False
    )
    elapsed = (time.perf_counter() - start) * 1e6
    if done.returncode != 0:
        raise SystemExit(f"a measured process exited {done.returncode}, not 0")
    return elapsed


def _hook(directory: Path) -> tuple[list[float], list[float]]:
    """The hook's samples and a bare interpreter's, one of each in turn, so
    drift on a shared runner lands on both sides of the subtraction."""
    folder = directory / ".paveo"
    folder.mkdir()
    starter = Path(paveo.__file__).parent / "starters" / "claude-code.json"
    policy = json.loads(starter.read_text(encoding="utf-8"))
    bash = next(r for r in policy["agents"][0]["tools"]["allow"] if r["name"] == "Bash")
    bash["rate"] = {"calls": 10_000, "seconds": 60}
    (folder / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    hook = [sys.executable, "-c", _HOOK, "guard", "claude-code", "--dir", str(folder)]
    bare = [sys.executable, "-c", "pass"]
    for _ in range(_WARMUP):
        _run(hook, _HOOK_CALL)
        _run(bare, b"")
    with_hook, without = [], []
    for _ in range(_HOOK_RUNS):
        with_hook.append(_run(hook, _HOOK_CALL))
        without.append(_run(bare, b""))
    return with_hook, without


def main() -> int:
    """0 if every median is inside its budget, 1 if not."""
    with tempfile.TemporaryDirectory() as scratch:
        samples = _library(Path(scratch))
        with_hook, bare = _hook(Path(scratch))
    medians = {name: statistics.median(runs) for name, runs in samples.items()}
    medians["hook, less the interpreter"] = statistics.median(
        with_hook
    ) - statistics.median(bare)
    print(f"Paveo overhead, Python {sys.version.split()[0]} on {sys.platform}")
    for name, runs in samples.items():
        p99 = statistics.quantiles(runs, n=100)[98]
        print(f"  {name:28} median {medians[name]:8,.0f} us  p99 {p99:8,.0f} us")
    print(
        f"  {'hook, less the interpreter':28} median "
        f"{medians['hook, less the interpreter']:8,.0f} us  "
        f"(whole process {statistics.median(with_hook):,.0f} us)"
    )
    failures = over_budget(medians)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
