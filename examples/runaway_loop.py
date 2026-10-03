"""An agent stuck in a loop, stopped by a $1.00 ceiling before the breaching call.

Run it::

    make example          # or: PYTHONPATH=src python examples/runaway_loop.py

No network, no API key, no provider SDK, and no money: the "responses" are the
usage an Anthropic response reports, recorded here as plain dicts. Everything
else is the real thing, the policy, the price table, the ledger and the audit
log, which is what makes the refusal at the end mean something.

Each call may cost up to about $0.22 (20,000 output tokens of Claude Sonnet 5 at
the US-inference rate, since `inference_geo` is left unset). Paveo holds that
worst case before each call, charges what the response says it cost, and refuses
the call whose worst case no longer fits. Then it checks its own log.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from paveo import BudgetExceeded, Paveo, verify_chain

POLICY = {
    "version": 1,
    "policy_id": "runaway-loop-example",
    "agents": [
        {
            "id": "research-agent",
            "budget": {"period": "day", "limit_usd": "1.00"},
            "models": {"allow": ["claude-sonnet-5"]},
        }
    ],
}

REQUEST = {
    "model": "claude-sonnet-5",
    "max_tokens": 20_000,
    "messages": [{"role": "user", "content": "Try the search again, harder."}],
}

# What `response.usage.model_dump()` reports for one turn of the loop.
USAGE = {"input_tokens": 3_000, "output_tokens": 18_000}


def main() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        log = Path(workdir) / "audit.jsonl"
        with (
            Paveo.from_policy(POLICY, audit_path=log) as pf,
            pf.session(agent_id="research-agent", principal="user_123") as s,
        ):
            for turn in range(1, 20):
                try:
                    call = s.check_llm(REQUEST, shape="anthropic")
                except BudgetExceeded as refused:
                    print(f"turn {turn:>2}: REFUSED before it was sent")
                    print(f"          {refused}")
                    break
                # ...here the real agent would call the provider...
                charged = call.record(USAGE)
                print(
                    f"turn {turn:>2}: allowed, charged ${charged:.4f}, "
                    f"${s.remaining():.4f} left"
                )

        status = verify_chain(log)
        print(f"\naudit log: {status.records} records, chain intact: {status.ok}")


if __name__ == "__main__":
    main()
