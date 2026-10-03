"""The audit log shared by processes, not only threads (D49).

A Claude Code hook is a new process on every tool call, and matching hooks run
in parallel. Before B2 each process chained onto the tail it read when it
opened, so two of them appending at once forked the chain. These tests start
real processes, because a lock that only holds within one interpreter is exactly
the bug being tested for.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import paveo
from paveo import verify_chain

SOURCE = str(Path(paveo.__file__).parent.parent)

# One child: open the shared log, wait for the go file, append its records.
CHILD = """
import sys, time
from pathlib import Path
from paveo.audit import AuditLog

log, go, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
audit = AuditLog(log)
while not go.exists():
    time.sleep(0.001)
for _ in range(count):
    audit.append({
        "agent_id": "hook", "action": {"kind": "tool", "name": "Bash"},
        "decision": "allow", "reason": None, "rule": None,
    })
audit.close()
"""


def run_children(log: Path, *, processes: int, each: int) -> None:
    go = log.with_name("go")
    env = {**os.environ, "PYTHONPATH": SOURCE}
    children = [
        subprocess.Popen(  # noqa: S603 - our own interpreter, running our own test child
            [sys.executable, "-c", CHILD, str(log), str(go), str(each)], env=env
        )
        for _ in range(processes)
    ]
    go.touch()
    for child in children:
        assert child.wait(timeout=60) == 0


def test_processes_appending_at_once_leave_one_chain_that_verifies(
    tmp_path: Path,
) -> None:
    log = tmp_path / "audit.jsonl"
    run_children(log, processes=8, each=40)

    status = verify_chain(log)
    assert status.ok, status.detail
    assert status.records == 320


def test_a_process_opening_while_others_append_does_not_see_truncation(
    tmp_path: Path,
) -> None:
    """Opening reads the tail and the anchor. Without the lock, another process
    could append between the two reads, and the anchor would appear to be ahead
    of the log: a false "records removed from the end", which refuses to open."""
    log = tmp_path / "audit.jsonl"
    run_children(log, processes=4, each=10)
    for _ in range(5):
        run_children(log, processes=6, each=5)

    status = verify_chain(log)
    assert status.ok, status.detail
    assert status.records == 40 + 5 * 30
