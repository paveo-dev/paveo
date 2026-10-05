"""A stdio MCP server for tests: it records every line it receives, then answers.

``python mcp_fake_server.py RECORD [EXIT_CODE] [linger]``. Each line read from
stdin is appended to RECORD before anything else, so a test can prove what the
server saw. It answers ``initialize``, ``tools/list`` and ``tools/call`` (the
tool ``long`` answers with 300 KB of text); notifications get no answer. It
exits with EXIT_CODE when stdin closes, unless ``linger``: then it keeps
running after end of input, as a badly behaved server does, until SIGTERM,
when it writes ``RECORD.stopped`` and exits.
"""

from __future__ import annotations

import io
import json
import signal
import sys
import time


def main() -> int:
    record, code = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    linger = sys.argv[3:] == ["linger"]

    def stopped(_signum: int, _frame: object) -> None:
        with open(record + ".stopped", "w", encoding="utf-8") as marker:
            marker.write("stopped\n")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stopped)
    # Read as the MCP Python SDK reads (src/mcp/server/stdio.py): a text wrapper
    # in universal-newline mode, which also ends a line at a bare \r.
    text = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    for line in text:
        with open(record, "a", encoding="utf-8") as seen:
            seen.write(line if line.endswith("\n") else line + "\n")
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if "id" not in message:
            continue
        method = message.get("method")
        if method == "initialize":
            result: object = {"protocolVersion": "2025-06-18", "capabilities": {}}
        elif method == "tools/list":
            result = {"tools": [{"name": "read_file"}, {"name": "write_file"}]}
        elif method == "tools/call":
            ran = message["params"]["name"]
            text = "x" * 300_000 if ran == "long" else f"ran {ran}"
            result = {"content": [{"type": "text", "text": text}]}
        else:
            result = {}
        answer = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()
    while linger:
        time.sleep(0.05)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
