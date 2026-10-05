"""The MCP guard: `paveo mcp --agent NAME -- <server>` (B6, D80).

It sits between any MCP client and a local stdio server and judges every
``tools/call`` before the server sees it. What is tested here:

- ``judge_line``, pure: what is forwarded, what is answered, what is dropped,
  and that every message it cannot read safely is refused, never passed on;
- ``Gate``, the decision: the same checkpoint the hook uses, with memory for the
  run, the stop file, one record per call and no payload in it;
- the whole command, end to end, with a real server process that records every
  message it receives: a refused call never reaches it.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from conftest import make_clock, records_in
from paveo._mcp import Gate, judge_line
from paveo.errors import ConfigError, PolicyDenied

ROOT = Path(__file__).parent.parent
FAKE_SERVER = Path(__file__).parent / "mcp_fake_server.py"

POLICY = {
    "version": 1,
    "policy_id": "mcp-test",
    "agents": [
        {
            "id": "files",
            "tools": {
                "allow": [
                    {"name": "read_file", "constraints": {"path": {}}},
                    {
                        "name": "write_file",
                        "constraints": {
                            "path": {"not_matches": ["secret"]},
                            "content": {},
                        },
                    },
                    {"name": "ping", "rate": {"calls": 2, "seconds": 3600}},
                    {"name": "long"},
                ]
            },
        }
    ],
}


def call(id_: object, name: object, arguments: object = None) -> bytes:
    params: dict[str, object] = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    message = {"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": params}
    return (json.dumps(message) + "\n").encode()


def allow_all(_tool: str, _arguments: Mapping[str, object]) -> None:
    return None


def refuse(_tool: str, _arguments: Mapping[str, object]) -> None:
    raise PolicyDenied(
        reason="constraint_violated",
        rule="write_file.path.not_matches",
        remedy="the operator's remedy, never shown to the model",
    )


def answer(to_client: bytes | None) -> dict[str, object]:
    assert to_client is not None
    assert to_client.endswith(b"\n")
    assert to_client.count(b"\n") == 1
    parsed = json.loads(to_client)
    assert isinstance(parsed, dict)
    return parsed


# --------------------------------------------------------------------------
# judge_line: pure
# --------------------------------------------------------------------------


def test_messages_that_are_not_tool_calls_pass_unjudged_and_unchanged() -> None:
    for line in (
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n',
        b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n',
        b'{"jsonrpc":"2.0","id":7,"result":{"roots":[]}}\n',
        b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n',
    ):
        to_server, to_client, note = judge_line(line, refuse)
        assert to_client is None
        assert note is None
        assert to_server is not None
        assert json.loads(to_server) == json.loads(line)


def test_a_tool_call_hidden_behind_bare_carriage_returns_never_runs() -> None:
    """The /security-review finding (D80): Python's JSON parser reads a bare \\r
    as whitespace, so this is one notification to us; the MCP Python SDK ends a
    line at it and would run the tools/call in the middle. Forwarded
    re-serialized, it stays one message to every reader."""
    smuggled = (
        b'{"jsonrpc":"2.0","method":"notifications/progress","params":{"x":\r'
        b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":'
        b'{"name":"delete_everything","arguments":{}}}\r}}\n'
    )
    to_server, _, _ = judge_line(smuggled, refuse)

    assert to_server is not None
    assert b"\r" not in to_server
    lines = list(io.TextIOWrapper(io.BytesIO(to_server), encoding="utf-8"))
    assert len(lines) == 1
    assert json.loads(lines[0])["method"] == "notifications/progress"


def test_nothing_forwarded_can_be_split_by_any_line_reader() -> None:
    for line in (
        b'{"jsonrpc":"2.0","method":"n","params":{"a":"\xe2\x80\xa8x\\u000by"}}\n',
        b'[{"jsonrpc":"2.0","id":1,"method":"tools/list"}]\n',
        call(1, "read_file", {"path": "a\u2028b"}),
    ):
        to_server, _, _ = judge_line(line, allow_all)
        assert to_server is not None
        body = to_server[:-1]
        assert all(byte >= 0x20 and byte < 0x7F for byte in body)
        assert len(to_server.decode("ascii").splitlines()) == 1


def test_an_allowed_call_reaches_the_server_exactly_as_judged() -> None:
    seen: list[tuple[str, Mapping[str, object]]] = []

    def record(tool: str, arguments: Mapping[str, object]) -> None:
        seen.append((tool, arguments))

    line = call(3, "read_file", {"path": "a.txt"})
    to_server, to_client, note = judge_line(line, record)

    assert to_client is None
    assert note is None
    assert to_server is not None
    assert to_server.endswith(b"\n")
    assert json.loads(to_server) == json.loads(line)
    assert seen == [("read_file", {"path": "a.txt"})]


def test_a_call_with_no_arguments_is_judged_with_none() -> None:
    seen: list[Mapping[str, object]] = []
    judge_line(call(4, "ping"), lambda _t, a: seen.append(a))
    assert seen == [{}]


def test_a_refused_call_never_reaches_the_server_and_the_model_hears_why() -> None:
    to_server, to_client, _ = judge_line(
        call("req-9", "write_file", {"path": "secret.txt", "content": "x"}), refuse
    )

    assert to_server is None
    reply = answer(to_client)
    assert reply["id"] == "req-9"
    result = reply["result"]
    assert isinstance(result, dict)
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "write_file.path.not_matches" in text
    assert "secret.txt" not in text
    assert "operator's remedy" not in text


def test_a_decision_that_fails_refuses_without_saying_what_was_asked() -> None:
    def broken(_tool: str, _arguments: Mapping[str, object]) -> None:
        raise ConfigError("the audit log could not be written.", remedy="check it.")

    def crashed(_tool: str, _arguments: Mapping[str, object]) -> None:
        raise RuntimeError("boom with secret.txt in it")

    for decide, expected in ((broken, "ConfigError"), (crashed, "RuntimeError")):
        to_server, to_client, note = judge_line(
            call(5, "write_file", {"path": "secret.txt", "content": "x"}), decide
        )
        assert to_server is None
        result = answer(to_client)["result"]
        assert isinstance(result, dict)
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert expected in text
        assert "secret.txt" not in text
        # The operator's remedy goes to stderr, never to the model (D48).
        assert "check it" not in text
        assert note is not None
        assert "secret.txt" not in note


@pytest.mark.parametrize(
    "params",
    [
        {"name": 7},
        {"name": ""},
        {"name": "read_file", "arguments": ["a"]},
        {"arguments": {}},
        "read_file",
    ],
)
def test_a_tool_call_it_cannot_read_is_answered_as_an_error(params: object) -> None:
    line = json.dumps(
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": params}
    ).encode()
    to_server, to_client, _ = judge_line(line + b"\n", allow_all)

    assert to_server is None
    reply = answer(to_client)
    assert reply["id"] == 6
    assert "error" in reply


@pytest.mark.parametrize(
    "line",
    [
        b"not json at all\n",
        b"\xff\xfe{}\n",
        b'{"jsonrpc":"2.0","method":"tools/call","params":{"name":"read_file"}}\n',
        b'{"jsonrpc":"2.0","id":true,"method":"tools/call","params":{"name":"x"}}\n',
        # The same key twice: one parser keeps the first, another the last.
        b'{"jsonrpc":"2.0","id":1,"method":"tools/list","method":"tools/call","params":{"name":"write_file"}}\n',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"write_file","arguments":{"path":"ok","path":"secret"}}}\n',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"read_file","arguments":{"n":NaN}}}\n',
    ],
)
def test_what_cannot_be_read_safely_is_dropped_and_said_so(line: bytes) -> None:
    to_server, to_client, note = judge_line(line, allow_all)

    assert to_server is None
    assert to_client is None
    assert note is not None
    assert "secret" not in note
    assert "write_file" not in note


def test_a_number_too_large_for_a_float_is_dropped_not_sent_as_infinity() -> None:
    to_server, to_client, note = judge_line(
        call(1, "read_file", {"n": 1e999}), allow_all
    )
    assert to_server is None
    assert to_client is None
    assert note is not None


def test_a_batch_with_a_tool_call_and_nothing_to_answer_is_dropped_and_said_so() -> (
    None
):
    batch = [{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "x"}}]
    to_server, to_client, note = judge_line(
        (json.dumps(batch) + "\n").encode(), allow_all
    )
    assert to_server is None
    assert to_client is None
    assert note is not None


@pytest.mark.parametrize(
    "line",
    [
        # Go's decoder (mcp-go) matches these keys ignoring case: each would be a
        # tools/call there that Paveo never judged, or judged as another tool.
        b'{"jsonrpc":"2.0","id":1,"Method":"tools/call","params":{"name":"evil","arguments":{}}}\n',
        b'{"jsonrpc":"2.0","id":1,"method":"ping","METHOD":"tools/call","params":{"name":"evil"}}\n',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"safe","Name":"evil","arguments":{},"Arguments":{"x":1}}}\n',
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","param\u017f":{"name":"evil"}}\n'.encode(),
        b'[{"jsonrpc":"2.0","id":1,"Method":"tools/call","params":{"name":"evil"}}]\n',
        '{"jsonrpc":"2.0","id":1,"\u0131d":2,"method":"tools/list"}\n'.encode(),
    ],
)
def test_keys_that_differ_only_in_case_are_dropped(line: bytes) -> None:
    seen: list[str] = []
    to_server, to_client, note = judge_line(line, lambda t, _a: seen.append(t))
    assert to_server is None
    assert to_client is None
    assert note is not None
    assert seen == []


def test_arguments_may_use_any_case_they_like() -> None:
    seen: list[Mapping[str, object]] = []
    line = call(1, "add_user", {"Name": "Bob", "ID": 7})
    to_server, _, _ = judge_line(line, lambda _t, a: seen.append(a))
    assert to_server is not None
    assert seen == [{"Name": "Bob", "ID": 7}]


def test_two_arguments_that_differ_only_in_case_are_refused() -> None:
    """A Go server decoding into typed fields reads BRANCH where Paveo judged
    branch: a `requires` with `same: [branch]` would compare the wrong value."""
    seen: list[str] = []
    for arguments in (
        {"branch": "feature", "BRANCH": "main"},
        {"task": "build", "ta\u017fk": "deploy"},
        {"image": "safe", "\u0131mage": "evil"},
        {"image": "safe", "\u0130mage": "evil"},
        {"kind": "safe", "\u212aind": "evil"},
    ):
        to_server, to_client, _ = judge_line(
            call(9, "deploy", arguments), lambda t, _a: seen.append(t)
        )
        assert to_server is None
        assert "error" in answer(to_client)
    assert seen == []


def test_an_oversized_message_is_dropped() -> None:
    line = call(1, "read_file", {"path": "x" * (65 << 20)})
    to_server, to_client, note = judge_line(line, allow_all)
    assert to_server is None
    assert to_client is None
    assert note is not None


def test_a_batch_holding_a_tool_call_is_refused_whole() -> None:
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "x"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    to_server, to_client, _ = judge_line((json.dumps(batch) + "\n").encode(), allow_all)

    assert to_server is None
    assert to_client is not None
    replies = json.loads(to_client)
    assert [reply["id"] for reply in replies] == [1, 2]
    assert all("error" in reply for reply in replies)


def test_a_batch_with_no_tool_call_passes_unjudged() -> None:
    line = b'[{"jsonrpc":"2.0","id":1,"method":"tools/list"}]\n'
    to_server, to_client, _ = judge_line(line, refuse)
    assert to_client is None
    assert to_server is not None
    assert json.loads(to_server) == json.loads(line)


# --------------------------------------------------------------------------
# Gate: the decision
# --------------------------------------------------------------------------


def seat(tmp_path: Path, policy: object = POLICY) -> Path:
    home = tmp_path / ".paveo"
    home.mkdir()
    (home / "policy.json").write_text(json.dumps(policy), "utf-8")
    return home


def gate(home: Path, stopped: Callable[[], bool] = lambda: False) -> Gate:
    return Gate(
        home,
        agent="files",
        principal="run-1",
        salt=b"s",
        now=make_clock(),
        stopped=stopped,
    )


def test_the_gate_admits_refuses_and_records_once_each(tmp_path: Path) -> None:
    home = seat(tmp_path)
    with gate(home) as decide:
        decide("read_file", {"path": "a.txt"})
        with pytest.raises(PolicyDenied):
            decide("write_file", {"path": "secret.txt", "content": "x"})

    records = records_in(home / "audit.jsonl")
    assert [r["decision"] for r in records] == ["allow", "deny"]
    assert all(r["principal"] == "run-1" for r in records)
    assert "secret.txt" not in (home / "audit.jsonl").read_text("utf-8")


def test_the_gate_remembers_the_run_for_rate_rules(tmp_path: Path) -> None:
    home = seat(tmp_path)
    with gate(home) as decide:
        decide("ping", {})
        decide("ping", {})
        with pytest.raises(PolicyDenied) as refused:
            decide("ping", {})
    assert refused.value.reason == "rate_limited"


def test_paveo_stop_reaches_a_running_gate(tmp_path: Path) -> None:
    home = seat(tmp_path)
    stop = [False]
    with gate(home, stopped=lambda: stop[0]) as decide:
        decide("read_file", {"path": "a.txt"})
        stop[0] = True
        with pytest.raises(PolicyDenied) as refused:
            decide("read_file", {"path": "a.txt"})
        stop[0] = False
        decide("read_file", {"path": "a.txt"})
    assert refused.value.reason == "stopped"


def test_a_policy_that_will_not_load_stops_the_gate_before_it_starts(
    tmp_path: Path,
) -> None:
    home = seat(tmp_path, policy={"version": 1})
    with pytest.raises(ConfigError):
        gate(home)


# --------------------------------------------------------------------------
# The command, end to end, with a real server process
# --------------------------------------------------------------------------


def run_mcp(
    home: Path, record: Path, messages: list[bytes], *, server_exit: int = 0
) -> subprocess.CompletedProcess[bytes]:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(  # noqa: S603 - our own interpreter and test server
        [
            sys.executable,
            "-m",
            "paveo.cli",
            "mcp",
            "--agent",
            "files",
            "--dir",
            str(home),
            "--",
            sys.executable,
            str(FAKE_SERVER),
            str(record),
            str(server_exit),
        ],
        input=b"".join(messages),
        capture_output=True,
        env=environment,
        timeout=60,
        check=False,
    )


def test_a_refused_call_never_reaches_the_server(tmp_path: Path) -> None:
    home = seat(tmp_path)
    record = tmp_path / "server-saw.jsonl"
    messages = [
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n',
        call(2, "read_file", {"path": "a.txt"}),
        call(3, "write_file", {"path": "secret.txt", "content": "x"}),
        b'{"jsonrpc":"2.0","id":4,"method":"tools/list"}\n',
    ]

    done = run_mcp(home, record, messages, server_exit=3)

    saw = [json.loads(line) for line in record.read_bytes().splitlines()]
    assert [m.get("id") for m in saw] == [1, 2, 4]
    assert all("secret.txt" not in json.dumps(m) for m in saw)

    replies = {
        reply["id"]: reply for reply in map(json.loads, done.stdout.splitlines())
    }
    assert set(replies) == {1, 2, 3, 4}
    assert replies[2]["result"]["content"][0]["text"] == "ran read_file"
    assert replies[3]["result"]["isError"] is True
    assert done.returncode == 3

    assert [r["decision"] for r in records_in(home / "audit.jsonl")] == [
        "allow",
        "deny",
    ]


def test_the_command_refuses_to_start_without_a_policy(tmp_path: Path) -> None:
    home = tmp_path / ".paveo"
    record = tmp_path / "server-saw.jsonl"

    done = run_mcp(home, record, [call(1, "read_file", {"path": "a.txt"})])

    assert done.returncode == 1
    assert b"paveo" in done.stderr
    assert not record.exists()


def test_the_command_says_when_the_server_cannot_be_started(tmp_path: Path) -> None:
    home = seat(tmp_path)
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run(  # noqa: S603 - our own interpreter, a path we made
        [
            sys.executable,
            "-m",
            "paveo.cli",
            "mcp",
            "--agent",
            "files",
            "--dir",
            str(home),
            "--",
            str(tmp_path / "no-such-server"),
        ],
        input=b"",
        capture_output=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert done.returncode == 1
    assert b"could not be started" in done.stderr


def test_the_relay_writes_only_whole_lines_to_the_client() -> None:
    from paveo._mcp import read_message  # noqa: PLC0415

    stream = io.BytesIO(b"x" * 100 + b"\nnext\n")
    assert read_message(stream, limit=10) is None
    assert read_message(stream, limit=10) == b"next\n"
    assert read_message(stream, limit=10) == b""


def test_refusals_never_land_inside_a_long_answer(tmp_path: Path) -> None:
    """Long answers from the server and refusals from Paveo go to the client at
    the same time, from two threads; every line the client reads must be whole."""
    home = seat(tmp_path)
    messages = []
    for n in range(40):
        messages.append(call(2 * n, "long", {}))
        messages.append(
            call(2 * n + 1, "write_file", {"path": "secret", "content": ""})
        )

    done = run_mcp(home, tmp_path / "saw.jsonl", messages)

    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert sorted(reply["id"] for reply in replies) == list(range(80))


def test_a_server_that_outlives_its_input_is_stopped_with_the_guard(
    tmp_path: Path,
) -> None:
    """MCP clients end a server by closing its input, then SIGTERM. Sent to
    paveo, it must reach the server paveo started, never orphan it."""
    home = seat(tmp_path)
    record = tmp_path / "saw.jsonl"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    guard = subprocess.Popen(  # noqa: S603 - our own interpreter and test server
        [
            sys.executable,
            "-m",
            "paveo.cli",
            "mcp",
            "--agent",
            "files",
            "--dir",
            str(home),
            "--",
            sys.executable,
            str(FAKE_SERVER),
            str(record),
            "0",
            "linger",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=environment,
    )
    assert guard.stdin is not None
    assert guard.stdout is not None
    guard.stdin.write(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n')
    guard.stdin.flush()
    assert guard.stdout.readline()
    guard.stdin.close()

    guard.terminate()
    guard.wait(timeout=30)

    assert Path(str(record) + ".stopped").read_text("utf-8") == "stopped\n"


def test_stop_server_ends_a_server_that_ignores_its_input_without_waiting() -> None:
    import time  # noqa: PLC0415

    from paveo._mcp import stop_server  # noqa: PLC0415

    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
    )
    started = time.monotonic()
    stop_server(sleeper, grace=2.0, patient=False)

    assert sleeper.poll() is not None
    assert time.monotonic() - started < 2.0


def test_a_patient_stop_lets_a_server_that_exits_finish_on_its_own() -> None:
    from paveo._mcp import stop_server  # noqa: PLC0415

    quick = subprocess.Popen([sys.executable, "-c", "pass"])
    stop_server(quick, grace=10.0)
    assert quick.returncode == 0
