"""The MCP guard: ``paveo mcp --agent NAME -- <server command>`` (B6, D80).

It sits between an MCP client that starts a local server over stdio (Claude
Desktop, Cursor, Windsurf) and a local stdio MCP server,
and judges every ``tools/call`` before the server sees it. A refused call never
reaches the server; the client gets a tool result with ``isError: true`` whose
text names the rule, which is what MCP gives the model to read (spec
2025-06-18, "Error Handling").

The transport is stdio: JSON-RPC, UTF-8, one message a line, no newline inside
one (spec 2025-06-18, "Transports"). Pipes only, so the package still opens no
socket. The server is the user's own program, started without a shell; what it
does with a call it was allowed is its own (THREAT_MODEL).

**The server receives exactly what was judged.** Every message is forwarded as
this module re-serialized it, never as the client's bytes: one line of ASCII
with every control character escaped. So no difference between our JSON parser
or line framing and the server's can turn a judged message into another one, or
split one message into several. Found by /security-review: the MCP Python SDK
reads stdin in universal-newline mode and ends a line at a bare ``\r``, which
Python's JSON parser treats as whitespace inside one message. A message holding
the same key twice is dropped for the same reason: one parser keeps the first,
another the last.

**Fails closed.** A tool call it cannot read, a decision that fails, a batch
holding a tool call: refused. A line it cannot read at all is dropped, and one
line on stderr says why without repeating it. Everything that is not a tool
call (initialize, tools/list, notifications, answers to the server's own
requests) passes unjudged, re-serialized like the rest: only tool calls are
judged.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, TextIO

from . import enforce
from ._licence import apply, plan_in
from ._memory import Memory
from ._policy_document import load_file
from .audit import AuditLog, utc_now
from .errors import ConfigError, PaveoError, PolicyDenied
from .policy import Policy
from .session import Identity

# The cap on one call, for this guard and the hook alike (cli uses it too).
MAX_MESSAGE_BYTES = 64 << 20

Decide = Callable[[str, Mapping[str, object]], None]
_OVERSIZED = f"dropped a message over {MAX_MESSAGE_BYTES >> 20} MiB"
# How long a server Paveo started gets at each step of being stopped.
STOP_GRACE_S = 5.0

# JSON-RPC's codes for a request it will not run (spec 2.0, section 5.1).
_INVALID_PARAMS = -32602
_INVALID_REQUEST = -32600


class _DuplicateKeyError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise _DuplicateKeyError
    return dict(pairs)


def _no_constants(_name: str) -> object:
    raise ValueError  # NaN and Infinity are not JSON; another parser may differ


def _finite(text: str) -> float:
    # 1e400 parses to inf, which would be sent on as the non-JSON word Infinity:
    # a value the server reads differently from the one that was judged.
    value = float(text)
    if value in {float("inf"), float("-inf")}:
        raise ValueError
    return value


def _line(message: object) -> bytes:
    # ASCII, so a lone surrogate the client sent still encodes; json.dumps escapes
    # every newline inside a string, so the result is one line.
    return json.dumps(message, separators=(",", ":")).encode("ascii") + b"\n"


def _refusal(request_id: object, text: str) -> bytes:
    return _line(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": text}], "isError": True},
        }
    )


def _error(request_id: object, code: int, message: str) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _is_id(value: object) -> bool:
    # JSON-RPC ids are strings or numbers; a bool is neither, though Python agrees.
    return isinstance(value, str | int | float) and not isinstance(value, bool)


# The keys of the JSON-RPC envelope and of a tool call's params. Go's JSON
# decoder, under mcp-go, matches these to its fields ignoring letter case, the
# last of several winning, and Unicode lookalikes count (U+017F folds to "s").
# Python matches exactly. So a key that folds to one of these and is not spelled
# exactly so, or two keys that fold alike, would be read one way here and
# another by the server: dropped (/security-review, D80).
_ENVELOPE = frozenset({"jsonrpc", "id", "method", "params", "name", "arguments"})


def _fold(key: str) -> str:
    """At least as broad as any decoder's case folding: Python's casefold, plus
    the dotless and dotted Turkish i, which Go's decoder may fold to ``I`` and
    casefold does not (a re-review, D80). Broader only refuses more."""
    return key.casefold().replace("\u0131", "i").replace("i\u0307", "i")


def _plain_envelope(message: object) -> bool:
    """Whether a message and its params name their keys only one way. The
    tool's own arguments are checked in ``_judge``, for keys that collide by
    case only: a lone ``{"Name": "Bob"}`` is an ordinary argument."""
    for part in (message, message.get("params") if isinstance(message, dict) else None):
        if not isinstance(part, dict):
            continue
        folded = [_fold(key) for key in part]
        if len(set(folded)) != len(folded):
            return False
        if any(key not in _ENVELOPE and _fold(key) in _ENVELOPE for key in part):
            return False
    return True


def _is_tool_call(message: object) -> bool:
    return isinstance(message, dict) and message.get("method") == "tools/call"


def _parse(body: bytes) -> tuple[object, str | None]:
    """The message, or ``None`` and why it cannot be read safely."""
    try:
        message: object = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_no_constants,
            parse_float=_finite,
        )
    except _DuplicateKeyError:
        return None, "dropped a message that names the same key twice"
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None, "dropped a message that is not JSON"
    parts = message if isinstance(message, list) else [message]
    if not all(map(_plain_envelope, parts)):
        return None, "dropped a message whose keys differ only in letter case"
    return message, None


def _batch(message: list[object]) -> bytes | None:
    """Every request in a batch that holds a tool call, answered with an error."""
    errors = [
        _error(
            each["id"],
            _INVALID_REQUEST,
            "paveo does not pass a batch that holds a tool call; send it alone.",
        )
        for each in message
        if isinstance(each, dict) and "method" in each and _is_id(each.get("id"))
    ]
    return _line(errors) if errors else None


_BATCH_DROPPED = "dropped a batch that holds a tool call and no request to answer"


def _parts(params: object) -> tuple[str, dict[str, object]] | str:
    """A tool call's name and arguments, or why it cannot be judged as written."""
    name = params.get("name") if isinstance(params, dict) else None
    arguments = params.get("arguments", {}) if isinstance(params, dict) else None
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return (
            "paveo refused this tool call: it needs a name and an object of arguments."
        )
    folded = [_fold(key) for key in arguments]
    if len(set(folded)) != len(folded):
        # A server that decodes arguments into typed fields ignoring case (Go's
        # decoder) reads one of these where Paveo judged the other (D80).
        return "paveo refused this tool call: two arguments differ only in case."
    return name, arguments


def _judge(
    request: dict[str, object], decide: Decide
) -> tuple[bytes | None, bytes | None, str | None]:
    """A tool call: forwarded as judged, or refused to the client."""
    request_id = request.get("id")
    if not _is_id(request_id):
        return None, None, "dropped a tool call with no usable id"
    parts = _parts(request.get("params"))
    if isinstance(parts, str):
        return None, _line(_error(request_id, _INVALID_PARAMS, parts)), None
    name, arguments = parts
    try:
        decide(name, arguments)
    except PolicyDenied as refused:
        return None, _refusal(request_id, refused.for_model), None
    except PaveoError as e:
        # The model hears that it failed, never the remedy, which tells whoever
        # reads it how to change the guard (D48); the operator reads it on stderr.
        # Paveo's own errors carry no payload by construction (§8).
        failure = type(e).__name__
        refusal = f"paveo refused this call because it could not decide it ({failure})."
        return None, _refusal(request_id, refusal), f"could not decide a call: {e}"
    except Exception as e:  # any failure refuses; its message may quote the call
        refusal = f"paveo refused this call because it failed ({type(e).__name__})."
        return None, _refusal(request_id, refusal), refusal
    return _line(request), None, None


def judge_line(
    line: bytes, decide: Decide
) -> tuple[bytes | None, bytes | None, str | None]:
    """One line from the client: ``(to_server, to_client, note)``.

    ``note`` is for stderr, when a line is dropped or a decision fails; it never
    repeats the line.
    ``decide`` returns when a call is allowed and raises when it is not.
    """
    if len(line) > MAX_MESSAGE_BYTES:
        return None, None, _OVERSIZED
    body = line.rstrip(b"\r\n")
    if not body.strip():
        return None, None, None
    message, unreadable = _parse(body)
    if unreadable is not None:
        return None, None, unreadable
    if isinstance(message, list) and any(map(_is_tool_call, message)):
        answers = _batch(message)
        return None, answers, None if answers is not None else _BATCH_DROPPED
    if isinstance(message, dict) and _is_tool_call(message):
        return _judge(message, decide)
    # Re-serialized, never the client's bytes: a server that ends a line at a
    # bare \r, as the MCP Python SDK's text reader does, would split the original
    # into messages nobody judged (/security-review, D80).
    return _line(message), None, None


class Gate:
    """The decision for one run of ``paveo mcp``: the hook's checkpoint, kept open.

    The policy is read once, when the run starts, and a policy that will not
    load raises ``ConfigError`` before any server is started. The plan is
    re-read when the date changes, so a lapsed licence applies in a long run.
    ``paveo stop`` is read on every call, through ``stopped``. The run's memory
    (``rate``, ``repeat``, ``requires``) lives in this process, as a library
    ``Session``'s does.

    Not safe to share across threads: the relay judges every call on one.
    """

    def __init__(  # noqa: PLR0913 - keyword-only; clock and randomness injected (Rule 14)
        self,
        directory: Path,
        *,
        agent: str,
        principal: str,
        salt: bytes,
        stopped: Callable[[], bool],
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._directory = directory
        self._document = load_file(directory / "policy.json")
        self._identity = Identity(agent_id=agent, principal=principal)
        self._memory = Memory(salt)
        self._stopped = stopped
        self._clock = utc_now if now is None else now
        self._audit = AuditLog(directory / "audit.jsonl", now=now)
        self._applied: tuple[date, Policy] | None = None

    def _policy(self, today: date) -> Policy:
        if self._applied is None or self._applied[0] != today:
            plan = plan_in(self._directory, today=today)
            self._applied = (today, apply(self._document, plan, today=today))
        return self._applied[1]

    def declares(self) -> bool:
        """Whether the policy names this run's agent at all."""
        return self._identity.agent_id in self._document.agents

    def __call__(self, tool: str, arguments: Mapping[str, object]) -> None:
        clock = self._clock()
        policy = self._policy(clock.date())
        agent = self._identity.agent_id
        recall = self._memory.recall(clock.timestamp())
        admitting, admitted = self._memory.steps(policy, agent, tool, arguments, recall)
        enforce.check_tool(
            policy=policy,
            audit=self._audit,
            identity=self._identity,
            tool=tool,
            arguments=arguments,
            stopped=self._stopped(),
            recall=recall,
            admitting=admitting,
            admitted=admitted,
        )

    def close(self) -> None:
        self._audit.close()

    def __enter__(self) -> Gate:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def read_message(stream: BinaryIO, *, limit: int = MAX_MESSAGE_BYTES) -> bytes | None:
    """The next line, ``b""`` at the end, or ``None`` for one over ``limit``,
    which is read to its end and thrown away so the next line starts clean."""
    line = stream.readline(limit + 1)
    if len(line) <= limit or line.endswith(b"\n"):
        return line
    while True:
        rest = stream.readline(limit)
        if not rest or rest.endswith(b"\n"):
            return None


def stop_server(
    server: subprocess.Popen[bytes],
    *,
    grace: float = STOP_GRACE_S,
    patient: bool = True,
) -> None:
    """Make sure a server Paveo started ends: if ``patient``, time to exit on
    its own first; then a request to stop, then a stop it cannot refuse. Never
    touches its pipes, which a wedged server can leave blocked (/code-review)."""
    first = (None,) if patient else ()
    for step in (*first, server.terminate, server.kill):
        if server.poll() is not None:
            return
        if step is not None:
            step()
        try:
            server.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
        return


def relay(
    server: subprocess.Popen[bytes],
    decide: Decide,
    *,
    client_in: BinaryIO,
    client_out: BinaryIO,
    err: TextIO,
) -> int:
    """Carry messages both ways until the server exits; return its exit code.

    Two threads, one each way. Writes to the client go through one lock, a whole
    line at a time, so a refusal never lands inside a line the server is
    writing. Every call is judged on the client's thread while ``deciding`` is
    held, and it is taken before returning, so the caller closes the log only
    once no decision is half-written.
    """
    to_server, from_server = server.stdin, server.stdout
    if to_server is None or from_server is None:
        raise ConfigError(
            "the server was started without pipes.",
            remedy="start it with stdin and stdout as pipes.",
        )
    writing, deciding = threading.Lock(), threading.Lock()
    ended = threading.Event()

    def write(data: bytes) -> None:
        with writing:
            client_out.write(data)
            client_out.flush()

    def from_client() -> None:
        try:
            while (line := read_message(client_in)) != b"":
                if line is None:
                    err.write(f"paveo: {_OVERSIZED}\n")
                    continue
                with deciding:
                    if ended.is_set():
                        return  # the run is ending: nothing more is judged
                    forward, reply, note = judge_line(line, decide)
                if note:
                    err.write(f"paveo: {note}\n")
                if reply is not None:
                    write(reply)
                if forward is not None:
                    to_server.write(forward)
                    to_server.flush()
        except (BrokenPipeError, ValueError):
            return  # the server has gone; its exit ends the run
        finally:
            # Closed already if the server went first; closing it again is moot.
            with contextlib.suppress(OSError):
                to_server.close()
        # The client is gone. A server that ignores the end of its input would
        # keep the run, and itself, alive forever: it gets its grace, then stops.
        stop_server(server)

    client = threading.Thread(target=from_client, daemon=True)
    client.start()
    try:
        for line in iter(from_server.readline, b""):
            write(line)
    except (BrokenPipeError, ValueError):
        # The client has gone. Stopped, not sent end of input: closing the pipe
        # waits on a write the other thread may have stuck in a wedged server.
        stop_server(server, patient=False)
    finally:
        # However the run ends, a signal included, nothing more is judged.
        with deciding:
            ended.set()
    code = server.wait()
    return code if code >= 0 else 1
