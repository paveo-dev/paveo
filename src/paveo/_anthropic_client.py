"""``wrap_anthropic``: an Anthropic client whose every call is checked first (§2.1, S6).

``s.wrap_anthropic(client)`` returns a stand-in that exposes exactly
``messages.create``, ``messages.stream`` and the same two under ``beta``, where
fast mode lives. Each one asks ``check_llm`` first, makes the real call only if
it is admitted, and records the response's usage, so a caller cannot forget any
of the three steps. **Nothing else is exposed**: an attribute the stand-in does
not offer raises rather than passing through, because a pass-through would let
``messages.batches.create`` and the rest go out unchecked by accident (§2.1's
interception allowlist). Use the unwrapped client for anything Paveo does not
check, knowingly. This guards against accidents, not against the caller: code
that holds the raw client can always call it (THREAT_MODEL.md §3).

**Nothing here imports the Anthropic SDK** (zero dependencies, locked decision
#6 and D6): the client is used through the attributes its documentation names,
and tested against a fake built from the same names (Rule 7).

**A call that raises is charged its worst case**, whatever the error (D44).
Anthropic does not document which failed requests go unbilled, so none is
assumed to (Rule 4, locked decision #7).

**Streaming**: usage arrives in pieces, ``message_start`` then ``message_delta``,
whose counts are cumulative. A stream is charged what it reported once a
``message_delta`` carrying the output count has arrived, and its worst case
otherwise: abandoned, broken, never read, or still open when the session ends.
Nothing here reads a stream further than the caller did, since the rest of a
generation is what the caller stopped paying for.

Nothing here is safe to share across threads or tasks: every object belongs to
one session, which is not either. Wrap the client once per session.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, NoReturn

from .errors import ConfigError

if TYPE_CHECKING:
    from .session import LLMCall, Session

_UNSETTLED = "unsettled_at_block_exit"


class Wrapped:
    """The stand-in ``wrap_anthropic`` returns: ``messages``, and ``beta`` if the
    client has one. Anything else raises.

    Not safe to share across threads or tasks: it belongs to one session.
    """

    def __init__(self, session: Session, client: object) -> None:
        self.messages = _Messages(session, _part(client, "messages"))
        beta = getattr(client, "beta", None)
        if beta is not None:
            self.beta = _Beta(session, beta)

    def __getattr__(self, name: str) -> object:
        raise AttributeError(
            f"a client wrapped by paveo offers messages.create and messages.stream "
            f"only, not {name!r}. Use the unwrapped client for calls paveo does not "
            f"check, knowingly (SPEC_V1.md §2.1)."
        )


class _Beta:
    def __init__(self, session: Session, beta: object) -> None:
        self.messages = _Messages(session, _part(beta, "messages"))

    def __getattr__(self, name: str) -> object:
        raise AttributeError(
            f"a wrapped client's beta offers messages only, not {name!r}."
        )


class _Messages:
    def __init__(self, session: Session, messages: object) -> None:
        self._session = session
        self._messages = messages

    def create(self, **request: object) -> object:
        """``messages.create``, checked first and recorded after.

        A streamed call is handed to the stream, which settles it when it ends.
        ``stream`` is read as the SDK reads it, by truth, not identity.
        """
        call = self._session.check_llm(request, shape="anthropic")
        try:
            response = self._messages.create(**request)  # type: ignore[attr-defined]  # the SDK's documented method
        except BaseException:
            call._close(_UNSETTLED)
            raise
        if inspect.iscoroutine(response):
            _refuse_async(call, response)
        if request.get("stream"):
            return _Stream(call, response)
        with call:
            call.record(_usage(getattr(response, "usage", None)))
        return response

    def stream(self, **request: object) -> _StreamManager:
        """``messages.stream``, checked when entered and settled when it closes."""
        return _StreamManager(self._session, self._messages, request)

    def __getattr__(self, name: str) -> object:
        raise AttributeError(
            f"a wrapped client's messages offers create and stream only, not "
            f"{name!r} (SPEC_V1.md §2.1)."
        )


class _Stream:
    """The raw event stream of ``create(stream=True)``, settled when it ends.

    An iterator, like the SDK's own, plus ``response`` and ``close()``. Nothing
    else of the SDK stream is offered: its client is reachable from it.

    Not safe to share across threads or tasks: it belongs to one session.
    """

    def __init__(self, call: LLMCall, stream: object) -> None:
        self._call = call
        self._stream = stream
        self._usage: dict[str, object] = {}
        self._reported = False
        self._events = self._read()

    def __iter__(self) -> Iterator[object]:
        return self._events

    def __next__(self) -> object:
        return next(self._events)

    @property
    def response(self) -> object:
        """The HTTP response the SDK stream carries, for its headers."""
        return getattr(self._stream, "response", None)

    def close(self) -> None:
        """Stop reading; charged its worst case unless its usage had arrived."""
        try:
            self._settle()
        finally:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> _Stream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _read(self) -> Iterator[object]:
        try:
            for event in self._stream:  # type: ignore[attr-defined]  # the SDK's Stream is iterable
                self._reported |= _collect(self._usage, event)
                yield event
        finally:
            self._settle()

    def _settle(self) -> None:
        if not self._call._open:
            return
        if self._reported:
            self._call.record(self._usage)
        else:
            self._call._close(_UNSETTLED)


class _StreamManager:
    """``messages.stream(...)``: a context manager, like the SDK's own.

    Entering it returns the SDK's own stream, so ``text_stream`` and
    ``get_final_message()`` work as documented. Leaving it reads the usage from
    that stream's ``current_message_snapshot``: every way of reading the stream
    goes through the SDK's own iterator, so the snapshot is the one place the
    counts can be seen without patching the SDK (locked decision #3). The property
    is public but not in the SDK's helpers guide, so **every way it can fail
    charges the worst case** (D44).

    Not safe to share across threads or tasks: it belongs to one session.
    """

    def __init__(
        self, session: Session, messages: object, request: dict[str, object]
    ) -> None:
        self._session = session
        self._messages = messages
        self._request = request
        self._call: LLMCall | None = None
        self._manager: object = None
        self._stream: object = None

    def __enter__(self) -> object:
        call = self._session.check_llm(self._request, shape="anthropic")
        try:
            manager = self._messages.stream(**self._request)  # type: ignore[attr-defined]  # the SDK's documented method
            if not hasattr(manager, "__enter__"):
                _refuse_async(call, manager)
            stream = manager.__enter__()
        except BaseException:
            call._close(_UNSETTLED)
            raise
        self._call, self._manager, self._stream = call, manager, stream
        return stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        call = self._call
        try:
            if call is not None and call._open:
                # An error leaving the block may have come from the stream itself,
                # part-way through folding the event that carries the counts.
                usage = None if exc is not None else _reported(self._stream)
                if usage is None:
                    call._close(_UNSETTLED)
                else:
                    call.record(usage)
        finally:
            if call is not None and call._open:
                call._close(_UNSETTLED)
            self._manager.__exit__(exc_type, exc, traceback)  # type: ignore[attr-defined]  # a context manager


def _refuse_async(call: LLMCall, pending: object) -> NoReturn:
    """An async client in a sync session: nothing was sent, so the hold goes back.

    A coroutine that was never awaited, or a stream manager that was never
    entered, sent nothing: releasing is exact here, not a guess about billing.
    Neither is handed back, so the caller cannot send it later unchecked, and a
    coroutine is closed as well.
    """
    if inspect.iscoroutine(pending):
        pending.close()
    call.release()
    raise ConfigError(
        "an async Anthropic client was wrapped by a sync session.",
        remedy="pass anthropic.Anthropic(), not AsyncAnthropic(), to this session.",
    )


def _reported(stream: object) -> dict[str, object] | None:
    """The usage of a stream whose ``message_delta`` has arrived, else ``None``."""
    try:
        snapshot = getattr(stream, "current_message_snapshot", None)
    except AssertionError:
        # The SDK asserts a snapshot exists, and none does before the first event:
        # nothing has reported, which is exactly the answer.
        return None
    if getattr(snapshot, "stop_reason", None) is None:
        return None
    return _usage(getattr(snapshot, "usage", None))


def _collect(usage: dict[str, object], event: object) -> bool:
    """Fold one stream event's usage into the total; True once it is final.

    ``message_start`` carries the input side on its message, and a provisional
    output count that must never be what is charged, so it is dropped.
    ``message_delta`` carries cumulative counts, and leaves out any that do not
    apply, so a value it omits keeps the one from ``message_start``. Only a delta
    that carries an output count makes the usage final.
    """
    kind = getattr(event, "type", None)
    if kind == "message_start":
        start = _usage(getattr(getattr(event, "message", None), "usage", None))
        start.pop("output_tokens", None)
        usage.update(start)
    elif kind == "message_delta":
        delta = _usage(getattr(event, "usage", None))
        usage.update({k: v for k, v in delta.items() if v is not None})
        return "output_tokens" in usage
    return False


def _usage(obj: object) -> dict[str, object]:
    """A usage object as the plain dict the adapter reads.

    Missing usage becomes ``{}``, which ``settle_llm`` refuses as unreadable and
    charges at the worst case (D44): never a call that cost nothing.
    """
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        data = dump()
        if isinstance(data, Mapping):
            return dict(data)
    raise ConfigError(
        "the response's usage is neither a mapping nor a model with model_dump().",
        remedy="pass an Anthropic SDK client, or one whose responses carry usage.",
    )


def _part(client: object, name: str) -> object:
    part = getattr(client, name, None)
    if part is None:
        raise ConfigError(
            f"the client has no {name!r}.",
            remedy="pass an anthropic.Anthropic() client to wrap_anthropic.",
        )
    return part
