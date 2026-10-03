"""``wrap_anthropic``: the checked stand-in for an Anthropic client (S6, D44).

The client is a fake built from the names the SDK documents (Rule 7): no SDK, no
network, no money. Everything behind it is real, the policy, the price table, the
ledger and the audit log, so what these prove is what a customer gets.

Each charge is compared with the same call made through ``check_llm`` by hand,
so a test states what the wrapper must agree with rather than a price.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import make_clock, records_in
from paveo import BudgetExceeded, ConfigError, Paveo, PolicyDenied, Session

POLICY: dict[str, object] = {
    "version": 1,
    "policy_id": "s6",
    "agents": [
        {
            "id": "bot",
            "budget": {"period": "day", "limit_usd": "1.00"},
            "models": {"allow": ["claude-sonnet-5"]},
        }
    ],
}

REQUEST: dict[str, object] = {
    "model": "claude-sonnet-5",
    "max_tokens": 20_000,
    "messages": [{"role": "user", "content": "Draft the reply."}],
}

USAGE: dict[str, object] = {"input_tokens": 100, "output_tokens": 900}


class Usage:
    """Shaped like the SDK's pydantic models: read through ``model_dump()``."""

    def __init__(self, fields: dict[str, object]) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, object]:
        return dict(self._fields)


DELTA: object = {"input_tokens": None, "output_tokens": 900}


def events(*, through_delta: bool = True, delta_usage: object = DELTA) -> list[object]:
    """A raw stream as the API sends it: counts on message_start, then cumulative
    ones on message_delta, which omits the input count it does not repeat."""
    sent: list[object] = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=Usage({"input_tokens": 100, "output_tokens": 1})
            ),
        ),
        SimpleNamespace(type="content_block_delta"),
    ]
    if through_delta:
        sent += [
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=Usage(delta_usage) if isinstance(delta_usage, dict) else None,
            ),
            SimpleNamespace(type="message_stop"),
        ]
    return sent


class RawStream:
    """What ``messages.create(stream=True)`` returns: iterable, closable."""

    def __init__(self, sent: list[object]) -> None:
        self._sent = iter(sent)
        self.closed = False
        self.response = SimpleNamespace(headers={"request-id": "req_1"})

    def __iter__(self) -> Iterator[object]:
        yield from self._sent

    def close(self) -> None:
        self.closed = True


class MessageStream:
    """What ``messages.stream(...)`` enters: iterable, with a running snapshot
    whose property asserts one exists, as the SDK's does."""

    def __init__(self, sent: list[object]) -> None:
        self._sent = iter(sent)
        self._snapshot: SimpleNamespace | None = None

    def __iter__(self) -> Iterator[object]:
        for event in self._sent:
            kind = getattr(event, "type", None)
            if kind == "message_start":
                usage = event.message.usage.model_dump()  # type: ignore[attr-defined]  # the fake's own event
                self._snapshot = SimpleNamespace(stop_reason=None, usage=Usage(usage))
            elif kind == "message_delta" and self._snapshot is not None:
                self._snapshot.stop_reason = event.delta.stop_reason  # type: ignore[attr-defined]  # the fake's own event
                merged = self._snapshot.usage.model_dump()
                merged["output_tokens"] = event.usage.model_dump()["output_tokens"]  # type: ignore[attr-defined]  # the fake's own event
                self._snapshot.usage = Usage(merged)
            yield event

    @property
    def current_message_snapshot(self) -> SimpleNamespace:
        assert self._snapshot is not None
        return self._snapshot


class StreamManager:
    def __init__(self, messages: Messages) -> None:
        self._messages = messages

    def __enter__(self) -> MessageStream:
        if self._messages.error is not None:
            raise self._messages.error
        return MessageStream(self._messages.sent)

    def __exit__(self, *exc: object) -> None:
        self._messages.exited = True


class Messages:
    def __init__(
        self,
        *,
        error: BaseException | None = None,
        usage: object = USAGE,
        through_delta: bool = True,
        delta_usage: object = DELTA,
    ) -> None:
        self.error = error
        self.usage = usage
        self.sent = events(through_delta=through_delta, delta_usage=delta_usage)
        self.requests: list[dict[str, object]] = []
        self.exited = False
        self.raw: RawStream | None = None

    def create(self, **request: object) -> object:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if request.get("stream"):  # the SDK streams on any true value
            self.raw = RawStream(self.sent)
            return self.raw
        usage = Usage(self.usage) if isinstance(self.usage, dict) else self.usage
        return SimpleNamespace(usage=usage, content=[])

    def stream(self, **request: object) -> StreamManager:
        self.requests.append(request)
        return StreamManager(self)


class Client:
    def __init__(self, **messages: object) -> None:
        self.messages = Messages(**messages)  # type: ignore[arg-type]  # test fixture passthrough
        self.beta = SimpleNamespace(messages=Messages(**messages))  # type: ignore[arg-type]  # test fixture passthrough
        self.models = SimpleNamespace(list=lambda: [])


@pytest.fixture
def pf(tmp_path: Path) -> Iterator[Paveo]:
    with Paveo.from_policy(
        POLICY, audit_path=tmp_path / "audit.jsonl", now=make_clock()
    ) as paveo:
        yield paveo


@pytest.fixture
def s(pf: Paveo) -> Iterator[Session]:
    with pf.session(agent_id="bot", principal="user_1") as session:
        yield session


@pytest.fixture
def by_hand(tmp_path: Path) -> Iterator[Session]:
    """A second, separate ledger for the charges the wrapper must agree with."""
    with (
        Paveo.from_policy(POLICY, audit_path=tmp_path / "hand.jsonl") as other,
        other.session(agent_id="bot", principal="user_1") as session,
    ):
        yield session


def recorded(by_hand: Session) -> Decimal:
    return by_hand.check_llm(REQUEST, shape="anthropic").record(USAGE)


STREAMED = {**REQUEST, "stream": True}


def worst_case(by_hand: Session, request: dict[str, object] = REQUEST) -> Decimal:
    before = by_hand.remaining()
    call = by_hand.check_llm(request, shape="anthropic")
    held = before - by_hand.remaining()
    call.release()
    return held


def spent(s: Session) -> Decimal:
    return Decimal("1.00") - s.remaining()


def settle_reasons(tmp_path: Path) -> list[object]:
    return [
        r["reason"]
        for r in records_in(tmp_path / "audit.jsonl")
        if r.get("decision") == "settle"
    ]


# --- messages.create -----------------------------------------------------------


def test_create_is_charged_what_its_response_reported(
    s: Session, by_hand: Session
) -> None:
    client = Client()
    response = s.wrap_anthropic(client).messages.create(**REQUEST)

    assert response.usage.model_dump() == USAGE  # type: ignore[attr-defined]  # the fake's response
    assert client.messages.requests == [REQUEST]
    assert spent(s) == recorded(by_hand)


def test_a_call_the_policy_refuses_never_reaches_the_client(s: Session) -> None:
    client = Client()

    with pytest.raises(PolicyDenied):
        s.wrap_anthropic(client).messages.create(
            **{**REQUEST, "model": "claude-fable-5-1"}
        )

    assert client.messages.requests == []
    assert spent(s) == 0


def test_the_call_that_would_breach_the_ceiling_is_refused_before_it_leaves(
    s: Session,
) -> None:
    client = Client()
    wrapped = s.wrap_anthropic(client)

    for _ in range(1_000):
        try:
            wrapped.messages.create(**REQUEST)
        except BudgetExceeded:
            break

    admitted = len(client.messages.requests)
    assert 0 < admitted < 1_000
    assert s.remaining() >= 0
    with pytest.raises(BudgetExceeded):
        wrapped.messages.create(**REQUEST)
    assert len(client.messages.requests) == admitted


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("read timed out"),
        type("RateLimitError", (Exception,), {"status_code": 429})(),
        type("InternalServerError", (Exception,), {"status_code": 500})(),
    ],
)
def test_a_call_that_raises_is_charged_its_worst_case(
    s: Session, by_hand: Session, error: BaseException
) -> None:
    """Anthropic does not document which failed requests are unbilled, so none is
    assumed to be, a 429 included (D44)."""
    with pytest.raises(type(error)):
        s.wrap_anthropic(Client(error=error)).messages.create(**REQUEST)

    assert spent(s) == worst_case(by_hand)


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 100}])
def test_a_response_without_readable_usage_is_charged_its_worst_case(
    s: Session, by_hand: Session, usage: object
) -> None:
    """Never a call that cost nothing (D44)."""
    with pytest.raises(ConfigError):
        s.wrap_anthropic(Client(usage=usage)).messages.create(**REQUEST)

    assert spent(s) == worst_case(by_hand)


def test_beta_messages_are_checked_too(s: Session, by_hand: Session) -> None:
    client = Client()
    s.wrap_anthropic(client).beta.messages.create(**REQUEST)  # type: ignore[union-attr]  # the fake has a beta

    assert client.beta.messages.requests == [REQUEST]
    assert spent(s) == recorded(by_hand)


# --- messages.create(stream=True) ---------------------------------------------


def test_a_stream_read_to_its_end_is_charged_what_it_reported(
    s: Session, by_hand: Session
) -> None:
    stream = s.wrap_anthropic(Client()).messages.create(**REQUEST, stream=True)
    kinds = [event.type for event in stream]  # type: ignore[attr-defined]  # the wrapper's stream

    assert kinds[0] == "message_start"
    assert kinds[-1] == "message_stop"
    assert spent(s) == recorded(by_hand)


def test_a_stream_is_not_settled_before_it_is_read(s: Session, tmp_path: Path) -> None:
    """Handing the stream back must not settle the call: a regression once."""
    s.wrap_anthropic(Client()).messages.create(**STREAMED)

    assert settle_reasons(tmp_path) == []


def test_a_stream_abandoned_before_its_usage_is_charged_its_worst_case(
    s: Session, by_hand: Session
) -> None:
    client = Client()
    with s.wrap_anthropic(client).messages.create(**REQUEST, stream=True) as stream:  # type: ignore[attr-defined]  # the wrapper's stream
        for _ in stream:
            break

    assert spent(s) == worst_case(by_hand, STREAMED)
    assert client.messages.raw is not None
    assert client.messages.raw.closed


def test_a_stream_left_after_its_usage_arrived_is_charged_what_it_reported(
    s: Session, by_hand: Session
) -> None:
    """Settled when closed, as the SDK's own stream is: a caller that breaks out
    of the loop still holds the iterator, and may read on."""
    stream = s.wrap_anthropic(Client()).messages.create(**REQUEST, stream=True)
    for event in stream:  # type: ignore[attr-defined]  # the wrapper's stream
        if event.type == "message_delta":
            break
    stream.close()  # type: ignore[attr-defined]  # the wrapper's stream

    assert spent(s) == recorded(by_hand)


def test_a_stream_that_ends_without_its_usage_is_charged_its_worst_case(
    s: Session, by_hand: Session
) -> None:
    """A dropped connection: the events stop before message_delta."""
    stream = s.wrap_anthropic(Client(through_delta=False)).messages.create(
        **REQUEST, stream=True
    )
    list(stream)  # type: ignore[call-overload]  # the wrapper's stream

    assert spent(s) == worst_case(by_hand, STREAMED)


def test_a_stream_never_read_is_charged_its_worst_case_when_closed(
    s: Session, by_hand: Session
) -> None:
    s.wrap_anthropic(Client()).messages.create(**REQUEST, stream=True).close()  # type: ignore[attr-defined]  # the wrapper's stream

    assert spent(s) == worst_case(by_hand, STREAMED)


# --- messages.stream -----------------------------------------------------------


def test_messages_stream_read_to_its_end_is_charged_what_it_reported(
    s: Session, by_hand: Session
) -> None:
    client = Client()
    with s.wrap_anthropic(client).messages.stream(**REQUEST) as stream:
        for _ in stream:  # type: ignore[attr-defined]  # the SDK's own stream
            pass

    assert spent(s) == recorded(by_hand)
    assert client.messages.exited


def test_messages_stream_left_early_is_not_read_further(
    s: Session, by_hand: Session
) -> None:
    """The rest of a generation is what the caller stopped paying for: leaving must
    not pull it (the SDK's get_final_message would)."""
    with s.wrap_anthropic(Client()).messages.stream(**REQUEST) as stream:
        events_read = iter(stream)  # type: ignore[call-overload]  # the SDK's own stream
        next(events_read)

    assert next(events_read).type == "content_block_delta"
    assert spent(s) == worst_case(by_hand)


def test_messages_stream_left_before_any_event_is_charged_its_worst_case(
    s: Session, by_hand: Session
) -> None:
    with s.wrap_anthropic(Client()).messages.stream(**REQUEST):
        pass

    assert spent(s) == worst_case(by_hand)


def test_messages_stream_that_raises_inside_is_charged_and_the_error_kept(
    s: Session, by_hand: Session
) -> None:
    client = Client()
    with (
        pytest.raises(KeyError),
        s.wrap_anthropic(client).messages.stream(**REQUEST),
    ):
        raise KeyError("the caller's own bug")

    assert spent(s) == worst_case(by_hand)
    assert client.messages.exited


def test_messages_stream_that_fails_to_open_is_charged_its_worst_case(
    s: Session, by_hand: Session
) -> None:
    with (
        pytest.raises(TimeoutError),
        s.wrap_anthropic(Client(error=TimeoutError())).messages.stream(**REQUEST),
    ):
        pass

    assert spent(s) == worst_case(by_hand)


# --- the allowlist -------------------------------------------------------------


@pytest.mark.parametrize(
    "reach",
    [
        lambda w: w.models,
        lambda w: w.completions,
        lambda w: w.messages.batches,
        lambda w: w.messages.count_tokens,
        lambda w: w.beta.files,
        lambda w: w.beta.messages.batches,
    ],
)
def test_anything_the_wrapper_does_not_check_is_not_reachable(
    s: Session, reach: object
) -> None:
    """A pass-through would let unchecked calls out by accident (§2.1)."""
    wrapped = s.wrap_anthropic(Client())

    with pytest.raises(AttributeError, match="only"):
        reach(wrapped)  # type: ignore[operator]  # each case is a lambda


def test_a_client_without_messages_is_a_wiring_error(s: Session) -> None:
    with pytest.raises(ConfigError, match="messages"):
        s.wrap_anthropic(object())


def test_a_session_that_was_never_entered_cannot_wrap(pf: Paveo) -> None:
    with pytest.raises(ConfigError, match="never entered"):
        pf.session(agent_id="bot", principal="user_1").wrap_anthropic(Client())


def test_every_wrapped_call_leaves_both_records(s: Session, tmp_path: Path) -> None:
    wrapped = s.wrap_anthropic(Client())
    wrapped.messages.create(**REQUEST)
    list(wrapped.messages.create(**REQUEST, stream=True))  # type: ignore[call-overload]  # the wrapper's stream
    with pytest.raises(TimeoutError):
        s.wrap_anthropic(Client(error=TimeoutError())).messages.create(**REQUEST)

    assert settle_reasons(tmp_path) == [
        "recorded",
        "recorded",
        "unsettled_at_block_exit",
    ]


class AsyncMessages:
    """``AsyncAnthropic().messages``: create returns a coroutine, stream a manager
    entered with ``async with``."""

    def __init__(self) -> None:
        self.sent = False
        self.pending: object = None

    async def _send(self) -> object:
        self.sent = True
        return SimpleNamespace(usage=Usage(USAGE))

    def create(self, **_request: object) -> object:
        self.pending = self._send()
        return self.pending

    def stream(self, **_request: object) -> object:
        return SimpleNamespace(__aenter__=None)


def test_an_async_client_in_a_sync_session_is_refused_and_nothing_charged(
    s: Session,
) -> None:
    """The coroutine never ran, so nothing was sent: the hold goes back exactly,
    and it is closed so it cannot be awaited later, unchecked."""
    messages = AsyncMessages()
    client = SimpleNamespace(messages=messages)

    with pytest.raises(ConfigError, match="async"):
        s.wrap_anthropic(client).messages.create(**REQUEST)
    with pytest.raises(ConfigError, match="async"):
        s.wrap_anthropic(client).messages.stream(**REQUEST).__enter__()

    assert s.remaining() == Decimal("1.00")
    assert not messages.sent
    with pytest.raises(RuntimeError, match="cannot reuse already awaited coroutine"):
        messages.pending.send(None)  # type: ignore[attr-defined]  # a coroutine


# --- found by both reviews, 2026-09-24 (D44) ------------------------------------


@pytest.mark.parametrize("delta_usage", [None, {}, {"output_tokens": None}])
def test_a_delta_without_its_output_count_is_not_settled_on_the_provisional_one(
    s: Session, by_hand: Session, delta_usage: object
) -> None:
    """message_start carries output_tokens=1 as a placeholder. A stream whose
    delta brings no final count (a proxy on base_url, a regression) was charged
    that 1 token: $0.0002 of a $0.22 call (/security-review)."""
    stream = s.wrap_anthropic(Client(delta_usage=delta_usage)).messages.create(
        **STREAMED
    )
    list(stream)  # type: ignore[call-overload]  # the wrapper's stream

    assert spent(s) == worst_case(by_hand, STREAMED)


def test_messages_stream_left_by_an_error_is_charged_its_worst_case_even_if_read(
    s: Session, by_hand: Session
) -> None:
    """The SDK sets stop_reason before it reads the delta's counts, so an error out
    of the stream can leave a snapshot that looks final and is not
    (/security-review)."""

    def read_then_fail() -> None:
        with s.wrap_anthropic(Client()).messages.stream(**REQUEST) as stream:
            for _ in stream:  # type: ignore[attr-defined]  # the SDK's own stream
                pass
            raise KeyError("raised after the last event")

    with pytest.raises(KeyError):
        read_then_fail()

    assert spent(s) == worst_case(by_hand)


def test_messages_stream_without_a_snapshot_is_charged_its_worst_case(
    s: Session, by_hand: Session
) -> None:
    """The snapshot property is not in the SDK's helpers guide: if it goes, the
    call is charged in full, never at nothing (/code-review)."""
    client = Client()
    client.messages.stream = lambda **_: SimpleNamespace(  # type: ignore[method-assign]  # a stream with no snapshot
        __enter__=lambda: iter(()), __exit__=lambda *_: None
    )
    manager = s.wrap_anthropic(client).messages.stream(**REQUEST)
    manager.__enter__()
    manager.__exit__(None, None, None)

    assert spent(s) == worst_case(by_hand)


def test_a_true_stream_value_that_is_not_true_itself_is_still_a_stream(
    s: Session, by_hand: Session
) -> None:
    """The SDK streams on any true value; so must the wrapper (/code-review)."""
    request = {**REQUEST, "stream": 1}
    list(s.wrap_anthropic(Client()).messages.create(**request))  # type: ignore[call-overload]  # the wrapper's stream

    assert spent(s) == recorded(by_hand)


def test_the_raw_stream_is_an_iterator_with_its_response(s: Session) -> None:
    """next() and response.headers work, as on the SDK's own (/code-review)."""
    stream = s.wrap_anthropic(Client()).messages.create(**STREAMED)

    assert next(stream).type == "message_start"  # type: ignore[call-overload]  # the wrapper's stream
    assert stream.response.headers["request-id"] == "req_1"  # type: ignore[attr-defined]  # the wrapper's stream
    stream.close()  # type: ignore[attr-defined]  # the wrapper's stream


def test_a_client_without_beta_explains_itself(s: Session) -> None:
    wrapped = s.wrap_anthropic(SimpleNamespace(messages=Messages()))

    with pytest.raises(AttributeError, match="only"):
        wrapped.beta  # noqa: B018 - the access is the test
