"""The audit log: the chain holds, the file is private, and nothing else gets in.

Three things are load-bearing here and each has a test rather than a convention:
the chain detects tampering *and says where*, the log refuses any field §6 does
not define, and an unwritable log raises instead of quietly carrying on.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import stat
import threading
from datetime import datetime
from pathlib import Path

import pytest

from conftest import SENTINEL, make_clock, records_in
from paveo.audit import (
    AuditLog,
    _write_all,
    anchor_path,
    read_anchor,
    verify_chain,
)
from paveo.errors import ConfigError, PolicyUnavailable


def tool_record(
    decision: str = "allow", *, agent_id: str = "refund-bot", **extra: object
) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "principal": "user_123",
        "action": {"kind": "tool", "name": "refund"},
        "decision": decision,
        "policy_id": "prod-2026-09",
        "policy_hash": "sha256:" + "ab" * 32,
        **extra,
    }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_a_record_carries_the_fields_the_spec_defines(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", now=make_clock())
    log.append(tool_record("deny", reason="constraint_violated", rule="refund.a.max"))
    log.close()

    (record,) = records_in(tmp_path / "audit.jsonl")
    assert record["v"] == 1
    assert record["seq"] == 1
    assert record["ts"] == "2026-09-20T11:02:04.881Z"
    assert record["agent_id"] == "refund-bot"
    assert record["principal"] == "user_123"
    assert record["action"] == {"kind": "tool", "name": "refund"}
    assert record["decision"] == "deny"
    assert record["rule"] == "refund.a.max"
    assert record["prev_hash"] == "sha256:" + "0" * 64
    assert str(record["hash"]).startswith("sha256:")


def test_sequence_numbers_are_consecutive(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", now=make_clock())
    for _ in range(5):
        log.append(tool_record())
    log.close()

    assert [r["seq"] for r in records_in(tmp_path / "audit.jsonl")] == [1, 2, 3, 4, 5]


def test_each_record_is_chained_to_the_one_before(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", now=make_clock())
    for _ in range(3):
        log.append(tool_record())
    log.close()

    records = records_in(tmp_path / "audit.jsonl")
    for earlier, later in itertools.pairwise(records):
        assert later["prev_hash"] == earlier["hash"]


def test_the_log_file_is_private(tmp_path: Path) -> None:
    location = tmp_path / "audit.jsonl"
    AuditLog(location, now=make_clock()).close()
    assert stat.S_IMODE(location.stat().st_mode) == 0o600


def test_an_existing_log_is_made_private(tmp_path: Path) -> None:
    """A file that already exists keeps its old mode unless we change it."""
    location = tmp_path / "audit.jsonl"
    location.touch(mode=0o644)
    os.chmod(location, 0o644)
    AuditLog(location, now=make_clock()).close()
    assert stat.S_IMODE(location.stat().st_mode) == 0o600


def test_a_world_writable_directory_is_refused(tmp_path: Path) -> None:
    exposed = tmp_path / "exposed"
    exposed.mkdir()
    # The thing under test is that a world-writable log directory is refused, so
    # the test has to make one. S103 is the linter correctly objecting to exactly
    # the situation this asserts we reject.
    os.chmod(exposed, 0o777)  # noqa: S103
    with pytest.raises(ConfigError, match="world-writable"):
        AuditLog(exposed / "audit.jsonl")


def test_an_unwritable_log_denies_rather_than_carrying_on(tmp_path: Path) -> None:
    """An unlogged decision did not happen (§7)."""
    log = AuditLog(tmp_path / "audit.jsonl", now=make_clock())
    log.close()  # the handle is now closed, so the next write fails
    with pytest.raises(PolicyUnavailable, match="could not be written"):
        log.append(tool_record())


def test_a_log_moved_while_open_is_not_appended_to(tmp_path: Path) -> None:
    """The tail is read by path and the record written to the open file, so
    after a rename they would be two files (/code-review, D49)."""
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    log.append(tool_record())
    location.rename(tmp_path / "audit.jsonl.1")  # rotated, with its anchor
    anchor_path(location).rename(anchor_path(tmp_path / "audit.jsonl.1"))
    AuditLog(location, now=make_clock()).close()  # a new, empty log at the path

    with pytest.raises(PolicyUnavailable, match="moved or replaced"):
        log.append(tool_record())
    log.close()
    assert verify_chain(tmp_path / "audit.jsonl.1").ok


def test_a_missing_directory_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        AuditLog(tmp_path / "nowhere" / "audit.jsonl")


# --------------------------------------------------------------------------
# Reopening — one chain, not two
# --------------------------------------------------------------------------


def test_reopening_continues_the_chain(tmp_path: Path) -> None:
    location = tmp_path / "audit.jsonl"
    first = AuditLog(location, now=make_clock())
    first.append(tool_record())
    first.append(tool_record())
    first.close()

    second = AuditLog(location, now=make_clock())
    second.append(tool_record())
    second.close()

    records = records_in(location)
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert records[2]["prev_hash"] == records[1]["hash"]
    assert verify_chain(location).ok


def test_appending_after_a_broken_tail_is_refused(tmp_path: Path) -> None:
    """Chaining onto garbage produces a log that can never verify."""
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    log.append(tool_record())
    log.close()

    with location.open("ab") as handle:
        handle.write(b'{"seq": 2, "truncated"\n')

    with pytest.raises(ConfigError, match="not valid JSON"):
        AuditLog(location)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def test_an_untouched_chain_verifies(tmp_path: Path) -> None:
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    for _ in range(4):
        log.append(tool_record())
    log.close()

    status = verify_chain(location)
    assert status.ok
    assert status.records == 4
    assert status.broken_at is None


def test_an_empty_log_verifies(tmp_path: Path) -> None:
    location = tmp_path / "audit.jsonl"
    AuditLog(location, now=make_clock()).close()
    assert verify_chain(location) == verify_chain(location)
    assert verify_chain(location).ok
    assert verify_chain(location).records == 0


@pytest.mark.parametrize("target", [0, 1, 2, 3, 4])
def test_editing_any_record_is_detected_at_that_record(
    tmp_path: Path, target: int
) -> None:
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    for _ in range(5):
        log.append(tool_record())
    log.close()

    records = records_in(location)
    records[target]["decision"] = (
        "allow" if records[target]["decision"] == "deny" else "deny"
    )
    location.write_text(
        "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records
        ),
        encoding="utf-8",
    )

    status = verify_chain(location)
    assert not status.ok
    assert status.broken_at == target + 1
    assert status.detail == "the record's contents do not match its hash"


def test_deleting_a_record_is_detected(tmp_path: Path) -> None:
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    for _ in range(4):
        log.append(tool_record())
    log.close()

    lines = location.read_text(encoding="utf-8").splitlines()
    location.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")

    status = verify_chain(location)
    assert not status.ok
    assert status.broken_at == 2


def test_a_reordered_log_is_detected(tmp_path: Path) -> None:
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    for _ in range(3):
        log.append(tool_record())
    log.close()

    lines = location.read_text(encoding="utf-8").splitlines()
    location.write_text(
        "\n".join([lines[1], lines[0], lines[2]]) + "\n", encoding="utf-8"
    )

    assert not verify_chain(location).ok


def test_verifying_a_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="could not be read"):
        verify_chain(tmp_path / "absent.jsonl")


# --------------------------------------------------------------------------
# Locked decision #5, made structural
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"arguments": {"amount_usd": 600}}, id="tool-arguments"),
        pytest.param({"prompt": SENTINEL}, id="prompt"),
        pytest.param({"completion": SENTINEL}, id="completion"),
    ],
)
def test_a_field_the_spec_does_not_define_is_refused(
    tmp_path: Path, extra: dict[str, object]
) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", now=make_clock())
    with pytest.raises(ConfigError, match="unrecognised field"):
        log.append(tool_record(**extra))
    log.close()


@pytest.mark.parametrize("field", ["v", "seq", "ts", "prev_hash", "hash"])
def test_a_caller_cannot_write_its_own_sequence_or_hash(
    tmp_path: Path, field: str
) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", now=make_clock())
    with pytest.raises(ConfigError, match="log-owned field"):
        log.append(tool_record(**{field: 99}))
    log.close()


def test_an_action_may_name_what_was_called_but_not_its_arguments(
    tmp_path: Path,
) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", now=make_clock())
    with pytest.raises(ConfigError, match="unrecognised field"):
        log.append(
            tool_record(action={"kind": "tool", "name": "refund", "args": SENTINEL})
        )
    with pytest.raises(ConfigError, match=r"action\.kind"):
        log.append(tool_record(action={"kind": "shell", "name": "rm"}))
    log.close()


def test_no_tool_argument_reaches_the_file(tmp_path: Path) -> None:
    """The end-to-end version of the same claim, asserted on the bytes on disk."""
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    log.append(
        tool_record("deny", reason="constraint_violated", rule="refund.amount_usd.max")
    )
    log.close()
    assert SENTINEL not in location.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The concurrent path (Rule 18.1)
# --------------------------------------------------------------------------


def test_many_threads_produce_one_valid_chain(tmp_path: Path) -> None:
    """seq and prev_hash form a chain; two threads interleaving would corrupt it."""
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location)
    writers = 8
    per_writer = 25
    start = threading.Barrier(writers)

    def write_many() -> None:
        start.wait()
        for _ in range(per_writer):
            log.append(tool_record())

    threads = [threading.Thread(target=write_many) for _ in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log.close()

    status = verify_chain(location)
    assert status.ok, status.detail
    assert status.records == writers * per_writer

    sequences = [r["seq"] for r in records_in(location)]
    assert sequences == list(range(1, writers * per_writer + 1))


# --------------------------------------------------------------------------
# Verification checks the bytes, not just the parsed object (D19)
# --------------------------------------------------------------------------


def genuine_line(tmp_path: Path) -> tuple[Path, str]:
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    log.append(tool_record())
    log.close()
    return location, location.read_text(encoding="utf-8").strip()


def test_duplicate_keys_cannot_hide_a_forged_record(tmp_path: Path) -> None:
    """`json.loads` keeps the LAST of duplicate keys, so re-hashing the parsed
    object matches while a human, a grep or any first-wins parser reads the
    attacker's values instead."""
    location, genuine = genuine_line(tmp_path)
    forged = (
        '{"decision":"deny","agent_id":"mallory",'
        '"action":{"kind":"tool","name":"transfer_funds"},' + genuine[1:]
    )
    location.write_text(forged + "\n", encoding="utf-8")

    status = verify_chain(location)
    assert not status.ok
    assert status.broken_at == 1
    assert status.detail == "the record is not in the canonical form the log writes"


def test_a_reserialised_record_does_not_verify(tmp_path: Path) -> None:
    """Same object, different bytes. The log only ever writes canonical form."""
    location, genuine = genuine_line(tmp_path)
    location.write_text(
        json.dumps(json.loads(genuine), indent=2) + "\n", encoding="utf-8"
    )
    assert not verify_chain(location).ok


def test_a_non_json_constant_is_reported_not_raised(tmp_path: Path) -> None:
    """`json.loads` accepts NaN; canonical_json refuses to emit it. The
    tamper-detector must report that, not die of it (D16)."""
    location, genuine = genuine_line(tmp_path)
    location.write_text(genuine[:-1] + ',"zz":NaN}\n', encoding="utf-8")

    status = verify_chain(location)
    assert not status.ok
    assert status.detail == "the record is not canonical JSON"


def test_an_untouched_record_still_verifies(tmp_path: Path) -> None:
    location, genuine = genuine_line(tmp_path)
    location.write_text(genuine + "\n", encoding="utf-8")
    assert verify_chain(location).ok


# --------------------------------------------------------------------------
# The injected clock is a contract (D19)
# --------------------------------------------------------------------------


def test_a_naive_clock_is_refused_not_silently_reinterpreted(tmp_path: Path) -> None:
    """astimezone would read it as local time and shift the record silently."""
    log = AuditLog(
        tmp_path / "audit.jsonl", now=lambda: datetime(2026, 9, 20, 11, 2, 4)
    )
    with pytest.raises(ConfigError, match=r"no\s+timezone"):
        log.append(tool_record())
    log.close()


# --------------------------------------------------------------------------
# A short write is not a written record (D19)
# --------------------------------------------------------------------------


class ShortWriter:
    """A handle that writes a few bytes at a time, as FileIO is allowed to."""

    def __init__(self, chunk: int) -> None:
        self.chunk = chunk
        self.written = bytearray()

    def write(self, data: memoryview | bytes) -> int:
        piece = bytes(data)[: self.chunk]
        self.written.extend(piece)
        return len(piece)


class RefusingWriter:
    def write(self, _data: object) -> int:
        return 0


def test_a_short_write_is_completed_not_assumed() -> None:
    handle = ShortWriter(chunk=7)
    _write_all(handle, b"a" * 100)  # type: ignore[arg-type]  # minimal write-only stub
    assert bytes(handle.written) == b"a" * 100


def test_a_handle_that_accepts_nothing_raises() -> None:
    with pytest.raises(OSError, match="accepted no bytes"):
        _write_all(RefusingWriter(), b"x")  # type: ignore[arg-type]  # as above


# --------------------------------------------------------------------------
# The end anchor: how far the log is known to reach (§10.13, D20)
# --------------------------------------------------------------------------


def filled_log(tmp_path: Path, count: int = 5) -> Path:
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    for _ in range(count):
        log.append(tool_record())
    log.close()
    return location


def test_the_anchor_is_written_beside_the_log_and_is_private(tmp_path: Path) -> None:
    location = filled_log(tmp_path)
    anchor = anchor_path(location)
    assert anchor.exists()
    assert stat.S_IMODE(anchor.stat().st_mode) == 0o600
    assert read_anchor(location) == (5, records_in(location)[-1]["hash"])


def test_deleting_the_end_of_the_log_is_now_detected(tmp_path: Path) -> None:
    """The one tamper the hash chain alone cannot see: the log just looks shorter."""
    location = filled_log(tmp_path)
    lines = location.read_text(encoding="utf-8").splitlines()
    location.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")

    status = verify_chain(location)
    assert not status.ok
    assert status.records == 3
    assert status.broken_at == 4
    assert status.detail is not None
    assert "removed from the end" in status.detail


def test_a_truncated_log_refuses_to_be_reopened(tmp_path: Path) -> None:
    """Appending to it would build on a history known to be incomplete."""
    location = filled_log(tmp_path)
    lines = location.read_text(encoding="utf-8").splitlines()
    location.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="removed from the end"):
        AuditLog(location)


def test_a_rebuilt_log_does_not_pass_the_anchor(tmp_path: Path) -> None:
    """An attacker who knows the format can forge a whole consistent chain.

    The chain alone cannot refuse it — every hash is genuinely correct. The
    anchor can, because it pins how far the log reached and which record was
    there.
    """
    genuine = filled_log(tmp_path, count=5)

    forge_dir = tmp_path / "forge"
    forge_dir.mkdir()
    forged = filled_log(forge_dir, count=3)
    genuine.write_text(forged.read_text(encoding="utf-8"), encoding="utf-8")

    status = verify_chain(genuine)
    assert not status.ok
    assert status.detail is not None


def test_rewriting_the_last_record_is_detected_on_open(tmp_path: Path) -> None:
    """The attacker rewrites the tail *and* recomputes its hash correctly.

    The chain cannot object — every hash is genuinely right. The anchor can,
    because it recorded which record was at that sequence number.
    """
    location = filled_log(tmp_path, count=3)
    genuine_anchor = anchor_path(location).read_bytes()

    # Replace record 3 with a different one, validly chained.
    lines = location.read_text(encoding="utf-8").splitlines()
    location.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    anchor_path(location).unlink()
    rewriting = AuditLog(location, now=make_clock())
    rewriting.append(tool_record(agent_id="mallory"))
    rewriting.close()

    assert verify_chain(location).ok  # the chain alone is satisfied
    anchor_path(location).write_bytes(genuine_anchor)

    with pytest.raises(ConfigError, match="does not match"):
        AuditLog(location)


def test_losing_the_anchor_warns_and_does_not_brick_the_agent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The deliberate trade (D20). An anchor that can stop production traffic
    when a sidecar goes missing is a denial of service with extra steps."""
    location = filled_log(tmp_path)
    anchor_path(location).unlink()

    assert verify_chain(location).ok  # the records present are still consistent
    with caplog.at_level(logging.WARNING, logger="paveo"):
        log = AuditLog(location)
        log.append(tool_record())
        log.close()

    assert any("end anchor" in message for message in caplog.messages)
    assert read_anchor(location) == (6, records_in(location)[-1]["hash"])


def test_an_anchor_that_cannot_be_adopted_is_a_config_error_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one anchor write outside the lock: a full disk there must still be the
    module's own error, and must not leave the anchor open (D77)."""
    location = filled_log(tmp_path)
    anchor = anchor_path(location)
    anchor.unlink()
    opened: list[int] = []
    closed: list[int] = []
    real_open, real_close = os.open, os.close

    def track_open(path: str | Path, flags: int, mode: int = 0o777) -> int:
        descriptor = real_open(path, flags, mode)
        if Path(path) == anchor:
            opened.append(descriptor)
        return descriptor

    def full(_fd: int, _data: bytes, _offset: int) -> int:
        raise OSError(28, "No space left on device")

    def close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "pwrite", full)
    monkeypatch.setattr(os, "close", close)
    with pytest.raises(ConfigError, match="could not be written"):
        AuditLog(location)
    assert len(opened) == 1
    assert opened[0] in closed


def test_the_anchor_never_claims_a_record_the_log_lacks(tmp_path: Path) -> None:
    """Order matters: record first, anchor second. The reverse would refuse to
    open a log that is perfectly intact."""
    location = tmp_path / "audit.jsonl"
    log = AuditLog(location, now=make_clock())
    for expected in range(1, 4):
        log.append(tool_record())
        anchor = read_anchor(location)
        assert anchor is not None
        assert anchor[0] == expected == len(records_in(location))
    log.close()


def test_an_untouched_log_with_its_anchor_verifies(tmp_path: Path) -> None:
    location = filled_log(tmp_path)
    status = verify_chain(location)
    assert status.ok
    assert status.records == 5


def test_closing_the_log_twice_is_harmless(tmp_path: Path) -> None:
    """A `with` block and an explicit close is ordinary; the second `os.close`
    on a descriptor already given back would raise, or close another file."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.close()
    log.close()
