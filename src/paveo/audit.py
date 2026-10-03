"""The append-only, hash-chained record of every decision (``docs/SPEC_V1.md`` §6).

**Tamper-evident, not tamper-proof.** Each record's hash covers the record and
the hash before it, so editing record *n* invalidates every record after it.
Anyone with write access can still rewrite the whole chain from that point — they
simply cannot do it invisibly. Claiming more than that is how you lose a security
reviewer permanently, so the README says exactly this too.

**No payloads, structurally.** ``append`` refuses any field it does not
recognise, and refuses the log-owned fields outright. So a future edit that adds
``"arguments"`` to a record does not quietly start writing customer data to
disk — it fails immediately, in tests, on the first call. That is locked decision
#5 as a guard rather than a habit (Rule 5). It is a guard against *our* mistakes,
not a content filter: a caller determined to put a prompt in ``reason`` still
can.

**Thread safety.** ``AuditLog`` is safe to share across threads. Every append
holds its own lock for the whole build-hash-write sequence, because ``seq`` and
``prev_hash`` form a chain and two threads interleaving there would corrupt it.
That lock is **separate from the budget lock** and must stay that way: they
protect different invariants and merging them would make the audit write part of
the budget's critical section.

**Processes, too** (D49). Every append takes an exclusive ``flock`` on the log
and re-reads the chain's tail and the anchor under it, so a record always chains
onto the one actually last in the file, whichever process wrote it. Opening does
the same, so a log another process is appending to never looks truncated. This
is what lets a Claude Code hook, a new process on every tool call and several at
once, share one log. It needs a POSIX ``flock``, like the rest of this module
(``os.pwrite``, ``os.fchmod``).
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import stat
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO

from ._canonical import canonical_json
from .errors import ConfigError, PolicyUnavailable

_logger = logging.getLogger("paveo")

_RECORD_VERSION = 1

# The first record has nothing before it. A named constant rather than an empty
# string, so a chain that starts mid-file is distinguishable from one that does
# not start at all.
_GENESIS_HASH = "sha256:" + "0" * 64

# The end anchor (§10.13, D20). The hash chain is anchored at its start, so
# deleting records from the *end* leaves something internally perfect. This is
# the other end of the rope: a tiny file beside the log recording how far the log
# is known to reach.
_ANCHOR_SUFFIX = ".anchor"
_ANCHOR_WIDTH = 160

# Exactly the fields §6 defines, and nothing else. A caller may supply these.
_CALLER_FIELDS = frozenset(
    {
        "agent_id",
        "principal",
        "action",
        "decision",
        "reason",
        "rule",
        "estimated_cost_usd",
        "actual_cost_usd",
        "rate_key",
        "usage_by_class",
        "price_table_stale",
        "token_count_mode",
        "fail_open",
        "policy_id",
        "policy_hash",
        # The plan in force: the same policy admits a fourth agent under Team and
        # refuses it under Developer, so the hash alone cannot say which (D62).
        "plan",
        "prices_version",
        # D23 #2: an allowed LLM call writes a decision record before the call and
        # a settlement record after it, and this is what pairs them. It is the one
        # field added since D17 closed the set, for the reason D17 allows.
        "reservation_id",
    }
)

# Set by the log itself. A caller supplying one of these is trying to write its
# own sequence number or its own hash, which is exactly what must not be possible.
_LOG_OWNED_FIELDS = frozenset({"v", "seq", "ts", "prev_hash", "hash"})

_ACTION_FIELDS: Mapping[str, frozenset[str]] = {
    "tool": frozenset({"kind", "name"}),
    "llm": frozenset({"kind", "model"}),
}


@dataclass(frozen=True, slots=True)
class ChainStatus:
    """The result of verifying a log.

    ``verify_chain`` reports rather than raises. §3's "every denial raises" governs
    *guard decisions*, where a caller must not be able to ignore a refusal — this
    is not one. Someone auditing a log a year later wants the sequence number
    where it broke and how many records were good, not a traceback. See D16.
    """

    ok: bool
    records: int
    broken_at: int | None = None
    detail: str | None = None


def utc_now() -> datetime:
    return datetime.now(UTC)


def _write_all(handle: IO[bytes], data: bytes) -> None:
    """Write every byte, because a raw unbuffered file is allowed to write fewer.

    ``FileIO.write`` returns a count and may be short — classically on a filling
    disk, where the first write returns a partial count and only the *next* one
    raises. Taking that as success would report an unlogged decision as logged,
    which is the one thing this module exists not to do.
    """
    view = memoryview(data)
    while view:
        written = handle.write(view)
        if not written:
            raise OSError(errno.EIO, "the audit log accepted no bytes")
        view = view[written:]


def anchor_path(log: Path) -> Path:
    """Where the end anchor for ``log`` lives. Beside it, same directory."""
    return log.with_name(log.name + _ANCHOR_SUFFIX)


def _anchor_bytes(seq: int, digest: str) -> bytes:
    """A fixed-width anchor record, so an update is one overwrite in place.

    Padding to a constant width means the file is never grown or truncated after
    it is created: every update is a single ``pwrite`` at offset 0, and there is
    no window where the file is shorter than a whole record.
    """
    payload = canonical_json({"v": _RECORD_VERSION, "seq": seq, "hash": digest})
    if len(payload) > _ANCHOR_WIDTH:  # pragma: no cover — hash and seq are bounded
        raise ValueError("anchor record is wider than its fixed width")
    return payload.ljust(_ANCHOR_WIDTH, b" ")


def read_anchor(log: Path) -> tuple[int, str] | None:
    """The ``(seq, hash)`` the anchor claims the log reaches, if it is readable.

    Unreadable, absent or malformed all return ``None``. That is deliberate: see
    ``AuditLog._check_anchor`` for why a missing anchor warns rather than
    refuses.
    """
    try:
        raw = anchor_path(log).read_bytes()
    except OSError:
        return None
    try:
        record = json.loads(raw.strip() or b"null")
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    seq = record.get("seq")
    digest = record.get("hash")
    if isinstance(seq, int) and not isinstance(seq, bool) and isinstance(digest, str):
        return seq, digest
    return None


def _format_timestamp(moment: datetime) -> str:
    """ISO 8601 in UTC to the millisecond, as §6's example shows.

    A naive datetime is refused rather than converted. ``astimezone`` would read
    it as local time and silently shift the record by the machine's offset, and
    the one thing an injection point must not do is quietly rewrite the value it
    was handed (Rule 14).
    """
    if moment.tzinfo is None:
        raise ConfigError(
            "the clock supplied to the audit log returned a datetime with no timezone.",
            remedy=(
                "return an aware datetime, for example datetime.now(UTC). A "
                "naive one would be read as local time and the record would be "
                "stamped with the wrong instant."
            ),
        )
    utc = moment.astimezone(UTC)
    return f"{utc:%Y-%m-%dT%H:%M:%S}.{utc.microsecond // 1000:03d}Z"


def _chain_hash(record: Mapping[str, object], previous: str) -> str:
    """``sha256(canonical_json(record_without_hash) || prev_hash)`` — §6."""
    digest = hashlib.sha256(canonical_json(record) + previous.encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


class AuditLog:
    """A local, append-only JSONL log with a hash chain over its records."""

    def __init__(
        self, path: str | Path, *, now: Callable[[], datetime] | None = None
    ) -> None:
        """Open or create the log at ``path``.

        ``now`` is injected so timestamps are testable (Rule 14); nothing in this
        module reaches for the clock any other way. ``None`` means the real one,
        so a caller passing a clock through does not have to import our default.
        """
        self._path = Path(path)
        self._now = utc_now if now is None else now
        self._lock = threading.Lock()
        self._check_directory()
        self._handle: IO[bytes] = self._open()
        try:
            with self._across_processes(ConfigError):
                self._seq, self._prev_hash = self._recover_position()
                self._check_anchor()
                self._anchor_fd = self._open_anchor()
        except BaseException:
            self._handle.close()
            raise

    def append(self, record: Mapping[str, object]) -> str:
        """Write one decision and return its hash.

        Raises ``PolicyUnavailable`` if the record cannot be written. **The
        failure is not itself logged** — a full disk would otherwise recurse
        forever (§6). The caller turns this into a denial, because an unlogged
        decision did not happen.
        """
        _reject_unrecognised_fields(record)
        with self._lock, self._across_processes(PolicyUnavailable):
            try:
                self._check_same_file()
                # Another process may have appended since this one last did.
                self._seq, self._prev_hash = self._recover_position()
                self._check_anchor()
            except ConfigError as e:
                raise PolicyUnavailable(
                    f"the audit log at {self._path} cannot be appended to ({e}).",
                    remedy=e.remedy,
                ) from e
            entry: dict[str, object] = {
                "v": _RECORD_VERSION,
                "seq": self._seq + 1,
                "ts": _format_timestamp(self._now()),
                **record,
                "prev_hash": self._prev_hash,
            }
            digest = _chain_hash(entry, self._prev_hash)
            entry["hash"] = digest
            # Serialised outside the try, so that a record we cannot encode
            # surfaces as the programming error it is rather than being reported
            # as a disk problem.
            line = canonical_json(entry) + b"\n"
            try:
                _write_all(self._handle, line)
            except (OSError, ValueError) as e:
                # ValueError as well as OSError: a closed handle raises it, and
                # "the log is closed" is just as much a reason to deny as "the
                # disk is full". Whatever the cause, the decision is unlogged.
                raise PolicyUnavailable(
                    f"the audit log at {self._path} could not be written ({e}).",
                    remedy=(
                        "free space or fix permissions on the log. Until it can "
                        "be written, every decision is denied: an unlogged "
                        "decision did not happen."
                    ),
                ) from e
            self._seq += 1
            self._prev_hash = digest
            # After the record, never before. The anchor is a floor, so it must
            # never claim a record the log does not yet contain — a crash between
            # the two leaves the anchor one behind, which is harmless, while the
            # reverse would refuse to open a log that is perfectly intact.
            self._update_anchor(self._seq, digest)
            return digest

    def close(self) -> None:
        """Close the log and its anchor. A second call does nothing.

        Idempotent because closing twice is ordinary (a ``with`` block and an
        explicit ``close``), and the second ``os.close`` on a descriptor already
        given back would raise, or close whatever file the process opened next
        under the same number.
        """
        with self._lock:
            if self._handle.closed:
                return
            self._handle.close()
            os.close(self._anchor_fd)

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def _across_processes(
        self, failure: type[ConfigError | PolicyUnavailable]
    ) -> Iterator[None]:
        """Hold an exclusive ``flock`` on the log for the block (D49).

        Blocks until any other process's append finishes, which is microseconds.
        A lock that cannot be taken is raised as ``failure``: an unreadable
        position at open is wiring, and at append it is an unlogged decision.
        """
        if self._handle.closed:
            raise failure(
                f"the audit log at {self._path} could not be written (it is closed).",
                remedy=(
                    "open a new Paveo; every decision is denied while the log "
                    "cannot be written."
                ),
            )
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except (OSError, ValueError) as e:
            raise failure(
                f"the audit log at {self._path} could not be locked ({e}).",
                remedy=(
                    "the log is shared by every process that records to it, and "
                    "is locked while each record is written. Use a local file "
                    "system that supports flock."
                ),
            ) from e
        try:
            yield
        finally:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)

    def _check_same_file(self) -> None:
        """Refuse when the path no longer names the file this process has open.

        The tail and the anchor are re-read by path and the record is written to
        the open file, so after a rename or a replacement the two would be
        different files: a record chained onto one file's tail and written into
        another (/code-review, D49). Rotating a live log is not supported.
        """
        opened = os.fstat(self._handle.fileno())
        try:
            named = os.stat(self._path)
        except OSError:
            named = None
        if named is None or (named.st_dev, named.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ConfigError(
                f"the audit log at {self._path} was moved or replaced after this "
                f"process opened it.",
                remedy=(
                    "open a new Paveo, which starts on the file now at that path. "
                    "Rotate a log only while nothing is writing to it."
                ),
            )

    def _check_directory(self) -> None:
        """Refuse a world-writable parent directory (§9)."""
        parent = self._path.parent
        try:
            mode = parent.stat().st_mode
        except OSError as e:
            raise ConfigError(
                f"the audit log directory {parent} cannot be inspected ({e.strerror}).",
                remedy="create it, and make sure this process can reach it.",
            ) from e
        if mode & stat.S_IWOTH:
            raise ConfigError(
                f"the audit log directory {parent} is world-writable.",
                remedy=(
                    "put the log somewhere only this service can write. Anyone "
                    "who can write there can rewrite the chain — the log would "
                    "still show tampering, but it would no longer be evidence."
                ),
            )

    def _open(self) -> IO[bytes]:
        """Open append-only, readable and writable by the owner alone (§9).

        ``fchmod`` after the fact is what actually makes the guarantee: the mode
        passed to ``os.open`` is filtered through the process umask, and a file
        that already exists keeps whatever mode it already had.

        Unbuffered, so a record reaches the operating system as it is written and
        there is no buffer to lose. It is deliberately not ``fsync``ed — that
        costs milliseconds, which is the entire per-call latency budget
        (Rule 17), and the cost of the trade is published in the README.
        """
        try:
            descriptor = os.open(
                self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
        except OSError as e:
            raise ConfigError(
                f"the audit log at {self._path} could not be opened ({e.strerror}).",
                remedy=(
                    "check the directory exists and this process can write to "
                    "it. Paveo denies every call while the log is "
                    "unwritable."
                ),
            ) from e
        try:
            os.fchmod(descriptor, 0o600)
        except OSError as e:
            os.close(descriptor)
            raise ConfigError(
                f"the audit log at {self._path} could not be made private "
                f"({e.strerror}).",
                remedy="the log records who did what; it must not be world-readable.",
            ) from e
        return os.fdopen(descriptor, "wb", buffering=0)

    def _open_anchor(self) -> int:
        location = anchor_path(self._path)
        remedy = (
            "the anchor records how far the log reaches, so tail truncation is "
            "detectable. It lives beside the log and needs the same permissions."
        )
        try:
            descriptor = os.open(location, os.O_WRONLY | os.O_CREAT, 0o600)
        except OSError as e:
            raise ConfigError(
                f"the audit anchor at {location} could not be opened ({e.strerror}).",
                remedy=remedy,
            ) from e
        try:
            os.fchmod(descriptor, 0o600)
            if self._seq:
                # Adopt the current position, so a log that already existed
                # without an anchor gains one rather than staying unprotected.
                os.pwrite(descriptor, _anchor_bytes(self._seq, self._prev_hash), 0)
        except OSError as e:
            os.close(descriptor)
            raise ConfigError(
                f"the audit anchor at {location} could not be written ({e.strerror}).",
                remedy=remedy,
            ) from e
        return descriptor

    def _update_anchor(self, seq: int, digest: str) -> None:
        """Record how far the log now reaches. Called under the lock."""
        try:
            os.pwrite(self._anchor_fd, _anchor_bytes(seq, digest), 0)
        except OSError as e:
            raise PolicyUnavailable(
                f"the audit anchor beside {self._path} could not be written ({e}).",
                remedy=(
                    "free space or fix permissions. Without the anchor, deleting "
                    "the end of the log would be undetectable, so a decision that "
                    "cannot be anchored is treated as one that cannot be logged."
                ),
            ) from e

    def _check_anchor(self) -> None:
        """Refuse a log that has gone backwards from where the anchor left it.

        **A missing or unreadable anchor warns rather than refuses.** That is a
        deliberate trade and it is the whole reason this was not built on the
        first pass: an anchor that can brick the agent is a denial-of-service with
        extra steps, and losing a sidecar file at 3am is not a reason to stop a
        customer's production traffic. The consequence is stated rather than
        hidden — an attacker who deletes the anchor *as well as* the records is
        back to being undetectable, and §10.13 says so.

        What it does buy: deleting records now requires noticing the anchor
        exists and removing it too, and the removal is itself visible.
        """
        anchor = read_anchor(self._path)
        if anchor is None:
            if self._seq:
                _logger.warning(
                    "paveo: no readable end anchor beside %s, so deletion of "
                    "records from the end of the log cannot be detected. A new "
                    "anchor will be written from the current position.",
                    self._path,
                )
            return
        anchored_seq, anchored_hash = anchor
        if self._seq < anchored_seq:
            raise ConfigError(
                f"the audit log at {self._path} ends at record {self._seq}, but "
                f"its anchor records {anchored_seq}. "
                f"{anchored_seq - self._seq} record(s) have been removed from "
                f"the end.",
                remedy=(
                    "the log has been truncated. Preserve both files as they are "
                    "and investigate; appending to it now would build on a "
                    "history that is known to be incomplete."
                ),
            )
        if self._seq == anchored_seq and self._prev_hash != anchored_hash:
            raise ConfigError(
                f"the audit log at {self._path} ends at record {self._seq}, and "
                f"that record does not match the one its anchor recorded.",
                remedy=(
                    "the last record has been replaced. Preserve both files and "
                    "investigate."
                ),
            )

    def _recover_position(self) -> tuple[int, str]:
        """Continue an existing chain rather than starting a second one.

        A tail that cannot be parsed is refused instead of chained onto: appending
        after garbage produces a log that can never verify, and the operator would
        find out at the worst possible moment.
        """
        line = _last_line(self._path)
        if line is None:
            return 0, _GENESIS_HASH
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"the last record in {self._path} is not valid JSON.",
                remedy=(
                    "the file is truncated or is not an audit log. Move it aside; "
                    "appending after it would produce a chain that never verifies."
                ),
            ) from e
        if not isinstance(record, dict):
            raise ConfigError(
                f"the last record in {self._path} is not an audit record.",
                remedy="move the file aside; this is not a Paveo audit log.",
            )
        seq = record.get("seq")
        digest = record.get("hash")
        if (
            not isinstance(seq, int)
            or isinstance(seq, bool)
            or not isinstance(digest, str)
        ):
            raise ConfigError(
                f"the last record in {self._path} has no usable seq or hash.",
                remedy="move the file aside; this is not a Paveo audit log.",
            )
        return seq, digest


def _reject_unrecognised_fields(record: Mapping[str, object]) -> None:
    """Refuse anything §6 does not define. This is the redaction guard."""
    owned = sorted(set(record) & _LOG_OWNED_FIELDS)
    if owned:
        raise ConfigError(
            f"an audit record supplied the log-owned field(s): "
            f"{', '.join(repr(key) for key in owned)}.",
            remedy="the log sets v, seq, ts, prev_hash and hash itself.",
        )
    unknown = sorted(set(record) - _CALLER_FIELDS)
    if unknown:
        raise ConfigError(
            f"an audit record carries the unrecognised field(s): "
            f"{', '.join(repr(key) for key in unknown)}.",
            remedy=(
                "audit records hold identities, decisions, reasons, costs and "
                "hashes — never prompts, completions or tool arguments. Add the "
                "field to SPEC_V1.md §6 and to _CALLER_FIELDS if it is genuinely "
                "payload-free."
            ),
        )
    _check_action(record.get("action"))


def _check_action(action: object) -> None:
    if action is None:
        return
    if not isinstance(action, Mapping):
        raise ConfigError(
            "an audit record's action is not an object.",
            remedy='write {"kind": "tool", "name": "refund"}.',
        )
    kind = action.get("kind")
    permitted = _ACTION_FIELDS.get(kind) if isinstance(kind, str) else None
    if permitted is None:
        raise ConfigError(
            f"an audit record's action.kind is {kind!r}.",
            remedy=f"use one of: {', '.join(sorted(_ACTION_FIELDS))}.",
        )
    unknown = sorted(set(action) - permitted)
    if unknown:
        raise ConfigError(
            f"an audit record's action carries the unrecognised field(s): "
            f"{', '.join(repr(key) for key in unknown)}.",
            remedy=(
                "an action names what was called, never what it was called with. "
                "Tool arguments are where the customer's data lives."
            ),
        )


def _last_line(path: Path) -> bytes | None:
    """Read the final line without reading the whole file.

    An audit log grows without bound, so starting up must not be proportional to
    how long the service has been running.
    """
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            buffer = b""
            while position > 0:
                step = min(4096, position)
                position -= step
                handle.seek(position)
                buffer = handle.read(step) + buffer
                trimmed = buffer.rstrip(b"\n")
                if b"\n" in trimmed:
                    return trimmed[trimmed.rindex(b"\n") + 1 :]
            return buffer.rstrip(b"\n") or None
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ConfigError(
            f"the audit log at {path} could not be read ({e.strerror}).",
            remedy="check the path and the permissions on it.",
        ) from e


def verify_chain(path: str | Path) -> ChainStatus:
    """Check that a log is intact, and say where it is not if it is not.

    Recomputes every record's hash from its own contents and the hash before it,
    checks that sequence numbers run consecutively, and — if an anchor is present
    beside the log — that the log still reaches as far as the anchor says and
    that the record at that point is the one the anchor recorded.

    Without the anchor this can only prove the records present are consistent
    with each other, never that none were deleted from the end. With it, a
    wholesale rewrite starting from record 1 fails too, because the rewritten
    record at the anchored sequence has a different hash.

    Reports rather than raises (see ``ChainStatus``).
    """
    count = 0
    try:
        location = Path(path)
        for _ in verified_records(location, anchor=read_anchor(location)):
            count += 1
    except BrokenChain as e:
        return e.status
    return ChainStatus(ok=True, records=count)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """How far the log reached at one instant, taken between appends."""

    anchor: tuple[int, str] | None
    size: int


def snapshot(location: Path) -> Snapshot:
    """The anchor and the log's length, read under a shared ``flock``.

    Appends hold the exclusive lock while they write a record and its anchor, so
    this sees neither half-done: every byte up to ``size`` is whole records, and
    the anchor names one of them. Reading only that far is what lets the audit
    evidence export run while an agent is writing (D66). Held for two reads, so
    an append waits microseconds.
    """
    try:
        with location.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return Snapshot(
                    read_anchor(location), os.fstat(handle.fileno()).st_size
                )
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        raise ConfigError(
            f"the audit log at {location} could not be read ({e.strerror}).",
            remedy="check the path and the permissions on it.",
        ) from e


class BrokenChain(Exception):  # noqa: N818 - it is the state found, not a failure to act
    """Raised by ``verified_records`` where the chain breaks."""

    def __init__(self, status: ChainStatus) -> None:
        super().__init__(status.detail)
        self.status = status


def verified_records(
    location: Path, *, anchor: tuple[int, str] | None, size: int | None = None
) -> Iterator[tuple[bytes, dict[str, object]]]:
    """Each record of the log, as its line and parsed, checked before it is
    yielded; ``BrokenChain`` where the chain breaks, and at the end if the log
    falls short of ``anchor``. The caller reads the anchor, so that what it
    reports about the anchor is the anchor that was checked.

    ``size`` stops at that many bytes, from a ``snapshot``: what lies past it
    was written after, and may be half-written now. ``verify_chain`` reads
    everything.
    """
    previous = _GENESIS_HASH
    count = 0
    consumed = 0
    hash_at_anchor: str | None = None

    try:
        with location.open("rb") as handle:
            for raw in handle:
                consumed += len(raw)
                if size is not None and consumed > size:
                    break
                line = raw.strip()
                if not line:
                    continue
                record, failure = _verify_record(line, previous, count + 1)
                if record is None:
                    raise BrokenChain(
                        ChainStatus(
                            ok=False, records=count, broken_at=count + 1, detail=failure
                        )
                    )
                count += 1
                previous = str(record["hash"])
                if anchor is not None and count == anchor[0]:
                    hash_at_anchor = previous
                yield line, record
    except OSError as e:
        raise ConfigError(
            f"the audit log at {location} could not be read ({e.strerror}).",
            remedy="check the path and the permissions on it.",
        ) from e

    if anchor is not None:
        # The chain proves the records present are consistent with each other.
        # The anchor is the other end of the rope: it proves how many there
        # should be, and pins one of them by hash so a wholesale rewrite from
        # record 1 does not verify either (§10.13, D20).
        anchored_seq, anchored_hash = anchor
        if count < anchored_seq:
            raise BrokenChain(
                ChainStatus(
                    ok=False,
                    records=count,
                    broken_at=count + 1,
                    detail=(
                        f"the log holds {count} records; its anchor records "
                        f"{anchored_seq}, so {anchored_seq - count} have been "
                        f"removed from the end"
                    ),
                )
            )
        if hash_at_anchor != anchored_hash:
            raise BrokenChain(
                ChainStatus(
                    ok=False,
                    records=count,
                    broken_at=anchored_seq,
                    detail="record does not match the hash its anchor recorded",
                )
            )


def _verify_record(
    line: bytes, previous: str, expected_seq: int
) -> tuple[dict[str, object] | None, str | None]:
    """Return ``(the record, None)``, or ``(None, why it failed)``."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None, "the record is not valid JSON"
    if not isinstance(record, dict):
        return None, "the record is not an object"

    # Verify the BYTES, not just the parsed object. `json.loads` is not injective:
    # whitespace, key order and — the dangerous one — duplicate keys all collapse
    # to the same object, and Python keeps the last occurrence. Without this, an
    # attacker prepends shadowed duplicates carrying whatever they like, the
    # recomputed hash still matches, and a human or a log shipper reading the raw
    # line sees the forged values in a record `verify_chain` certifies as intact.
    # `append` writes exactly `canonical_json(entry)`, so this is byte-exact and
    # free. It also catches a tampered record carrying NaN or Infinity, which
    # `json.loads` accepts and `canonical_json` refuses to emit.
    try:
        if canonical_json(record) != line:
            return None, "the record is not in the canonical form the log writes"
    except (TypeError, ValueError):
        return None, "the record is not canonical JSON"

    claimed = record.get("hash")
    if not isinstance(claimed, str):
        return None, "the record has no hash"
    failure = _mismatch(record, claimed, previous, expected_seq)
    return (None, failure) if failure is not None else (record, None)


def _mismatch(
    record: Mapping[str, object], claimed: str, previous: str, expected_seq: int
) -> str | None:
    if record.get("seq") != expected_seq:
        return f"expected seq {expected_seq}, found {record.get('seq')!r}"
    if record.get("prev_hash") != previous:
        return "prev_hash does not match the record before it"
    body = {key: value for key, value in record.items() if key != "hash"}
    if _chain_hash(body, previous) != claimed:
        return "the record's contents do not match its hash"
    return None
