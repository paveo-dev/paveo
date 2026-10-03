"""What a session remembers for ``requires``, ``rate`` and ``repeat`` (D58, D59).

``Memory`` is that memory in one process, which a library ``Session`` holds.
``SessionMemory`` is the same memory kept on disk for the ``paveo`` guard, a new
process for every tool call: one file per agent session under ``.paveo/memory/``,
named by a digest of the session id. One class for both, so the two cannot come
to count differently (/code-review, D59).

**Digests and times, never values** (locked decision #5). A file holds a salt,
the footprints ``requires`` compares and the admitted calls ``rate`` and
``repeat`` count, each a SHA-256 over the salt and the values. The salt is random
per file, so no table precomputed once reverses them all. It lives in the same
file, so it does not stop someone who can read the file from guessing a
low-entropy value one at a time: the file and its folder are private to their
owner, like the audit log, for that reason, and a file untouched for a week is
deleted.

**Time never runs backwards, and never freezes.** Rule time moves on by however
far the clock moved forward since its last reading, and not at all when it moved
back. A clock that steps back is held only until the next reading, never until it
catches up; a call already forgotten cannot fall back inside a window.

**Locked for the whole decision.** ``SessionMemory`` holds an exclusive
``flock`` from reading the memory to writing it back, and the guard decides and
records the call inside it, so hooks run at once for one session take turns. The
audit log's lock is taken inside this one, never around it, so the two cannot
deadlock.

**Anything wrong refuses** (locked decision #4): a file that is not what this
module writes, a folder or file that is a link, not ours, or writable by others,
or a file over its limits. A write is made over the old bytes and only then cut
to length, so an interrupted one leaves a file that does not parse, and refuses,
rather than an empty one that would read as a new session. Nothing here opens a
socket.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType

from .errors import ConfigError
from .policy import Call, Footprint, Policy, Recall

_FOLDER = "memory"
_VERSION = 1
_SALT_BYTES = 16
# A day of calls at a hook's pace is far under this; a file over it is refused
# rather than read, since it is not one this module wrote.
_MAX_FILE_BYTES = 16 << 20
# `requires` keeps one footprint per distinct value. Past this a call leaves
# none, so a later call that needs it is refused: never a false match.
_MAX_FOOTPRINTS = 10_000
# Every window is at most a day (D59), so a file untouched for a week holds no
# call any rule still counts; its footprints only let `requires` pass, so losing
# them refuses more. Deleted when a new session's file is made.
_STALE_SECONDS = 7 * 86_400


class Memory:
    """One session's memory, in this process.

    Not safe to share across threads or tasks: it belongs to one session, which
    is not either.
    """

    def __init__(self, salt: bytes = b"") -> None:
        self._salt = salt
        self._footprints: set[Footprint] = set()
        self._calls: list[Call] = []
        # Rule time, and the clock reading it was last moved on from.
        self._latest = 0.0
        self._reading: float | None = None

    def recall(self, clock: float) -> Recall:
        """This session's memory at rule time, moved on from ``clock``, a
        reading in seconds since the epoch."""
        if self._reading is None:
            self._latest = max(self._latest, clock)
        else:
            self._latest += max(0.0, clock - self._reading)
        self._reading = clock
        return Recall(
            now=self._latest,
            footprints=self._footprints,
            calls=self._calls,
            salt=self._salt,
        )

    def count(self, call: Call | None, *, horizon: int) -> bool:
        """Count an admitted call for ``rate`` and ``repeat``, and forget calls
        past every window. Called before the decision is recorded (``enforce``).
        Returns whether anything changed."""
        since = self._latest - horizon
        kept = [earlier for earlier in self._calls if earlier.at > since]
        changed = len(kept) != len(self._calls) or call is not None
        self._calls = kept if call is None else [*kept, call]
        return changed

    def leave(self, footprints: frozenset[Footprint]) -> bool:
        """Leave an admitted call's footprints for ``requires``. Called only
        after the decision is recorded. Returns whether anything changed."""
        room = max(0, _MAX_FOOTPRINTS - len(self._footprints))
        new = sorted(footprints - self._footprints, key=repr)[:room]
        self._footprints.update(new)
        return bool(new)

    def steps(
        self,
        policy: Policy,
        agent_id: str,
        tool: str,
        arguments: Mapping[str, object],
        recall: Recall,
    ) -> tuple[Callable[[], object], Callable[[], object]]:
        """``enforce.check_tool``'s ``admitting`` and ``admitted`` for one call.

        What the call leaves is worked out only once the rules have admitted
        it: a refused call never pays for hashing its arguments (/code-review).
        """
        footprints: list[frozenset[Footprint]] = []

        def admitting() -> None:
            left, call = policy.remember(agent_id, tool, arguments, recall)
            footprints.append(left)
            self.count(call, horizon=policy.horizon(agent_id))

        return admitting, lambda: self.leave(footprints[0])


class SessionMemory(Memory):
    """One agent session's memory file, locked while open. A context manager.

    Not safe to share across threads: it is one process's view of one file for
    one decision, and the guard opens it once per call.
    """

    def __init__(
        self,
        directory: Path,
        agent: str,
        session: str,
        *,
        salt: Callable[[int], bytes] = os.urandom,
    ) -> None:
        """One file per agent *and* session: two ``--agent`` hooks that share a
        session id would otherwise count each other's calls and forget them at
        each other's horizons (/code-review, D59). ``salt`` is injected so tests
        are repeatable (Rule 14)."""
        super().__init__()
        key = f"{agent}\0{session}".encode()
        name = hashlib.sha256(key).hexdigest()[:32]
        self._folder = directory / _FOLDER
        self._path = self._folder / f"{name}.json"
        self._new_salt = salt
        self._fd = -1
        self._fresh = False

    def __enter__(self) -> SessionMemory:
        # A sweep may delete the file between this process opening it and
        # locking it; the lock would then guard a file no one else can find.
        # So the file is checked to still be the one at the path, once more on
        # a retry (/code-review, D59).
        for _ in range(2):
            self._fd = self._open()
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX)
                if self._still_at_path():
                    self._read()
                    return self
            except BaseException:
                os.close(self._fd)
                raise
            os.close(self._fd)
        raise ConfigError(
            f"the guard's memory at {self._path} was replaced while opening it.",
            remedy="try again; if it keeps happening, remove the .paveo/memory folder.",
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        os.close(self._fd)  # closing releases the lock

    def recall(self, clock: float) -> Recall:
        if self._fresh:
            self._sweep(clock)
        stepped_back = self._reading is not None and clock < self._reading
        recalled = super().recall(clock)
        if stepped_back:
            # Saved now, not with the next admitted call: a limit that refuses
            # every call would otherwise never save it, and would hold until the
            # clock caught up (/security-review, D59). Raises ConfigError if it
            # cannot be; the guard then judges the call with no memory.
            self._write()
        return recalled

    def count(self, call: Call | None, *, horizon: int) -> bool:
        """As ``Memory.count``, then write the file back if anything changed.
        Raises ``ConfigError`` if it cannot be written, and the call is refused."""
        changed = super().count(call, horizon=horizon)
        if changed or self._fresh:
            self._write()
        return changed

    def leave(self, footprints: frozenset[Footprint]) -> bool:
        """As ``Memory.leave``, then write the file back if anything changed."""
        changed = super().leave(footprints)
        if changed:
            self._write()
        return changed

    def _still_at_path(self) -> bool:
        try:
            named = os.stat(self._path, follow_symlinks=False)
        except FileNotFoundError:
            return False
        held = os.fstat(self._fd)
        return (named.st_dev, named.st_ino) == (held.st_dev, held.st_ino)

    def _write(self) -> None:
        document = {
            "version": _VERSION,
            "salt": self._salt.hex(),
            "latest": self._latest,
            "reading": self._reading,
            "footprints": sorted(
                [footprint.tool, list(footprint.same), footprint.digest]
                for footprint in self._footprints
            ),
            "calls": [[kept.tool, kept.at, kept.digest] for kept in self._calls],
        }
        data = json.dumps(document, separators=(",", ":")).encode("ascii")
        try:
            written = 0
            while written < len(data):
                written += os.pwrite(self._fd, data[written:], written)
            os.ftruncate(self._fd, len(data))
        except OSError as e:
            raise ConfigError(
                f"the guard's memory at {self._path} could not be written "
                f"({e.strerror}).",
                remedy="free some space, or check the .paveo folder is writable.",
            ) from e

    def _open(self) -> int:
        try:
            self._folder.mkdir(mode=0o700, exist_ok=True)
            folder = os.lstat(self._folder)
        except OSError as e:
            raise ConfigError(
                f"the guard's memory folder {self._folder} could not be made "
                f"({e.strerror}).",
                remedy="check the .paveo folder exists and is writable.",
            ) from e
        if not stat.S_ISDIR(folder.st_mode) or not _private(folder):
            raise ConfigError(
                f"the guard's memory folder {self._folder} is a link, not yours, "
                f"or writable by others.",
                remedy="remove it; the guard makes a private one.",
            )
        self._ignore_in_git()
        try:
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        except OSError as e:
            raise ConfigError(
                f"the guard's memory at {self._path} could not be opened "
                f"({e.strerror}).",
                remedy="remove the file if it is a link; the guard makes a new one.",
            ) from e
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not _private(opened, links=True):
            os.close(fd)
            raise ConfigError(
                f"the guard's memory at {self._path} is not a private file of yours.",
                remedy="remove it; the guard makes a new one.",
            )
        os.fchmod(fd, 0o600)
        return fd

    def _ignore_in_git(self) -> None:
        """The folder keeps itself out of the repository, so a project set up
        before it existed does not start committing session files (D59)."""
        try:
            fd = os.open(
                self._folder / ".gitignore",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            return
        except OSError as e:
            raise ConfigError(
                f"the guard's memory folder {self._folder} could not be kept out "
                f"of git ({e.strerror}).",
                remedy="check the .paveo folder is writable.",
            ) from e
        try:
            written = os.write(fd, b"*\n")
        except OSError as e:
            written = 0
            failure: OSError | None = e
        else:
            failure = None
        os.close(fd)
        if written != len(b"*\n"):
            # Never left empty: an empty .gitignore reads as done (/security-review).
            (self._folder / ".gitignore").unlink(missing_ok=True)
            raise ConfigError(
                f"the guard's memory folder {self._folder} could not be kept out "
                f"of git.",
                remedy="check the .paveo folder is writable.",
            ) from failure

    def _read(self) -> None:
        size = os.fstat(self._fd).st_size
        if size > _MAX_FILE_BYTES:
            raise self._unreadable(f"it is over {_MAX_FILE_BYTES >> 20} MiB")
        raw = os.pread(self._fd, size, 0)
        if not raw:
            # Only a file this call just made is empty: a write is never cut to
            # nothing first, so an interrupted one does not parse instead.
            self._fresh = True
            self._salt = self._new_salt(_SALT_BYTES)
            return
        try:
            document = json.loads(raw)
            if document["version"] != _VERSION:
                raise self._unreadable("it is from another version")
            self._salt = bytes.fromhex(document["salt"])
            self._latest = _number(document["latest"])
            reading = document["reading"]
            self._reading = None if reading is None else _number(reading)
            self._footprints = {
                Footprint(_text(tool), tuple(map(_text, _list(same))), _text(digest))
                for tool, same, digest in map(_triple, _list(document["footprints"]))
            }
            self._calls = [
                Call(
                    _text(tool), _number(at), None if digest is None else _text(digest)
                )
                for tool, at, digest in map(_triple, _list(document["calls"]))
            ]
        except (ValueError, TypeError, KeyError) as e:
            raise self._unreadable("it is not what the guard writes") from e
        if len(self._salt) != _SALT_BYTES:
            raise self._unreadable("its salt is the wrong length")
        if len(self._footprints) > _MAX_FOOTPRINTS:
            raise self._unreadable("it holds more than the guard keeps")

    def _sweep(self, clock: float) -> None:
        """Delete the memory of sessions untouched for a week, once per new
        session. A file some guard holds right now is left alone.

        Housekeeping: a file that is already gone, is a link, or cannot be read
        is skipped, since nothing is decided from what it held. One that can be
        read but not removed refuses, as the folder is then not ours to manage.
        """
        for path in self._folder.glob("*.json"):
            if path == self._path:
                continue
            try:
                seen = os.lstat(path)
                if (
                    not stat.S_ISREG(seen.st_mode)
                    or seen.st_mtime > clock - _STALE_SECONDS
                ):
                    continue
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held, named = os.fstat(fd), os.lstat(path)
                # A guard may have written to it, or replaced it, between the
                # first look and the lock (/security-review, D59).
                if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino) or (
                    held.st_mtime > clock - _STALE_SECONDS
                ):
                    continue
                path.unlink()
            except OSError as e:
                if e.errno not in {errno.EWOULDBLOCK, errno.ENOENT}:
                    raise ConfigError(
                        f"a stale session memory at {path} could not be removed "
                        f"({e.strerror}).",
                        remedy="check the .paveo/memory folder is writable.",
                    ) from e
            finally:
                os.close(fd)

    def _unreadable(self, why: str) -> ConfigError:
        return ConfigError(
            f"the guard's memory at {self._path} cannot be used: {why}.",
            remedy=(
                "delete the file to start this session's memory afresh. Until "
                "then every call from the session is refused."
            ),
        )


def _private(seen: os.stat_result, *, links: bool = False) -> bool:
    """Ours, and writable by no one else. With ``links``, one name only: a hard
    link planted elsewhere would let a second path reach the same memory."""
    return (
        seen.st_uid == os.getuid()
        and not seen.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        and (not links or seen.st_nlink == 1)
    )


def _list(value: object) -> list[object]:
    """A JSON list, not a string that would be read one character at a time."""
    if not isinstance(value, list):
        raise TypeError("expected a list")
    return value


def _triple(value: object) -> tuple[object, object, object]:
    match _list(value):
        case [first, second, third]:
            return first, second, third
    raise ValueError("expected three fields")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value


def _number(value: object) -> float:
    """A finite time. NaN would sit outside every window and count for nothing."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("expected a number")
    if not math.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)
