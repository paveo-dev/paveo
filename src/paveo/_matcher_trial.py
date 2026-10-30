"""Trying a repository's own matchers where they cannot hang paveo (D116, D118).

A hook's matcher is a regular expression from a project's settings, and
Python's can backtrack for minutes, even over the four letters of ``Bash``.
So paveo never compiles or tries one itself: a child Python, isolated
(``-I``, so a project's files cannot stand in for ``re``), compiles and tries
them one at a time, in the order given, answering a line each.

**One budget per run.** Every child in one process shares ``TIMEOUT_S``
seconds, counted from the first try: once it is spent no child starts, and
each matcher not answered is could-not-tell. So a file with a thousand slow
matchers costs five seconds, not a thousand times five. A matcher longer than
``MAX_LENGTH`` is not read at all: no real one is, and the checks of a
pattern's text that run in paveo (``_PYTHON_ONLY``, ``_NOT_RUST``, ``_NOT_JS``)
then stay quick. ``paveo test`` (all three agents) and ``--selftest`` read
matchers only through here.

Opens no socket; the child imports only ``json``, ``re`` and ``sys``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence

TIMEOUT_S = 5.0
MAX_LENGTH = 1000
# Less than this left, and a child could not even start in time.
_MIN_LEFT_S = 0.05
UNREAD = "its matcher is not a pattern paveo test can read"
LATE = "its matcher could not be tried in time, so it is not run"
FAILED = "its matcher could not be tried, so it is not run"
# Each answer is the indexes of the texts found; null for a pattern Python
# refuses; false for any other failure, which costs only that matcher.
_TRY = (
    "import json, re, sys\n"
    "d = json.load(sys.stdin)\n"
    "for matcher, key in d['matchers']:\n"
    "    texts = d['texts'][key]\n"
    "    try:\n"
    "        pattern = re.compile(matcher)\n"
    "    except (re.error, RecursionError, OverflowError):\n"
    "        print('null', flush=True)\n"
    "        continue\n"
    "    except Exception:\n"
    "        print('false', flush=True)\n"
    "        continue\n"
    "    try:\n"
    "        found = [i for i, t in enumerate(texts) if pattern.search(t)]\n"
    "    except Exception:\n"
    "        print('false', flush=True)\n"
    "        continue\n"
    "    print(json.dumps(found), flush=True)\n"
)


class Budget:
    """The seconds every try in one run shares, counted from the first.

    Not safe to share across threads: one run, one thread, one budget."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._deadline: float | None = None
        # Only answers are remembered: a late or failed try is asked again,
        # and a spent budget answers late at once.
        self.answers: dict[tuple[str, tuple[str, ...]], frozenset[str]] = {}

    def left(self) -> float:
        now = self._clock()
        if self._deadline is None:
            self._deadline = now + TIMEOUT_S
        return self._deadline - now


_BUDGET = Budget()


def reset(budget: Budget | None = None) -> None:
    """Start a fresh budget: each command is one run (tests start one each)."""
    global _BUDGET  # noqa: PLW0603 - one budget per run, by design
    _BUDGET = budget or Budget()


def try_all(
    wanted: Sequence[tuple[str, str]],
    texts: Mapping[str, Sequence[str]],
) -> list[frozenset[str] | str]:
    """For each (matcher, key), in order, which of ``texts[key]`` it finds,
    or why that could not be told: one isolated child for all, ended when
    the run's budget is spent, its answers so far kept."""
    if not wanted:
        return []
    left = _BUDGET.left()
    if left <= _MIN_LEFT_S:
        return [LATE] * len(wanted)
    payload = json.dumps(
        {"matchers": list(wanted), "texts": {k: list(v) for k, v in texts.items()}}
    ).encode()
    late = False
    try:
        done = subprocess.run(  # noqa: S603 - this interpreter, isolated, our program
            [sys.executable, "-I", "-c", _TRY],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=left,
            check=False,
        )
        out = done.stdout
    except subprocess.TimeoutExpired as e:
        late = True
        out = e.stdout or b""
    except (OSError, ValueError):  # ValueError: no interpreter to name
        out = b""
    # A line cut off mid-write, by the limit or a crash, is no answer.
    lines = out.split(b"\n")[:-1]
    answers = [
        _answered(line, texts[key])
        for line, (_matcher, key) in zip(lines, wanted, strict=False)
    ]
    return answers + [LATE if late else FAILED] * (len(wanted) - len(answers))


def reach(matcher: str, names: Sequence[str]) -> frozenset[str] | str:
    """Which of ``names`` one matcher finds, or why that could not be told.
    An answer is remembered for the run, since ``--selftest`` asks about the
    same matcher once per hook and event."""
    if len(matcher) > MAX_LENGTH:
        return UNREAD
    key = (matcher, tuple(sorted(set(names))))
    if key in _BUDGET.answers:
        return _BUDGET.answers[key]
    (hits,) = try_all([(matcher, "names")], {"names": key[1]})
    if not isinstance(hits, str):
        _BUDGET.answers[key] = hits
    return hits


def _answered(line: bytes, texts: Sequence[str]) -> frozenset[str] | str:
    """One line of the child's answer."""
    try:
        found = json.loads(line)
    except ValueError:
        return FAILED
    if found is None:
        return UNREAD
    if found is False:
        return FAILED
    if not (
        isinstance(found, list)
        and all(type(i) is int and 0 <= i < len(texts) for i in found)
    ):
        return FAILED
    return frozenset(texts[i] for i in found)
