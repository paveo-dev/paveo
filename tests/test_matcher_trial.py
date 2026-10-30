"""A repository's matchers are tried in an isolated child with a time limit,
never in paveo itself, for every agent (D116, D118)."""

from __future__ import annotations

import io
import json
import subprocess
import time
from pathlib import Path

import pytest

from paveo import _codex_hooks, _matcher_trial, _setup
from paveo._guard_test import Handler, Run, command
from paveo._harnesses import CLAUDE_CODE, CODEX, CURSOR, Harness

# Each optional repeat doubles the work: minutes per text if tried here.
SLOW = ".?" * 40 + "Q"
# The same over a tool name, which is only a few letters long.
SLOW_NAME = "(?:.?){400}Q"


def test_answers_come_back_in_order_and_say_which_python_cannot_read() -> None:
    texts = {"cmd": ["rm -rf x", "ls"], "tool": ["Bash"]}
    answers = _matcher_trial.try_all(
        [("rm", "cmd"), ("(?<name>x)", "cmd"), ("^Ba", "tool")], texts
    )
    assert answers == [
        frozenset({"rm -rf x"}),
        _matcher_trial.UNREAD,
        frozenset({"Bash"}),
    ]


def test_answers_before_the_limit_count_and_only_those_owed_are_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_matcher_trial, "TIMEOUT_S", 1.0)
    started = time.monotonic()
    answers = _matcher_trial.try_all(
        [("rm", "t"), (SLOW, "t"), ("ls", "t")], {"t": ["rm -rf x" * 20, "ls"]}
    )
    assert time.monotonic() - started < 5
    assert answers == [
        frozenset({"rm -rf x" * 20}),
        _matcher_trial.LATE,
        _matcher_trial.LATE,
    ]


def test_a_line_cut_off_at_the_limit_is_no_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cut_off(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("python", 1.0, output=b"[0]\n[1")

    monkeypatch.setattr(subprocess, "run", cut_off)
    answers = _matcher_trial.try_all([("a", "t"), ("b", "t")], {"t": ["x", "y"]})
    assert answers == [frozenset({"x"}), _matcher_trial.LATE]


@pytest.mark.parametrize("line", [b"false", b"{}", b"[9]", b"[true]", b"not json"])
def test_any_answer_but_indexes_or_null_is_a_failure(line: bytes) -> None:
    assert _matcher_trial._answered(line, ["x"]) == _matcher_trial.FAILED


def test_a_project_s_own_re_module_cannot_answer_for_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "re.py").write_text("def compile(*a, **k):\n    raise SystemExit(0)\n")
    (tmp_path / "json.py").write_text("raise SystemExit(0)\n")
    monkeypatch.chdir(tmp_path)
    answers = _matcher_trial.try_all([("rm", "t")], {"t": ["rm -rf x", "ls"]})
    assert answers == [frozenset({"rm -rf x"})]


# --------------------------------------------------------------------------
# Every agent: a slow matcher is could-not-tell, never a hang or a pass
# --------------------------------------------------------------------------


def _settings(folder: Path, matcher: str) -> Path:
    path = folder / ".claude" / "settings.json"
    path.parent.mkdir(parents=True)
    entry = {"type": "command", "command": "guard"}
    path.write_text(
        json.dumps({"hooks": {"PreToolUse": [{"matcher": matcher, "hooks": [entry]}]}})
    )
    return path


def _codex_hooks_json(folder: Path, matcher: str) -> Path:
    path = folder / ".codex" / "hooks.json"
    path.parent.mkdir(parents=True)
    entry = {"type": "command", "command": "guard"}
    path.write_text(
        json.dumps({"hooks": {"PreToolUse": [{"matcher": matcher, "hooks": [entry]}]}})
    )
    return path


def _blocks(_handler: Handler, _payload: bytes) -> Run:
    return Run(2, b"", wrote_stderr=True)


def test_a_slow_claude_code_matcher_makes_paveo_test_unsure_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_matcher_trial, "TIMEOUT_S", 0.5)
    started = time.monotonic()
    out = io.StringIO()
    code = command(
        [_settings(tmp_path, SLOW_NAME)],
        project=tmp_path,
        expect=[],
        out=out,
        run=_blocks,
    )
    assert time.monotonic() - started < 10
    assert "could not be tried in time" in out.getvalue()
    assert code == 2, out.getvalue()


def test_a_slow_codex_matcher_makes_paveo_test_unsure_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_matcher_trial, "TIMEOUT_S", 0.5)
    started = time.monotonic()
    out = io.StringIO()
    code = command(
        [_codex_hooks_json(tmp_path, SLOW_NAME)],
        project=tmp_path,
        expect=[],
        out=out,
        run=_blocks,
        agent=_codex_hooks.agent("/bin/sh"),
    )
    assert time.monotonic() - started < 10
    assert "could not be tried in time" in out.getvalue()
    assert code == 2, out.getvalue()


def test_a_plain_regex_matcher_still_reaches_what_it_did(tmp_path: Path) -> None:
    out = io.StringIO()
    code = command(
        [_settings(tmp_path, "^Ba.h$")],
        project=tmp_path,
        expect=[],
        out=out,
        run=_blocks,
    )
    assert "could not" not in out.getvalue()
    assert "force-push  stopped 17 of 17" in out.getvalue()
    assert code == 1  # Read spellings of .env are not reached, so secrets is partial


@pytest.mark.parametrize("harness", [CLAUDE_CODE, CODEX, CURSOR])
def test_selftest_counts_a_slow_matcher_as_covering_nothing(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed: a hook whose matcher cannot be tried is reported as not
    seeing the call, never as covering it."""
    monkeypatch.setattr(_matcher_trial, "TIMEOUT_S", 0.5)
    started = time.monotonic()
    assert not _setup._covers(harness, SLOW_NAME, ("Bash",))
    assert time.monotonic() - started < 10
    _matcher_trial.reset()
    assert _setup._covers(harness, "Ba.h", ("Bash",))


# --------------------------------------------------------------------------
# One budget per run, and a cap on length (the review of D118)
# --------------------------------------------------------------------------


def test_many_slow_matchers_share_one_budget_not_one_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thousand distinct slow matchers would cost a thousand limits if each
    child had its own; they share one, and once it is spent none starts."""
    monkeypatch.setattr(_matcher_trial, "TIMEOUT_S", 0.5)
    started = time.monotonic()
    answers = [
        _matcher_trial.reach(f"(?:.?){{{400 + n}}}Q", ("Bash",)) for n in range(40)
    ]
    assert time.monotonic() - started < 3
    assert set(answers) == {_matcher_trial.LATE}


def test_a_spent_budget_starts_no_child(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0])
    _matcher_trial.reset(_matcher_trial.Budget(clock=lambda: next(ticks, 99.0)))

    def never(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no child may start once the budget is spent")

    _matcher_trial._BUDGET.left()  # the first try starts the budget
    monkeypatch.setattr(subprocess, "run", never)
    assert _matcher_trial.reach("Ba.h", ("Bash",)) == _matcher_trial.LATE


def test_only_answers_are_remembered_never_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_run = subprocess.run

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("cannot start")

    monkeypatch.setattr(subprocess, "run", refuse)
    assert _matcher_trial.reach("Ba.h", ("Bash",)) == _matcher_trial.FAILED
    monkeypatch.setattr(subprocess, "run", real_run)
    assert _matcher_trial.reach("Ba.h", ("Bash",)) == frozenset({"Bash"})


def test_a_matcher_longer_than_any_real_one_is_not_read_and_costs_nothing() -> None:
    """Quadratic in its length in the in-process syntax check before the cap:
    a megabyte of `[` took minutes in Codex's (the review of D118)."""
    huge = "[" * 1_000_000
    started = time.monotonic()
    assert _codex_hooks._reaches(huge)[1] == _matcher_trial.UNREAD
    assert _matcher_trial.reach("x" * 1001, ("Bash",)) == _matcher_trial.UNREAD
    from paveo._doctor import reaches_tools  # noqa: PLC0415

    assert reaches_tools("(" * 1_000_000, ("Bash",))[1] == _matcher_trial.UNREAD
    assert time.monotonic() - started < 1


def test_selftest_reads_a_codex_names_matcher_as_exact_names() -> None:
    """`Bas|apply_patc` names no tool Codex has; read as a regex search it would
    have covered both, and --selftest would pass a guard Codex never runs (the
    second review of D118)."""
    assert not _setup._covers(CODEX, "Bas|apply_patc", ("Bash", "apply_patch"))
    assert _setup._covers(CODEX, "Bash|apply_patch", ("Bash", "apply_patch"))
    assert not _setup._covers(CODEX, "(?=Bash)Bash", ("Bash",))  # Rust refuses it


def test_a_long_names_only_matcher_is_not_read_either() -> None:
    from paveo._doctor import reaches_tools  # noqa: PLC0415

    long_names = "|".join(["Bash"] * 300)
    assert reaches_tools(long_names, ("Bash",))[1] == _matcher_trial.UNREAD


def test_each_command_starts_a_fresh_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget spent by an earlier command must not answer late for the next
    one, for `paveo test` and `--selftest` alike."""
    from paveo import cli  # noqa: PLC0415

    ticks = iter([0.0])
    spent = _matcher_trial.Budget(clock=lambda: next(ticks, 99.0))
    spent.left()  # deadline at 5.0; every later reading is 99.0, so it is spent
    assert spent.left() <= 0
    seen: list[float] = []

    def record(*_args: object, **_kwargs: object) -> int:
        seen.append(_matcher_trial._BUDGET.left())
        return 0

    monkeypatch.chdir(tmp_path)
    _matcher_trial.reset(spent)
    monkeypatch.setattr("paveo._guard_test.command", record)
    cli.main(["test"])
    _matcher_trial.reset(spent)
    monkeypatch.setattr("paveo.cli.selftest", record)
    cli.main(["guard", "--selftest"])
    assert len(seen) == 2
    assert all(left > 0 for left in seen)


def test_no_child_starts_with_under_fifty_milliseconds_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0.0, 4.97])
    _matcher_trial.reset(_matcher_trial.Budget(clock=lambda: next(ticks, 4.97)))
    _matcher_trial._BUDGET.left()

    def never(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a child that cannot answer in time must not start")

    monkeypatch.setattr(subprocess, "run", never)
    assert _matcher_trial.try_all([("Ba.h", "t")], {"t": ["Bash"]}) == [
        _matcher_trial.LATE
    ]


def test_selftest_reads_a_cursor_matcher_as_its_test_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Cursor branch: JavaScript-only syntax and an untried matcher cover
    nothing; a plain regex covers what it finds."""
    assert _setup._covers(CURSOR, "Write|Delete", ("Write", "Delete"))
    assert not _setup._covers(CURSOR, "(?P<x>Write)", ("Write",))  # Python-only
    assert not _setup._covers(CURSOR, "W" * 1001, ("Write",))

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("cannot start")

    _matcher_trial.reset()
    monkeypatch.setattr(subprocess, "run", refuse)
    assert not _setup._covers(CURSOR, "Wri.e", ("Write",))
