"""Audit evidence export (D66).

The claims:

1. Only a plan that includes it exports: Developer and a lapsed key are refused
   with what to do, and nothing is written.
2. Only a log that verifies is exported. A broken chain writes nothing, not
   even a partial folder, and says which record broke.
3. The exported records are the log's own lines, byte for byte, one unbroken run
   that chains from the hash before it; the report and the CSV count the same.
4. A record being written by another process while the export runs is left out,
   never read as tampering.
5. Values from the log are escaped for a browser and for a spreadsheet.
6. The policy file ships only when the period ran under it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from conftest import records_in
from paveo import _evidence, audit
from paveo._evidence import command, export
from paveo._harnesses import CLAUDE_CODE
from paveo._licence import LICENCE_FILE, TRIAL_PREFIX
from paveo._policy_document import load_file
from paveo.audit import AuditLog, anchor_path
from paveo.cli import guard, main
from paveo.errors import ConfigError

NOW = datetime(2026, 9, 27, 12, tzinfo=UTC)
POLICY = {
    "version": 1,
    "policy_id": "evidence",
    "agents": [{"id": "claude-code", "tools": {"allow": [{"name": "Read"}]}}],
}


def trial(started: date) -> str:
    """Team, as a trial written by 0.1.0 or 0.1.1 grants it (D85)."""
    return f"{TRIAL_PREFIX}{started.isoformat()}"


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / ".paveo"
    directory.mkdir()
    (directory / "policy.json").write_text(json.dumps(POLICY), encoding="utf-8")
    (directory / LICENCE_FILE).write_text(trial(NOW.date()), encoding="utf-8")
    return directory


def days(*dates: str) -> Callable[[], datetime]:
    """A clock that gives each record the next of ``dates``, at noon UTC."""
    moments: Iterator[datetime] = (
        datetime.fromisoformat(f"{d}T12:00:00+00:00") for d in dates
    )
    return lambda: next(moments)


def write_log(home: Path, *dated: tuple[str, dict[str, object]]) -> Path:
    policy_hash = load_file(home / "policy.json").policy_hash
    path = home / "audit.jsonl"
    with AuditLog(path, now=days(*(d for d, _ in dated))) as log:
        for _, fields in dated:
            log.append(
                {
                    "agent_id": "claude-code",
                    "principal": "session-1",
                    "action": {"kind": "tool", "name": "Read"},
                    "decision": "allow",
                    "policy_id": "evidence",
                    "policy_hash": policy_hash,
                    "fail_open": False,
                    **fields,
                }
            )
    return path


REFUSED = {"decision": "deny", "reason": "tool_not_allowed", "rule": "tools.allow"}


def run(home: Path, out: Path, *extra: str) -> tuple[int, str]:
    stdout = io.StringIO()
    since = until = None
    for flag, value in zip(extra[::2], extra[1::2], strict=True):
        if flag == "--since":
            since = date.fromisoformat(value)
        else:
            until = date.fromisoformat(value)
    code = command(
        directory=home,
        log=None,
        policy=None,
        out=out,
        since=since,
        until=until,
        now=NOW,
        stdout=stdout,
    )
    return code, stdout.getvalue()


def rows(out: Path) -> list[dict[str, str]]:
    with (out / "records.csv").open(encoding="utf-8", newline="") as table:
        return list(csv.DictReader(table))


# --------------------------------------------------------------------------
# 1. The plan
# --------------------------------------------------------------------------


def test_the_developer_plan_is_refused_and_nothing_is_written(
    home: Path, tmp_path: Path
) -> None:
    (home / LICENCE_FILE).unlink()
    write_log(home, ("2026-09-26", {}))

    code, said = run(home, tmp_path / "out")

    assert code == 1
    assert "Team plan" in said
    assert "licence key" in said
    assert "paveo trial" not in said
    assert list(tmp_path.iterdir()) == [home]


def test_a_trial_that_ended_says_so(home: Path, tmp_path: Path) -> None:
    started = NOW.date() - timedelta(days=40)
    (home / LICENCE_FILE).write_text(trial(started), encoding="utf-8")
    write_log(home, ("2026-09-26", {}))

    code, said = run(home, tmp_path / "out")

    assert code == 1
    assert "team trial licence ended on" in said
    assert "paveo trial" not in said
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------
# 2-3. What is exported
# --------------------------------------------------------------------------


def test_the_export_holds_the_log_lines_and_counts_them(
    home: Path, tmp_path: Path
) -> None:
    log = write_log(
        home,
        ("2026-09-25", {}),
        ("2026-09-25", REFUSED),
        ("2026-09-26", {"decision": "would_deny", "reason": "x", "rule": "y"}),
        ("2026-09-26", {}),
    )
    out = tmp_path / "out"

    code, said = run(home, out)

    assert code == 0, said
    assert "4 records (seq 1-4)" in said
    assert (out / "audit.jsonl").read_bytes() == log.read_bytes()
    assert [r["decision"] for r in rows(out)] == [
        "allow",
        "deny",
        "would_deny",
        "allow",
    ]
    report = (out / "report.html").read_text(encoding="utf-8")
    # allowed 2, refused 1, would refuse 1, nothing settled.
    assert "<td>claude-code</td><td>2</td><td>1</td><td>1</td><td>$0</td>" in report
    assert "<td>tool_not_allowed</td><td>tools.allow</td><td>1</td>" in report
    assert "matches its anchor" in said
    for name in ("audit.jsonl", "records.csv", "policy.json"):
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert f"<td>{name}</td><td>{digest}</td>" in report


def test_the_period_is_one_run_that_chains_from_the_hash_before_it(
    home: Path, tmp_path: Path
) -> None:
    log = write_log(
        home,
        ("2026-09-24", {}),
        ("2026-09-25", REFUSED),
        ("2026-09-25", {}),
        ("2026-09-26", {}),
    )
    out = tmp_path / "out"

    code, said = run(home, out, "--since", "2026-09-25", "--until", "2026-09-25")

    assert code == 0, said
    everything = records_in(log)
    exported = records_in(out / "audit.jsonl")
    assert [r["seq"] for r in exported] == [2, 3]
    report = (out / "report.html").read_text(encoding="utf-8")
    assert f"<td>before record 2</td><td>{everything[0]['hash']}</td>" in report
    assert f"<td>record 3</td><td>{everything[2]['hash']}</td>" in report
    # The whole log was verified, not just the period.
    assert "verified from record 1 to 4" in said


def test_a_clock_that_went_back_is_refused_not_silently_trimmed(
    home: Path, tmp_path: Path
) -> None:
    """Record 3 is dated inside the period but comes after one dated past it.
    Leaving it out would hand an auditor a period with a record missing and
    nothing to say so (D77)."""
    write_log(home, ("2026-09-25", {}), ("2026-09-26", {}), ("2026-09-25", {}))
    out = tmp_path / "out"

    code, said = run(home, out, "--until", "2026-09-25")

    assert code == 1
    assert "clock went back" in said
    assert "pass --until 2026-09-26 or later" in said
    assert not out.exists()
    code, said = run(home, out, "--until", "2026-09-26")
    assert code == 0, said
    assert [r["seq"] for r in records_in(out / "audit.jsonl")] == [1, 2, 3]


def test_a_broken_chain_is_reported_before_a_clock_that_went_back(
    home: Path, tmp_path: Path
) -> None:
    """Tampering is the finding that matters; a remedy about --until would point
    the operator away from it (D77)."""
    log = write_log(
        home,
        ("2026-09-25", {}),
        ("2026-09-26", {}),
        ("2026-09-25", {}),
        ("2026-09-26", REFUSED),
    )
    lines = log.read_bytes().splitlines()
    lines[3] = lines[3].replace(b'"deny"', b'"allow"')
    log.write_bytes(b"\n".join(lines) + b"\n")

    code, said = run(home, tmp_path / "out", "--until", "2026-09-25")

    assert code == 1
    assert "does not verify: record 4" in said
    assert "clock went back" not in said


def test_a_record_past_the_period_is_never_read_for_its_date() -> None:
    """Records after the run are not exported, so an unreadable date among them
    decides nothing; before D77 they were not parsed at all."""
    until = date(2026, 9, 30)
    assert _evidence._period("after", "not-a-date", None, None, until) == "after"
    with pytest.raises(ConfigError, match="not a date"):
        _evidence._period("in", "not-a-date", None, None, until)


def test_a_period_with_no_records_is_refused(home: Path, tmp_path: Path) -> None:
    write_log(home, ("2026-09-20", {}))

    code, said = run(home, tmp_path / "out", "--since", "2026-09-25")

    assert code == 1
    assert "no records between 2026-09-25" in said
    assert list(tmp_path.iterdir()) == [home]


def test_a_period_backwards_is_refused(home: Path, tmp_path: Path) -> None:
    write_log(home, ("2026-09-20", {}))

    code, said = run(
        home, tmp_path / "out", "--since", "2026-09-26", "--until", "2026-09-20"
    )

    assert code == 1
    assert "after it ends" in said


def test_an_existing_folder_is_never_written_into(home: Path, tmp_path: Path) -> None:
    write_log(home, ("2026-09-26", {}))
    out = tmp_path / "out"
    out.mkdir()

    code, said = run(home, out)

    assert code == 1
    assert "could not be created" in said
    assert list(out.iterdir()) == []


def test_a_broken_chain_writes_nothing_and_names_the_record(
    home: Path, tmp_path: Path
) -> None:
    log = write_log(home, ("2026-09-25", REFUSED), ("2026-09-26", {}))
    lines = log.read_bytes().splitlines()
    lines[0] = lines[0].replace(b'"deny"', b'"allow"')
    log.write_bytes(b"\n".join(lines) + b"\n")

    code, said = run(home, tmp_path / "out")

    assert code == 1
    assert "does not verify: record 1" in said
    # No partial folder either, hidden or not.
    assert list(tmp_path.iterdir()) == [home]


def test_records_removed_from_the_end_are_caught(home: Path, tmp_path: Path) -> None:
    log = write_log(home, ("2026-09-25", {}), ("2026-09-26", {}))
    log.write_bytes(log.read_bytes().splitlines(keepends=True)[0])

    code, said = run(home, tmp_path / "out")

    assert code == 1
    assert "removed from the end" in said


def test_the_recipe_the_report_gives_recomputes_the_chain(
    home: Path, tmp_path: Path
) -> None:
    """Followed as the report words it, with nothing from Paveo: an auditor who
    does this reaches the hash the report prints."""
    write_log(home, ("2026-09-25", {}), ("2026-09-26", REFUSED))
    out = tmp_path / "out"
    run(home, out)

    previous = "sha256:" + "0" * 64
    for line in (out / "audit.jsonl").read_bytes().splitlines():
        record = json.loads(line)
        claimed = record.pop("hash")
        assert record["prev_hash"] == previous
        body = json.dumps(record, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256((body + record["prev_hash"]).encode("ascii"))
        assert claimed == f"sha256:{digest.hexdigest()}"
        previous = claimed
    report = (out / "report.html").read_text(encoding="utf-8")
    assert f"<td>record 2</td><td>{previous}</td>" in report


# --------------------------------------------------------------------------
# 4. A write in progress
# --------------------------------------------------------------------------


@pytest.mark.parametrize("anchored", [True, False])
def test_a_record_begun_after_the_export_started_is_left_out(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, anchored: bool
) -> None:
    """Another process starts appending the moment after the snapshot: its half
    line is past the snapshot, so it is neither exported nor called tampering,
    with or without an anchor (/code-review)."""
    log = write_log(home, ("2026-09-26", {}), ("2026-09-26", {}))
    if not anchored:
        anchor_path(log).unlink()

    def then_a_writer_starts(location: Path) -> audit.Snapshot:
        taken = audit.snapshot(location)
        with location.open("ab") as handle:
            handle.write(b'{"action":{"kind":"tool","na')
        return taken

    monkeypatch.setattr(_evidence, "snapshot", then_a_writer_starts)
    out = tmp_path / "out"

    code, said = run(home, out)

    assert code == 0, said
    assert len(records_in(out / "audit.jsonl")) == 2


def test_a_log_with_no_anchor_exports_and_says_so(home: Path, tmp_path: Path) -> None:
    write_log(home, ("2026-09-26", {}))
    anchor_path(home / "audit.jsonl").unlink()
    out = tmp_path / "out"

    code, said = run(home, out)

    assert code == 0, said
    assert "but has no anchor" in said
    assert "no anchor file was found" in (out / "report.html").read_text("utf-8")


# --------------------------------------------------------------------------
# 5. Escaping
# --------------------------------------------------------------------------


def test_values_are_escaped_for_the_browser_and_the_spreadsheet(
    home: Path, tmp_path: Path
) -> None:
    write_log(
        home,
        (
            "2026-09-26",
            {
                "agent_id": "<script>alert(1)</script>",
                "principal": '=HYPERLINK("http://x","y")',
                "rule": "x;=HYPERLINK(CHAR(104)&A1);y",
            },
        ),
    )
    out = tmp_path / "out"

    code, said = run(home, out)

    assert code == 0, said
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    (row,) = rows(out)
    assert row["principal"] == '\'=HYPERLINK("http://x","y")'
    # Where `;` separates cells, a formula can start mid-value (/security-review).
    assert row["rule"].startswith("'")


# --------------------------------------------------------------------------
# 6. The policy
# --------------------------------------------------------------------------


def test_a_policy_the_period_never_ran_under_is_left_out(
    home: Path, tmp_path: Path
) -> None:
    write_log(home, ("2026-09-26", {}))
    changed = {**POLICY, "policy_id": "evidence-v2"}
    (home / "policy.json").write_text(json.dumps(changed), encoding="utf-8")
    out = tmp_path / "out"

    code, said = run(home, out)

    assert code == 0, said
    assert not (out / "policy.json").exists()
    assert "policy.json is not included" in said
    assert "is not one the period ran under" in (out / "report.html").read_text("utf-8")


def test_fail_open_records_are_called_out(home: Path, tmp_path: Path) -> None:
    write_log(home, ("2026-09-26", {"fail_open": True}), ("2026-09-26", {}))
    out = tmp_path / "out"

    run(home, out)

    report = (out / "report.html").read_text("utf-8")
    assert "set to fail open for 1 record<" in report
    assert "not whether it was ever used" in report


# --------------------------------------------------------------------------
# End to end: the guard writes, the command exports
# --------------------------------------------------------------------------


def test_the_command_exports_what_the_guard_decided(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for tool in ("Read", "Write"):
        guard(
            CLAUDE_CODE,
            directory=home,
            agent="claude-code",
            stdin=io.BytesIO(
                json.dumps(
                    {
                        "session_id": "abc",
                        "hook_event_name": "PreToolUse",
                        "tool_name": tool,
                        "tool_input": {},
                    }
                ).encode()
            ),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            stopped=lambda: False,
        )
    # The trial was written for NOW; the command reads the real clock.
    (home / LICENCE_FILE).write_text(trial(datetime.now(UTC).date()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["evidence", "--dir", str(home), "--out", "evidence"]) == 0

    decided = [r["decision"] for r in rows(tmp_path / "evidence")]
    assert decided == ["allow", "deny"]
    assert (tmp_path / "evidence" / "policy.json").exists()


def test_export_refuses_a_log_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="could not be read"):
        export(
            log=tmp_path / "missing.jsonl",
            policy=tmp_path / "policy.json",
            out=tmp_path / "out",
            since=None,
            until=NOW.date(),
            now=NOW,
        )
    assert list(tmp_path.iterdir()) == []
