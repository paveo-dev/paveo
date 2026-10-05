"""Audit evidence: a period of the audit log, packaged for an auditor (D66).

``paveo evidence`` writes one folder:

- ``report.html``: what an auditor reads, or prints to PDF. Who was allowed and
  refused what, under which policy, by which rule, and what the hash chain proves.
- ``records.csv``: every record in the period, the population an auditor samples.
- ``audit.jsonl``: the same records, byte for byte as the log holds them, so the
  chain can be recomputed without trusting the report.
- ``policy.json``: the policy file, only when its hash is one the period ran
  under, since a policy that was never in force is not evidence of anything.

**It exports only a log that verifies.** Evidence from a broken chain is not
evidence, so the whole log is checked, from record 1, before the folder appears;
where it breaks, nothing is written and the break is named. It reads as far as
the log reached when it began (``audit.snapshot``), so an agent appending while
it runs is neither exported half-written nor mistaken for tampering.

**Tamper-evident, not tamper-proof**, and the report says so. What the export
adds is a copy of the chain's hash held by someone else: once the folder is with
an auditor, rewriting any record it covers changes that hash.

**No payloads**: the log holds none (locked decision #5), so neither can this.
Every value is escaped for the file it lands in, HTML or a spreadsheet, because a
log someone rewrote is still read by an auditor's browser and Excel.

A paid feature, Team and up (D61). Opens no socket, like everything here.
"""

from __future__ import annotations

import csv
import hashlib
import html
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TextIO

from ._licence import plan_in
from ._policy_document import load_file
from .audit import BrokenChain, snapshot, verified_records
from .errors import ConfigError

_REPORT, _CSV = "report.html", "records.csv"
_SLICE, _POLICY = "audit.jsonl", "policy.json"
_COLUMNS = (
    "seq",
    "ts",
    "agent_id",
    "principal",
    "action_kind",
    "tool_or_model",
    "decision",
    "reason",
    "rule",
    "estimated_cost_usd",
    "actual_cost_usd",
    "fail_open",
    "policy_id",
    "policy_hash",
    "plan",
    "reservation_id",
    "hash",
)
_DECISIONS = {"deny": "refused", "would_deny": "would refuse (shadow)"}
_STYLE = (
    "body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;"
    "padding:0 1rem;color:#111;background:#fff}"
    "table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}"
    "th,td{border:1px solid #bbb;padding:.3rem .5rem;text-align:left;"
    "vertical-align:top;overflow-wrap:anywhere}"
    "th{background:#f2f2f2}.warn{background:#fff3cd;padding:.5rem}"
)
# A spreadsheet runs a cell that starts with one of these as a formula, and in a
# locale whose list separator is `;`, a `;` starts a new cell mid-value
# (/security-review, D66). Nothing Paveo writes in these fields needs them.
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_FORMULA_ANYWHERE = frozenset(";=+@\t\r")


@dataclass(frozen=True, slots=True)
class Exported:
    """What was written, for the command to say."""

    first_seq: int
    last_seq: int
    records: int
    chain_records: int
    anchored: bool
    policy_included: bool


@dataclass(slots=True)
class _Agent:
    decisions: Counter[str] = field(default_factory=Counter)
    settled_usd: Decimal = Decimal(0)


@dataclass(slots=True)
class _Policy:
    policy_id: str
    first: str
    last: str
    records: int = 0


@dataclass(slots=True)
class _Tally:
    """The period's records, counted as they stream past."""

    agents: dict[str, _Agent] = field(default_factory=dict)
    refusals: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    policies: dict[str, _Policy] = field(default_factory=dict)
    fail_open: int = 0
    first_seq: int = 0
    last_seq: int = 0
    prev_hash: str = ""
    last_hash: str = ""
    first_ts: str = ""
    last_ts: str = ""

    def add(self, seq: int, record: dict[str, object]) -> None:
        ts = _text(record.get("ts"))
        if not self.first_seq:
            self.first_seq, self.prev_hash, self.first_ts = (
                seq,
                _text(record.get("prev_hash")),
                ts,
            )
        self.last_seq, self.last_hash, self.last_ts = seq, _text(record["hash"]), ts
        decision = _text(record.get("decision"))
        agent = self.agents.setdefault(_text(record.get("agent_id")), _Agent())
        agent.decisions[decision] += 1
        if decision == "settle":
            agent.settled_usd += _usd(record.get("actual_cost_usd"), seq)
        if decision in {"deny", "would_deny"}:
            reason, rule = _text(record.get("reason")), _text(record.get("rule"))
            self.refusals[(decision, reason, rule)] += 1
        digest = _text(record.get("policy_hash"))
        policy = self.policies.setdefault(
            digest, _Policy(_text(record.get("policy_id")), ts, ts)
        )
        policy.last, policy.records = ts, policy.records + 1
        if record.get("fail_open") is True:
            self.fail_open += 1


def command(  # noqa: PLR0913 - keyword-only; every path and the clock are injected (Rule 14)
    *,
    directory: Path,
    log: Path | None,
    policy: Path | None,
    out: Path,
    since: date | None,
    until: date | None,
    now: datetime,
    stdout: TextIO,
) -> int:
    """``paveo evidence``: check the plan, export, and say what was written."""
    try:
        plan = plan_in(directory, today=now.date())
        if "evidence" not in plan.features:
            stdout.write(_upgrade(plan.lapsed, plan.expires, directory))
            return 1
        done = export(
            log=log or directory / "audit.jsonl",
            policy=policy or directory / _POLICY,
            out=out,
            since=since,
            until=until or now.date(),
            now=now,
        )
    except ConfigError as e:
        stdout.write(f"paveo: {e} Nothing was written.\n")
        return 1
    anchor = "and matches its anchor" if done.anchored else "but has no anchor"
    policy_note = (
        "" if done.policy_included else f" {_POLICY} is not included: see the report."
    )
    stdout.write(
        f"paveo: {_records(done.records)} (seq {done.first_seq}-{done.last_seq}) "
        f"written to {out}. The chain verified from record 1 to "
        f"{done.chain_records}, {anchor}.{policy_note}\n"
    )
    return 0


def _upgrade(lapsed: str | None, expires: date | None, directory: Path) -> str:
    """What to do, for the plan in force."""
    key = f"put a licence key for Team or up in {directory}/licence.key"
    if lapsed is None:
        return (
            f"paveo: audit evidence is part of the Team plan and up. To export, "
            f"{key}.\n"
        )
    return (
        f"paveo: the {lapsed} licence ended on {expires}, and audit evidence is "
        f"part of the Team plan and up. To export, {key}.\n"
    )


def export(  # noqa: PLR0913 - keyword-only; every path and the clock are injected (Rule 14)
    *,
    log: Path,
    policy: Path,
    out: Path,
    since: date | None,
    until: date,
    now: datetime,
) -> Exported:
    """Write the evidence for ``since``..``until`` (UTC dates, inclusive) to
    ``out``, which must not exist; ``ConfigError`` if the log does not verify,
    or the period holds no records, and then nothing is left behind.

    The period is one unbroken run of records, so it chains: from the first
    record dated on or after ``since`` to the last before one dated after
    ``until``.
    """
    if since is not None and since > until:
        raise ConfigError(
            f"the period starts on {since}, after it ends on {until}.",
            remedy="check --since and --until; --until is today unless given.",
        )
    # `mkdir` claims the name or fails if it is taken, so nothing already there
    # is ever written into or replaced (/code-review). The files are built beside
    # it and moved in at the end, the report last: a folder holding report.html
    # is a whole export.
    try:
        out.mkdir()
    except OSError as e:
        raise ConfigError(
            f"{out} could not be created ({e.strerror}).",
            remedy="choose an --out that does not exist yet, somewhere writable.",
        ) from e
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    except OSError as e:
        out.rmdir()
        raise ConfigError(
            f"{out} could not be written ({e.strerror}).",
            remedy="check the space and permissions where --out points.",
        ) from e
    try:
        done = _write(
            log=log, policy=policy, staging=staging, since=since, until=until, now=now
        )
        for name in (_SLICE, _CSV, _POLICY, _REPORT):
            if (staging / name).exists():
                os.rename(staging / name, out / name)
        staging.rmdir()
    except BaseException as e:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
        if isinstance(e, OSError):
            raise ConfigError(
                f"the evidence could not be written ({e.strerror}).",
                remedy="check the space and permissions where --out points.",
            ) from e
        raise
    return done


def _write(  # noqa: PLR0913 - one export's inputs, passed straight through
    *,
    log: Path,
    policy: Path,
    staging: Path,
    since: date | None,
    until: date,
    now: datetime,
) -> Exported:
    tally = _Tally()
    taken = snapshot(log)
    anchor = taken.anchor
    chain_records = 0
    with (
        (staging / _SLICE).open("wb") as lines,
        (staging / _CSV).open("w", encoding="utf-8", newline="") as table,
    ):
        rows = csv.writer(table)
        rows.writerow(_COLUMNS)
        state = "before"
        latest: date | None = None  # the latest date seen past the period
        went_back: tuple[str, date] | None = None
        try:
            for line, record in verified_records(log, anchor=anchor, size=taken.size):
                chain_records += 1
                day = _text(record.get("ts"))[:10]
                dated = _date(day)
                state = _period(state, day, dated, since, until)
                if state == "back":
                    # Kept going: a broken chain must be reported as that.
                    if latest is not None:
                        went_back = (day, latest)
                    state = "after"
                elif state == "after" and dated and (not latest or dated > latest):
                    latest = dated
                elif state == "in":
                    lines.write(line + b"\n")
                    rows.writerow(_row(record))
                    tally.add(chain_records, record)
        except BrokenChain as e:
            raise ConfigError(
                f"the audit log at {log} does not verify: record "
                f"{e.status.broken_at}: {e.status.detail}.",
                remedy=(
                    "evidence from a broken chain is not evidence. Keep the log "
                    "as it is for whoever investigates, and start a new one."
                ),
            ) from e
    if went_back:
        day, needed = went_back
        raise ConfigError(
            f"a record dated {day} comes after records dated as late as {needed}: "
            "the clock went back while the log was written, so the period is not "
            "one unbroken run of it.",
            remedy=f"pass --until {needed} or later, so the export covers that run.",
        )
    if not tally.first_seq:
        raise ConfigError(
            f"the audit log holds no records between {since or 'its start'} "
            f"and {until}.",
            remedy="check --since and --until, and that --log is the right log.",
        )
    included, policy_note = _copy_policy(policy, staging, tally)
    files = [name for name in (_SLICE, _CSV, _POLICY) if (staging / name).exists()]
    report = _report(
        tally,
        since=since,
        until=until,
        now=now,
        chain_records=chain_records,
        anchored=anchor is not None,
        policy_note=policy_note,
        digests={name: _sha256(staging / name) for name in files},
    )
    (staging / _REPORT).write_text(report, encoding="utf-8")
    return Exported(
        first_seq=tally.first_seq,
        last_seq=tally.last_seq,
        records=tally.last_seq - tally.first_seq + 1,
        chain_records=chain_records,
        anchored=anchor is not None,
        policy_included=included,
    )


def _date(day: str) -> date | None:
    try:
        return date.fromisoformat(day)
    except ValueError:
        return None


def _period(
    state: str, day: str, dated: date | None, since: date | None, until: date
) -> str:
    """``before``, ``in``, ``after`` or ``back`` for one record of the period.

    The period is one unbroken run, so the slice still verifies as a chain. A
    record dated inside it after the run has ended is ``back``: the clock went
    back, and leaving that record out would hand an auditor a period with records
    silently missing, so the export refuses once the whole chain has verified.
    Records after the run are not exported, so an unreadable date there decides
    nothing; anywhere else it refuses.
    """
    if state == "after":
        if dated and dated <= until and (since is None or dated >= since):
            return "back"
        return state
    if dated is None:
        raise ConfigError(
            f"a record carries the timestamp {day!r}, which is not a date.",
            remedy="the log was not written by Paveo; it cannot be exported.",
        )
    if dated > until:
        return "after" if state == "in" else "before"
    if state == "in" or since is None or dated >= since:
        return "in"
    return state


def _copy_policy(policy: Path, staging: Path, tally: _Tally) -> tuple[bool, str]:
    """Include the policy file if the period ran under it, and say why not.

    Copied once and judged by the copy, so what ships is what was hashed."""
    copy = staging / _POLICY
    try:
        copy.write_bytes(policy.read_bytes())
        digest = load_file(copy).policy_hash
    except (OSError, ConfigError) as e:
        copy.unlink(missing_ok=True)
        return False, f"The policy file could not be read, so it is not included: {e}"
    if digest not in tally.policies:
        copy.unlink()
        return False, (
            f"The policy file at the time of export ({digest}) is not one the "
            f"period ran under, so it is not included. Earlier versions are in "
            f"your version control; each hashes to the value shown above."
        )
    return True, f"{_POLICY} in this folder is the policy with hash {digest}."


def _report(  # noqa: PLR0913 - keyword-only; the report's inputs
    tally: _Tally,
    *,
    since: date | None,
    until: date,
    now: datetime,
    chain_records: int,
    anchored: bool,
    policy_note: str,
    digests: dict[str, str],
) -> str:
    """The page an auditor reads. Every value from the log is escaped."""
    first, last = tally.first_seq, tally.last_seq
    asked = f"{since or 'the start of the log'} to {until}"
    anchor = (
        "and the log reaches the record its anchor names, so none were removed "
        "from the end."
        if anchored
        else "<strong>but no anchor file was found beside the log, so records "
        "removed from the end could not be detected.</strong>"
    )
    fail_open = (
        f"<p class=warn><strong>The policy was set to fail open for "
        f"{_records(tally.fail_open)}</strong>: had Paveo been unable to judge one "
        "of those calls, it would have let it through rather than refuse it. The "
        "setting is recorded, not whether it was ever used.</p>"
        if tally.fail_open
        else "<p>The policy was set to fail closed for every record: any call "
        "Paveo could not judge was refused.</p>"
    )
    agents = _table(
        ("agent", "allowed", "refused", "would refuse (shadow)", "spend settled"),
        [
            (
                name,
                a.decisions["allow"],
                a.decisions["deny"],
                a.decisions["would_deny"],
                f"${a.settled_usd}",
            )
            for name, a in sorted(tally.agents.items())
        ],
    )
    refusals = (
        _table(
            ("decision", "reason", "rule", "records"),
            [
                (_DECISIONS[decision], reason, rule, count)
                for (decision, reason, rule), count in sorted(tally.refusals.items())
            ],
        )
        if tally.refusals
        else "<p>No call was refused in the period.</p>"
    )
    policies = _table(
        ("policy_id", "policy_hash", "first record", "last record", "records"),
        [
            (p.policy_id, digest, p.first, p.last, p.records)
            for digest, p in tally.policies.items()
        ],
    )
    chain = _table(
        ("", "hash"),
        [
            (f"before record {first}", tally.prev_hash),
            (f"record {last}", tally.last_hash),
        ],
    )
    files = _table(("file", "sha256"), list(digests.items()))
    return "\n".join(
        (
            "<!doctype html>",
            "<html lang=en><head><meta charset=utf-8>",
            f"<title>Paveo audit evidence, {_e(asked)}</title>",
            f"<style>{_STYLE}</style></head><body>",
            "<h1>Paveo audit evidence</h1>",
            f"<p>Period asked for: {_e(asked)}, UTC. Records exported: "
            f"{last - first + 1} (seq {first} to {last}, {_e(tally.first_ts)} to "
            f"{_e(tally.last_ts)}). Generated "
            f"{_e(now.isoformat(timespec='seconds'))} by Paveo {_e(_version())}.</p>",
            "<h2>What this shows</h2>",
            "<p>Every decision Paveo made in the period: which agent, which tool or "
            "model, whether the call was allowed or refused, and by which rule. "
            "Paveo decides before the call is made, and a refusal stops it there. "
            "A call shadow mode would have refused went ahead, and is counted "
            "apart. Paveo never records prompts, completions or tool arguments, so "
            "none are here.</p>",
            fail_open,
            "<h2>Integrity</h2>",
            f"<p>The log's hash chain verified from record 1 to record "
            f"{chain_records}, {anchor} Each record's hash covers the record and "
            "the hash before it, so editing any record breaks every hash after "
            "it.</p>",
            "<p><strong>Tamper-evident, not tamper-proof.</strong> Someone with "
            "write access to the log could rewrite the whole chain. Once this "
            "folder is with you, that is detectable for every record it covers: "
            f"record {last} of the live log must still carry the hash below.</p>",
            chain,
            f"<p>To recompute it from <code>{_SLICE}</code>: take a line, remove "
            "its <code>hash</code> field, and write what is left as JSON with keys "
            "sorted, no spaces and non-ASCII escaped. Append its "
            "<code>prev_hash</code> exactly as written, including "
            "<code>sha256:</code>. The SHA-256 of those bytes, in hex after "
            "<code>sha256:</code>, is its <code>hash</code>, and is the next "
            "line's <code>prev_hash</code>.</p>",
            "<h2>Agents</h2>",
            agents,
            "<h2>Refusals by rule</h2>",
            refusals,
            "<h2>Policies in force</h2>",
            policies,
            f"<p>{_e(policy_note)}</p>",
            "<h2>Files</h2>",
            files,
            "</body></html>",
            "",
        )
    )


def _table(headers: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_e(str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _row(record: dict[str, object]) -> list[str]:
    action = record.get("action")
    kind, target = (
        (_text(action.get("kind")), _text(action.get("name") or action.get("model")))
        if isinstance(action, dict)
        else ("", "")
    )
    values = {
        **{name: _text(record.get(name)) for name in _COLUMNS},
        "action_kind": kind,
        "tool_or_model": target,
    }
    return [_cell(values[name]) for name in _COLUMNS]


def _cell(value: str) -> str:
    """A spreadsheet must show the value, never run it."""
    risky = value.startswith(_FORMULA_START) or not _FORMULA_ANYWHERE.isdisjoint(value)
    return "'" + value if risky else value


def _records(count: int) -> str:
    return f"{count} record" if count == 1 else f"{count} records"


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _usd(value: object, seq: int) -> Decimal:
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        amount = Decimal("NaN")
    if not amount.is_finite():
        raise ConfigError(
            f"record {seq} carries a cost that is not a number.",
            remedy="the log was not written by Paveo; it cannot be exported.",
        )
    return amount


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _version() -> str:
    try:
        return version("paveo")
    except PackageNotFoundError:
        return "(run from source, not installed)"
