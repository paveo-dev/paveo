"""Claims the repo makes about itself, asserted rather than remembered.

Neither of these is about library behaviour. They exist because a licence that
drifts by a word and a README that outlives the code it describes are both
defects the rest of the gate — ruff, mypy, pytest, the egress check — cannot
see, and both would be discovered by someone we would rather not have find
them. Rule 5: if correctness depends on remembering, it is eventually wrong.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import paveo

ROOT = Path(__file__).parent.parent

# sha256 of the Elastic License 2.0 body, fetched 2026-09-20 from
# raw.githubusercontent.com/elastic/elasticsearch/main/licenses/ELASTIC-LICENSE-2.0.txt
# The copyright header above it is ours; everything from the title down is not.
ELASTIC_2_0_SHA256 = "48255018b41fc0e965b1115af7e6779bc218bb8a6747d561da800d5022622aa2"
ELASTIC_TITLE = b"Elastic License 2.0\n"
# sha256 of the Apache License 2.0 text from its title down, fetched 2026-10-15
# from www.apache.org/licenses/LICENSE-2.0.txt (D95).
APACHE_2_0_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
APACHE_TITLE = b"                                 Apache License\n"
LICENSOR = b"Copyright (c) 2026 MD Faiz Jamal\n"
ELASTIC_SPDX = "SPDX-License-Identifier: Elastic-2.0"


def _canonical_body(name: str, title: bytes, digest: str) -> None:
    """A reworded licence clause is a legal defect no other check would catch.

    Both licences are shipped *unmodified*, and both carry notices a licensee
    must keep. This pins each body so an accidental edit, a reflow, an editor
    stripping trailing whitespace or a well-meaning tidy-up, fails here.
    """
    raw = (ROOT / name).read_bytes()
    assert raw.startswith(LICENSOR), f"{name} must name the licensor"
    assert raw.count(title) == 1, f"{name}: the licence title must appear once"
    body = raw[raw.index(title) :]
    assert hashlib.sha256(body).hexdigest() == digest, (
        f"{name} is not the canonical licence text. Do not update this hash to "
        "make the test pass: restore the text, or record a deliberate relicensing "
        "decision first."
    )


def test_the_licence_is_canonical_apache_2_0() -> None:
    _canonical_body("LICENSE", APACHE_TITLE, APACHE_2_0_SHA256)


def test_the_paid_files_licence_is_canonical_elastic_2_0() -> None:
    _canonical_body("LICENSE-ELASTIC", ELASTIC_TITLE, ELASTIC_2_0_SHA256)


def test_the_elastic_files_are_exactly_the_ones_listed() -> None:
    """LICENSE-ELASTIC names its files; each carries the SPDX line, and no other
    file does. A paid file added without both, or one listed that went back to
    Apache, would leave a reader unable to tell which terms apply (D95)."""
    raw = (ROOT / "LICENSE-ELASTIC").read_text("utf-8")
    preamble = raw[: raw.index(ELASTIC_TITLE.decode())]
    listed = {line.strip() for line in preamble.splitlines() if line.startswith("    ")}
    marked = {
        path.relative_to(ROOT).as_posix()
        for tree in ("src", "tools", "rollout", "plugins")
        for path in (ROOT / tree).rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh", ""}
        and ELASTIC_SPDX in path.read_text("utf-8", errors="replace")[:400]
    }
    assert listed == marked == {"src/paveo/_evidence.py"}


def test_the_free_core_never_loads_the_elastic_file() -> None:
    """A company declined Paveo because licence-keyed, non-OSI code would sit
    in their guard path (D95). Importing the CLI, which holds the
    guard, doctor, init and every free command, must not load ``_evidence``."""
    script = "import sys, paveo, paveo.cli\nprint('paveo._evidence' in sys.modules)\n"
    result = subprocess.run(  # noqa: S603 - our own interpreter and our own script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert result.stdout.strip() == "False"


def test_the_readme_does_not_claim_what_is_not_built() -> None:
    """The README's "Not built" list, asserted rather than maintained by hand.

    Prime directive 5: every claim in the README must be true and testable. This
    list has already drifted once — it was prose, and a reader could copy a line
    that did not work. The day ``async_session`` lands, this fails and the
    README has to be corrected in the same commit rather than quietly becoming
    false (Rule 8). ``wrap_anthropic`` left this list with S6 (D44).
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for name in ("async_session",):
        assert f"`{name}`" in readme, (
            f"README no longer mentions {name}; if it shipped, update the README "
            "and this test in the same commit"
        )
        assert name not in paveo.__all__, f"{name} is public but README says unbuilt"
        assert not hasattr(paveo.Session, name), f"Session.{name} exists"
        assert not hasattr(paveo.Paveo, name), f"Paveo.{name} exists"


def test_the_readme_first_policy_refuses_what_the_readme_says(tmp_path: Path) -> None:
    """The five-minute example, read from the README itself, so the page a
    stranger copies from cannot drift from what it does (prime directive 5)."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("```json\n", 1)[1].split("```", 1)[0]
    pf = paveo.Paveo.from_policy(json.loads(block), audit_path=tmp_path / "audit.jsonl")
    with pf, pf.session(agent_id="refund-bot", principal="user_123") as s:
        with pytest.raises(paveo.PolicyDenied, match="requires"):
            s.check_tool("refund", {"order_id": "A-1", "amount_usd": 100})
        s.check_tool("lookup_order", {"order_id": "A-1"})
        s.check_tool("refund", {"order_id": "A-1", "amount_usd": 100})
        with pytest.raises(paveo.PolicyDenied, match="max"):
            s.check_tool("refund", {"order_id": "A-1", "amount_usd": 600})
        with pytest.raises(paveo.PolicyDenied):
            s.check_tool("delete_customer", {})


def test_windows_is_told_why_not_shown_a_traceback() -> None:
    """The README says macOS and Linux; on Windows the import says so too, and
    names what works instead, rather than failing on a missing module."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    run = subprocess.run(
        [sys.executable, "-c", "import sys; sys.platform = 'win32'; import paveo"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert run.returncode != 0
    assert "Paveo runs on macOS and Linux" in run.stderr
    assert "WSL" in run.stderr
    assert "fcntl" not in run.stderr


def test_every_file_the_security_review_cites_exists() -> None:
    """The review answers a security team by pointing at evidence. A citation to a
    file that was renamed or deleted would be an answer nobody can check, so the
    page fails the build the day it drifts (Rule 5). The source distribution does
    not ship ``.github/``, so there those citations are checked by the repository's
    own run of this test instead."""
    text = (ROOT / "docs" / "SECURITY_REVIEW.md").read_text(encoding="utf-8")
    cited = set(
        re.findall(
            r"`((?:src|tests|docs|\.github)/[\w./-]+|SECURITY\.md|LICENSE|pyproject\.toml)`",
            text,
        )
    )
    assert cited, "the review cites no files; the pattern above has drifted"
    if not (ROOT / ".github").is_dir():
        cited = {path for path in cited if not path.startswith(".github/")}
    missing = sorted(path for path in cited if not (ROOT / path).exists())
    assert not missing, f"SECURITY_REVIEW.md cites files that do not exist: {missing}"


def test_the_readme_states_the_overhead_budgets_the_gate_enforces() -> None:
    """The README's overhead table names the budgets ``make check`` fails over,
    so a budget raised in the gate cannot leave the page promising less (D73)."""
    from tests.overhead import BUDGET_US  # noqa: PLC0415

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("\n## Overhead\n", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ")][1:]
    assert len(rows) == len(BUDGET_US), "one README row per gated figure, in order"
    for row, (name, budget) in zip(rows, BUDGET_US.items(), strict=True):
        stated = f"{budget / 1000:g} ms" if budget >= 1000 else f"{budget:g} µs"
        cells = [cell.strip() for cell in row.strip("| ").split("|")]
        assert cells[-1].startswith(stated), f"README row for {name}: {cells[-1]!r}"
