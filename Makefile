# Clone to green tests in one command (Rule 19). No tribal knowledge.
#
#   make dev     once, to build .venv and install the package editable
#   make check   the verification gate, in order, before a commit

PY := .venv/bin/python

# The gate finds the package by path, never through an editable install. On this
# machine, files pip writes into the venv get flagged UF_HIDDEN shortly after
# they are created, and since Python 3.13 `site.addpackage` silently skips hidden
# .pth files — so `pip install -e .` reports success and leaves the package
# unimportable, with no error anywhere. A verification gate that can be defeated
# by a filesystem flag is not a gate (Rule 5).
export PYTHONPATH := src

.PHONY: dev check lint types test egress overhead example release-check clean

# The editable install is a convenience for working in a REPL. Nothing in the
# gate relies on it, deliberately — see the PYTHONPATH note above.
dev:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

types:
	$(PY) -m mypy --strict src/

test:
	$(PY) -m pytest -q

# Standalone ON PURPOSE: pytest plugins can open sockets themselves and mask
# the result. This is the run that counts.
egress:
	$(PY) -m tests.test_no_egress

# What a call costs us, on the real clock; fails over the budget (Rule 17, D73).
# Standalone because it measures this machine, which is not a unit test's job.
overhead:
	$(PY) -m tests.overhead

example:
	$(PY) examples/refund_bot.py
	$(PY) examples/runaway_loop.py

check: lint types test egress overhead

# Before any release, on top of `check`: refuses a price table not re-read within
# 30 days (§12.5, D43). Not part of `check`, so an old table never blocks a commit,
# only a release.
release-check: check
	$(PY) -m tests.price_freshness

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache src/*.egg-info
