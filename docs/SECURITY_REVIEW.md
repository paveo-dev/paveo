# Paveo: answers for a security review

For the person deciding whether Paveo may sit in the path of their agents' calls.
Each answer names where to check it: a test, a file, or a command you can run. If
an answer and the code ever disagree, the code is right and this page is a bug;
please report it (`SECURITY.md`).

Paveo is a Python library and a command, `paveo`. It checks each call an AI agent
makes (a tool call, a shell command, a model call) against a policy file, before
the call runs, and refuses what the policy does not allow. It runs inside your own
process, or as a hook your coding agent starts. Nothing of ours runs anywhere.

## 1. Data

**What data does Paveo send to the vendor?**
None. Paveo opens no network sockets: not to us, not to anyone. There is no
telemetry, no update check, and no licence server. Check: `tests/test_no_egress.py`
fails the build if any operation constructs a socket, and runs outside pytest on
purpose so that no test plugin can mask it. You can confirm it on your own machine
with a firewall rule or `tcpdump` while your agent runs.

**What does Paveo store, and where?**
Only on your disk, in files you choose:
- The audit log (`paveo-audit.jsonl` by default, `.paveo/audit.jsonl` for the
  coding-agent guard): one record per decision, and one more settling each allowed
  model call at its real cost, with the agent, the principal identifier your
  code supplies, the tool name or model, the decision, the rule, the policy's hash,
  and costs and token counts. **Never prompts, completions or tool arguments.**
  Check: `src/paveo/audit.py` refuses any record field it does not recognise, so a
  code change that tried to add one fails in tests (`tests/test_audit.py`).
- The guard's memory (`.paveo/memory/`), used only by `requires`, `rate` and
  `repeat` rules: salted SHA-256 digests of the compared values and timestamps,
  never the values. A session's file untouched for a week is deleted when the next
  new session starts. Check: `src/paveo/_memory.py`.
- The policy file you write, and a licence key if you have one.

**Does a tool name ever carry model-chosen text into the log?**
No. A tool name the policy does not declare is recorded as `<undeclared>`, because a
model under prompt injection chooses that name. Check: `tests/test_session.py` and
`src/paveo/enforce.py`.

**Is anything encrypted?**
Paveo writes no secrets and no customer content, so it adds no encryption of its
own; the audit log and memory are protected by file permissions (below) and by
whatever disk encryption you already use.

## 2. Access and files

- The audit log and its anchor are created `0600` (owner only). The library
  refuses a log in a directory anyone may write to.
- The guard's memory folder is created `0700` and its files `0600`; it refuses a
  folder or file that is group- or world-writable, a link, or owned by another user.
- `paveo init` refuses to write through a symbolic link, and keeps the files that
  name this machine's paths out of git.

Check: `src/paveo/audit.py`, `src/paveo/_memory.py`, `src/paveo/_setup.py`.

## 3. Failure

**What happens when Paveo cannot judge a call?**
It refuses it. A policy that will not load, a model call that cannot be priced, a
log that cannot be written, a guard that takes longer than 4 seconds, or an
unexpected error inside the guard: each is a refusal, never a pass. Failing open
exists only as a setting you switch on, it warns on every call while on, and it
never overrides a rule's refusal. Two exceptions are stated in section 6: a guard
that never runs, and an agent you put in shadow mode. Check: `docs/SPEC_V1.md` §7 and `tests/test_cli.py`.

**What happens when a licence key expires?**
The free plan applies, with a warning: every rule keeps enforcing, the first two
agents keep working, and an agent past two is refused and told why. Nothing is
ever let through because a key lapsed. A signed key that has been altered is
refused outright rather than guessed at. A 30-day trial, which only 0.1.0 and
0.1.1 could start, is a start date written on your machine, not a signed key, so
whoever can edit that file can restart it; that changes which plan applies, never
what a rule refuses. Check:
`tests/test_licence.py` and `src/paveo/_licence.py`.

**Can concurrent calls overspend a budget?**
Not through one store. A model call's worst case is reserved under a lock before
it is sent and settled at its real cost after; one lock serves threads and asyncio.
Check: the property test in `tests/test_budget.py` and the thread and coroutine
tests in `tests/test_stores.py`. Budgets are per process: two processes do not share
one ceiling yet.

## 4. The code and its supply chain

**What does installing Paveo pull in?**
Nothing: zero required dependencies. Paid plans add one optional package,
`cryptography`, used only to verify a licence key's signature, offline, and
imported only when a key is read. Check: `pyproject.toml`.

**How do releases reach PyPI?**
Only through `.github/workflows/release.yml`, from a tag in the public repository,
using PyPI trusted publishing, after the maintainer approves the release in the
`pypi` environment (a repository setting, so not visible in the file). No upload
token exists for anyone to hold. The build tools
are pinned by hash (`.github/requirements-build.txt`) and run on a separate machine
from the tests; PyPI keeps a provenance attestation for each file, and the GitHub
release carries an SBOM written by `.github/sbom.py`.

**Can we read the code?**
All of it. The licence is the Elastic License 2.0 (`LICENSE`): source-available, not
open source. You may read, run, modify and embed it; you may not resell Paveo itself
as a hosted service, or remove its licence-key checks.

**How is the code checked?**
Lint, `mypy --strict`, the full test suite (including property and concurrency
tests) and the standalone no-egress run: `make check` runs them all in one command,
and `.github/workflows/ci.yml` runs the same on every push to the public repository.
The shipped source distribution includes the tests, so you can run them on what you
installed.

## 5. Vulnerabilities

Report privately through GitHub's private vulnerability reporting, as described in
`SECURITY.md`; acknowledgement within 72 hours. While Paveo is at 0.x, fixes go into
a new release, not into older ones.

## 6. What Paveo does not protect against

Stated so your review does not have to discover it:

- **It is a guard rail, not a sandbox.** Code that calls a provider or a tool
  without asking Paveo is never seen. Code already running inside your process can
  bypass it.
- **A coding-agent guard that never runs cannot refuse.** Claude Code and Codex let
  a call through when the hook's binary is missing or the hook times out, and Codex
  runs no hook you have not trusted. The guard refuses at its own 4-second deadline,
  so leave the hook's timeout at the agent's default, or well above 4 seconds.
  `paveo guard <agent> --selftest` checks for a missing binary and Codex's trust
  record; it does not read the timeout. Cursor's hooks are written to refuse on a
  crash or timeout.
- **An agent in shadow mode is guarded by its budget and nothing else.** Its rule
  refusals are recorded as `would_deny` and let through, with a warning on every
  call. The mode is written in the policy, so it is part of the policy hash in every
  audit record.
- **The coding-agent guard matches patterns.** It does not understand the shell: a
  command split across flags, hidden in a variable or run from a script can get
  past it. An agent that finds a way to edit its own hook settings can switch it
  off; the starter policies refuse the obvious ways, not all of them.
- **The audit log is tamper-evident, not tamper-proof.** Someone with write access
  can rewrite it, but not invisibly; deleting its anchor file as well makes
  removal from the end undetectable.
- **It does not inspect model output**, filter content or detect prompt injection.
  It limits what a manipulated agent can do, not what it is told.

The full threat model is `docs/THREAT_MODEL.md`.
