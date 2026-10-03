# Security

## Reporting a vulnerability

**Report it privately on GitHub:** this repository's **Security** tab, then
**Report a vulnerability**. Only the maintainer can read it. Please do not put
details in an issue or anywhere public. If that button is not there (GitHub
offers it only on public repositories), open an issue saying only that you have
a security report, and you will be invited to a private repository for the
details.

Expect an acknowledgement within 72 hours. This project is maintained by one
person, so a fix may take longer than an acknowledgement — you will be told
which is happening. While Paveo is at 0.x, a fix goes into a new release; older
releases are not patched.

## How releases reach you

Every release on PyPI is built and uploaded by
[`.github/workflows/release.yml`](https://github.com/paveo-dev/paveo/blob/main/.github/workflows/release.yml),
from a tag in this repository, through PyPI's trusted publishing: there is no
upload token for anyone to hold. The tools that build it are pinned by hash,
PyPI keeps a provenance attestation for each file, and the GitHub release for
the tag carries an SBOM with each file's SHA-256.

Reviewing Paveo for your organisation? `docs/SECURITY_REVIEW.md` answers the usual
questions, each with the test or file that proves it.

## What we claim, precisely

Overclaiming is itself a security problem, so:

- **Paveo opens no network sockets.** Enforced by `tests/test_no_egress.py`.
  Verify it yourself with a firewall rule or `tcpdump` — we would rather you did.
- **Prompts, completions and tool arguments never leave your process** and are
  never written to the audit log.
- **The audit log is tamper-evident, not tamper-proof.** Anyone with write access
  can rewrite the chain; editing, reordering or removing a record from the middle
  cannot be done invisibly. Truncating the *end* is caught by an anchor file
  beside the log — **unless that anchor is deleted as well**, which is a warning
  rather than a refusal by design. Stated here rather than discovered.
- **Paveo is a guard rail, not a sandbox.** Code that calls your provider SDK
  directly bypasses it. Tool policy applies only to calls routed through
  `check_tool`.
- **The coding-agent guard (Claude Code, Codex, Cursor) refuses on any failure
  of its own, but cannot refuse if it never runs.** Claude Code and Codex let a
  call through when a hook is missing or times out, and Codex runs no hook you
  have not trusted; `paveo guard <agent> --selftest` checks for a missing hook
  and Codex's trust record, and you should run it. It does not read the hook's
  timeout: leave it at the default, or well above the guard's 4-second deadline. Cursor's hooks are written with `failClosed`, which Cursor
  documents as refusing the call on a crash or timeout. Its
  patterns are a guard rail against common destructive commands, not a sandbox:
  a determined or disguised command gets past them. See THREAT_MODEL T10.
- **An agent in shadow mode is guarded by its budget and nothing else.** Its
  rule refusals are recorded as `would_deny` and allowed through, with a
  warning on every call. The mode is written in the policy and is part of the
  policy hash in every audit record.
- **Zero required runtime dependencies**, because our own supply chain is the
  gravest risk we pose to you. See `docs/THREAT_MODEL.md` §5. The one optional
  dependency is for paid plans only: `paveo[team]` adds `cryptography`, used
  solely to verify a licence key's signature, offline, and imported only when a
  key is read. The free plan never loads it, and a test proves that.
- **You can read every line.** The licence is the
  [Elastic License 2.0](https://github.com/paveo-dev/paveo/blob/main/LICENSE) —
  source-available, not open source. Audit it, run it, modify it, embed it in a
  commercial product; you may not resell Paveo itself as a hosted service.
  Being able to read the code is half of why a security team would accept it in
  their call path. The other half is that nothing leaves their machine, and
  neither substitutes for the other. See D22.

## Scope

In scope: budget bypass, policy bypass, payload leakage into logs or errors,
audit chain forgery, ReDoS in policy loading, anything that makes the library
fail open silently.

Out of scope: bypassing Paveo by not calling it, provider SDK issues, TLS,
and attacks requiring code execution already inside the host process — all
documented in `docs/THREAT_MODEL.md` §6.
