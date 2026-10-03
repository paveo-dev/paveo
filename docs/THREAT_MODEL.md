# Paveo — Threat Model

We are asking people to run our code in the path of every call their agent
makes. **The most important threat in this document is the one we ourselves
pose.** A threat model that only covers threats *to* the product, and not
threats *from* it, is marketing.

---

## 1. Assets

| Asset | Why it matters |
|---|---|
| Customer prompts, completions, tool arguments | Contains their customers' data. **Never leaves the process, never logged.** |
| Provider API keys | Held by the SDK, not by us. We must never read, log or transmit them. |
| The budget ceiling | The control being sold. If it can be exceeded, the product is a lie. |
| The audit log | The record of who did what. If it can be silently rewritten, it proves nothing. |
| The policy | The definition of allowed. Silent modification = silent privilege escalation. |

---

## 2. Trust boundaries

```
  ┌─ customer process ───────────────────────────────────┐
  │                                                      │
  │   agent code ──▶ PAVEO ──▶ provider SDK ─────────┼──▶ provider API
  │                     │                                │
  │                     └──▶ audit file (local disk)     │
  └──────────────────────────────────────────────────────┘
        ▲                                        ▲
        │                                        │
   boundary A                              boundary B
   (our code enters                 (the only egress — and it is
    their process)                   the SDK's, never ours)
```

**Boundary A is the one a reviewer will care about**, and the mitigation is that
we never cross boundary B.

---

## 3. What an attacker can do, by access level

| Attacker has | They can | Mitigated? |
|---|---|---|
| Ability to influence the agent's **prompt** (injection) | Attempt tool calls the agent shouldn't make | **Yes — this is the product.** Policy is evaluated on the call, not on the prompt. Injection cannot exceed policy. |
| **Code execution** inside the customer process | Bypass Paveo entirely by calling the SDK directly | **No.** Accepted, documented: we are a guard rail, not a sandbox. |
| **Write access** to the policy file | Grant themselves permissions, or set an agent to shadow mode | **Partially.** Policy hash in every audit record makes the change visible after the fact. Prevention is the customer's file permissions, not ours. |
| **Write access** to the audit file | Rewrite history | **Partly detectable, not preventable.** Editing, reordering or removing a record from the middle forces them to rewrite every record after it, and `verify_chain` names where. Truncating the tail is caught by the anchor file beside the log — **unless they delete that too**, which is undetectable by design (§10.13). Once an audit evidence export is in an auditor's hands, a rewrite of the records it covers is detectable too: it holds their hash off the machine (D66). |
| **Ability to publish to our package index account** | Ship malicious code into every customer's process | **The gravest threat. See §5.** |
| Network position between customer and provider | Read traffic | **Out of scope** — TLS is the SDK's job, not ours. |

---

## 4. Threats and mitigations

**T1 · Budget bypass via concurrency.**
Two calls race, both see budget available, both proceed, ceiling exceeded.
→ Reserve and reconcile are atomic under a lock. Property and concurrency tests
gate every release. *Spec §4.3.*

**T2 · Budget bypass via unbounded output.**
Caller omits `max_tokens`; actual cost far exceeds the estimate.
→ We reserve the **worst case**, never a likely case. No `max_tokens` and no
configured assumption ⇒ deny. *Spec §4.4.*

**T3 · Prompt injection drives an unintended tool call.**
→ Policy is evaluated at the call site against the tool name and its arguments.
A prompt cannot talk its way past a numeric constraint. **This is the strongest
security argument for the product and should lead the README.**

**Qualified, because the original wording overstated it:** this holds only for
tool invocations that actually pass through `check_tool`. If an agent framework
dispatches a tool without calling us, injection wins and we never see it.
Integration correctness is therefore load-bearing, and the docs must say so —
the claim is *"a checked call cannot be talked past"*, not *"your agent cannot
be injected"*.

**And only for an agent that enforces.** An agent in shadow mode (D48) has its
rule refusals recorded as `would_deny` and let through, so injection beats its
rules exactly as if Paveo were absent. Its budget still holds. Shadow mode is for
trying a policy, and the warning on every call is there so it is not left on.

**T4 · Reservation leak becomes a self-DoS.**
A crash or abandoned stream leaves budget reserved; the customer's own agent is
denied.
→ Release in `finally`, including generator close. Explicit test for abandoned
streams. Process death clears in-memory state (documented limitation).

**T5 · ReDoS through a policy regex.**
→ `fullmatch` so anchoring is structural, a length cap on the pattern and on the
value it is matched against, and rejection at load time of nested quantifiers,
repeated groups that hold an alternation (`(?:a|aa)+`), backreferences and every
group form but `(?:`. Policy loading is the only place regexes are compiled.
**Partially mitigated, not closed:** a chain of optional alternation groups,
`(?:a|aa)?(?:a|aa)?...`, and polynomial shapes such as `a*a*a*b` are not detected,
and the stdlib engine has no timeout. The coding-agent guard refuses any call it
cannot decide within 4 seconds, so there a slow pattern costs a refusal, never a
bypass; a library caller has no such deadline and waits. The residual risk needs
write access to the policy file, which is its own row in §3. *D15.*

**T6 · Code execution via the policy file.**
→ Policy is **JSON in the core** (stdlib parser, no object construction). The
optional YAML extra uses `yaml.safe_load` only — never `yaml.load`, never
pickle, no `!!python/` tags. No expression language, no lambdas, no plugin
loading. Unknown keys are a hard error.

**T7 · Data leakage through our audit log.**
→ Payloads are never written, and this is **structural rather than a
convention**: the log refuses any field §6 does not define, refuses the fields it
owns itself, and checks the `action` object the same way, so an edit that adds
`"arguments"` fails on the first call rather than shipping (D17). A sentinel test
asserts no argument value reaches the file or a denial. File mode `0600`,
enforced with `fchmod` after opening because umask can strip bits from the mode
passed to `open`. **Not a content filter:** a caller who puts a prompt in
`reason` still gets a prompt in the log — nothing here inspects values, because
inspecting them would mean reading customer data to decide whether to write it.

**T8 · Data leakage through our error messages or exceptions.**
→ Errors carry identifiers, costs and rule names. **Never request content.**
Same sentinel test covers exception strings.
`PolicyDenied.for_model` is text meant to go *back to the model*, which may be
under injection. It carries the rule and the reason and no remedy, because the
remedy says how to change the policy: handed to a model, that is advice on
switching off its own guard (D47 #7, D48). Covered by the same sentinel test.

**T9 · Silent failure of the guard.**
A guard that stops guarding without telling anyone is worse than none.
→ Fail closed by default. Every fail-open use is logged, every call. Audit
unwritable ⇒ deny, because an unlogged decision did not happen. Shadow mode, the
one deliberate way to stop rules enforcing, is chosen per agent in the policy,
warns on every call, and records every refusal it lets through. **It never
reaches the budget** (D48).

**T10 · The Claude Code seatbelt switched off** (D49).
The guard is a hook that Claude Code runs, so Claude Code's contract decides what
a failure means: **exit 2 blocks, and everything else lets the call through**, a
crash, a timeout, a missing binary included.
→ Every failure *inside* the guard exits 2, and a test forces each one: no
policy, input that is not a call, an exception, a stop file it cannot check. What
it cannot close from inside is its own absence. `paveo guard --selftest` runs
the configured hook and fails if it is missing or does not refuse, and the README
tells the user to run it.
→ **The agent turning it off.** The agent can write files and run commands, so
it can try to edit the hook settings, the policy, the stop file or the log. The
Claude Code starter policy refuses a Bash command that names `.pave`, a literal
`.claude` (except `.claude.c…` and `.claude.a…`, so that `claude.com` and
`claude.ai` addresses pass), a `settings.json` or `.mcp.json`, and refuses
`paveo resume`. Write, Edit and NotebookEdit are refused under `.paveo`, anywhere
under `.claude` except `.claude/plans` (where plan mode saves plans), on
`.claude.json` and `.mcp.json`, and on any path with a `..` step. Everything else
under `.claude` configures Claude Code or runs code it loads (settings, hooks,
plugins, agents and skills with hooks in their frontmatter, commands), and an
MCP server added to `.mcp.json` runs tools the hook never sees. Paths are matched
as macOS and Linux write them; the guard does not run on Windows. A narrower
Bash rule was tried and dropped because it opened paths the blanket one closed
(D54). **Best effort only:** a command assembled at run time, a script, a
symlink, a variable, or a glob or quote inside the name (`~/.clau*`,
`.cl''aude`, `${X%.ai}` after `X=~/.claude.ai`) gets past a pattern. A guard
rail, not a sandbox.
→ **The agent wiping its own memory** (D59). `requires`, `rate` and `repeat`
remember through files in `.paveo/memory/`. Deleting one resets that session's
counts, and a new session id starts from none. The starters' `.pave` rules cover
the folder like the rest of `.paveo`, with the same best-effort limits; a damaged
file refuses every call from its session rather than forget it. A rate limit is
a brake on a loop, not a quota an agent that means to evade it cannot reset.
`git clean -fdx`, which never names `.paveo`, wipes it too, as it does the stop
file: resetting `rate` and `repeat`, and only tightening `requires`
(/security-review, D59).
→ **Claude Code removing the hook itself.** A setup command has replaced a user's
settings file and dropped every hook in it (claude-code#96987). Nothing inside
the guard can notice its own absence, so `paveo guard --selftest` after any
settings change is the check, and the README says to run it.
→ **Disguised destructive commands.** `not_matches` finds patterns; it does not
parse the shell. `rm -r -f`, `find -delete`, or anything piped into `sh` gets
past it. Stated in the README and in `src/paveo/starters/README.md`.
→ **A guard made slow.** A timed-out hook is a pass, so anything that can keep
the guard waiting switches it off: an agent holding the audit log's `flock`
(`flock .pave?/audit.jsonl sleep 999`), or a command crafted so a pattern
searches a megabyte in quadratic time. **The guard refuses at its own 4-second
deadline** by exiting 2 from a signal handler. It uses `os._exit`, not an
exception, because an exception could be caught, or arrive after the answer was
chosen. Both are tested with real processes. The starter patterns are also
bounded (`[^\n]{0,100}`, not `.*`). A hook `timeout` set below 4 seconds would
reopen this, and the README says so. *Found by /code-review, D49.*
→ **`paveo init` run inside a cloned repository** (D50). The repository controls
every file `init` reads and writes. It writes only real files inside the project
and refuses if any of them is a symbolic link. Its temporary file gets a fresh
name from `mkstemp`: a fixed name was a place the repository could plant a link
to the user's global Claude Code settings, found by /security-review. It asks a
policy already in the project what it refuses, rather than trusting it, because
the repository may have shipped a permissive one. And it never reports the
seatbelt on without checking the hook's matcher and every settings file.
→ **`--selftest` running what a repository says.** Settings files ship with
repositories, and a cloned one could name any command. The self-test runs only
a plain `paveo guard claude-code` command, with no shell syntax, whose program
is named `paveo`. A program inside the project (a project's own virtualenv is
the ordinary case) runs only after the person confirms, at a terminal, that they
installed it; with no terminal it is never run. A test tries seven forms of the
attack.
→ **Replay and learn read what a model wrote** (D53). Session files hold every
prompt, command and result, so `replay` prints only counts, sums and names the
policy or price table already holds. A tool name in them was chosen by the
model, and one under injection can "call" a tool that does not exist, with
anything in its name: Claude Code refuses to run it, but the name is on disk.
`learn` writes names into a policy people commit, so it takes them only from
calls whose result came back without an error, and only names shaped like
identifiers. That proves the tool is real, not every argument name: an MCP server
may ignore one it does not know, so `learn` prints each name it adds. It writes
beside the policy, never over it, and not through a link.
→ **Codex and Cursor** (D57). The same guard behind each agent's own hook.
**Codex runs no hook the person has not trusted**, in a project they have not
trusted, and says nothing when it skips one: `--selftest codex` fails when
Codex's config holds no trust record for the hook or the project, or turns hooks
off. It cannot tell a hook changed since it was trusted, which Codex also skips,
and says so. **Text typed into a shell Codex already started never reaches
`PreToolUse`** (`write_stdin`, by Codex's design): a model that opens `bash` and
types into it is not seen. Codex's `apply_patch` is judged by the files its
headers name, read in one pass, so a patch cannot be made slow to judge and its
content is not mistaken for a path. **Cursor lets a crashed or timed-out hook
through unless `failClosed` is set**: `init` sets it and `--selftest` fails a
hook without it. Cursor's file tools send fields it does not document, so the
starter names a guess, and a field it does not name is refused. Each agent's
starter refuses its own hook folder (`.codex`, `.cursor`) the way Claude Code's
refuses `.claude`, with the same limits.
→ **The guard's own code, and the agent a hook names** (D57, /security-review).
paveo installed in a virtualenv inside the project is a file the agent can
write: every starter refuses a write under `site-packages/paveo` and to any `.pth`
file, and the recommended install under `~/.paveo` is covered by `.pave`. A
shell command reaching a project-local install is best effort, as above. And
`init` asks a kept policy what it refuses **as every agent id the project's hooks
name with `--agent`**, not only the default, or a repository could pair a strict
agent with a lax one its hook really uses.

**T11 · A rewritten log attacks whoever reads its export** (D66).
Someone who can write the log can rechain it, so the export verifying proves
nothing about who wrote a value, and an auditor opens the report in a browser and
the records in a spreadsheet.
→ Every value from the log is escaped for HTML, and a CSV cell that a spreadsheet
could run as a formula is prefixed with `'`: one starting with a formula
character, or holding `;` (a cell break where it is the list separator), `=`,
`+`, `@`, a tab or a return anywhere. The report carries no script. The
export reads the log, never a payload (there is none), and opens no socket.

---

## 5. The threat we pose: supply chain

If our package index account is compromised, an attacker ships code into the
inner loop of every customer's agent. This is the highest-severity threat in
this document and the one we control most directly.

**Controls, from first release:**

- **Zero required runtime dependencies.** Nothing to compromise transitively.
  This is the single largest reduction in attack surface available to us, and it
  is why it is a locked decision rather than a preference.
  *Revisited 2026-09-27 (D62), when the trigger below fired:* paid plans add one
  optional dependency, `cryptography`, installed only with `paveo[team]` and
  imported only to verify a licence key's signature, offline. A compromised
  `cryptography` reaches only those who install the extra, and it is the most
  scrutinised Python cryptography package there is; writing Ed25519 ourselves
  would have been the riskier choice. The free core still installs nothing.
- **No network egress**, enforced by a test — a compromised build that adds one
  is detectable by the customer with `strace`, `tcpdump`, or a firewall rule.
  **We should tell them to check.**
- Signed tags; published SBOM; build dependencies pinned and hash-verified.
- 2FA and a hardware key on the package index and source host accounts.
- Trusted publishing from CI only. **No human ever publishes from a laptop.**
- `SECURITY.md` with a disclosure contact before the first public release.

*How each is met, 2026-09-27 (D67):* `.github/workflows/release.yml` publishes
from a `v*` tag in the public repository only, through PyPI trusted publishing
in a `pypi` environment that waits for the maintainer's approval, so no upload
token exists; it refuses a tag that does not match the version, runs the whole
gate and the price-freshness check first, builds with tools pinned by hash
(`.github/requirements-build.txt`, no build isolation), and attaches a CycloneDX
SBOM to the GitHub release; PyPI keeps a provenance attestation per file.
`SECURITY.md` points to GitHub's private vulnerability reporting. **Signed tags
and the hardware key are the maintainer's to set up before the first tag**; the
workflow does not check a tag's signature.

---

## 6. Explicitly out of scope for v1

Say these out loud in the README. A reviewer who finds an unstated limit assumes
there are others.

1. **We are not a sandbox.** Code that can call the provider SDK directly
   bypasses us. We constrain an agent that is *using* us, not an attacker who
   has already executed code.
2. **We do not inspect model output.** No content filtering, no jailbreak
   detection, no evaluation.
3. **We do not manage secrets.** API keys stay with the SDK.
4. **We do not defend the audit log against an attacker with write access** —
   only make tampering evident.
5. **We do not provide cross-process or cross-host budget enforcement in v1.**
6. **We do not protect against a malicious provider SDK**, which sits closer to
   the network than we do.

---

## 7. Review triggers

Revisit this document when any of the following changes — not on a schedule:

- A network call is added anywhere (this invalidates §5's strongest control)
- A runtime dependency is added
- The audit log gains payload capture, even opt-in
- Policy gains any form of expression evaluation
- Budget state moves out of process
