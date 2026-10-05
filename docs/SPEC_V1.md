# Paveo v1 — Technical Specification

**Status:** draft for implementation · **Read this before writing code.**

v1 is the smallest thing that is genuinely trustworthy. Not the smallest thing
that demos. Every requirement below exists because a security reviewer will ask
about it.

---

## 1. Scope

**In:** agent identity · budget ceilings with correct accounting · policy on
models and tools · pre-call enforcement · tamper-evident local audit log ·
Python 3.11+.

**Out:** hosted anything · network calls · dashboards · prompt-injection
detection · output evaluation · distributed budget state · framework
integrations.

**The one sentence v1 must earn:** *"This library decides whether your agent's
next call is allowed, in your own process, and it cannot overspend your ceiling."*

That is **admission control** — the Kubernetes term for a gate that inspects a
request and admits or rejects it before it takes effect. Prefer it to invented
vocabulary throughout: it is precise, it is already understood, and it draws the
line against observability without an argument. D12.

---

## 2. Architecture

```
your process
┌──────────────────────────────────────────────┐
│  agent code                                  │
│      │ explicit call                         │
│      ▼                                       │
│  paveo.Session                           │
│      ├─ identity      who is calling         │
│      ├─ budget        reserve → reconcile    │
│      ├─ policy        model + tool rules     │
│      └─ audit         append-only, hashed    │
│      │                                       │
│      ├── allow ──▶ provider SDK ──▶ network  │
│      └── deny  ──▶ raise, nothing sent       │
└──────────────────────────────────────────────┘
```

**Paveo itself opens no sockets.** The provider SDK makes the network call,
after we have allowed it. This distinction is the security story and must never
blur.

### 2.1 Interception model
Wrapping is **explicit**. The developer writes the call that installs it:

```python
client = session.wrap_anthropic(anthropic.Anthropic())
```

`wrap_*` returns a proxy that intercepts **an explicit allowlist of methods**
that cost money — for Anthropic, `messages.create` and `messages.stream`. It does
not subclass, does not patch the module, and does not rely on SDK internals
beyond the documented request/response shape.

**Do not implement this as a permissive `__getattr__` passthrough.** Provider
clients are deep attribute trees (`client.messages.create`, `.messages.stream`,
`.beta.*`), and a proxy that forwards anything it does not recognise will
silently pass an unchecked, billable call straight through. **That is a security
failure, not a cosmetic one.**

Rule: an attribute path that is not on the allowlist and could plausibly make a
request **raises** rather than forwarding. Adding a provider means extending the
allowlist deliberately — which is the point.

**Rejected:** import-time monkey-patching, `sys.settrace`, sitecustomize hooks,
and anything a reviewer would describe as "magic". Unauditable, fragile across
SDK versions, and an immediate no in a security review.

---

## 3. Public API

The entire public surface. Anything not listed here is private and may change.

```python
from paveo import Paveo, PolicyDenied, BudgetExceeded

pf = Paveo.from_file("paveo.json")       # or Paveo.from_policy(obj)
# Both take licence="paveo1...." for a paid plan; without one the Developer
# plan covers two agents per policy (D61, D62, D74). Checked offline.

with pf.session(agent_id="refund-bot", principal="user_123") as s:

    client = s.wrap_anthropic(anthropic.Anthropic())
    resp = client.messages.create(
        model="claude-sonnet-5", max_tokens=1024, messages=[...]
    )                                # checked before it leaves; raises on deny

    s.check_tool("refund", {"amount_usd": 250})   # raises PolicyDenied
    s.record_external_cost(Decimal("0.02"), note="embedding")

    s.remaining()                    # -> Decimal, budget left this period
```

Two module-level functions complete the surface:

```python
from paveo import verify_chain, ChainStatus

verify_chain("audit.jsonl")      # -> ChainStatus(ok, records, broken_at, detail)
```

**Built in S4 (D23 #1, D40), the door for every framework the proxy cannot
reach:** a model call checked by its request, and a handle that settles it.

```python
with s.check_llm(request, shape="anthropic") as call:   # raises on any refusal
    response = client.messages.create(**request)
    call.record(response.usage.model_dump())            # or call.release()

s.remaining()
```

`shape` names the request's format; a model whose price the policy declares is
read whatever its shape (§4.8.4, D39). A call that leaves the block, or the
session, without `record` or `release` is charged its whole worst case.

**Built in B1 (D48), the refusal a tool loop hands back to the model:**

```python
try:
    s.check_tool(name, arguments)
except PolicyDenied as refused:
    result = refused.for_model       # str: the rule, the reason, what to do
```

`for_model` carries no payload and **no remedy**: the remedy is addressed to the
operator and says how to change the policy, which is not advice to give a model.

**Built in B2 (D49), the `paveo` command.** Standard library only, and it
opens no socket:

```
paveo init {claude-code,codex,cursor}                  # set a project up (D50, D57)
paveo guard {claude-code,codex,cursor} [--dir .paveo] [--agent NAME]   # the hook
paveo guard [AGENT] --selftest [--settings FILE ...]   # AGENT defaults to claude-code
paveo stop   [--dir .paveo]
paveo resume [--dir .paveo]
paveo mcp --agent NAME [--dir .paveo] -- SERVER [ARGS ...]   # the MCP guard (B6, D80)
```

**`paveo mcp`** starts the MCP server named after `--`, without a shell, and
relays stdio between it and the client that started `paveo mcp`. Every
`tools/call` is judged under `--agent` before the server sees it: allowed, it is
forwarded as Paveo re-serialized it; refused, the server sees nothing and the
client gets a result with `isError: true` and `PolicyDenied.for_model` as its
text. Everything else passes unjudged. **Every message is forwarded as Paveo
re-serialized it**, one line of ASCII with every control character escaped, so
no reader's line framing can split it (D80). A line that cannot be read safely
(not JSON or UTF-8, over 64 MiB, a key twice) is dropped with a note on stderr;
a batch holding a tool call is answered with JSON-RPC errors. The run keeps one
session's memory in the process, reads `stop` for every call, and re-applies the
plan when the date changes. If the policy does not load, the server is not
started and it exits 1. Its exit code is the server's. On `SIGTERM` it stops
the server it started at once (`SIGTERM`, then `SIGKILL` a few seconds later)
rather than leave it running.

`--dir` holds `policy.json`, `audit.jsonl` and, while stopped, `stop`. The guard
reads one call on stdin and **exits 0 and prints nothing** when the policy has no
objection, so the agent's own permission flow still applies, or **exits 2**
with `PolicyDenied.for_model` on stderr (for Cursor, also as a
`{"permission": "deny"}` answer on stdout). `--agent` defaults to the agent's
name, which is the agent id in its starter policy. What differs between Claude
Code, Codex and Cursor is data in `_harnesses.py` (D57). **Every failure inside it also exits 2**:
Claude Code lets a call through on any other exit. `--selftest` runs each
configured hook as Claude Code would, and fails a hook that is missing or does
not refuse. It runs only a plain `paveo guard claude-code` command, and a
program inside the project only after the person confirms they installed it. While `stop` exists, every call is refused as `stopped`,
never shadowed. The stop reaches the command only; the library's sessions do not
read it (trigger in D49).

`verify_chain` **reports rather than raises** — see D16. The "every denial
raises" rule above governs guard decisions, where a caller must not be able to
ignore a refusal. Verifying a log a year later is not one of those: the operator
wants the sequence number where the chain broke, not a traceback.

The same surface exists for `asyncio`, because most real agent code is async
(§4.9). Same names, same semantics, `await` on anything that touches the budget:

```python
async with pf.async_session(agent_id="refund-bot", principal="user_123") as s:

    client = s.wrap_anthropic(anthropic.AsyncAnthropic())
    resp = await client.messages.create(
        model="claude-sonnet-5", max_tokens=1024, messages=[...]
    )

    await s.check_tool("refund", {"amount_usd": 250})
    await s.record_external_cost(Decimal("0.02"), note="embedding")
    await s.remaining()
```

`wrap_anthropic` returns a proxy matching the client it was handed. A sync client
passed to an async session — or the reverse — is a wiring error and **raises**
(`ConfigError`), rather than working by accident until it is under load.

**Every denial raises.** No `(ok, reason)` tuples, no boolean returns, no
"result objects" a caller can forget to check. **A guard you can accidentally
ignore is not a guard.** This is a security decision, not a style one.

`session()` is a context manager and **must** release outstanding reservations
in a `finally` block, including on exception, `return`, and generator close.

---

## 4. Budget accounting — the hard part

The decision must be made **before** the call, but the cost is not known until
**after** it. Resolved with two-phase accounting.

### 4.1 Reserve (before the call)

Cost is a function of three things, not one:

```
cost = f( model , price-affecting request parameters , per-class token counts )
```

At reserve time we know the first two. **We cannot know the third** — how the
tokens divide between classes (fresh input, cache read, cache write) is only
reported in the response. So we reserve at the most expensive class that could
apply. See §4.8 for the table this resolves against.

```
input_upper_bound = UPPER BOUND on tokens sent (see 4.5 — not "exact")
max_out_tokens    = request.max_tokens
                    ?? policy.assumed_max_output_tokens
                    ?? DENY (PricingUnknown)

rates             = prices.resolve(model, modifiers(request))
                    ?? DENY (PricingUnknown)          # unknown model, or an
                                                      # unknown value for a
                                                      # price-affecting param

worst_case        = input_upper_bound * max(rates.input_classes)
                  + max_out_tokens    * max(rates.output_classes)

atomically:
    if spent + reserved + worst_case > limit:  ──▶ DENY (BudgetExceeded)
    else: reserved += worst_case, return reservation_id
```

We reserve the **worst case**, never an estimate of the likely case. A ceiling
that can be exceeded by an unlucky long completion is not a ceiling.

**Why `max()` over the classes and not the likely class.** A request that enables
caching may come back as almost all cache reads (0.1× on most models, 0.05× on
Opus 5.5, 0.025× on Fable 5.1) or almost all cache writes (2.0×) — a spread of
20× to 80× we cannot predict. Taking the maximum can never under-reserve. It over-reserves input by
up to 2× the base rate, which on a cache-read-heavy request is **up to 20× the
actual cost of that input, 40× on Opus 5.5 and 80× on Fable 5.1** (0.025×
charged as 2.0×). Output usually dominates the bill, so the practical cost is
usually small; see §10.9 for the case where it is not.
*Corrected 2026-09-19: the original formula had a single `price.input`, which
cannot express a response carrying three separately-priced input classes. D9.*

### 4.2 Reconcile (after the call, always)

```
on success:  actual = Σ  over every class reported in usage:
                          usage[class] * rates[class]
             spent += actual ; reserved -= worst_case

             a class present in usage but ABSENT from the table:
                 charge it at the row's highest rate, either basis
                                             (best effort, may be low)
                 mark the price table STALE                      — see §4.8.3
                 record price_table_stale in the audit line

on error:    reserved -= worst_case        (nothing was consumed, or unknowable)
             *Corrected 2026-09-23 (D40): "unknowable" is charged, not
             released. A call that errors may still have been billed, so
             unless the caller releases it (the request never left) it is
             charged its worst case.*
on stream:   reservation held for the whole stream; released in finally
```

`actual` is **never** recomputed from the reserve-time estimate. The response's
own per-class counts are the only input to it.

Reconciliation runs in a `finally`. **A leaked reservation is a denial-of-service
against the customer's own agent**, which is a bug of the same severity as an
overspend.

### 4.3 Requirements

- **`Decimal` for all money. Never float.** Enforced by type and by test.
- **Reserve and reconcile are atomic** with respect to concurrent sessions
  sharing a budget. In-process: a lock — and *which* lock is not a detail: it
  must be correct in threads and in an event loop at once, and only one kind is.
  See §4.9. Interface allows an external store later.
- **Retries reserve independently.** Each attempt is a separate reservation and
  a separate audit record. The SDK's internal retries are invisible to us and
  are a **known gap** — documented in §10, not hidden.
- **Process death clears in-memory reservations.** Acceptable for v1, documented.

### 4.4 The `max_tokens` tension — state it plainly

If a caller does not set `max_tokens`, the worst case is the model's full output
window, and we will deny calls that would have been affordable. Options, in order
of preference:

1. Caller sets `max_tokens` (recommended, documented loudly)
2. Policy sets `assumed_max_output_tokens` — **and every call using it emits a
   warning in the audit record**, because the ceiling is now approximate
3. Deny with a message that says exactly how to fix it

**We never silently assume a small output to avoid a false denial.** A quiet
under-estimate turns the ceiling into a suggestion.

### 4.5 Counting input tokens without breaking our own rules

An **exact** input count needs either a tokenizer library (a required
dependency — forbidden by D6) or the provider's count-tokens endpoint (a network
call — forbidden by D2). **We can have exact counts, zero dependencies, or zero
egress. Not all three.**

Resolved by noticing that **over-estimating is safe.** A ceiling enforced with an
inflated input count denies too eagerly; it can never overspend. So:

| mode | how | accuracy | deps |
|---|---|---|---|
| `conservative` *(default)* | `len(request.encode("utf-8"))` | guaranteed upper bound — no BPE token is under one byte | none |
| `exact` | optional tokenizer extra | accurate | `paveo[tokenizer]` |

The conservative bound typically over-estimates input by **3–4×**. On a
completion-heavy workload that barely matters, because output dominates cost. On
a prompt-heavy one it will deny too early — **so the mode and its cost are
printed at startup and recorded in every audit line.** Users must not discover
this from a mystery denial.

Never silently switch to an under-estimate to avoid a false denial. *Spec §4.4.*

**Corrected 2026-09-23 (D38): the byte count bounds only the tokens that are in
the bytes.** Three kinds of billed input are not, and each is on the provider's
pricing page: the system prompt the API adds whenever `tools` is present (up to
804 tokens), the definitions of Anthropic-defined tools (325 for bash, about
4,500 for the computer toolset and 6,600 for the browser toolset), and content
the request only *refers to* — an image by URL, a document by `file_id` — whose
tokens are the content and whose bytes are the reference. **S4 must close this
before `check_llm` claims a bound**; until it does, "guaranteed upper bound" in
the table above is false for any request carrying tools or references.

### 4.6 Rounding — money math needs a direction

All monetary values are `Decimal`, quantised to **8 decimal places**.

- **Reserving: always round UP** (`ROUND_CEILING`)
- **Releasing: always round DOWN** (`ROUND_FLOOR`)
- **Recording actual spend: round UP**

Every rounding decision moves in the direction that protects the ceiling. The
cumulative drift is fractions of a cent and always in the customer's favour.
*Unspecified in the original draft — a silent source of overspend.*

### 4.7 Period boundaries

`period: day` resets at **00:00 UTC**, not local time and not on process start.
`hour` resets on the hour, UTC. `session` is the lifetime of the context manager.

The reset instant is computed from the wall clock, so a process running across a
boundary sees the budget refill. **In v1 the in-memory store means a restart also
clears spend** — a process restarting every hour effectively has no daily
ceiling. This is a real limitation of the in-memory store, listed in §10.

### 4.8 Pricing

A versioned table in `prices.py`, overridable in policy. It carries an `as_of`
date, and the audit record carries the table version used for the decision.

**The table is not keyed on the model alone.** That was the original design and
it is wrong in two ways that both fail in the unsafe direction. D9 records why.

#### 4.8.1 Shape

```
model → {
    base_input   : Decimal          # per token
    base_output  : Decimal          # per token
    classes      : { name → (basis: input|output, multiplier: Decimal) }
}
```

Multipliers, not absolute rates, **because that is how the provider defines
them** — Anthropic prices cache reads at 0.1× the base input rate (0.05× on
Opus 5.5 and 0.025× on Fable 5.1, so the multiplier belongs to the model, not to
the class), 5-minute cache writes at 1.25×, 1-hour cache writes at 2.0×. One base rate to maintain,
derived rates that cannot drift out of step with it. Every multiplier resolves
to an absolute `Decimal` at table load; no arithmetic anywhere else touches a
multiplier.

For Anthropic the classes are `input` (1.0×), `cache_read` (0.1×, or the
model's own), `cache_write_5m` (1.25×), `cache_write_1h` (2.0×), and `output`
(1.0× of base_output). Verified 2026-09-23 against the provider's published
pricing, where the per-model cache-read rates were found (D38); re-verify before
any release (Rule 4).

#### 4.8.2 Modifiers — the same model, two prices

A **price-affecting request parameter** selects a different base pair. Claude
Opus 5 is $5/$25 per MTok normally and **$10/$50 with `speed: "fast"`** — the
same model string, a parameter on the request. A table keyed on the model alone
under-prices that call by half.

**There are two such parameters as of 2026-09-23, not one** (D38).
`inference_geo: "us"` multiplies every class by 1.1 on Claude 4.6 and later, and
stacks with fast mode. It is the §10.10 gap arriving in practice: this spec named
only `speed`, and a table built to it would have under-priced every US-only call
by a tenth.

**An unset `inference_geo` is priced as `us`, not as the API's documented
`global` default** (D38, found by `/security-review`). A workspace's
`default_inference_geo` decides it and we cannot see the workspace; orgs that had
opted out of global routing were migrated to a US default with no code change.
Reserving at the dearer geo is Rule 6. Setting the parameter makes it exact. The
row is its own, `…|inference_geo=unset`, so the audit record never claims a geo
the request did not ask for.

```
prices.resolve(model, modifiers) → rates
```

- The table declares, per model, **which parameters are price-affecting** and the
  base pair for each permitted value.
- A request that sets a declared price-affecting parameter to a value the table
  does not carry ⇒ **DENY** (`PricingUnknown`). We do not guess.
- Parameters not declared price-affecting are ignored for pricing. **This is a
  real gap, not a closed door** — a provider can introduce a new price-affecting
  parameter we do not know to look at, and we would under-price silently.
  Published as §10.10.

**Not required in v1: batch pricing.** The Batch API runs at roughly 50% of
standard cost, but `messages.batches.create` is not on the §2.1 interception
allowlist, so a batch call raises rather than being priced. *Trigger: the first
time batch is added to that allowlist — and it is a modifier like any other, not
a special case.*

#### 4.8.3 Unknown classes — fail closed, loudly

If a response reports a token class the table does not know, the table is stale
and **we can no longer bound a call**. Locked decision #4 applies:

1. Reconcile the unknown class at the row's highest rate of **either** basis,
   because which basis it is priced off is exactly what we do not know. Best
   effort — it may be an under-count, and the audit line says so. *Corrected
   2026-09-23 (D38): this said "for its basis".*
2. Mark the table **stale** for this `Paveo` (each owns its table, D40).
3. **Every subsequent reserve denies** with `PricingUnknown`, naming the
   unrecognised class and the table version. *In v1 `fail_open` does not
   override this (D39); trigger for letting it: the first user who asks.*

This is a deliberate self-inflicted outage with a loud, actionable reason, and
it is the honest end of "cannot overspend". The alternative — carry on with a
price table known to be incomplete — is silently lying about a ceiling, which is
the one thing this library exists not to do. Recovery is a table update.
Operational consequence published as §10.12.

#### 4.8.4 Unknown model

Denied (`PricingUnknown`) unless the policy declares its price. **Built in S4
(D39)**, as the door every provider without an adapter comes in by:

```json
"prices": {
  "mistral-large-3": {"input_per_mtok": "2.00", "output_per_mtok": "6.00",
                      "cached_input_per_mtok": "0.20"}
}
```

A declared price may not name a model a built-in table carries (`ConfigError`):
a typo there could only under-price a model whose real price we know. A call to
a declared model is bounded by its bytes, and anything in it that is not text is
refused (§10.15).

---

### 4.9 Threads and asyncio — one core, one lock

**Decided: both, from the start** (D10), **with one lock that is correct in
both** (D33).

*Corrected 2026-09-23.* This section used to say the two worlds need different
locks, a `threading.Lock` for threads and an `asyncio.Lock` for coroutines, and
that neither works in the other's world. That is true of a critical section that
awaits. It is false of ours, which cannot: the ledger is arithmetic with no I/O
(Rule 14), so nothing inside the lock ever awaits.

| what runs inside the lock | `threading.Lock` | `asyncio.Lock` |
|---|---|---|
| **never awaits** (ours) | correct in threads, in a loop, and in both at once | excludes nothing inside one loop; **hung 7 runs in 10** shared across threads (D33) |
| awaits (a store doing I/O) | deadlocks the loop | correct inside one loop |

Why the first row holds:

- **Inside one event loop** a task runs until it awaits, so no other task can
  reach the middle of a call that never awaits. The loop is already the lock.
- **Across threads** the threading lock excludes, as it always has.
- **A threading lock never held across an `await` cannot deadlock a loop.** That
  takes a task suspending while it holds the lock and another task on the same
  thread asking for it — which needs an `await` inside.

**The cost, stated rather than discovered:** a coroutine waits while a thread
holds the lock. Measured on 3.14 (D33): about 3 µs per reserve and settle with
nothing contending; about 1 ms at the 99th percentile when eight threads do
nothing but hammer the same ceiling. That is the trade §4.9.3 already makes for
the audit log, whose lock is held across a disk write.

#### 4.9.1 The split that makes this cheap

**The accounting is pure; the store adds the lock.** So the arithmetic exists
exactly once:

```
_BudgetCore          pure, no locking, no I/O, no clock of its own
                     try_reserve(worst_case, now) -> Reservation | Denial
                     settle(reservation_id, actual, now)
                     release(reservation_id)
      │
      └── InMemoryBudgetStore   threading.Lock around each core call —
                                threads, coroutines, or both at once
```

This is not an abstraction for a second case that does not exist (Rule 12). **The
property tests in §11 run against `_BudgetCore` alone**, so the guarantee is
proved once, with no lock in the picture; the store then needs only tests proving
its lock is actually applied, which is a far smaller surface. It is also what
makes one lock enough: a core with no I/O gives the lock nothing to await.

**The core returns a `Denial`; the store raises it.** That is not a breach of
§3's "every denial raises" — §3 governs the *public* surface, and `_BudgetCore`
is private. The split is deliberate: a property test wants to inspect thousands
of outcomes without wrapping every one in a `try`, while a caller must not be
able to ignore one. Converting value to exception is the store's job, and it is
the only place that conversion happens.

#### 4.9.2 There is no wrong world to catch

*Replaced 2026-09-23.* This section required a sync store touched inside a
running event loop to raise `ConfigError`. S2 built that guard; D33 removed it.
It refused a call that was safe, and the remedy it would have named — a second,
`asyncio` store — was the design measured hanging.

What it protected is kept by construction instead of by a check: **no store may
hold its lock across an `await`.** The store's methods are plain functions, so
they cannot. A store that must await inside its critical section — one doing I/O
— is a different design and reopens this section (D33, *Reversed by*).

#### 4.9.3 Audit writes stay synchronous in both worlds

**Not required for v1: an async audit writer.** The audit log is a local file
append under its own lock (§6), measured in microseconds. Making it async buys
back a sliver of event-loop time and costs either a dependency or a thread pool —
both forbidden or unwarranted today. It does mean an async program briefly blocks
the loop on every decision, and that is stated here rather than discovered.
*Trigger: the first measurement showing audit writes are a material share of the
overhead budget (Rule 17), or the first user on a slow or networked filesystem.*

---

## 5. Policy

```yaml
version: 1
policy_id: "prod-2026-09"

defaults:
  decision: deny                 # anything unmatched is denied. "deny" is the
                                 # only accepted value — a policy file may not
                                 # switch off locked decision #6 (D15).
  assumed_max_output_tokens: null
  fail_open: false               # §7. Off by default, noisy when on.

agents:
  - id: "refund-bot"
    mode: enforce                # enforce | shadow. Shadow records a rule's
                                 # refusal as would_deny and lets the call on;
                                 # the budget enforces in both (D48).
    budget:
      period: day                # day | hour | session
      limit_usd: "50.00"
    models:
      allow: ["claude-sonnet-5"]
      deny: []                   # explicit deny wins here too (D15)
    tools:
      allow:
        - name: "lookup_order"
        - name: "refund"
          constraints:
            order_id:   {}
            amount_usd: { max: "500.00" }
            currency:   { in: ["USD", "EUR"] }
          requires:              # only after an admitted lookup_order, earlier
            tool: "lookup_order" # in this session, with the same order_id
            same: ["order_id"]   # (D58)
          rate:   { calls: 20, seconds: 3600 }   # D59
          repeat: { seconds: 60 }                # not the same call twice in 60s
      deny: ["transfer_funds"]   # explicit deny always wins
```

### 5.1 Rules
- **Deny by default.** No matching agent, model or tool rule ⇒ denied.
- **Arguments are a closed set** when a tool rule declares `constraints` (D14).
  Those are the only argument names that tool may be called with; an unnamed one
  is denied. A name carrying predicates **must be supplied** — a constraint you
  can dodge by omitting the argument is not a constraint. Declare a name as `{}`
  to permit it without constraining or requiring it. A tool rule with no
  `constraints` block does not check arguments at all.
- **An agent's `budget` is optional**, for tool-only agents. An agent with no
  declared ceiling has every LLM call denied: no ceiling is not an open one.
- **Explicit deny beats allow**, always, at every level.
- **Shadow mode covers rules, never the budget** (D48). For an agent with
  `mode: shadow`, a refusal by a model or tool rule is recorded as `would_deny`
  and the call goes on, recorded as any other call is; a warning is logged on
  every call. A ceiling that would be breached, a call that cannot be priced and
  an agent with no budget are refused in every mode. "No budget" is its own
  check, made after the rules and never offered to shadow mode, and a model a
  shadowed rule let through is refused unless the price table carries it, so
  its name cannot reach an error (D26).
- **Constraints are declarative and dumb**: `max`, `min`, `in`, `equals`,
  `matches` (anchored regex, no backtracking-unsafe constructs), and
  `not_matches` (D49): refused if the pattern is found **anywhere** in the value,
  ignoring case, under the same pattern limits. It is how a policy refuses
  `rm -rf` or `DROP TABLE` inside an otherwise permitted shell tool. A value
  that is not a string, or is over 1 MiB, fails it.
  **No expression language, no lambdas, no eval.** An expression language in a
  policy file is a remote code execution vector.
- **Three rules remember** (D58, D59): `requires`, `rate` and `repeat`. A tool with
  `requires: {tool, same}` is permitted only after a call to `tool` that the
  checkpoint admitted earlier in the same session, with equal values for every
  argument in `same` (`same` may be empty). Only a string, an integer, a
  boolean or null can be compared, with its type; any other value meets nothing.
  The session keeps a SHA-256 digest of those values, never the values, and
  drops it when its `with` block ends. Admitted means admitted by the rules: a
  call shadow mode let through does not count, and admitted is not succeeded,
  since Paveo sees a call before it runs, never its outcome. A `requires` that
  could never be met (a tool the agent may not call, a chain that comes back to
  itself, or an argument either tool could never be passed) fails to load.
  `rate: {calls, seconds}` refuses a call once that many admitted calls to the
  tool fall inside the last `seconds`; `repeat: {seconds, same?}` refuses a call
  whose arguments, every one or those `same` names, equal an admitted call's
  inside the last `seconds`. Both
  are whole numbers from 1 (at most 10,000 calls and 86,400 seconds). Both count
  per session. The time they judge by moves on by how far the clock moved
  forward since its last reading, and not at all when it moved back, so it never
  runs backwards and never freezes: a call already forgotten cannot fall back
  inside a window, and a clock corrected after running fast holds the windows for
  one call, not until it catches up. **A checkpoint
  with no memory refuses such a tool rather than skip the rule**: replay, init's
  check, and the guard when the agent sends no session id. The `paveo` guard
  keeps each agent session's memory in `.paveo/memory/`, one private file per
  session holding a random salt, salted digests and times, never a value, private
  to its owner and deleted after a week untouched; it is
  locked from the moment the memory is read until it is written back, so hooks
  run at once take turns, and a file that is damaged, too large or a link
  refuses every call from its session.
- Policy is **immutable once loaded**. Reloading creates a new policy object with
  a new hash. Live mutation is not supported.
- Unknown keys are a **hard error**, not ignored. A typo'd `deny:` that silently
  does nothing is a breach waiting to happen.
- **Policy is JSON in the core** (`json` is stdlib), so the zero-dependency
  guarantee survives. YAML is an **optional extra** (`paveo[yaml]`) parsed
  with `yaml.safe_load` only — never `yaml.load`, never pickle. The YAML above
  is shown for readability; `paveo.json` is the canonical form.
  *Corrected 2026-09-19: the original draft required PyYAML while claiming zero
  dependencies in four places.*

### 5.2 Identity
`agent_id` — which agent. `principal` — on whose behalf (optional, opaque
string). Both appear in every audit record. Together they are the beginning of
an answer to the confused-deputy problem: **an action is attributable to an
agent *and* the human it claims to act for.** v1 records the pair and enforces
policy on `agent_id`; delegation chains are v2.

---

## 6. Audit log

Append-only JSONL, one record per decision, written locally with mode `0600`.

```json
{"v":1,"seq":42,"ts":"2026-09-19T11:02:04.881Z",
 "agent_id":"refund-bot","principal":"user_123",
 "action":{"kind":"llm","model":"claude-sonnet-5"},
 "decision":"deny","reason":"budget_exceeded",
 "estimated_cost_usd":"0.0421","actual_cost_usd":null,
 "rate_key":"claude-sonnet-5|speed=standard",
 "usage_by_class":{"input":512,"cache_read":8192,"cache_write_5m":2048,"output":0},
 "price_table_stale":false,
 "policy_id":"prod-2026-09","policy_hash":"sha256:…","plan":"developer",
 "prices_version":"2026-09-01",
 "prev_hash":"sha256:…","hash":"sha256:…"}
```

A tool decision uses the same record with a different `action`, and **no
arguments** — the constraint that fired is named, the value that failed it is not:

```json
 "action":{"kind":"tool","name":"refund"},
 "decision":"deny","reason":"constraint_violated","rule":"refund.amount_usd.max"
```

`usage_by_class`, `rate_key` and the cost fields are `null` for a tool decision.

A tool call refused by `paveo stop` is a `deny` with `reason` `stopped` and
`rule` `paveo.stop` (D49).

`decision` is `allow`, `deny`, `settle`, or **`would_deny`** for a refusal that
shadow mode let through (D48). A `would_deny` record is written where the `deny`
would have been, carrying its `reason` and `rule`, and the call's own `allow` or
`deny` follows it. So `allow` always counts the calls that went out, and
`would_deny` counts what enforcing would have stopped.
Recording the *rule* rather than the *value* is what keeps locked decision #5
true while still making a denial explainable a year later.

**`action.name` carries a tool name only when the policy declares it.** A name
matching an `allow` or `deny` entry is one of a fixed set the operator wrote, so
it is safe and it is the most useful thing in the record. A name matching nothing
was chosen by the model — and a model under injection chooses it — so it is
recorded as `<undeclared>` and the permitted names go in the remedy instead. The
deny path is the one an attacker can always reach, which is what makes this the
cheapest exfiltration channel in the library if it is left open. D26.

- `hash = sha256(canonical_json(record_without_hash) || prev_hash)` — a hash
  chain. Editing record *n* invalidates every record after it.
- **This is tamper-EVIDENT, not tamper-proof.** Say exactly that in the README.
  Someone with write access can rewrite the whole chain. Overclaiming here is
  how you lose a security reviewer permanently.
- **The chain is anchored at both ends.** Its start is `seq 1`; its finish is a
  fixed-width `<log>.anchor` file written `0600` beside the log, recording the
  last `seq` and hash after every append. Written *after* the record, never
  before, so the anchor is a floor and can never claim a record the log lacks.
  A log that has gone backwards from its anchor is refused on open. **A missing
  anchor warns and is rebuilt rather than refusing** — §10.13 and D20 explain the
  trade.
- **Verification compares bytes, not just the parsed record.** `json.loads` is
  not injective — duplicate keys collapse, and Python keeps the last — so a
  forged line carrying shadowed values would otherwise re-hash to the same
  digest. `verify_chain` requires the line to *be* `canonical_json(record)`. D19.
- **No prompts. No completions. No tool arguments. No principal PII beyond the
  opaque identifier the caller supplied.** Tool *names* yes; tool *arguments*
  no — arguments are where the customer data lives.
- `rate_key` and `usage_by_class` record **why a number was that number** — which
  base pair was selected and how the tokens divided. Counts and costs, never
  content, so locked decision #5 is untouched. Without them a disputed charge is
  unauditable a year later (Rule 19).
- **Audit writes are serialised under their own lock**, separate from the budget
  lock. `seq` and `prev_hash` form a chain; two threads appending concurrently
  corrupt it. The record is built, hashed and flushed while holding the lock.
  **Across processes too** (D49): each append also holds an exclusive `flock`
  on the log and re-reads the tail and the anchor under it, so any number of
  processes (a Claude Code hook is one per tool call) share one chain.
- **When the audit log cannot be written, we raise and do not attempt to log the
  failure.** Otherwise a full disk produces infinite recursion. The exception
  surfaces to the caller with the underlying OS error; the process's own stderr
  is the last resort, and we do not pretend otherwise.
- `verify_chain(path)` ships in v1 and is part of the public API.

---

## 7. Failure modes

| Condition | v1 behaviour |
|---|---|
| Policy file missing / invalid | `ConfigError` at load. Nothing starts. |
| Budget store unavailable | **Deny** (`PolicyUnavailable`) |
| Unknown model, no default price | **Deny** (`PricingUnknown`) |
| Price-affecting request parameter set to a value the table lacks | **Deny** (`PricingUnknown`) — §4.8.2 |
| Response reports a token class the table lacks | Reconcile at `max(rates)`, mark table stale, **deny every subsequent reserve** — §4.8.3 |
| Token counting fails | **Deny** |
| Audit log unwritable | **Deny** — an unlogged decision did not happen |
| The Claude Code guard fails in any way: no policy, an unwritable log, input it cannot read, an exception | **Deny**: exit 2. Claude Code treats any other exit as no objection (D49) |
| The guard takes longer than 4 seconds: a held lock, a slow pattern | **Deny**: it exits 2 at its own deadline, before Claude Code's timeout could let the call through (D49) |
| The guard cannot run at all: a missing binary, or a hook timeout set below 4 seconds | **Claude Code lets the call through.** Not ours to close; `paveo guard --selftest` detects the first (D49) |
| Codex has not been told to trust the hook, or the project | **Codex never runs it.** `paveo guard codex --selftest` fails on a missing trust record (D57) |
| A Cursor hook crashes, times out or cannot be found | **Deny**, because `init cursor` sets `failClosed: true`; `--selftest cursor` fails a hook without it (D57) |
| Agent has `mode: shadow` | A rule's refusal is recorded as `would_deny` and the call proceeds, with a warning **on every call**. Never the budget: every row above still refuses (D48) |
| `fail_open: true` set | Allow, and emit a warning **on every call**. It covers the fail-*closed* rows above — never a policy denial, which is a decision rather than a failure to decide (D18) |

`fail_open` exists because some teams genuinely prefer availability to control,
and pretending otherwise just gets the library ripped out. It is off by default,
noisy when on, and recorded in every audit line.

---

## 8. Errors

```
PaveoError
├── ConfigError          wiring is wrong: policy invalid or unloadable, an audit
│                     log or clock that cannot be trusted, or a store handed
│                     a ceiling it cannot honour
├── PolicyDenied         model or tool not permitted, or constraint violated
├── BudgetExceeded       reservation would breach the ceiling
├── PolicyUnavailable    could not evaluate — fail-closed
└── PricingUnknown       cannot price this call, therefore cannot bound it
```

`BudgetExceeded` carries `limit`, `spent`, `reserved`, `requested`.
`PolicyDenied` carries the rule that denied it, and `for_model`: the refusal as
text for the model, without the remedy (D48). **Error messages must say how to
fix it** and must never contain payload content.

---

## 9. Security requirements

- **No network egress.** `tests/test_no_egress.py` patches `socket.socket` and
  fails the suite if it is constructed during any library operation.
- **No `eval`, `exec`, `pickle`, `marshal`, or dynamic import of caller-supplied
  module paths.** Checked by a lint rule in CI.
- **Zero required runtime dependencies.** Tokenizers and provider SDKs are
  optional extras. Every dependency is something we ask the customer to trust.
- Audit file created `0600`; parent directory not world-writable (checked).
- Policy regexes are compiled only at policy load, capped in length, matched
  with `fullmatch` so anchoring is structural, and rejected if they apply a
  quantifier to a group that already contains one, repeat a group that holds an
  alternation, use a backreference, or use any group form but `(?:`. **This
  reduces ReDoS risk rather than eliminating it** — a chain of optional
  alternation groups or a polynomial shape such as `a*a*a*b` is not detected, so
  the matched value is length-capped as well. `re` has no timeout in the standard library,
  so static rejection at load is the only guard available without a dependency.
  D15.
- Releases: signed tags, published SBOM, pinned and hash-verified build deps.
- `SECURITY.md` with a disclosure address before the first public release.

---

## 10. Known gaps — publish these, don't hide them

1. **SDK-internal retries are invisible.** If the provider SDK retries inside a
   single call, we see one reservation and one usage figure. Bounded by the
   worst case, but the audit log under-counts attempts.
2. **In-memory budget is per-process.** Two processes sharing a ceiling both
   think they own it. The `BudgetStore` interface exists for this; only the
   in-memory implementation ships in v1.
3. **A crash mid-call leaks a reservation** until the process exits.
4. **We price from a static table.** Provider price changes silently make
   estimates wrong until the table is updated. The table carries an `as_of` date
   and every audit line carries its version, so a wrong number is at least
   attributable after the fact — but nothing detects the change at the time.
5. **We cannot stop an agent that doesn't call through us.** Paveo is a
   guard rail, not a sandbox. Say so first, before someone else does.
6. **A process that restarts loses its spend record.** With the in-memory store,
   something restarting hourly effectively has no daily ceiling. *§4.7.*
7. **Conservative token estimation over-denies by 3–4× on prompt-heavy
   workloads.** Fixable with the tokenizer extra; the default is safe, not
   accurate. *§4.5.*
8. **Tool enforcement only covers calls routed through `check_tool`.** A
   framework that dispatches tools itself bypasses it entirely.

9. **Reserving at `max()` over token classes over-reserves input by up to 2× the
   base rate — and up to 20× the actual input cost when the workload is
   cache-read-heavy** (true rate 0.1×, reserved at 2.0×), **40× on Opus 5.5 and
   80× on Fable 5.1**, whose cache reads are 0.05× and 0.025×.
   On a completion-heavy workload this is invisible, because output dominates.
   On a prompt-heavy one that relies on cache reads it can deny calls the
   customer could easily afford. Safe, never silent: the reserve-time class and rate are in every
   audit line. *§4.1. Fixable only by a provider telling us the split in
   advance, which none does.*

10. **A new price-affecting request parameter would be invisible to us.** We
    price against a declared list of parameters per model. If a provider ships a
    parameter that changes the rate and we have not added it, we under-price and
    the ceiling is wrong in the unsafe direction. **This gap cannot be closed by
    design, only by maintenance** — which is why the table carries a version and
    §12.5 asks how it gets re-verified. *§4.8.2.*

11. **Threads and asyncio against the same budget are safe only through one
    store object.** *Narrowed 2026-09-23 (D33); this used to say the mix was
    unsupported and undetected.* One lock now serves both worlds (§4.9), so a
    thread pool and an event loop spending one ceiling through one
    `InMemoryBudgetStore` cannot race. **Two store objects are two sets of
    books**, and each lets the agent spend the whole ceiling — §10.2's gap, one
    level down, and just as silent. `Paveo` must therefore hand one shared store
    to every session it opens, sync or async (S4); a session-period ceiling's own
    store is the one exception, by design (D31). And a coroutine can wait while a
    thread holds the lock: microseconds normally, about a millisecond at the
    99th percentile with eight threads contending for one ceiling.

13. **Tail truncation is detected only while the anchor survives.** The
    `<log>.anchor` file records how far the log reached, so deleting records from
    the end is caught on open and by `verify_chain` — and a wholesale rewrite
    from record 1 fails too, because the record at the anchored sequence has a
    different hash. **An attacker who deletes the anchor as well is back to being
    undetectable**, and a missing anchor is deliberately a warning rather than a
    refusal: one that could stop production traffic when a sidecar goes missing
    would be a denial of service with extra steps. The bar is raised from
    "delete three lines" to "delete three lines and a file you had to know
    existed", not to zero. *Trigger for going further — an anchor held somewhere
    the agent cannot write, which is the only real fix: the first user who needs
    the log as evidence against an insider rather than against accident.* D20.

12. **A new token class stops every agent until the table is updated.** §4.8.3
    fails closed on an unrecognised class, so a provider introducing one turns
    into an outage for every user on the old table, not a silent mis-count. This
    is the deliberate trade and it is the correct one — but it is an operational
    risk a customer must be told about before they deploy, not after. `fail_open`
    is the escape hatch, and it is noisy by design.

15. **A model priced in the policy is only as bounded as the operator's price.**
    Its calls go through a reader that knows no provider (D39): text is bounded
    by its bytes, which holds for every tokenizer, and anything that is not text
    is refused. What it cannot see is what a provider adds on its own side (for
    formatting tools, for example) and any request parameter that changes that
    provider's price. The operator wrote the price and is the one who knows what
    their provider bills. *For a guarantee that does not rest on the operator,
    use a provider with an adapter; trigger for a new adapter: its first user.*

14. **The table is the Claude API's own price list.** Claude Platform on AWS and
    Microsoft Foundry bill at the same rates. **Amazon Bedrock and Google Cloud
    set their own prices**, and their regional endpoints carry a 10% premium
    over global ones, chosen in client configuration rather than on the request,
    so nothing we can read says which was used. A Bedrock model id carries an
    `anthropic.` prefix and is refused as unknown; **a Google Cloud id is the
    bare first-party one and would be priced at first-party rates.** *Trigger
    for a partner table: the first user on either platform.* D38.

16. **Two things on OpenAI are charged high because they cannot be seen** (D41).
    The 10% regional uplift is chosen by endpoint or project region, so the ten
    eligible models are always charged it; and the tokens OpenAI adds rendering
    function definitions are not published, so 1,000 are added whenever tools
    are present. Both err high. *Trigger: the first user whose ceiling is
    measurably tighter than their bill because of either.*

17. **Two things are inferred rather than read, and one check stands behind
    both** (D42). Gemini's `max_output_tokens` is read as capping thinking too,
    from a guide written for another of Google's APIs; and OpenAI's and Gemini's
    tool-rendering overhead is a chosen 1,000 tokens. **A call that costs more than
    its reservation stops the table**, so an inference that is wrong fails closed
    on the first call that shows it.

---

## 11. Tests that must exist before v1 ships

- **Property:** across any interleaving of reserve/reconcile/error, `spent`
  never exceeds `limit` and `reserved` returns to zero. (`hypothesis`)
- **Concurrency — threads:** N threads against one budget, no overspend, no lost
  release.
- **Concurrency — asyncio:** N coroutines against one budget via
  `asyncio.gather`, same two assertions, with reservations held across an
  `await`, and again with worker threads spending the same ceiling. Must be a
  separate test: passing the thread test says nothing about whether the lock
  blocks or deadlocks a loop.
- **Concurrency — the core is proved once:** the §11 property test runs against
  `_BudgetCore` with no lock at all, so the arithmetic is proved independently of
  the store (§4.9.1). The store's tests then assert only that its lock is
  applied — between two threads, and between a thread and a coroutine in a
  running loop.
- **Both worlds, one lock:** a coroutine in a running event loop and a thread are
  never inside the ledger at once (§4.9, D33). *Replaces the wrong-world guard
  test, whose guard D33 removed.*
- **Client/session mismatch:** a sync client handed to an async session raises,
  and the reverse raises, rather than half-working.
- **Egress:** no socket constructed, anywhere, ever.
- **Policy:** malformed, unknown-key and empty policies all raise — never
  silently permit.
- **Audit:** chain verifies; mutating any record fails verification.
- **Redaction:** a test that asserts no request content appears in the audit file
  given a request containing a known sentinel string.
- **Pricing — the shape:** a response carrying `input`, `cache_read` and
  `cache_write_5m` in one usage object reconciles to the sum of three
  separately-rated classes. A table with one input rate cannot pass this.
- **Pricing — the modifier:** the same model string with and without the
  price-affecting parameter resolves to two different base pairs, and the more
  expensive one is never priced as the cheaper. Property: for every model and
  every permitted modifier value, the resolved rate is the table's, never a
  default.
- **Pricing — fail closed:** an unknown model, an unknown value for a declared
  price-affecting parameter, and an unrecognised class in a response each
  produce the §7 behaviour. The last one must also deny the *next* reserve.
- **Pricing — the reserve bound:** property, over random per-class splits
  summing to the same total — `actual ≤ worst_case`, always. This is the one
  that proves §4.1's `max()` is doing its job.

---

## 12. Open questions — decide before building, not during

1. Do we count tokens ourselves (a dependency, and drift risk) or require the
   caller to pass `max_tokens` and use the provider's returned usage only?
   **Leaning: require `max_tokens`, count input with an optional tokenizer
   extra, deny if neither available.**
2. Session-scoped vs process-scoped budget as the default period.
   **Leaning: `day`, because that's the runaway-overnight case people actually
   have.**
3. **Licence — decide before the first public commit.** *Upgraded 2026-09-19
   from a thirty-minute formality to a strategic question — see D13: our nearest
   competitor gives away more than our v1, free, under Apache 2.0, and you cannot
   out-restrict a free competitor. This must now be answered together with "what
   do we hold back that anyone would pay for?"* The strategy is
   source-available but not open source: readable, because we ask people to run
   it in their call path; not freely commercialisable, because the control plane
   is the business. Candidates: BSL 1.1 with a change date, Elastic License 2.0,
   or PolyForm. ~~**This is unresolved and blocks the first release**~~
   **RESOLVED 2026-09-20 — Elastic License 2.0. See D22.** BSL was rejected on
   the change-date maintenance it imposes per release, PolyForm on unfamiliarity
   to enterprise procurement. The cost is recorded rather than hidden: ELv2 is
   not OSI-approved, and organisations with a blanket policy against non-OSI
   licences will not evaluate us.

5. **How does the price table get re-verified, and by whom?** §10.10 is a
   maintenance gap, not a design gap — which means it is only closed by a person
   doing something on a schedule. A solo maintainer with limited hours is
   the wrong control for a manual monthly check (Rule 5: mechanical guards, never
   vigilance). Options: pin the `as_of` date and warn loudly in the README once
   it is over N days old; or accept it and say so. **Unresolved. It does not
   block `budget.py`, and it does block the first release** — an un-maintained
   price table is a ceiling that quietly stops being true.
   **RESOLVED 2026-09-24 (D43):** each provider's table records the day it was
   read; Paveo warns past 60 days; `make release-check` refuses past 30;
   `docs/PRICES.md` is the procedure.

4. ~~Does `check_tool` belong in v1, or is v1 budget-only?~~ **RESOLVED
   2026-09-19 — included. See D11.** The leaning was right for a reason that only
   became visible on inspection: the budget ceiling is declared *in the policy
   file* (§5) and every budget decision must reach the audit log (§7), so
   `policy.py` and `audit.py` were never optional for a budget-only v1. Tool
   permissions are constraint evaluation on top of two modules already required.
