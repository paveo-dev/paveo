# Starter policies

Copy one, change it to fit, and run it in shadow mode (`"mode": "shadow"` on
the agent) for a day before it enforces. Everything not listed is refused, so a
starter policy is a place to begin, not a finished one.

- **`claude-code.json`**, which `paveo init claude-code` installs: Claude Code on a repo
  that matters. Bash, Write, Edit and NotebookEdit are allowed. Bash refuses `rm -rf`, force
  pushes, `git reset --hard`, `git clean -f`, `DROP`/`TRUNCATE`,
  `terraform destroy`, `kubectl delete` and `mkfs`. Nothing may touch the
  guard's own `.paveo/` folder or Claude Code's settings (a Bash command that
  mentions `.pave`, `.claude`, a `settings.json` or `.mcp.json` at all, except
  `.claude.c…` and `.claude.a…` so that `claude.com` and `claude.ai` addresses
  pass; Write and Edit anywhere under `.claude` except plan mode's
  `.claude/plans`, on `.mcp.json`, on any path with `..`, and on paveo's own
  installed code under `site-packages/paveo` or any `.pth` file), and Claude may not
  run `paveo resume`. Pair it with the matcher `Bash|Write|Edit|NotebookEdit`: any other
  tool the hook sees is refused. The tool fields it lists are the ones Claude
  Code sent on 2026-09-25. If a later version adds one, its calls are refused
  with a message naming the fix, rather than let through.
- **`codex.json`** and **`cursor.json`**, which `paveo init codex` and
  `paveo init cursor` install (D57). The same nine destructive-command patterns
  as `claude-code.json`, word for word (a test keeps the three copies equal),
  plus the guard's own folder, the agent's hook folder (`.codex`, `.cursor`),
  any `hooks.json`, and `paveo resume`. Codex's `apply_patch` is judged by `paths`, the files
  its patch headers name, which the guard reads from the patch: no path under
  `.paveo` or `.codex`, and none with a `..` step. Cursor's `Write` and
  `Delete` are refused under `.paveo` and `.cursor`, on any path with
  `..`, and, in both, on `site-packages/paveo` and any `.pth` file. **Cursor does not document its file tools' fields, so `file_path` and
  `content` are a guess**: if Cursor sends others, every write is refused
  until this file is corrected.
- **`support-agent.json`**, for the library: a support agent that looks up
  orders and customers, replies, and refunds up to $200 in three currencies,
  only an order it has looked up earlier in the same session (`requires`), and
  no order refunded twice within an hour of one session (`repeat`, on the order
  alone, since an amount or a currency can change between tries). It may never delete
  an account, change an email or payout account, or issue credit. $20 a day on Sonnet 5 or Haiku 4.5.
- **`finance-ops.json`**, for the library: reads the ledger, invoices up to
  $5,000, and pays known vendors (`V-` and six digits) up to $1,000 in USD, no
  vendor twice within an hour of one session. It may never change bank details, add
  a payee, approve a payment or delete a ledger entry. $10 a day on Sonnet 5.

The two library policies give their agent no tool that writes files, so it
cannot reach its own policy or log. If you add one, refuse those paths with
`not_matches` the way `claude-code.json` does.

**These patterns are a guard rail, not a sandbox.** They catch the common
spellings of a destructive command, not every way to write one.
