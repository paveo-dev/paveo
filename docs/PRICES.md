# Re-verifying the price tables

**A price table is a ceiling only while it is true.** A rate the provider raised
since the table was read is under-charged, which is the one direction that
breaches a ceiling (§10.4, §10.10). So the tables carry the day each provider
was verified (`prices.VERIFIED`), Paveo warns once one is over 60 days old, and
**`make release-check` refuses to ship one over 30 days old** (§12.5, D43).
**A release is made through `make release-check`, and only through it**: the gate
protects nothing a manual `python -m build` skips, which is why S5's checklist
starts with it.

This file is how to re-read them. It takes about an hour for all three.

## The rule that matters most

**Read the provider's own pages, in the session that edits the table, and
confirm anything surprising on a second page.** Not a cached table, not a
session card, not memory (Rule 4). Twice this repo's plan was wrong about a
price and the page was right (D38), and once a page's own summary contradicted
its table (D41). Tests pin the **figures the page prints**, not our multipliers,
so a wrong multiplier disagrees with the tests instead of being copied into them.

## Per provider

For each: fetch the pages, compare every row in the listing and the test's
`PUBLISHED`/`STANDARD`/`FAST`/`PRIORITY` tables, update both, then update
`VERIFIED[provider]` to today. Add a model only if someone uses it; remove one
the page retires.

### Anthropic (`_LISTINGS`, `tests/test_prices.py`)

- `platform.claude.com/docs/en/about-claude/pricing.md`: base input, 5m and 1h
  cache writes, cache hits, output, per model; fast-mode rows; the footnotes on
  cache-hit multipliers (0.025× Fable 5.1, 0.05× Opus 5.5 as of 2026-09-23).
- `platform.claude.com/docs/en/manage-claude/data-residency`: which models take
  `inference_geo`, the multiplier, and **whether an unset geo can still be
  billed as US** through a workspace default (D38).
- Check: any new price-affecting request parameter (§10.10); any new usage
  field (it would stop the table, §4.8.3); the per-image token cap on the vision
  page, used by the adapter's allowance.

### OpenAI (`_OPENAI_LISTINGS`, `tests/test_openai.py`)

- `developers.openai.com/api/docs/pricing.md`: Standard, Fast, Flex, and the
  long-context rows; cache-write columns.
- The model pages for the newest models (`/api/docs/models/<id>`): the
  long-context threshold and whether it bills the whole request.
- `/api/docs/guides/priority-processing.md`: **whether an unset `service_tier`
  can still be billed as Fast** through a project setting (D41).
- `/api/docs/guides/your-data.md`: the data-residency list and uplift; a model
  that joins the list must get `regional=True`.
- `/api/docs/guides/prompt-caching.md`: which models charge for cache writes.

### Gemini (`_GEMINI_LISTINGS`, `tests/test_gemini.py`)

- `ai.google.dev/gemini-api/docs/pricing`: every text model's input, cached,
  output and Priority rows, the "> 200k" tiers, and **dated price changes**
  ("through December 31, 2026 … starting January 1, 2027"), which go in
  `changes` with the date they take effect (D42).
- The google-genai SDK's `types.py` (raw from GitHub): `ServiceTier` (is unset
  still standard?), `GenerateContentResponseUsageMetadata` (what `total` sums),
  `HttpOptions` (anything new that reaches the request body).

## After

`make release-check` must pass. Record in the decision record what changed and
anything the pages showed that the adapters must now refuse or price.
