#!/usr/bin/env bash
# The Action's test step (action.yml): `paveo test` with each expected action,
# the report in the job summary too, and paveo's own exit as the job's, so 1 (a
# spelling got past) and 2 (could not tell) both fail the build.
set -euo pipefail
set -f  # a * in expect stays a word, never this repository's file names

args=()
for action in ${PAVEO_EXPECT:-}; do
  args+=(--expect "$action")
done

set +e
report=$(paveo test ${args[@]+"${args[@]}"} --json "${PAVEO_JSON:-paveo-test.json}" 2>&1)
code=$?
set -e

printf '%s\n' "$report"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '### paveo test\n\n```\n%s\n```\n' "$report" >> "$GITHUB_STEP_SUMMARY"
fi
exit "$code"
