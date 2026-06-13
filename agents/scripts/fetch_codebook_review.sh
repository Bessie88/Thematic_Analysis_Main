#!/usr/bin/env bash
# Fetch an approved codebook review from Supabase and materialize local JSON.
#
# Usage:
#   ./fetch_codebook_review.sh <review_id>
#   CODEBOOK_REVIEW_ID=<uuid> ./fetch_codebook_review.sh
#
# Requires agents/scripts/.env.supabase (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY).
# Optional: PIPELINE_SLUG (default: codebook-review-smoke-test)

set -euo pipefail

AGENTS_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_ROOT="$(cd "$AGENTS_SCRIPTS/.." && pwd)"
REPO_ROOT="$(cd "$AGENTS_ROOT/.." && pwd)"

SECRETS_FILE="${SECRETS_FILE:-$AGENTS_SCRIPTS/.env.supabase}"
if [ -f "$SECRETS_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_FILE"
  set +a
else
  echo "Error: $SECRETS_FILE not found" >&2
  exit 1
fi

export CODEBOOK_REVIEW_ID="${1:-${CODEBOOK_REVIEW_ID:-}}"
if [ -z "$CODEBOOK_REVIEW_ID" ]; then
  echo "Usage: $0 <review_id>" >&2
  echo "   or: CODEBOOK_REVIEW_ID=<uuid> $0" >&2
  exit 1
fi

export PIPELINE_SLUG="${PIPELINE_SLUG:-codebook-review-smoke-test}"
export PYTHONPATH="$REPO_ROOT"

exec python "$AGENTS_SCRIPTS/fetch_approved_codebook.py"
