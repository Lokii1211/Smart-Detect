#!/usr/bin/env bash
# scripts/run_tests.sh — CI entry point for the identity-arbitration suite.
#
# Hermetic: no camera, no external database, no network, no model weights.
# Exits non-zero if any test fails or coverage of the identity-arbitration
# modules falls below the floor.
#
#   ./scripts/run_tests.sh            # tests + coverage gate
#   ./scripts/run_tests.sh --html     # also write htmlcov/
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

# Coverage floor for the identity decision layer. Raise as coverage improves;
# never lower it to make a red build green.
MIN_COVERAGE=80

# macOS ships bash 3.2, where "${EXTRA[@]}" on an empty array trips `set -u`.
EXTRA=""
if [[ "${1:-}" == "--html" ]]; then
  EXTRA="--cov-report=html"
fi

echo "== identity-arbitration test suite =="
"$PY" -m pytest \
  --cov=recognition.smart_identifier \
  --cov=config.identity_config \
  --cov-report=term-missing \
  --cov-fail-under="$MIN_COVERAGE" \
  ${EXTRA}

echo
echo "PASS: suite green and coverage >= ${MIN_COVERAGE}% on the identity layer."
