#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> 1. Running tests..."
python -m pytest tests/ -x -q 2>&1 | tail -5

echo ""
echo "==> 2. Syncing .env.example to profile templates..."
python scripts/sync_env_examples.py

echo ""
echo "==> 3. Staging all changes..."
git add -A

echo ""
echo "==> 4. Changes staged:"
git diff --cached --stat

echo ""
read -r -p "==> Commit message (or empty to skip commit): " msg

if [ -z "$msg" ]; then
    echo "Commit skipped."
    exit 0
fi

git commit -m "$msg"

echo ""
echo "==> 5. Pushing to origin..."
git push

echo ""
echo "Done."
