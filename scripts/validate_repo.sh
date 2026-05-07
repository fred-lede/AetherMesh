#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

echo "==> validate shell files"
find "$ROOT" -type f \( -name "*.sh" -o -name "*.bash" -o -name "*.zsh" \) -print0 | \
while IFS= read -r -d '' f; do
  echo "[CHECK] $f"
  bash -n "$f"
done

echo
echo "==> validate python files"
find "$ROOT" -type f -name "*.py" -print0 | \
while IFS= read -r -d '' f; do
  echo "[CHECK] $f"
  python3 -m py_compile "$f"
done

echo
echo "==> validate yaml files"
python3 - <<'PY' "$ROOT"
import sys
from pathlib import Path

root = Path(sys.argv[1])

try:
    import yaml
except Exception:
    print("PyYAML not installed, skip YAML validation")
    raise SystemExit(0)

for p in root.rglob("*"):
    if p.suffix.lower() in {".yaml", ".yml"} and p.is_file():
        print(f"[CHECK] {p}")
        with p.open("r", encoding="utf-8") as fh:
            yaml.safe_load(fh)

print("All validations passed.")
PY