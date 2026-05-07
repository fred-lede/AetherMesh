#!/usr/bin/env bash
set -u

ROOT="${1:-.}"

fixed_bom=0
fixed_crlf=0
tab_found=0
yaml_ok=0
yaml_fail=0
shell_ok=0
shell_fail=0
python_ok=0
python_fail=0

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

print_section() {
  printf '\n==> %s\n' "$1"
}

fix_bom() {
  local f="$1"
  if head -c 3 "$f" 2>/dev/null | grep -q $'^\xEF\xBB\xBF'; then
    echo "[FIX BOM] $f"
    local tmp
    tmp="$(mktemp)"
    tail -c +4 "$f" > "$tmp" && cat "$tmp" > "$f"
    rm -f "$tmp"
    fixed_bom=$((fixed_bom + 1))
  fi
}

fix_crlf() {
  local f="$1"
  if grep -q $'\r' "$f" 2>/dev/null; then
    echo "[FIX CRLF] $f"
    perl -pi -e 's/\r$//' "$f"
    fixed_crlf=$((fixed_crlf + 1))
  fi
}

check_tabs() {
  local f="$1"
  if grep -n $'\t' "$f" >/dev/null 2>&1; then
    echo "[TAB FOUND] $f"
    grep -n $'\t' "$f" || true
    tab_found=$((tab_found + 1))
  fi
}

validate_yaml() {
  local f="$1"

  if have_cmd python3; then
    if python3 - <<'PY' "$f"
import sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    import yaml
except Exception:
    sys.exit(2)

try:
    with p.open('r', encoding='utf-8') as fh:
        yaml.safe_load(fh)
    sys.exit(0)
except Exception as e:
    print(e)
    sys.exit(1)
PY
    then
      echo "[YAML OK] $f"
      yaml_ok=$((yaml_ok + 1))
    else
      rc=$?
      if [ "$rc" -eq 2 ]; then
        echo "[YAML SKIP] $f (PyYAML not installed)"
      else
        echo "[YAML FAIL] $f"
        yaml_fail=$((yaml_fail + 1))
      fi
    fi
  else
    echo "[YAML SKIP] $f (python3 not found)"
  fi
}

validate_shell() {
  local f="$1"
  if bash -n "$f" >/dev/null 2>&1; then
    echo "[SHELL OK] $f"
    shell_ok=$((shell_ok + 1))
  else
    echo "[SHELL FAIL] $f"
    bash -n "$f" || true
    shell_fail=$((shell_fail + 1))
  fi
}

validate_python() {
  local f="$1"
  if have_cmd python3; then
    if python3 -m py_compile "$f" >/dev/null 2>&1; then
      echo "[PY OK] $f"
      python_ok=$((python_ok + 1))
    else
      echo "[PY FAIL] $f"
      python3 -m py_compile "$f" || true
      python_fail=$((python_fail + 1))
    fi
  else
    echo "[PY SKIP] $f (python3 not found)"
  fi
}

process_file() {
  local f="$1"

  fix_bom "$f"
  fix_crlf "$f"

  case "$f" in
    *.yaml|*.yml)
      check_tabs "$f"
      validate_yaml "$f"
      ;;
    *.sh|*.bash|*.zsh)
      check_tabs "$f"
      validate_shell "$f"
      ;;
    *.py)
      validate_python "$f"
      ;;
  esac
}

print_section "掃描目錄: $ROOT"

while IFS= read -r -d '' file; do
  process_file "$file"
done < <(
  find "$ROOT" -type f \
    \( -name "*.sh" \
    -o -name "*.bash" \
    -o -name "*.zsh" \
    -o -name "*.py" \
    -o -name "*.yaml" \
    -o -name "*.yml" \
    -o -name "*.json" \
    -o -name "*.md" \
    -o -name "*.txt" \
    -o -name "*.toml" \
    -o -name "*.ini" \
    -o -name "*.conf" \) \
    -print0
)

print_section "結果摘要"
echo "BOM 修正數量      : $fixed_bom"
echo "CRLF 修正數量     : $fixed_crlf"
echo "Tab 問題檔案數    : $tab_found"
echo "YAML 驗證成功數   : $yaml_ok"
echo "YAML 驗證失敗數   : $yaml_fail"
echo "Shell 驗證成功數  : $shell_ok"
echo "Shell 驗證失敗數  : $shell_fail"
echo "Python 驗證成功數 : $python_ok"
echo "Python 驗證失敗數 : $python_fail"