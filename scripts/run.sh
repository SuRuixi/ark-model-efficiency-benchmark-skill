#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$SKILL_DIR/.venv"
PYTHON="${PYTHON:-python3}"

command -v arkcli >/dev/null 2>&1 || {
  echo "arkcli is required. Install and authenticate Ark CLI first." >&2
  exit 1
}

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi

STAMP="$VENV/.requirements.sha256"
CURRENT="$(
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$SKILL_DIR/requirements.txt" | awk '{print $1}'
  else
    "$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$SKILL_DIR/requirements.txt"
  fi
)"

if [[ ! -f "$STAMP" ]] || [[ "$(cat "$STAMP")" != "$CURRENT" ]]; then
  "$VENV/bin/python" -m pip install --disable-pip-version-check -q -r "$SKILL_DIR/requirements.txt"
  printf '%s\n' "$CURRENT" > "$STAMP"
fi

exec "$VENV/bin/python" "$SCRIPT_DIR/ark_bench.py" "$@"
