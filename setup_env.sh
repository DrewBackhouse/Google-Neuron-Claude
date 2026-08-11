#!/usr/bin/env bash
# Rebuild the Python environment in the Linux sandbox.
# The sandbox resets between sessions, so run this at the start of any chat
# that executes code. Takes ~15 s.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install --break-system-packages --only-binary=:all: -r "$DIR/requirements.txt"
python3 - <<'PY'
import importlib
mods = ["numpy","scipy","matplotlib","sympy","cirq","cirq_google","qiskit","qiskit_aer"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        bad.append(m); print(f"  FAIL {m:14s} {type(e).__name__}")
raise SystemExit(1 if bad else 0)
PY
