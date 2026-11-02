#!/bin/bash
# ClimateSense AI launcher for macOS / Linux
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Install it from https://www.python.org/downloads/ or run: brew install python"
  exit 1
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 9):
    sys.exit("Python 3.9+ is required (3.10+ recommended). Found: " + sys.version.split()[0])
PY

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Running offline self-tests..."
python -m unittest discover -s tests

streamlit run app.py
