#!/bin/bash
cd "$(dirname "$0")"

PYTHON_BIN=".venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
"$PYTHON_BIN" -m streamlit run scripts/app_revisao_faces.py
