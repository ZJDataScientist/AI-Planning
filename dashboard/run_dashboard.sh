#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d ".venv-dashboard" ]]; then
  source .venv-dashboard/bin/activate
fi
streamlit run dashboard/1_Research_Overview.py
