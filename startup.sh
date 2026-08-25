#!/usr/bin/env bash
set -euo pipefail

APP_FILE="${APP_FILE:-app.py}"
PORT="${PORT:-8501}"
STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

export STREAMLIT_SERVER_PORT="${PORT}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS}"
export ENABLE_LOCAL_LLM="${ENABLE_LOCAL_LLM:-False}"

exec streamlit run "${APP_FILE}" \
  --server.address="${STREAMLIT_SERVER_ADDRESS}" \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
