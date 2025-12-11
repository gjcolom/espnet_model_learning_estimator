#!/bin/bash
# ESPnet Training Analyzer - Run Script
# Activates the virtual environment and runs the analyzer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at ${VENV_DIR}"
    echo "Please run ./setup_env.sh first to create the environment."
    exit 1
fi

# Activate virtual environment and run analyzer
source "${VENV_DIR}/bin/activate"
python "${SCRIPT_DIR}/espnet_log_parser.py" "$@"
