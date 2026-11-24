#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This setup script is intended for Raspberry Pi OS / Linux." >&2
    exit 1
fi

if command -v python3.13 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3.13)"
elif command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3)"
else
    echo "Python 3.13 is required. Install it and try again." >&2
    exit 1
fi

BASE_VERSION="$("$BASE_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$BASE_VERSION" != "3.13" ]]; then
    echo "Python 3.13 is required; found Python $BASE_VERSION at $BASE_PYTHON." >&2
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/config.json" ]]; then
    cp "$PROJECT_DIR/config.example.json" "$PROJECT_DIR/config.json"
    echo "Created config.json from the example. Add your RTSP URL, then run setup.sh again."
    exit 1
fi

if [[ ! -x "$VENV_PY" ]]; then
    echo "Creating project virtual environment..."
    "$BASE_PYTHON" -m venv "$VENV_DIR"
else
    VENV_VERSION="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ "$VENV_VERSION" != "3.13" ]]; then
        echo "The existing .venv does not use Python 3.13; rebuilding it..."
        rm -rf -- "$VENV_DIR"
        "$BASE_PYTHON" -m venv "$VENV_DIR"
    fi
fi

if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
    echo "requirements.txt is missing from $PROJECT_DIR." >&2
    exit 1
fi

# pip compares installed packages with requirements and only installs or changes
# packages that are missing or incompatible; there is no unconditional upgrade.
"$VENV_PY" -m pip install --disable-pip-version-check -r "$PROJECT_DIR/requirements.txt"

cd "$PROJECT_DIR"
exec "$VENV_PY" "$PROJECT_DIR/script.py" "$@"
