#!/usr/bin/env bash

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$project_root/.runtime"
log_dir="$runtime_dir/logs"
state_file="$runtime_dir/pathshongi-linux.pids"
python_bin="$project_root/.venv/bin/python"
gui_port="${PATHSHONGI_GUI_PORT:-8000}"
answer_port="${PATHSHONGI_ANSWER_PORT:-8080}"
embedding_port="${PATHSHONGI_EMBEDDING_PORT:-8081}"
open_browser=1

usage() {
    cat <<'EOF'
Usage: ./Launch-Pathshongi.sh [options]

Options:
  --port PORT       Web interface port (default: 8000)
  --no-browser      Do not open the web interface automatically
  -h, --help        Show this help

Environment overrides:
  PATHSHONGI_GUI_PORT
  PATHSHONGI_ANSWER_PORT
  PATHSHONGI_EMBEDDING_PORT
  PATHSHONGI_LLAMA_SERVER
  PATHSHONGI_MODEL
EOF
}

while (($#)); do
    case "$1" in
        --port)
            if (($# < 2)); then
                echo "Missing value for --port." >&2
                exit 2
            fi
            gui_port="$2"
            shift 2
            ;;
        --no-browser)
            open_browser=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for port in "$gui_port" "$answer_port" "$embedding_port"; do
    if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
        echo "Invalid port: $port" >&2
        exit 2
    fi
done

mkdir -p "$log_dir"

port_is_open() {
    "$python_bin" - "$1" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.socket() as connection:
    connection.settimeout(0.35)
    raise SystemExit(connection.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)
PY
}

wait_for_port() {
    local port="$1"
    local label="$2"
    local timeout="$3"
    local elapsed=0

    printf '  %s' "$label"
    while ((elapsed < timeout)); do
        if port_is_open "$port"; then
            printf ' ready\n'
            return 0
        fi
        printf '.'
        sleep 1
        ((elapsed += 1))
    done
    printf ' failed\n' >&2
    return 1
}

record_pid() {
    printf '%s=%s\n' "$1" "$2" >>"$state_file"
}

start_background() {
    local name="$1"
    shift
    nohup "$@" >"$log_dir/$name.out.log" 2>"$log_dir/$name.error.log" &
    local process_id=$!
    record_pid "$name" "$process_id"
    printf '%s' "$process_id"
}

if [[ ! -x "$python_bin" ]]; then
    echo "Project Python was not found at $python_bin." >&2
    echo "Create it with: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

if ! "$python_bin" -c 'import fastapi, numpy, torch, transformers, uvicorn' >/dev/null 2>&1; then
    echo "Core Python dependencies are missing." >&2
    echo "Install them with: .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

cd "$project_root"

index_manifest="$project_root/dataset/index/manifest.json"
index_schema="$("$python_bin" - "$index_manifest" 2>/dev/null <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    print(int(json.loads(path.read_text(encoding="utf-8")).get("schema_version", 0)))
except (OSError, ValueError, TypeError):
    print(0)
PY
)"
if [[ "$index_schema" != "2" ]]; then
    echo "No chapter-aware textbook index was found; building a lexical index..."
    if ! "$python_bin" -m bangla_rag ingest --lexical-only; then
        echo "Could not build the textbook index." >&2
        exit 1
    fi
fi

# Never overwrite state for a launcher instance that is still alive.
if [[ -f "$state_file" ]]; then
    while IFS='=' read -r name process_id; do
        if [[ "$process_id" =~ ^[0-9]+$ ]] && kill -0 "$process_id" 2>/dev/null; then
            echo "Pathshongi is already launcher-managed ($name PID $process_id)."
            echo "Run ./Stop-Pathshongi.sh before starting another instance."
            exit 0
        fi
    done <"$state_file"
fi
: >"$state_file"

started_pids=()
cleanup_failed_start() {
    local process_id
    for process_id in "${started_pids[@]}"; do
        kill "$process_id" 2>/dev/null || true
    done
    rm -f "$state_file"
}
trap cleanup_failed_start ERR INT TERM

llama_server="${PATHSHONGI_LLAMA_SERVER:-$runtime_dir/llama.cpp/llama-server}"
answer_model="${PATHSHONGI_MODEL:-$runtime_dir/models/Qwen3-4B-Q4_K_M.gguf}"

if [[ ! -x "$llama_server" ]] && command -v llama-server >/dev/null 2>&1; then
    llama_server="$(command -v llama-server)"
fi

echo "Starting Pathshongi (পাঠসঙ্গী)..."
if [[ -x "$llama_server" ]]; then
    if ! port_is_open "$embedding_port"; then
        process_id="$(start_background embedding env HF_HOME="$runtime_dir/huggingface" "$llama_server" \
            --embd-gemma-default --alias local-embedding-model \
            --host 127.0.0.1 --port "$embedding_port" \
            -c 8192 --parallel 8 -ngl 0 --embedding --no-webui)"
        started_pids+=("$process_id")
    fi

    wait_for_port "$embedding_port" "Retrieval service" 900
    if [[ -f "$answer_model" ]]; then
        if ! port_is_open "$answer_port"; then
            process_id="$(start_background answer env HF_HOME="$runtime_dir/huggingface" "$llama_server" \
                -m "$answer_model" --alias local-model \
                --host 127.0.0.1 --port "$answer_port" \
                -c 8192 -ngl 0 --reasoning off --jinja --no-webui)"
            started_pids+=("$process_id")
        fi
        wait_for_port "$answer_port" "Answer and quiz service" 240
    else
        echo "  Qwen model not found; grounded answers and quizzes are unavailable."
        echo "  Expected model: $answer_model"
    fi
else
    echo "  llama-server not found; retrieval diagnostics remain available, but answers and quizzes are unavailable."
    echo "  Expected server: $llama_server"
    echo "  Install llama.cpp and Qwen: ./Install-Pathshongi-Model.sh"
fi

if port_is_open "$gui_port"; then
    echo "Port $gui_port is already in use; leaving that service untouched." >&2
    echo "Choose another port with: ./Launch-Pathshongi.sh --port 8001" >&2
    cleanup_failed_start
    trap - ERR INT TERM
    exit 1
fi

process_id="$(start_background gui "$python_bin" -m uvicorn bangla_rag.webapp:app \
    --host 127.0.0.1 --port "$gui_port")"
started_pids+=("$process_id")
wait_for_port "$gui_port" "Web app" 60

trap - ERR INT TERM
url="http://127.0.0.1:$gui_port"
echo
echo "Pathshongi is ready: $url"
echo "Logs: $log_dir"
echo "Stop it with: ./Stop-Pathshongi.sh"

if ((open_browser)); then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 &
    elif command -v gio >/dev/null 2>&1; then
        gio open "$url" >/dev/null 2>&1 &
    else
        echo "No browser opener was found; open $url manually."
    fi
fi
