#!/usr/bin/env bash

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
state_file="$project_root/.runtime/pathshongi-linux.pids"

if [[ ! -f "$state_file" ]]; then
    echo "No launcher-managed Pathshongi services were found."
    exit 0
fi

stopped=0
while IFS='=' read -r name process_id; do
    if [[ ! "$process_id" =~ ^[0-9]+$ ]] || ! kill -0 "$process_id" 2>/dev/null; then
        continue
    fi

    command_line="$(tr '\0' ' ' <"/proc/$process_id/cmdline" 2>/dev/null || true)"
    case "$name" in
        gui)
            expected="uvicorn bangla_rag.webapp:app"
            ;;
        answer|embedding)
            expected="llama-server"
            ;;
        *)
            echo "Skipped unknown process record: $name=$process_id" >&2
            continue
            ;;
    esac

    if [[ "$command_line" != *"$expected"* ]]; then
        echo "Skipped PID $process_id: it is no longer the recorded $name service." >&2
        continue
    fi

    kill "$process_id"
    ((stopped += 1))
    echo "Stopped $name service."
done <"$state_file"

rm -f "$state_file"
echo "Pathshongi stopped ($stopped managed process(es))."
