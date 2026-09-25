#!/usr/bin/env bash

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$project_root/.runtime"
download_dir="$runtime_dir/downloads"
llama_dir="$runtime_dir/llama.cpp"
model_dir="$runtime_dir/models"
model_name="Qwen3-4B-Q4_K_M.gguf"
model_url="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/$model_name"
release_api="https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"

for command_name in curl tar; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command is missing: $command_name" >&2
        exit 1
    fi
done

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This installer currently supports Linux x86-64 only." >&2
    exit 1
fi

mkdir -p "$download_dir" "$llama_dir" "$model_dir"

echo "Resolving the latest official llama.cpp Linux x86-64 release..."
release_json="$download_dir/llama-release.json"
curl --fail --location --retry 5 --retry-delay 2 \
    "$release_api" --output "$release_json"

release_details_text="$("$project_root/.venv/bin/python" - "$release_json" <<'PY'
import json
import re
import sys
from pathlib import Path

releases = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for release in releases:
    matches = [
        asset for asset in release.get("assets", [])
        if re.search(
            r"^llama-.*-bin-ubuntu-x64\.tar\.gz$",
            asset.get("name", ""),
            re.IGNORECASE,
        )
    ]
    if len(matches) == 1:
        print(release.get("tag_name", "unknown"))
        print(matches[0]["name"])
        print(matches[0]["browser_download_url"])
        break
else:
    raise SystemExit("Could not identify a CPU Ubuntu x64 archive in recent releases")
PY
)"
readarray -t release_details <<<"$release_details_text"

release_tag="${release_details[0]}"
archive_name="${release_details[1]}"
archive_url="${release_details[2]}"
archive_path="$download_dir/$archive_name"

echo "Downloading llama.cpp $release_tag..."
curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    "$archive_url" --output "$archive_path"

extract_dir="$(mktemp -d "$runtime_dir/llama-extract.XXXXXX")"
cleanup() {
    rm -rf -- "$extract_dir"
}
trap cleanup EXIT
tar -xzf "$archive_path" -C "$extract_dir"

server_path="$(find "$extract_dir" -type f -name llama-server -print -quit)"
if [[ -z "$server_path" ]]; then
    echo "The llama.cpp archive did not contain llama-server." >&2
    exit 1
fi

rm -rf -- "$llama_dir"
mkdir -p "$llama_dir"
cp -a -- "$(dirname -- "$server_path")/." "$llama_dir/"
chmod +x "$llama_dir/llama-server"

model_path="$model_dir/$model_name"
partial_model="$model_path.part"
if [[ ! -f "$model_path" ]]; then
    echo "Downloading the official Qwen3-4B Q4_K_M model (about 2.5 GB)..."
    curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
        "$model_url" --output "$partial_model"
    if [[ "$(stat -c '%s' "$partial_model")" -lt 1000000000 ]]; then
        echo "The downloaded model is unexpectedly small; leaving it as $partial_model." >&2
        exit 1
    fi
    mv -- "$partial_model" "$model_path"
else
    echo "Model already exists: $model_path"
fi

echo
echo "Local model runtime installed successfully."
echo "llama-server: $llama_dir/llama-server"
echo "answer model: $model_path"
echo "Restart Pathshongi with:"
echo "  ./Stop-Pathshongi.sh"
echo "  ./Launch-Pathshongi.sh"
