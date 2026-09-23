param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found at $python. Follow README.md first."
}

Set-Location -LiteralPath $projectRoot
Write-Host "Starting পাঠসঙ্গী at http://127.0.0.1:$Port"
Write-Host "Keep this terminal open. Press Ctrl+C to stop the GUI."
& $python -m uvicorn bangla_rag.webapp:app --host 127.0.0.1 --port $Port
