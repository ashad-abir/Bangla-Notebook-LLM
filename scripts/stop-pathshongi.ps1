$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$statePath = Join-Path $projectRoot ".runtime\pathshongi-processes.json"

if (-not (Test-Path -LiteralPath $statePath)) {
    Write-Host "No launcher-managed Pathshongi services were found."
    exit 0
}

$state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
$stopped = 0
foreach ($name in @("gui", "answer", "embedding")) {
    $processId = $state.$name
    if (-not $processId) { continue }
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    $allowed = if ($name -eq "gui") { @("python", "pythonw") } else { @("llama-server") }
    if ($process.ProcessName -notin $allowed) {
        Write-Warning "Skipped PID $processId because it is now $($process.ProcessName), not a Pathshongi $name process."
        continue
    }
    Stop-Process -Id $processId -Force
    $stopped += 1
    Write-Host "Stopped $name service."
}

Remove-Item -LiteralPath $statePath -Force
Write-Host "Pathshongi stopped ($stopped managed process(es))." -ForegroundColor Green
