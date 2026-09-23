param(
    [int]$GuiPort = 8000,
    [int]$AnswerPort = 8080,
    [int]$EmbeddingPort = 8081
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$server = "C:\Users\Ashad Bin Rashid\Desktop\Bangla-Notebook\.runtime\llama-b10516\llama-server.exe"
$model = Join-Path $projectRoot ".runtime\models\Qwen3-4B-Q4_K_M.gguf"
$runtimeDir = Join-Path $projectRoot ".runtime"
$logDir = Join-Path $runtimeDir "logs"
$statePath = Join-Path $runtimeDir "pathshongi-processes.json"

function Test-LocalPort([int]$Port) {
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync("127.0.0.1", $Port)
        return $task.Wait(350) -and $client.Connected
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Wait-LocalPort([int]$Port, [string]$Label, [int]$TimeoutSeconds) {
    Write-Host -NoNewline "  $Label"
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-LocalPort $Port) {
            Write-Host " ready" -ForegroundColor Green
            return
        }
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 1
    }
    Write-Host " failed" -ForegroundColor Red
    throw "$Label did not become ready on port $Port. Check $logDir for details."
}

function Start-HiddenProcess([string]$FilePath, [string[]]$Arguments, [string]$LogName) {
    $stdout = Join-Path $logDir "$LogName.out.log"
    $stderr = Join-Path $logDir "$LogName.error.log"
    return Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
foreach ($required in @($python, $server, $model)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file was not found: $required"
    }
}

$saved = @{}
if (Test-Path -LiteralPath $statePath) {
    try {
        $previous = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
        foreach ($name in @("embedding", "answer", "gui")) {
            if ($previous.$name) { $saved[$name] = [int]$previous.$name }
        }
    }
    catch { $saved = @{} }
}

$started = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()
try {
    Write-Host "Starting Pathshongi (পাঠসঙ্গী)..." -ForegroundColor Cyan

    if (-not (Test-LocalPort $EmbeddingPort)) {
        $process = Start-HiddenProcess $server @(
            "--embd-gemma-default", "--alias", "local-embedding-model",
            "--host", "127.0.0.1", "--port", "$EmbeddingPort",
            "-c", "8192", "--parallel", "8", "-ngl", "0", "--embedding", "--no-webui"
        ) "embedding"
        $started.Add($process)
        $saved["embedding"] = $process.Id
    }

    if (-not (Test-LocalPort $AnswerPort)) {
        $quotedModel = '"' + $model + '"'
        $process = Start-HiddenProcess $server @(
            "-m", $quotedModel, "--alias", "local-model",
            "--host", "127.0.0.1", "--port", "$AnswerPort",
            "-c", "8192", "-ngl", "0", "--reasoning", "off", "--jinja", "--no-webui"
        ) "answer"
        $started.Add($process)
        $saved["answer"] = $process.Id
    }

    Wait-LocalPort $EmbeddingPort "Retrieval service" 180
    Wait-LocalPort $AnswerPort "Answer service" 240

    if (-not (Test-LocalPort $GuiPort)) {
        $process = Start-HiddenProcess $python @(
            "-m", "uvicorn", "bangla_rag.webapp:app",
            "--host", "127.0.0.1", "--port", "$GuiPort"
        ) "gui"
        $started.Add($process)
        $saved["gui"] = $process.Id
    }
    Wait-LocalPort $GuiPort "Web app" 60

    [ordered]@{
        embedding = $saved["embedding"]
        answer = $saved["answer"]
        gui = $saved["gui"]
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8

    $url = "http://127.0.0.1:$GuiPort"
    Write-Host "`nPathshongi is ready: $url" -ForegroundColor Green
    Write-Host "Use Stop-Pathshongi.cmd when the demonstration is finished."
    Start-Process $url
}
catch {
    Write-Host "`nCould not start Pathshongi: $($_.Exception.Message)" -ForegroundColor Red
    foreach ($process in $started) {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
    }
    exit 1
}
