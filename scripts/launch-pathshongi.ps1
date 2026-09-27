param(
    [int]$GuiPort = 8000,
    [int]$AnswerPort = 8080,
    [int]$EmbeddingPort = 8081,
    [string]$Server = $env:PATHSHONGI_LLAMA_SERVER,
    [string]$Model
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not $Server) {
    $Server = Join-Path $projectRoot ".runtime\llama.cpp\llama-server.exe"
}
if (-not $Model) {
    $Model = Join-Path $projectRoot ".runtime\models\Qwen3-4B-Q4_K_M.gguf"
}
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
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found: $python"
}
& $python -c "import fastapi, numpy, torch, transformers, uvicorn"
if ($LASTEXITCODE -ne 0) {
    throw "Python dependencies are missing. Run: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
$saved = @{}
$indexManifest = Join-Path $projectRoot "dataset\index\manifest.json"
$indexSchema = 0
if (Test-Path -LiteralPath $indexManifest) {
    try { $indexSchema = [int](Get-Content -Raw -LiteralPath $indexManifest | ConvertFrom-Json).schema_version }
    catch { $indexSchema = 0 }
}
if ($indexSchema -ne 2) {
    Write-Host "No chapter-aware textbook index was found; building a lexical index..."
    & $python -m bangla_rag ingest --lexical-only
    if ($LASTEXITCODE -ne 0) { throw "Could not build the textbook index." }
}

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

    if (Test-Path -LiteralPath $Server) {
        if (-not (Test-LocalPort $EmbeddingPort)) {
            $process = Start-HiddenProcess $Server @(
                "--embd-gemma-default", "--alias", "local-embedding-model",
                "--host", "127.0.0.1", "--port", "$EmbeddingPort",
                "-c", "8192", "--parallel", "8", "-ngl", "0", "--embedding", "--no-webui"
            ) "embedding"
            $started.Add($process)
            $saved["embedding"] = $process.Id
        }
        Wait-LocalPort $EmbeddingPort "Retrieval service" 180

        if (Test-Path -LiteralPath $Model) {
            if (-not (Test-LocalPort $AnswerPort)) {
                $quotedModel = '"' + $Model + '"'
                $process = Start-HiddenProcess $Server @(
                    "-m", $quotedModel, "--alias", "local-model",
                    "--host", "127.0.0.1", "--port", "$AnswerPort",
                    "-c", "8192", "-ngl", "0", "--reasoning", "off", "--jinja", "--no-webui"
                ) "answer"
                $started.Add($process)
                $saved["answer"] = $process.Id
            }
            Wait-LocalPort $AnswerPort "Answer and quiz service" 240
        }
        else {
            Write-Host "  Qwen model not found; grounded answers and quizzes are unavailable."
        }
    }
    else {
        Write-Host "  llama-server not found; retrieval diagnostics remain available, but answers and quizzes are unavailable."
    }

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
