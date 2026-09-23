param(
    [string]$Server = $env:PATHSHONGI_LLAMA_SERVER,
    [string]$Model = "$PSScriptRoot\..\.runtime\models\Qwen3-4B-Q4_K_M.gguf",
    [int]$EmbeddingPort = 8081,
    [int]$AnswerPort = 8080
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
if (-not $Server) {
    $Server = Join-Path $projectRoot ".runtime\llama.cpp\llama-server.exe"
}
$resolvedServer = (Resolve-Path -LiteralPath $Server).Path
$resolvedModel = (Resolve-Path -LiteralPath $Model).Path

Write-Host "Starting the local Qwen3-4B server on http://127.0.0.1:8080"
Write-Host "Starting the local EmbeddingGemma server on http://127.0.0.1:$EmbeddingPort"
Write-Host "Keep this terminal open while using question answering or quizzes."

$embeddingArguments = @(
    "--embd-gemma-default",
    "--alias", "local-embedding-model",
    "--host", "127.0.0.1",
    "--port", "$EmbeddingPort",
    "-c", "8192",
    "--parallel", "8",
    "-ngl", "0",
    "--embedding",
    "--no-webui"
)
$embeddingProcess = Start-Process -FilePath $resolvedServer -ArgumentList $embeddingArguments -WindowStyle Hidden -PassThru

try {
    & $resolvedServer `
        -m $resolvedModel `
        --alias local-model `
        --host 127.0.0.1 `
        --port $AnswerPort `
        -c 8192 `
        -ngl 0 `
        --reasoning off `
        --jinja `
        --no-webui
}
finally {
    if ($embeddingProcess -and -not $embeddingProcess.HasExited) {
        Stop-Process -Id $embeddingProcess.Id
    }
}
