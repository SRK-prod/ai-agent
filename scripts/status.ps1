# Windows equivalent of `make status`.
$ErrorActionPreference = "SilentlyContinue"
Set-Location (Join-Path $PSScriptRoot "..")

$backendPort = if ($env:MEETING_COPILOT_PORT) { $env:MEETING_COPILOT_PORT } else { "8765" }
$backendHost = if ($env:MEETING_COPILOT_HOST) { $env:MEETING_COPILOT_HOST } else { "127.0.0.1" }

try {
    $resp = Invoke-RestMethod -Uri "http://${backendHost}:${backendPort}/health" -TimeoutSec 2
    Write-Host $resp
} catch {
    Write-Host "backend: not running"
}

$overlay = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*meeting_copilot.desktop.app*" }
if ($overlay) { Write-Host "overlay: running" } else { Write-Host "overlay: not running" }
