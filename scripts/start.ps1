# Detached start: backend + overlay in the background, terminal stays free.
# Windows equivalent of `make start` (see Makefile) -- no --loop uvloop, since
# uvloop is Unix-only and isn't installed on Windows (see pyproject.toml).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'uvicorn.exe'" |
    Where-Object { $_.CommandLine -like "*meeting_copilot*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

docker compose up -d qdrant redis

New-Item -ItemType Directory -Force -Path logs | Out-Null

$backendPort = if ($env:MEETING_COPILOT_PORT) { $env:MEETING_COPILOT_PORT } else { "8765" }
$backendHost = if ($env:MEETING_COPILOT_HOST) { $env:MEETING_COPILOT_HOST } else { "127.0.0.1" }

Start-Process -NoNewWindow -RedirectStandardOutput "logs\backend.log" -RedirectStandardError "logs\backend.err.log" `
    -FilePath ".venv\Scripts\uvicorn.exe" `
    -ArgumentList "meeting_copilot.server.main:app", "--host", $backendHost, "--port", $backendPort

Write-Host "waiting for backend..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri "http://${backendHost}:${backendPort}/health" -TimeoutSec 2
        if ($resp -match "running") { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $ready) { Write-Warning "backend did not report healthy within 30s -- check logs\backend.log" }

Start-Process -NoNewWindow -RedirectStandardOutput "logs\overlay.log" -RedirectStandardError "logs\overlay.err.log" `
    -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList "-m", "meeting_copilot.desktop.app"

Start-Sleep -Seconds 2
& "$PSScriptRoot\status.ps1"
