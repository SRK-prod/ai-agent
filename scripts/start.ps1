# Detached start: backend + overlay in the background, terminal stays free.
# Windows equivalent of `make start` (see Makefile) -- no --loop uvloop, since
# uvloop is Unix-only and isn't installed on Windows (see pyproject.toml).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Make PATH (Docker CLI) and the HF model cache location explicit so the backend,
# however it's launched, finds the models pre-downloaded on D: instead of re-fetching
# to %USERPROFILE%\.cache (and failing on the gated pyannote model).
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User") + ";C:\Program Files\Docker\Docker\resources\bin"
if (-not $env:HF_HOME) {
    $u = [Environment]::GetEnvironmentVariable("HF_HOME","User")
    $env:HF_HOME = if ($u) { $u } else { "D:\meeting-copilot\hf-cache" }
}

Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'uvicorn.exe'" |
    Where-Object { $_.CommandLine -like "*meeting_copilot*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# `docker compose` writes its normal progress ("Container ... Running") to STDERR even when
# it succeeds. Windows PowerShell 5.1 turns any stderr from a native exe into a
# NativeCommandError, which $ErrorActionPreference="Stop" then promotes to a terminating
# error -- so this line aborted the script before the backend was ever started, while
# printing something that looked like a Docker failure. Redirecting with 2>&1 does NOT help
# (5.1 still wraps each stderr line in an ErrorRecord); the fix is to drop the Stop
# preference around the call and judge the process exit code, which is the real signal.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
docker compose up -d qdrant redis
$composeExit = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($composeExit -ne 0) {
    throw "docker compose failed (exit $composeExit) -- is Docker Desktop running?"
}

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

# Raise the pipeline above normal priority. STT is CPU-bound and this machine has only two
# cores, which the meeting app (Chrome/Teams), Docker and the browser happily saturate --
# measured live 2026-09-01: with the CPU pegged at 99% by other apps, a 2.2s utterance took
# 33s to decode instead of the ~0.4s it takes with headroom. At High the decoder wins the
# scheduler against background tabs rather than time-slicing behind them.
foreach ($p in (Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'uvicorn.exe'" |
                Where-Object { $_.CommandLine -like "*meeting_copilot*" })) {
    try { (Get-Process -Id $p.ProcessId).PriorityClass = "High" } catch {}
}

& "$PSScriptRoot\status.ps1"
