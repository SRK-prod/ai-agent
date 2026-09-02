# One-shot setup for a fresh Windows machine.
#
# WHAT THIS AUTOMATES: system tools, the clone, the venv, dependencies, model
# downloads, and starting the stack.
#
# WHAT YOU STILL DO BY HAND (the script stops and tells you):
#   - accept UAC prompts for each winget install
#   - reboot after Docker + WSL2 (the script detects it needs one and stops)
#   - install VB-Audio Virtual Cable (a driver -- winget cannot do it) + reboot
#   - drop in .env and data\ from the transfer bundle
#   - set the audio routing (Windows + the meeting app)
#
# RUN IT:  right-click > Run with PowerShell  is NOT enough (execution policy).
#   Open PowerShell, then:
#       powershell -ExecutionPolicy Bypass -File C:\path\to\bootstrap-new-machine.ps1
#   Re-run it after each reboot -- it is idempotent and picks up where it stopped.

$ErrorActionPreference = "Stop"
$Root  = "D:\meeting-copilot"
$Repo  = "$Root\ai-agent"
$Repo_url = "https://github.com/SRK-prod/ai-agent.git"
$Branch = "feature/windows-support"

function Step($n) { Write-Host "`n=== $n ===" -ForegroundColor Cyan }
function Have($cmd) { $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue) }

# --- 0. drive check ---------------------------------------------------------
Step "0. Prerequisites"
$d = Get-PSDrive D -ErrorAction SilentlyContinue
if (-not $d) { throw "No D: drive. This build needs ~20GB on a second drive -- edit `$Root at the top of this script if yours is elsewhere." }
$freeGB = [math]::Round($d.Free / 1GB, 1)
Write-Host "D: has $freeGB GB free"
if ($freeGB -lt 18) { Write-Warning "Under 18GB free -- the install may not fit. Clear space or point `$Root at a bigger drive." }

# --- 1. system tools -----------------------------------------------------------
Step "1. System tools (winget -- accept each UAC prompt)"
if (-not (Have winget)) { throw "winget not found. Install 'App Installer' from the Microsoft Store, then re-run." }

if (-not (Have py))     { winget install --id Python.Python.3.13   -e --source winget --accept-package-agreements --accept-source-agreements }
if (-not (Have ffmpeg)) { winget install --id Gyan.FFmpeg          -e --source winget --accept-package-agreements --accept-source-agreements }
$dockerExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
if (-not (Test-Path $dockerExe)) {
    winget install --id Docker.DockerDesktop -e --source winget --accept-package-agreements --accept-source-agreements
}

# Docker's install enables WSL2 / Virtual Machine Platform, which need a reboot.
$rebootPending = Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"
if ($rebootPending) {
    Write-Host "`nA REBOOT IS PENDING (Docker enabled Windows features)." -ForegroundColor Yellow
    Write-Host "Reboot now, then run this script again. It will continue from here."
    exit 0
}

# --- 2. caches on the big drive ---------------------------------------------
Step "2. Redirect caches to D:"
[Environment]::SetEnvironmentVariable("HF_HOME",       "$Root\hf-cache",  "User")
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", "$Root\pip-cache", "User")
$env:HF_HOME       = "$Root\hf-cache"
$env:PIP_CACHE_DIR = "$Root\pip-cache"
New-Item -ItemType Directory -Force -Path "$Root\hf-cache","$Root\pip-cache","$Root\tmp" | Out-Null

# --- 3. clone --------------------------------------------------------------
Step "3. Clone the repo"
if (-not (Test-Path "$Repo\.git")) {
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
    git clone -c core.longpaths=true $Repo_url $Repo
    Push-Location $Repo; git checkout $Branch; Pop-Location
} else {
    Write-Host "already cloned at $Repo"
}
Set-Location $Repo

# --- 4. venv + deps -----------------------------------------------------------
Step "4. Python environment (10-20 min)"
if (-not (Test-Path "$Repo\.venv\Scripts\python.exe")) {
    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
}
$pipOk = $false
try { .\.venv\Scripts\python.exe -m pip check 2>&1 | Out-Null; $pipOk = ($LASTEXITCODE -eq 0) } catch {}
if (-not $pipOk) {
    # keep pip's multi-GB wheel staging off C:
    $env:TMP = "$Root\tmp"; $env:TEMP = "$Root\tmp"
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
} else {
    Write-Host "dependencies already installed"
}

# --- 5. secrets ----------------------------------------------------------------
Step "5. Secrets"
if (-not (Test-Path "$Repo\.env")) {
    Copy-Item .env.example .env
    Write-Host "`nSTOP: fill in D:\meeting-copilot\ai-agent\.env" -ForegroundColor Yellow
    Write-Host "  - copy it from the transfer bundle, OR"
    Write-Host "  - open it and paste ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, HF_TOKEN"
    Write-Host "Then run this script again."
    notepad .env
    exit 0
}
$envtext = Get-Content .env -Raw
foreach ($k in "ANTHROPIC_API_KEY","DEEPGRAM_API_KEY","HF_TOKEN") {
    if ($envtext -notmatch "(?m)^$k=.+") { Write-Warning "$k looks empty in .env -- fill it before a real interview" }
}

# --- 6. data -----------------------------------------------------------------
Step "6. Voice + profile"
if (-not (Test-Path "$Repo\data\speaker_enrollment.sqlite3")) {
    Write-Warning "data\ is missing. Copy it from the transfer bundle into $Repo\data\ ."
    Write-Warning "Without it you have no enrolled voice and no profile grounding."
}

# --- 7. VB-Cable ------------------------------------------------------------
Step "7. VB-Audio Virtual Cable"
$hasCable = $false
try {
    .\.venv\Scripts\python.exe -c "import sounddevice as sd; import sys; sys.exit(0 if any('CABLE Output' in d['name'] for d in sd.query_devices()) else 1)"
    $hasCable = ($LASTEXITCODE -eq 0)
} catch {}
if (-not $hasCable) {
    Write-Host "`nSTOP: VB-Audio Virtual Cable is not installed (winget cannot install a driver)." -ForegroundColor Yellow
    Write-Host "  1. https://vb-audio.com/Cable/  -> download the driver pack"
    Write-Host "  2. extract, right-click VBCABLE_Setup_x64.exe -> Run as administrator -> Install Driver"
    Write-Host "  3. REBOOT"
    Write-Host "  4. run this script again"
    Start-Process "https://vb-audio.com/Cable/"
    exit 0
}
Write-Host "VB-Cable present"

# --- 8. models + containers ------------------------------------------------
Step "8. Models + containers"
if (-not (Test-Path "$env:HF_HOME\hub")) {
    .\.venv\Scripts\python.exe scripts\download_models.py
} else {
    Write-Host "models already downloaded"
}
if (-not (Test-Path $dockerExe)) { throw "Docker Desktop missing -- step 1 did not complete" }
if (-not (Get-Process "Docker Desktop" -ErrorAction SilentlyContinue)) {
    Write-Host "starting Docker Desktop..."
    Start-Process $dockerExe
}
$env:Path += ";C:\Program Files\Docker\Docker\resources\bin"
Write-Host "waiting for the Docker daemon..."
$up = $false
for ($i = 0; $i -lt 24; $i++) {
    docker info *> $null
    if ($LASTEXITCODE -eq 0) { $up = $true; break }
    Start-Sleep 10
}
if (-not $up) { throw "Docker daemon did not come up. Open Docker Desktop, accept its terms, wait for 'Engine running', then re-run." }
docker compose up -d qdrant redis

# --- 9. start -------------------------------------------------------------
Step "9. Start the stack"
& .\scripts\start.ps1
Start-Process -NoNewWindow -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-u","scripts\audio_monitor.py"

Write-Host "`n=== DONE ===" -ForegroundColor Green
Write-Host "Health:  Invoke-RestMethod http://127.0.0.1:8765/health"
Write-Host ""
Write-Host "Audio routing you still have to set by hand:"
Write-Host "  Windows Output = CABLE Input (VB-Audio Virtual Cable)"
Write-Host "  Windows Input  = your real Realtek mic"
Write-Host "  Meeting app Speaker = CABLE Input,  Mic = your real mic"
Write-Host "  Use HEADPHONES with audio_monitor.py or the interviewer hears an echo."
Write-Host ""
Write-Host "Verify before relying on it:"
Write-Host "  .\.venv\Scripts\python.exe -m pytest -m `"not slow and not e2e`" -q"
Write-Host "  .\.venv\Scripts\python.exe scripts\smoke_e2e.py `"How would you design a highly available EKS platform?`""
