# Stop backend + overlay (leaves Qdrant/Redis containers running --
# `docker compose down` to stop those too). Windows equivalent of `make stop`.
$ErrorActionPreference = "SilentlyContinue"

Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'uvicorn.exe'" |
    Where-Object { $_.CommandLine -like "*meeting_copilot*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Write-Host "stopped (docker services still up; 'docker compose down' to stop them too)"
