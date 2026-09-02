# Enable "Listen to this device" on CABLE Output, playing through the Realtek speakers.
#
# WHY THIS EXISTS: with the virtual-cable setup the meeting app's audio is sent into
# CABLE Input so the copilot can transcribe it -- which means it no longer reaches your
# speakers, and you cannot hear the interviewer. Windows' "Listen to this device" monitor
# is what plays the cable's audio back out to real hardware so BOTH happen at once.
#
# It is normally a checkbox in Sound > Recording > CABLE Output > Properties > Listen.
# This script writes the same setting, for when that dialog is awkward to drive.
#
# MUST BE RUN AS ADMINISTRATOR (the setting lives under HKLM).
#   Right-click this file > "Run with PowerShell" will NOT be enough.
#   Instead: Start > type "powershell" > right-click Windows PowerShell > Run as
#   administrator > then run:
#       & "D:\meeting-copilot\ai-agent\scripts\enable_cable_listen.ps1"

$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "NOT RUNNING AS ADMINISTRATOR." -ForegroundColor Red
    Write-Host "Open PowerShell as administrator and run this script again:"
    Write-Host '    & "D:\meeting-copilot\ai-agent\scripts\enable_cable_listen.ps1"'
    exit 1
}

# Resolve the devices by name rather than hardcoding GUIDs, so this still works after a
# driver reinstall (which regenerates them).
function Get-MMDevice([string]$Kind, [string]$Match) {
    $base = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\$Kind"
    Get-ChildItem $base -ErrorAction SilentlyContinue | ForEach-Object {
        $props = Join-Path $_.PSPath "Properties"
        $name = (Get-ItemProperty -Path $props -ErrorAction SilentlyContinue).
                "{a45c254e-df1c-4efd-8020-67d146a850e0},2"
        $state = (Get-ItemProperty -Path $_.PSPath -ErrorAction SilentlyContinue).DeviceState
        if ($name -like $Match -and $state -eq 1) {
            [pscustomobject]@{ Guid = $_.PSChildName; Name = $name; Path = $_.PSPath }
        }
    } | Select-Object -First 1
}

$cable = Get-MMDevice -Kind "Capture" -Match "CABLE Output*"
$speaker = Get-MMDevice -Kind "Render" -Match "Speaker*"

if (-not $cable) { throw "No active 'CABLE Output' recording device -- is VB-Cable installed?" }
if (-not $speaker) { throw "No active 'Speaker' playback device found." }

Write-Host "listen source : $($cable.Name)   $($cable.Guid)"
Write-Host "play back to  : $($speaker.Name) $($speaker.Guid)"

$props = Join-Path $cable.Path "Properties"
# {24dbb0fc-...},1 = listen enabled (DWORD). ,0 = target render device (string GUID).
Set-ItemProperty -Path $props -Name "{24dbb0fc-9311-4b3d-9cf0-18ff155639d4},1" -Value 1 -Type DWord
Set-ItemProperty -Path $props -Name "{24dbb0fc-9311-4b3d-9cf0-18ff155639d4},0" -Value $speaker.Guid -Type String

Write-Host "`nSetting written. Restarting Windows Audio so it takes effect..." -ForegroundColor Yellow
Write-Host "(all sound will cut out for a couple of seconds)"
Restart-Service audiosrv -Force
Start-Sleep -Seconds 3

Write-Host "`nDONE." -ForegroundColor Green
Write-Host "You should now hear anything sent to CABLE Input through your speakers."
Write-Host "If you hear an echo, the playback target is wrong -- it must be the real"
Write-Host "speakers, never CABLE Input."
