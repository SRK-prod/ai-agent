# Installation (Windows)

The pipeline itself is cross-platform (PySide6, pynput, sounddevice/PortAudio,
faster-whisper, pyannote/torch all run natively on Windows). The two things
that differ from [`docs/installation.md`](installation.md) (macOS) are the
virtual-audio driver used for meeting capture and the shell used to run
things. Everything else -- Claude backend, knowledge base, voice enrollment --
is identical.

## 1. System prerequisites

- **Python 3.13** (python.org installer, or `winget install Python.Python.3.13`)
- **Docker Desktop** (for Qdrant + Redis) -- `winget install Docker.DockerDesktop`
- **Git**, and **PowerShell** (built in) or **Git Bash** if you'd rather run
  the existing `.sh`/Makefile flow under WSL2 -- see the WSL2 note below
  before choosing that route.
- **FFmpeg** -- `winget install Gyan.FFmpeg` (needed by `faster-whisper`)

`sounddevice` ships its own PortAudio binary wheel on Windows, so there's no
separate PortAudio install step like on macOS.

## 2. Route meeting audio through a virtual audio cable

macOS uses BlackHole for this; Windows has no equivalent built in, so install
one of:

- **[VB-Audio Virtual Cable](https://vb-audio.com/Cable/)** (free) --
  simplest option, one virtual input/output pair.
- **[VoiceMeeter](https://vb-audio.com/Voicemeeter/)** -- more setup, but
  lets you mix your own mic and the meeting audio into one device if you
  want both captured (the app already filters your own voice out via voice
  enrollment either way, so this is optional).

Steps with VB-Cable:

1. Install VB-Audio Virtual Cable and reboot if prompted.
2. In your meeting app's audio settings, set **Speaker/output** to
   **CABLE Input (VB-Audio Virtual Cable)**.
3. In Windows **Sound settings > Playback**, you can enable "Listen to this
   device" on **CABLE Output** if you still want to hear the meeting through
   your normal speakers/headphones while it's also being captured.
4. In `configs/settings.yaml`, set `audio.input_device` to a name that
   matches **CABLE Output** (that's the capture side of the virtual pair).

Run this to see what PortAudio can see once the virtual cable is installed:

```powershell
.venv\Scripts\python.exe -c "from meeting_copilot.audio.capture import list_input_devices; print(list_input_devices())"
```

**WSL2 note**: don't run this app under WSL2. WSL2 doesn't get native
passthrough to Windows audio devices, so it can't see the virtual cable's
capture device -- run natively on Windows instead.

## 3. Python environment

```powershell
cd meeting-copilot
py -3.13 -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\python.exe -m playwright install chromium
```

`mlx-whisper` (the Apple Silicon STT backend) is skipped automatically on
Windows -- the dependency is marked `sys_platform == 'darwin' and
platform_machine == 'arm64'` in `pyproject.toml`, so `pip install` never even
attempts it. Leave `stt.backend: faster-whisper` in `configs/settings.yaml`
(that's already the default) -- it runs on CPU, or on an NVIDIA GPU if you
have CUDA-enabled `torch` installed.

## 4. Backing services

```powershell
docker compose up -d qdrant redis
```

## 5. Credentials

```powershell
copy .env.example .env
```

Same credentials as macOS -- `ANTHROPIC_API_KEY` (or `claude login` for the
`cli` backend), and `HF_TOKEN` for the pyannote speaker-embedding model. See
[`docs/installation.md`](installation.md#5-credentials) for details, they're
identical here.

## 6. Pre-download models (optional but recommended)

```powershell
.venv\Scripts\python.exe scripts\download_models.py
```

## 7. Enroll your voice

```powershell
.venv\Scripts\python.exe scripts\enroll_voice.py
```

## 8. Build the knowledge base

Edit `configs/topics.yaml`, then:

```powershell
.venv\Scripts\python.exe scripts\ingest_knowledge.py --topics configs/topics.yaml
```

## 9. Hotkeys

The default overlay hotkeys in `configs/settings.yaml` use `<cmd>+<shift>+...`,
which is macOS-only. On Windows, `<cmd>` maps to the Windows key, and
Win+Shift+* combos are heavily reserved by the OS (window snapping, virtual
desktops) -- they often won't register reliably for a third-party global
hotkey. Change the `overlay.hotkeys` block in `configs/settings.yaml` to
something Windows doesn't reserve, e.g.:

```yaml
overlay:
  hotkeys:
    hide: "<ctrl>+<alt>+h"
    pin: "<ctrl>+<alt>+p"
    expand: "<ctrl>+<alt>+e"
    copy_answer: "<ctrl>+<alt>+c"
```

The app also logs a startup warning if it detects a `<cmd>` hotkey while
running on a non-macOS platform, as a reminder.

## 10. Run

```powershell
scripts\start.ps1
```

This brings up Qdrant/Redis, starts the FastAPI backend (auto-starts the live
pipeline) and launches the overlay, both detached so the terminal stays free.
Use `scripts\stop.ps1` to stop them (this leaves the Docker containers
running -- `docker compose down` to stop those too).

The first time Windows launches the overlay, it may prompt for microphone
access (**Settings > Privacy & security > Microphone**) -- grant it, since
`sounddevice` needs it even though the actual voice being captured is the
meeting app's output routed through the virtual cable, not your physical mic.

Note: `--loop uvloop` is dropped from the Windows run scripts -- `uvloop` is
Unix-only and is already excluded from the Windows dependency set in
`pyproject.toml`; uvicorn falls back to its default asyncio loop, which is
fine for this workload.
