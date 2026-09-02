# Setting this up on a fresh Windows machine

Written 2026-09-01 from an actual end-to-end setup, including everything that went
wrong. Follow it in order -- several steps exist only because a later step fails
without them.

The end state: the meeting app's audio is routed through a virtual cable into the
copilot, which transcribes the far end, detects questions, and streams an answer to a
floating overlay in ~3 seconds, while you still hear the call.

---

## 0. What you need before starting

| Thing | Why | Where |
|---|---|---|
| `ANTHROPIC_API_KEY` | generates the answers | console.anthropic.com > API Keys |
| `DEEPGRAM_API_KEY` | transcription (see step 6 for why not local) | console.deepgram.com, free tier |
| `HF_TOKEN` | downloads the Silero VAD / embedding models | huggingface.co/settings/tokens |
| `data/` from the old machine | your enrolled voice + profile. **Cannot be recreated from git** | see step 7 |

**A second drive with ~20GB free.** The install is roughly 5GB of Python packages,
2GB of models, and 3-4GB of Docker. On the machine this was written on, C: had 11GB
free and that was not enough -- everything sizable goes on D:.

---

## 1. System tools

Run from an ordinary PowerShell; each triggers a UAC prompt.

    winget install --id Python.Python.3.13 -e --source winget
    winget install --id Gyan.FFmpeg        -e --source winget
    winget install --id Docker.DockerDesktop -e --source winget

Then **reboot** -- Docker Desktop enables Windows features (WSL2, Virtual Machine
Platform) that only take effect after a restart. Launch Docker Desktop once
afterwards and let it finish starting.

Verify in a *new* shell:

    py -3.13 --version      # 3.13.x
    ffmpeg -version
    docker compose version

### Keep Docker's data off C:

Before Docker's first successful start, point its disk image at the big drive:
Docker Desktop > Settings > Resources > Disk image location > `D:\DockerData`.

> If you write `%APPDATA%\Docker\settings-store.json` by hand instead, it **must be
> UTF-8 with no BOM**. PowerShell 5.1's `Set-Content -Encoding utf8` adds a BOM, and
> Docker's parser then dies on every startup with
> `invalid character` looking for beginning of value -- the backend crash-loops and
> the UI just says it is starting forever. Use
> `[System.IO.File]::WriteAllText($p, $json, (New-Object System.Text.UTF8Encoding($false)))`.

---

## 2. VB-Audio Virtual Cable

This is the BlackHole equivalent -- it carries the meeting app's output into
something the copilot can record from. It is not on winget.

1. Download from <https://vb-audio.com/Cable/>
2. Extract, right-click **`VBCABLE_Setup_x64.exe`** > **Run as administrator**
3. **Reboot**

After the reboot you should have `CABLE Input` (a playback device) and
`CABLE Output` (a recording device).

---

## 3. Redirect the caches, then get the code

    setx HF_HOME       "D:\meeting-copilot\hf-cache"
    setx PIP_CACHE_DIR "D:\meeting-copilot\pip-cache"

Open a **new** shell so those take effect, then:

    mkdir D:\meeting-copilot
    git clone -c core.longpaths=true https://github.com/SRK-prod/ai-agent.git D:\meeting-copilot\ai-agent
    cd D:\meeting-copilot\ai-agent
    git checkout feature/windows-support

> `core.longpaths=true` is not optional. Cloning into a deep path (e.g. under
> `AppData\Roaming\...`) fails partway with `Filename too long` because Windows caps
> paths at 260 characters. A short path plus this flag avoids it.

> Use the `feature/windows-support` branch, not `main`. It is the only ref with both
> the Windows scripts and the reliability work.

---

## 4. Python environment

    cd D:\meeting-copilot\ai-agent
    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    mkdir D:\meeting-copilot\tmp -Force
    $env:TMP="D:\meeting-copilot\tmp"; $env:TEMP="D:\meeting-copilot\tmp"
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

> **Set TMP/TEMP to the big drive for the install only.** pip stages wheel builds in
> `%TEMP%` on C:, and torch/PySide6 unpack to several GB. Set them back afterwards --
> leaving the whole system's temp dir pointed at a project folder is not something you
> want permanently.

> **The numpy pin.** `pyproject.toml` asks for `numpy>=1.26,<2.3`. It used to say
> `<2.0`, which has no cp313 wheel -- pip then tries to build numpy from source and
> fails with `Unknown compiler(s)` because there is no MSVC. If you ever see that,
> the fix is the pin, not installing a compiler.

Takes 10-20 minutes and lands ~5GB in `.venv`. Verify:

    .\.venv\Scripts\python.exe -m pip check
    .\.venv\Scripts\python.exe -c "import torch,numpy; print(torch.__version__, numpy.__version__)"

Expect `2.7.0+cpu 2.2.6`. Playwright is only for browser e2e tests -- skip it.

---

## 5. Secrets

    Copy-Item .env.example .env
    notepad .env

Fill in `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY` and `HF_TOKEN`. Leave the
`QDRANT_URL` / `REDIS_URL` / host / port defaults alone.

`.env` is gitignored and must stay that way. Committing it would put live keys in
GitHub history permanently -- and GitHub's secret scanning would very likely get the
Anthropic key auto-revoked, breaking the setup you were trying to preserve. Move it
between machines on a USB stick or private cloud folder instead.

---

## 6. Why transcription is in the cloud

`configs/settings.yaml` ships with `stt.backend: deepgram`. That is a deliberate
choice for a CPU-only laptop, measured on a 2-core i5-7300U:

| model | 2.5s clip | 12s clip | 28s clip |
|---|---|---|---|
| faster-whisper `base` (local) | 3.03s | 2.98s | 4.54s |
| Deepgram `nova-3` (cloud) | **0.56s** | **0.82s** | **0.81s** |

And local degraded to **33s** for the same clip when Chrome and the video call were
competing for the two cores -- a network call does not. Accuracy was also better:
local `base` turned "wastage" into "missed" and "unattached volumes" into "out of
time volumes", where Deepgram got both right.

**Trade-off:** raw meeting audio leaves the machine. To keep it local instead, set
`stt.backend: faster-whisper`; everything still works, roughly 5 seconds slower per
question and less accurate. On a machine with a real GPU, reconsider from scratch --
these numbers are specific to CPU-only.

---

## 7. Restore your voice and profile

`data/` is gitignored and **cannot be reconstructed** -- copy it from the old machine
(USB, OneDrive, anything):

    data/speaker_enrollment.sqlite3     your enrolled voice
    data/profile/                       your real background, used for grounding
    data/research/                      interview notes

If you genuinely lost it, re-enroll with
`.\.venv\Scripts\python.exe scripts\enroll_voice.py` (records ~45s). Note that
`speaker.enabled` is `false` in the shipped config -- see step 9 -- so enrollment
only matters if you turn diarization back on.

---

## 8. Models and containers

    .\.venv\Scripts\python.exe scripts\download_models.py
    docker compose up -d qdrant redis

~1.6GB into `D:\meeting-copilot\hf-cache`. Qdrant starts but is unused
(`retrieval.enabled: false`); Redis is a hard dependency for caching.

> pyannote's model does **not** honour `HF_HOME` -- it caches to
> `%USERPROFILE%\.cache\torch\pyannote\`. If you only see two models under
> `hf-cache\hub`, nothing is wrong.

---

## 9. Audio routing -- the part that actually breaks

Three settings, and they are easy to get subtly wrong.

**Windows** (taskbar speaker icon > Sound settings):

| | Set to |
|---|---|
| Output | `CABLE Input (VB-Audio Virtual Cable)` |
| Input | `Microphone Array (Realtek...)` -- **your real mic** |

> Setting the *Input* to `CABLE Output` is the trap. The meeting app then uses the
> cable as your microphone, so the interviewer hears their own audio echoed back and
> never hears you.

**The meeting app** (Zoom / Teams / Meet > audio settings):

| | Set to |
|---|---|
| Speaker | `CABLE Input (VB-Audio Virtual Cable)` |
| Microphone | your real mic |

**Hearing the call.** Once the app's audio goes into the cable it no longer reaches
your speakers. Windows has a "Listen to this device" checkbox for this
(Sound > Recording > CABLE Output > Properties > Listen), and if you can drive that
dialog, use it -- it is the cleanest answer.

If you cannot, run the userspace equivalent:

    .\.venv\Scripts\python.exe -u scripts\audio_monitor.py

It copies the cable's audio to your speakers (~21ms). Leave it running for the call.

> **Use headphones.** The monitor plays out of the *speakers* by default and a laptop
> mic will pick that up and feed it back into the call -- this produced a real echo
> that other participants complained about mid-interview. Headphones eliminate it, or
> pass `--target` to aim the monitor at a headset.

> `scripts\enable_cable_listen.ps1` writes the registry setting directly, but on the
> machine this was written on it failed even elevated: the MMDevices keys are owned by
> SYSTEM. Do not go taking ownership of system registry keys for this.

---

## 10. Run it

    .\scripts\start.ps1

Brings up the containers, the backend, and the overlay, and raises them to High
priority (STT is CPU-bound and loses the scheduler fight against browser tabs
otherwise). Then, in a separate window, the audio monitor from step 9.

`.\scripts\status.ps1` to check, `.\scripts\stop.ps1` to stop.

**Overlay controls:** drag to move, double-click to snap back to the corner,
`Ctrl+Alt+H` hide/show, `Ctrl+Alt+P` pin, `Ctrl+Alt+E` expand, `Ctrl+Alt+C` copy.

> If you share your **whole screen**, the interviewer sees the overlay. Share a
> specific window, or hide it while sharing.

---

## 11. Verify before you rely on it

    .\.venv\Scripts\python.exe -m pytest -m "not slow and not e2e" -q

Expect **117 passed**. Then a real round trip:

    .\.venv\Scripts\python.exe scripts\smoke_e2e.py "How would you design a highly available ECS platform?"

Synthesizes speech, runs it through STT, question detection and a real Claude call.
And through the actual audio path:

    .\.venv\Scripts\python.exe scripts\play_into_cable.py logs\sample_client.wav --seconds 12

Watch `logs\backend.err.log` for a `TTFA=` line. Healthy is **2.5-4 seconds**.

---

## 12. When something is wrong

| Symptom | Cause | Fix |
|---|---|---|
| Overlay silent, no transcripts | audio not reaching the cable | check `peak=` in `logs\backend.err.log`; all `0.0000` means the meeting app is not outputting to `CABLE Input`. Reload the call tab -- apps cache the device |
| You cannot hear the call | expected; the audio is in the cable | run `audio_monitor.py` (step 9) |
| Interviewer hears an echo | monitor playing to speakers, your mic picks it up | headphones |
| `AUDIO INPUT LOST` on the overlay | the audio service or device changed under the running capture | restart the backend |
| Answers take 30s+ | CPU contention with local STT | you are on `faster-whisper`; switch to `deepgram`, or close Chrome |
| First answer after a restart is slow | prompt cache is cold | normal, ~1-2s extra on question one only |
| `start.ps1` prints a Docker error and stops | PowerShell 5.1 turns native stderr into a terminating error | already fixed in the script; if it recurs, check `$ErrorActionPreference` around the compose call |
| Overlay asks the interviewer to clarify | a guard should catch this | it does -- `Answer sought clarification, forcing retry` in the log means it worked |

---

## What lives where

    D:\meeting-copilot\
      ai-agent\          the repo
        .venv\           ~5GB
        .env             SECRETS - never commit
        data\            voice + profile - never commit, cannot be regenerated
        logs\            backend.err.log is the one to read
      hf-cache\          ~1.6GB models   (HF_HOME)
      pip-cache\                          (PIP_CACHE_DIR)
    D:\DockerData\       Docker disk image
    %USERPROFILE%\.cache\torch\pyannote\  pyannote ignores HF_HOME

**Back up before wiping a machine:** `.env` and `data\`. Everything else is in git or
re-downloadable.
