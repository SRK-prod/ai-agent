# Installation

## 1. System prerequisites

```bash
brew install ffmpeg portaudio blackhole-2ch
```

- `ffmpeg` / `portaudio` are build/runtime deps for `faster-whisper` and
  `sounddevice`.
- `blackhole-2ch` installs a virtual audio driver. **This is a system-level
  change** -- macOS will prompt you to approve a new audio driver in
  **System Settings > Privacy & Security** (you may need to restart Core
  Audio, or log out/in, for it to appear as an input/output device).

## 2. Route meeting audio through BlackHole

You need one audio stream that contains both your mic and the meeting app's
output (Zoom/Meet/Teams/etc.), so the pipeline hears every participant.

1. Open **Audio MIDI Setup** (Spotlight: "Audio MIDI Setup").
2. Click **+** > **Create Multi-Output Device**. Check both your normal
   output (e.g. built-in speakers/headphones) and **BlackHole 2ch**, so you
   still hear the meeting yourself.
3. In your meeting app's audio settings, set:
   - **Speaker/output** → the Multi-Output Device you just created.
   - Keep your normal **microphone/input** as-is.
4. In `configs/settings.yaml`, set `audio.input_device` to a name that
   matches **BlackHole 2ch** (for the meeting's side) -- capturing your own
   mic simultaneously requires either a second capture stream or an
   aggregate device; see `audio/capture.py::list_input_devices()` to see
   what CoreAudio exposes and adjust the aggregate/multi-output setup to
   route both into one device if you want your own mic captured too (it's
   filtered out by voice enrollment either way).

Run this to see what CoreAudio can see once BlackHole is installed:

```bash
python -c "from meeting_copilot.audio.capture import list_input_devices; print(list_input_devices())"
```

## 3. Python environment

```bash
cd meeting-copilot
make setup   # creates .venv with python3.13, installs the package + dev deps, installs Playwright's Chromium
```

## 4. Backing services

```bash
make services   # docker compose up -d qdrant redis
```

## 5. Credentials

```bash
cp .env.example .env
```

- **Claude (LLM)**: default backend is `api` (fast, model
  `claude-haiku-4-5-20251001`) -- needs `ANTHROPIC_API_KEY` from
  console.anthropic.com (separate account/billing from a Claude Pro/Max
  subscription; see console.anthropic.com > Settings > API Keys). Switch
  `llm.backend` to `cli` in `configs/settings.yaml` instead if you'd rather
  use your Claude Pro/Max subscription (via `claude login`) and accept
  slower answers (measured 1-3s to ~106s per call) to avoid API billing.
- **Embeddings**: run locally via `sentence-transformers` (no API key, no
  billing) -- the model downloads once (~80MB) on first use or via
  `make download-models`.
- **HuggingFace (`HF_TOKEN`)**: needed for the pyannote.audio speaker
  embedding model. Get a token at https://huggingface.co/settings/tokens,
  and accept the license on
  https://huggingface.co/pyannote/embedding (and
  https://huggingface.co/pyannote/speaker-diarization-3.1 if you later wire
  in full diarization).

## 6. Pre-download models (optional but recommended)

```bash
make download-models
```

Downloads Silero VAD, the configured Faster-Whisper model, and (if
`HF_TOKEN` is set) the pyannote embedding model, so the first real meeting
isn't slowed down by cold downloads.

## 7. Enroll your voice

```bash
make enroll
```

Records 30-60 seconds (default 45s) and stores a voice embedding in
`data/speaker_enrollment.sqlite3`. Re-run any time your voice/mic setup
changes meaningfully.

## 8. Build the knowledge base

Edit `configs/topics.yaml` with the topics you want covered, then:

```bash
make ingest
```

## 9. Run

```bash
make run
```

This brings up Qdrant/Redis, starts the FastAPI backend (which auto-starts
the live pipeline), and launches the overlay. The first time macOS launches
the overlay, it will ask for **Accessibility**/**Input Monitoring**
permission (needed by `pynput` for global hotkeys) and **Microphone**
permission (needed for audio capture) -- grant both in **System Settings >
Privacy & Security**.
