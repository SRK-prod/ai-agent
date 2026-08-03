# Troubleshooting

**`MissingCredentialError: HF_TOKEN is not set...`**
Get a token at https://huggingface.co/settings/tokens and accept the
license on https://huggingface.co/pyannote/embedding. Needed for speaker
enrollment/identification.

**`claude CLI failed (exit ...)` / CLI backend errors**
Run `claude login` once, interactively, to establish your Pro/Max
subscription session -- the CLI backend (`llm.backend: cli`, the default)
relies on that session, not an API key. If it still fails, run the same
`claude -p "..."` command manually in a terminal to see the raw error.

**Answers take 1-3+ seconds to appear**
Expected with the default `llm.backend: cli` -- see
[performance.md](performance.md). Switch to `llm.backend: api` +
`ANTHROPIC_API_KEY` if you need the spec's original <400ms LLM target.

**Global hotkeys (Hide/Pin/Expand/Copy) don't do anything**
macOS needs to grant `pynput`'s listener process **Accessibility** and/or
**Input Monitoring** permission: **System Settings > Privacy & Security >
Accessibility / Input Monitoring**, add your terminal or the packaged
`.app`. You may need to quit and relaunch after granting.

**No audio / wrong participants captured**
Confirm `audio.input_device` in `configs/settings.yaml` matches a real
CoreAudio device name -- run
`python -c "from meeting_copilot.audio.capture import list_input_devices; print(list_input_devices())"`.
If BlackHole isn't listed, it isn't installed/approved yet -- see
[installation.md](installation.md) steps 1-2.

**My own voice is still being transcribed (or I'm being incorrectly ignored)**
Re-run `make enroll` in a quiet room with your normal meeting mic setup.
Tune `speaker.ignore_similarity_threshold` in `configs/settings.yaml` --
lower it if you're not being ignored enough, raise it if other people are
being mistaken for you.

**Qdrant/Redis connection errors**
`make services` (or `docker compose up -d qdrant redis`) must be running.
Check `docker ps` and `QDRANT_URL`/`REDIS_URL` in `.env`.

**`AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'`**
`torch`/`torchaudio` are pinned to `2.7.0` in `pyproject.toml` for exactly
this reason -- newer `torchaudio` (2.8+) removed `AudioMetaData`, which
`pyannote.audio` 3.4.0 still imports. If you see this, something in your
environment overrode the pin (`pip install -U torch torchaudio`, etc.) --
reinstall with `pip install -e ".[dev]"` to restore it.

**`TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'`**
`huggingface_hub` is pinned to `<1.0` in `pyproject.toml` for exactly this
reason -- `huggingface_hub` 1.0 removed the `use_auth_token` kwarg that
`pyannote.audio` 3.4.0 still passes internally. Reinstall with
`pip install -e ".[dev]"` if something overrode the pin.

**`GatedRepoError: ... Access to model pyannote/embedding is restricted`**
Your token is valid but your HuggingFace account hasn't clicked through the
model's access terms yet. Visit
https://huggingface.co/pyannote/embedding while logged in and click "Agree
and access repository" (it's an instant click-through, not a real review) --
this can't be done via API/token alone.

**`_pickle.UnpicklingError: Weights only load failed ... Unsupported global: pytorch_lightning.callbacks.early_stopping.EarlyStopping`**
PyTorch 2.6+ defaults `torch.load` to `weights_only=True`, which rejects the
pickled `pytorch_lightning` objects inside pyannote's official checkpoints.
`speaker/diarization.py::_trust_pyannote_checkpoint()` scopes a
`weights_only=False` override to just that load (official, signed pyannote
HuggingFace models, not arbitrary files). You may also see loud
"Bad things might happen unless you revert torch/pyannote.audio to
1.x/0.x..." warnings when this loads -- that's pyannote's stock warning for
its 2021-era checkpoint running on a modern torch/pyannote.audio, not an
error; the model still loads and works.

**`make ingest` says a topic has no sources/chunks**
`configs/topics.yaml` entries with empty `sources: []` and no
`research_brief` have nothing to embed -- either point `sources` at local
files/directories or add a `research_brief` so Claude generates notes for
that topic.
