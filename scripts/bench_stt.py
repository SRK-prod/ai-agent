#!/usr/bin/env python3
"""Benchmark faster-whisper decode time vs accuracy on THIS machine, across model
sizes and beam widths, so the stt.* settings are chosen from measurement instead
of guesswork.

Uses Windows SAPI to synthesize a fixed technical question (same text every run,
so transcripts are directly comparable), then decodes it with each candidate
config and reports wall-clock decode time and the transcript produced.

    .venv\\Scripts\\python.exe scripts\\bench_stt.py
"""

from __future__ import annotations

import time
import wave

import numpy as np

# Deliberately loaded with the technical vocabulary that a weaker model mangles --
# that is exactly the failure mode we are trading against.
SENTENCE = (
    "How would you troubleshoot a Kubernetes pod that is OOMKilled and stuck in "
    "CrashLoopBackOff behind an ALB in EKS?"
)

CANDIDATES = [
    # (model_size, beam_size)
    ("small", 5),
    ("small", 1),
    ("base", 1),
    ("tiny", 1),
]


def synth_sapi(text: str, path: str) -> None:
    import win32com.client

    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
    fmt.Type = 22  # SAFT22kHz16BitMono
    stream.Format = fmt
    stream.Open(path, 3)
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    voice.AudioOutputStream = stream
    voice.Speak(text)
    stream.Close()


def load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        tgt = int(len(a) * 16000 / sr)
        a = np.interp(
            np.linspace(0, len(a), tgt, endpoint=False), np.arange(len(a)), a
        ).astype(np.float32)
    return a


def main() -> None:
    from faster_whisper import WhisperModel

    from meeting_copilot.config import get_config

    cfg = get_config().stt
    wav = "logs/bench_question.wav"
    synth_sapi(SENTENCE, wav)
    samples = load_wav_16k_mono(wav)
    dur = len(samples) / 16000
    print(f"audio: {dur:.1f}s\nreference: {SENTENCE}\n")
    print(f"{'model':<8} {'beam':<5} {'load_s':<8} {'decode_s':<9} {'xRT':<6} transcript")
    print("-" * 110)

    for model_size, beam in CANDIDATES:
        t = time.perf_counter()
        model = WhisperModel(model_size, device="cpu", compute_type=cfg.compute_type)
        load_s = time.perf_counter() - t

        # warm the model so the timed run reflects steady state, not first-call overhead
        list(model.transcribe(samples[: 16000 * 1], language="en", beam_size=beam)[0])

        t = time.perf_counter()
        segments, _ = model.transcribe(
            samples,
            language="en",
            beam_size=beam,
            condition_on_previous_text=True,
            vad_filter=False,
            initial_prompt=cfg.vocabulary_hint,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        decode_s = time.perf_counter() - t

        print(
            f"{model_size:<8} {beam:<5} {load_s:<8.1f} {decode_s:<9.1f} "
            f"{decode_s / dur:<6.2f} {text}"
        )
        del model


if __name__ == "__main__":
    main()
