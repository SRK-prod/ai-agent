#!/usr/bin/env python3
"""Benchmark faster-whisper *decode options* (not model size) on this machine.

bench_stt.py answered "which model"; this answers "which transcribe() flags".
The engine currently passes a ~530-token `initial_prompt` (the technical
vocabulary hint) and `condition_on_previous_text=True` on every call, and lets
timestamps be generated -- none of which have ever been measured for cost.

    .venv\\Scripts\\python.exe scripts\\bench_stt_opts.py
"""

from __future__ import annotations

import time
import wave

import numpy as np

SENTENCE = (
    "How would you troubleshoot a Kubernetes pod that is OOMKilled and stuck in "
    "CrashLoopBackOff behind an ALB in EKS?"
)


def synth_sapi(text: str, path: str) -> None:
    import win32com.client

    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
    fmt.Type = 22
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
    hint = cfg.vocabulary_hint
    # A drastically shorter hint: only the terms Whisper actually mangles phonetically.
    short_hint = (
        "Kubernetes, EKS, AKS, OOMKilled, CrashLoopBackOff, ALB, NLB, Terraform, "
        "IAM, Lambda, DynamoDB, Aurora, Bedrock, Agentic AI, RAG, CI/CD, GitOps, SLO"
    )

    wav = "logs/bench_opts.wav"
    synth_sapi(SENTENCE, wav)
    samples = load_wav_16k_mono(wav)
    dur = len(samples) / 16000

    model = WhisperModel(cfg.model_size, device="cpu", compute_type=cfg.compute_type,
                         cpu_threads=cfg.cpu_threads)
    # warm
    list(model.transcribe(samples, language="en", beam_size=1)[0])

    variants = [
        ("current (full hint, cond=True, ts)", {"initial_prompt": hint, "condition_on_previous_text": True, "without_timestamps": False}),
        ("without_timestamps=True", {"initial_prompt": hint, "condition_on_previous_text": True, "without_timestamps": True}),
        ("cond=False", {"initial_prompt": hint, "condition_on_previous_text": False, "without_timestamps": True}),
        ("short hint", {"initial_prompt": short_hint, "condition_on_previous_text": False, "without_timestamps": True}),
        ("no hint", {"initial_prompt": None, "condition_on_previous_text": False, "without_timestamps": True}),
    ]

    print(f"audio {dur:.1f}s | model={cfg.model_size} beam={cfg.beam_size}")
    print(f"full hint = {len(hint)} chars, short hint = {len(short_hint)} chars\n")
    print(f"{'variant':<38} {'sec':>6}  transcript")
    print("-" * 120)
    for name, kw in variants:
        # 3 runs, take the best -- CPU timing on a 2-core box is noisy
        best, text = 1e9, ""
        for _ in range(3):
            t = time.perf_counter()
            segs, _ = model.transcribe(samples, language="en", beam_size=cfg.beam_size,
                                       vad_filter=False, **kw)
            out = " ".join(s.text.strip() for s in segs).strip()
            el = time.perf_counter() - t
            if el < best:
                best, text = el, out
        print(f"{name:<38} {best:>6.2f}  {text}")


if __name__ == "__main__":
    main()
