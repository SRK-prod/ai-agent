#!/usr/bin/env python3
"""Transcribe a real recorded call through the SAME path the live pipeline uses --
Silero VAD segmentation, then per-segment Whisper decode, then term normalization --
so model choice can be judged on real call audio instead of clean synthesized speech.

    .venv\\Scripts\\python.exe scripts\\transcribe_sample.py logs\\sample_client.wav
    .venv\\Scripts\\python.exe scripts\\transcribe_sample.py logs\\sample_client.wav --models tiny,small
    .venv\\Scripts\\python.exe scripts\\transcribe_sample.py logs\\sample_client.wav --seconds 90
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np


def load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"expected 16kHz mono, got {sr}Hz/{ch}ch -- re-run ffmpeg")
    return a


def vad_segments(samples: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """Segment with Silero VAD using the same thresholds the live pipeline uses."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    from meeting_copilot.config import get_config

    cfg = get_config().vad
    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(samples),
        model,
        sampling_rate=16000,
        threshold=cfg.threshold,
        min_speech_duration_ms=cfg.min_speech_ms,
        min_silence_duration_ms=cfg.min_silence_ms,
    )
    return [(t["start"] / 16000, t["end"] / 16000, samples[t["start"] : t["end"]]) for t in ts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--models", default="tiny")
    ap.add_argument("--seconds", type=float, default=0.0, help="only the first N seconds")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    from meeting_copilot.config import get_config
    from meeting_copilot.stt.faster_whisper_engine import _is_hallucinated, _is_near_silent
    from meeting_copilot.stt.term_normalizer import normalize

    cfg = get_config().stt
    samples = load_wav_16k_mono(args.wav)
    if args.seconds:
        samples = samples[: int(args.seconds * 16000)]
    total = len(samples) / 16000
    print(f"audio: {total:.0f}s ({total / 60:.1f} min)  rms={np.sqrt(np.mean(samples**2)):.4f}")

    segs = vad_segments(samples)
    spoken = sum(e - s for s, e, _ in segs)
    print(f"VAD: {len(segs)} speech segments, {spoken:.0f}s of speech "
          f"({100 * spoken / total:.0f}% of the file)\n")

    for model_size in args.models.split(","):
        model_size = model_size.strip()
        print("=" * 100)
        print(f"MODEL: {model_size}  (beam={cfg.beam_size})")
        print("=" * 100)
        model = WhisperModel(model_size, device="cpu", compute_type=cfg.compute_type,
                             cpu_threads=cfg.cpu_threads)
        decode_total = 0.0
        skipped_silent = skipped_halluc = 0
        for i, (start, end, chunk) in enumerate(segs, 1):
            if _is_near_silent(chunk):
                skipped_silent += 1
                continue
            t = time.perf_counter()
            out, _ = model.transcribe(
                chunk, language=cfg.language, beam_size=cfg.beam_size,
                condition_on_previous_text=False, vad_filter=False,
                without_timestamps=True, initial_prompt=cfg.initial_prompt,
            )
            raw = " ".join(s.text.strip() for s in out).strip()
            dt = time.perf_counter() - t
            decode_total += dt
            if _is_hallucinated(raw):
                skipped_halluc += 1
                print(f"[{start:6.1f}s] ({dt:4.1f}s) <HALLUCINATION FILTERED> {raw[:70]!r}")
                continue
            text, fixes = normalize(raw)
            flag = f"  (+{fixes} term fixes)" if fixes else ""
            print(f"[{start:6.1f}s] ({dt:4.1f}s) {text}{flag}")
        print(f"\n-- {model_size}: {decode_total:.0f}s decode for {spoken:.0f}s of speech "
              f"({decode_total / max(spoken, 1e-9):.2f}x realtime), "
              f"{skipped_silent} silent-skipped, {skipped_halluc} hallucination-filtered\n")
        del model


if __name__ == "__main__":
    main()
