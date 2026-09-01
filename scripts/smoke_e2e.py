#!/usr/bin/env python3
"""Offline end-to-end smoke test: synthesize a spoken question with Windows SAPI,
run it through the real STT engine, the real question detector, and a real
streaming Claude call. Proves the expensive/risky parts of the pipeline work on
this machine without needing a live meeting or audio routing.

    .venv\\Scripts\\python.exe scripts\\smoke_e2e.py
    .venv\\Scripts\\python.exe scripts\\smoke_e2e.py "How does an ALB differ from an NLB?"
"""

from __future__ import annotations

import asyncio
import sys
import wave

import numpy as np

QUESTION_TEXT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "How would you design a multi region disaster recovery strategy on AWS?"
)


def synth_sapi(text: str, path: str) -> None:
    """Windows SAPI text-to-speech -> 22 kHz 16-bit mono WAV file."""
    import win32com.client  # from pywin32 (installed)

    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
    fmt.Type = 22  # SAFT22kHz16BitMono
    stream.Format = fmt
    stream.Open(path, 3)  # SSFMCreateForWrite
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    voice.AudioOutputStream = stream
    voice.Speak(text)
    stream.Close()


def load_wav_as_16k_mono_float(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        ch = w.getnchannels()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        tgt = int(len(audio) * 16000 / sr)
        audio = np.interp(
            np.linspace(0, len(audio), tgt, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


async def main() -> None:
    loop = asyncio.get_event_loop()
    from meeting_copilot.config import get_config
    from meeting_copilot.llm.claude_client import ClaudeClient
    from meeting_copilot.llm.prompt_templates import build_system_prompt, build_user_prompt
    from meeting_copilot.nlp.question_detector import get_question_detector
    from meeting_copilot.pipeline.events import RetrievedContext, Transcript
    from meeting_copilot.stt.faster_whisper_engine import FasterWhisperEngine
    from meeting_copilot.stt.term_normalizer import normalize

    cfg = get_config()
    wav_path = "logs/smoke_question.wav"

    print(f"[1/4] Synthesizing speech: {QUESTION_TEXT!r}")
    synth_sapi(QUESTION_TEXT, wav_path)
    samples = load_wav_as_16k_mono_float(wav_path)
    print(f"      {len(samples) / 16000:.1f}s of audio")

    print("[2/4] STT (faster-whisper, CPU)...")
    engine = FasterWhisperEngine(cfg.stt)
    t0 = loop.time()
    raw = engine.transcribe_samples(samples, 16000)
    text, _fixes = normalize(raw)
    print(f"      decode {loop.time() - t0:.1f}s")
    print(f"      raw:        {raw!r}")
    print(f"      normalized: {text!r}")

    print("[3/4] Question detection...")
    detector = get_question_detector(cfg.question_detector)
    tr = Transcript(speaker_id="speaker_a", text=text, start_time=0.0, end_time=len(samples) / 16000)
    detected = detector.detect(tr)
    print(f"      detected: {detected is not None}")
    if detected is None:
        print("      (not classified as a question -- stopping before the LLM call)")
        return

    print("[4/4] Claude (api, streaming)...")
    client = ClaudeClient(cfg.llm)
    system_prompt = build_system_prompt(question_text=text)
    user_prompt = build_user_prompt(RetrievedContext(question=detected, chunks=[]))
    t0 = loop.time()
    first = None
    answer = ""
    async for tok in client.stream(user_prompt, system=system_prompt):
        if first is None:
            first = loop.time() - t0
        answer += tok
    print(f"      TTFT {first:.2f}s, total {loop.time() - t0:.2f}s, {len(answer)} chars")
    print("\n----- ANSWER -----")
    print(answer.strip())
    print("------------------")
    print("\nOK: STT + detection + Claude all working on this machine.")


if __name__ == "__main__":
    asyncio.run(main())
