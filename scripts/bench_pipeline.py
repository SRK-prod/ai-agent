#!/usr/bin/env python3
"""Per-stage latency breakdown of the real pipeline on THIS machine.

Answers "where does the time actually go" with measurements instead of guesses,
so optimization effort is spent on the stage that dominates. Runs the same code
the live pipeline runs (diarizer, STT stage, detector, prompt build, Claude) on a
synthesized utterance.

    .venv\\Scripts\\python.exe scripts\\bench_pipeline.py
"""

from __future__ import annotations

import asyncio
import time
import wave

import numpy as np

SENTENCE = (
    "How would you design a multi region disaster recovery strategy for a "
    "Kubernetes platform running on EKS?"
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


async def main() -> None:
    from meeting_copilot.config import get_config
    from meeting_copilot.llm.claude_client import ClaudeClient
    from meeting_copilot.llm.prompt_templates import build_system_prompt, build_user_prompt
    from meeting_copilot.nlp.question_detector import get_question_detector
    from meeting_copilot.pipeline.events import RetrievedContext, SpeechSegment, Transcript
    from meeting_copilot.speaker.diarization import SpeakerDiarizer
    from meeting_copilot.stt.faster_whisper_engine import FasterWhisperEngine
    from meeting_copilot.stt.term_normalizer import normalize

    cfg = get_config()
    wav = "logs/bench_pipeline.wav"
    synth_sapi(SENTENCE, wav)
    samples = load_wav_16k_mono(wav)
    dur = len(samples) / 16000
    print(f"audio: {dur:.1f}s | stt.model={cfg.stt.model_size} beam={cfg.stt.beam_size}\n")

    results: list[tuple[str, float]] = []

    # --- cold init (paid once at backend boot, not per question) ---
    t = time.perf_counter()
    diarizer = SpeakerDiarizer()
    init_diar = time.perf_counter() - t
    t = time.perf_counter()
    engine = FasterWhisperEngine(cfg.stt)
    init_stt = time.perf_counter() - t
    print(f"[init, once at boot] diarizer {init_diar:.1f}s | stt model {init_stt:.1f}s\n")

    segment = SpeechSegment(samples=samples, sample_rate=16000, start_time=0.0, end_time=dur)

    # --- per-question stages (run twice; report the warm second run) ---
    for run in ("cold", "warm"):
        t = time.perf_counter()
        diarized = diarizer.diarize(segment)
        d_diar = time.perf_counter() - t

        t = time.perf_counter()
        raw = engine.transcribe_samples(samples, 16000)
        d_stt = time.perf_counter() - t

        t = time.perf_counter()
        text, _ = normalize(raw)
        d_norm = time.perf_counter() - t

        detector = get_question_detector(cfg.question_detector)
        tr = Transcript(speaker_id=diarized.speaker_id, text=text, start_time=0.0, end_time=dur)
        t = time.perf_counter()
        detected = detector.detect(tr)
        d_detect = time.perf_counter() - t

        t = time.perf_counter()
        system_prompt = build_system_prompt(question_text=text)
        user_prompt = build_user_prompt(RetrievedContext(question=detected, chunks=[]))
        d_prompt = time.perf_counter() - t

        if run == "warm":
            results += [
                ("diarization (pyannote)", d_diar),
                ("STT decode", d_stt),
                ("term normalize", d_norm),
                ("question detect", d_detect),
                ("prompt build", d_prompt),
            ]
            print(f"transcript: {text!r}\n")

    # --- Claude ---
    client = ClaudeClient(cfg.llm)
    t = time.perf_counter()
    ttft = None
    answer = ""
    async for tok in client.stream(user_prompt, system=system_prompt):
        if ttft is None:
            ttft = time.perf_counter() - t
        answer += tok
    d_total_llm = time.perf_counter() - t
    results.append(("Claude time-to-first-token", ttft or 0.0))

    print(f"{'stage':<28} {'seconds':>9}   share of TTFA")
    print("-" * 60)
    ttfa = sum(s for _, s in results)
    for name, secs in results:
        bar = "#" * max(1, int(40 * secs / ttfa))
        print(f"{name:<28} {secs:>9.2f}   {bar} {100 * secs / ttfa:.0f}%")
    print("-" * 60)
    print(f"{'TTFA (sum of stages)':<28} {ttfa:>9.2f}")
    print(f"{'+ VAD silence gate':<28} {cfg.vad.min_silence_ms / 1000:>9.2f}  (fixed, before any of this)")
    print(f"{'= perceived latency':<28} {ttfa + cfg.vad.min_silence_ms / 1000:>9.2f}")
    print(f"\n(full answer streamed in {d_total_llm:.1f}s, {len(answer)} chars)")


if __name__ == "__main__":
    asyncio.run(main())
