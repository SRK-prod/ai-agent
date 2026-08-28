"""Record real interviewer audio, then compare STT models on the metric that actually
matters: end-of-speech to first visible answer token, not raw STT decode time in isolation.

Why this exists: the STT model is the single largest latency lever in the pipeline
(measured 2026-08-27 -- transcription was 81% of end-to-end latency), but the cheaper model
was abandoned on 2026-08-07 after it hallucinated repetition loops and mangled technical
vocabulary in a live interview. Deciding between them on vibes is how that regression
happened the first time, so this harness makes the trade-off measurable on the candidate's
OWN audio path and the candidate's OWN full pipeline rather than on synthetic tone or a
raw-decode-speed number that isn't what the interviewer experiences.

Two steps, deliberately separate so recording happens once and comparison can be re-run:

    python scripts/record_stt_samples.py --duration 20 --label short-technical-q
    python scripts/compare_stt_models.py

Corrected 2026-08-28: the first version of this harness measured only
engine.transcribe_samples() runtime. That is necessary but not sufficient -- a model can
decode fast and still lose the question (mis-transcribed badly enough that the rule-based
detector doesn't recognize it as one) or hallucinate (rejected before the LLM ever sees it),
either of which means the REAL latency is infinite: no answer appears at all. The primary
number this harness reports is therefore the full path:

    STT decode -> hallucination/term-repair -> question detection -> LLM call -> first token

exactly mirroring what the live orchestrator measures as TTFA. This runs a real LLM call
per sample per model, which costs real API tokens -- pass --stt-only on the compare script
to skip that and get only the (still useful, but incomplete) decode-speed number.

PRIVACY. Recordings are real interview audio. They are written under `data/stt_samples/`,
which is gitignored, they never leave the machine except for the (expected, necessary)
Anthropic API call each full comparison makes, and nothing here uploads the audio itself.
Delete the directory when the comparison is done -- `--clean` on the compare script does it.
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meeting_copilot.config import SttConfig, get_config
from meeting_copilot.paths import DATA_DIR
from meeting_copilot.stt.faster_whisper_engine import _is_hallucinated
from meeting_copilot.stt.term_normalizer import normalize as normalize_terms
from meeting_copilot.utils.logging import get_logger

logger = get_logger()

# Models worth comparing on Apple Silicon. distil-large-v3 is deliberately included even
# though a quick benchmark showed it slower and erratic on synthetic audio -- a one-off
# measurement on synthetic audio is exactly the kind of evidence this harness replaces.
CANDIDATE_MODELS = ("large-v3-turbo", "small", "distil-large-v3")

SAMPLE_RATE = 16000

# Coverage checklist -- printed before recording so a comparison isn't accidentally run on
# eight variations of the same easy case. Judging a model on short clean questions alone
# would have missed the exact failure (hallucination on noisy/compressed call audio) that
# caused the 2026-08-07 regression this harness exists to prevent from repeating blind.
SAMPLE_CHECKLIST = (
    ("short-technical-q", "A short, single-concept technical question (~10-15s)"),
    ("medium-technical-q", "A medium technical question with some elaboration (~20-30s)"),
    ("long-architecture-q", "A full system-design / architecture question (~30-60s)"),
    ("cloud-terminology", "Dense in AWS/Azure/GCP/Kubernetes vocabulary specifically"),
    ("numbers-acronyms", "Contains numbers, versions, or several acronyms in a row"),
    ("follow-up-q", "A short follow-up referencing something just discussed"),
    ("natural-pauses", "Natural interviewer speech with mid-sentence pauses"),
    ("noisy-call-audio", "Real call audio quality -- background noise, compression, cross-talk"),
)


def samples_dir() -> Path:
    return DATA_DIR / "stt_samples"


def print_checklist() -> None:
    have = {p.stem.split("-", 1)[1] if "-" in p.stem else "" for p in samples_dir().glob("*.wav")}
    print("\nSuggested coverage (record one sample per row for a real comparison):")
    for label, description in SAMPLE_CHECKLIST:
        mark = "x" if any(label in name for name in have) else " "
        print(f"  [{mark}] {label:22} {description}")
    print()


# --- recording ---------------------------------------------------------------------


def record_sample(duration_seconds: float, label: str) -> Path:
    """Capture `duration_seconds` from the SAME input device the live pipeline uses.

    Recording from the configured device (BlackHole in production) rather than the default
    microphone is the whole point: the thing being evaluated is how the models behave on
    compressed, routed call audio, which is where the original quality regression appeared.
    """
    import sounddevice as sd

    from meeting_copilot.audio.capture import resolve_device

    cfg = get_config().audio
    device = resolve_device(cfg.input_device)

    out_dir = samples_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
    path = out_dir / f"{int(time.time())}-{safe_label}.wav"

    print(f"Recording {duration_seconds:.0f}s from device {device!r} ({cfg.input_device})...")
    print("Play or speak the interviewer audio now.")
    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()

    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    print(f"Saved {path}  (peak level {peak:.4f})")
    if peak < 0.01:
        print(
            "  WARNING: that is effectively silence. Check that call audio is routed to "
            f"{cfg.input_device!r} before recording again -- otherwise the comparison is "
            "measuring nothing."
        )
    print_checklist()
    return path


def load_sample(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0


# --- comparison --------------------------------------------------------------------


@dataclass
class ModelResult:
    model: str
    sample: str
    stt_ms: float
    text: str
    hallucinated: bool
    term_repairs: int
    detected_as_question: bool
    # Below are None when --stt-only is used, or when the transcript never reached the LLM
    # (hallucination-rejected or not detected as a question -- matching real pipeline
    # behavior, where those cases never generate an answer at all).
    llm_ttft_ms: float | None = None
    llm_total_ms: float | None = None
    end_to_end_ms: float | None = None  # STT + question-detect + LLM first token
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


def _engine_for(model_size: str):
    from meeting_copilot.stt.mlx_whisper_engine import MlxWhisperEngine

    base = get_config().stt
    return MlxWhisperEngine(
        SttConfig(
            backend="mlx-whisper",
            model_size=model_size,
            device=base.device,
            compute_type=base.compute_type,
            language=base.language,
            vocabulary_hint=base.vocabulary_hint,
        )
    )


async def _run_llm_leg(question_text: str) -> tuple[float, float, int | None, int | None, str | None]:
    """Real system-prompt + real Claude call, timing first-token and total. Returns
    (ttft_ms, total_ms, input_tokens, output_tokens, error)."""
    from meeting_copilot.llm.claude_client import ClaudeClient
    from meeting_copilot.llm.prompt_templates import build_system_prompt, build_user_prompt
    from meeting_copilot.pipeline.events import DetectedQuestion, RetrievedContext, Transcript

    transcript = Transcript(
        speaker_id="interviewer", text=question_text, start_time=0.0, end_time=1.0
    )
    question = DetectedQuestion(transcript=transcript, matched_keywords=[], ends_with_question_mark=True)
    context = RetrievedContext(question=question, chunks=[])
    system = build_system_prompt(question_text=question_text)
    user = build_user_prompt(context)

    client = ClaudeClient()
    started = time.monotonic()
    first_token_at: float | None = None
    try:
        async for _chunk in client.stream(user, system=system):
            if first_token_at is None:
                first_token_at = time.monotonic()
        total = time.monotonic() - started
        ttft = (first_token_at or time.monotonic()) - started
        usage = client.last_usage()
        return ttft * 1000, total * 1000, (usage[0] if usage else None), (usage[1] if usage else None), None
    except Exception as exc:  # a real reliability signal, not just noise to hide
        return 0.0, 0.0, None, None, f"{type(exc).__name__}: {exc}"


async def run_comparison(
    models: tuple[str, ...] = CANDIDATE_MODELS, stt_only: bool = False
) -> list[ModelResult]:
    from meeting_copilot.nlp.question_detector import get_question_detector

    paths = sorted(samples_dir().glob("*.wav"))
    if not paths:
        raise SystemExit(
            f"No samples in {samples_dir()}. Record some first:\n"
            "  python scripts/record_stt_samples.py --duration 20 --label short-technical-q"
        )

    detector = get_question_detector()
    results: list[ModelResult] = []
    for model in models:
        print(f"\nLoading {model}...")
        try:
            engine = _engine_for(model)
        except Exception as exc:  # a model that won't even load is a real result
            print(f"  SKIPPED -- failed to load: {type(exc).__name__}: {exc}")
            continue
        # Warm up so the first real sample isn't charged for weight loading.
        engine.transcribe_samples(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

        for path in paths:
            audio = load_sample(path)
            t0 = time.monotonic()
            text = engine.transcribe_samples(audio, SAMPLE_RATE)
            stt_ms = (time.monotonic() - t0) * 1000

            hallucinated = _is_hallucinated(text) if text else False
            normalized, repairs = normalize_terms(text) if text else (text, 0)

            from meeting_copilot.pipeline.events import Transcript as _T

            detected = None
            if text and not hallucinated:
                detected = detector.detect(
                    _T(speaker_id="interviewer", text=normalized, start_time=0.0, end_time=1.0)
                )

            result = ModelResult(
                model=model,
                sample=path.name,
                stt_ms=stt_ms,
                text=normalized,
                hallucinated=hallucinated,
                term_repairs=repairs,
                detected_as_question=detected is not None,
            )

            if not stt_only and detected is not None:
                ttft_ms, total_ms, in_tok, out_tok, err = await _run_llm_leg(normalized)
                result.llm_ttft_ms = ttft_ms
                result.llm_total_ms = total_ms
                result.input_tokens = in_tok
                result.output_tokens = out_tok
                result.error = err
                if err is None:
                    result.end_to_end_ms = stt_ms + ttft_ms

            results.append(result)
            status = "OK" if result.end_to_end_ms is not None else (
                "HALLUCINATED" if hallucinated else
                "NOT DETECTED AS QUESTION" if not detected else
                "LLM ERROR" if result.error else "stt-only"
            )
            e2e = f"{result.end_to_end_ms:.0f}ms end-to-end" if result.end_to_end_ms else status
            print(f"  {path.name}: stt={stt_ms:.0f}ms  {e2e}")
    return results


def report(results: list[ModelResult]) -> None:
    if not results:
        print("No results.")
        return

    by_sample: dict[str, list[ModelResult]] = {}
    for r in results:
        by_sample.setdefault(r.sample, []).append(r)

    print("\n" + "=" * 84)
    print("TRANSCRIPTS -- read these first. Speed only matters if the words are right.")
    print("=" * 84)
    for sample, rows in by_sample.items():
        print(f"\n--- {sample} ---")
        for r in rows:
            flags = []
            if r.hallucinated:
                flags.append("HALLUCINATION-LOOP -- rejected before reaching the LLM")
            if r.term_repairs:
                flags.append(f"{r.term_repairs} term repair(s) needed")
            if not r.hallucinated and not r.detected_as_question:
                flags.append("NOT RECOGNIZED AS A QUESTION -- would never be answered live")
            if r.error:
                flags.append(f"LLM ERROR: {r.error}")
            suffix = f"   [{', '.join(flags)}]" if flags else ""
            print(f"\n  {r.model} (stt {r.stt_ms:.0f}ms){suffix}")
            print(f"    {r.text or '(empty)'}")

    print("\n" + "=" * 84)
    print("LATENCY -- end-to-end is end-of-speech -> first visible answer token,")
    print("the number the interviewer actually experiences. STT-only is decode time alone.")
    print("=" * 84)
    models = sorted({r.model for r in results})
    header = f"  {'model':18} {'stt (mean)':>12} {'e2e (mean)':>12} {'e2e (P95)':>12} {'reached LLM':>12} {'halluc.':>9}"
    print(header)
    for model in models:
        rows = [r for r in results if r.model == model]
        stt_times = sorted(r.stt_ms for r in rows)
        e2e_times = sorted(r.end_to_end_ms for r in rows if r.end_to_end_ms is not None)
        loops = sum(1 for r in rows if r.hallucinated)
        reached = len(e2e_times)
        stt_mean = sum(stt_times) / len(stt_times)
        if e2e_times:
            e2e_mean = sum(e2e_times) / len(e2e_times)
            e2e_p95 = e2e_times[min(len(e2e_times) - 1, int(len(e2e_times) * 0.95))]
            e2e_mean_s, e2e_p95_s = f"{e2e_mean:.0f}ms", f"{e2e_p95:.0f}ms"
        else:
            e2e_mean_s = e2e_p95_s = "n/a"
        print(
            f"  {model:18} {stt_mean:10.0f}ms {e2e_mean_s:>12} {e2e_p95_s:>12} "
            f"{reached:>8}/{len(rows):<3} {loops:>6}/{len(rows)}"
        )

    token_rows = [r for r in results if r.input_tokens is not None]
    if token_rows:
        print("\n" + "-" * 84)
        print("COST -- local STT itself has no API cost. This is the LLM call each model's")
        print("transcript produced -- token counts should be similar across models since the")
        print("prompt is dominated by the shared system prompt, not the transcript. A model")
        print("whose transcripts need real formatting cleanup before the LLM sees them is the")
        print("cost signal to watch here, not raw token count.")
        print("-" * 84)
        for model in models:
            rows = [r for r in token_rows if r.model == model]
            if not rows:
                continue
            avg_in = sum(r.input_tokens for r in rows) / len(rows)
            avg_out = sum(r.output_tokens for r in rows) / len(rows)
            print(f"  {model:18} input~{avg_in:.0f} tok  output~{avg_out:.0f} tok")

    print(
        "\nDecide on the transcripts first, then reached-LLM rate, then hallucination rate, "
        "then latency. A model that is faster on paper but loses or mangles the question "
        "before it reaches the LLM produced an infinite latency for that question, not a "
        "fast one."
    )
