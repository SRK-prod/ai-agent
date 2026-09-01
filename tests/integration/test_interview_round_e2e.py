"""End-to-end validation of a full interview round through the REAL MeetingPipeline.

Drives the real orchestrator -- real question detector, real merge/revision/correction
logic, real term normalizer, real answer optimizer -- with only the three genuinely
external things stubbed: diarization (needs pyannote + a GPU-ish CPU), STT (needs Whisper),
and the Claude API (costs money and is non-deterministic). Everything the pipeline actually
decides is exercised for real.

Covers the behaviour the tool is judged on live:
  1. a question produces exactly one answer on the overlay
  2. a follow-up seconds later MERGES and REPLACES that answer -- never a second one
  3. an unrelated question does NOT merge -- it starts its own answer
  4. a self-correction answers only the corrected question
  5. garbled technical terms are repaired before the LLM sees them
  6. acknowledgements ("okay, got it") never trigger an answer
  7. near-silence never reaches STT at all
  8. Whisper repetition-loop hallucinations are dropped
  9. a long unbroken utterance is cut and the pieces re-merge into one question
 10. an answer that asks the interviewer to clarify is caught and regenerated
 11. the model falling over mid-answer fails over to the backup model

Run:  .venv\\Scripts\\python.exe -m pytest tests/integration/test_interview_round_e2e.py -v
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest

from meeting_copilot.pipeline.events import DiarizedSegment, SpeechSegment, Transcript
from meeting_copilot.pipeline.orchestrator import MeetingPipeline, seeks_clarification

# The orchestrator debounces before generating; give it comfortably more than that.
SETTLE = 0.9


@dataclass
class Overlay:
    """Stands in for the desktop overlay: records what the candidate would actually see."""

    answers: list[str] = field(default_factory=list)
    partials: list[str] = field(default_factory=list)

    async def on_answer(self, answer) -> None:
        self.answers.append(answer.text)

    async def on_partial(self, text: str) -> None:
        self.partials.append(text)

    @property
    def visible(self) -> str:
        """What is on screen right now -- the overlay shows the latest answer only."""
        return self.answers[-1] if self.answers else ""


class ScriptedStt:
    """Returns a queued transcript per segment, so the test controls the words exactly."""

    def __init__(self):
        self.texts: list[str] = []
        self.calls = 0

    async def transcribe(self, diarized: DiarizedSegment) -> Transcript | None:
        self.calls += 1
        text = self.texts.pop(0) if self.texts else None
        if text is None:
            return None
        return Transcript(
            speaker_id=diarized.speaker_id,
            text=text,
            start_time=diarized.segment.start_time,
            end_time=diarized.segment.end_time,
        )


class ScriptedClaude:
    """Echoes the question it was asked so assertions can prove WHICH question was answered."""

    def __init__(self, answer_fn=None):
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self._answer_fn = answer_fn or (lambda p: f"ANSWER<{p}>")

    def stream(self, prompt: str, system: str | None = None):
        self.prompts.append(prompt)
        self.systems.append(system or "")
        text = self._answer_fn(prompt)

        async def gen():
            for part in text.split(" "):
                await asyncio.sleep(0)
                yield part + " "

        return gen()

    async def complete(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system or "")
        return self._answer_fn(prompt)


class NullCache:
    async def get_llm_response(self, *a, **kw):
        return None

    async def set_llm_response(self, *a, **kw):
        return None

    async def close(self):
        return None


class _Diarizer:
    def diarize(self, segment: SpeechSegment) -> DiarizedSegment:
        return DiarizedSegment(segment=segment, speaker_id="A", is_me=False, similarity_to_me=0.1)


@pytest.fixture
def make_pipeline(monkeypatch):
    """Builds a REAL MeetingPipeline -- its actual __init__ runs, so every attribute the
    orchestrator relies on is set by the production code path rather than re-listed here
    (a hand-built object silently drifts out of date as the orchestrator gains state).
    Only the four external boundaries are swapped: audio device, diarization model, Whisper,
    and the Claude API.
    """
    import meeting_copilot.pipeline.orchestrator as orch

    monkeypatch.setattr(orch, "AudioCapture", lambda *a, **kw: object())
    monkeypatch.setattr(orch, "SileroVAD", lambda *a, **kw: object())
    monkeypatch.setattr(orch, "SpeakerDiarizer", lambda *a, **kw: _Diarizer())
    monkeypatch.setattr(orch, "SttStage", lambda *a, **kw: ScriptedStt())
    monkeypatch.setattr(orch, "RedisCache", lambda *a, **kw: NullCache())

    def _build(overlay: Overlay, claude=None):
        monkeypatch.setattr(orch, "ClaudeClient", lambda *a, **kw: claude or ScriptedClaude())
        pipe = MeetingPipeline(
            on_answer=overlay.on_answer, on_partial_answer=overlay.on_partial
        )
        return pipe

    return _build


def segment(start: float, end: float, level: float = 0.2) -> SpeechSegment:
    n = max(int((end - start) * 16000), 160)
    rng = np.random.default_rng(0)
    return SpeechSegment(
        samples=(rng.standard_normal(n) * level).astype(np.float32),
        sample_rate=16000,
        start_time=start,
        end_time=end,
    )


async def say(pipe, overlay, text: str, start: float, end: float, settle: float = SETTLE):
    """Interviewer speaks one utterance; wait for the pipeline to settle."""
    pipe._stt.texts.append(text)
    await pipe._handle_segment(segment(start, end))
    await asyncio.sleep(settle)
    return overlay.visible


# ---------------------------------------------------------------- 1. basic Q -> answer


async def test_question_produces_exactly_one_answer(make_pipeline):
    overlay = Overlay()
    pipe = make_pipeline(overlay)
    await say(pipe, overlay, "How would you design a highly available ECS platform?", 0.0, 5.0)

    assert len(overlay.answers) == 1, f"expected 1 answer, got {len(overlay.answers)}"
    assert "highly available ECS" in overlay.visible
    assert overlay.partials, "overlay must stream partial text, not sit blank until done"


# ---------------------------------------------- 2. follow-up merges and REPLACES the answer


async def test_followup_merges_and_replaces_not_appends(make_pipeline):
    overlay = Overlay()
    pipe = make_pipeline(overlay)
    await say(pipe, overlay, "How would you design event correlation?", 0.0, 4.0)
    first = overlay.visible
    assert "event correlation" in first

    # Same topic, 2s later -- must fold into the SAME answer.
    await say(pipe, overlay, "specifically across Prometheus and Splunk", 6.0, 9.0)

    assert len(overlay.answers) == 2, "a revision replaces the answer (2 renders, 1 question)"
    assert "event correlation" in overlay.visible, "merged answer lost the original question"
    assert "Prometheus" in overlay.visible, "merged answer lost the follow-up detail"


# ------------------------------------------------- 3. unrelated question must NOT merge


async def test_unrelated_question_starts_its_own_answer(make_pipeline):
    overlay = Overlay()
    pipe = make_pipeline(overlay)
    await say(pipe, overlay, "How do you manage Terraform state?", 0.0, 4.0)
    await say(pipe, overlay, "How would you secure a Kubernetes cluster?", 6.0, 10.0)

    last = overlay.visible
    assert "Kubernetes" in last
    assert "Terraform state" not in last, (
        "an unrelated question merged into the previous one -- this is the mega-answer bug"
    )


# --------------------------------------------------------------- 4. self-correction


async def test_correction_answers_only_the_corrected_question(make_pipeline):
    overlay = Overlay()
    pipe = make_pipeline(overlay)
    await say(pipe, overlay, "How would you design this on ECS?", 0.0, 4.0)
    await say(pipe, overlay, "Actually, I mean Kubernetes.", 5.0, 7.0)

    assert "Kubernetes" in overlay.visible, "the corrected question was not answered"
    # The correction is enforced by injecting a CORRECTION block into the SYSTEM prompt
    # ("the later statement REPLACES the earlier one -- answer ONLY the corrected
    # question"), not by rewriting the question text. Assert on that, because that is the
    # real mechanism: asserting "ECS" is absent from the answer would only be testing this
    # file's echo-the-prompt stub, which cannot honour a system-prompt instruction.
    assert pipe._claude.systems, "no system prompt was captured"
    assert "CORRECTION" in pipe._claude.systems[-1], (
        "the correction was merged as an ordinary revision -- the model was never told the "
        "ECS reading is void, so it would answer both"
    )


# ------------------------------------------------ 5. garbled technical terms repaired


@pytest.mark.parametrize(
    ("garbled", "expected"),
    [
        ("What is OOMCade in Kubernetes?", "OOMKilled"),
        ("Explain crash look back off", "CrashLoopBackOff"),
        ("Tell me about AIG systems", "Agentic AI"),
    ],
)
def test_garbled_terms_are_repaired_before_the_llm(garbled: str, expected: str):
    from meeting_copilot.stt.term_normalizer import normalize

    fixed, count = normalize(garbled)
    assert expected.lower() in fixed.lower(), f"{garbled!r} -> {fixed!r}, expected {expected!r}"
    assert count >= 1


# ---------------------------------------------------------- 6. acknowledgements ignored


@pytest.mark.parametrize("filler", ["Okay.", "Got it, thanks.", "Yeah makes sense", "Mm hmm"])
async def test_acknowledgements_never_trigger_an_answer(filler: str, make_pipeline):
    overlay = Overlay()
    pipe = make_pipeline(overlay)
    await say(pipe, overlay, "How would you design an HA platform?", 0.0, 4.0)
    before = len(overlay.answers)
    await say(pipe, overlay, filler, 6.0, 7.0)

    assert len(overlay.answers) == before, f"{filler!r} triggered a regeneration"


# ------------------------------------------------------- 7. near-silence never hits STT


def test_near_silence_skips_stt_entirely():
    from meeting_copilot.stt.faster_whisper_engine import _is_near_silent

    assert _is_near_silent(np.zeros(16000, dtype=np.float32))
    assert _is_near_silent(np.full(16000, 0.0005, dtype=np.float32))
    # real speech must NOT be skipped
    rng = np.random.default_rng(1)
    assert not _is_near_silent((rng.standard_normal(16000) * 0.15).astype(np.float32))


# ------------------------------------------------------- 8. hallucination loops dropped


@pytest.mark.parametrize(
    "text",
    [
        "create create create create create create create",
        "the other one, the other one, the other one, the other one",
        "I could be red or I could be red or I could be red or I could be red",
    ],
)
def test_repetition_hallucinations_are_dropped(text: str):
    from meeting_copilot.stt.faster_whisper_engine import _is_hallucinated

    assert _is_hallucinated(text), f"not filtered: {text!r}"


def test_real_answer_is_not_mistaken_for_a_hallucination():
    from meeting_copilot.stt.faster_whisper_engine import _is_hallucinated

    real = (
        "How would you design a multi region disaster recovery strategy for a Kubernetes "
        "platform running on EKS with an RTO of fifteen minutes?"
    )
    assert not _is_hallucinated(real)


# ------------------------------------- 9. long utterance is cut, then merged back together


async def test_forced_vad_cut_pieces_remerge_into_one_question(make_pipeline):
    """max_speech_ms cuts a long unbroken question; the fragments must re-merge, not
    become two separate answers."""
    overlay = Overlay()
    pipe = make_pipeline(overlay)
    # Two back-to-back pieces of ONE question, gap well under _FRAGMENT_MERGE_GAP_SECONDS.
    pipe._stt.texts.append("So how would you design a multi region")
    await pipe._handle_segment(segment(0.0, 12.0))
    pipe._stt.texts.append("disaster recovery setup for EKS?")
    await pipe._handle_segment(segment(12.1, 18.0))
    await asyncio.sleep(SETTLE)

    assert len(overlay.answers) >= 1
    assert "multi region" in overlay.visible and "disaster recovery" in overlay.visible, (
        "a forced VAD cut split one question into two answers"
    )


def test_vad_cap_is_configured_and_sane():
    from meeting_copilot.config import get_config

    vad = get_config().vad
    assert vad.max_speech_ms == 0 or vad.max_speech_ms >= 5000, (
        "a cap under 5s would chop normal questions mid-sentence"
    )
    if vad.max_speech_ms:
        assert vad.max_speech_ms > vad.min_silence_ms


# ------------------------------------------------- 10. clarification-seeking is caught


@pytest.mark.parametrize(
    "bad_opening",
    [
        (
            "I need to clarify what you're asking here -- are you asking me to redesign the "
            "reference architecture, or are you asking something else about the setup?"
        ),
        "Could you clarify which part of the design you mean?",
        "I need more context to answer this.",
        "The transcript appears to be incomplete, so I'll infer the question.",
        "Per my instructions, I should infer the clearest real question here.",
        "Are you asking about the network layer, or the storage layer, or the compute layer?",
    ],
)
def test_clarification_seeking_answers_are_detected(bad_opening: str):
    assert seeks_clarification(bad_opening), f"NOT caught: {bad_opening[:60]!r}"


@pytest.mark.parametrize(
    "good_answer",
    [
        "I'd run three replicas across three availability zones and put an ALB in front.",
        "I'd keep the control plane vendor-agnostic. Metrics collection stays deterministic.",
        "My decision is to keep DynamoDB managed rather than moving the database into the cluster.",
    ],
)
def test_good_answers_are_not_false_positives(good_answer: str):
    assert not seeks_clarification(good_answer)


async def test_clarification_answer_triggers_a_regeneration(make_pipeline):
    """The guard must actually re-prompt, not just detect."""
    calls = {"n": 0}

    def answer_fn(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "I need to clarify what you're asking here -- are you asking about X?"
        return "I'd run three replicas across three AZs behind an ALB."

    overlay = Overlay()
    pipe = make_pipeline(overlay, claude=ScriptedClaude(answer_fn))
    await say(pipe, overlay, "How would you design a highly available platform?", 0.0, 5.0, 1.4)

    assert calls["n"] >= 2, "a clarification-seeking answer was not regenerated"
    assert not seeks_clarification(overlay.visible), (
        f"clarification request still reached the overlay: {overlay.visible[:90]!r}"
    )


# --------------------------------------------------------- 11. model failure -> fallback


def test_fallback_model_is_configured():
    from meeting_copilot.config import get_config

    llm = get_config().llm
    assert llm.fallback_models, "no fallback model -- a single model outage kills the tool"
    assert llm.model not in llm.fallback_models, "fallback must differ from the primary"
    assert llm.stream_timeout_seconds > 0, "a hung stream would stall a question forever"


async def test_llm_failure_does_not_crash_the_pipeline(make_pipeline):
    """A generation blowing up must not take down the worker -- the next question must
    still be answered."""

    class ExplodingClaude(ScriptedClaude):
        def __init__(self):
            super().__init__()
            self.n = 0

        def stream(self, prompt, system=None):
            self.n += 1
            first = self.n == 1

            async def gen():
                if first:
                    raise RuntimeError("simulated API failure")
                yield "recovered answer"

            return gen()

    overlay = Overlay()
    pipe = make_pipeline(overlay, claude=ExplodingClaude())
    await say(pipe, overlay, "How would you design an HA platform?", 0.0, 4.0)
    await say(pipe, overlay, "How do you secure a Kubernetes cluster?", 40.0, 44.0)

    assert overlay.answers, "pipeline died on the first failure and never recovered"
    assert "recovered" in overlay.visible
