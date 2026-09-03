"""The two pause behaviours a live interview depends on, through the real orchestrator.

Both were verified by hand before the 2026-09-03 interview and are pinned here because both
are silent failures: nothing errors, you just get two answers where you wanted one, or an
answer that lost the scenario it depended on.

  A. A thinking pause mid-question is still ONE question.
       "How would you architect Terraform..."  [~18s]  "...for GCP and AWS"
  B. A long scenario buildup ending in the real ask is ONE answer carrying the whole
     context, and none of the setup sentences trigger an answer of their own.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest

from meeting_copilot.pipeline.events import DiarizedSegment, SpeechSegment, Transcript

SETTLE = 0.9


@dataclass
class Overlay:
    answers: list[str] = field(default_factory=list)

    async def on_answer(self, answer) -> None:
        self.answers.append(answer.text)

    async def on_partial(self, text: str) -> None:
        pass

    @property
    def visible(self) -> str:
        return self.answers[-1] if self.answers else ""


class ScriptedStt:
    def __init__(self):
        self.texts: list[str] = []

    async def transcribe(self, diarized: DiarizedSegment) -> Transcript | None:
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
    """Echoes the prompt so a test can prove WHICH question and context were sent."""

    def __init__(self):
        self.prompts: list[str] = []

    def stream(self, prompt: str, system: str | None = None):
        self.prompts.append(prompt)

        async def gen():
            for part in f"ANSWER<{prompt}>".split(" "):
                await asyncio.sleep(0)
                yield part + " "

        return gen()

    async def complete(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return f"ANSWER<{prompt}>"


class NullCache:
    async def get_llm_response(self, *a, **kw):
        return None

    async def set_llm_response(self, *a, **kw):
        return None

    async def close(self):
        return None


class _Diarizer:
    def diarize(self, segment: SpeechSegment) -> DiarizedSegment:
        return DiarizedSegment(
            segment=segment, speaker_id="A", is_me=False, similarity_to_me=0.1
        )


@pytest.fixture
def make_pipeline(monkeypatch):
    import meeting_copilot.pipeline.orchestrator as orch

    monkeypatch.setattr(orch, "AudioCapture", lambda *a, **kw: object())
    monkeypatch.setattr(orch, "SileroVAD", lambda *a, **kw: object())
    monkeypatch.setattr(orch, "SpeakerDiarizer", lambda *a, **kw: _Diarizer())
    monkeypatch.setattr(orch, "SttStage", lambda *a, **kw: ScriptedStt())
    monkeypatch.setattr(orch, "RedisCache", lambda *a, **kw: NullCache())
    monkeypatch.setattr(orch, "ClaudeClient", lambda *a, **kw: ScriptedClaude())

    def _build(overlay: Overlay):
        return orch.MeetingPipeline(
            on_answer=overlay.on_answer, on_partial_answer=overlay.on_partial
        )

    return _build


def segment(start: float, end: float) -> SpeechSegment:
    n = max(int((end - start) * 16000), 160)
    rng = np.random.default_rng(0)
    return SpeechSegment(
        samples=(rng.standard_normal(n) * 0.2).astype(np.float32),
        sample_rate=16000,
        start_time=start,
        end_time=end,
    )


async def say(pipe, text: str, start: float, end: float) -> None:
    pipe._stt.texts.append(text)
    await pipe._handle_segment(segment(start, end))
    await asyncio.sleep(SETTLE)


async def test_thinking_pause_mid_question_stays_one_question(make_pipeline):
    """An ~18s pause before a grammatical continuation must not split the question."""
    overlay = Overlay()
    pipe = make_pipeline(overlay)

    await say(pipe, "How would you architect Terraform", 0.0, 4.0)
    await say(pipe, "for GCP and AWS", 22.0, 25.0)

    visible = overlay.visible
    assert "Terraform" in visible and "GCP" in visible, (
        "the continuation did not merge back into the original question"
    )
    assert "Question 1:" not in visible and "Question 2:" not in visible, (
        "a grammatical continuation was numbered as a second question"
    )


async def test_long_scenario_buildup_produces_one_answer_with_full_context(make_pipeline):
    """Setup sentences are held, not answered; the final ask carries all of them."""
    overlay = Overlay()
    pipe = make_pipeline(overlay)

    await say(pipe, "We run an enterprise platform across AWS.", 0.0, 7.0)
    await say(pipe, "There are strict compliance requirements.", 12.0, 19.0)
    await say(pipe, "The platform needs to support GCP as well.", 24.0, 31.0)
    await say(pipe, "So how would you architect this?", 36.0, 40.0)

    # Renders > 1 is BY DESIGN: each buildup sentence revises the answer in place and the
    # overlay shows the latest, so the candidate only ever sees one. What must hold is that
    # the FINAL question carries the whole scenario -- before the 2026-09-03 fix it carried
    # only the last sentence, having thrown "AWS" and "compliance" away.
    prompt = pipe._claude.prompts[-1]
    for fragment in ("AWS", "compliance", "GCP"):
        assert fragment in prompt, (
            f"{fragment!r} from the buildup never reached the model -- the scenario "
            f"context was dropped"
        )
    assert "architect this" in prompt, "the actual question was lost"


async def test_closing_ask_after_buildup_does_not_start_a_fresh_turn(make_pipeline):
    """"So what would you do?" ends in a bare demonstrative -- it has no subject of its own
    and must attach to the buildup rather than being answered alone."""
    overlay = Overlay()
    pipe = make_pipeline(overlay)

    await say(pipe, "Two services show correlated latency spikes.", 0.0, 6.0)
    await say(pipe, "There is no dependency between them.", 11.0, 17.0)
    await say(pipe, "So what would you do?", 22.0, 25.0)

    prompt = pipe._claude.prompts[-1]
    assert "correlated latency" in prompt and "no dependency" in prompt, (
        "the closing ask was answered without the scenario it depends on"
    )
