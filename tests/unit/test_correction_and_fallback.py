"""Stage 1 reliability behaviour: correction detection and model fallback.

Both were identified in the 2026-08-27 end-to-end audit. Corrections previously merged
("Would you use ECS?" + "Actually, I mean Kubernetes" -> an answer covering both), and a
single model with no fallback was a single point of failure for the whole app.
"""

from __future__ import annotations

import pytest

from meeting_copilot.llm.claude_client import _stream_with_fallback
from meeting_copilot.pipeline.orchestrator import _is_correction

# --- correction detection ---------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Actually, I mean Kubernetes",
        "Sorry, I meant EKS not ECS",
        "Let me rephrase that",
        "No, I was asking about the database",
        "What I mean is, how would you scale it",
        "To clarify, I'm asking about the control plane",
    ],
)
def test_correction_phrases_are_detected(text):
    assert _is_correction(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "How would you design a highly available system?",
        "What is Terraform?",
        "How do you handle database failover?",
    ],
)
def test_ordinary_questions_are_not_corrections(text):
    assert _is_correction(text) is False


def test_correction_marker_deep_in_a_real_question_does_not_fire():
    # The head-only check exists for this case: these words appear mid-sentence in ordinary
    # speech, and treating that as a correction would discard a genuine question.
    assert (
        _is_correction(
            "How would you design the platform so that the trade-off I mean here is cost-bounded?"
        )
        is False
    )


# --- model fallback ---------------------------------------------------------------


def _stream_factory(behaviour: dict[str, list[str] | Exception]):
    """Build a fake _raw_stream whose behaviour is keyed by model name."""

    async def _raw_stream(prompt, system, model=None):
        outcome = behaviour[model]
        if isinstance(outcome, Exception):
            raise outcome
        for chunk in outcome:
            yield chunk

    return _raw_stream


async def _collect(agen):
    return [chunk async for chunk in agen]


async def test_primary_model_success_never_reaches_fallback():
    stream = _stream_factory({"primary": ["hello ", "world"], "backup": RuntimeError("unused")})
    out = await _collect(_stream_with_fallback(stream, ["primary", "backup"], "q", None))
    assert out == ["hello ", "world"]


async def test_falls_back_when_primary_produces_no_output():
    stream = _stream_factory(
        {"primary": RuntimeError("provider down"), "backup": ["fallback ", "answer"]}
    )
    out = await _collect(_stream_with_fallback(stream, ["primary", "backup"], "q", None))
    assert out == ["fallback ", "answer"]


async def test_raises_when_every_model_fails():
    stream = _stream_factory(
        {"primary": RuntimeError("down"), "backup": RuntimeError("also down")}
    )
    with pytest.raises(RuntimeError):
        await _collect(_stream_with_fallback(stream, ["primary", "backup"], "q", None))


async def test_no_fallback_configured_still_works():
    stream = _stream_factory({"primary": ["only ", "model"]})
    out = await _collect(_stream_with_fallback(stream, ["primary"], "q", None))
    assert out == ["only ", "model"]
