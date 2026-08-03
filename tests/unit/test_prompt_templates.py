from meeting_copilot.config import LlmConfig
from meeting_copilot.llm.prompt_templates import (
    CONFIDENCE_MARKER,
    build_system_prompt,
    build_user_prompt,
)
from meeting_copilot.pipeline.events import (
    DetectedQuestion,
    RetrievedChunk,
    RetrievedContext,
    Transcript,
)


def test_system_prompt_includes_persona_and_confidence_instruction():
    cfg = LlmConfig(persona="You are a Staff Engineer.")
    prompt = build_system_prompt(cfg)
    assert "You are a Staff Engineer." in prompt
    assert CONFIDENCE_MARKER in prompt


def test_user_prompt_includes_question_and_chunks():
    transcript = Transcript(speaker_id="speaker_a", text="How should we scale this?", start_time=0, end_time=1)
    question = DetectedQuestion(transcript=transcript, matched_keywords=["scale"], ends_with_question_mark=True)
    chunks = [RetrievedChunk(text="Use horizontal autoscaling.", source="notes.md", topic="aws", score=0.9)]
    context = RetrievedContext(question=question, chunks=chunks)

    prompt = build_user_prompt(context)

    assert "How should we scale this?" in prompt
    assert "Use horizontal autoscaling." in prompt
    assert "aws" in prompt


def test_user_prompt_handles_no_retrieved_chunks():
    # Pure-LLM mode (retrieval.enabled=false, the default): no reference-notes section at
    # all, just the question -- Claude answers from its own expertise.
    transcript = Transcript(speaker_id="speaker_a", text="What's a kafka partition?", start_time=0, end_time=1)
    question = DetectedQuestion(transcript=transcript, matched_keywords=["kafka"], ends_with_question_mark=True)
    context = RetrievedContext(question=question, chunks=[])

    prompt = build_user_prompt(context)

    assert "What's a kafka partition?" in prompt
    assert "Reference notes" not in prompt
