import pytest

from meeting_copilot.config import LlmConfig
from meeting_copilot.llm.prompt_templates import (
    CONFIDENCE_MARKER,
    _classify_category,
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


# Routing regression suite from the 20-question audit run 2026-09-02. A question that lands
# in the wrong category gets the wrong shape and the wrong word budget no matter how good the
# model is, so these are worth pinning. The secrets cases specifically guard the
# security-vs-cicd_devops split, which is decided by a keyword guard that is easy to break.
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # Cloud-agnostic secret handling -> security. Before 2026-09-02 these matched nothing
        # and fell through to `default`, a 110-word generic answer for a core topic.
        ("How do you manage secrets across multiple clouds?", "security"),
        ("What is your approach to secret management?", "security"),
        ("How would you rotate secrets across environments?", "security"),
        ("How do you store secrets for Kubernetes workloads?", "security"),
        # ...but a named delivery tool, or "artifacts", keeps it in the pipeline shape.
        ("How would you handle secrets in GitHub Actions?", "cicd_devops"),
        ("How do you manage secrets and artifacts across clouds?", "cicd_devops"),
        ("How do you handle secrets in a Terraform pipeline?", "iac_terraform"),
        # Spot checks across the rest of the interview surface.
        ("Can you explain where you have used GitHub Actions?", "cicd_devops"),
        ("How are you going to design the terraform for multi cloud systems?", "iac_terraform"),
        ("How would you migrate 500 applications from EC2 to EKS?", "migration"),
        ("Blue-green vs canary deployment - which would you choose?", "trade_off"),
        ("How would you establish observability standards across services?", "observability"),
        ("How do you prevent 50 teams from building 50 different pipelines?", "platform_engineering"),
        ("How would you reduce our AWS bill?", "cost_finops"),
        ("Why not just use CloudFormation instead of Terraform?", "why_not"),
        ("What is OpenTelemetry?", "definition"),
        ("Tell me about yourself", "career_narrative"),
        ("Tell me about a time you disagreed with a senior stakeholder", "behavioral"),
    ],
)
def test_question_routes_to_expected_category(question: str, expected: str):
    assert _classify_category(question) == expected
