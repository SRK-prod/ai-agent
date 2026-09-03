"""The deterministic bullet-density backstop (answer_optimizer._tighten_bullets).

Five rounds of prompt tightening failed to hold bullets to a glanceable length -- real
Claude output kept returning 20-30 word bullets with inline CLI commands. Live feedback was
that the overlay must carry short memory triggers while the candidate explains out loud, so
the guarantee moved into code. These tests pin the behaviour that guarantee depends on.
"""

from __future__ import annotations

from meeting_copilot.llm.answer_optimizer import _tighten_bullets


def test_cuts_at_semicolon_and_drops_inline_command():
    text = (
        "* **Task IAM role** -- permissions denied on Secrets Manager, S3, or downstream "
        "services; run `aws sts get-caller-identity` from inside the task"
    )
    out = _tighten_bullets(text)
    assert "aws sts get-caller-identity" not in out, "inline command survived"
    assert ";" not in out, "the appended second thought survived"
    assert out.startswith("* **Task IAM role** -- permissions denied")


def test_leaves_an_already_short_bullet_untouched():
    text = "* IAM -- task/execution role and permission failures"
    assert _tighten_bullets(text) == text


def test_cuts_long_tail_at_a_comma_boundary_not_mid_phrase():
    text = (
        "* **Recent deploy** -- changed task definition, container image, environment "
        "variables, or infrastructure in the last 24 hours"
    )
    out = _tighten_bullets(text).strip()
    tail = out.split(" -- ", 1)[1]
    assert len(tail.split()) <= 9
    assert not tail.endswith(","), "cut left a dangling comma"
    # A comma-boundary cut keeps whole list items rather than severing one.
    assert tail == "changed task definition, container image, environment variables"


def test_never_touches_fenced_code_or_headings():
    text = (
        "## Troubleshooting\n"
        "```\n"
        "* this line is inside a fence and is deliberately very long indeed, way past any "
        "budget at all\n"
        "```\n"
        "Deploy -> task starts -> health check fails -> circuit breaker\n"
    )
    assert _tighten_bullets(text) == text


def test_leaves_a_bullet_with_no_label_structure_alone():
    """No 'LABEL --' split means there is no safe place to cut, so don't guess."""
    text = "* a plain bullet with no label separator that runs on for quite a long while here"
    assert _tighten_bullets(text) == text


def test_preserves_bullet_marker_and_indentation():
    text = "  - **Network** -- security group blocks outbound to database or API; check rules"
    out = _tighten_bullets(text)
    assert out.startswith("  - **Network** -- ")


def test_optimizer_applies_tightening_for_a_sweep_category():
    from meeting_copilot.llm.answer_optimizer import AnswerOptimizer
    from meeting_copilot.pipeline.events import DetectedQuestion, RetrievedContext, Transcript

    question = "An ECS deployment is intermittently failing. How would you troubleshoot it?"
    transcript = Transcript(speaker_id="a", text=question, start_time=0.0, end_time=1.0)
    context = RetrievedContext(question=DetectedQuestion(transcript=transcript), chunks=[])
    raw = (
        "## Troubleshooting\n"
        "* **Container image** -- pull failing from ECR; check the image digest exists and "
        "the execution role has ecr:BatchGetImage\n"
    )
    answer = AnswerOptimizer().optimize(context, raw)
    assert ";" not in answer.text
    assert "ecr:BatchGetImage" not in answer.text
