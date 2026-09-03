"""Colloquial US interview phrasing must reach the right answer shape.

Today's interview is with US interviewers, who ask questions in ways the classifier's
formal openers missed: "run me through...", "gimme an example of something you
automated...", "let's say the app can't reach RDS...". Detection already handled these; the
ROUTING did not, so two textbook STAR questions were being answered with the generic
`default` shape and a 110-word cap.

The guard cases matter as much as the positive ones: the colloquial rules are broad, and a
rule that steals a design question is worse than one that misses a story question.
"""

from __future__ import annotations

import pytest

from meeting_copilot.llm.prompt_templates import _classify_category
from meeting_copilot.nlp.question_detector import RuleBasedQuestionDetector
from meeting_copilot.pipeline.events import Transcript


@pytest.mark.parametrize(
    "question,expected",
    [
        # Past experience, colloquially asked -- all STAR.
        ("Take me through a time you fixed a nasty IAM issue", "behavioral"),
        ("Tell us about a time you led a migration", "behavioral"),
        ("Gimme an example of something you automated with Python", "behavioral"),
        ("Anything you've built with Python for ops?", "behavioral"),
        # Scenario handed over without "how would you".
        ("Let's say the app can't reach RDS after a security group change",
         "scenario_troubleshooting"),
        ("Suppose a Harness stage fails on deploy, what then", "scenario_troubleshooting"),
        ("I'm curious how you'd handle a TLS handshake failure", "scenario_troubleshooting"),
        ("Run me through how you'd debug an ECS deploy that keeps failing",
         "scenario_troubleshooting"),
        # Tool questions -- for ECS and Harness these are honest-gap answers.
        ("Talk to me about your experience with Harness", "tool_technology"),
        ("Have you used Harness before?", "tool_technology"),
        ("Are you familiar with ECS?", "tool_technology"),
    ],
)
def test_colloquial_phrasing_routes_correctly(question, expected):
    assert _classify_category(question) == expected


@pytest.mark.parametrize(
    "question,expected",
    [
        # A domain noun inside a STAR question must not steal it (this one regressed:
        # "migration" returns far earlier in the classifier than the STAR openers did).
        ("Tell us about a time you led a migration", "behavioral"),
        # ...but a real migration design question still routes to migration.
        ("How would you migrate 500 applications from EC2 to ECS?", "migration"),
        # "Suppose" alone is not a failure scenario.
        ("Suppose you have 50 teams, how do you standardize pipelines?",
         "platform_engineering"),
        ("How would you design a highly available ECS platform?", "ha_dr"),
        ("What is a security group?", "definition"),
        ("ECS vs EKS?", "trade_off"),
        ("Tell me about yourself", "career_narrative"),
        ("Production is down. What do you do?", "incident_rca"),
    ],
)
def test_colloquial_rules_do_not_steal_other_shapes(question, expected):
    assert _classify_category(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Run me through how you'd debug an ECS deploy that keeps failing",
        "Talk to me about your experience with Harness",
        "Take me through a time you fixed a nasty IAM issue",
        "Let's say the app can't reach RDS after a security group change",
        "I'm curious how you'd handle a TLS handshake failure",
        "Gimme an example of something you automated with Python",
        "Suppose a Harness stage fails on deploy, what then",
        "Any thoughts on how to cut down repeat tickets",
    ],
)
def test_colloquial_phrasing_is_detected_as_a_question(question):
    """None of these carry a formal interrogative opener, and conversational speech often
    loses the trailing '?' in transcription -- so the opener list has to cover them."""
    transcript = Transcript(speaker_id="i", text=question, start_time=0.0, end_time=2.0)
    assert RuleBasedQuestionDetector().detect(transcript) is not None


def test_source_has_no_mangled_regex_escapes():
    """A shell heredoc once collapsed `\\b` into a literal backspace inside these regexes,
    producing patterns that silently never matched. Cheap permanent guard."""
    import re
    from pathlib import Path

    import meeting_copilot.llm.prompt_templates as pt

    source = Path(pt.__file__).read_text(encoding="utf-8")
    bad = re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", source)
    assert not bad, f"control characters in prompt_templates.py: {[hex(ord(c)) for c in bad]}"
