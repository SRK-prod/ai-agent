from meeting_copilot.config import LlmConfig
from meeting_copilot.llm.answer_optimizer import (
    AnswerOptimizer,
    detect_format_type,
    parse_confidence,
)
from meeting_copilot.pipeline.events import DetectedQuestion, RetrievedContext, Transcript


def _question(text: str = "What's the tradeoff here?") -> DetectedQuestion:
    transcript = Transcript(speaker_id="speaker_a", text=text, start_time=0.0, end_time=1.0)
    return DetectedQuestion(transcript=transcript, matched_keywords=[], ends_with_question_mark=True)


def test_parse_confidence_extracts_trailing_marker():
    raw = "Use option B here because of latency.\nCONFIDENCE: 72"
    text, confidence = parse_confidence(raw)
    assert text == "Use option B here because of latency."
    assert confidence == 0.72


def test_parse_confidence_defaults_when_marker_missing():
    text, confidence = parse_confidence("Just an answer, no marker.", default=1.0)
    assert text == "Just an answer, no marker."
    assert confidence == 1.0


def test_detect_format_type_code():
    assert detect_format_type(_question(), "```python\nprint(1)\n```") == "code"


def test_detect_format_type_table():
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert detect_format_type(_question(), table) == "table"


def test_detect_format_type_bullets():
    bullets = "- point one\n- point two\n- point three"
    assert detect_format_type(_question(), bullets) == "bullets"


def test_detect_format_type_prose_fallback():
    assert detect_format_type(_question(), "Just a plain sentence answer.") == "prose"


def test_optimizer_flags_low_confidence():
    optimizer = AnswerOptimizer(LlmConfig(low_confidence_threshold=0.80))
    context = RetrievedContext(question=_question(), chunks=[])
    answer = optimizer.optimize(context, "Not sure about this one.\nCONFIDENCE: 40")
    assert answer.low_confidence is True
    assert answer.confidence == 0.40
    assert "[Low Confidence]" in answer.text


def test_optimizer_does_not_flag_high_confidence():
    optimizer = AnswerOptimizer(LlmConfig(low_confidence_threshold=0.80))
    context = RetrievedContext(question=_question(), chunks=[])
    answer = optimizer.optimize(context, "Confident answer.\nCONFIDENCE: 95")
    assert answer.low_confidence is False
    assert "[Low Confidence]" not in answer.text
