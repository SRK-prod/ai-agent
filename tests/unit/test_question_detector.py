from meeting_copilot.config import QuestionDetectorConfig
from meeting_copilot.nlp.question_detector import RuleBasedQuestionDetector
from meeting_copilot.pipeline.events import Transcript

CFG = QuestionDetectorConfig(
    denylist_phrases=["how are you", "good morning"],
    keywords=["kubernetes", "terraform", "root cause"],
)


def _transcript(text: str) -> Transcript:
    return Transcript(speaker_id="speaker_a", text=text, start_time=0.0, end_time=1.0)


def test_greeting_is_ignored():
    detector = RuleBasedQuestionDetector(CFG)
    assert detector.detect(_transcript("Good morning everyone")) is None


def test_plain_question_mark_triggers():
    detector = RuleBasedQuestionDetector(CFG)
    result = detector.detect(_transcript("What time is it?"))
    assert result is not None
    assert result.ends_with_question_mark is True


def test_keyword_without_question_mark_triggers():
    detector = RuleBasedQuestionDetector(CFG)
    result = detector.detect(_transcript("We should talk about terraform state locking"))
    assert result is not None
    assert "terraform" in result.matched_keywords


def test_greeting_with_keyword_still_triggers():
    detector = RuleBasedQuestionDetector(CFG)
    result = detector.detect(_transcript("Good morning, quick kubernetes question"))
    assert result is not None
    assert "kubernetes" in result.matched_keywords


def test_small_talk_without_keyword_or_question_mark_is_ignored():
    detector = RuleBasedQuestionDetector(CFG)
    assert detector.detect(_transcript("Thanks everyone, sounds good")) is None


def test_empty_text_is_ignored():
    detector = RuleBasedQuestionDetector(CFG)
    assert detector.detect(_transcript("   ")) is None


def test_short_filler_with_bare_question_mark_is_filtered():
    """Short filler ("Okay, no?") must never produce an answer, but short REAL follow-ups
    ("Why Sonnet?", "Trade-offs?") must.

    This is deliberately a two-layer guarantee, and the test was updated 2026-09-01 to
    match where each layer actually lives. The detector used to reject every sub-N-word
    utterance, which silently dropped the rapid-fire one-liners a real interviewer asks
    after a long answer -- see the "EXCEPTION" comment in question_detector.detect(). So
    the detector now admits anything ending in "?", and the orchestrator's
    _is_acknowledgement_only strips the pure-filler ones before any LLM call. Asserting
    `detector.detect(...) is None` here tested the old single-layer design and had been
    failing ever since; what actually matters is that filler is never ANSWERED.
    """
    from meeting_copilot.pipeline.orchestrator import _is_acknowledgement_only

    detector = RuleBasedQuestionDetector(CFG)

    def would_be_answered(text: str) -> bool:
        return detector.detect(_transcript(text)) is not None and not _is_acknowledgement_only(text)

    for filler in ("Okay, no?", "Right?", "Yeah, sure?", "Okay."):
        assert not would_be_answered(filler), f"filler reached the LLM: {filler!r}"

    for real in ("Why Sonnet?", "Trade-offs?", "MCP?"):
        assert would_be_answered(real), f"real one-line follow-up was dropped: {real!r}"


def test_short_question_with_keyword_still_triggers():
    detector = RuleBasedQuestionDetector(CFG)
    result = detector.detect(_transcript("terraform?"))
    assert result is not None
    assert "terraform" in result.matched_keywords


def test_min_words_threshold_is_configurable():
    lenient_cfg = QuestionDetectorConfig(
        denylist_phrases=[], keywords=[], min_words_for_bare_question_mark=1
    )
    detector = RuleBasedQuestionDetector(lenient_cfg)
    assert detector.detect(_transcript("Right?")) is not None


def test_interview_phrasing_without_question_mark_triggers():
    # "Tell me about..." / "Walk me through..." style questions carry no "?" and
    # may match no tech keyword, but must still trigger.
    detector = RuleBasedQuestionDetector(CFG)
    assert detector.detect(_transcript("Tell me about a time you handled a major outage")) is not None
    assert detector.detect(_transcript("Walk me through your deployment process")) is not None
    assert detector.detect(_transcript("Describe your experience leading a migration")) is not None


def test_interrogative_opener_still_filtered_when_too_short():
    detector = RuleBasedQuestionDetector(CFG)
    assert detector.detect(_transcript("How so")) is None


def test_plain_statement_still_ignored():
    detector = RuleBasedQuestionDetector(CFG)
    assert detector.detect(_transcript("Yeah that all makes sense to me, sounds good")) is None
