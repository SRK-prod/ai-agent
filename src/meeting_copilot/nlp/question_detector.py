"""Decide whether a Transcript is worth spending an LLM call on.

Only questions/requests/technical-discussion utterances should reach Claude;
greetings and small talk must not. This is a small ABC so the rule-based v1
implementation can later be swapped for a model-based classifier without
touching pipeline.orchestrator.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from meeting_copilot.config import QuestionDetectorConfig, get_config
from meeting_copilot.pipeline.events import DetectedQuestion, Transcript
from meeting_copilot.stt.term_normalizer import technical_term_count


class QuestionDetector(ABC):
    @abstractmethod
    def detect(self, transcript: Transcript) -> DetectedQuestion | None: ...


# Interview questions are frequently imperative/interrogative openings with no "?" in the
# transcript at all (STT drops punctuation, and "Tell me about..." has none to begin with).
_INTERROGATIVE_OPENERS = (
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
    "can you",
    "could you",
    "would you",
    "tell me",
    "tell us",
    "walk me through",
    "walk us through",
    "describe",
    "explain",
    "share an example",
    "give me an example",
    "give an example",
    "have you",
    "do you",
    "did you",
    "are you",
    "is this",
    "is that",
    "is it",
    "was this",
    "i want to understand",
    "i want to know",
    # Colloquial US interview phrasing, added 2026-09-03. American interviewers routinely
    # open a question with these and none of them contain a classic interrogative word, so
    # the question would only be caught by a trailing "?" -- which conversational speech
    # often loses in transcription.
    "run me through",
    "run us through",
    "talk me through",
    "talk us through",
    "talk to me about",
    "take me through",
    "take us through",
    "walk through",
    "gimme",
    "give us an example",
    "lemme ask",
    "let me ask",
    "i'm curious",
    "im curious",
    "curious how",
    "curious about",
    "any thoughts",
    "thoughts on",
    "your take on",
    "speak to",
    "get into",
    "dig into",
    # Scenario setups -- not interrogative, but they always precede the real ask, and
    # holding them as context is exactly what the scenario-buildup path is for.
    "let's say",
    "lets say",
    "say you",
    "suppose",
    "imagine",
    "picture this",
)


class RuleBasedQuestionDetector(QuestionDetector):
    def __init__(self, config: QuestionDetectorConfig | None = None):
        self._cfg = config or get_config().question_detector

    def detect(self, transcript: Transcript) -> DetectedQuestion | None:
        text_lower = transcript.text.lower().strip()
        if not text_lower:
            return None

        matched_keywords = [kw for kw in self._cfg.keywords if kw.lower() in text_lower]
        is_denylisted = any(phrase.lower() in text_lower for phrase in self._cfg.denylist_phrases)
        ends_with_question_mark = "?" in text_lower  # not just endswith -- STT often runs a
        # trailing clause on after the "?" ("...have you used it? You said...") and the
        # question mark can land mid-segment, not at the very end.
        # A real question is very often NOT the first word, or even the first clause, in
        # the segment -- interviewers preface it ("Anyway, we'll move on. How you use the
        # MCP...") or embed it naturally mid-sentence ("I want to understand how your agent
        # interacts...", "what component is interacting", "or what it does"). Measured live:
        # requiring the opener at a clause boundary still missed most real questions in an
        # actual interview. Missing a real question is far more costly than one extra,
        # unnecessary LLM call on a false positive, so search the whole segment for any
        # interrogative opener as a whole word/phrase, not just at a clause start.
        starts_interrogative = any(
            re.search(r"\b" + re.escape(opener) + r"\b", text_lower) for opener in _INTERROGATIVE_OPENERS
        )

        # TECHNICAL-DENSITY RESCUE. Whisper mangles domain vocabulary badly enough that a
        # real question can arrive grammatically broken with no clean interrogative opener
        # ("Okay, already stuck in, CrashLoopBackOff."). Rejecting those silently drops real
        # interview questions -- observed repeatedly in production. So: speech that is dense
        # in genuine technical terms is treated as a probable question even when the grammar
        # signals are absent. A false positive costs one LLM call; a false negative costs an
        # unanswered interview question.
        tech_terms = technical_term_count(transcript.text)

        # Greetings/small talk are ignored unless they also carry a technical keyword
        # (e.g. "hey, quick question about kubernetes" should still trigger).
        if is_denylisted and not matched_keywords and tech_terms < 2:
            return None

        if (
            not ends_with_question_mark
            and not matched_keywords
            and not starts_interrogative
            and tech_terms < 2
        ):
            return None

        # Without a keyword match, very short utterances are almost always filler
        # ("Okay, no?", "Right?", a bare "Why") rather than an answerable question --
        # and with no conversation history in the prompt, the LLM can't do anything
        # useful with them anyway. A keyword match always triggers regardless of length.
        # EXCEPTION: a short segment ending in "?" is a strong enough signal on its own --
        # these are exactly the rapid-fire one-line follow-ups a real interviewer asks
        # after a longer answer ("Why Sonnet?", "MCP?", "Trade-offs?"), and none of them
        # necessarily contain a classic interrogative opener word or a technical keyword.
        # Measured live: requiring one of those on top of the "?" was silently dropping
        # real follow-ups. A false positive here costs one unnecessary LLM call; a false
        # negative silently drops a real question mid-interview -- not a close trade-off.
        if (
            not matched_keywords
            and not ends_with_question_mark
            and tech_terms < 2
            and len(text_lower.split()) < self._cfg.min_words_for_bare_question_mark
        ):
            return None

        return DetectedQuestion(
            transcript=transcript,
            matched_keywords=matched_keywords,
            ends_with_question_mark=ends_with_question_mark,
            has_interrogative_signal=ends_with_question_mark or starts_interrogative,
        )


def get_question_detector(config: QuestionDetectorConfig | None = None) -> QuestionDetector:
    cfg = config or get_config().question_detector
    if cfg.backend == "rule_based":
        return RuleBasedQuestionDetector(cfg)
    raise ValueError(f"Unknown question_detector.backend: {cfg.backend}")
