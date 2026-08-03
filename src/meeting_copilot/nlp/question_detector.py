"""Decide whether a Transcript is worth spending an LLM call on.

Only questions/requests/technical-discussion utterances should reach Claude;
greetings and small talk must not. This is a small ABC so the rule-based v1
implementation can later be swapped for a model-based classifier without
touching pipeline.orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from meeting_copilot.config import QuestionDetectorConfig, get_config
from meeting_copilot.pipeline.events import DetectedQuestion, Transcript


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
        ends_with_question_mark = text_lower.endswith("?")
        starts_interrogative = text_lower.startswith(_INTERROGATIVE_OPENERS)

        # Greetings/small talk are ignored unless they also carry a technical keyword
        # (e.g. "hey, quick question about kubernetes" should still trigger).
        if is_denylisted and not matched_keywords:
            return None

        if not ends_with_question_mark and not matched_keywords and not starts_interrogative:
            return None

        # Without a keyword match, very short utterances are almost always filler
        # ("Okay, no?", "Right?", a bare "Why") rather than an answerable question --
        # and with no conversation history in the prompt, the LLM can't do anything
        # useful with them anyway. A keyword match always triggers regardless of length.
        if (
            not matched_keywords
            and len(text_lower.split()) < self._cfg.min_words_for_bare_question_mark
        ):
            return None

        return DetectedQuestion(
            transcript=transcript,
            matched_keywords=matched_keywords,
            ends_with_question_mark=ends_with_question_mark,
        )


def get_question_detector(config: QuestionDetectorConfig | None = None) -> QuestionDetector:
    cfg = config or get_config().question_detector
    if cfg.backend == "rule_based":
        return RuleBasedQuestionDetector(cfg)
    raise ValueError(f"Unknown question_detector.backend: {cfg.backend}")
