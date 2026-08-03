"""Post-process a raw Claude completion into a formatted, confidence-gated Answer."""

from __future__ import annotations

import re

from meeting_copilot.config import LlmConfig, get_config
from meeting_copilot.llm.prompt_templates import CONFIDENCE_MARKER
from meeting_copilot.pipeline.events import Answer, DetectedQuestion, RetrievedContext


def parse_confidence(raw_text: str, default: float = 1.0) -> tuple[str, float]:
    """Strips the trailing `CONFIDENCE: N` line Claude was asked to add, returns (text, 0-1 score)."""
    idx = raw_text.rfind(CONFIDENCE_MARKER)
    if idx == -1:
        return raw_text.strip(), default

    body = raw_text[:idx].strip()
    tail = raw_text[idx + len(CONFIDENCE_MARKER) :]
    match = re.search(r"\d+", tail)
    if not match:
        return body, default

    value = max(0, min(100, int(match.group())))
    return body, value / 100.0


def detect_format_type(question: DetectedQuestion, text: str) -> str:
    if "```" in text:
        return "code"
    if "|" in text and "-|-" in text.replace(" ", "").replace("--", "-"):
        return "table"
    bullet_lines = sum(1 for line in text.splitlines() if line.strip()[:1] in ("-", "*", "•"))
    if bullet_lines >= 2:
        return "bullets"
    return "prose"


class AnswerOptimizer:
    def __init__(self, config: LlmConfig | None = None):
        self._cfg = config or get_config().llm

    def optimize(self, context: RetrievedContext, raw_text: str) -> Answer:
        text, confidence = parse_confidence(raw_text)
        format_type = detect_format_type(context.question, text)
        low_confidence = confidence < self._cfg.low_confidence_threshold
        if low_confidence:
            text = f"[Low Confidence]\n{text}"

        return Answer(
            question=context.question,
            text=text,
            format_type=format_type,
            confidence=confidence,
            low_confidence=low_confidence,
        )
