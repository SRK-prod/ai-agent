"""Post-process a raw Claude completion into a formatted, confidence-gated Answer."""

from __future__ import annotations

import re

from meeting_copilot.config import LlmConfig, get_config
from meeting_copilot.llm.prompt_templates import CONFIDENCE_MARKER, _classify_category
from meeting_copilot.pipeline.events import Answer, DetectedQuestion, RetrievedContext

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_DEFINITION_MAX_SENTENCES = 3
_DEFINITION_MAX_WORDS = 70
_BULLET_PREFIXES = ("*", "-", "•")


def _cap_definition_length(text: str) -> str:
    """Deterministic backstop for the `definition` category's short keyword block. Prompt
    instructions alone plateaued at 104-127 words even after two rounds of tightening
    (measured 2026-08-25) -- a live interview definition answer needs a hard guarantee, not
    a best-effort one, the same reasoning as the clarification backstop. No-op if the answer
    is already within budget (the normal case).

    Drops whole BULLETS from the end, since 2026-09-02 the category emits 4-6 keyword
    bullets rather than three flowing sentences. The previous version split on sentences and
    rejoined with " ", which would have flattened a bulleted answer onto a single unreadable
    line -- the exact opposite of what the overlay needs. Falls back to the sentence logic
    for unbulleted text, which the model can still occasionally produce."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    bullets = [ln for ln in lines if ln.strip()[:1] in _BULLET_PREFIXES]

    # Bulleted answer (the expected shape): drop trailing bullets until within budget.
    if len(bullets) >= 2:
        for n in range(len(lines), 0, -1):
            candidate = "\n".join(lines[:n]).rstrip()
            if len(candidate.split()) <= _DEFINITION_MAX_WORDS:
                return candidate
        return "\n".join(lines[:1]).rstrip()

    # Unbulleted fallback -- drop whole sentences rather than truncating mid-clause.
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    for n in range(min(_DEFINITION_MAX_SENTENCES, len(sentences)), 0, -1):
        candidate = " ".join(sentences[:n]).strip()
        if len(candidate.split()) <= _DEFINITION_MAX_WORDS:
            return candidate
    # Even one sentence is over budget (rare) -- word-truncate as a last resort.
    words = sentences[0].split() if sentences else text.split()
    return " ".join(words[:_DEFINITION_MAX_WORDS]).rstrip(",;:") + "."


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


def _reorganize_architecture_for_pacing(text: str) -> str:
    """For complex architecture answers (typically 500+ words), reorganize to put key
    decisions upfront and trim secondary detail. Target 300-400 words for interview pacing.

    Preserves: Brief Context, Components/Agent Responsibilities, Automation Model, Failure
    Handling, Trade-offs, Principal Architect Decision. Trims: Architecture Flow diagram
    (optional -- can describe the flow in the key-decision bullets instead), Tool Layer
    (usually detail), Knowledge/State (usually optional), Debugging/Observability (usually
    optional for pure-architecture questions).
    """
    lines = text.splitlines(keepends=True)
    if len(text.split()) <= 500:
        return text  # not oversized, leave as-is

    sections = {}
    current_section = None
    current_content = []

    # Parse sections by heading
    for line in lines:
        if line.strip().startswith("##"):
            if current_section:
                sections[current_section] = "".join(current_content).strip()
            current_section = line.strip()
            current_content = []
        else:
            current_content.append(line)
    if current_section:
        sections[current_section] = "".join(current_content).strip()

    # Preserve only the critical sections, reorder for pacing
    priority_order = [
        "## Brief Context",
        "## Architecture Components",
        "## Agent Responsibilities",  # variant name for agentic questions
        "## Components",
        "## Automation Model",
        "## Failure Handling",
        "## Trade-offs",
        "## Principal Architect Decision",
        "## Why This Design?",  # less common variant
    ]

    result_sections = []
    for heading in priority_order:
        if heading in sections:
            result_sections.append((heading, sections[heading]))

    # Reassemble with all preserved sections
    result = []
    for heading, content in result_sections:
        result.append(f"{heading}\n{content}\n")

    result_text = "".join(result).strip()
    word_count = len(result_text.split())

    # If still oversized, trim secondary bullets from non-critical sections
    if word_count > 400:
        # Trim elaboration from most sections, keeping the core (first 1-2 bullets per section)
        trimmed_sections = []
        for heading, content in result_sections:
            if heading in ["## Principal Architect Decision"]:
                # Never trim the closing decision statement
                trimmed_sections.append((heading, content))
            elif heading in ["## Brief Context"]:
                # Keep Brief Context intact (usually short)
                trimmed_sections.append((heading, content))
            else:
                # For other sections: keep only the first 2-3 bullets if there are many
                lines = content.split("\n")
                bullet_count = sum(1 for l in lines if l.strip().startswith(("*", "-")))
                if bullet_count > 3:
                    # Keep the section heading and first few bullets
                    kept_lines = []
                    bullet_kept = 0
                    for line in lines:
                        if line.strip().startswith(("*", "-")):
                            if bullet_kept < 3:
                                kept_lines.append(line)
                                bullet_kept += 1
                        elif line.strip() or bullet_kept == 0:  # keep non-bullets at start
                            kept_lines.append(line)
                    trimmed_sections.append((heading, "\n".join(kept_lines).strip()))
                else:
                    trimmed_sections.append((heading, content))

        result = []
        for heading, content in trimmed_sections:
            result.append(f"{heading}\n{content}\n")
        result_text = "".join(result).strip()

    return result_text


class AnswerOptimizer:
    def __init__(self, config: LlmConfig | None = None):
        self._cfg = config or get_config().llm

    def optimize(self, context: RetrievedContext, raw_text: str) -> Answer:
        text, confidence = parse_confidence(raw_text)
        category = _classify_category(context.question.transcript.text)
        if category == "definition":
            text = _cap_definition_length(text)
        elif category == "architecture" and len(text.split()) > 500:
            text = _reorganize_architecture_for_pacing(text)
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
