"""Exercises question-detection -> retrieval -> prompt-building -> answer-optimization
as one slice, with fake retrieval/embedding stand-ins so no network/models are needed.
CI-safe (not marked slow): validates that the event contracts in
pipeline/events.py actually compose end-to-end.
"""

from __future__ import annotations

from meeting_copilot.config import LlmConfig, QuestionDetectorConfig, RetrievalConfig
from meeting_copilot.llm.answer_optimizer import AnswerOptimizer
from meeting_copilot.llm.prompt_templates import build_system_prompt, build_user_prompt
from meeting_copilot.nlp.question_detector import RuleBasedQuestionDetector
from meeting_copilot.pipeline.events import Transcript
from meeting_copilot.retrieval.hybrid_search import HybridSearcher
from meeting_copilot.retrieval.qdrant_store import ScoredChunk


class FakeQdrantStore:
    def __init__(self, hits: list[ScoredChunk]):
        self._hits = hits

    def search(self, vector, top_k, topic_filter=None):
        return self._hits[:top_k]


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0]  # semantic scoring is Qdrant's job in prod; irrelevant to this slice


async def test_question_to_answer_slice():
    detector = RuleBasedQuestionDetector(
        QuestionDetectorConfig(keywords=["kafka", "partition"], denylist_phrases=["good morning"])
    )
    transcript = Transcript(
        speaker_id="speaker_a",
        text="kafka partitions",
        start_time=0.0,
        end_time=2.0,
    )
    question = detector.detect(transcript)
    assert question is not None
    assert "kafka" in question.matched_keywords

    hits = [
        ScoredChunk(
            text="Kafka partitions distribute load across brokers",
            topic="platform",
            source="notes.md",
            chunk_id="1",
            score=0.4,
        ),
        ScoredChunk(
            text="Completely unrelated content about databases",
            topic="platform",
            source="other.md",
            chunk_id="2",
            score=0.9,
        ),
    ]
    # Full keyword overlap ("kafka partitions" matches notes.md exactly) combined with a
    # keyword-heavy alpha should outrank other.md's higher-but-irrelevant semantic score.
    searcher = HybridSearcher(
        qdrant=FakeQdrantStore(hits),
        embedder=FakeEmbedder(),
        config=RetrievalConfig(top_k_vector=20, top_k_final=2, hybrid_alpha=0.4),
    )
    context = await searcher.retrieve(question)
    assert len(context.chunks) == 2
    assert context.chunks[0].source == "notes.md"

    system_prompt = build_system_prompt(LlmConfig())
    user_prompt = build_user_prompt(context)
    assert "kafka partitions" in user_prompt.lower()
    assert "CONFIDENCE" in system_prompt

    raw_llm_response = "Use Kafka partitions keyed by customer ID for ordering.\nCONFIDENCE: 88"
    answer = AnswerOptimizer(LlmConfig(low_confidence_threshold=0.80)).optimize(
        context, raw_llm_response
    )

    assert answer.low_confidence is False
    assert answer.confidence == 0.88
    assert "Use Kafka partitions" in answer.text
