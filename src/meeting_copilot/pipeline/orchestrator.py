"""Wires every stage into the live pipeline:

  audio capture -> VAD -> speaker diarization (ignore me) -> STT -> question
  detector -> hybrid retrieval -> Claude -> answer optimizer -> on_answer callback

VAD segmentation always keeps consuming the live audio stream -- detected segments are
queued, never blocked on a slow LLM call. But segments are then handled ONE AT A TIME by a
single worker, strictly in the order they were spoken: if two questions land close together
(a fast follow-up before the first answer finishes), concurrent handling previously let both
write to the same single-answer overlay at once, silently clobbering whichever one lost the
race. Serial processing trades a few extra seconds of queue wait (rare -- answers take ~10s,
real questions are rarely that close together) for a guarantee that no answer is ever lost or
overwritten.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from meeting_copilot.audio.capture import AudioCapture
from meeting_copilot.cache.redis_cache import RedisCache
from meeting_copilot.config import get_config
from meeting_copilot.knowledge.embeddings import LocalEmbedder
from meeting_copilot.llm.answer_optimizer import AnswerOptimizer
from meeting_copilot.llm.claude_client import ClaudeClient
from meeting_copilot.llm.prompt_templates import build_system_prompt, build_user_prompt
from meeting_copilot.nlp.question_detector import get_question_detector
from meeting_copilot.pipeline.events import Answer, RetrievedContext, SpeechSegment
from meeting_copilot.pipeline.metrics import PIPELINE_TOTAL_LATENCY_SECONDS, StageTimer
from meeting_copilot.retrieval.hybrid_search import HybridSearcher
from meeting_copilot.retrieval.qa_bank import QaBankStore
from meeting_copilot.speaker.diarization import SpeakerDiarizer
from meeting_copilot.stt.faster_whisper_engine import SttStage
from meeting_copilot.utils.logging import get_logger
from meeting_copilot.vad.silero_vad import SileroVAD

logger = get_logger()

OnAnswer = Callable[[Answer], Awaitable[None]]
OnPartialAnswer = Callable[[str], Awaitable[None]]


class MeetingPipeline:
    def __init__(
        self,
        on_answer: OnAnswer | None = None,
        on_partial_answer: OnPartialAnswer | None = None,
    ):
        """on_answer: async callable(Answer) -> None, e.g. push over the overlay WebSocket.
        on_partial_answer: async callable(text_so_far) -> None, called as the answer streams in
        (skipped on a cache hit, since that's already instant)."""
        self._cfg = get_config()
        self._capture = AudioCapture()
        self._vad = SileroVAD()
        self._diarizer = SpeakerDiarizer()
        self._stt = SttStage()
        self._question_detector = get_question_detector()
        # Pure-LLM mode (default): both disabled, so skip loading the embedding model
        # entirely -- see configs/settings.yaml retrieval.enabled for why.
        needs_embedder = self._cfg.retrieval.enabled or self._cfg.qa_bank.enabled
        self._embedder = LocalEmbedder() if needs_embedder else None
        self._retriever = (
            HybridSearcher(embedder=self._embedder) if self._cfg.retrieval.enabled else None
        )
        self._qa_bank = QaBankStore() if self._cfg.qa_bank.enabled else None
        self._claude = ClaudeClient()
        self._optimizer = AnswerOptimizer()
        self._cache = RedisCache()
        self._on_answer = on_answer
        self._on_partial_answer = on_partial_answer

        self._running = False
        self._run_task: asyncio.Task | None = None
        self._worker_task: asyncio.Task | None = None
        self._segment_queue: asyncio.Queue[SpeechSegment] = asyncio.Queue()

    async def _handle_segment(self, segment: SpeechSegment) -> None:
        started_at = time.monotonic()
        try:
            with StageTimer("speaker_id"):
                diarized = await asyncio.to_thread(self._diarizer.diarize, segment)
            if diarized.is_me:
                return

            with StageTimer("stt"):
                transcript = await self._stt.transcribe(diarized)
            if transcript is None:
                return

            question = self._question_detector.detect(transcript)
            if question is None:
                logger.debug(f"Not a question, skipping: {transcript.text!r}")
                return

            # Flip the overlay to "answering..." with the heard question right away --
            # visible feedback within ~a second of speech ending, well before the
            # retrieval+LLM answer starts streaming in over it.
            if self._on_partial_answer:
                await self._on_partial_answer(f"*Q: {question.transcript.text}*")

            # Pre-generated Q&A bank: a close-enough banked question serves its stored
            # answer instantly -- no retrieval, no LLM call. (disabled by default; see
            # configs/settings.yaml qa_bank.enabled)
            if self._qa_bank is not None:
                assert self._embedder is not None
                with StageTimer("qa_bank"):
                    vector = await self._embedder.embed(question.transcript.text)
                    banked = self._qa_bank.lookup(vector)
                if banked is not None:
                    logger.info(
                        f"QA-bank hit (score={banked.score:.2f}) asked={question.transcript.text!r} "
                        f"-> matched={banked.question!r}"
                    )
                    answer = self._optimizer.optimize(
                        RetrievedContext(question=question, chunks=[]), banked.answer
                    )
                    PIPELINE_TOTAL_LATENCY_SECONDS.observe(time.monotonic() - started_at)
                    if self._on_answer:
                        await self._on_answer(answer)
                    return

            if self._retriever is not None:
                with StageTimer("retrieval"):
                    context = await self._retriever.retrieve(question)
            else:
                context = RetrievedContext(question=question, chunks=[])

            chunk_texts = [c.text for c in context.chunks]
            cached_text = await self._cache.get_llm_response(question.transcript.text, chunk_texts)
            if cached_text is not None:
                raw_text = cached_text
            else:
                raw_text = ""
                with StageTimer("llm"):
                    system_prompt = build_system_prompt(question_text=question.transcript.text)
                    async for delta in self._claude.stream(
                        build_user_prompt(context), system=system_prompt
                    ):
                        raw_text += delta
                        if self._on_partial_answer:
                            await self._on_partial_answer(raw_text)
                await self._cache.set_llm_response(question.transcript.text, chunk_texts, raw_text)

            answer = self._optimizer.optimize(context, raw_text)
            PIPELINE_TOTAL_LATENCY_SECONDS.observe(time.monotonic() - started_at)

            if self._on_answer:
                await self._on_answer(answer)
        except Exception:
            logger.exception("Error handling speech segment")

    async def _consume_queue(self) -> None:
        """Single worker: handles exactly one segment at a time, strictly in spoken order,
        so overlay updates from different questions can never race each other."""
        while True:
            segment = await self._segment_queue.get()
            try:
                queued_behind = self._segment_queue.qsize()
                if queued_behind:
                    logger.info(f"Handling segment with {queued_behind} more already queued")
                await self._handle_segment(segment)
            finally:
                self._segment_queue.task_done()

    async def run(self) -> None:
        self._running = True
        logger.info("Meeting pipeline started")
        self._worker_task = asyncio.create_task(self._consume_queue())
        async for segment in self._vad.segments(self._capture.frames()):
            if not self._running:
                break
            # Non-blocking: VAD keeps segmenting live audio even if the worker is still
            # busy on a previous segment -- nothing is ever dropped, only queued.
            self._segment_queue.put_nowait(segment)
        logger.info("Meeting pipeline stopped")

    def start(self) -> None:
        self._run_task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._running = False
        if self._run_task:
            self._run_task.cancel()
        if self._worker_task:
            self._worker_task.cancel()
        await self._cache.close()
