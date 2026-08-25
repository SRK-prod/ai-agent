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
import re
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
from meeting_copilot.pipeline.events import Answer, RetrievedContext, SpeechSegment, Transcript
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

# VAD's min_silence_ms gap can split one continuous, paused-and-continued question into
# multiple segments. If the next segment from the SAME speaker arrives within this many
# seconds of the previous one ending, treat it as a continuation and merge the text
# instead of answering a fragment out of context.
_FRAGMENT_MERGE_GAP_SECONDS = 3.0

# ANSWER REVISION WINDOW (redesigned 2026-08-25, replacing the earlier "wait before
# answering" compound-question approach). Every detected question is answered immediately --
# no upfront wait beyond the short fragment debounce below. The trade-off moves to AFTER
# generation instead of before it: once an answer is shown, it stays "open for revision" for
# this many seconds. If another legitimate interviewer question arrives inside that window
# (an explicit pivot like "forget that"/"let's move to" always excepted -- see
# _NEW_QUESTION_OPENERS), it is folded into the SAME question set, the combined question is
# re-answered, and the new answer REPLACES the one on the overlay -- the interviewer never
# sees two separate write-ups for what was really one connected exchange. This covers both
# a quick back-to-back follow-up ("What is Terraform?" [3s] "How should you have integrated
# AI?") and a genuine thinking-pause addition ("tell me about Terraform" [~30s pause] "how
# can you integrate it with Agentic AI") with one mechanism, because both are just "a new
# question inside the window" -- there's no need to guess in advance whether a second
# question is coming, so the ~1-1.5s baseline time-to-first-answer is never taxed for
# questions that turn out to be standalone. Capped at _MAX_REVISION_QUESTIONS so an
# interviewer who keeps talking for the full window doesn't get one giant answer covering
# six unrelated questions. Configurable in this one place; 30s is an initial default to be
# tuned from real interview use, not a permanent value.
_ANSWER_REVISION_WINDOW_SECONDS = 30.0

# How many questions may be folded into one revised answer before a further question inside
# the window is instead treated as the start of a brand-new turn. Prevents unbounded
# accumulation ("Q1... Q2... Q3... Q4... Q5...") into a single unreadable answer.
_MAX_REVISION_QUESTIONS = 3

# An interviewer describing a multi-sentence scenario ("Two services show correlated
# anomalies with no dependency between them... a BigPanda alert fires for the payment
# service... but infrastructure metrics look healthy...") pauses for breath between
# sentences. Measured live 2026-08-13: gaps of 10-18s between those sentences. None of them
# are questions on their own, and discarding them as "not a question" loses the scenario the
# eventual question depends on -- observed live producing an answer grounded in the WRONG
# prior Q&A (attached via the follow-up path) instead of the scenario actually spoken.
_SCENARIO_CONTEXT_WINDOW_SECONDS = 25.0

# Wait this long after speech stops before committing to an answer. Guards against
# answering half a question when the interviewer is mid-thought ("...how would you design
# an AIOps platform" [2s] "...for a bank with 10,000 services?"). Any new speech inside
# this window cancels the timer and extends the question instead of starting a 2nd answer.
# Kept deliberately short -- this is added directly to time-to-first-token.
# Measured live 2026-08-13: generation itself is fast (TTFA 0.8-1.4s), but VAD silence
# (1.0s) + debounce was adding ~2.2s of dead air BEFORE generation started -- the delay the
# candidate actually feels. Cut to 0.5s: the supersede path already re-answers late
# additions, so a long debounce is largely redundant protection paid for on every question.
_QUESTION_DEBOUNCE_SECONDS = 0.5

# How many (question, answer) pairs to keep for follow-up context.
_CONVERSATION_MEMORY_TURNS = 3

# Openers that mark a genuinely NEW question rather than a continuation of the previous one.
# Without this check, a real follow-up question asked shortly after an answer would get
# glued onto the previous question's text and answered as one confused hybrid.
_NEW_QUESTION_OPENERS = (
    "next question", "moving on", "let's move", "lets move", "another question",
    "second question", "third question", "my next", "ok so now", "okay so now",
    "let me ask", "different question", "switching", "one more question",
    "let's talk about", "lets talk about", "changing topic", "different topic",
    "moving to", "let's switch", "lets switch", "new topic",
)

# Bare drill-down questions a senior interviewer uses to challenge an answer. These are
# short, high-context, and MEANINGLESS without the previous question and answer attached --
# "Why not Kinesis?" answered standalone is a generic essay; answered with context it's a
# crisp defence of a decision you just made.
_FOLLOWUP_MARKERS = (
    "why", "why not", "how", "what if", "what about", "how about", "and if",
    "can you explain", "explain that", "elaborate", "go deeper", "tell me more",
    "what else", "anything else", "give me an example", "for example", "such as",
    "trade-off", "tradeoff", "trade off", "downside", "drawback", "risk",
    "at scale", "how would you scale", "what would you do differently",
    "but ", "however", "instead", "alternative", "versus", " vs ",
)

# ANAPHORIC REFERENCES -- a question that leans on "this/that/it" to mean "the thing we
# were just talking about" is unanswerable without the prior question attached, regardless
# of whether it happens to start with an interrogative word. Missed live 2026-08-13:
# "what kind of tools we have to use for this" and "...walk me through the RCA approach"
# (referring to a scenario stated two turns earlier) both slipped through the marker list
# above and were answered with zero context -- one asked the interviewer to clarify (a
# banned response), the other silently defaulted to the wrong toolset.
_ANAPHORA_MARKERS = (
    "for this", "for that", "with this", "with that", "in this case", "in that case",
    "using this", "using that", "on this", "on that", "from this", "from that",
    "this approach", "that approach", "this one", "that one", "this way",
    # Added 2026-08-25: bare pronoun objects on a dependency verb ("integrate IT with...",
    # "combine IT with...") -- these carry the same "refers to something already said"
    # signal as "with this"/"with that" above but without that exact phrase, and were
    # observed live merging incorrectly because of it.
    "integrate it", "integrate that", "combine it", "combine that", "connect it",
    "connect that", "extend it", "extend that", "use it with", "work with it",
    "how does it work with", "how would it work with",
    # Added 2026-08-25 (compound-turn testing): subject-position pronoun -- "how does IT
    # integrate" (it = the thing just discussed) is a different construction from
    # "integrate IT with" above (object-position) but carries the identical dependency.
    "does it integrate", "would it integrate", "does that integrate", "would that integrate",
    "does it combine", "would it combine", "does it connect", "would it connect",
    "does it compare", "would it compare", "does it relate", "would it relate",
    "how does it ", "how would it ", "how does that ", "how would that ",
    "walk me through the", "walk me through this", "walk me through that",
    "for the same", "the same time", "at the same time",
)

# An interviewer statement that is NOT a question and must never trigger an answer.
_FRAGMENT_STARTERS = (
    "and", "or", "but", "also", "plus", "with", "for", "from", "into", "across",
    "specifically", "particularly", "especially", "including", "such", "like",
    "the", "a", "an", "that", "this", "these", "those", "in", "on", "at", "to",
)


# Deterministic, code-level backstop for "asks the interviewer to clarify" -- proven live
# 2026-08-13 that prompt instructions alone get rephrased around indefinitely (the model
# swaps "I need you to restate" for "I don't have visibility into what this refers to" and
# every variant in between). Detecting the BEHAVIOR via regex, rather than trusting the
# model to police its own wording, is the only reliable fix.
_CLARIFICATION_SEEKING_PATTERNS = (
    r"\bi (?:can't|cannot|don't|do not) (?:answer|know|tell) (?:this|that|which)\b",
    r"\bi (?:need|require) (?:more )?context\b",
    r"\bi (?:need|require) you to (?:restate|clarify|specify|repeat)\b",
    r"\b(?:could|can|would) you (?:restate|clarify|specify|repeat|rephrase)\b",
    r"\bwhich (?:one|thing|topic) (?:are you|do you mean|were you)\b",
    r"\bwithout (?:more|additional) (?:context|information|signal)\b",
    r"\bi don'?t have visibility into\b",
    r"\bi'?m not (?:sure|certain) (?:which|what) (?:you'?re|you are) (?:asking|referring)\b",
    r"\bplease (?:clarify|specify|restate|elaborate on what)\b",
    r"\bmy best guess\b.{0,40}\bif you\b",
    r"\bare you asking about\b.{0,80}\bor\b.{0,80}\bor\b",  # the "A, B, or C?" menu pattern
)
_CLARIFICATION_RE = re.compile("|".join(_CLARIFICATION_SEEKING_PATTERNS), re.IGNORECASE)


def seeks_clarification(answer_text: str) -> bool:
    """True if the answer asks the interviewer to explain/restate the question.

    Checked only in the first ~300 chars -- that is where this failure mode always shows
    up (it is how the response opens), and searching the whole answer risks a false
    positive from an unrelated later sentence that happens to contain "context" or similar.
    """
    return bool(_CLARIFICATION_RE.search(answer_text[:300]))


_CLARIFICATION_RETRY_INSTRUCTION = (
    "\n\nMANDATORY CORRECTION -- your previous attempt at this exact question asked the "
    "interviewer to clarify, restate, or explain what was meant. That is not permitted, in "
    "any wording, under any framing. You do not have the option of asking for more "
    "information. Pick the single most defensible interpretation of the question given "
    "your grounding and this interview's domain, and answer it directly and confidently, "
    "starting from substance in the very first sentence. If you are genuinely uncertain, "
    "mark the answer [Low Confidence] -- that is allowed. Asking the interviewer anything "
    "is not."
)


def _materially_different(new_text: str, old_text: str, min_new_words: int = 3) -> bool:
    """True only if the new text adds real content over the old.

    Whisper re-emits near-duplicate text on echo/noise tails. Without this check each
    duplicate triggers a full cancel+regenerate cycle, so the answer restarts forever and
    never completes -- observed live 2026-08-13.
    """
    new_words = new_text.lower().split()
    old_words = old_text.lower().split()
    if len(new_words) <= len(old_words):
        return False
    added = new_words[len(old_words):]
    # Ignore filler-only additions ("okay", "so", "and", "yeah")
    meaningful = [w for w in added if w.strip(".,!?;:") not in
                  {"okay", "ok", "so", "and", "the", "a", "yeah", "right", "um", "uh", "you"}]
    return len(meaningful) >= min_new_words


def _strip_leading_repeated_filler(text: str) -> str:
    """Strip a short phrase immediately repeated at the start of text.

    Whisper occasionally hallucinates a short filler repeat ('and the and the') that
    survives the fragment-hold path and gets concatenated onto the next real segment,
    polluting the front of the actual question -- observed live 2026-08-13 producing
    q='and the You have 500 AWS accounts...'. Only strips units of 1-3 words repeated at
    least twice, so genuine short emphatic speech elsewhere in a real sentence ('no no no,
    that's not right') is untouched -- this only fires on a repeat sitting at the very
    front of a held fragment.
    """
    words = text.strip().split()
    for unit_len in (1, 2, 3):
        if len(words) < unit_len * 2:
            continue
        unit = [w.lower().strip(".,!?") for w in words[:unit_len]]
        pos = unit_len
        repeats = 1
        while pos + unit_len <= len(words):
            candidate = [w.lower().strip(".,!?") for w in words[pos : pos + unit_len]]
            if candidate != unit:
                break
            repeats += 1
            pos += unit_len
        if repeats >= 2:
            return " ".join(words[pos:]).strip()
    return text


def _is_sentence_fragment(text: str) -> bool:
    """True for a dangling continuation that is not answerable on its own.

    Observed live: 'infrastructure telemetric.' and 'and the' were answered as standalone
    questions because they happened to contain a detector keyword. A fragment like that
    carries no question -- it belongs merged onto the utterance it continues.
    """
    cleaned = text.strip().strip(".,!?;: ")
    if not cleaned:
        return True
    words = cleaned.split()
    if "?" in text:
        return False  # an explicit question mark is a strong standalone signal
    if len(words) > 5:
        return False
    lowered = words[0].lower()
    # Starts with a conjunction/preposition/article => it is continuing a previous clause.
    if lowered in _FRAGMENT_STARTERS:
        return True
    # Very short with no interrogative opener at all => not an answerable question.
    interrogatives = {"why", "how", "what", "when", "where", "which", "who", "explain",
                      "describe", "tell", "give", "can", "could", "would", "do", "did",
                      "have", "is", "are", "was"}
    return len(words) <= 3 and lowered not in interrogatives


_ACKNOWLEDGEMENT_ONLY = {
    "ok", "okay", "right", "sure", "yes", "yeah", "yep", "mm", "mmhmm", "uh huh",
    "makes sense", "interesting", "got it", "understood", "good", "great", "nice",
    "thats good", "that's good", "fair enough", "go on", "continue", "i see",
    "perfect", "excellent", "cool", "alright", "all right", "true", "exactly",
    "thank you", "thanks", "thank you very much", "many thanks", "cheers",
    "no worries", "not a problem", "sounds good", "sounds great", "noted",
    "appreciate it", "appreciated", "gotcha", "for sure", "absolutely",
}


def _looks_like_new_question(text: str) -> bool:
    lowered = text.lower().strip()
    return any(lowered.startswith(opener) or opener in lowered[:40]
               for opener in _NEW_QUESTION_OPENERS)


def _is_question_enhancement(text: str) -> bool:
    """True when this looks like added detail on the SAME question, not a new topic.

    An explicit topic-change marker ("next question", "let's move on") is the reliable
    negative signal; absent that, a short addition shortly after the previous question is
    far more likely to be a qualifier than a brand-new question.
    """
    return not _looks_like_new_question(text)


def _is_acknowledgement_only(text: str) -> bool:
    """Interviewer noise ('Okay.', 'Right.', 'Makes sense.') that must not be answered."""
    cleaned = text.lower().strip().strip(".!?,;: ")
    if not cleaned:
        return True
    if cleaned in _ACKNOWLEDGEMENT_ONLY:
        return True
    # Also catch short combinations like "okay, right" / "yeah makes sense"
    words = [w.strip(".!?,;:") for w in cleaned.split()]
    if len(words) <= 4 and all(
        w in {x for phrase in _ACKNOWLEDGEMENT_ONLY for x in phrase.split()} for w in words if w
    ):
        return True
    return False


def _is_followup(text: str) -> bool:
    """True for a drill-down/challenge that needs the previous Q&A attached to make sense."""
    lowered = text.lower().strip().strip(".!?")
    if _looks_like_new_question(lowered):
        return False
    # Anaphoric references ("for this", "at the same time", "walk me through this") can
    # land anywhere in the sentence, including the end -- check the WHOLE text, not just
    # an opening window. A question leaning on "this/that" to mean "what we were just
    # discussing" is unanswerable standalone regardless of where that reference sits.
    if any(m in lowered for m in _ANAPHORA_MARKERS):
        return True
    word_count = len(lowered.split())
    # Short questions are almost always drill-downs on what was just said.
    if word_count <= 8:
        return True
    return any(lowered.startswith(m) or f" {m}" in lowered[:60] for m in _FOLLOWUP_MARKERS)


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
        self._pending_transcript: Transcript | None = None
        self._last_answered: Transcript | None = None
        # How many questions are folded into the answer currently in _last_answered.
        # Tracked as explicit state rather than parsed back out of the merged text --
        # parsing "Question N:" labels undercounts once a dependent follow-up in the middle
        # of a merge chain falls back to plain concatenation instead of numbering (a
        # genuine question can depend on the previous one via anaphora -- "what about
        # security for THAT setup?" -- while still being a distinct question that should
        # count against the cap). Reset to 1 on every genuinely new turn, incremented on
        # every successful revision merge -- see _MAX_REVISION_QUESTIONS.
        self._revision_count = 0
        # Rolling conversation memory: the last few (question, answer) pairs. Without this,
        # every question is answered in isolation -- which is why bare follow-ups ("Why?",
        # "What about 100k events/sec?", "Give me an example") were previously answered as
        # if they were standalone questions with no idea what they referred to.
        self._conversation: list[tuple[str, str]] = []
        # In-flight generation, so new speech can cancel a stale answer mid-stream.
        self._active_generation: asyncio.Task | None = None
        # Monotonic generation id. Cancellation is not instantaneous -- a task can be between
        # awaits when cancelled, or an in-flight HTTP response can land afterwards. Every
        # overlay write is gated on "am I still the newest generation?", so a slow or
        # cancelled answer can never overwrite a newer one.
        self._generation_seq = 0
        # Debounce timer: gives the interviewer a beat to keep talking before we commit.
        self._debounce_task: asyncio.Task | None = None
        # Accumulated non-question scenario-setup sentences, held rather than discarded so
        # the eventual question carries the scenario instead of losing it. Separate from
        # _pending_transcript because the gap between scenario sentences (10-18s observed)
        # is much longer than _FRAGMENT_MERGE_GAP_SECONDS (3s), which is tuned for a single
        # paused-and-continued utterance, not a multi-sentence scenario walkthrough.
        self._scenario_context_text: str = ""
        self._scenario_context_end_time: float = 0.0

    async def _handle_segment(self, segment: SpeechSegment) -> None:
        """Turn a speech segment into pending question text, then (re)start the debounce.

        Deliberately does NOT answer directly -- it hands off to a debounce timer so that
        an interviewer who keeps talking (a pause mid-question, or an added qualifier)
        updates the same pending question instead of triggering a second answer.
        """
        try:
            with StageTimer("speaker_id"):
                diarized = await asyncio.to_thread(self._diarizer.diarize, segment)
            if diarized.is_me:
                return

            with StageTimer("stt"):
                transcript = await self._stt.transcribe(diarized)
            if transcript is None:
                return

            # DO NOT cancel yet. Measured live 2026-08-13: cancelling on every incoming
            # segment meant sub-second noise blips (864ms, 896ms) each killed the in-flight
            # answer and restarted it -- the same question regenerated 4+ times and the
            # candidate never saw a completed answer. Cancel only once we know this segment
            # actually changes the question (below).
            if _is_acknowledgement_only(transcript.text):
                logger.debug(f"Acknowledgement mid-answer, ignoring: {transcript.text!r}")
                return

            pending = self._pending_transcript
            # NOTE: deliberately NOT comparing speaker_id. Measured live 2026-08-13: with no
            # enrolled voice, pyannote assigns a NEW speaker id to nearly every segment
            # (A,B,C,D,E,F within one minute), so any speaker-equality check silently never
            # fires and fragments stop merging. We only capture the far end via BlackHole --
            # the candidate's own mic is never in this stream -- so effectively every segment
            # is the interviewer and speaker matching buys nothing while breaking merging.
            if (
                pending is not None
                and (transcript.start_time - pending.end_time) < _FRAGMENT_MERGE_GAP_SECONDS
            ):
                # Same speaker, short gap -- continuation of the same utterance. Strip a
                # leading repeated-filler artifact from the held fragment first so it
                # doesn't pollute the front of the merged text.
                cleaned_pending_text = _strip_leading_repeated_filler(pending.text)
                pending.text = f"{cleaned_pending_text} {transcript.text}".strip()
                pending.end_time = transcript.end_time
                wait_seconds = _QUESTION_DEBOUNCE_SECONDS
            elif (
                pending is None
                and self._last_answered is not None
                and (transcript.start_time - self._last_answered.end_time)
                < _ANSWER_REVISION_WINDOW_SECONDS
                and _is_question_enhancement(transcript.text)  # no explicit pivot phrase
                and self._revision_count < _MAX_REVISION_QUESTIONS
            ):
                # ANSWER REVISION: we already answered, and within the revision window the
                # same speaker either (a) added detail to the SAME question -- "how would
                # you design event correlation?" [pause] "specifically across Splunk and
                # Prometheus in a bank" -- or (b) asked a genuinely NEW question -- "What is
                # Terraform?" [3s] "How should you have integrated AI?". Both cases fold into
                # the SAME answer: re-answer the combined question set and replace the
                # previous answer on the overlay, rather than leaving the addition unanswered
                # or starting a disconnected second answer. _format_pending_addition tells
                # the two cases apart (numbered "Question 1:/Question 2:" for a genuinely
                # independent new question, plain concatenation for a same-question addition)
                # -- see its docstring.
                merged_text = self._format_pending_addition(self._last_answered.text, transcript)
                # Guard against re-answering the same thing forever. Whisper re-emits
                # near-identical text for echo/noise tails, and without this the merged
                # question barely changes yet still triggers a full cancel+regenerate --
                # observed live as the same answer restarting 4 times in 20 seconds.
                if not _materially_different(merged_text, self._last_answered.text):
                    logger.info(
                        f"Revision adds nothing new, keeping current answer: "
                        f"{transcript.text[:60]!r}"
                    )
                    return
                logger.info(f"ANSWER REVISION -- merged question: {merged_text!r}")
                pending = Transcript(
                    speaker_id=transcript.speaker_id,
                    text=merged_text,
                    start_time=self._last_answered.start_time,
                    end_time=transcript.end_time,
                    language=transcript.language,
                )
                self._revision_count += 1
                wait_seconds = _QUESTION_DEBOUNCE_SECONDS
            else:
                # Genuinely new turn -- no active answer to revise (none yet, outside the
                # revision window, an explicit pivot, or the revision cap was already hit).
                # Answered immediately: only the short fragment debounce applies, same as
                # any other question. If pending is already set here, a second segment
                # arrived inside that same short debounce (a tight race, not the normal
                # path) -- fold it in with the same formatting rather than losing it.
                if pending is not None:
                    transcript.text = self._format_pending_addition(pending.text, transcript)
                    transcript.start_time = pending.start_time
                pending = transcript
                self._revision_count = 1
                wait_seconds = _QUESTION_DEBOUNCE_SECONDS

            # Only NOW is it worth interrupting: this segment genuinely changes the question.
            self._cancel_active_generation("question changed")
            if self._debounce_task and not self._debounce_task.done():
                self._debounce_task.cancel()

            self._pending_transcript = pending
            self._debounce_task = asyncio.create_task(self._debounced_process(wait_seconds))
        except Exception:
            logger.exception("Error handling speech segment")

    def _reads_as_independent_question(self, transcript: Transcript) -> bool:
        """True if this segment has its own complete interrogative signal -- a "?" or an
        opener like "have you"/"can I"/"how would you" -- meaning it is very likely a
        genuinely NEW question, not a qualifier being added to the previous one.

        Observed live 2026-08-13, twice in one session: "Tell me about yourself" was
        answered, then "Have you designed any kind of agent-KI automation?" (and later
        "Can I explain the Agentic AI solution...") arrived inside the supersede window
        and got merged into the FIRST question's text, because _is_question_enhancement
        only refuses to merge when the new segment uses an explicit transition phrase
        ("next question", "moving on") -- real interviewers almost never say those, they
        just ask the next question directly. Checking the segment's own interrogative
        strength catches this regardless of phrasing.

        EXCEPTION -- anaphoric dependency wins over interrogative shape. Observed live
        2026-08-25: "tell me about Terraform" [pause] "how can you integrate it with
        Agentic AI" was treated as independent (it starts with "how", has a "?") and
        answered as a disconnected second question, when it's clearly a continuation --
        "integrate IT" only means something with the prior turn's topic as antecedent. A
        sentence can be grammatically a complete question and still be semantically
        dependent on what was just said; the anaphora check catches the case the bare
        interrogative check cannot. An explicit new-topic opener (_NEW_QUESTION_OPENERS)
        still wins over both -- that's handled by the caller via _looks_like_new_question.
        """
        lowered = transcript.text.lower().strip()
        if any(m in lowered for m in _ANAPHORA_MARKERS):
            return False
        detected = self._question_detector.detect(transcript)
        return detected is not None and detected.has_interrogative_signal

    def _format_pending_addition(self, old_text: str, new_transcript: Transcript) -> str:
        """Join a newly-arrived segment onto prior question text -- either a still-pending
        (undebounced) transcript, or (via the answer revision window) an already-answered
        question being revised.

        Whether to number depends ONLY on the NEW segment's own independence (has its own
        interrogative signal, no anaphoric dependency -- see _reads_as_independent_question),
        not on re-checking old_text. old_text can already be a multi-question accumulated
        blob (from an earlier merge in this same revision chain); re-running the anaphora
        check against that whole blob is unreliable once it contains an earlier dependent
        clause -- e.g. after folding in "what about security for THAT setup?", the phrase
        "for that" now sits somewhere in the middle of old_text and would wrongly flag the
        entire blob as anaphoric on every later check, breaking numbering for a genuinely
        independent question that comes after it. old_text was already validated as
        answerable when it was first set, so it doesn't need re-validating here.

        If the new segment reads independent -- format it as an explicit numbered compound
        turn ("Question 1: ... Question 2: ...") so the answer generator addresses both
        distinctly, rather than a bare space-joined concatenation that reads as one
        run-on/garbled sentence. A genuinely dependent addition (leans on "it"/"that" to
        mean what was just discussed) still gets the plain concatenation this always did,
        since numbering a fragment or an anaphoric clause as its own "question" would be
        wrong -- it folds into the question it depends on instead.
        """
        if not self._reads_as_independent_question(new_transcript):
            cleaned = _strip_leading_repeated_filler(old_text)
            return f"{cleaned} {new_transcript.text}".strip()

        existing_numbers = re.findall(r"^Question (\d+):", old_text, re.MULTILINE)
        if existing_numbers:
            # Already a compound turn (3rd+ question in the same grouping window) --
            # append as the next number rather than restarting at 1.
            next_n = int(existing_numbers[-1]) + 1
            return f"{old_text}\nQuestion {next_n}: {new_transcript.text}"
        return f"Question 1: {old_text}\nQuestion 2: {new_transcript.text}"

    def _cancel_active_generation(self, reason: str) -> None:
        if self._active_generation and not self._active_generation.done():
            self._active_generation.cancel()
            logger.info(f"Cancelled in-flight answer generation: {reason}")

    async def _debounced_process(self, wait_seconds: float = _QUESTION_DEBOUNCE_SECONDS) -> None:
        """Wait a short beat before committing to an answer -- guards against answering half
        a question if the interviewer is pausing mid-sentence. Any new speech inside the
        window cancels this timer and extends the pending question instead. Deliberately
        always short: compound-question grouping happens AFTER an answer exists, via the
        answer revision window (_ANSWER_REVISION_WINDOW_SECONDS), not by delaying the first
        answer.
        """
        try:
            await asyncio.sleep(wait_seconds)
        except asyncio.CancelledError:
            return

        transcript = self._pending_transcript
        if transcript is None:
            return
        self._pending_transcript = None
        self._generation_seq += 1
        gen_id = self._generation_seq
        self._active_generation = asyncio.create_task(
            self._process_transcript(transcript, gen_id)
        )
        try:
            await self._active_generation
        except asyncio.CancelledError:
            logger.debug("Answer generation cancelled before completion")

    async def _process_transcript(self, transcript: Transcript, gen_id: int = 0) -> None:
        started_at = time.monotonic()

        def is_current() -> bool:
            """False once a newer question has superseded this generation."""
            return gen_id == 0 or gen_id == self._generation_seq

        try:
            # Interviewer acknowledgements ("Okay." / "Right." / "Makes sense.") are not
            # questions and must never trigger an answer -- they'd wipe a useful answer off
            # the overlay mid-read.
            if _is_acknowledgement_only(transcript.text):
                logger.debug(f"Acknowledgement, not answering: {transcript.text!r}")
                return

            # A dangling continuation ("infrastructure telemetric.", "and the") is not an
            # answerable question -- answering it wipes a good answer off the overlay and
            # replaces it with nonsense. Hold it as pending so the NEXT segment merges with
            # it, rather than burning a generation on it.
            if _is_sentence_fragment(transcript.text):
                logger.info(f"Fragment held for merge, not answered: {transcript.text!r}")
                self._pending_transcript = transcript
                return

            question = self._question_detector.detect(transcript)
            mid_scenario_buildup = bool(self._scenario_context_text) and (
                transcript.start_time - self._scenario_context_end_time
            ) < _SCENARIO_CONTEXT_WINDOW_SECONDS
            # A detection with no real interrogative signal (no "?", no interrogative
            # opener -- it only survived on a keyword match or technical-term density)
            # arriving WHILE a scenario is already being built up is far more likely to be
            # another scenario sentence than the interviewer's actual ask -- observed live
            # 2026-08-13: "production BigPanda application suddenly produces false alarms"
            # matched the keyword "bigpanda" and got answered immediately, truncating the
            # scenario buildup before the real question ("how would you determine...")
            # arrived. A real ask overwhelmingly uses an interrogative phrasing, so treat
            # this as more scenario instead. When there's NO active buildup, this stays
            # exactly as before: catching a real STANDALONE question that has a keyword or
            # technical density but happened to lose its interrogative wording to STT.
            if question is None or (
                not question.has_interrogative_signal and mid_scenario_buildup
            ):
                # Not phrased as a question, but a real declarative sentence is almost
                # always scenario setup ("the two services show anomalies with no direct
                # dependency") that the interviewer will assume as given once the actual
                # question lands. Hold it instead of discarding it -- see
                # _SCENARIO_CONTEXT_WINDOW_SECONDS.
                if mid_scenario_buildup:
                    self._scenario_context_text = (
                        f"{self._scenario_context_text} {transcript.text}".strip()
                    )
                else:
                    self._scenario_context_text = transcript.text
                self._scenario_context_end_time = transcript.end_time
                logger.info(f"Held as scenario context, not answered: {transcript.text!r}")
                return

            # If scenario-setup sentences were held recently, this question almost
            # certainly depends on them -- prepend them so the LLM sees the full scenario,
            # not just the trailing question. Consume-and-clear: this scenario belongs to
            # THIS question only, not future ones.
            have_scenario_context = bool(self._scenario_context_text) and (
                question.transcript.start_time - self._scenario_context_end_time
            ) < _SCENARIO_CONTEXT_WINDOW_SECONDS
            if have_scenario_context:
                logger.info(
                    f"Prepending held scenario context: {self._scenario_context_text!r}"
                )
                question.transcript.text = (
                    f"{self._scenario_context_text} {question.transcript.text}".strip()
                )
            self._scenario_context_text = ""
            self._scenario_context_end_time = 0.0

            # A bare drill-down ("Why not Kinesis?", "At scale?", "Give me an example")
            # is meaningless standalone. Attach the running conversation so the answer
            # defends the decision just made instead of restarting from first principles.
            # Skipped when scenario context was just attached -- that scenario IS the
            # context, so pulling in an unrelated earlier Q&A on top would be wrong (this
            # is exactly how the wrong-context bug reproduced live 2026-08-13: a "How would
            # you..." scenario question got misread as a follow-up to a DIFFERENT, already-
            # answered question because "how" is a generic follow-up marker).
            conversation_context = ""
            if not have_scenario_context and _is_followup(transcript.text) and self._conversation:
                prev_q, prev_a = self._conversation[-1]
                conversation_context = (
                    "\n\nCONVERSATION SO FAR -- the interviewer is drilling into the answer "
                    "you just gave, not asking a fresh question. Treat this as ONE continuing "
                    "architectural discussion: defend, refine or honestly revise the position "
                    "you already took rather than restarting the explanation from scratch. "
                    "Keep it SHORT and precise -- a drill-down deserves a tight, direct answer, "
                    "not a re-lecture.\n"
                    f"  PREVIOUS QUESTION: {prev_q}\n"
                    f"  YOUR PREVIOUS ANSWER (abridged): {prev_a[:1200]}\n"
                    "  NOTE: your own previous answer is NOT evidence of your real experience. "
                    "If it claimed hands-on experience you do not actually have per the persona "
                    "grounding, correct that now rather than building on it.\n"
                )
                logger.info(f"FOLLOW-UP detected, attaching prior context: {transcript.text!r}")

            # Remember what we answered so a later addition from the same speaker can be
            # merged with it and re-answered, instead of being answered context-free.
            self._last_answered = transcript

            # Flip the overlay to "answering..." with the heard question right away --
            # visible feedback within ~a second of speech ending, well before the
            # retrieval+LLM answer starts streaming in over it.
            if self._on_partial_answer and is_current():
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
                    if self._on_answer and is_current():
                        await self._on_answer(answer)
                    return

            if self._retriever is not None:
                with StageTimer("retrieval"):
                    context = await self._retriever.retrieve(question)
            else:
                context = RetrievedContext(question=question, chunks=[])

            chunk_texts = [c.text for c in context.chunks]
            # Follow-ups depend on prior context, so they must never serve a cached answer
            # keyed only on their own (short, ambiguous) text -- "Why?" would collide across
            # completely unrelated discussions.
            cached_text = (
                None
                if conversation_context
                else await self._cache.get_llm_response(question.transcript.text, chunk_texts)
            )
            # Time-based compound-turn grouping (2026-08-25): _format_pending_addition
            # marks a compound turn with literal "Question 1:"/"Question 2:" numbering.
            # Detect that here and tell the model explicitly to answer all of them
            # together in one natural response, rather than picking just one or writing
            # them up as disconnected mini-essays.
            compound_instruction = ""
            compound_matches = re.findall(r"^Question (\d+):", question.transcript.text, re.MULTILINE)
            if len(compound_matches) >= 2:
                compound_instruction = (
                    "\n\nCOMPOUND TURN -- the interviewer asked "
                    f"{len(compound_matches)} questions in close succession, numbered "
                    "above. Answer ALL of them, in one natural "
                    "flowing response as a candidate would actually speak it -- not "
                    "'Answer 1: ... Answer 2: ...' unless the questions are complex enough "
                    "that explicit separation genuinely improves clarity. Do not ignore any "
                    "of them and do not answer only the last one. If a later question "
                    "explicitly says to forget/ignore the earlier one or pivots to a "
                    "clearly different scenario (\"forget that\", \"different scenario\", "
                    "\"let's design something else\"), respect that pivot -- don't force "
                    "an artificial connection between unrelated questions just because they "
                    "arrived in the same turn.\n"
                )
            system_prompt = (
                build_system_prompt(question_text=question.transcript.text)
                + conversation_context
                + compound_instruction
            )
            q_prefix = f"**Q:** {question.transcript.text}\n\n---\n\n"
            if cached_text is not None:
                raw_text = cached_text
            else:
                raw_text = ""
                with StageTimer("llm"):
                    first_token_at: float | None = None
                    async for delta in self._claude.stream(
                        build_user_prompt(context), system=system_prompt
                    ):
                        raw_text += delta
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                            # TIME TO FIRST USEFUL ANSWER -- the metric that actually decides
                            # whether this is usable live. Everything before this point is
                            # dead air the candidate is standing in.
                            logger.info(
                                f"TTFA {first_token_at - started_at:.2f}s "
                                f"(debounce {_QUESTION_DEBOUNCE_SECONDS}s excluded) "
                                f"q={question.transcript.text!r}"
                            )
                        if not is_current():
                            logger.debug("Stale generation -- dropping partial")
                            return
                        if self._on_partial_answer:
                            await self._on_partial_answer(q_prefix + raw_text)
                if not conversation_context:
                    await self._cache.set_llm_response(
                        question.transcript.text, chunk_texts, raw_text
                    )

            # DETERMINISTIC BACKSTOP: prompt instructions alone proved unreliable at
            # stopping the model from asking the interviewer to clarify -- it just
            # rephrases around whatever banned strings are listed. Detect the behavior via
            # regex and force one non-streamed regeneration with an amplified instruction
            # rather than ever showing a clarification request on the overlay.
            if seeks_clarification(raw_text) and is_current():
                logger.warning(
                    f"Answer sought clarification, forcing retry: {raw_text[:100]!r}"
                )
                retry_system = system_prompt + _CLARIFICATION_RETRY_INSTRUCTION
                try:
                    raw_text = await self._claude.complete(
                        build_user_prompt(context), system=retry_system
                    )
                except Exception:
                    logger.exception("Clarification-retry generation failed")
                if is_current() and self._on_partial_answer:
                    await self._on_partial_answer(q_prefix + raw_text)

            answer = self._optimizer.optimize(context, raw_text)
            PIPELINE_TOTAL_LATENCY_SECONDS.observe(time.monotonic() - started_at)

            # Record the exchange so the NEXT drill-down has something to build on.
            self._conversation.append((question.transcript.text, answer.text))
            if len(self._conversation) > _CONVERSATION_MEMORY_TURNS:
                self._conversation.pop(0)

            if self._on_answer and is_current():
                await self._on_answer(answer)
        except asyncio.CancelledError:
            logger.debug("Generation cancelled mid-answer (interviewer spoke)")
            raise
        except Exception:
            logger.exception("Error processing transcript")

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
