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
from meeting_copilot.llm.prompt_templates import (
    _classify_category,
    build_system_prompt,
    build_user_prompt,
)
from meeting_copilot.nlp.question_detector import get_question_detector
from meeting_copilot.pipeline.events import (
    Answer,
    AudioHealth,
    DiarizedSegment,
    RetrievedContext,
    SpeechSegment,
    Transcript,
)
from meeting_copilot.pipeline.metrics import (
    LLM_TTFT_SECONDS,
    PIPELINE_TOTAL_LATENCY_SECONDS,
    TTFA_SECONDS,
    StageTimer,
)
from meeting_copilot.retrieval.hybrid_search import HybridSearcher
from meeting_copilot.retrieval.qa_bank import QaBankStore
from meeting_copilot.speaker.diarization import SpeakerDiarizer
from meeting_copilot.stt.faster_whisper_engine import SttStage
from meeting_copilot.utils.logging import get_logger
from meeting_copilot.vad.silero_vad import SileroVAD

logger = get_logger()

OnAnswer = Callable[[Answer], Awaitable[None]]
OnPartialAnswer = Callable[[str], Awaitable[None]]
OnAudioHealth = Callable[[AudioHealth], Awaitable[None]]

# How often the watchdog polls AudioCapture.health(). A state CHANGE is what actually gets
# logged/pushed (see _watchdog_loop) -- this just bounds how quickly a transition is
# noticed, so it can stay short without spamming anything.
_AUDIO_HEALTH_POLL_SECONDS = 2.0

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
# Kept deliberately short -- this is added directly to time-to-first-token, on EVERY
# question, guaranteed.
# Measured live 2026-08-13: generation itself is fast (TTFA 0.8-1.4s), but VAD silence
# (1.0s) + debounce was adding ~2.2s of dead air BEFORE generation started -- the delay the
# candidate actually feels. Cut to 0.5s then.
# Cut further 2026-08-25 after the TTFA instrumentation (metrics.py TTFA_SECONDS) showed
# this fixed 500ms was the single most deterministic, cheapest-to-remove chunk of the
# ~1.7-2.4s real pipeline: STT ~400-600ms + this 500ms + Claude TTFT ~800-1300ms. Not
# removed entirely -- fragmented STT ("What is Terraform" / "and how" / "have you used
# it?") still needs SOME window to merge back into one question before answering; this is
# not the ANSWER REVISION WINDOW (which handles a fully separate, already-answered question
# arriving later) but the guard against answering a mid-utterance fragment as if it were
# the whole question. Verified against a live-pipeline test at this value before landing:
# fragment merging and the answer revision window both still work correctly (see
# scratchpad test run referenced in the commit).
_QUESTION_DEBOUNCE_SECONDS = 0.2

# How many (question, answer) pairs to keep for follow-up context.
_CONVERSATION_MEMORY_TURNS = 5

# How long a conversation thread stays "live". Past this gap the interviewer has almost
# certainly moved on, so prior turns are dropped rather than attached to an unrelated
# question. Only relevant because context is now attached by DEFAULT (see the gate in
# _process_transcript) instead of only on a lexical follow-up match.
_CONVERSATION_RECENCY_SECONDS = 300.0

# Openers that mark a genuinely NEW question rather than a continuation of the previous one.
# Without this check, a real follow-up question asked shortly after an answer would get
# glued onto the previous question's text and answered as one confused hybrid.
_NEW_QUESTION_OPENERS = (
    "next question", "moving on", "let's move", "lets move", "another question",
    "second question", "third question", "my next", "ok so now", "okay so now",
    "let me ask", "different question", "switching", "one more question",
    "let's talk about", "lets talk about", "changing topic", "different topic",
    "moving to", "let's switch", "lets switch", "new topic",
    # Added 2026-08-25: an interviewer pivoting to a new design scenario often phrases it
    # as an IMPERATIVE, not a question -- "Now let's design a Databricks streaming
    # platform" has no "?", no interrogative opener, and isn't a literal "next question" --
    # so it fell through every existing check and got merged as an ENHANCEMENT onto the
    # previous, unrelated answer instead of starting a fresh scenario. Category
    # classification and conversation context are independent: a category change alone
    # must never reset context, but an explicit new design/scenario prompt must.
    "let's design", "lets design", "let's build", "lets build", "let's architect",
    "lets architect", "now let's", "now lets", "let's discuss a different",
    "let's discuss another", "different scenario", "another scenario", "new scenario",
    "completely different", "switching to", "let's now design", "let's now build",
    "for this next one", "in this next scenario",
    # Added 2026-08-25, second pass: explicit scenario RESETS, phrased as an instruction to
    # discard prior context rather than as "let's move to X" -- "forget the previous
    # architecture, assume you're designing a real-time data platform now" contains none of
    # the phrases above.
    "forget the previous", "forget everything", "ignore the previous", "ignore what we",
    "disregard the previous", "start fresh", "start over", "clean slate",
    # Added 2026-08-25, third pass: the SHORT, natural way an interviewer actually says
    # this live -- "Okay, forget that. You have a legacy application now." -- reproduced
    # exactly this failure mode, mechanically merging onto the previous, already-answered
    # question instead of starting fresh. "forget the previous architecture" (added last
    # pass) is the formal phrasing; nobody actually talks like that in a fast interview.
    "forget that", "forget it", "scratch that", "never mind that", "never mind, ",
    "disregard that",
    "assume a completely different", "assume you're designing", "assume a different",
    "totally different problem", "entirely different problem", "unrelated system",
    "separate scenario", "hypothetically, a different", "consider a different",
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

# CORRECTIONS -- the interviewer restating what they meant, which must REPLACE the previous
# interpretation rather than be merged onto it. Identified as a real defect 2026-08-27:
# "Would you use ECS?" followed by "Actually, I mean Kubernetes" hit the answer-revision
# path and produced one answer covering BOTH, which is a wrong answer on screen rather than
# merely a slow one. Distinct from _NEW_QUESTION_OPENERS (which start an unrelated topic)
# and from anaphora (which extend the same question) -- a correction keeps the previous
# question's SUBJECT but replaces one of its terms.
_CORRECTION_MARKERS = (
    "actually, i mean", "actually i mean", "actually, i meant", "actually i meant",
    "sorry, i mean", "sorry i mean", "sorry, i meant", "sorry i meant",
    "i mean ", "i meant ", "what i mean is", "what i meant was",
    "let me rephrase", "let me restate", "rephrase that", "to clarify",
    "no, i was asking", "no i was asking", "no, i'm asking", "no i'm asking",
    "that's not what i", "thats not what i", "i should say", "correction",
    "rather than", " i mean,",
)


def _is_correction(text: str) -> bool:
    """True when this turn restates/corrects the previous question rather than extending it.

    Deliberately checked on the FIRST part of the utterance only: a correction marker is an
    opener ("Actually, I mean Kubernetes"), whereas the same words appearing deep in a long
    sentence are usually ordinary speech ("...the trade-off I mean here is cost"), and
    treating those as corrections would wrongly discard a real question."""
    lowered = text.lower().strip()
    head = lowered[:60]
    return any(marker in head for marker in _CORRECTION_MARKERS)


# An interviewer statement that is NOT a question and must never trigger an answer.
_FRAGMENT_STARTERS = (
    "and", "or", "but", "also", "plus", "with", "for", "from", "into", "across",
    "specifically", "particularly", "especially", "including", "such", "like",
    "the", "a", "an", "that", "this", "these", "those", "in", "on", "at", "to",
)


# Openers that signal this segment hangs off the PREVIOUS clause -- it completes,
# qualifies, or adds scope, rather than starting a new question. Deliberately narrower than
# _FRAGMENT_STARTERS: bare articles/pronouns ("the", "this", "that") and location
# prepositions ("in", "on", "at") are dropped because a real new question often opens with
# them ("In Kubernetes, how do you...?"), whereas nothing below naturally opens a fresh,
# independent interview question.
_CONTINUATION_OPENERS = (
    "and", "or", "but", "also", "plus", "with", "without", "for", "from", "into",
    "across", "alongside", "specifically", "particularly", "especially", "including",
    "regarding", "concerning", "considering", "given", "using", "versus", "vs",
    "as well as", "along with",
)

# A trailing continuation is a QUALIFIER, not a whole new sentence. A segment that opens
# with a connective but runs long is far more likely to be a full standalone question that
# just happens to start with "for"/"given" ("For a bank with 10,000 microservices, how
# would you design the platform?") -- those should still split.
_MAX_WORDS_FOR_TRAILING_CONTINUATION = 12


def _is_trailing_continuation(text: str) -> bool:
    """True when this segment grammatically CONTINUES a previous utterance rather than
    starting a new one -- it opens with a connective ("for GCP and AWS with multi-cloud",
    "and how would that scale", "with multi-region failover") and is short enough to be a
    qualifier rather than a new sentence.

    Such a segment completes, qualifies, or adds scope to the question already asked, so it
    belongs merged onto that question NO MATTER HOW LONG the interviewer paused -- a 20s
    thinking pause between "How can you architect Terraform..." and "...for GCP and AWS" is
    still one question. This is the signal that lets the merge ignore the revision-window
    clock, which is otherwise the only thing keeping a late continuation attached.

    An explicit pivot still wins: "and now let's move on to Kubernetes" opens with "and"
    but _looks_like_new_question fires on "let's move on", so this returns False and the
    segment starts its own turn.
    """
    lowered = text.lower().strip().strip("\"'“”")
    if not lowered:
        return False
    if _looks_like_new_question(text):
        return False
    words = lowered.split()
    if len(words) > _MAX_WORDS_FOR_TRAILING_CONTINUATION:
        return False
    first = words[0].strip(".,!?;:")
    first_two = " ".join(w.strip(".,!?;:") for w in words[:2])
    first_three = " ".join(w.strip(".,!?;:") for w in words[:3])
    return first in _CONTINUATION_OPENERS or first_two in _CONTINUATION_OPENERS or first_three in _CONTINUATION_OPENERS


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
    # Reached the overlay live 2026-09-01 on a garbled transcript, unfiltered: "I need to
    # clarify what you're asking here -- are you asking me to redesign the AIOps reference
    # architecture ... or are you asking something else about the multi-cloud setup?"
    # Two gaps: (a) "I need to clarify" is not "I need YOU to clarify", which is what the
    # pattern above matched; (b) the menu pattern required THREE branches ("A, B, or C"),
    # but the real failure offered only two. Both are the same behaviour and both are banned.
    r"\bi (?:need|want|have) to clarify\b",
    r"\b(?:just|first) to clarify\b",
    r"\bare you asking\b.{0,100}\bor are you\b",  # two-option menu: "are you asking X or are you Y"
    r"\bdo you mean\b.{0,80}\bor\b",
    r"\bwhat exactly (?:do you|are you) (?:mean|asking|referring)\b",
)
_CLARIFICATION_RE = re.compile("|".join(_CLARIFICATION_SEEKING_PATTERNS), re.IGNORECASE)

# The same failure wearing a technical excuse: instead of asking the interviewer to
# repeat, the answer complains about the transcript or narrates its own instructions.
# Measured live 2026-08-24 on real interview fragments -- "I'm looking at an incomplete
# question fragment", "The transcription appears to be incomplete", "Per my instructions,
# I should infer the clearest real question". None of these tripped the clarification
# patterns above, so they reached the overlay unfiltered mid-interview.
_TRANSCRIPT_COMPLAINT_PATTERNS = (
    r"\b(question|text|transcript\w*)\b.{0,40}\b(is|appears?|looks?|seems?)\b.{0,20}"
    r"\b(incomplete|garbled|cut off|truncated|fragment\w*)\b",
    r"\b(incomplete|garbled|partial)\b.{0,20}\b(question|fragment|transcript\w*)\b",
    # No linking verb: "The transcript cut off mid-question", "the question cut off".
    # Missed on the first pass 2026-08-24 and reached the answer opening.
    r"\b(transcript\w*|question|text)\b.{0,30}\bcut off\b",
    r"\bcut off mid[- ]?(question|sentence|way)\b",
    r"\binferr?ing\b.{0,40}\bquestion\b",
    r"\bvoice[- ]to[- ]text\b",
    r"\bspeech[- ]to[- ]text\b",
    r"\bper my instructions?\b",
    r"\bmy (instructions?|guidelines?|persona|grounding)\b.{0,20}\b(say|state|tell|require)",
    r"\bI should infer\b",
    r"\bas transcribed\b",
)
_TRANSCRIPT_COMPLAINT_RE = re.compile(
    "|".join(_TRANSCRIPT_COMPLAINT_PATTERNS), re.IGNORECASE
)


def seeks_clarification(answer_text: str) -> bool:
    """True if the answer asks the interviewer to explain/restate the question, OR dodges
    by complaining about the transcript / narrating its own instructions.

    Checked only in the first ~300 chars -- that is where this failure mode always shows
    up (it is how the response opens), and searching the whole answer risks a false
    positive from an unrelated later sentence that happens to contain "context" or similar.
    """
    head = answer_text[:300]
    return bool(_CLARIFICATION_RE.search(head) or _TRANSCRIPT_COMPLAINT_RE.search(head))


_CLARIFICATION_RETRY_INSTRUCTION = (
    "\n\nMANDATORY CORRECTION -- your previous attempt at this exact question either asked "
    "the interviewer to clarify/restate what was meant, or dodged by commenting on the "
    "question text itself (calling it incomplete, garbled, a fragment, a transcription "
    "artifact) or by narrating your own instructions. None of that is permitted, in any "
    "wording, under any framing. You do not have the option of asking for more "
    "information, and the interviewer must never hear anything about the transcript or "
    "about these instructions. Pick the single most defensible interpretation of the "
    "question given your grounding and this interview's domain, and answer it directly and "
    "confidently, starting from substance in the very first sentence. If you are genuinely "
    "uncertain, mark the answer [Low Confidence] -- that is allowed. Commenting on the "
    "question or asking the interviewer anything is not."
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


# _classify_category's catch-all bucket -- landing here on its own says nothing reliable
# about topic (many genuinely different real questions share it purely because they don't
# hit a specific keyword), so it can't be trusted for a same/different comparison the way a
# specific category can.
_TOPIC_AMBIGUOUS_CATEGORY = "default"

# A genuinely additive follow-up ("...for a bank with 10,000 services?", "...specifically for
# a payment service?") is almost always short -- it leans on the previous question for its
# subject and just adds a qualifier. Used only as the fallback signal when category alone
# can't decide (see _is_topic_change).
_MIN_WORDS_FOR_STANDALONE_QUESTION = 8


def _is_topic_change(new_text: str, previous_text: str) -> bool:
    """Cheap, deterministic topic-discontinuity signal, complementing (never replacing)
    _looks_like_new_question. That phrase-based check only catches a topic change when the
    interviewer explicitly narrates the transition ('next question', 'let's move on') --
    measured live 2026-08-27: three genuinely unrelated questions asked back-to-back with no
    such marker ('tell me about yourself' -> 'AWS multi-zone landing setup' -> 'telemetry in
    the DevOps system') all got merged into one 22-second, 1356-token mega-answer, because
    none of them used a marker phrase.

    Two-tier signal, in priority order:
    1. If BOTH the new and previous question land in a specific (non-'default') category,
       trust that directly -- a category match ('design HA' -> 'how would you handle
       failover specifically for a payment service', both ha_dr) means stay merged
       regardless of length; a mismatch ('what is Kubernetes' -> 'tell me about yourself',
       definition vs career_narrative) means split even if both are short.
    2. Otherwise (either side landed in the 'default' catch-all, which carries no reliable
       topic signal on its own -- measured live: both real unrelated follow-ups above landed
       here) fall back to length: a substantial, grammatically complete question is far more
       likely to be the interviewer's next prepared question than a trailing qualifier.

    A false split here is safe: conversation context still attaches by default (see the
    ATTACH BY DEFAULT block above) regardless of merge-vs-split, so a genuinely connected
    follow-up that splits anyway still gets answered with full awareness of the prior turn,
    just as its own answer instead of a merged rewrite. A false MERGE is the expensive
    failure (a confused multi-topic answer taking 20+ seconds), so this is deliberately
    biased toward splitting when uncertain."""
    new_category = _classify_category(new_text)
    previous_category = _classify_category(previous_text)
    if new_category != _TOPIC_AMBIGUOUS_CATEGORY and previous_category != _TOPIC_AMBIGUOUS_CATEGORY:
        return new_category != previous_category
    return len(new_text.split()) >= _MIN_WORDS_FOR_STANDALONE_QUESTION


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


class _NullDiarizer:
    """Stand-in used when speaker.enabled is False: every segment is the far end.

    Deliberately assigns ONE stable speaker id rather than a per-segment one. The
    orchestrator does not compare speaker ids when merging fragments (see the note in
    _handle_segment), but the id reaches the prompt as "Question (from X)", and a value
    that changed every segment would read as a room full of people to the model.
    """

    def diarize(self, segment: SpeechSegment) -> DiarizedSegment:
        return DiarizedSegment(
            segment=segment, speaker_id="speaker_a", is_me=False, similarity_to_me=0.0
        )


class MeetingPipeline:
    def __init__(
        self,
        on_answer: OnAnswer | None = None,
        on_partial_answer: OnPartialAnswer | None = None,
        on_audio_health: OnAudioHealth | None = None,
    ):
        """on_answer: async callable(Answer) -> None, e.g. push over the overlay WebSocket.
        on_partial_answer: async callable(text_so_far) -> None, called as the answer streams in
        (skipped on a cache hit, since that's already instant).
        on_audio_health: async callable(AudioHealth) -> None, called only on a STATE
        transition (not every poll) -- see _watchdog_loop."""
        self._cfg = get_config()
        self._capture = AudioCapture()
        self._vad = SileroVAD()
        # Skipping diarization avoids loading pyannote+torch at all -- see SpeakerConfig.enabled.
        self._diarizer = SpeakerDiarizer() if self._cfg.speaker.enabled else _NullDiarizer()
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
        self._on_audio_health = on_audio_health

        self._running = False
        self._run_task: asyncio.Task | None = None
        self._worker_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
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
        # Set when the current pending question is a CORRECTION of the previous one, so the
        # answer generator is told to replace the earlier interpretation rather than cover
        # both. Consumed (and cleared) by _process_transcript.
        self._pending_is_correction = False
        # Rolling conversation memory: the last few (question, answer) pairs. Without this,
        # every question is answered in isolation -- which is why bare follow-ups ("Why?",
        # "What about 100k events/sec?", "Give me an example") were previously answered as
        # if they were standalone questions with no idea what they referred to.
        self._conversation: list[tuple[str, str]] = []
        self._last_conversation_at: float | None = None
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
        # STT duration for the segment most recently handled -- read back in
        # _process_transcript to fold into the per-question latency breakdown. Passed via
        # instance state rather than a return value because _handle_segment's result can
        # take several different paths (fragment merge, revision, genuinely new) before
        # _process_transcript is eventually invoked, sometimes turns later -- see
        # _handle_segment's own "started_at" for why the SAME pattern is used for TTFA.
        self._last_stt_seconds: float = 0.0

    async def _handle_segment(self, segment: SpeechSegment) -> None:
        """Turn a speech segment into pending question text, then (re)start the debounce.

        Deliberately does NOT answer directly -- it hands off to a debounce timer so that
        an interviewer who keeps talking (a pause mid-question, or an added qualifier)
        updates the same pending question instead of triggering a second answer.
        """
        # TTFA is measured from HERE -- the moment this segment is ready to handle, before
        # diarize/STT even run -- not from when _process_transcript happens to start. That
        # earlier, narrower definition undercounted real perceived latency by excluding
        # diarize+STT+debounce entirely. Threaded through _debounced_process ->
        # _process_transcript below.
        started_at = time.monotonic()
        try:
            with StageTimer("speaker_id"):
                diarized = await asyncio.to_thread(self._diarizer.diarize, segment)
            if diarized.is_me:
                return

            with StageTimer("stt") as stt_timer:
                transcript = await self._stt.transcribe(diarized)
            self._last_stt_seconds = stt_timer.elapsed_seconds
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
                and (
                    (transcript.start_time - pending.end_time) < _FRAGMENT_MERGE_GAP_SECONDS
                    # A grammatical continuation ("...for GCP and AWS") folds onto the still
                    # -pending question no matter how long the interviewer paused -- an
                    # incomplete sentence that gets completed later is one question, not two.
                    or _is_trailing_continuation(transcript.text)
                )
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
                and (
                    (transcript.start_time - self._last_answered.end_time)
                    < _ANSWER_REVISION_WINDOW_SECONDS
                    # A grammatical continuation of the answered question ("...for GCP and
                    # AWS with multi-cloud") is still the same question however long the
                    # pause -- the revision-window clock does not apply to it.
                    or _is_trailing_continuation(transcript.text)
                )
                and _is_question_enhancement(transcript.text)  # no explicit pivot phrase
                # A correction ALWAYS belongs in this branch: by definition it restates the
                # question just asked, so it must replace that answer rather than start a
                # new turn -- it bypasses the topic-change check, which could otherwise
                # split it off as unrelated.
                and (
                    _is_correction(transcript.text)
                    or not _is_topic_change(transcript.text, self._last_answered.text)
                    # A continuation overrides a topic-change split: "...for GCP and AWS"
                    # may classify into a different category on its own, but it is a scope
                    # qualifier on the prior question, not a new topic.
                    or _is_trailing_continuation(transcript.text)
                    # A DECLARATIVE SENTENCE IS SCENARIO SETUP, NEVER A TOPIC CHANGE.
                    # Fixed 2026-09-03 by tracing the exact live pattern -- interviewer
                    # gives ~40s of background, then asks. "The platform needs to support
                    # GCP as well." is 8 words, which tripped _is_topic_change's LENGTH
                    # fallback (_MIN_WORDS_FOR_STANDALONE_QUESTION) and started a fresh
                    # turn, discarding the AWS and compliance context already accumulated.
                    # That fallback is explicitly documented as a heuristic for "a
                    # substantial, grammatically complete QUESTION" -- a statement with no
                    # interrogative signal at all is not one, so it must not be judged on
                    # length. An explicit pivot ("let's move on") still splits, via
                    # _is_question_enhancement above.
                    or not self._reads_as_independent_question(transcript)
                )
                and self._revision_count < _MAX_REVISION_QUESTIONS
            ):
                # ANSWER REVISION: we already answered, and within the revision window the
                # same speaker added detail to the SAME question or asked a short, clearly
                # connected follow-up -- "how would you design event correlation?" [pause]
                # "specifically across Splunk and Prometheus in a bank". Both fold into the
                # SAME answer: re-answer the combined question set and replace the previous
                # answer on the overlay. A genuinely NEW, unrelated question inside this same
                # window (caught by _is_topic_change above, since it rarely comes with an
                # explicit pivot phrase) does NOT reach this branch -- it starts its own fresh
                # turn instead, still with conversation context attached. Corrected 2026-08-27
                # after live use: this branch used to fold ANY non-explicitly-pivoted question
                # in here regardless of topic, so three unrelated real interview questions
                # merged into one 22s/1356-token mega-answer. _format_pending_addition tells
                # apart a genuinely independent new question that still belongs in this branch
                # (numbered "Question 1:/Question 2:") from a same-question addition (plain
                # concatenation) -- see its docstring.
                merged_text = self._format_pending_addition(self._last_answered.text, transcript)
                # Carried into _process_transcript, which turns it into an explicit
                # "the second statement REPLACES the first" instruction. Tracked as state
                # rather than re-detected from merged_text, because by then the correction
                # marker sits mid-string where a head-only check can't see it.
                self._pending_is_correction = _is_correction(transcript.text)
                if self._pending_is_correction:
                    logger.info(f"CORRECTION detected -- replacing interpretation: {transcript.text!r}")
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
                # SCENARIO SETUP DOES NOT SPEND THE REVISION BUDGET. Fixed 2026-09-03
                # after a preflight run of the exact live pattern -- interviewer gives ~60s
                # of background, then asks. The setup sentences match a technical keyword,
                # so each one came through here as a "revision", exhausted
                # _MAX_REVISION_QUESTIONS before the real question arrived, and the real
                # question then started a FRESH turn that had lost two thirds of the
                # context ("compliance" and "GCP" never reached the model).
                #
                # That cap exists to stop six genuinely UNRELATED QUESTIONS merging into one
                # unreadable answer. A declarative sentence carrying no interrogative signal
                # is not a question, so it must not consume that budget -- it is the
                # scenario the eventual question depends on. Counting only real questions
                # keeps the cap doing its job while letting arbitrarily long buildup
                # accumulate.
                addition_is_a_question = self._reads_as_independent_question(transcript)
                if addition_is_a_question:
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
            self._debounce_task = asyncio.create_task(
                self._debounced_process(wait_seconds, started_at)
            )
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
        # A question ENDING in a bare demonstrative is anaphoric by construction: "so what
        # would you do?", "how would you architect this?", "what would you do about that?"
        # carry no subject of their own -- the subject is everything the interviewer just
        # said. Found 2026-09-03 tracing the live pattern: after ~40s of scenario buildup
        # the closing ask started a FRESH turn and threw the entire scenario away.
        # Deliberately anchored to the END of the utterance, so "how would you design this
        # PLATFORM?" -- which names its own subject -- is still independent.
        if re.search(r"\b(this|that|it)\s*[?.!]*\s*$", lowered):
            return False
        # Opens with a conjunction/preposition/article ("for GCP and AWS with multi-cloud")
        # -- grammatically a continuation of the prior utterance, so it folds in with plain
        # concatenation, never numbered as its own "Question 2" even if STT tacked a "?" on.
        if _is_trailing_continuation(transcript.text):
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

    async def _debounced_process(
        self, wait_seconds: float = _QUESTION_DEBOUNCE_SECONDS, started_at: float = 0.0
    ) -> None:
        """Wait a short beat before committing to an answer -- guards against answering half
        a question if the interviewer is pausing mid-sentence. Any new speech inside the
        window cancels this timer and extends the pending question instead. Deliberately
        always short: compound-question grouping happens AFTER an answer exists, via the
        answer revision window (_ANSWER_REVISION_WINDOW_SECONDS), not by delaying the first
        answer.

        started_at: passed through from _handle_segment (the moment THIS segment became
        ready to handle) so the eventual TTFA measurement in _process_transcript covers the
        real pipeline -- diarize+STT+this debounce+generation -- not just generation.
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
            self._process_transcript(transcript, gen_id, started_at or time.monotonic())
        )
        try:
            await self._active_generation
        except asyncio.CancelledError:
            logger.debug("Answer generation cancelled before completion")

    async def _process_transcript(
        self, transcript: Transcript, gen_id: int = 0, started_at: float | None = None
    ) -> None:
        started_at = started_at if started_at is not None else time.monotonic()
        stt_seconds = self._last_stt_seconds

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
            # Widened 2026-09-03 from "... and mid_scenario_buildup" to hold a
            # declarative statement from the FIRST sentence, not just once a buildup is
            # already running. Measured by replaying a real recorded US interview: five
            # minutes of the interviewer explaining the role produced ~20 answers, because
            # sentence one was answered (no buildup active yet, so this guard could not
            # fire), which set _last_answered, and every following sentence then merged
            # into it through the revision path at one Claude call each.
            #
            # The cost, stated plainly and previously accepted as deliberate: a real
            # question that loses BOTH its question mark and every interrogative word to
            # STT is now held instead of answered. Real audio says that is much rarer than
            # the monologue case, and the held text is picked up by the next utterance
            # rather than lost.
            if question is None or not question.has_interrogative_signal:
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
            # Drop a thread the interviewer has clearly left behind, so stale turns can't
            # attach to an unrelated question now that attachment is the default.
            if (
                self._last_conversation_at is not None
                and (time.monotonic() - self._last_conversation_at)
                > _CONVERSATION_RECENCY_SECONDS
            ):
                logger.info(
                    f"Conversation thread stale (>{_CONVERSATION_RECENCY_SECONDS:.0f}s idle) "
                    f"-- dropping {len(self._conversation)} turn(s) of context"
                )
                self._conversation.clear()
                self._last_conversation_at = None

            # ATTACH BY DEFAULT, opt out only on an explicit topic change. This was
            # previously opt-IN via _is_followup()'s lexical markers ("why", "how", "but",
            # <=8 words), which measured 2026-08-24 against the 60 most recent REAL
            # interview questions matched only 17 (28%) -- 41 (68%) continued the thread but
            # were answered with no context at all, including "I just reminded you the
            # question was high availability" (the interviewer repeating himself because the
            # thread was lost) and "So Terraform obviously doesn't know it exists, right?".
            # Real interviewers simply do not phrase drill-downs using a fixed marker list,
            # so any marker list keeps losing. _looks_like_new_question() is the guard that
            # matters -- it catches the explicit hand-offs ("next question", "let's move on
            # to database") where carrying context forward would genuinely be wrong.
            if (
                not have_scenario_context
                and self._conversation
                and not _looks_like_new_question(transcript.text)
            ):
                # Use the FULL rolling window (up to _CONVERSATION_MEMORY_TURNS), not just the
                # single most recent turn -- a bare single-turn lookback loses the original
                # framing by turn 3+ of a drill-down chain. Reproduced live: "design a HA AWS
                # architecture" -> "how would you handle DB failover?" (correctly grounded in
                # turn 1) -> "how would you optimize the cost?" (turn 3 only saw turn 2's
                # failover answer, lost the architecture from turn 1 entirely, and answered
                # generic cost optimization instead of cost-optimizing the actual design).
                turns_text = "\n".join(
                    f"  Q{i + 1}: {q}\n  A{i + 1} (abridged): {a[:500]}"
                    for i, (q, a) in enumerate(self._conversation)
                )
                conversation_context = (
                    "\n\nCONVERSATION SO FAR -- this is ONE continuing discussion thread, not "
                    "isolated questions. The interviewer is drilling into, extending, or "
                    "circling back within the SAME topic established below. A question like "
                    "\"how would you optimize the cost?\" after discussing an architecture means "
                    "the cost of THAT architecture, not cost optimization in general -- resolve "
                    "any pronoun or implicit reference (\"that\", \"it\", \"the cost\", \"this "
                    "approach\") against the thread below, even if it points back further than "
                    "the immediately previous turn. Defend, refine, extend or honestly revise "
                    "the position already taken rather than restarting from scratch or "
                    "answering a generic version of the question. Keep it SHORT and precise -- "
                    "a drill-down deserves a tight, direct answer, not a re-lecture.\n"
                    "  QUESTION TYPE AND ACTIVE SCENARIO ARE INDEPENDENT. A different kind of "
                    "question (architecture, then database, then cost, then monitoring) does "
                    "NOT mean a different scenario -- by default assume every question below "
                    "is still about the SAME system/architecture/scenario already established, "
                    "just viewed through a different lens. \"How would you optimize the cost?\" "
                    "after an HA architecture discussion must name the actual components "
                    "already chosen (e.g. \"for the three-AZ EKS and RDS Multi-AZ setup we "
                    "just designed, the cost drivers are...\"), NOT a generic cost-optimization "
                    "answer that could apply to any AWS account. Only treat this as a genuinely "
                    "new, unrelated scenario if the thread below actually shows an explicit "
                    "pivot (\"let's design a completely different system\", \"different "
                    "scenario\", \"now let's build X instead\") -- absent that, inherit "
                    "everything: the architecture, the requirements, the constraints, the "
                    "decisions already made.\n"
                    f"{turns_text}\n"
                    "  NOTE: your own previous answers are NOT evidence of your real experience. "
                    "If they claimed hands-on experience you do not actually have per the "
                    "persona grounding, correct that now rather than building on it.\n"
                )
                logger.info(
                    f"FOLLOW-UP detected, attaching {len(self._conversation)} prior turn(s): "
                    f"{transcript.text!r}"
                )

            # Remember what we answered so a later addition from the same speaker can be
            # merged with it and re-answered, instead of being answered context-free.
            self._last_answered = transcript

            # Flip the overlay to "answering..." with the heard question right away --
            # visible feedback within ~a second of speech ending, well before the
            # retrieval+LLM answer starts streaming in over it.
            #
            # Keep the LAST COMPLETED answer on screen underneath while the new one is
            # generated. Previously this wiped the overlay to a bare question line, so every
            # regenerate (a follow-up, or the interviewer adding a qualifier mid-question)
            # blanked an answer the candidate was still reading and left them staring at
            # dead space for ~2-3s. Nothing here changes actual latency -- it removes the
            # blank screen, which is what the wait actually felt like. The first streamed
            # token replaces this wholesale.
            if self._on_partial_answer and is_current():
                placeholder = f"*Q: {question.transcript.text}*"
                if self._conversation:
                    _, previous_answer = self._conversation[-1]
                    placeholder += (
                        "\n\n---\n\n*(previous answer -- new one generating...)*\n\n"
                        f"{previous_answer}"
                    )
                await self._on_partial_answer(placeholder)

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

            # Follow-ups depend on prior context, so they must never serve an answer cached
            # under their own (short, ambiguous) text alone -- "Why?" would collide across
            # completely unrelated discussions. That is why this used to skip the cache
            # entirely whenever conversation context was present. Corrected 2026-08-27:
            # context now attaches by DEFAULT on nearly every turn, so "skip when context
            # exists" had silently disabled the cache almost entirely -- it was never
            # written, therefore never read. Fix is to make the context part of the KEY
            # rather than a reason to bypass the cache: identical question + identical
            # conversation state is genuinely the same request and is safe to reuse, while
            # "Why?" in two different discussions now hashes to two different keys.
            cache_key_parts = [*(c.text for c in context.chunks), conversation_context]
            cached_text = await self._cache.get_llm_response(
                question.transcript.text, cache_key_parts
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
            # CORRECTION -- the interviewer restated what they meant, so the earlier reading
            # is void. Without this the model sees "Would you use ECS? Actually, I mean
            # Kubernetes" as a compound turn and answers BOTH, which is a wrong answer on
            # screen rather than merely a slow one (identified 2026-08-27).
            correction_instruction = ""
            if self._pending_is_correction:
                correction_instruction = (
                    "\n\nCORRECTION -- the interviewer CORRECTED their own question partway "
                    "through this turn. The later statement REPLACES the earlier one; the "
                    "earlier wording is void. Answer ONLY the corrected question. Do not "
                    "answer the original, do not compare the two, and do not mention that a "
                    "correction happened -- just answer what they actually meant. Where the "
                    "correction only swaps one term (e.g. the platform, service or "
                    "technology), keep the rest of the original question's intent intact and "
                    "substitute that term.\n"
                )
            self._pending_is_correction = False

            # "classification" (_classify_category inside build_system_prompt) is cheap
            # deterministic string matching, not an LLM call -- folded into this one
            # "context_prep" stage rather than split out, since separately timing a
            # sub-millisecond string match would just add noise to the breakdown.
            with StageTimer("context_prep") as context_timer:
                system_prompt = (
                    build_system_prompt(question_text=question.transcript.text)
                    + conversation_context
                    + compound_instruction
                    + correction_instruction
                )
            q_prefix = f"**Q:** {question.transcript.text}\n\n---\n\n"
            llm_ttft_seconds = 0.0

            def _log_ttfa(first_token_at: float) -> None:
                # TIME TO FIRST USEFUL ANSWER -- the metric that actually decides whether
                # this is usable live. Everything before this point is dead air the
                # candidate is standing in. Logged once, whether the answer came from a
                # cache hit (near-instant) or a real generation, so the breakdown is
                # comparable across both.
                ttfa_seconds = first_token_at - started_at
                TTFA_SECONDS.observe(ttfa_seconds)
                turn = "first" if self._revision_count <= 1 else f"revision{self._revision_count}"
                logger.info(
                    f"Q{gen_id} turn={turn} stt={stt_seconds * 1000:.0f}ms "
                    f"context={context_timer.elapsed_seconds * 1000:.0f}ms "
                    f"llm_ttft={llm_ttft_seconds * 1000:.0f}ms "
                    f"TTFA={ttfa_seconds * 1000:.0f}ms "
                    f"q={question.transcript.text!r}"
                )

            if cached_text is not None:
                raw_text = cached_text
                _log_ttfa(time.monotonic())
            else:
                raw_text = ""
                llm_call_started_at = time.monotonic()
                with StageTimer("llm"):
                    first_token_at: float | None = None
                    async for delta in self._claude.stream(
                        build_user_prompt(context), system=system_prompt
                    ):
                        raw_text += delta
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                            llm_ttft_seconds = first_token_at - llm_call_started_at
                            LLM_TTFT_SECONDS.observe(llm_ttft_seconds)
                            _log_ttfa(first_token_at)
                        if not is_current():
                            logger.debug("Stale generation -- dropping partial")
                            return
                        if self._on_partial_answer:
                            await self._on_partial_answer(q_prefix + raw_text)
                await self._cache.set_llm_response(
                    question.transcript.text, cache_key_parts, raw_text
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

            with StageTimer("optimizer") as optimizer_timer:
                answer = self._optimizer.optimize(context, raw_text)
            PIPELINE_TOTAL_LATENCY_SECONDS.observe(time.monotonic() - started_at)

            # Record the exchange so the NEXT drill-down has something to build on.
            self._conversation.append((question.transcript.text, answer.text))
            self._last_conversation_at = time.monotonic()
            if len(self._conversation) > _CONVERSATION_MEMORY_TURNS:
                self._conversation.pop(0)

            if self._on_answer and is_current():
                with StageTimer("overlay_delivery") as overlay_timer:
                    await self._on_answer(answer)
                usage = self._claude.last_usage()
                turn = "first" if self._revision_count <= 1 else f"revision{self._revision_count}"
                usage_str = (
                    f"input_tokens={usage[0]} output_tokens={usage[1]} "
                    if usage is not None
                    else ""
                )
                logger.info(
                    f"Q{gen_id} turn={turn} {usage_str}"
                    f"optimizer={optimizer_timer.elapsed_seconds * 1000:.0f}ms "
                    f"overlay={overlay_timer.elapsed_seconds * 1000:.0f}ms "
                    f"total={(time.monotonic() - started_at) * 1000:.0f}ms"
                )
        except asyncio.CancelledError:
            logger.debug("Generation cancelled mid-answer (interviewer spoke)")
            raise
        except Exception:
            logger.exception("Error processing transcript")

    async def _watchdog_loop(self) -> None:
        """Poll AudioCapture.health() on a timer and surface every STATE transition.

        Separate from the VAD/STT pipeline on purpose: those only run when speech is
        detected, so they have no way to notice "nothing is coming through at all" -- a
        dead capture pipeline looks identical, downstream, to an interviewer who just isn't
        talking right now. Observability only for now, no automatic recovery -- see the
        module docstring in audio/capture.py for why AUDIO_INPUT_LOST is driven only by
        infrastructure evidence (callback/device), never by how long the signal is quiet.

        AUDIO_INPUT_LOST is a real problem (logged at warning); AUDIO_ACTIVE<->AUDIO_SILENT
        is routine conversational silence (logged at info, not warning) -- the interviewer
        going silent while the candidate answers is expected and will transition through
        here on essentially every question, so treating it as a warning would bury the
        signal that actually matters in noise.
        """
        last_state: str | None = None
        while self._running:
            try:
                health = self._capture.health()
                if health.state != last_state:
                    log = logger.warning if health.state == "AUDIO_INPUT_LOST" else logger.info
                    log(
                        f"Audio health: {last_state} -> {health.state}"
                        + (f" reason={health.reason}" if health.reason else "")
                        + f" (peak={health.last_peak:.4f} rms={health.last_rms:.4f} "
                        f"callbacks={health.callback_count})"
                    )
                    last_state = health.state
                    if self._on_audio_health:
                        await self._on_audio_health(health)
            except Exception:
                logger.exception("Error in audio watchdog")
            await asyncio.sleep(_AUDIO_HEALTH_POLL_SECONDS)

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
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        async for segment in self._vad.segments(self._capture.frames()):
            if not self._running:
                break
            # Non-blocking: VAD keeps segmenting live audio even if the worker is still
            # busy on a previous segment -- nothing is ever dropped, only queued.
            self._segment_queue.put_nowait(segment)
        logger.info("Meeting pipeline stopped")

    def audio_health(self) -> AudioHealth:
        """Current audio health, queryable on demand -- not just on a state transition.

        Needed because _watchdog_loop only pushes on CHANGE: a client (the overlay) that
        connects between two transitions would otherwise never learn the current state
        until the next one happens to fire, which could be arbitrarily far away if the
        state is stable. Observed live: the backend transitioned to AUDIO_INPUT_LOST at
        T, the overlay connected at T+12s, and never saw it -- this is what the server's
        websocket endpoint calls right after a new connection to close that gap.
        """
        return self._capture.health()

    def start(self) -> None:
        self._run_task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._running = False
        if self._run_task:
            self._run_task.cancel()
        if self._worker_task:
            self._worker_task.cancel()
        if self._watchdog_task:
            self._watchdog_task.cancel()
        await self._cache.close()
