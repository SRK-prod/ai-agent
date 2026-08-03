#!/usr/bin/env python3
"""Build the pre-generated interview Q&A bank (run offline, before interviews).

For each knowledge topic: ask Claude for the most likely interview questions
(including follow-ups) for this candidate, then answer each one through the
same retrieval + structured-answer prompt the live pipeline uses, and store
question-embedding -> answer in Qdrant. At interview time a matching spoken
question is served from the bank in <1s instead of ~4-5s live generation.

Re-runnable: banked answers are keyed deterministically by question text, so
rebuilding updates in place. Roughly 10 questions/topic x 14 topics; expect
~10-15 minutes and a few tens of cents of Haiku usage per full build.
"""

from __future__ import annotations

import argparse
import asyncio
import re

from meeting_copilot.config import get_config
from meeting_copilot.knowledge.embeddings import LocalEmbedder
from meeting_copilot.knowledge.ingestion import load_topics
from meeting_copilot.llm.claude_client import ClaudeClient
from meeting_copilot.llm.prompt_templates import build_system_prompt, build_user_prompt
from meeting_copilot.nlp.question_detector import get_question_detector
from meeting_copilot.pipeline.events import DetectedQuestion, Transcript
from meeting_copilot.retrieval.hybrid_search import HybridSearcher
from meeting_copilot.retrieval.qa_bank import QaBankStore
from meeting_copilot.utils.logging import configure_logging, get_logger

logger = get_logger()

_PROFILE_SUMMARY = """Candidate profile summary: 14+ years, Principal Cloud & DevOps Platform Architect at Reach Mobile
(previously Technical Lead at Innominds), AWS/GCP, Terraform, Kubernetes (EKS/GKE), CI/CD
(Jenkins/Spinnaker/GitHub Actions), observability (Grafana/Prometheus/ELK/Datadog), FinOps
(25-30% cost reduction), SOC compliance, and an enterprise Agentic AI transformation: production
FinOps cost-anomaly agent and Terraform AI plan-reviewer, RAG, LangChain/LangGraph/CrewAI, LLMOps.
Target roles: DevOps Manager / DevOps Architect / Principal DevOps (senior compensation band)."""

# Tiered coverage: basics get asked as warm-ups/screeners even at senior levels; scenarios
# and case studies are where senior interviews are actually decided.
_TIERS = {
    "basics": (
        8,
        (
            "fundamental definition/concept questions (asked as warm-ups or by screeners), "
            "phrased simply, e.g. 'What is X', 'What is the difference between X and Y'"
        ),
    ),
    "intermediate": (
        8,
        "practical how-do-you / how-does-it-work questions about day-to-day usage",
    ),
    "advanced": (
        10,
        (
            "architecture, tradeoff, at-scale, and 'design this' questions expecting "
            "principal-level judgment, plus probing follow-ups to earlier answers"
        ),
    ),
    "scenario": (
        8,
        (
            "scenario and troubleshooting questions ('you see X in production, walk me "
            "through what you do'), including outage/postmortem stories"
        ),
    ),
    "case-study": (
        6,
        (
            "open-ended case studies and system-design prompts an interviewer would give, "
            "including migration/cost/reliability programs with constraints"
        ),
    ),
}

_QUESTION_GEN_PROMPT = """You are preparing a candidate for senior DevOps interviews at top product companies.
{profile}

Research notes on what interviews at this level actually ask:
{research}

List the {count} most likely {tier_description} on the topic "{topic}" for this candidate.
Output ONLY the questions, one per line, no numbering, no commentary."""


def _load_research_notes() -> str:
    from meeting_copilot.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "research" / "interview_research_notes.txt"
    return path.read_text()[:6000] if path.exists() else "(no research notes found)"


async def _generate_questions(
    claude: ClaudeClient, topic: str, count: int, tier_description: str, research: str
) -> list[str]:
    raw = await claude.complete(
        _QUESTION_GEN_PROMPT.format(
            profile=_PROFILE_SUMMARY,
            research=research,
            topic=topic,
            count=count,
            tier_description=tier_description,
        )
    )
    questions = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^[\s\d\.\-\*\)]+", "", line).strip()
        if len(cleaned.split()) >= 4:
            questions.append(cleaned)
    return questions[:count]


# Universal staples every interview opens/closes with -- answered from the profile.
_STAPLE_QUESTIONS = [
    "Tell me about yourself",
    "Walk me through your career journey",
    "What is your biggest achievement in your career",
    "Tell me about your biggest failure and what you learned from it",
    "Why are you looking to leave your current role",
    "Why do you want to join our company",
    "Where do you see yourself in five years",
    "What are your strengths and weaknesses",
    "How do you stay current with new technologies",
    "Tell me about your current role and responsibilities",
    "What does your day to day look like in your current role",
    "Describe a time you had a conflict with a colleague and how you resolved it",
    "Tell me about a time you influenced a decision without having authority",
    "Why should we hire you for this role",
    "Do you have any questions for us",
]

# Guaranteed literal "What is X?" coverage per topic -- discovered missing for "kubernetes"
# after a real mismatch (asked "What is Kubernetes?", no exact bank entry, fell through to
# a wrong, only-topically-adjacent question at 0.67 similarity). AI-generated basics-tier
# questions don't reliably include the single most obvious opening phrasing, so these are
# added explicitly rather than left to chance.
_TOPIC_DISPLAY_NAMES = {
    "aws": "AWS",
    "terraform": "Terraform",
    "kubernetes": "Kubernetes",
    "ci-cd": "CI/CD",
    "devops-leadership": "DevOps leadership",
    "incident-management": "incident management and SRE",
    "mlops-ai": "MLOps and AI in DevOps",
    "platform-engineering": "platform engineering",
    "devsecops": "DevSecOps",
    "migration-modernization": "cloud migration and modernization",
    "observability": "observability",
    "system-design": "system design",
}


def _guaranteed_questions_for_topic(topic: str) -> list[str]:
    name = _TOPIC_DISPLAY_NAMES.get(topic)
    if name is None:
        return []
    return [
        f"What is {name}?",
        f"What is {name} and why is it used?",
        f"Can you explain {name} in simple terms?",
    ]


async def _answer_question(
    claude: ClaudeClient, searcher: HybridSearcher, question_text: str
) -> str:
    transcript = Transcript(speaker_id="interviewer", text=question_text, start_time=0, end_time=0)
    question = DetectedQuestion(
        transcript=transcript, matched_keywords=[], ends_with_question_mark=True
    )
    context = await searcher.retrieve(question)
    return await claude.complete(build_user_prompt(context), system=build_system_prompt())


async def _bank_one(
    claude: ClaudeClient,
    searcher: HybridSearcher,
    embedder: LocalEmbedder,
    bank: QaBankStore,
    detector,
    topic: str,
    question_text: str,
) -> None:
    answer = await _answer_question(claude, searcher, question_text)
    vector = await embedder.embed(question_text)
    bank.upsert(question_text, answer, topic, vector)
    # Sanity: warn if the live detector would never route this question to the bank.
    probe = Transcript(speaker_id="x", text=question_text, start_time=0, end_time=0)
    if detector.detect(probe) is None:
        logger.warning(f"[{topic}] banked but live detector would skip: {question_text!r}")


async def build() -> None:
    cfg = get_config()
    # Same structured-answer prompt as live, but allow full-length generations.
    claude = ClaudeClient(cfg.llm.model_copy(update={"max_tokens": 2000}))
    embedder = LocalEmbedder()
    searcher = HybridSearcher(embedder=embedder)
    bank = QaBankStore()
    bank.ensure_collection()
    detector = get_question_detector()
    research = _load_research_notes()

    total = 0

    logger.info(f"[staples] banking {len(_STAPLE_QUESTIONS)} universal questions")
    for question_text in _STAPLE_QUESTIONS:
        await _bank_one(claude, searcher, embedder, bank, detector, "staples", question_text)
        total += 1
    logger.info(f"[staples] done (running total {total})")

    topics = [t.name for t in load_topics()]
    for topic in topics:
        guaranteed = _guaranteed_questions_for_topic(topic)
        for question_text in guaranteed:
            await _bank_one(claude, searcher, embedder, bank, detector, topic, question_text)
            total += 1
        if guaranteed:
            logger.info(f"[{topic}/guaranteed] banked {len(guaranteed)} (running total {total})")

        for tier, (count, tier_description) in _TIERS.items():
            questions = await _generate_questions(claude, topic, count, tier_description, research)
            for question_text in questions:
                await _bank_one(
                    claude, searcher, embedder, bank, detector, topic, question_text
                )
                total += 1
            logger.info(f"[{topic}/{tier}] banked {len(questions)} (running total {total})")

    logger.info(f"Q&A bank build complete: {total} answers stored ({bank.count()} in collection)")


def main() -> None:
    configure_logging()
    argparse.ArgumentParser(description="Build the pre-generated interview Q&A bank").parse_args()
    asyncio.run(build())


if __name__ == "__main__":
    main()
