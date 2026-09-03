#!/usr/bin/env python3
"""Run the representative interview question bank through the real answer path and flag
format failures automatically.

WHY THIS EXISTS: the answer format was changed on 2026-09-02 from speakable sentences to a
keyword cheat sheet. Prompt instructions are a request, not a guarantee -- the only way to
know whether Claude actually obeys them is to look at real output across the whole question
surface, not one lucky example. Eyeballing 20 answers by hand is slow and misses things, so
this scores each one against the rules the prompt claims to enforce and prints only what
looks wrong.

It deliberately SKIPS audio and STT (see smoke_e2e.py for that path). Transcription quality
is a separate concern; this validates answer generation only.

    .venv\\Scripts\\python.exe scripts\\validate_answers.py
    .venv\\Scripts\\python.exe scripts\\validate_answers.py --only secrets terraform
    .venv\\Scripts\\python.exe scripts\\validate_answers.py --out logs/answers.md

Needs ANTHROPIC_API_KEY in .env. Costs one Claude call per question (~20 calls by default);
the ~13K-token system prompt prefix is cached, so runs after the first are much cheaper.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time

# (topic, question) -- the 15 topics identified as the live interview surface, plus the
# shape-only categories (definition/career/behavioral) that have their own formatting rules.
QUESTION_BANK: list[tuple[str, str]] = [
    ("github-actions", "Can you explain where you have used GitHub Actions?"),
    ("secrets", "How would you handle secrets in GitHub Actions?"),
    ("secrets", "How do you manage secrets across multiple clouds?"),
    ("terraform", "How would you structure Terraform for multiple teams?"),
    ("multi-cloud", "How are you going to design the terraform for multi cloud systems?"),
    ("eks", "How would you design a highly available EKS platform?"),
    ("migration", "How would you migrate 500 applications from EC2 to EKS?"),
    ("deployment", "Blue-green vs canary deployment - which would you choose?"),
    ("ha-dr", "What happens if the entire region fails?"),
    ("observability", "How would you establish observability standards across services?"),
    ("observability", "Why would you use Datadog instead of CloudWatch?"),
    ("devsecops", "How would you add security gates to a CI/CD pipeline?"),
    ("troubleshooting", "Payment service is throwing 503s and pods are OOMKilled, walk me through your triage"),
    ("platform-eng", "How do you prevent 50 teams from building 50 different pipelines?"),
    ("cost", "How would you reduce our AWS bill?"),
    ("networking", "How would you design DNS and load balancing for multi-region?"),
    ("failure", "What if the Terraform state file gets corrupted?"),
    ("why-not", "Why not just use CloudFormation instead of Terraform?"),
    ("definition", "What is OpenTelemetry?"),
    ("career", "Tell me about yourself"),
    # Support DevOps Engineer JD surface (added 2026-09-03) -- operational tickets, not design.
    ("support-ecs", "An application intermittently fails to deploy to ECS, how do you debug it?"),
    ("support-harness", "A Harness pipeline stage keeps failing on deploy, walk me through it"),
    ("support-iam", "A role is getting AccessDenied calling S3, how do you resolve it?"),
    ("support-network", "After a security group change the app can't reach RDS, what do you check?"),
    ("support-tls", "Service onboarding fails with a TLS handshake error, how do you troubleshoot?"),
    ("support-rca", "How do you do root cause analysis for a recurring deployment failure?"),
    # Seven-shape coverage (added 2026-09-03) -- one probe per master shape.
    ("star", "Tell me about a time you resolved a difficult production issue"),
    ("star", "Tell me about a technical mistake you made"),
    ("incident", "Production is down. What do you do?"),
    ("knowledge", "How does Terraform work?"),
    ("rapidfire", "What is a security group?"),
    ("comparison", "ECS vs EKS - which would you choose?"),
]

_MAX_BULLET_WORDS = 14
_BULLET_RE = re.compile(r"^\s*[*\-•]\s+")
_HEADING_RE = re.compile(r"^\s*#{1,4}\s")
# Phrases the prompt explicitly bans as content-free filler. Matching one is not proof the
# answer is bad, but it is always worth a human look.
_GENERIC_PHRASES = (
    "ensure scalability", "ensure security", "ensure high availability",
    "best practices", "as needed", "it depends", "consider the requirements",
    "properly configured", "robust solution", "leverage the power",
)


def _analyse(text: str, cap: int) -> tuple[list[str], dict[str, int]]:
    """Score one answer against the format rules. Returns (flags, stats)."""
    flags: list[str] = []
    lines = text.splitlines()

    in_code = False
    bullets: list[str] = []
    prose: list[str] = []
    for raw in lines:
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line or _HEADING_RE.match(line):
            continue
        if _BULLET_RE.match(line):
            bullets.append(_BULLET_RE.sub("", line))
        # A long unbulleted line outside a code block is the paragraph failure mode. Short
        # ones are usually a diagram caption or a stray label, so they don't count.
        elif len(line.split()) > 12:
            prose.append(line)

    words = len(text.split())
    if words > cap:
        flags.append(f"OVER BUDGET: {words}w > {cap}w cap")
    if prose:
        flags.append(f"PARAGRAPH: {len(prose)} unbulleted line(s), e.g. {prose[0][:70]!r}")
    if not bullets:
        flags.append("NO BULLETS at all")

    long_bullets = [b for b in bullets if len(b.split()) > _MAX_BULLET_WORDS]
    if long_bullets:
        flags.append(
            f"LONG BULLETS: {len(long_bullets)}/{len(bullets)} over {_MAX_BULLET_WORDS}w, "
            f"e.g. {long_bullets[0][:70]!r}"
        )

    # "I'd run three replicas..." is the pre-2026-09-02 style the format change removed.
    narrated = [b for b in bullets if re.match(r"^(I|I'd|I would|We|We'd|My )\b", b)]
    if narrated:
        flags.append(f"FIRST-PERSON: {len(narrated)} bullet(s), e.g. {narrated[0][:70]!r}")

    # A bullet of one or two bare words is a topic label, not an answer -- the "Security /
    # IAM / CI/CD" failure mode.
    stubs = [b for b in bullets if len(b.rstrip(".").split()) <= 2]
    if len(stubs) >= 3:
        flags.append(f"TOPIC LIST: {len(stubs)} bare-label bullet(s), e.g. {stubs[0]!r}")

    hits = [p for p in _GENERIC_PHRASES if p in text.lower()]
    if hits:
        flags.append(f"GENERIC: {', '.join(hits)}")

    return flags, {"words": words, "bullets": len(bullets)}


async def _run_one(question: str) -> tuple[str, float]:
    from meeting_copilot.config import get_config
    from meeting_copilot.llm.answer_optimizer import AnswerOptimizer
    from meeting_copilot.llm.claude_client import ClaudeClient
    from meeting_copilot.llm.prompt_templates import build_system_prompt, build_user_prompt
    from meeting_copilot.pipeline.events import DetectedQuestion, RetrievedContext, Transcript

    cfg = get_config()
    transcript = Transcript(speaker_id="interviewer", text=question, start_time=0.0, end_time=1.0)
    detected = DetectedQuestion(
        transcript=transcript, matched_keywords=[], ends_with_question_mark=True
    )
    context = RetrievedContext(question=detected, chunks=[])

    client = ClaudeClient(cfg.llm)
    t0 = time.monotonic()
    raw = ""
    async for token in client.stream(
        build_user_prompt(context), system=build_system_prompt(question_text=question)
    ):
        raw += token
    elapsed = time.monotonic() - t0

    # Go through the optimizer so what we score is what the overlay actually renders.
    return AnswerOptimizer(cfg.llm).optimize(context, raw).text, elapsed


async def _run_bank(bank: list[tuple[str, str]]) -> tuple[int, str]:
    """Returns (flagged_count, markdown_report)."""
    from meeting_copilot.llm.prompt_templates import _CATEGORY_WORD_LIMITS, _classify_category

    report: list[str] = ["# Answer format validation\n"]
    failed = 0

    for i, (topic, question) in enumerate(bank, 1):
        category = _classify_category(question)
        cap = _CATEGORY_WORD_LIMITS[category]
        print(f"[{i}/{len(bank)}] {topic:16} {category:24} {question[:48]}")
        try:
            answer, elapsed = await _run_one(question)
        except Exception as exc:  # noqa: BLE001 -- one bad question must not kill the run
            print(f"          ERROR: {type(exc).__name__}: {exc}")
            report.append(f"\n## {topic} -- {question}\n\n**ERROR:** `{exc}`\n")
            failed += 1
            continue

        flags, stats = _analyse(answer, cap)
        status = "FAIL" if flags else "ok"
        print(f"          {status}  {stats['words']:3}w/{cap}w  "
              f"{stats['bullets']:2} bullets  {elapsed:.1f}s")
        for flag in flags:
            print(f"            - {flag}")
        if flags:
            failed += 1

        report.append(
            f"\n## {topic} -- {question}\n\n"
            f"`{category}` | {stats['words']}w of {cap}w | {stats['bullets']} bullets | "
            f"{elapsed:.1f}s | **{status}**\n\n"
            + ("".join(f"- FLAG: {f}\n" for f in flags) + "\n" if flags else "")
            + f"```\n{answer}\n```\n"
        )

    print(f"\n{len(bank) - failed}/{len(bank)} clean, {failed} flagged for review")
    return failed, "".join(report)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", metavar="TOPIC", help="run just these topics")
    ap.add_argument("--out", metavar="PATH", help="also write a full markdown report here")
    args = ap.parse_args()

    bank = QUESTION_BANK
    if args.only:
        wanted = {t.lower() for t in args.only}
        bank = [(t, q) for t, q in bank if t in wanted]
        if not bank:
            print(f"no questions match {sorted(wanted)}; topics are "
                  f"{sorted({t for t, _ in QUESTION_BANK})}")
            return 2

    failed, report = asyncio.run(_run_bank(bank))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"full report: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
