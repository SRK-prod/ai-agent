"""Typed configuration: secrets from .env/environment, tunables from configs/settings.yaml.

Usage:
    from meeting_copilot.config import get_config
    cfg = get_config()
    cfg.stt.model_size
    cfg.secrets.require_hf_token()  # raises MissingCredentialError with setup instructions
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from meeting_copilot.paths import ENV_FILE, SETTINGS_FILE


class MissingCredentialError(RuntimeError):
    """Raised when a required API key/token is absent, with a pointer to setup docs."""

    def __init__(self, var_name: str, purpose: str):
        super().__init__(
            f"{var_name} is not set but is required for {purpose}. "
            f"Add it to your .env file (see .env.example) -- "
            f"see docs/installation.md for how to obtain it."
        )


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), extra="ignore")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    deepgram_api_key: str | None = Field(default=None, alias="DEEPGRAM_API_KEY")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    host: str = Field(default="127.0.0.1", alias="MEETING_COPILOT_HOST")
    port: int = Field(default=8765, alias="MEETING_COPILOT_PORT")

    def require_hf_token(self) -> str:
        if not self.hf_token:
            raise MissingCredentialError(
                "HF_TOKEN", "downloading pyannote.audio diarization/embedding models"
            )
        return self.hf_token

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise MissingCredentialError(
                "ANTHROPIC_API_KEY", "calling the Anthropic Messages API (llm.backend=api)"
            )
        return self.anthropic_api_key

    def require_deepgram_key(self) -> str:
        if not self.deepgram_api_key:
            raise MissingCredentialError(
                "DEEPGRAM_API_KEY", "cloud transcription (stt.backend=deepgram)"
            )
        return self.deepgram_api_key

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise MissingCredentialError(
                "OPENAI_API_KEY", "calling the OpenAI API (llm.backend=openai)"
            )
        return self.openai_api_key


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 30
    input_device: str | None = None
    noise_reduction: bool = True


class VadConfig(BaseModel):
    backend: Literal["silero"] = "silero"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    # Force-close a segment once speech has run this long, even with no silence yet, so a
    # continuously-talking speaker still gets transcribed incrementally instead of the whole
    # pipeline stalling until they stop. 0 disables. Only safe when STT decode cost is
    # roughly linear in audio length -- see the comment in vad/silero_vad.py.
    max_speech_ms: int = 12000


class SpeakerConfig(BaseModel):
    # Diarization exists to drop the candidate's OWN voice before it is transcribed. That is
    # only necessary when the capture stream can contain it. With the virtual-cable setup
    # (BlackHole/VB-Cable) the app captures the meeting app's OUTPUT only -- the far end --
    # so the candidate's microphone is never in this stream and is_me can never be true.
    # Disabling then costs nothing and buys back a pyannote+torch model load (~1GB RSS) and
    # ~0.3s of CPU per segment, which matters a lot on a 2-core machine that is also running
    # the video call. Set True if the capture device could ever carry your own voice
    # (system-wide loopback, an aggregate device, or a shared room mic).
    enabled: bool = True
    embedding_model: str = "pyannote/embedding"
    window_seconds: float = 5.0
    window_overlap_seconds: float = 1.0
    ignore_similarity_threshold: float = 0.75
    new_speaker_similarity_threshold: float = 0.70
    max_tracked_speakers: int = 6
    enrollment_db_path: str = "data/speaker_enrollment.sqlite3"


class SttConfig(BaseModel):
    backend: Literal["faster-whisper", "mlx-whisper", "deepgram"] = "faster-whisper"
    # deepgram only -- see stt/deepgram_engine.py for why a cloud engine exists at all.
    deepgram_model: str = "nova-3"
    # Deepgram caps keyterm prompting; the full vocabulary list is longer than the cap, and
    # sending more than this is rejected rather than truncated server-side.
    deepgram_max_keyterms: int = 100
    # A cloud call must fail fast: the pipeline can survive losing one utterance, but not a
    # request that hangs past the point where the answer would still be useful.
    cloud_timeout_seconds: float = 8.0
    # Hold the idle HTTPS connection open across a whole interview. Questions are minutes
    # apart and a cold connection costs ~1s of TLS setup vs ~250ms warm (measured) -- the
    # library default of 5s would put every single question on the cold path.
    cloud_keepalive_seconds: float = 3600.0
    model_size: str = "distil-large-v3"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    compute_type: str = "int8"
    # faster-whisper's own default is beam_size=5 -- five decode hypotheses searched in
    # parallel. On a GPU that is nearly free; on a weak CPU it multiplies decode time for
    # a marginal accuracy gain. 1 = greedy decoding.
    beam_size: int = 5
    # 0 = let CTranslate2 choose (all cores). Pin it when the machine has few cores and
    # other pipeline stages (pyannote diarization) are competing for them.
    cpu_threads: int = 0
    chunk_seconds: float = 2.5
    chunk_overlap_seconds: float = 0.5
    language: str = "en"
    vocabulary_hint: str = (
        "LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, MCP, Model Context Protocol, "
        "GenAI, Generative AI, Agentic AI, RAG, vector database, embeddings, Amazon Bedrock, "
        "Claude Sonnet, Claude Haiku, Amazon Q in Connect, Amazon Connect, DIEZ Mobile, "
        "Reach Mobile, JIRA, IAM, KMS, Lambda, DynamoDB, Aurora, OpenSearch, Terraform, "
        "Kubernetes, Databricks, Kafka, Kinesis, CI/CD, DevSecOps, OpenTofu, Rancher Desktop, "
        "AKS, EKS, RBAC, Azure DevOps, Key Vault, Azure Container Registry, Prometheus, "
        "Grafana, OpenTelemetry, Loki, PowerShell, PaaS, multi-tenant, SOP, GitOps, ArgoCD, "
        "Helm, Ansible, Istio, service mesh, "
        # AIOps / SRE / observability vocabulary (EPAM Cloud AIOps Architect)
        "AIOps, SRE, SLI, SLO, SLA, error budget, burn rate, MTTR, MTTD, toil, "
        "observability, telemetry, cardinality, anomaly detection, event correlation, "
        "root cause analysis, RCA, runbook, remediation, postmortem, blameless, "
        "OOMKilled, CrashLoopBackOff, PodDisruptionBudget, Karpenter, Kyverno, "
        "OPA Gatekeeper, Falco, Trivy, Cosign, Linkerd, Tempo, Mimir, Thanos, Cortex, "
        "Alertmanager, PromQL, EFK, ELK, Datadog, AppDynamics, New Relic, Dynatrace, "
        "BigPanda, Moogsoft, ServiceNow, Jaeger, tail sampling, RED metrics, USE method, "
        "golden signals, vector store, pgvector, OpenSearch, Pinecone, Weaviate, Qdrant, "
        "FAISS, embeddings, Titan Text Embeddings, Cohere Embed, reranking, BM25, "
        "hybrid search, chunking, grounding, hallucination, guardrails, human-in-the-loop, "
        "Bicep, ARM template, CloudFormation, Transit Gateway, ExpressRoute, Direct Connect, "
        "landing zone, SCP, permission boundary, IRSA, Managed Identity, Entra ID, FinOps, "
        # Wells Fargo Principal Engineer (AIOps) JD vocabulary
        "Splunk, ITRS Geneos, Geneos, BigPanda, Moogsoft, Dynatrace, AppDynamics, "
        "self-healing, closed-loop remediation, intelligent alerting, event-driven "
        "architecture, Kafka, Kinesis, streaming, telemetry ingestion, data pipeline, "
        "Ansible, playbook, forecasting, capacity forecasting, incident summarization, "
        "knowledge mining, automated runbook generation, GitHub Copilot, Confluence, "
        "target-state architecture, operating model, roadmap, stakeholder management, "
        "model risk management, segregation of duties, change control, audit trail, "
        "regulated, compliance, SOX, PCI, Wells Fargo, principal engineer"
    )  # primes Whisper's decoder toward this domain's proper nouns/acronyms -- measured
    # live: without this, "LangChain, LangGraph, LlamaIndex" transcribed as "line chain,
    # line graph, non-index", "Gen AI" as "gender TV", "JIRA" as "PIA". Passed as
    # initial_prompt, which biases decoding without forcing verbatim repetition.
    #
    # NOTE (2026-09-01, scripts/bench_stt_opts.py on the Windows/CPU box): this full
    # ~2100-char hint is NOT the best-performing option. It is long enough to dilute the
    # decoder's attention AND it costs decode time. Measured on one utterance, model=tiny:
    #   full hint (2121 chars)  2.13s -> "OOM killed",  "crash loop back off"   (worse)
    #   short hint (147 chars)  1.87s -> "OOMKilled",   "CrashLoopBackOff"      (better)
    #   no hint                 1.58s -> "OOM killed",  "crash loop back off"
    # So the short hint is both faster than the full one and the only variant that kept the
    # compound technical terms intact. `vocabulary_hint_short` below is what the engine uses
    # by default now; the full list is kept because it was built from real transcription
    # failures and is the right fallback on a machine fast enough not to care.

    # Deliberately short: only terms Whisper actually mis-hears PHONETICALLY, and only ones
    # that plausibly occur in these interviews. Adding more here has been measured to make
    # transcription worse, not better -- do not grow this back into a full glossary.
    vocabulary_hint_short: str = (
        "Kubernetes, EKS, AKS, OOMKilled, CrashLoopBackOff, ALB, NLB, Terraform, OpenTofu, "
        "IAM, KMS, Lambda, DynamoDB, Aurora, Bedrock, Agentic AI, RAG, CI/CD, GitOps, "
        "ArgoCD, SLO, SRE, AIOps, LangChain, LangGraph, GenAI, JIRA, Databricks"
    )
    # Which of the two above the engine passes as initial_prompt. "short" is the measured
    # default on this hardware; "full" restores the original behaviour; "none" is fastest
    # but loses the compound terms.
    vocabulary_hint_mode: Literal["short", "full", "none"] = "short"

    # DEEPGRAM KEYTERMS -- deliberately LONGER than vocabulary_hint_short, and that is not a
    # contradiction of the measurement above. That measurement is about Whisper's
    # initial_prompt, which primes a decoder: a long hint dilutes its attention and measurably
    # turned "OOMKilled" back into "OOM killed". Deepgram keyterm prompting is a keyword BOOST
    # list, not a decoder prime -- nova-3 accepts up to 100 terms and extra entries do not
    # compete for attention the same way. Using the 28-term Whisper hint on Deepgram was
    # leaving 72 slots unused while missing every proper noun this interview turns on.
    #
    # Scope rule: only terms that are (a) phonetically fragile and (b) plausible in THIS
    # interview. "Harness" is the sharpest example -- as a common English noun it will
    # transcribe as lowercase "harness" in a sentence like "our harness pipeline failed", and
    # the classifier keys on the product name.
    deepgram_keyterms: str = (
        # today's Support DevOps Engineer JD surface
        "Harness, ECS, Fargate, ECR, task definition, task role, execution role, "
        "AccessDenied, OIDC, ALB, NLB, target group, security group, NACL, PrivateLink, "
        "VPC endpoint, NAT gateway, ENI, subnet, Route 53, ACM, SNI, TLS handshake, "
        "CloudTrail, CloudWatch, Secrets Manager, SCP, permission boundary, IAM, KMS, "
        "policy simulator, assume role, trust policy, least privilege, "
        "circuit breaker, health check, rollback, blue-green, canary, "
        # container / platform terms that mis-transcribe phonetically
        "Kubernetes, EKS, AKS, GKE, OOMKilled, CrashLoopBackOff, kubectl, Helm, "
        "Terraform, OpenTofu, GitOps, ArgoCD, CI/CD, "
        # AWS services that get mangled
        "Lambda, DynamoDB, Aurora, RDS, S3, EC2, Bedrock, "
        # role / practice vocabulary
        "SLO, SLI, SRE, AIOps, RCA, root cause, runbook, on-call, observability, "
        "Prometheus, Grafana, Datadog, Splunk, JIRA, Agentic AI, RAG"
    )

    @property
    def keyterms(self) -> list[str]:
        """Keyterm list for the cloud backend. Falls back to the Whisper hint if the
        Deepgram list is somehow blank, so this can never silently send nothing."""
        source = self.deepgram_keyterms or self.vocabulary_hint_short or ""
        seen: set[str] = set()
        out: list[str] = []
        for term in (t.strip() for t in source.split(",")):
            key = term.lower()
            if term and key not in seen:
                seen.add(key)
                out.append(term)
        return out

    @property
    def initial_prompt(self) -> str | None:
        """The initial_prompt actually handed to faster-whisper -- see vocabulary_hint_mode."""
        if self.vocabulary_hint_mode == "none":
            return None
        if self.vocabulary_hint_mode == "full":
            return self.vocabulary_hint
        return self.vocabulary_hint_short


class QuestionDetectorConfig(BaseModel):
    backend: Literal["rule_based"] = "rule_based"
    denylist_phrases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    min_words_for_bare_question_mark: int = 4  # filters filler like "Okay, no?" from triggering
    # the full pipeline on a "?" alone; a keyword match always triggers regardless of length


class KnowledgeConfig(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 75
    collection_name: str = "meeting_copilot_knowledge"


class RetrievalConfig(BaseModel):
    enabled: bool = False  # disabled: retrieval-grounded confidence was mislabeling good
    # answers as low-confidence whenever the knowledge base didn't happen to cover a topic
    # (e.g. a correct ALB vs NLB answer scored 22%). Pure-LLM mode answers from Claude's own
    # expertise with confidence reflecting factual certainty instead.
    top_k_vector: int = 20
    top_k_final: int = 5
    hybrid_alpha: float = 0.6


class QaBankConfig(BaseModel):
    enabled: bool = False  # disabled alongside retrieval.enabled -- see note above
    collection_name: str = "meeting_copilot_qa_bank"
    similarity_threshold: float = 0.90  # biased toward precision: a wrong-topic bank answer in
    # a real interview is worse than the ~4-5s live-generation fallback. True near-exact
    # matches score ~0.95-0.99; see configs/settings.yaml for the incident that set this.


class LlmConfig(BaseModel):
    backend: Literal["cli", "api", "openai"] = "cli"
    cli_binary: str = "claude"
    cli_timeout_seconds: int = 180
    model: str = "claude-haiku-4-5-20251001"  # fastest model; answer quality is driven
    # mainly by the interview-format system prompt (see llm/prompt_templates.py)
    # Tried in order when the primary model fails every retry. Audited 2026-08-27: a single
    # model with no fallback was a single point of failure for the whole app -- a sustained
    # provider issue left the overlay blank mid-interview with nothing to show. Empty list
    # disables fallback.
    fallback_models: list[str] = ["claude-sonnet-4-5-20250929"]
    # Hard ceiling on ONE streaming generation. The API path previously had no timeout at
    # all (only the CLI backend did), so a hung connection could stall a question forever.
    # Generous relative to the ~1.05s P50 / 1.5s max time-to-first-token measured in a real
    # session -- this is a stall guard, not a latency control.
    stream_timeout_seconds: float = 45.0
    openai_model: str = "gpt-4o"  # only used when backend=openai
    persona: str = (
        "You are a candidate with 14+ years of technology architecture experience "
        "interviewing for a senior-band Enterprise / AI Solution Architect role (also "
        "framed as Principal SRE / DevOps Architect / Engineering Manager). Your depth "
        "spans Agentic AI and GenAI solution architecture on AWS (Anthropic Claude via "
        "Amazon Bedrock, prompt engineering, tool/function calling, agent orchestration, "
        "evaluation), RAG architecture (vector databases, embeddings, knowledge "
        "ingestion), AI governance/security/model risk/auditability in regulated "
        "financial services, and the underlying platform engineering (AWS, Kubernetes, "
        "Terraform, CI/CD, observability, FinOps, SRE)."
    )
    answer_min_words: int = 30
    answer_max_words: int = 320  # ~2 min spoken -- the right length for a senior interview
    # answer. Measured: a 1157-word answer took 23.6s to generate and 7.7 min to say aloud,
    # which is both too slow to read live and too long to be a good answer. Tight answers
    # that invite follow-ups read as more senior than exhaustive monologues.
    max_tokens: int = 4200  # caps generation time; sized for answer_max_words prose plus
    # headroom for diagrams/tables/code blocks in category-specific templates (see
    # llm/prompt_templates.py _CATEGORY_SHAPES). Raised again 2026-08-03 alongside a second
    # word-limit increase (real interview follow-ups demanded more depth than the first
    # round of ceilings) -- top tier is now ~1100 words (architecture) / ~950 (ai_genai),
    # needing ~1600-1900 tokens of prose alone before diagram/table/heading overhead
    # (ingestion overrides this higher -- see ingestion.py)
    low_confidence_threshold: float = 0.60  # 0.80 stamped decent answers "[Low Confidence]"
    # mid-interview (observed: a solid answer self-scored 75%); only flag genuinely weak ones


class CacheConfig(BaseModel):
    embedding_ttl_seconds: int = 604800
    llm_response_ttl_seconds: int = 3600


class HotkeysConfig(BaseModel):
    hide: str = "<cmd>+<shift>+h"
    pin: str = "<cmd>+<shift>+p"
    expand: str = "<cmd>+<shift>+e"
    copy_answer: str = Field(default="<cmd>+<shift>+c", alias="copy")

    model_config = {"populate_by_name": True}


class OverlayConfig(BaseModel):
    always_on_top: bool = True
    opacity: float = 0.92
    hotkeys: HotkeysConfig = Field(default_factory=HotkeysConfig)


class PerformanceTargetsConfig(BaseModel):
    speech_detection: int = 50
    stt: int = 200
    retrieval: int = 50
    llm: int = 400
    overlay: int = 50
    total: int = 800
    note: str = ""


class AppConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    question_detector: QuestionDetectorConfig = Field(default_factory=QuestionDetectorConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    qa_bank: QaBankConfig = Field(default_factory=QaBankConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    performance_targets_ms: PerformanceTargetsConfig = Field(
        default_factory=PerformanceTargetsConfig
    )
    secrets: Secrets = Field(default_factory=Secrets)

    model_config = {"arbitrary_types_allowed": True}


def _load_yaml(path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_config() -> AppConfig:
    """Load configs/settings.yaml + .env once per process and cache the result."""
    raw = _load_yaml(SETTINGS_FILE)
    return AppConfig(**raw, secrets=Secrets())


def reload_config() -> AppConfig:
    """Clear the cache and reload -- useful in tests that monkeypatch env/yaml."""
    get_config.cache_clear()
    return get_config()
