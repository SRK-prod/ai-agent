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


class SpeakerConfig(BaseModel):
    embedding_model: str = "pyannote/embedding"
    window_seconds: float = 5.0
    window_overlap_seconds: float = 1.0
    ignore_similarity_threshold: float = 0.75
    new_speaker_similarity_threshold: float = 0.70
    max_tracked_speakers: int = 6
    enrollment_db_path: str = "data/speaker_enrollment.sqlite3"


class SttConfig(BaseModel):
    backend: Literal["faster-whisper", "mlx-whisper"] = "faster-whisper"
    model_size: str = "distil-large-v3"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    compute_type: str = "int8"
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
