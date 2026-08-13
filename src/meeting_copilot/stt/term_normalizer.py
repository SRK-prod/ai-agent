"""Deterministic recovery of technical vocabulary that Whisper mis-transcribes.

Whisper is trained on general speech, so domain acronyms and CamelCase product names come
back phonetically mangled -- observed live in real interviews:

    "OOMCade status"            -> OOMKilled
    "crash look back off"       -> CrashLoopBackOff
    "easy-to-instances"         -> EC2 instances
    "Porties, Port, Distribution, Budgets, PDBs" -> PodDisruptionBudgets

This matters far more than a cosmetic spelling fix: the question detector scores on keyword
matches, so a mangled technical term can turn a real interview question into "not a question"
and the answer is never generated. That happened repeatedly in production use.

Deliberately NO model and NO LLM call here -- this runs on every transcript on the hot path,
so it is exact-phrase replacement first (cheap, deterministic) and only then a bounded fuzzy
pass over a small canonical vocabulary. Typical cost is well under a millisecond.
"""

from __future__ import annotations

import difflib
import re

# --- Multi-word phrase repairs -------------------------------------------------------
# Applied first, longest-first, case-insensitively. These are the high-confidence repairs:
# each left-hand side is a phonetic mangling that has essentially no legitimate meaning in
# an infrastructure interview, so replacing it is safe.
_PHRASE_FIXES: dict[str, str] = {
    # Kubernetes ---------------------------------------------------------------------
    "crash look back off": "CrashLoopBackOff",
    "crash loop back off": "CrashLoopBackOff",
    "crashloop back off": "CrashLoopBackOff",
    "crash lube back off": "CrashLoopBackOff",
    "crash loopback off": "CrashLoopBackOff",
    "oom cade": "OOMKilled",
    "oomcade": "OOMKilled",
    "oom cad": "OOMKilled",
    "oom killed": "OOMKilled",
    "oom kill": "OOMKilled",
    "o o m killed": "OOMKilled",
    "pod disruption budget": "PodDisruptionBudget",
    "port disruption budget": "PodDisruptionBudget",
    "porties port distribution budgets": "PodDisruptionBudgets",
    "port distribution budget": "PodDisruptionBudget",
    # Whisper often shatters this one into comma-separated fragments and only the trailing
    # acronym survives intact ("Porties, Port, Port, Distribution, Budgets, PDBs") -- so
    # expand the acronym itself, which is what actually carries the meaning.
    "pdbs": "PodDisruptionBudgets",
    "pdb": "PodDisruptionBudget",
    "hpa": "HorizontalPodAutoscaler (HPA)",
    "irsa": "IRSA",
    "rbac": "RBAC",
    "bm twenty five": "BM25",
    "ttft": "time-to-first-token",
    "cube control": "kubectl",
    "cube cuddle": "kubectl",
    "kube control": "kubectl",
    "cube ctl": "kubectl",
    "vectl": "kubectl",
    "cube adm": "kubeadm",
    "demon set": "DaemonSet",
    "daemon set": "DaemonSet",
    "stateful set": "StatefulSet",
    "replica set": "ReplicaSet",
    "config map": "ConfigMap",
    "name space": "namespace",
    "side car": "sidecar",
    "helm chart": "Helm chart",
    "service mesh": "service mesh",
    "horizontal pod autoscaler": "HorizontalPodAutoscaler",
    "cluster autoscaler": "Cluster Autoscaler",
    # AWS ----------------------------------------------------------------------------
    "easy to instances": "EC2 instances",
    "easy-to-instances": "EC2 instances",
    "easy to instance": "EC2 instance",
    "easy two": "EC2",
    "e c two": "EC2",
    "s three bucket": "S3 bucket",
    "s three": "S3",
    "e k s": "EKS",
    "e c s": "ECS",
    "i am role": "IAM role",
    "i am policy": "IAM policy",
    "i am permissions": "IAM permissions",
    "star permissions": "wildcard (*) permissions",
    "cloud watch": "CloudWatch",
    "cloud trail": "CloudTrail",
    "cloud front": "CloudFront",
    "cloud formation": "CloudFormation",
    "dynamo db": "DynamoDB",
    "dynamodb": "DynamoDB",
    "step functions": "Step Functions",
    "event bridge": "EventBridge",
    "secrets manager": "Secrets Manager",
    "transit gateway": "Transit Gateway",
    "direct connect": "Direct Connect",
    "route fifty three": "Route53",
    "route 53": "Route53",
    "auto scaling": "Auto Scaling",
    "well architected": "Well-Architected",
    "landing zone": "landing zone",
    # Observability / AIOps ----------------------------------------------------------
    "big panda": "BigPanda",
    "bigpanda": "BigPanda",
    "app dynamics": "AppDynamics",
    "appdynamics": "AppDynamics",
    "app dynamic": "AppDynamics",
    "new relic": "New Relic",
    "data dog": "Datadog",
    "datadog": "Datadog",
    "graphana": "Grafana",
    "grafana": "Grafana",
    "prometheus": "Prometheus",
    "promethus": "Prometheus",
    "promotheus": "Prometheus",
    "open telemetry": "OpenTelemetry",
    "opentelemetry": "OpenTelemetry",
    "otel": "OpenTelemetry",
    "open tracing": "OpenTracing",
    "geneos": "ITRS Geneos",
    "genios": "ITRS Geneos",
    "genius": "ITRS Geneos",
    "gene os": "ITRS Geneos",
    "itrs": "ITRS",
    "moogsoft": "Moogsoft",
    "mook soft": "Moogsoft",
    "dyna trace": "Dynatrace",
    "dynatrace": "Dynatrace",
    "splunk": "Splunk",
    "spelunk": "Splunk",
    "service now": "ServiceNow",
    "pager duty": "PagerDuty",
    "ops genie": "Opsgenie",
    "elastic search": "Elasticsearch",
    "log stash": "Logstash",
    "fluent bit": "Fluent Bit",
    "fluentd": "Fluentd",
    "loki": "Loki",
    "tempo": "Tempo",
    "mimir": "Mimir",
    "thanos": "Thanos",
    "alert manager": "Alertmanager",
    "prom ql": "PromQL",
    "promql": "PromQL",
    # SRE ----------------------------------------------------------------------------
    "error budget": "error budget",
    "burn rate": "burn rate",
    "golden signals": "golden signals",
    "s l o": "SLO",
    "s l i": "SLI",
    "s l a": "SLA",
    "m t t r": "MTTR",
    "m t t d": "MTTD",
    "blameless post mortem": "blameless postmortem",
    "post mortem": "postmortem",
    "run book": "runbook",
    "self healing": "self-healing",
    "closed loop": "closed-loop",
    "chaos engineering": "chaos engineering",
    # Data / streaming ---------------------------------------------------------------
    "kafka": "Kafka",
    "kaffka": "Kafka",
    "kafca": "Kafka",
    "kinesis": "Kinesis",
    "data bricks": "Databricks",
    "databricks": "Databricks",
    "air flow": "Airflow",
    "spark streaming": "Spark Streaming",
    # IaC / CI-CD --------------------------------------------------------------------
    "terra form": "Terraform",
    "terraform": "Terraform",
    "open tofu": "OpenTofu",
    "ansible": "Ansible",
    "answerable": "Ansible",
    "argo cd": "ArgoCD",
    "argocd": "ArgoCD",
    "flux cd": "FluxCD",
    "git ops": "GitOps",
    "git hub actions": "GitHub Actions",
    "git lab": "GitLab",
    "jenkins": "Jenkins",
    "blue green": "blue/green",
    # AI / GenAI ---------------------------------------------------------------------
    "bed rock": "Bedrock",
    "bedrock": "Bedrock",
    "sage maker": "SageMaker",
    "lang chain": "LangChain",
    "langchain": "LangChain",
    "lang graph": "LangGraph",
    "crew ai": "CrewAI",
    "llama index": "LlamaIndex",
    "rag": "RAG",
    "r a g": "RAG",
    "l l m": "LLM",
    "vector db": "vector database",
    "vector data base": "vector database",
    "p g vector": "pgvector",
    "open search": "OpenSearch",
    "pine cone": "Pinecone",
    "we aviate": "Weaviate",
    "q drant": "Qdrant",
    "hybrid search": "hybrid search",
    "re ranking": "reranking",
    "re ranker": "reranker",
    "model context protocol": "Model Context Protocol (MCP)",
    "m c p": "MCP",
    "co pilot": "Copilot",
    "github co pilot": "GitHub Copilot",
    "agentic": "agentic",
    "gen ai": "GenAI",
    "generative a i": "Generative AI",
    "hallucination": "hallucination",
    "grounded ness": "groundedness",
    "guard rails": "guardrails",
    "human in the loop": "human-in-the-loop",
    "fine tuning": "fine-tuning",
    "fine tune": "fine-tune",
    "prompt injection": "prompt injection",
    "embeddings": "embeddings",
    # Banking / regulated ------------------------------------------------------------
    "segregation of duties": "segregation of duties",
    "change control": "change control",
    "audit trail": "audit trail",
    "model risk management": "model risk management",
    "wells fargo": "Wells Fargo",
    # --- Wells Fargo JD vocabulary, with the manglings Whisper actually produces -----
    # AIOps platforms named explicitly in the JD
    "i t r s": "ITRS",
    "itrs genios": "ITRS Geneos",
    "itrs geneos": "ITRS Geneos",
    "geneus": "ITRS Geneos",
    "genios": "ITRS Geneos",
    "jean os": "ITRS Geneos",
    "jeneos": "ITRS Geneos",
    "big pandas": "BigPanda",
    "bigpandas": "BigPanda",
    "big panda's": "BigPanda",
    "app dynamics apm": "AppDynamics APM",
    "appdynamic": "AppDynamics",
    "splunck": "Splunk",
    "splung": "Splunk",
    "s plunk": "Splunk",
    "splunk enterprise": "Splunk Enterprise",
    "splunk observability": "Splunk Observability",
    "open telemetry collector": "OpenTelemetry Collector",
    "otel collector": "OpenTelemetry Collector",
    "o tel": "OpenTelemetry",
    "otlp": "OTLP",
    # AI/ML for IT ops -- JD explicitly names these four
    "anomaly detection": "anomaly detection",
    "event correlation": "event correlation",
    "intelligent alerting": "intelligent alerting",
    "capacity forecasting": "capacity forecasting",
    "noise reduction": "noise reduction",
    "alert fatigue": "alert fatigue",
    "alert storm": "alert storm",
    "root cause analysis": "root cause analysis",
    "r c a": "RCA",
    "topology": "topology",
    "service topology": "service topology",
    "dependency graph": "dependency graph",
    # Self-healing / closed-loop -- the centre of gravity for this role
    "self healing systems": "self-healing systems",
    "closed loop remediation": "closed-loop remediation",
    "closed loop automation": "closed-loop automation",
    "auto remediation": "automated remediation",
    "automated remediation": "automated remediation",
    "intelligent runbook": "intelligent runbook",
    "automated runbook generation": "automated runbook generation",
    "runbook automation": "runbook automation",
    "circuit breaker": "circuit breaker",
    "blast radius": "blast radius",
    "idempotent": "idempotent",
    "human approval": "human approval",
    # Automation frameworks / event-driven
    "ansable": "Ansible",
    "hansible": "Ansible",
    "ansible tower": "Ansible Tower",
    "ansible playbook": "Ansible playbook",
    "event driven architecture": "event-driven architecture",
    "event driven": "event-driven",
    "message queue": "message queue",
    "pub sub": "pub/sub",
    "web hook": "webhook",
    # Streaming / data pipelines -- JD names Kafka specifically
    "kavka": "Kafka",
    "kafka lag": "Kafka lag",
    "consumer lag": "consumer lag",
    "kafka connect": "Kafka Connect",
    "kafka streams": "Kafka Streams",
    "confluent": "Confluent",
    "zoo keeper": "ZooKeeper",
    "zookeeper": "ZooKeeper",
    "telemetry ingestion": "telemetry ingestion",
    "data pipeline": "data pipeline",
    "streaming platform": "streaming platform",
    "back pressure": "backpressure",
    "flink": "Flink",
    "pulsar": "Pulsar",
    # SRE vocabulary (careful: never map the ordinary word "slow" to SLO)
    "s l o s": "SLOs",
    "s l i s": "SLIs",
    "service level objective": "Service Level Objective (SLO)",
    "service level indicator": "Service Level Indicator (SLI)",
    "service level agreement": "Service Level Agreement (SLA)",
    "error budgets": "error budgets",
    "burn rate alerting": "burn-rate alerting",
    "reliability engineering": "reliability engineering",
    "site reliability": "Site Reliability",
    "s r e": "SRE",
    "toil": "toil",
    "on call": "on-call",
    "incident commander": "Incident Commander",
    "mean time to recovery": "MTTR",
    "mean time to detect": "MTTD",
    # GenAI-assisted operations -- JD names these three explicitly
    "incident summarization": "incident summarization",
    "incident summarisation": "incident summarization",
    "knowledge mining": "knowledge mining",
    "foundation model": "foundation model",
    "foundation models": "foundation models",
    "large language model": "large language model (LLM)",
    "agentic a i": "Agentic AI",
    "agentic ai": "Agentic AI",
    "agent building": "agent building",
    "multi agent": "multi-agent",
    "tool calling": "tool calling",
    "function calling": "function calling",
    "github co pilot": "GitHub Copilot",
    "git hub copilot": "GitHub Copilot",
    "chain of thought": "chain-of-thought",
    "context window": "context window",
    "token limit": "token limit",
    "prompt engineering": "prompt engineering",
    "semantic search": "semantic search",
    "knowledge base": "knowledge base",
    "retrieval augmented generation": "Retrieval Augmented Generation (RAG)",
    # Documentation / ways of working named in Job Expectations
    "conflu ence": "Confluence",
    "confluance": "Confluence",
    "jira": "JIRA",
    "architecture diagram": "architecture diagram",
    "target state architecture": "target-state architecture",
    "operating model": "operating model",
    "stakeholder management": "stakeholder management",
    # Kubernetes / microservices (JD: "cloud-native architectures")
    "cubernetes": "Kubernetes",
    "kubernets": "Kubernetes",
    "kuber netes": "Kubernetes",
    "k eight s": "K8s",
    "k8s": "K8s",
    "micro services": "microservices",
    "micro service": "microservice",
    "cloud native": "cloud-native",
    "container orchestration": "container orchestration",
    "readiness probe": "readiness probe",
    "liveness probe": "liveness probe",
    "node pool": "node pool",
    "control plane": "control plane",
    "ingress controller": "ingress controller",
    "network policy": "NetworkPolicy",
    "resource quota": "ResourceQuota",
    "admission controller": "admission controller",
}

# Canonical single tokens for the bounded fuzzy pass. Kept small on purpose: fuzzy matching
# a large vocabulary produces false positives on ordinary English, which is worse than
# leaving a term slightly wrong.
_CANONICAL_TOKENS: tuple[str, ...] = (
    "Kubernetes", "Prometheus", "Grafana", "Datadog", "Splunk", "Kafka", "Terraform",
    "Ansible", "Bedrock", "OpenTelemetry", "Databricks", "Kinesis", "DynamoDB",
    "CloudWatch", "Lambda", "Aurora", "OpenSearch", "Elasticsearch", "Jenkins",
    "ArgoCD", "Istio", "Linkerd", "Karpenter", "Kyverno", "BigPanda", "Moogsoft",
    "Dynatrace", "AppDynamics", "ServiceNow", "PagerDuty", "Helm", "Loki", "Jaeger",
    "SageMaker", "LangChain", "Anthropic", "Claude",
)

# Tokens that must never be fuzzy-rewritten: ordinary English that happens to sit close to
# a technical term in edit distance.
_FUZZY_STOPWORDS = {
    "container", "containers", "cluster", "clusters", "service", "services", "system",
    "systems", "platform", "platforms", "process", "processes", "problem", "customer",
    "company", "critical", "practice", "produce", "product", "question", "solution",
    "instance", "instances", "database", "databases", "architect", "architecture",
    "management", "monitoring", "operations", "engineering", "experience", "important",
}

_FUZZY_MIN_LEN = 5
_FUZZY_THRESHOLD = 0.86  # deliberately strict -- a wrong "fix" is worse than no fix

_word_re = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


def _build_phrase_pattern() -> re.Pattern[str]:
    """One combined alternation, longest phrase first.

    Built once and reused. Single-pass matters for correctness, not just speed: applying
    each fix in its own pass lets an earlier replacement's OUTPUT be re-matched by a later
    rule, producing cascades like "itrs genios" -> "ITRS Geneos" -> "ITRS ITRS Geneos".
    A single pass consumes each position exactly once, so that cannot happen.
    """
    alternatives = sorted(_PHRASE_FIXES, key=len, reverse=True)
    # Allow whitespace / commas / hyphens between words, since Whisper punctuates
    # unpredictably ("crash look back off" vs "crash, look back off").
    escaped = [re.escape(a).replace(r"\ ", r"[\s,\-]+") for a in alternatives]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


_PHRASE_PATTERN: re.Pattern[str] | None = None


def _apply_phrase_fixes(text: str) -> tuple[str, int]:
    global _PHRASE_PATTERN
    if _PHRASE_PATTERN is None:
        _PHRASE_PATTERN = _build_phrase_pattern()

    fixes = 0
    # Normalise the matched span back to a canonical dict key: collapse internal
    # whitespace/commas/hyphens to single spaces so "crash, look  back off" finds
    # "crash look back off".
    def repl(match: re.Match[str]) -> str:
        nonlocal fixes
        raw = match.group(0)
        key = re.sub(r"[\s,\-]+", " ", raw).strip().lower()
        canonical = _PHRASE_FIXES.get(key)
        if canonical is None:
            # Hyphenated source ("easy-to-instances") whose key uses spaces, or vice versa.
            canonical = _PHRASE_FIXES.get(key.replace(" ", ""))
        if canonical is None or canonical == raw:
            return raw
        fixes += 1
        return canonical

    return _PHRASE_PATTERN.sub(repl, text), fixes


def _apply_fuzzy(text: str, canonical_lookup: dict[str, str]) -> tuple[str, int]:
    fixes = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal fixes
        word = match.group(0)
        lowered = word.lower()
        if len(word) < _FUZZY_MIN_LEN or lowered in _FUZZY_STOPWORDS:
            return word
        if lowered in canonical_lookup:  # already correct
            return word
        close = difflib.get_close_matches(
            lowered, canonical_lookup.keys(), n=1, cutoff=_FUZZY_THRESHOLD
        )
        if not close:
            return word
        fixes += 1
        return canonical_lookup[close[0]]

    return _word_re.sub(repl, text), fixes


def normalize(text: str) -> tuple[str, int]:
    """Repair mis-transcribed technical vocabulary.

    Returns (normalized_text, number_of_repairs). The repair count doubles as a weak
    "this looked technical" signal the question detector can use to avoid discarding an
    otherwise-garbled but genuinely technical question.
    """
    if not text or not text.strip():
        return text, 0

    normalized, phrase_fixes = _apply_phrase_fixes(text)
    lookup = {t.lower(): t for t in _CANONICAL_TOKENS}
    normalized, fuzzy_fixes = _apply_fuzzy(normalized, lookup)
    return normalized, phrase_fixes + fuzzy_fixes


def technical_term_count(text: str) -> int:
    """How many canonical technical terms appear. Used as a question-rescue signal:
    grammatically mangled speech that is dense in real technical terms is far more likely
    to be a real question than background noise."""
    lowered = text.lower()
    canonical_values = {v.lower() for v in _PHRASE_FIXES.values()}
    canonical_values.update(t.lower() for t in _CANONICAL_TOKENS)
    return sum(1 for term in canonical_values if term in lowered)
