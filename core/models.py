"""Pydantic v2 domain, telemetry, checkpoint, and benchmark schemas for ZERO-TAX."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

USD_TO_INR: float = 84.50


class TeacherMode(str, Enum):
    AUTO = "auto"
    SIMULATED = "simulated"
    GEMINI = "gemini"
    OPENAI = "openai"


class ProductCategory(str, Enum):
    FOOTWEAR = "FOOTWEAR"
    ELECTRONICS = "ELECTRONICS"
    APPAREL = "APPAREL"
    HOME_KITCHEN = "HOME_KITCHEN"
    ACCESSORIES = "ACCESSORIES"


class IntentLabel(str, Enum):
    DAMAGED_ITEM = "DAMAGED_ITEM"
    LATE_DELIVERY = "LATE_DELIVERY"
    WRONG_SIZE = "WRONG_SIZE"
    FRAUD_SUSPECT = "FRAUD_SUSPECT"
    EMPTY_BOX_SCAM = "EMPTY_BOX_SCAM"


class PolicyAction(str, Enum):
    INSTANT_REFUND = "INSTANT_REFUND"
    REPLACEMENT_PICKUP = "REPLACEMENT_PICKUP"
    MANUAL_INSPECTION = "MANUAL_INSPECTION"


class ExecutionMode(str, Enum):
    LLM_TEACHER = "LLM_TEACHER"
    SKLEARN_STUDENT = "SKLEARN_STUDENT"
    LOW_CONF_FALLBACK = "LOW_CONF_FALLBACK"


class NodeStatus(str, Enum):
    TEACHER_ACTIVE = "TEACHER_ACTIVE"
    CANDIDATE_READY = "CANDIDATE_READY"
    STUDENT_ACTIVE = "STUDENT_ACTIVE"
    DEMOTED_DRIFT = "DEMOTED_DRIFT"


class SupportTicket(BaseModel):
    ticket_id: str
    customer_name: str = "Aarav Sharma"
    city: str = "Bengaluru"
    order_id: str = "ORD-10001"
    category: ProductCategory
    item_price_inr: float = Field(..., ge=0.0)
    customer_return_rate: float = Field(..., ge=0.0, le=1.0)
    days_since_delivery: int = Field(..., ge=0)
    ticket_text: str
    split: Literal["train_stream", "eval_holdout", "drift_wave"] = "train_stream"
    is_drift: bool = False
    ground_truth_intent: Optional[IntentLabel] = None
    ground_truth_policy: Optional[PolicyAction] = None


class DatasetBundle(BaseModel):
    seed: int
    train_stream: list[SupportTicket] = Field(default_factory=list)
    eval_holdout: list[SupportTicket] = Field(default_factory=list)
    drift_wave: list[SupportTicket] = Field(default_factory=list)

    @property
    def all_tickets(self) -> list[SupportTicket]:
        return [*self.train_stream, *self.eval_holdout, *self.drift_wave]


class NodePrediction(BaseModel):
    node_name: str
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    execution_mode: ExecutionMode
    latency_ms: float = Field(..., ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    cost_inr: float = Field(default=0.0, ge=0.0)
    teacher_model: Optional[str] = None
    reply_text: Optional[str] = None


class ShadowAuditRecord(BaseModel):
    ticket_id: str
    node_name: str
    student_prediction: str
    student_confidence: float = Field(..., ge=0.0, le=1.0)
    teacher_prediction: str
    agreed: bool
    audit_source: Literal["SHADOW_SAMPLE", "LOW_CONF_AUDIT"] = "SHADOW_SAMPLE"
    is_drift_sample: bool = False
    timestamp: str


class PromotionCandidate(BaseModel):
    node_name: str
    sample_count: int = Field(..., ge=1)
    cv_accuracy: float = Field(..., ge=0.0, le=1.0)
    ece: float = Field(..., ge=0.0, le=1.0)
    model_version: int = Field(..., ge=1)
    class_distribution: dict[str, int] = Field(default_factory=dict)
    estimated_cost_savings_pct: float = Field(..., ge=0.0, le=100.0)
    estimated_latency_savings_ms: float = Field(..., ge=0.0)


class NodeDistillationMetrics(BaseModel):
    node_name: str
    status: NodeStatus = NodeStatus.TEACHER_ACTIVE
    is_promoted: bool = False
    trace_count: int = 0
    traces_since_last_train: int = 0
    model_version: int = 0
    cv_accuracy: Optional[float] = None
    ece: Optional[float] = None
    teacher_calls: int = 0
    student_calls: int = 0
    low_conf_fallbacks: int = 0
    shadow_audit_count: int = 0
    rolling_shadow_agreement: Optional[float] = None
    promotion_count: int = 0
    demotion_count: int = 0
    total_cost_usd: float = 0.0
    total_cost_inr: float = 0.0
    avg_latency_ms: float = 0.0


class TicketTelemetry(BaseModel):
    ticket_id: str
    thread_id: str
    intent_result: NodePrediction
    policy_result: NodePrediction
    reply_result: NodePrediction
    reply_text: str
    shadow_audits: list[ShadowAuditRecord] = Field(default_factory=list)
    total_cost_usd: float = Field(..., ge=0.0)
    total_cost_inr: float = Field(..., ge=0.0)
    total_latency_ms: float = Field(..., ge=0.0)
    baseline_cost_usd: float = Field(default=0.0, ge=0.0)
    baseline_cost_inr: float = Field(default=0.0, ge=0.0)
    baseline_latency_ms: float = Field(default=0.0, ge=0.0)
    is_drift: bool = False


class CheckpointSummary(BaseModel):
    checkpoint_id: str
    parent_checkpoint_id: Optional[str] = None
    thread_id: str
    step: Optional[int] = None
    created_at: str
    ticket_id: Optional[str] = None
    node_statuses: dict[str, str] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    seed: int
    normal_count: int
    drift_count: int
    teacher_mode: str
    baseline_total_cost_usd: float
    baseline_total_cost_inr: float
    zerotax_total_cost_usd: float
    zerotax_total_cost_inr: float
    cost_reduction_pct: float
    distillable_baseline_cost_inr: float
    distillable_zerotax_cost_inr: float
    distillable_cost_reduction_pct: float
    baseline_avg_latency_ms: float
    zerotax_avg_latency_ms: float
    latency_reduction_pct: float
    intent_accuracy: float
    policy_accuracy: float
    evaporation_rate_pct: float
    fallback_rate: float
    drift_detection_ticket_idx: Optional[int] = None
    drift_recovery_ticket_idx: Optional[int] = None
    post_recovery_accuracy: float
    per_node_metrics: dict[str, Any] = Field(default_factory=dict)
