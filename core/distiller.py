"""Online scikit-learn distillation, 5-fold CV, 10-bin ECE calibration, and drift circuit-breaker."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core.models import (
    ExecutionMode,
    NodeDistillationMetrics,
    NodeStatus,
    PromotionCandidate,
    ShadowAuditRecord,
)

REQUIRED_POLICY_KEYS: tuple[str, ...] = (
    "category",
    "item_price_inr",
    "customer_return_rate",
    "days_since_delivery",
    "ticket_text",
    "intent",
)

POLICY_NUMERIC_COLS: list[str] = [
    "item_price_inr",
    "customer_return_rate",
    "days_since_delivery",
]
POLICY_CATEGORICAL_COLS: list[str] = ["category", "intent"]

CORE_CLASSES: dict[str, set[str]] = {
    "intent_classifier": {"DAMAGED_ITEM", "LATE_DELIVERY", "WRONG_SIZE", "FRAUD_SUSPECT"},
    "policy_router": {"INSTANT_REFUND", "REPLACEMENT_PICKUP", "MANUAL_INSPECTION"},
}


class DistillerConfig(BaseModel):
    min_samples_for_promotion: int = Field(default=45, ge=5)
    min_samples_per_class: int = Field(default=5, ge=2)
    min_cv_accuracy: float = Field(default=0.95, ge=0.0)
    max_ece: float = Field(default=0.10, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    shadow_audit_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    drift_agreement_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    shadow_window_size: int = Field(default=10, ge=2)
    min_audits_for_breaker: int = Field(default=4, ge=1)
    retrain_interval_traces: int = Field(default=10, ge=1)
    audit_on_low_conf_fallback: bool = True
    cv_folds: int = Field(default=5, ge=2)
    ece_bins: int = Field(default=10, ge=2)


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute multi-class top-label Expected Calibration Error across n_bins."""
    if len(y_true) == 0:
        return 0.0
    confidences = np.max(y_proba, axis=1)
    pred_indices = np.argmax(y_proba, axis=1)
    preds = classes[pred_indices]
    accuracies = (preds == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y_true))
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if np.any(mask):
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (float(np.sum(mask)) / n) * abs(bin_acc - bin_conf)
    return round(float(ece), 6)


def _validate_policy_dict(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError(f"Expected dict for policy_router features, got {type(item).__name__}")
    for key in REQUIRED_POLICY_KEYS:
        if key not in item:
            raise KeyError(f"Missing required policy_router feature key: '{key}'")
    return {
        "category": str(item["category"]),
        "intent": str(item["intent"]),
        "item_price_inr": float(max(0.0, min(float(item["item_price_inr"]), 1e7))),
        "customer_return_rate": float(max(0.0, min(float(item["customer_return_rate"]), 1.0))),
        "days_since_delivery": float(max(0, min(int(item["days_since_delivery"]), 3650))),
    }


def _policy_records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    normalized = [_validate_policy_dict(r) for r in records]
    return pd.DataFrame(normalized, columns=POLICY_CATEGORICAL_COLS + POLICY_NUMERIC_COLS)


class NodeDistiller:
    """Manages online trace collection, CV/ECE validation, promotion, and drift demotion for one node."""

    def __init__(
        self,
        node_name: str,
        config: Optional[DistillerConfig] = None,
        seed: int = 42,
    ) -> None:
        self.node_name = node_name
        self.config = config or DistillerConfig()
        self.seed = seed
        self._init_runtime_state()

    def _init_runtime_state(self) -> None:
        self._traces_x: list[Any] = []
        self._traces_y: list[str] = []
        self._trace_ids: list[str] = []
        self._observed_trace_ids: set[str] = set()
        self._traces_is_drift: list[bool] = []
        self._classes_seen_before_demotion: set[str] = set()
        self._drift_classes_seen_since_demotion: Counter[str] = Counter()

        self._status: NodeStatus = NodeStatus.TEACHER_ACTIVE
        self._is_promoted: bool = False
        self._rejected_once: bool = False
        self._traces_since_last_train: int = 0
        self._has_evaluated: bool = False
        self._model_version: int = 0
        self._cv_accuracy: Optional[float] = None
        self._ece: Optional[float] = None

        self._fitted_model: Optional[Pipeline] = None
        self._latest_candidate: Optional[PromotionCandidate] = None

        self._teacher_calls: int = 0
        self._student_calls: int = 0
        self._low_conf_fallbacks: int = 0
        self._shadow_audit_count: int = 0
        self._shadow_window: deque[bool] = deque(maxlen=self.config.shadow_window_size)
        self._promotion_count: int = 0
        self._demotion_count: int = 0
        self._total_cost_usd: float = 0.0
        self._total_cost_inr: float = 0.0
        self._total_latency_ms: float = 0.0

    def reset(self) -> None:
        """Reset all learned models, traces, and metrics back to cold start."""
        self._init_runtime_state()

    def _build_pipeline(self) -> Pipeline:
        if self.node_name == "intent_classifier":
            return Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=(1, 1),
                            sublinear_tf=True,
                            stop_words="english",
                            min_df=2,
                            max_features=4000,
                        ),
                    ),
                    (
                        "clf",
                        LogisticRegression(
                            C=100.0,
                            max_iter=1000,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
        if self.node_name == "policy_router":
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        POLICY_CATEGORICAL_COLS,
                    ),
                    ("num", "passthrough", POLICY_NUMERIC_COLS),
                ],
                remainder="drop",
            )
            return Pipeline(
                [
                    ("prep", preprocessor),
                    (
                        "clf",
                        GradientBoostingClassifier(
                            n_estimators=80,
                            max_depth=3,
                            learning_rate=0.12,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
        raise ValueError(f"Unsupported distillable node: {self.node_name}")

    def _prepare_X(self, items: list[Any]) -> Any:
        if self.node_name == "intent_classifier":
            return [str(x or "") for x in items]
        if self.node_name == "policy_router":
            return _policy_records_to_df(items)
        raise ValueError(f"Unsupported distillable node: {self.node_name}")

    def add_trace(
        self,
        x: Union[str, dict[str, Any]],
        y: str,
        ticket_id: str,
        is_drift: bool = False,
    ) -> None:
        """Record a Teacher-labeled (input, output) pair for online distillation."""
        if self.node_name == "reply_writer":
            return
        if self.node_name == "policy_router":
            _validate_policy_dict(x)  # type: ignore[arg-type]
        self._traces_x.append(x)
        self._traces_y.append(str(y))
        self._trace_ids.append(ticket_id)
        self._observed_trace_ids.add(ticket_id)
        self._traces_is_drift.append(bool(is_drift))
        self._traces_since_last_train += 1
        if self._status == NodeStatus.DEMOTED_DRIFT:
            self._drift_classes_seen_since_demotion[str(y)] += 1

    def record_observed_trace(self, ticket_id: str) -> None:
        """Record an observed execution trace ID in Store metrics without adding student labels to Teacher training."""
        if self.node_name == "reply_writer":
            return
        self._observed_trace_ids.add(ticket_id)

    def is_ready_for_evaluation(self) -> bool:
        """Return True when the node meets trace thresholds to attempt CV/ECE promotion."""
        if self.node_name == "reply_writer":
            return False
        if self._is_promoted:
            return False
        if len(self._traces_y) < self.config.min_samples_for_promotion:
            return False
        counts = Counter(self._traces_y)
        if self.node_name in CORE_CLASSES and not CORE_CLASSES[self.node_name].issubset(counts.keys()):
            return False
        if len(counts) < 2 or min(counts.values()) < self.config.min_samples_per_class:
            return False
        required_interval = self.config.retrain_interval_traces + (1 if self._rejected_once else 0)
        if self._has_evaluated and self._traces_since_last_train < required_interval:
            return False
        if self._status == NodeStatus.DEMOTED_DRIFT:
            if self._traces_since_last_train < self.config.retrain_interval_traces:
                return False
            new_classes = set(counts.keys()) - self._classes_seen_before_demotion
            for cls_name in new_classes:
                if self._drift_classes_seen_since_demotion[cls_name] < self.config.min_samples_per_class:
                    return False
        return True

    def evaluate_and_fit(self) -> Optional[PromotionCandidate]:
        """Run 5-fold StratifiedKFold CV and 10-bin ECE; fit full student if criteria pass."""
        if self.node_name == "reply_writer":
            return None
        n_total = len(self._traces_y)
        if n_total < self.config.min_samples_for_promotion:
            return None

        counts = Counter(self._traces_y)
        if self.node_name in CORE_CLASSES and not CORE_CLASSES[self.node_name].issubset(counts.keys()):
            return None
        min_cls = min(counts.values())
        if len(counts) < 2 or min_cls < self.config.min_samples_per_class:
            return None

        n_splits = min(self.config.cv_folds, min_cls)
        if n_splits < 2:
            return None

        X_all = self._prepare_X(self._traces_x)
        y_all = np.asarray(self._traces_y)
        classes = np.unique(y_all)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        cv_pipe = self._build_pipeline()
        oof_proba = cross_val_predict(
            cv_pipe,
            X_all,
            y_all,
            cv=skf,
            method="predict_proba",
        )
        oof_preds = classes[np.argmax(oof_proba, axis=1)]
        cv_acc = round(float(np.mean(oof_preds == y_all)), 6)
        ece = compute_ece(y_all, oof_proba, classes, n_bins=self.config.ece_bins)

        self._cv_accuracy = cv_acc
        self._ece = ece
        self._has_evaluated = True
        self._traces_since_last_train = 0

        if cv_acc < self.config.min_cv_accuracy or ece > self.config.max_ece:
            return None

        full_pipe = self._build_pipeline()
        full_pipe.fit(X_all, y_all)
        self._fitted_model = full_pipe
        self._model_version += 1

        avg_teacher_latency = 600.0
        if self._teacher_calls > 0:
            avg_teacher_latency = max(250.0, self._total_latency_ms / max(self.total_calls, 1))

        candidate = PromotionCandidate(
            node_name=self.node_name,
            sample_count=n_total,
            cv_accuracy=cv_acc,
            ece=ece,
            model_version=self._model_version,
            class_distribution=dict(counts),
            estimated_cost_savings_pct=98.5,
            estimated_latency_savings_ms=round(max(200.0, avg_teacher_latency - 1.5), 2),
        )
        self._latest_candidate = candidate
        if not self._is_promoted:
            self._status = NodeStatus.CANDIDATE_READY
        return candidate

    def approve_promotion(self) -> None:
        """Activate the fitted scikit-learn student as primary executor for this node."""
        if self._fitted_model is None:
            raise RuntimeError(f"Cannot promote {self.node_name} before evaluate_and_fit succeeds")
        self._is_promoted = True
        self._rejected_once = False
        self._status = NodeStatus.STUDENT_ACTIVE
        self._promotion_count += 1
        self._traces_since_last_train = 0
        self._shadow_window.clear()
        self._drift_classes_seen_since_demotion.clear()

    def reject_promotion(self) -> None:
        """Keep Teacher active and enforce cooldown before re-evaluation."""
        self._is_promoted = False
        self._rejected_once = True
        self._status = NodeStatus.TEACHER_ACTIVE
        self._traces_since_last_train = 0

    def restore_pre_promotion(self) -> None:
        """Reset promotion state during checkpoint time-travel rewind."""
        self._is_promoted = False
        self._rejected_once = False
        self._status = NodeStatus.TEACHER_ACTIVE
        self._traces_since_last_train = 0
        self._shadow_window.clear()
        self._drift_classes_seen_since_demotion.clear()

    def predict(self, x: Union[str, dict[str, Any]]) -> tuple[str, float, dict[str, float]]:
        """Predict label, confidence, and normalized class probability dict."""
        if self._fitted_model is None:
            raise RuntimeError(f"NodeDistiller('{self.node_name}') is not fitted yet")

        if self.node_name == "intent_classifier":
            text = str(x or "")
            tfidf = self._fitted_model.named_steps["tfidf"]
            clf = self._fitted_model.named_steps["clf"]
            classes = [str(c) for c in clf.classes_]
            vec = tfidf.transform([text])
            if vec.nnz == 0:
                uniform = round(1.0 / len(classes), 6)
                probs = {c: uniform for c in classes}
                top = classes[0]
                diff = round(1.0 - sum(probs.values()), 6)
                probs[top] = round(probs[top] + diff, 6)
                return top, probs[top], probs
            proba_row = clf.predict_proba(vec)[0]
        elif self.node_name == "policy_router":
            _validate_policy_dict(x)  # type: ignore[arg-type]
            df = self._prepare_X([x])
            classes = [str(c) for c in self._fitted_model.classes_]
            proba_row = self._fitted_model.predict_proba(df)[0]
        else:
            raise ValueError(f"Unsupported distillable node: {self.node_name}")

        probs = {cls_name: float(p) for cls_name, p in zip(classes, proba_row)}
        total = sum(probs.values())
        if not np.isfinite(total) or total <= 0.0:
            uniform = 1.0 / len(classes)
            probs = {c: uniform for c in classes}
        else:
            probs = {c: round(v / total, 6) for c, v in probs.items()}
            top = max(probs, key=lambda k: probs[k])
            probs[top] = round(probs[top] + round(1.0 - sum(probs.values()), 6), 6)

        best_label = max(probs, key=lambda k: probs[k])
        return best_label, float(probs[best_label]), probs

    def record_shadow_audit(
        self,
        *,
        ticket_id: str,
        student_prediction: str,
        student_confidence: float,
        teacher_prediction: str,
        audit_source: str = "SHADOW_SAMPLE",
        is_drift_sample: bool = False,
    ) -> tuple[ShadowAuditRecord, bool]:
        """Log shadow comparison and trip drift circuit-breaker if rolling agreement drops below threshold."""
        agreed = str(student_prediction) == str(teacher_prediction)
        rec = ShadowAuditRecord(
            ticket_id=ticket_id,
            node_name=self.node_name,
            student_prediction=str(student_prediction),
            student_confidence=float(max(0.0, min(1.0, student_confidence))),
            teacher_prediction=str(teacher_prediction),
            agreed=agreed,
            audit_source=audit_source,  # type: ignore[arg-type]
            is_drift_sample=bool(is_drift_sample),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._shadow_audit_count += 1
        self._shadow_window.append(agreed)

        tripped = False
        if (
            self._is_promoted
            and len(self._shadow_window) >= self.config.min_audits_for_breaker
        ):
            rolling_agr = sum(self._shadow_window) / len(self._shadow_window)
            if rolling_agr < self.config.drift_agreement_threshold:
                tripped = True
                self._classes_seen_before_demotion = set(self._traces_y)
                self._drift_classes_seen_since_demotion.clear()
                self._is_promoted = False
                self._status = NodeStatus.DEMOTED_DRIFT
                self._demotion_count += 1
                self._traces_since_last_train = 0
                self._shadow_window.clear()

        return rec, tripped

    def record_call(
        self,
        execution_mode: ExecutionMode,
        cost_usd: float,
        cost_inr: float,
        latency_ms: float,
    ) -> None:
        if execution_mode == ExecutionMode.LLM_TEACHER:
            self._teacher_calls += 1
        elif execution_mode == ExecutionMode.SKLEARN_STUDENT:
            self._student_calls += 1
        elif execution_mode == ExecutionMode.LOW_CONF_FALLBACK:
            self._low_conf_fallbacks += 1
        self._total_cost_usd += float(cost_usd)
        self._total_cost_inr += float(cost_inr)
        self._total_latency_ms += float(latency_ms)

    @property
    def total_calls(self) -> int:
        return self._teacher_calls + self._student_calls + self._low_conf_fallbacks

    @property
    def is_promoted(self) -> bool:
        return self._is_promoted

    @property
    def status(self) -> NodeStatus:
        return self._status

    @property
    def latest_candidate(self) -> Optional[PromotionCandidate]:
        return self._latest_candidate

    def get_metrics(self) -> NodeDistillationMetrics:
        calls = self.total_calls
        rolling_agr: Optional[float] = None
        if self._shadow_window:
            rolling_agr = round(sum(self._shadow_window) / len(self._shadow_window), 6)
        return NodeDistillationMetrics(
            node_name=self.node_name,
            status=self._status,
            is_promoted=self._is_promoted,
            trace_count=max(len(self._traces_y), len(self._observed_trace_ids)),
            traces_since_last_train=self._traces_since_last_train,
            model_version=self._model_version,
            cv_accuracy=self._cv_accuracy,
            ece=self._ece,
            teacher_calls=self._teacher_calls,
            student_calls=self._student_calls,
            low_conf_fallbacks=self._low_conf_fallbacks,
            shadow_audit_count=self._shadow_audit_count,
            rolling_shadow_agreement=rolling_agr,
            promotion_count=self._promotion_count,
            demotion_count=self._demotion_count,
            total_cost_usd=round(self._total_cost_usd, 8),
            total_cost_inr=round(self._total_cost_inr, 8),
            avg_latency_ms=round(self._total_latency_ms / calls, 4) if calls > 0 else 0.0,
        )
