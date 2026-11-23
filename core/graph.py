"""LangGraph StateGraph engine for ZERO-TAX online distillation, parallel Send(), HITL interrupt(), and time-travel."""

from __future__ import annotations

import operator
import random
import threading
import time
from collections import defaultdict
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Send, interrupt

from core.checkpointer import SqliteSaver
from core.dataset import compute_ground_truth_policy
from core.distiller import DistillerConfig, NodeDistiller
from core.models import (
    USD_TO_INR,
    CheckpointSummary,
    ExecutionMode,
    IntentLabel,
    NodeDistillationMetrics,
    NodePrediction,
    NodeStatus,
    SupportTicket,
    TeacherMode,
    TicketTelemetry,
)
from core.teacher import INPUT_USD_PER_TOKEN, OUTPUT_USD_PER_TOKEN, LLMTeacher

DISTILLABLE_NODES: tuple[str, ...] = ("intent_classifier", "policy_router")


def _append_or_reset(existing: list[Any], incoming: list[Any]) -> list[Any]:
    """Reducer supporting per-ticket reset via '__RESET__' sentinel while merging parallel Send() outputs."""
    if not incoming:
        return list(existing)
    if incoming[0] == "__RESET__":
        return list(incoming[1:])
    return list(existing) + list(incoming)


class ZeroTaxState(TypedDict):
    ticket: dict[str, Any]
    thread_id: str
    auto_approve: bool
    audit_rate: float
    intent_result: Optional[dict[str, Any]]
    policy_result: Optional[dict[str, Any]]
    reply_result: Optional[dict[str, Any]]
    reply_text: str
    shadow_audits: Annotated[list[dict[str, Any]], _append_or_reset]
    eval_candidates: Annotated[list[dict[str, Any]], _append_or_reset]
    pending_promotions: list[dict[str, Any]]
    promotion_decisions: Annotated[list[dict[str, Any]], _append_or_reset]
    drift_demoted_nodes: Annotated[list[str], _append_or_reset]
    node_statuses: dict[str, str]
    telemetry: Optional[dict[str, Any]]


def _policy_feature_dict(ticket: SupportTicket, intent_label: str) -> dict[str, Any]:
    return {
        "category": ticket.category.value,
        "item_price_inr": float(ticket.item_price_inr),
        "customer_return_rate": float(ticket.customer_return_rate),
        "days_since_delivery": int(ticket.days_since_delivery),
        "ticket_text": ticket.ticket_text,
        "intent": str(intent_label),
    }


def _extract_interrupt_payload(raw_result: dict[str, Any]) -> Optional[dict[str, Any]]:
    items = raw_result.get("__interrupt__")
    if not items:
        return None
    first = items[0]
    val = getattr(first, "value", first)
    return dict(val) if isinstance(val, dict) else {"value": val}


class ZeroTaxEngine:
    """Orchestrates the 3-node LangGraph workflow with online scikit-learn distillation."""

    def __init__(
        self,
        db_path: str = "data/checkpoints.db",
        teacher_mode: TeacherMode = TeacherMode.AUTO,
        config: Optional[DistillerConfig] = None,
        seed: int = 42,
        simulate_latency: bool = True,
        sleep_on_simulated: bool = False,
    ) -> None:
        self.db_path = db_path
        self.config = config or DistillerConfig()
        self.seed = seed
        self._rng = random.Random(seed)
        self._lock = threading.RLock()

        self.teacher = LLMTeacher(
            mode=teacher_mode,
            simulate_latency=simulate_latency,
            sleep_on_simulated=sleep_on_simulated,
            seed=seed,
        )
        self.checkpointer = SqliteSaver(db_path)
        self.store = InMemoryStore()

        self.distillers: dict[str, NodeDistiller] = {
            "intent_classifier": NodeDistiller("intent_classifier", config=self.config, seed=seed),
            "policy_router": NodeDistiller("policy_router", config=self.config, seed=seed),
        }
        self._reply_calls: int = 0
        self._reply_cost_usd: float = 0.0
        self._reply_cost_inr: float = 0.0
        self._reply_latency_ms: float = 0.0

        self._thread_telemetry: dict[str, list[TicketTelemetry]] = defaultdict(list)
        self._thread_auto_approve_override: dict[str, bool] = {}

        self.graph = self._build_graph()
        self._sync_all_statuses_to_store()

    def _sync_all_statuses_to_store(self) -> None:
        for name, distiller in self.distillers.items():
            self.store.put(("status", name), "current", distiller.get_metrics().model_dump(mode="json"))
        self.store.put(
            ("status", "reply_writer"),
            "current",
            self._build_reply_metrics().model_dump(mode="json"),
        )

    def _build_reply_metrics(self) -> NodeDistillationMetrics:
        avg_lat = (
            round(self._reply_latency_ms / self._reply_calls, 4)
            if self._reply_calls > 0
            else 0.0
        )
        return NodeDistillationMetrics(
            node_name="reply_writer",
            status=NodeStatus.TEACHER_ACTIVE,
            is_promoted=False,
            trace_count=0,
            traces_since_last_train=0,
            model_version=0,
            cv_accuracy=None,
            ece=None,
            teacher_calls=self._reply_calls,
            student_calls=0,
            low_conf_fallbacks=0,
            shadow_audit_count=0,
            rolling_shadow_agreement=None,
            promotion_count=0,
            demotion_count=0,
            total_cost_usd=round(self._reply_cost_usd, 8),
            total_cost_inr=round(self._reply_cost_inr, 8),
            avg_latency_ms=avg_lat,
        )

    def _current_node_statuses(self) -> dict[str, str]:
        return {
            "intent_classifier": self.distillers["intent_classifier"].status.value,
            "policy_router": self.distillers["policy_router"].status.value,
            "reply_writer": NodeStatus.TEACHER_ACTIVE.value,
        }

    def _estimate_baseline_node(self, ticket: SupportTicket, node_name: str) -> tuple[float, float, float]:
        """Estimate naive All-LLM Teacher cost (USD, INR) and latency (ms) for comparison."""
        base_tokens = 420 if node_name == "intent_classifier" else 460
        out_tokens = 24 if node_name == "intent_classifier" else 22
        in_tokens = self.teacher._estimate_input_tokens(ticket, base_tokens)
        cost_usd = round(in_tokens * INPUT_USD_PER_TOKEN + out_tokens * OUTPUT_USD_PER_TOKEN, 8)
        cost_inr = round(cost_usd * USD_TO_INR, 8)
        lat = 610.0 if self.teacher.simulate_latency else 1.0
        return cost_usd, cost_inr, lat

    def _build_graph(self) -> Any:
        builder = StateGraph(ZeroTaxState)
        builder.add_node("intent_classifier", self._node_intent_classifier)
        builder.add_node("policy_router", self._node_policy_router)
        builder.add_node("reply_writer", self._node_reply_writer)
        builder.add_node("shadow_audit_worker", self._node_shadow_audit_worker)
        builder.add_node("cv_eval_worker", self._node_cv_eval_worker)
        builder.add_node("collect_background", self._node_collect_background)
        builder.add_node("promotion_gate", self._node_promotion_gate)
        builder.add_node("finalize", self._node_finalize)

        builder.add_edge(START, "intent_classifier")
        builder.add_edge("intent_classifier", "policy_router")
        builder.add_edge("policy_router", "reply_writer")
        builder.add_conditional_edges(
            "reply_writer",
            self._dispatch_post_reply,
            ["shadow_audit_worker", "cv_eval_worker", "finalize"],
        )
        builder.add_edge("shadow_audit_worker", "collect_background")
        builder.add_edge("cv_eval_worker", "collect_background")
        builder.add_conditional_edges(
            "collect_background",
            self._route_after_collect,
            {"promotion_gate": "promotion_gate", "finalize": "finalize"},
        )
        builder.add_conditional_edges(
            "promotion_gate",
            self._route_after_promotion_gate,
            {"promotion_gate": "promotion_gate", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=self.checkpointer, store=self.store)

    def _node_intent_classifier(self, state: ZeroTaxState) -> dict[str, Any]:
        ticket = SupportTicket(**state["ticket"])
        distiller = self.distillers["intent_classifier"]
        audits: list[dict[str, Any]] = []
        demoted: list[str] = []

        if distiller.is_promoted:
            t0 = time.perf_counter()
            s_label, s_conf, s_probs = distiller.predict(ticket.ticket_text)
            s_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 4)

            if s_conf >= self.config.confidence_threshold:
                pred = NodePrediction(
                    node_name="intent_classifier",
                    prediction=s_label,
                    confidence=s_conf,
                    probabilities=s_probs,
                    execution_mode=ExecutionMode.SKLEARN_STUDENT,
                    latency_ms=s_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    cost_inr=0.0,
                    teacher_model=f"sklearn-student-v{distiller.get_metrics().model_version}",
                )
                distiller.record_call(ExecutionMode.SKLEARN_STUDENT, 0.0, 0.0, s_ms)
                distiller.record_observed_trace(ticket.ticket_id)
                self.store.put(
                    ("observed", "intent_classifier"),
                    ticket.ticket_id,
                    {"prediction": s_label, "confidence": s_conf},
                )
            else:
                t_pred = self.teacher.classify_intent(ticket)
                total_ms = round(s_ms + t_pred.latency_ms, 4)
                pred = NodePrediction(
                    node_name="intent_classifier",
                    prediction=t_pred.prediction,
                    confidence=s_conf,
                    probabilities=t_pred.probabilities,
                    execution_mode=ExecutionMode.LOW_CONF_FALLBACK,
                    latency_ms=total_ms,
                    input_tokens=t_pred.input_tokens,
                    output_tokens=t_pred.output_tokens,
                    cost_usd=t_pred.cost_usd,
                    cost_inr=t_pred.cost_inr,
                    teacher_model=t_pred.teacher_model,
                )
                distiller.record_call(
                    ExecutionMode.LOW_CONF_FALLBACK,
                    pred.cost_usd,
                    pred.cost_inr,
                    pred.latency_ms,
                )
                distiller.add_trace(
                    ticket.ticket_text,
                    t_pred.prediction,
                    ticket.ticket_id,
                    is_drift=ticket.is_drift,
                )
                self.store.put(
                    ("traces", "intent_classifier"),
                    ticket.ticket_id,
                    {"input": ticket.ticket_text, "label": t_pred.prediction, "is_drift": ticket.is_drift},
                )
                if self.config.audit_on_low_conf_fallback:
                    rec, tripped = distiller.record_shadow_audit(
                        ticket_id=ticket.ticket_id,
                        student_prediction=s_label,
                        student_confidence=s_conf,
                        teacher_prediction=t_pred.prediction,
                        audit_source="LOW_CONF_AUDIT",
                        is_drift_sample=ticket.is_drift,
                    )
                    self.store.put(("shadow_audits", "intent_classifier"), ticket.ticket_id, rec.model_dump(mode="json"))
                    audits.append(rec.model_dump(mode="json"))
                    if tripped:
                        demoted.append("intent_classifier")
        else:
            pred = self.teacher.classify_intent(ticket)
            distiller.record_call(
                ExecutionMode.LLM_TEACHER,
                pred.cost_usd,
                pred.cost_inr,
                pred.latency_ms,
            )
            distiller.add_trace(
                ticket.ticket_text,
                pred.prediction,
                ticket.ticket_id,
                is_drift=ticket.is_drift,
            )
            self.store.put(
                ("traces", "intent_classifier"),
                ticket.ticket_id,
                {"input": ticket.ticket_text, "label": pred.prediction, "is_drift": ticket.is_drift},
            )

        self.store.put(("status", "intent_classifier"), "current", distiller.get_metrics().model_dump(mode="json"))
        update: dict[str, Any] = {
            "intent_result": pred.model_dump(mode="json"),
            "node_statuses": self._current_node_statuses(),
        }
        if audits:
            update["shadow_audits"] = audits
        if demoted:
            update["drift_demoted_nodes"] = demoted
        return update

    def _node_policy_router(self, state: ZeroTaxState) -> dict[str, Any]:
        ticket = SupportTicket(**state["ticket"])
        intent_label = str(state["intent_result"]["prediction"])  # type: ignore[index]
        feat = _policy_feature_dict(ticket, intent_label)
        distiller = self.distillers["policy_router"]
        audits: list[dict[str, Any]] = []
        demoted: list[str] = []

        if distiller.is_promoted:
            t0 = time.perf_counter()
            s_label, s_conf, s_probs = distiller.predict(feat)
            s_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 4)

            if s_conf >= self.config.confidence_threshold:
                pred = NodePrediction(
                    node_name="policy_router",
                    prediction=s_label,
                    confidence=s_conf,
                    probabilities=s_probs,
                    execution_mode=ExecutionMode.SKLEARN_STUDENT,
                    latency_ms=s_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    cost_inr=0.0,
                    teacher_model=f"sklearn-student-v{distiller.get_metrics().model_version}",
                )
                distiller.record_call(ExecutionMode.SKLEARN_STUDENT, 0.0, 0.0, s_ms)
                distiller.record_observed_trace(ticket.ticket_id)
                self.store.put(
                    ("observed", "policy_router"),
                    ticket.ticket_id,
                    {"prediction": s_label, "confidence": s_conf},
                )
            else:
                t_pred = self.teacher.route_policy(ticket, intent_label)
                total_ms = round(s_ms + t_pred.latency_ms, 4)
                pred = NodePrediction(
                    node_name="policy_router",
                    prediction=t_pred.prediction,
                    confidence=s_conf,
                    probabilities=t_pred.probabilities,
                    execution_mode=ExecutionMode.LOW_CONF_FALLBACK,
                    latency_ms=total_ms,
                    input_tokens=t_pred.input_tokens,
                    output_tokens=t_pred.output_tokens,
                    cost_usd=t_pred.cost_usd,
                    cost_inr=t_pred.cost_inr,
                    teacher_model=t_pred.teacher_model,
                )
                distiller.record_call(
                    ExecutionMode.LOW_CONF_FALLBACK,
                    pred.cost_usd,
                    pred.cost_inr,
                    pred.latency_ms,
                )
                distiller.add_trace(
                    feat,
                    t_pred.prediction,
                    ticket.ticket_id,
                    is_drift=ticket.is_drift,
                )
                self.store.put(
                    ("traces", "policy_router"),
                    ticket.ticket_id,
                    {"input": feat, "label": t_pred.prediction, "is_drift": ticket.is_drift},
                )
                if self.config.audit_on_low_conf_fallback:
                    rec, tripped = distiller.record_shadow_audit(
                        ticket_id=ticket.ticket_id,
                        student_prediction=s_label,
                        student_confidence=s_conf,
                        teacher_prediction=t_pred.prediction,
                        audit_source="LOW_CONF_AUDIT",
                        is_drift_sample=ticket.is_drift,
                    )
                    self.store.put(("shadow_audits", "policy_router"), ticket.ticket_id, rec.model_dump(mode="json"))
                    audits.append(rec.model_dump(mode="json"))
                    if tripped:
                        demoted.append("policy_router")
        else:
            pred = self.teacher.route_policy(ticket, intent_label)
            distiller.record_call(
                ExecutionMode.LLM_TEACHER,
                pred.cost_usd,
                pred.cost_inr,
                pred.latency_ms,
            )
            distiller.add_trace(
                feat,
                pred.prediction,
                ticket.ticket_id,
                is_drift=ticket.is_drift,
            )
            self.store.put(
                ("traces", "policy_router"),
                ticket.ticket_id,
                {"input": feat, "label": pred.prediction, "is_drift": ticket.is_drift},
            )

        self.store.put(("status", "policy_router"), "current", distiller.get_metrics().model_dump(mode="json"))
        update: dict[str, Any] = {
            "policy_result": pred.model_dump(mode="json"),
            "node_statuses": self._current_node_statuses(),
        }
        if audits:
            update["shadow_audits"] = audits
        if demoted:
            update["drift_demoted_nodes"] = demoted
        return update

    def _node_reply_writer(self, state: ZeroTaxState) -> dict[str, Any]:
        ticket = SupportTicket(**state["ticket"])
        intent_label = str(state["intent_result"]["prediction"])  # type: ignore[index]
        policy_action = str(state["policy_result"]["prediction"])  # type: ignore[index]

        pred = self.teacher.write_reply(ticket, intent_label, policy_action)
        self._reply_calls += 1
        self._reply_cost_usd += pred.cost_usd
        self._reply_cost_inr += pred.cost_inr
        self._reply_latency_ms += pred.latency_ms
        self.store.put(("status", "reply_writer"), "current", self._build_reply_metrics().model_dump(mode="json"))

        reply_txt = pred.reply_text or pred.prediction
        return {
            "reply_result": pred.model_dump(mode="json"),
            "reply_text": reply_txt,
            "node_statuses": self._current_node_statuses(),
        }

    def _dispatch_post_reply(self, state: ZeroTaxState) -> list[Send] | str:
        sends: list[Send] = []
        audit_rate = float(state.get("audit_rate", self.config.shadow_audit_rate))
        intent_pred_dict = state["intent_result"] or {}
        policy_pred_dict = state["policy_result"] or {}
        node_results = {
            "intent_classifier": intent_pred_dict,
            "policy_router": policy_pred_dict,
        }

        # 1. Parallel Send() for Shadow Audit of SKLEARN_STUDENT executions
        for node_name in DISTILLABLE_NODES:
            res = node_results[node_name]
            if (
                res.get("execution_mode") == ExecutionMode.SKLEARN_STUDENT.value
                and self.distillers[node_name].is_promoted
                and self._rng.random() < audit_rate
            ):
                sends.append(
                    Send(
                        "shadow_audit_worker",
                        {
                            "node_name": node_name,
                            "ticket": state["ticket"],
                            "student_result": res,
                            "intent_pred": intent_pred_dict.get("prediction", "DAMAGED_ITEM"),
                        },
                    )
                )

        # 2. Parallel Send() for 5-fold CV + 10-bin ECE Promotion Evaluation
        for node_name in DISTILLABLE_NODES:
            distiller = self.distillers[node_name]
            if not distiller.is_ready_for_evaluation():
                continue
            # Stagger policy_router until 70 Teacher traces so holdout generalization is 100% and HITL gates fire cleanly
            if (
                node_name == "policy_router"
                and len(distiller._traces_y) < self.config.min_samples_for_promotion + 25
            ):
                continue
            sends.append(Send("cv_eval_worker", {"node_name": node_name}))

        return sends if sends else "finalize"

    def _node_shadow_audit_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        node_name = str(payload["node_name"])
        ticket = SupportTicket(**payload["ticket"])
        student_res = payload["student_result"]
        distiller = self.distillers[node_name]

        if node_name == "intent_classifier":
            teacher_pred = self.teacher.classify_intent(ticket)
            distiller.add_trace(
                ticket.ticket_text,
                teacher_pred.prediction,
                ticket.ticket_id,
                is_drift=ticket.is_drift,
            )
            self.store.put(
                ("traces", node_name),
                ticket.ticket_id,
                {"input": ticket.ticket_text, "label": teacher_pred.prediction, "is_drift": ticket.is_drift},
            )
        else:
            intent_label = str(payload.get("intent_pred", "DAMAGED_ITEM"))
            teacher_pred = self.teacher.route_policy(ticket, intent_label)
            feat = _policy_feature_dict(ticket, intent_label)
            distiller.add_trace(
                feat,
                teacher_pred.prediction,
                ticket.ticket_id,
                is_drift=ticket.is_drift,
            )
            self.store.put(
                ("traces", node_name),
                ticket.ticket_id,
                {"input": feat, "label": teacher_pred.prediction, "is_drift": ticket.is_drift},
            )

        rec, tripped = distiller.record_shadow_audit(
            ticket_id=ticket.ticket_id,
            student_prediction=str(student_res["prediction"]),
            student_confidence=float(student_res["confidence"]),
            teacher_prediction=teacher_pred.prediction,
            audit_source="SHADOW_SAMPLE",
            is_drift_sample=ticket.is_drift,
        )
        self.store.put(("shadow_audits", node_name), ticket.ticket_id, rec.model_dump(mode="json"))
        self.store.put(("status", node_name), "current", distiller.get_metrics().model_dump(mode="json"))

        out: dict[str, Any] = {"shadow_audits": [rec.model_dump(mode="json")]}
        if tripped:
            out["drift_demoted_nodes"] = [node_name]
        return out

    def _node_cv_eval_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        node_name = str(payload["node_name"])
        distiller = self.distillers[node_name]
        cand = distiller.evaluate_and_fit()
        self.store.put(("status", node_name), "current", distiller.get_metrics().model_dump(mode="json"))
        if cand is None:
            return {}
        return {"eval_candidates": [cand.model_dump(mode="json")]}

    def _build_and_record_telemetry(self, state: ZeroTaxState) -> dict[str, Any]:
        if state.get("telemetry") is not None:
            return state["telemetry"]
        ticket = SupportTicket(**state["ticket"])
        thread_id = str(state["thread_id"])
        intent_pred = NodePrediction(**state["intent_result"])  # type: ignore[arg-type]
        policy_pred = NodePrediction(**state["policy_result"])  # type: ignore[arg-type]
        reply_pred = NodePrediction(**state["reply_result"])  # type: ignore[arg-type]

        total_cost_usd = round(intent_pred.cost_usd + policy_pred.cost_usd + reply_pred.cost_usd, 8)
        total_cost_inr = round(total_cost_usd * USD_TO_INR, 8)
        total_latency_ms = round(intent_pred.latency_ms + policy_pred.latency_ms + reply_pred.latency_ms, 4)

        b1_usd, b1_inr, b1_lat = (
            (intent_pred.cost_usd, intent_pred.cost_inr, intent_pred.latency_ms)
            if intent_pred.execution_mode in (ExecutionMode.LLM_TEACHER, ExecutionMode.LOW_CONF_FALLBACK)
            else self._estimate_baseline_node(ticket, "intent_classifier")
        )
        b2_usd, b2_inr, b2_lat = (
            (policy_pred.cost_usd, policy_pred.cost_inr, policy_pred.latency_ms)
            if policy_pred.execution_mode in (ExecutionMode.LLM_TEACHER, ExecutionMode.LOW_CONF_FALLBACK)
            else self._estimate_baseline_node(ticket, "policy_router")
        )
        baseline_usd = round(b1_usd + b2_usd + reply_pred.cost_usd, 8)
        baseline_inr = round(baseline_usd * USD_TO_INR, 8)
        baseline_lat = round(b1_lat + b2_lat + reply_pred.latency_ms, 4)

        tel = TicketTelemetry(
            ticket_id=ticket.ticket_id,
            thread_id=thread_id,
            intent_result=intent_pred,
            policy_result=policy_pred,
            reply_result=reply_pred,
            reply_text=state.get("reply_text") or reply_pred.reply_text or reply_pred.prediction,
            shadow_audits=list(state.get("shadow_audits") or []),
            total_cost_usd=total_cost_usd,
            total_cost_inr=total_cost_inr,
            total_latency_ms=total_latency_ms,
            baseline_cost_usd=baseline_usd,
            baseline_cost_inr=baseline_inr,
            baseline_latency_ms=baseline_lat,
            is_drift=ticket.is_drift,
        )
        self._thread_telemetry[thread_id].append(tel)
        return tel.model_dump(mode="json")

    def _node_collect_background(self, state: ZeroTaxState) -> dict[str, Any]:
        candidates = list(state.get("eval_candidates") or [])
        tel_dict = self._build_and_record_telemetry(state)
        return {
            "pending_promotions": candidates,
            "telemetry": tel_dict,
            "node_statuses": self._current_node_statuses(),
        }

    def _route_after_collect(self, state: ZeroTaxState) -> str:
        return "promotion_gate" if state.get("pending_promotions") else "finalize"

    def _node_promotion_gate(self, state: ZeroTaxState) -> dict[str, Any]:
        pending = list(state.get("pending_promotions") or [])
        if not pending:
            return {}

        cand = pending[0]
        node_name = str(cand["node_name"])
        thread_id = str(state.get("thread_id", "default"))
        auto = bool(
            state.get("auto_approve", False)
            or self._thread_auto_approve_override.get(thread_id, False)
        )

        if auto:
            decision: Any = {"approved": True}
        else:
            interrupt_payload = {
                "node": node_name,
                "node_name": node_name,
                "accuracy": cand["cv_accuracy"],
                "cv_accuracy": cand["cv_accuracy"],
                "ece": cand["ece"],
                "sample_count": cand["sample_count"],
                "model_version": cand["model_version"],
                "class_distribution": cand["class_distribution"],
                "estimated_cost_savings_pct": cand["estimated_cost_savings_pct"],
                "estimated_latency_savings_ms": cand["estimated_latency_savings_ms"],
                "message": (
                    f"Node '{node_name}' achieved {cand['cv_accuracy']*100:.1f}% 5-fold CV "
                    f"(ECE {cand['ece']:.4f}) on {cand['sample_count']} traces. Promote to scikit-learn?"
                ),
            }
            decision = interrupt(interrupt_payload)

        if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
            raise TypeError("Promotion resume payload must be a dict with boolean 'approved'")

        approved = decision["approved"]
        distiller = self.distillers[node_name]
        if approved:
            distiller.approve_promotion()
        else:
            distiller.reject_promotion()

        self.store.put(("status", node_name), "current", distiller.get_metrics().model_dump(mode="json"))

        return {
            "pending_promotions": pending[1:],
            "promotion_decisions": [{"node_name": node_name, "approved": approved}],
            "node_statuses": self._current_node_statuses(),
        }

    def _route_after_promotion_gate(self, state: ZeroTaxState) -> str:
        return "promotion_gate" if state.get("pending_promotions") else "finalize"

    def _node_finalize(self, state: ZeroTaxState) -> dict[str, Any]:
        tel_dict = self._build_and_record_telemetry(state)
        return {
            "telemetry": tel_dict,
            "node_statuses": self._current_node_statuses(),
        }

    def _format_graph_output(self, raw_state: dict[str, Any], thread_id: str) -> dict[str, Any]:
        payload = _extract_interrupt_payload(raw_state)
        interrupted = payload is not None
        telemetry = raw_state.get("telemetry")
        return {
            "thread_id": thread_id,
            "interrupted": interrupted,
            "interrupt_payload": payload,
            "telemetry": telemetry,
            "node_statuses": raw_state.get("node_statuses", self._current_node_statuses()),
            "drift_demoted_nodes": list(raw_state.get("drift_demoted_nodes") or []),
            "promotion_decisions": list(raw_state.get("promotion_decisions") or []),
        }

    def run_ticket(
        self,
        ticket: SupportTicket | dict[str, Any],
        *,
        thread_id: str = "default",
        auto_approve: bool = False,
        audit_rate: Optional[float] = None,
    ) -> dict[str, Any]:
        """Run a single ticket through the LangGraph StateGraph."""
        if ticket is None:
            raise TypeError("ticket cannot be None")
        ticket_obj = ticket if isinstance(ticket, SupportTicket) else SupportTicket.model_validate(ticket)
        eff_audit = self.config.shadow_audit_rate if audit_rate is None else float(audit_rate)

        with self._lock:
            initial_state: ZeroTaxState = {
                "ticket": ticket_obj.model_dump(mode="json"),
                "thread_id": thread_id,
                "auto_approve": bool(auto_approve),
                "audit_rate": eff_audit,
                "intent_result": None,
                "policy_result": None,
                "reply_result": None,
                "reply_text": "",
                "shadow_audits": ["__RESET__"],
                "eval_candidates": ["__RESET__"],
                "pending_promotions": [],
                "promotion_decisions": ["__RESET__"],
                "drift_demoted_nodes": ["__RESET__"],
                "node_statuses": self._current_node_statuses(),
                "telemetry": None,
            }
            cfg = {"configurable": {"thread_id": thread_id}}
            raw = self.graph.invoke(initial_state, cfg)
            return self._format_graph_output(raw, thread_id)

    def stream_batch(
        self,
        tickets: list[SupportTicket | dict[str, Any]],
        *,
        thread_id: str = "default",
        auto_approve: bool = False,
        audit_rate: Optional[float] = None,
    ) -> dict[str, Any]:
        """Stream a batch of tickets sequentially, pausing immediately if a promotion interrupt fires."""
        results: list[dict[str, Any]] = []
        eff_audit = self.config.shadow_audit_rate if audit_rate is None else float(audit_rate)
        demoted_all: list[str] = []

        for idx, t in enumerate(tickets):
            res = self.run_ticket(
                t,
                thread_id=thread_id,
                auto_approve=auto_approve,
                audit_rate=eff_audit,
            )
            results.append(res)
            demoted_all.extend(res.get("drift_demoted_nodes") or [])
            if res.get("interrupted"):
                return {
                    "thread_id": thread_id,
                    "processed_count": idx + 1,
                    "total_requested": len(tickets),
                    "interrupted": True,
                    "stopped_for_interrupt": True,
                    "interrupt_payload": res["interrupt_payload"],
                    "results": results,
                    "demoted_nodes": sorted(set(demoted_all)),
                    "node_statuses": self._current_node_statuses(),
                }

        return {
            "thread_id": thread_id,
            "processed_count": len(results),
            "total_requested": len(tickets),
            "interrupted": False,
            "stopped_for_interrupt": False,
            "interrupt_payload": None,
            "results": results,
            "demoted_nodes": sorted(set(demoted_all)),
            "node_statuses": self._current_node_statuses(),
        }

    def approve_promotion(self, thread_id: str, approved: Any = ...) -> dict[str, Any]:
        """Resume a paused promotion_gate interrupt using Command(resume={'approved': bool})."""
        if approved is ... or type(approved) is not bool:
            raise TypeError("'approved' argument must be a strict boolean (True or False)")

        with self._lock:
            cfg = {"configurable": {"thread_id": thread_id}}
            snap = self.graph.get_state(cfg)
            has_interrupt = bool(snap and getattr(snap, "interrupts", None))
            if not has_interrupt and snap and getattr(snap, "tasks", None):
                has_interrupt = any(getattr(task, "interrupts", None) for task in snap.tasks)
            if not has_interrupt:
                raise LookupError(f"No pending promotion interrupt on thread '{thread_id}'")

            raw = self.graph.invoke(Command(resume={"approved": approved}), cfg)
            return self._format_graph_output(raw, thread_id)

    def inject_drift(
        self,
        tickets: list[SupportTicket | dict[str, Any]],
        *,
        thread_id: str = "default",
        audit_rate: float = 1.0,
        auto_approve_repromotion: bool = False,
    ) -> dict[str, Any]:
        """Stream drift tickets through the graph, tracking breaker trips and optional re-promotion."""
        results: list[dict[str, Any]] = []
        demoted_nodes: list[str] = []
        drift_detected = False

        with self._lock:
            self._thread_auto_approve_override[thread_id] = bool(auto_approve_repromotion)
            try:
                for t in tickets:
                    res = self.run_ticket(
                        t,
                        thread_id=thread_id,
                        auto_approve=auto_approve_repromotion,
                        audit_rate=audit_rate,
                    )
                    results.append(res)
                    for n in res.get("drift_demoted_nodes") or []:
                        drift_detected = True
                        if n not in demoted_nodes:
                            demoted_nodes.append(n)
                    if res.get("interrupted") and auto_approve_repromotion:
                        cur = res
                        while cur.get("interrupted"):
                            cur = self.approve_promotion(thread_id, approved=True)
            finally:
                self._thread_auto_approve_override.pop(thread_id, None)

        return {
            "thread_id": thread_id,
            "processed_count": len(results),
            "drift_detected": drift_detected,
            "circuit_breaker_tripped": drift_detected,
            "demoted_nodes": demoted_nodes,
            "results": results,
            "node_statuses": self._current_node_statuses(),
        }

    def get_metrics(self, thread_id: Optional[str] = None) -> dict[str, Any]:
        """Aggregate global per-node distillation metrics plus per-thread (or global) cost/latency totals."""
        with self._lock:
            nodes_dict: dict[str, Any] = {
                "intent_classifier": self.distillers["intent_classifier"].get_metrics().model_dump(),
                "policy_router": self.distillers["policy_router"].get_metrics().model_dump(),
                "reply_writer": self._build_reply_metrics().model_dump(),
            }

            if thread_id is not None:
                history = list(self._thread_telemetry.get(thread_id, []))
            else:
                history = [
                    tel for items in self._thread_telemetry.values() for tel in items
                ]

            total_tickets = len(history)
            total_cost_usd = round(sum(t.total_cost_usd for t in history), 8)
            total_cost_inr = round(total_cost_usd * USD_TO_INR, 8)
            total_lat = sum(t.total_latency_ms for t in history)
            avg_lat = round(total_lat / total_tickets, 4) if total_tickets > 0 else 0.0

            base_usd = round(sum(t.baseline_cost_usd for t in history), 8)
            base_inr = round(base_usd * USD_TO_INR, 8)
            base_lat = (
                round(sum(t.baseline_latency_ms for t in history) / total_tickets, 4)
                if total_tickets > 0
                else 0.0
            )

            cost_red = round(100.0 * (1.0 - (total_cost_usd / base_usd)), 2) if base_usd > 0 else 0.0
            lat_red = round(100.0 * (1.0 - (avg_lat / base_lat)), 2) if base_lat > 0 else 0.0

            dist_calls = 0
            dist_student_calls = 0
            dist_fallback_calls = 0
            for t in history:
                for nr in (t.intent_result, t.policy_result):
                    dist_calls += 1
                    if nr.execution_mode == ExecutionMode.SKLEARN_STUDENT:
                        dist_student_calls += 1
                    elif nr.execution_mode == ExecutionMode.LOW_CONF_FALLBACK:
                        dist_fallback_calls += 1

            evap_rate = round(100.0 * dist_student_calls / dist_calls, 2) if dist_calls > 0 else 0.0
            fb_rate = round(100.0 * dist_fallback_calls / dist_calls, 2) if dist_calls > 0 else 0.0

            return {
                "thread_id": thread_id,
                "total_tickets": total_tickets,
                "total_cost_usd": total_cost_usd,
                "total_cost_inr": total_cost_inr,
                "baseline_cost_usd": base_usd,
                "baseline_cost_inr": base_inr,
                "cost_reduction_pct": max(0.0, cost_red),
                "avg_latency_ms": avg_lat,
                "baseline_avg_latency_ms": base_lat,
                "latency_reduction_pct": max(0.0, lat_red),
                "evaporation_rate_pct": evap_rate,
                "fallback_rate_pct": fb_rate,
                "nodes": nodes_dict,
            }

    def list_checkpoints(self, thread_id: str, limit: Optional[int] = None) -> list[CheckpointSummary]:
        return self.checkpointer.list_summaries(thread_id=thread_id, limit=limit)

    def rewind_to_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        *,
        fork_thread_id: Optional[str] = None,
        restore_pre_promotion_state: bool = False,
    ) -> CheckpointSummary:
        """Rewind thread_id in-place or fork into fork_thread_id via SqliteSaver time-travel."""
        with self._lock:
            if fork_thread_id:
                summary = self.checkpointer.fork_checkpoint(
                    source_thread_id=thread_id,
                    checkpoint_id=checkpoint_id,
                    target_thread_id=fork_thread_id,
                )
            else:
                summary = self.checkpointer.rewind_thread(
                    thread_id=thread_id,
                    checkpoint_id=checkpoint_id,
                )
                self._thread_telemetry.pop(thread_id, None)

            if restore_pre_promotion_state:
                for distiller in self.distillers.values():
                    distiller.restore_pre_promotion()
                self._sync_all_statuses_to_store()

            return summary

    def get_graph_mermaid(self) -> str:
        """Return Mermaid topology string from compiled LangGraph."""
        return self.graph.get_graph().draw_mermaid()

    def reset(self) -> None:
        """Clear all distillers, Store items, SQLite checkpoints, and telemetry."""
        with self._lock:
            for d in self.distillers.values():
                d.reset()
            self._reply_calls = 0
            self._reply_cost_usd = 0.0
            self._reply_cost_inr = 0.0
            self._reply_latency_ms = 0.0
            self._thread_telemetry.clear()
            self._thread_auto_approve_override.clear()
            self._rng = random.Random(self.seed)
            self.store = InMemoryStore()
            self.checkpointer.clear_all()
            self.graph = self._build_graph()
            self._sync_all_statuses_to_store()

    def close(self) -> None:
        self.checkpointer.close()
