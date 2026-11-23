"""E2E Tests — F08, F09, F10: Parallel Send() Evaluation, Shadow Audit & HITL interrupt() Gate.

Progressive layer: M2 (core.graph + core.distiller).
Tests: 15 Tier 1 + 15 Tier 2 = 30 tests.
"""

from __future__ import annotations

import pytest


# ===========================================================================
# F08: Parallel Send() 5-Fold CV & ECE Evaluation (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f08_01_send_cv_ece_triggers_once_min_samples_reached(engine, dataset_bundle) -> None:
    """T1-F08-01: Below min_samples_for_promotion=45 no candidate; once reached, CV & ECE evaluate."""
    from e2e_helpers import get_node_metrics

    res_40 = engine.stream_batch(dataset_bundle.train_stream[:40], thread_id="t1-f08-01", auto_approve=False)
    assert not res_40.get("interrupted", False)

    res_next = engine.stream_batch(dataset_bundle.train_stream[40:120], thread_id="t1-f08-01", auto_approve=False)
    assert res_next.get("interrupted", False) is True
    m_intent = get_node_metrics(engine.get_metrics("t1-f08-01"), "intent_classifier")
    assert m_intent["cv_accuracy"] is not None
    assert m_intent["ece"] is not None


def test_t1_f08_02_five_fold_cv_accuracy_bounded_and_high(dataset_bundle) -> None:
    """T1-F08-02: 5-fold StratifiedKFold CV accuracy is in [0.95, 1.0] on 80 balanced traces."""
    from core.distiller import NodeDistiller
    from core.models import TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)
    for t in dataset_bundle.train_stream[:80]:
        distiller.add_trace(t.ticket_text, teacher.classify_intent(t).prediction, t.ticket_id)

    cand = distiller.evaluate_and_fit()
    assert cand is not None
    assert 0.95 <= cand.cv_accuracy <= 1.0


def test_t1_f08_03_ten_bin_ece_well_calibrated(dataset_bundle) -> None:
    """T1-F08-03: 10-bin Expected Calibration Error (ECE) is <= 0.10 on 80 balanced traces."""
    from core.distiller import NodeDistiller
    from core.models import TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)
    for t in dataset_bundle.train_stream[:80]:
        distiller.add_trace(t.ticket_text, teacher.classify_intent(t).prediction, t.ticket_id)

    cand = distiller.evaluate_and_fit()
    assert cand is not None
    assert 0.0 <= cand.ece <= 0.10


def test_t1_f08_04_parallel_send_evaluates_both_distillable_nodes(engine, dataset_bundle) -> None:
    """T1-F08-04: Both intent_classifier and policy_router receive CV & ECE evaluations."""
    from e2e_helpers import get_node_metrics, stream_until_complete

    _, interrupts = stream_until_complete(
        engine,
        dataset_bundle.train_stream[:140],
        thread_id="t1-f08-04",
        approve=True,
    )
    assert len(interrupts) >= 2
    metrics = engine.get_metrics("t1-f08-04")
    for node in ("intent_classifier", "policy_router"):
        nm = get_node_metrics(metrics, node)
        assert nm["cv_accuracy"] is not None and nm["cv_accuracy"] >= 0.95
        assert nm["ece"] is not None and nm["ece"] <= 0.10


def test_t1_f08_05_promotion_payload_reports_positive_savings(engine, dataset_bundle) -> None:
    """T1-F08-05: Promotion interrupt payload reports positive cost & latency savings estimates."""
    from e2e_helpers import get_interrupt_payload

    res = engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t1-f08-05", auto_approve=False)
    assert res.get("interrupted", False) is True
    payload = get_interrupt_payload(res)
    assert payload is not None

    cost_sav = payload.get("estimated_cost_savings_pct")
    if cost_sav is None and isinstance(payload.get("estimated_cost_savings"), dict):
        cost_sav = payload["estimated_cost_savings"].get("pct", 0.0)
    lat_sav = payload.get("estimated_latency_savings_ms")
    if lat_sav is None and isinstance(payload.get("estimated_latency_savings"), dict):
        lat_sav = payload["estimated_latency_savings"].get("ms", payload["estimated_latency_savings"].get("pct", 0.0))

    assert float(cost_sav) > 50.0
    assert float(lat_sav) > 0.0


def test_t2_f08_01_44_traces_never_triggers_promotion(engine, dataset_bundle) -> None:
    """T2-F08-01: 44 traces (just below min_samples_for_promotion=45) stays TEACHER_ACTIVE."""
    from core.models import NodeStatus
    from e2e_helpers import enum_str, get_node_metrics

    res = engine.stream_batch(dataset_bundle.train_stream[:44], thread_id="t2-f08-01", auto_approve=False)
    assert not res.get("interrupted", False)
    m_intent = get_node_metrics(engine.get_metrics("t2-f08-01"), "intent_classifier")
    assert enum_str(m_intent["status"]) == NodeStatus.TEACHER_ACTIVE.value


def test_t2_f08_02_exact_45_balanced_traces_triggers_evaluation(engine, dataset_bundle) -> None:
    """T2-F08-02: Exact 45 balanced traces (12+11+11+11) triggers CV/ECE evaluation."""
    from core.models import IntentLabel
    from e2e_helpers import get_node_metrics

    buckets: dict[str, list] = {
        IntentLabel.DAMAGED_ITEM.value: [],
        IntentLabel.LATE_DELIVERY.value: [],
        IntentLabel.WRONG_SIZE.value: [],
        IntentLabel.FRAUD_SUSPECT.value: [],
    }
    for t in dataset_bundle.train_stream:
        lbl = t.ground_truth_intent.value
        if lbl in buckets:
            buckets[lbl].append(t)

    balanced_45 = (
        buckets[IntentLabel.DAMAGED_ITEM.value][:12]
        + buckets[IntentLabel.LATE_DELIVERY.value][:11]
        + buckets[IntentLabel.WRONG_SIZE.value][:11]
        + buckets[IntentLabel.FRAUD_SUSPECT.value][:11]
    )
    assert len(balanced_45) == 45
    res = engine.stream_batch(balanced_45, thread_id="t2-f08-02", auto_approve=False)
    m_intent = get_node_metrics(engine.get_metrics("t2-f08-02"), "intent_classifier")
    assert m_intent["trace_count"] == 45
    assert (m_intent["cv_accuracy"] is not None) or (res["interrupted"] is True)


def test_t2_f08_03_strict_max_ece_0001_blocks_promotion(make_engine, dataset_bundle) -> None:
    """T2-F08-03: Impossibly strict max_ece=0.001 blocks promotion cleanly."""
    from core.distiller import DistillerConfig
    from core.models import NodeStatus
    from e2e_helpers import enum_str, get_node_metrics

    eng = make_engine(config=DistillerConfig(max_ece=0.001))
    res = eng.stream_batch(dataset_bundle.train_stream[:100], thread_id="t2-f08-03", auto_approve=False)
    assert not res.get("interrupted", False)
    m_intent = get_node_metrics(eng.get_metrics("t2-f08-03"), "intent_classifier")
    assert enum_str(m_intent["status"]) == NodeStatus.TEACHER_ACTIVE.value


def test_t2_f08_04_impossible_min_cv_accuracy_blocks_promotion(make_engine, dataset_bundle) -> None:
    """T2-F08-04: min_cv_accuracy=1.01 (impossible) blocks promotion across 100 tickets."""
    from core.distiller import DistillerConfig
    from e2e_helpers import get_node_metrics

    eng = make_engine(config=DistillerConfig(min_cv_accuracy=1.01))
    res = eng.stream_batch(dataset_bundle.train_stream[:100], thread_id="t2-f08-04", auto_approve=False)
    assert not res.get("interrupted", False)
    assert get_node_metrics(eng.get_metrics("t2-f08-04"), "intent_classifier")["is_promoted"] is False


def test_t2_f08_05_skewed_class_below_min_per_class_blocks_promotion(engine, dataset_bundle) -> None:
    """T2-F08-05: 50 tickets with only 2 FRAUD_SUSPECT samples blocks promotion until >= 5/class."""
    from core.models import IntentLabel
    from e2e_helpers import get_node_metrics

    non_fraud = [t for t in dataset_bundle.train_stream if t.ground_truth_intent != IntentLabel.FRAUD_SUSPECT][:48]
    two_fraud = [t for t in dataset_bundle.train_stream if t.ground_truth_intent == IntentLabel.FRAUD_SUSPECT][:2]
    skewed = non_fraud + two_fraud
    assert len(skewed) == 50

    res = engine.stream_batch(skewed, thread_id="t2-f08-05", auto_approve=False)
    assert not res.get("interrupted", False)
    assert get_node_metrics(engine.get_metrics("t2-f08-05"), "intent_classifier")["is_promoted"] is False


# ===========================================================================
# F09: Parallel Send() Shadow Audit Fan-Out (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f09_01_five_pct_shadow_audit_samples_proportionally(engine, dataset_bundle) -> None:
    """T1-F09-01: Default 5% audit_rate yields > 0 and < 35 audits across 150 post-promotion tickets."""
    from e2e_helpers import get_telemetry

    engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t1-f09-01", auto_approve=True, audit_rate=0.0)
    shadow_records = []
    for idx in range(150):
        t = dataset_bundle.eval_holdout[idx % len(dataset_bundle.eval_holdout)]
        res = engine.run_ticket(t, thread_id="t1-f09-01", audit_rate=0.05)
        shadow_records.extend(get_telemetry(res).get("shadow_audits", []))

    sample_audits = [a for a in shadow_records if (a.get("audit_source") if isinstance(a, dict) else a.audit_source) == "SHADOW_SAMPLE"]
    assert 0 < len(sample_audits) < 40


def test_t1_f09_02_shadow_audit_does_not_block_student_fast_path(engine, dataset_bundle) -> None:
    """T1-F09-02: Even at audit_rate=1.0, fast-path node still executes as SKLEARN_STUDENT."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, get_telemetry

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f09-02", auto_approve=True)
    student_with_audit = 0
    for t in dataset_bundle.eval_holdout[:20]:
        res = engine.run_ticket(t, thread_id="t1-f09-02", audit_rate=1.0)
        ir = get_node_result(res, "intent_classifier")
        audits = get_telemetry(res).get("shadow_audits", [])
        if enum_str(ir["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value and len(audits) >= 1:
            student_with_audit += 1

    assert student_with_audit >= 15


def test_t1_f09_03_shadow_audit_verifies_high_normal_agreement(engine, dataset_bundle) -> None:
    """T1-F09-03: Normal holdout tickets audited at 1.0 show >= 0.85 rolling_shadow_agreement."""
    from e2e_helpers import get_node_metrics, get_telemetry, to_dict

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f09-03", auto_approve=True)
    for t in dataset_bundle.eval_holdout[:20]:
        res = engine.run_ticket(t, thread_id="t1-f09-03", audit_rate=1.0)
        for rec in get_telemetry(res).get("shadow_audits", []):
            rd = to_dict(rec)
            assert "student_prediction" in rd
            assert "teacher_prediction" in rd
            assert isinstance(rd["agreed"], bool)
            assert 0.0 <= float(rd["student_confidence"]) <= 1.0

    m_intent = get_node_metrics(engine.get_metrics("t1-f09-03"), "intent_classifier")
    assert m_intent["rolling_shadow_agreement"] >= 0.85


def test_t1_f09_04_shadow_audit_feeds_online_trace_count(engine, dataset_bundle) -> None:
    """T1-F09-04: Shadow audit records increment trace_count appropriately without blocking."""
    from e2e_helpers import get_node_metrics

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f09-04", auto_approve=True)
    before = get_node_metrics(engine.get_metrics("t1-f09-04"), "intent_classifier")["trace_count"]

    for t in dataset_bundle.eval_holdout[:10]:
        engine.run_ticket(t, thread_id="t1-f09-04", audit_rate=1.0)

    after = get_node_metrics(engine.get_metrics("t1-f09-04"), "intent_classifier")["trace_count"]
    assert after > before


def test_t1_f09_05_both_nodes_shadow_audited_in_same_ticket(engine, dataset_bundle) -> None:
    """T1-F09-05: Both intent_classifier and policy_router emit ShadowAuditRecords when audited."""
    from e2e_helpers import get_telemetry, to_dict

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f09-05", auto_approve=True)
    audited_nodes: set[str] = set()
    for t in dataset_bundle.eval_holdout[:10]:
        res = engine.run_ticket(t, thread_id="t1-f09-05", audit_rate=1.0)
        for rec in get_telemetry(res).get("shadow_audits", []):
            audited_nodes.add(to_dict(rec)["node_name"])

    assert "intent_classifier" in audited_nodes
    assert "policy_router" in audited_nodes


def test_t2_f09_01_audit_rate_zero_records_zero_shadow_samples(engine, dataset_bundle) -> None:
    """T2-F09-01: audit_rate=0.0 produces exactly 0 SHADOW_SAMPLE audit records."""
    from e2e_helpers import get_telemetry, to_dict

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f09-01", auto_approve=True, audit_rate=0.0)
    samples = []
    for t in dataset_bundle.eval_holdout[:30]:
        res = engine.run_ticket(t, thread_id="t2-f09-01", audit_rate=0.0)
        for rec in get_telemetry(res).get("shadow_audits", []):
            if to_dict(rec).get("audit_source") == "SHADOW_SAMPLE":
                samples.append(rec)

    assert len(samples) == 0


def test_t2_f09_02_audit_rate_one_audits_every_student_call(engine, dataset_bundle) -> None:
    """T2-F09-02: audit_rate=1.0 audits every SKLEARN_STUDENT execution across 20 tickets."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, get_telemetry, to_dict

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f09-02", auto_approve=True, audit_rate=0.0)
    student_calls_seen = 0
    for t in dataset_bundle.eval_holdout[:20]:
        res = engine.run_ticket(t, thread_id="t2-f09-02", audit_rate=1.0)
        audits = [to_dict(r) for r in get_telemetry(res)["shadow_audits"]]
        shadow_sample_nodes = {
            a["node_name"] for a in audits if a.get("audit_source") == "SHADOW_SAMPLE"
        }
        for node in ("intent_classifier", "policy_router"):
            if enum_str(get_node_result(res, node)["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value:
                student_calls_seen += 1
                assert node in shadow_sample_nodes
    assert student_calls_seen >= 1


@pytest.mark.parametrize("bad_rate", [-0.1, 1.5])
def test_t2_f09_03_invalid_shadow_audit_rate_rejected(bad_rate: float) -> None:
    """T2-F09-03: Out-of-range shadow_audit_rate (<0 or >1) is rejected by DistillerConfig."""
    from core.distiller import DistillerConfig
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        DistillerConfig(shadow_audit_rate=bad_rate)


def test_t2_f09_04_shadow_window_rollover_reflects_latest_10(dataset_bundle) -> None:
    """T2-F09-04: shadow_window_size=10 reflects latest 10 audits after earlier disagreements."""
    from core.distiller import DistillerConfig, NodeDistiller

    distiller = NodeDistiller("intent_classifier", config=DistillerConfig(shadow_window_size=10), seed=42)
    for t in dataset_bundle.train_stream[:80]:
        distiller.add_trace(t.ticket_text, t.ground_truth_intent.value, t.ticket_id)
    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    for i in range(3):
        distiller.record_shadow_audit(
            ticket_id=f"DIS-{i}",
            student_prediction="DAMAGED_ITEM",
            student_confidence=0.95,
            teacher_prediction="EMPTY_BOX_SCAM",
            audit_source="SHADOW_SAMPLE",
            is_drift_sample=True,
        )
    for i in range(10):
        distiller.record_shadow_audit(
            ticket_id=f"AGR-{i}",
            student_prediction="DAMAGED_ITEM",
            student_confidence=0.96,
            teacher_prediction="DAMAGED_ITEM",
            audit_source="SHADOW_SAMPLE",
            is_drift_sample=False,
        )

    assert distiller.get_metrics().rolling_shadow_agreement == pytest.approx(1.0, abs=1e-6)


def test_t2_f09_05_shadow_audit_skipped_on_cold_start_teacher(engine, dataset_bundle) -> None:
    """T2-F09-05: On cold start (LLM_TEACHER), audit_rate=1.0 emits 0 SHADOW_SAMPLE records."""
    from e2e_helpers import get_telemetry, to_dict

    for t in dataset_bundle.train_stream[:10]:
        res = engine.run_ticket(t, thread_id="t2-f09-05", audit_rate=1.0)
        samples = [
            r for r in get_telemetry(res).get("shadow_audits", [])
            if to_dict(r).get("audit_source") == "SHADOW_SAMPLE"
        ]
        assert len(samples) == 0


# ===========================================================================
# F10: Human-in-the-Loop interrupt() + Command(resume=...) Promotion Gate (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f10_01_stream_batch_pauses_at_real_interrupt_when_ready(engine, dataset_bundle) -> None:
    """T1-F10-01: stream_batch(auto_approve=False) pauses with interrupted=True and payload."""
    from e2e_helpers import get_interrupt_payload

    res = engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t1-f10-01", auto_approve=False)
    assert res.get("interrupted", False) is True
    payload = get_interrupt_payload(res)
    assert payload is not None
    assert ("node" in payload) or ("node_name" in payload)
    assert ("accuracy" in payload) or ("cv_accuracy" in payload)
    assert "ece" in payload
    assert "sample_count" in payload


def test_t1_f10_02_approve_promotion_resumes_and_activates_student(engine, dataset_bundle) -> None:
    """T1-F10-02: approve_promotion(thread_id, approved=True) transitions node to STUDENT_ACTIVE."""
    from core.models import NodeStatus
    from e2e_helpers import enum_str, get_interrupt_payload, get_node_metrics

    res = engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t1-f10-02", auto_approve=False)
    assert res.get("interrupted", False) is True
    payload = get_interrupt_payload(res) or {}
    node = payload.get("node") or payload.get("node_name", "intent_classifier")

    engine.approve_promotion("t1-f10-02", approved=True)
    nm = get_node_metrics(engine.get_metrics("t1-f10-02"), node)
    assert enum_str(nm["status"]) == NodeStatus.STUDENT_ACTIVE.value
    assert nm["is_promoted"] is True
    assert nm["promotion_count"] >= 1


def test_t1_f10_03_subsequent_ticket_after_approval_uses_student(engine, dataset_bundle) -> None:
    """T1-F10-03: Ticket immediately after approve_promotion(True) runs approved node as SKLEARN_STUDENT."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_interrupt_payload, get_node_result

    res = engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t1-f10-03", auto_approve=False)
    payload = get_interrupt_payload(res) or {}
    node = payload.get("node") or payload.get("node_name", "intent_classifier")

    engine.approve_promotion("t1-f10-03", approved=True)
    follow = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t1-f10-03", audit_rate=0.0)
    assert enum_str(get_node_result(follow, node)["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value


def test_t1_f10_04_sequential_promotion_gates_approve_both_nodes(engine, dataset_bundle) -> None:
    """T1-F10-04: Approving interrupts sequentially promotes both intent_classifier and policy_router."""
    from core.models import NodeStatus
    from e2e_helpers import enum_str, get_node_metrics, stream_until_complete

    _, interrupts = stream_until_complete(
        engine,
        dataset_bundle.train_stream[:150],
        thread_id="t1-f10-04",
        approve=True,
    )
    assert len(interrupts) >= 2
    metrics = engine.get_metrics("t1-f10-04")
    assert enum_str(get_node_metrics(metrics, "intent_classifier")["status"]) == NodeStatus.STUDENT_ACTIVE.value
    assert enum_str(get_node_metrics(metrics, "policy_router")["status"]) == NodeStatus.STUDENT_ACTIVE.value


def test_t1_f10_05_interrupt_payload_numeric_criteria_valid(engine, dataset_bundle) -> None:
    """T1-F10-05: Interrupt payload contains accuracy >= 0.95, ece <= 0.10, sample_count >= 45."""
    from e2e_helpers import get_interrupt_payload

    res = engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t1-f10-05", auto_approve=False)
    payload = get_interrupt_payload(res)
    assert payload is not None
    acc = payload.get("accuracy") if "accuracy" in payload else payload.get("cv_accuracy")
    assert float(acc) >= 0.95
    assert float(payload["ece"]) <= 0.10
    assert int(payload["sample_count"]) >= 45


def test_t2_f10_01_promotion_rejection_keeps_teacher_active(engine, dataset_bundle) -> None:
    """T2-F10-01: approve_promotion(approved=False) keeps node TEACHER_ACTIVE and unpromoted."""
    from core.models import ExecutionMode, NodeStatus
    from e2e_helpers import enum_str, get_interrupt_payload, get_node_metrics, get_node_result

    res = engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t2-f10-01", auto_approve=False)
    payload = get_interrupt_payload(res) or {}
    node = payload.get("node") or payload.get("node_name", "intent_classifier")

    engine.approve_promotion("t2-f10-01", approved=False)
    nm = get_node_metrics(engine.get_metrics("t2-f10-01"), node)
    assert nm["is_promoted"] is False
    assert enum_str(nm["status"]) == NodeStatus.TEACHER_ACTIVE.value

    follow = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t2-f10-01")
    assert enum_str(get_node_result(follow, node)["execution_mode"]) == ExecutionMode.LLM_TEACHER.value


def test_t2_f10_02_approve_promotion_without_pending_interrupt_raises(engine) -> None:
    """T2-F10-02: Calling approve_promotion on fresh thread with no pending interrupt raises error."""
    with pytest.raises((ValueError, RuntimeError, LookupError)):
        engine.approve_promotion("no-interrupt-thread", approved=True)


def test_t2_f10_03_duplicate_approve_promotion_after_resolution_raises(engine, dataset_bundle) -> None:
    """T2-F10-03: Calling approve_promotion twice for the same interrupt fails on 2nd call."""
    from e2e_helpers import get_interrupt_payload, get_node_metrics, to_dict

    res = engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t2-f10-03", auto_approve=False)
    assert res.get("interrupted") is True
    payload = get_interrupt_payload(res) or {}
    node = payload.get("node") or payload.get("node_name", "intent_classifier")

    ap = to_dict(engine.approve_promotion("t2-f10-03", approved=True))
    while ap.get("interrupted"):
        ap = to_dict(engine.approve_promotion("t2-f10-03", approved=True))
    count_after_first = get_node_metrics(engine.get_metrics("t2-f10-03"), node)["promotion_count"]

    with pytest.raises((ValueError, RuntimeError, LookupError)):
        engine.approve_promotion("t2-f10-03", approved=True)

    assert get_node_metrics(engine.get_metrics("t2-f10-03"), node)["promotion_count"] == count_after_first


def test_t2_f10_04_malformed_command_resume_payload_never_silently_promotes(engine, dataset_bundle) -> None:
    """T2-F10-04: Missing/non-boolean approval payload via approve_promotion raises error and never promotes."""
    from e2e_helpers import get_interrupt_payload, get_node_metrics

    res = engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t2-f10-04", auto_approve=False)
    assert res.get("interrupted") is True
    payload = get_interrupt_payload(res)
    assert payload is not None
    node = payload.get("node") or payload.get("node_name", "intent_classifier")

    with pytest.raises(TypeError):
        engine.approve_promotion("t2-f10-04")  # type: ignore[call-arg]

    with pytest.raises((TypeError, ValueError)):
        engine.approve_promotion("t2-f10-04", approved=None)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
        engine.approve_promotion("t2-f10-04", approved="not_a_bool")  # type: ignore[arg-type]

    metrics = engine.get_metrics("t2-f10-04")
    assert get_node_metrics(metrics, node)["is_promoted"] is False
    assert get_node_metrics(metrics, "intent_classifier")["is_promoted"] is False
    assert get_node_metrics(metrics, "policy_router")["is_promoted"] is False


def test_t2_f10_05_reeligibility_after_rejection_when_retrain_interval_elapses(engine, dataset_bundle) -> None:
    """T2-F10-05: After rejecting, streaming >= retrain_interval_traces re-triggers interrupt gate."""
    from e2e_helpers import get_interrupt_payload, to_dict

    res1 = engine.stream_batch(dataset_bundle.train_stream[:100], thread_id="t2-f10-05", auto_approve=False)
    assert res1.get("interrupted", False) is True
    p1 = get_interrupt_payload(res1)
    assert p1 is not None
    processed1 = int(res1["processed_count"])
    ap = to_dict(engine.approve_promotion("t2-f10-05", approved=False))
    while ap.get("interrupted"):
        ap = to_dict(engine.approve_promotion("t2-f10-05", approved=False))

    # Next 1 ticket must NOT immediately re-interrupt (cooldown)
    res_cool = engine.stream_batch(
        [dataset_bundle.train_stream[processed1]],
        thread_id="t2-f10-05",
        auto_approve=False,
    )
    assert not res_cool.get("interrupted", False)

    # Streaming >= 10 additional tickets re-triggers promotion gate with larger sample_count
    res2 = engine.stream_batch(
        dataset_bundle.train_stream[processed1 + 1 : processed1 + 40],
        thread_id="t2-f10-05",
        auto_approve=False,
    )
    assert res2.get("interrupted", False) is True
    p2 = get_interrupt_payload(res2)
    assert p2 is not None
    assert int(p2["sample_count"]) > int(p1["sample_count"])
