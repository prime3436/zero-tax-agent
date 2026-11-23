"""E2E Tests — F11, F12, F13: SqliteSaver Time-Travel, Low-Confidence Fallback & Drift Circuit-Breaker.

Progressive layer: M2 (core.checkpointer, core.graph, core.distiller).
Tests: 15 Tier 1 + 15 Tier 2 = 30 tests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


# ===========================================================================
# F11: SqliteSaver Persistence & Time-Travel (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f11_01_sqlite_checkpoint_file_and_wal_mode(make_engine, dataset_bundle, temp_db_path) -> None:
    """T1-F11-01: SqliteSaver creates SQLite DB on disk in WAL mode with >= 5 checkpoints."""
    eng = make_engine(db_name=temp_db_path.name)
    for t in dataset_bundle.train_stream[:5]:
        eng.run_ticket(t, thread_id="t1-f11-01")

    assert temp_db_path.exists()
    assert temp_db_path.stat().st_size > 0

    conn = sqlite3.connect(str(temp_db_path))
    try:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(mode).lower() == "wal"
    finally:
        conn.close()

    cps = eng.list_checkpoints("t1-f11-01")
    assert len(cps) >= 5


def test_t1_f11_02_persistence_across_fresh_engine_reload(make_engine, dataset_bundle, temp_db_path) -> None:
    """T1-F11-02: Fresh ZeroTaxEngine on same db_path recovers persisted checkpoint history."""
    eng1 = make_engine(db_name=temp_db_path.name)
    for t in dataset_bundle.train_stream[:10]:
        eng1.run_ticket(t, thread_id="t1-f11-02")
    ids_1 = [c.checkpoint_id for c in eng1.list_checkpoints("t1-f11-02")]
    assert len(ids_1) >= 10

    eng2 = make_engine(db_name=temp_db_path.name)
    ids_2 = [c.checkpoint_id for c in eng2.list_checkpoints("t1-f11-02")]
    assert ids_2 == ids_1


def test_t1_f11_03_list_checkpoints_chain_ordering_and_metadata(engine, dataset_bundle) -> None:
    """T1-F11-03: CheckpointSummary entries contain distinct IDs, parent links, steps, timestamps."""
    for t in dataset_bundle.train_stream[:8]:
        engine.run_ticket(t, thread_id="t1-f11-03")

    cps = engine.list_checkpoints("t1-f11-03")
    assert len(cps) >= 8
    cp_ids = [c.checkpoint_id for c in cps]
    assert len(set(cp_ids)) == len(cp_ids)
    assert all(c.created_at is not None for c in cps)
    assert any(c.parent_checkpoint_id is not None for c in cps)


def test_t1_f11_04_time_travel_rewind_restores_pre_promotion_teacher_state(engine, dataset_bundle) -> None:
    """T1-F11-04: Rewinding to pre-promotion checkpoint restores TEACHER_ACTIVE and LLM_TEACHER mode."""
    from core.models import ExecutionMode, NodeStatus
    from e2e_helpers import enum_str, get_node_metrics, get_node_result

    for t in dataset_bundle.train_stream[:20]:
        engine.run_ticket(t, thread_id="t1-f11-04")
    pre_cp = engine.list_checkpoints("t1-f11-04")[0].checkpoint_id

    engine.stream_batch(dataset_bundle.train_stream[20:140], thread_id="t1-f11-04", auto_approve=True)
    assert get_node_metrics(engine.get_metrics("t1-f11-04"), "intent_classifier")["is_promoted"] is True

    engine.rewind_to_checkpoint("t1-f11-04", pre_cp, restore_pre_promotion_state=True)
    m_intent = get_node_metrics(engine.get_metrics("t1-f11-04"), "intent_classifier")
    assert enum_str(m_intent["status"]) == NodeStatus.TEACHER_ACTIVE.value
    assert m_intent["is_promoted"] is False

    res = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t1-f11-04")
    assert enum_str(get_node_result(res, "intent_classifier")["execution_mode"]) == ExecutionMode.LLM_TEACHER.value


def test_t1_f11_05_time_travel_fork_preserves_original_thread(engine, dataset_bundle) -> None:
    """T1-F11-05: Rewinding with fork_thread_id creates independent branch without mutating main."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    for t in dataset_bundle.train_stream[:15]:
        engine.run_ticket(t, thread_id="t1-f11-main")
    pre_cp = engine.list_checkpoints("t1-f11-main")[0].checkpoint_id
    engine.stream_batch(dataset_bundle.train_stream[15:140], thread_id="t1-f11-main", auto_approve=True)
    main_cps_before = [c.checkpoint_id for c in engine.list_checkpoints("t1-f11-main")]

    engine.rewind_to_checkpoint(
        "t1-f11-main",
        pre_cp,
        fork_thread_id="t1-f11-fork",
        restore_pre_promotion_state=True,
    )
    fork_cps = engine.list_checkpoints("t1-f11-fork")
    assert len(fork_cps) >= 1

    res = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t1-f11-fork")
    assert enum_str(get_node_result(res, "intent_classifier")["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
    assert [c.checkpoint_id for c in engine.list_checkpoints("t1-f11-main")] == main_cps_before


def test_t2_f11_01_rewind_nonexistent_checkpoint_raises(engine, dataset_bundle) -> None:
    """T2-F11-01: Rewinding to unknown checkpoint_id raises KeyError/ValueError/LookupError."""
    engine.run_ticket(dataset_bundle.train_stream[0], thread_id="t2-f11-01")
    with pytest.raises((KeyError, ValueError, LookupError)):
        engine.rewind_to_checkpoint("t2-f11-01", "nonexistent-cp-999")


def test_t2_f11_02_rewind_nonexistent_thread_raises(engine) -> None:
    """T2-F11-02: Rewinding on unknown thread_id raises KeyError/ValueError/LookupError."""
    with pytest.raises((KeyError, ValueError, LookupError)):
        engine.rewind_to_checkpoint("no-such-thread", "cp-1")


def test_t2_f11_03_list_checkpoints_unknown_thread_and_limit_one(engine, dataset_bundle) -> None:
    """T2-F11-03: Unknown thread returns []; limit=1 returns at most 1 checkpoint."""
    assert engine.list_checkpoints("unknown-thread-xyz") == []

    for t in dataset_bundle.train_stream[:5]:
        engine.run_ticket(t, thread_id="t2-f11-03")
    limited = engine.list_checkpoints("t2-f11-03", limit=1)
    assert len(limited) == 1


def test_t2_f11_04_rewind_restore_pre_promotion_flag(engine, dataset_bundle) -> None:
    """T2-F11-04: restore_pre_promotion_state=True resets NodeDistiller promotion status cleanly."""
    from core.models import NodeStatus
    from e2e_helpers import enum_str, get_node_metrics

    engine.run_ticket(dataset_bundle.train_stream[0], thread_id="t2-f11-04")
    early_cp = engine.list_checkpoints("t2-f11-04")[0].checkpoint_id
    engine.stream_batch(dataset_bundle.train_stream[1:140], thread_id="t2-f11-04", auto_approve=True)

    engine.rewind_to_checkpoint("t2-f11-04", early_cp, restore_pre_promotion_state=True)
    m_intent = get_node_metrics(engine.get_metrics("t2-f11-04"), "intent_classifier")
    assert enum_str(m_intent["status"]) == NodeStatus.TEACHER_ACTIVE.value
    assert m_intent["is_promoted"] is False


def test_t2_f11_05_invalid_directory_db_path_raises(tmp_path: Path) -> None:
    """T2-F11-05: Passing an existing directory path as db_path raises sqlite3/OSError."""
    from core.graph import ZeroTaxEngine
    from core.models import TeacherMode

    with pytest.raises((sqlite3.OperationalError, OSError, ValueError)):
        ZeroTaxEngine(
            db_path=str(tmp_path),
            teacher_mode=TeacherMode.SIMULATED,
            sleep_on_simulated=False,
        )


# ===========================================================================
# F12: Low-Confidence Fallback (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f12_01_ambiguous_ticket_triggers_low_conf_fallback(make_engine, dataset_bundle) -> None:
    """T1-F12-01: Ambiguous low-confidence input routes to LOW_CONF_FALLBACK with Teacher cost."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, make_ticket

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.98))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f12-01", auto_approve=True)

    ambiguous = make_ticket(
        ticket_id="AMB-1",
        ticket_text="maybe late or wrong size or something unclear not sure",
    )
    res = eng.run_ticket(ambiguous, thread_id="t1-f12-01", audit_rate=0.0)
    ir = get_node_result(res, "intent_classifier")

    assert enum_str(ir["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value
    assert ir["cost_usd"] > 0.0
    assert ir["latency_ms"] > 0.0


def test_t1_f12_02_fallback_uses_teacher_prediction_and_increments_counter(make_engine, dataset_bundle) -> None:
    """T1-F12-02: LOW_CONF_FALLBACK outputs valid Teacher prediction & increments low_conf_fallbacks."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode, IntentLabel, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import enum_str, get_node_metrics, get_node_result, make_ticket

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.98))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f12-02", auto_approve=True)

    ambiguous = make_ticket(ticket_id="AMB-2", ticket_text="maybe delayed or damaged parcel uncertain")
    teacher_pred = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42).classify_intent(ambiguous)

    res = eng.run_ticket(ambiguous, thread_id="t1-f12-02", audit_rate=0.0)
    ir = get_node_result(res, "intent_classifier")

    assert enum_str(ir["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value
    assert enum_str(ir["prediction"]) in {e.value for e in IntentLabel}
    assert enum_str(ir["prediction"]) == teacher_pred.prediction
    assert get_node_metrics(eng.get_metrics("t1-f12-02"), "intent_classifier")["low_conf_fallbacks"] >= 1


def test_t1_f12_03_fallback_records_low_conf_audit_when_enabled(make_engine, dataset_bundle) -> None:
    """T1-F12-03: LOW_CONF_FALLBACK emits ShadowAuditRecord with audit_source='LOW_CONF_AUDIT'."""
    from core.distiller import DistillerConfig
    from e2e_helpers import get_telemetry, make_ticket, to_dict

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.98, audit_on_low_conf_fallback=True))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f12-03", auto_approve=True)

    ambiguous = make_ticket(ticket_id="AMB-3", ticket_text="uncertain issue maybe size maybe late")
    res = eng.run_ticket(ambiguous, thread_id="t1-f12-03", audit_rate=0.0)
    audits = [to_dict(a) for a in get_telemetry(res).get("shadow_audits", [])]
    assert any(a.get("audit_source") == "LOW_CONF_AUDIT" for a in audits)


def test_t1_f12_04_fallback_feeds_teacher_label_back_as_online_trace(make_engine, dataset_bundle) -> None:
    """T1-F12-04: Fallback increments trace_count on the fallback node."""
    from core.distiller import DistillerConfig
    from e2e_helpers import get_node_metrics, make_ticket

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.98))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f12-04", auto_approve=True)
    before = get_node_metrics(eng.get_metrics("t1-f12-04"), "intent_classifier")["trace_count"]

    eng.run_ticket(make_ticket(ticket_id="AMB-4", ticket_text="vague order problem not clear"), thread_id="t1-f12-04", audit_rate=0.0)
    after = get_node_metrics(eng.get_metrics("t1-f12-04"), "intent_classifier")["trace_count"]
    assert after == before + 1


def test_t1_f12_05_mixed_clear_and_ambiguous_tickets_split_student_vs_fallback(make_engine, dataset_bundle) -> None:
    """T1-F12-05: Clear holdout ticket runs SKLEARN_STUDENT ($0) while ambiguous runs LOW_CONF_FALLBACK (>0)."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, make_ticket

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.90))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f12-05", auto_approve=True)

    clear_res = eng.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t1-f12-05", audit_rate=0.0)
    amb_res = eng.run_ticket(
        make_ticket(ticket_id="AMB-5", ticket_text="maybe late or size mismatch unsure"),
        thread_id="t1-f12-05",
        audit_rate=0.0,
    )

    ir_clear = get_node_result(clear_res, "intent_classifier")
    ir_amb = get_node_result(amb_res, "intent_classifier")
    assert enum_str(ir_clear["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
    assert ir_clear["cost_usd"] == 0.0
    assert enum_str(ir_amb["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value
    assert ir_amb["cost_usd"] > 0.0


def test_t2_f12_01_exact_threshold_090_routing_invariant(make_engine, dataset_bundle) -> None:
    """T2-F12-01: At threshold=0.90, SKLEARN_STUDENT implies conf>=0.90 and LOW_CONF_FALLBACK implies conf<0.90."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, make_ticket

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.90))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f12-01", auto_approve=True)

    probes = list(dataset_bundle.eval_holdout[:15]) + [
        make_ticket(ticket_id="EDGE-OOD", ticket_text="xqzv blarg flimflam 99999 zzz"),
    ]
    student_hits = 0
    fallback_hits = 0
    for t in probes:
        res = eng.run_ticket(t, thread_id="t2-f12-01", audit_rate=0.0)
        ir = get_node_result(res, "intent_classifier")
        mode = enum_str(ir["execution_mode"])
        if mode == ExecutionMode.SKLEARN_STUDENT.value:
            assert ir["confidence"] >= 0.90
            student_hits += 1
        elif mode == ExecutionMode.LOW_CONF_FALLBACK.value:
            assert ir["confidence"] < 0.90
            fallback_hits += 1
        else:
            pytest.fail(f"Unexpected execution mode {mode} for promoted intent_classifier")
    assert student_hits >= 1
    assert fallback_hits >= 1


def test_t2_f12_02_fallback_with_audit_on_low_conf_disabled(make_engine, dataset_bundle) -> None:
    """T2-F12-02: audit_on_low_conf_fallback=False still bills Teacher fallback without LOW_CONF_AUDIT."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, get_telemetry, make_ticket, to_dict

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.99, audit_on_low_conf_fallback=False))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f12-02", auto_approve=True)

    res = eng.run_ticket(make_ticket(ticket_id="AMB-NOAUD", ticket_text="ambiguous message"), thread_id="t2-f12-02", audit_rate=0.0)
    ir = get_node_result(res, "intent_classifier")
    assert enum_str(ir["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value
    assert ir["cost_usd"] > 0.0
    audits = [to_dict(a) for a in get_telemetry(res).get("shadow_audits", [])]
    assert not any(a.get("audit_source") == "LOW_CONF_AUDIT" for a in audits)


def test_t2_f12_03_empty_text_on_promoted_student_no_crash(engine, dataset_bundle) -> None:
    """T2-F12-03: Empty string ticket_text on promoted engine routes cleanly without crash."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, make_ticket

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f12-03", auto_approve=True)
    res = engine.run_ticket(make_ticket(ticket_id="EMPTY-PROMO", ticket_text=""), thread_id="t2-f12-03", audit_rate=0.0)
    mode = enum_str(get_node_result(res, "intent_classifier")["execution_mode"])
    assert mode in {ExecutionMode.LOW_CONF_FALLBACK.value, ExecutionMode.SKLEARN_STUDENT.value}


def test_t2_f12_04_ood_gibberish_routes_to_low_conf_fallback(engine, dataset_bundle) -> None:
    """T2-F12-04: Out-of-distribution gibberish routes to LOW_CONF_FALLBACK at threshold 0.90."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, make_ticket

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f12-04", auto_approve=True)
    res = engine.run_ticket(
        make_ticket(ticket_id="OOD-GIB", ticket_text="xqzv blarg flimflam 99999 zzz"),
        thread_id="t2-f12-04",
        audit_rate=0.0,
    )
    ir = get_node_result(res, "intent_classifier")
    assert enum_str(ir["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value
    assert ir["cost_usd"] > 0.0


@pytest.mark.parametrize("bad_thresh", [-0.1, 1.2])
def test_t2_f12_05_invalid_confidence_threshold_rejected(bad_thresh: float) -> None:
    """T2-F12-05: Out-of-range confidence_threshold (<0 or >1) is rejected."""
    from core.distiller import DistillerConfig
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        DistillerConfig(confidence_threshold=bad_thresh)


# ===========================================================================
# F13: Concept Drift Circuit-Breaker & Re-Promotion (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f13_01_drift_wave_drops_agreement_and_trips_breaker(engine, dataset_bundle) -> None:
    """T1-F13-01: Injecting drift_wave drops rolling_shadow_agreement < 0.90 and trips breaker."""
    from e2e_helpers import get_node_metrics

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f13-01", auto_approve=True)
    drift_res = engine.inject_drift(dataset_bundle.drift_wave[:25], thread_id="t1-f13-01", audit_rate=1.0)

    m_intent = get_node_metrics(engine.get_metrics("t1-f13-01"), "intent_classifier")
    assert m_intent["demotion_count"] >= 1
    assert (
        drift_res.get("drift_detected", False)
        or drift_res.get("circuit_breaker_tripped", False)
        or len(drift_res.get("demoted_nodes", [])) >= 1
    )


def test_t1_f13_02_demoted_node_immediately_uses_llm_teacher(engine, dataset_bundle) -> None:
    """T1-F13-02: Immediately after drift breaker trips, demoted node executes as LLM_TEACHER."""
    from core.models import ExecutionMode, IntentLabel
    from e2e_helpers import enum_str, get_node_result

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f13-02", auto_approve=True)
    engine.inject_drift(dataset_bundle.drift_wave[:25], thread_id="t1-f13-02", audit_rate=1.0)

    scam = next(t for t in dataset_bundle.drift_wave[25:] if t.ground_truth_intent == IntentLabel.EMPTY_BOX_SCAM)
    res = engine.run_ticket(scam, thread_id="t1-f13-02")
    ir = get_node_result(res, "intent_classifier")

    assert enum_str(ir["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
    assert enum_str(ir["prediction"]) == IntentLabel.EMPTY_BOX_SCAM.value


def test_t1_f13_03_drift_collection_learns_empty_box_scam_and_retrains(engine, dataset_bundle) -> None:
    """T1-F13-03: Full drift_wave collection retrains intent_classifier with higher model_version."""
    from e2e_helpers import get_node_metrics

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f13-03", auto_approve=True)
    v1 = get_node_metrics(engine.get_metrics("t1-f13-03"), "intent_classifier")["model_version"]

    engine.inject_drift(
        dataset_bundle.drift_wave,
        thread_id="t1-f13-03",
        audit_rate=1.0,
        auto_approve_repromotion=True,
    )
    v2 = get_node_metrics(engine.get_metrics("t1-f13-03"), "intent_classifier")["model_version"]
    assert v2 > v1


def test_t1_f13_04_repromotion_gate_restores_student_active_after_drift(engine, dataset_bundle) -> None:
    """T1-F13-04: After drift recovery, node re-promotes to STUDENT_ACTIVE with promotion_count >= 2."""
    from core.models import NodeStatus
    from e2e_helpers import enum_str, get_node_metrics

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f13-04", auto_approve=True)
    engine.inject_drift(
        dataset_bundle.drift_wave,
        thread_id="t1-f13-04",
        audit_rate=1.0,
        auto_approve_repromotion=True,
    )
    m_intent = get_node_metrics(engine.get_metrics("t1-f13-04"), "intent_classifier")
    assert enum_str(m_intent["status"]) == NodeStatus.STUDENT_ACTIVE.value
    assert m_intent["is_promoted"] is True
    assert m_intent["promotion_count"] >= 2


def test_t1_f13_05_post_recovery_student_handles_normal_and_empty_box_scam(engine, dataset_bundle) -> None:
    """T1-F13-05: Post-recovery student classifies both normal & EMPTY_BOX_SCAM accurately."""
    from core.models import ExecutionMode, IntentLabel
    from e2e_helpers import enum_str, get_node_result, make_ticket

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f13-05", auto_approve=True)
    engine.inject_drift(
        dataset_bundle.drift_wave,
        thread_id="t1-f13-05",
        audit_rate=1.0,
        auto_approve_repromotion=True,
    )

    probes = list(dataset_bundle.eval_holdout[:5]) + [
        make_ticket(
            ticket_id=f"POST-SCAM-{i}",
            category="ELECTRONICS",
            item_price_inr=9500.0,
            ticket_text="Opened the sealed parcel and found an empty box with a brick inside instead of the phone",
            is_drift=True,
            ground_truth_intent=IntentLabel.EMPTY_BOX_SCAM.value,
        )
        for i in range(5)
    ]

    correct = 0
    student_used = 0
    for p in probes:
        res = engine.run_ticket(p, thread_id="t1-f13-05", audit_rate=0.0)
        ir = get_node_result(res, "intent_classifier")
        if enum_str(ir["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value:
            student_used += 1
        if enum_str(ir["prediction"]) == p.ground_truth_intent.value:
            correct += 1

    assert student_used >= 5
    assert (correct / len(probes)) >= 0.80


def test_t2_f13_01_below_min_audits_for_breaker_3_audits_does_not_trip(dataset_bundle) -> None:
    """T2-F13-01: 3 disagreeing audits (< min_audits_for_breaker=4) do NOT trip circuit-breaker."""
    from core.distiller import DistillerConfig, NodeDistiller

    distiller = NodeDistiller(
        "intent_classifier",
        config=DistillerConfig(min_audits_for_breaker=4, drift_agreement_threshold=0.90),
        seed=42,
    )
    for t in dataset_bundle.train_stream[:80]:
        distiller.add_trace(t.ticket_text, t.ground_truth_intent.value, t.ticket_id)
    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    for i in range(3):
        _, tripped = distiller.record_shadow_audit(
            ticket_id=f"DRIFT-3-{i}",
            student_prediction="DAMAGED_ITEM",
            student_confidence=0.94,
            teacher_prediction="EMPTY_BOX_SCAM",
            audit_source="SHADOW_SAMPLE",
            is_drift_sample=True,
        )
        assert tripped is False

    assert distiller.get_metrics().demotion_count == 0
    assert distiller.get_metrics().is_promoted is True


def test_t2_f13_02_exact_4th_disagreeing_audit_trips_breaker(dataset_bundle) -> None:
    """T2-F13-02: Exact 4th disagreeing audit (min_audits_for_breaker=4) trips circuit-breaker."""
    from core.distiller import DistillerConfig, NodeDistiller
    from core.models import NodeStatus

    distiller = NodeDistiller(
        "intent_classifier",
        config=DistillerConfig(min_audits_for_breaker=4, drift_agreement_threshold=0.90),
        seed=42,
    )
    for t in dataset_bundle.train_stream[:80]:
        distiller.add_trace(t.ticket_text, t.ground_truth_intent.value, t.ticket_id)
    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    tripped_last = False
    for i in range(4):
        _, tripped_last = distiller.record_shadow_audit(
            ticket_id=f"DRIFT-4-{i}",
            student_prediction="DAMAGED_ITEM",
            student_confidence=0.94,
            teacher_prediction="EMPTY_BOX_SCAM",
            audit_source="SHADOW_SAMPLE",
            is_drift_sample=True,
        )

    assert tripped_last is True
    assert distiller.get_metrics().demotion_count == 1
    assert distiller.get_metrics().is_promoted is False
    assert distiller.get_metrics().status == NodeStatus.DEMOTED_DRIFT


def test_t2_f13_03_drift_injection_at_zero_audit_rate_never_trips_breaker(make_engine, dataset_bundle) -> None:
    """T2-F13-03: inject_drift with audit_rate=0.0 & high threshold misses shadow breaker (demotion_count==0)."""
    from core.distiller import DistillerConfig
    from e2e_helpers import get_node_metrics

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.0, shadow_audit_rate=0.0))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f13-03", auto_approve=True, audit_rate=0.0)
    eng.inject_drift(dataset_bundle.drift_wave[:20], thread_id="t2-f13-03", audit_rate=0.0)

    assert get_node_metrics(eng.get_metrics("t2-f13-03"), "intent_classifier")["demotion_count"] == 0


def test_t2_f13_04_drift_injection_on_cold_start_no_spurious_demotion(engine, dataset_bundle) -> None:
    """T2-F13-04: inject_drift on cold-start unpromoted engine runs LLM_TEACHER with 0 demotions."""
    from e2e_helpers import get_node_metrics

    engine.inject_drift(dataset_bundle.drift_wave[:20], thread_id="t2-f13-04", audit_rate=1.0)
    m_intent = get_node_metrics(engine.get_metrics("t2-f13-04"), "intent_classifier")
    assert m_intent["demotion_count"] == 0
    assert m_intent["trace_count"] == 20


def test_t2_f13_05_empty_drift_wave_list_graceful(engine) -> None:
    """T2-F13-05: inject_drift([]) returns cleanly without state corruption."""
    res = engine.inject_drift([], thread_id="t2-f13-empty")
    processed = int(res["processed_count"]) if "processed_count" in res else len(res["results"])
    assert processed == 0
