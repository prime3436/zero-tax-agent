"""Tier 3 E2E Tests — Cross-Feature Pairwise Interactions (T3-01 .. T3-18).

Tests: 18 pairwise combinatorial E2E tests covering shared state, data flow, and control flow.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_t3_01_promotion_low_conf_fallback_and_shadow_audit(engine, dataset_bundle) -> None:
    """T3-01: Promoted student takes SHADOW_SAMPLE on clear ticket and LOW_CONF_FALLBACK on OOD ticket."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_metrics, get_node_result, get_telemetry, make_ticket, to_dict

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t3-01", auto_approve=True, audit_rate=0.0)

    clear_res = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t3-01", audit_rate=1.0)
    ir_clear = get_node_result(clear_res, "intent_classifier")
    assert enum_str(ir_clear["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
    clear_audits = [to_dict(a) for a in get_telemetry(clear_res)["shadow_audits"]]
    assert any(a.get("audit_source") == "SHADOW_SAMPLE" for a in clear_audits)

    ambiguous = make_ticket(
        ticket_id="T3-01-OOD",
        ticket_text="xqzv blarg flimflam 99999 zzz",
    )
    amb_res = engine.run_ticket(ambiguous, thread_id="t3-01", audit_rate=1.0)
    ir_amb = get_node_result(amb_res, "intent_classifier")
    assert ir_amb["confidence"] < 0.90
    assert enum_str(ir_amb["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value
    amb_audits = [to_dict(a) for a in get_telemetry(amb_res)["shadow_audits"]]
    assert any(a.get("audit_source") == "LOW_CONF_AUDIT" for a in amb_audits)

    m_intent = get_node_metrics(engine.get_metrics("t3-01"), "intent_classifier")
    assert m_intent["demotion_count"] == 0


def test_t3_02_rejection_cooldown_reevaluation_and_approval(engine, dataset_bundle) -> None:
    """T3-02: Rejected promotion respects cooldown, re-evaluates after >=10 traces, then promotes on approval."""
    from core.models import NodeStatus
    from e2e_helpers import enum_str, get_interrupt_payload, get_node_metrics

    r1 = engine.stream_batch(dataset_bundle.train_stream[:100], thread_id="t3-02", auto_approve=False)
    assert r1.get("interrupted") is True
    p1 = get_interrupt_payload(r1) or {}
    node = p1.get("node") or p1.get("node_name", "intent_classifier")
    proc1 = int(r1.get("processed_count", 0))

    engine.approve_promotion("t3-02", approved=False)
    nm_rej = get_node_metrics(engine.get_metrics("t3-02"), node)
    assert nm_rej["is_promoted"] is False
    assert enum_str(nm_rej["status"]) == NodeStatus.TEACHER_ACTIVE.value

    r_cool = engine.stream_batch([dataset_bundle.train_stream[proc1]], thread_id="t3-02", auto_approve=False)
    assert not r_cool.get("interrupted", False)

    r2 = engine.stream_batch(dataset_bundle.train_stream[proc1 + 1 : proc1 + 35], thread_id="t3-02", auto_approve=False)
    assert r2.get("interrupted") is True
    p2 = get_interrupt_payload(r2) or {}
    assert int(p2.get("sample_count", 0)) > int(p1.get("sample_count", 0))

    engine.approve_promotion("t3-02", approved=True)
    nm_app = get_node_metrics(engine.get_metrics("t3-02"), node)
    assert nm_app["is_promoted"] is True
    assert enum_str(nm_app["status"]) == NodeStatus.STUDENT_ACTIVE.value


def test_t3_03_drift_shadow_audit_demotion_retrain_and_repromotion(engine, dataset_bundle) -> None:
    """T3-03: Full drift -> shadow audit <90% -> demotion -> retrain 5 classes -> re-promotion -> accurate scam prediction."""
    from core.models import ExecutionMode, IntentLabel, NodeStatus
    from e2e_helpers import enum_str, get_node_metrics, get_node_result, make_ticket

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t3-03", auto_approve=True, audit_rate=0.0)
    v1 = get_node_metrics(engine.get_metrics("t3-03"), "intent_classifier")["model_version"]

    drift_res = engine.inject_drift(
        dataset_bundle.drift_wave,
        thread_id="t3-03",
        audit_rate=1.0,
        auto_approve_repromotion=True,
    )
    m_intent = get_node_metrics(engine.get_metrics("t3-03"), "intent_classifier")
    assert m_intent["demotion_count"] >= 1
    assert m_intent["model_version"] > v1
    assert enum_str(m_intent["status"]) == NodeStatus.STUDENT_ACTIVE.value
    assert (
        drift_res.get("drift_detected", False)
        or drift_res.get("circuit_breaker_tripped", False)
        or len(drift_res.get("demoted_nodes", [])) >= 1
    )

    scam_hits = 0
    for i in range(10):
        scam_t = make_ticket(
            ticket_id=f"T3-03-SCAM-{i}",
            category="ELECTRONICS",
            item_price_inr=8500.0,
            ticket_text="Sealed box delivered with empty soap brick inside instead of headphones",
            is_drift=True,
        )
        res = engine.run_ticket(scam_t, thread_id="t3-03", audit_rate=0.0)
        ir = get_node_result(res, "intent_classifier")
        if enum_str(ir["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value and enum_str(ir["prediction"]) == IntentLabel.EMPTY_BOX_SCAM.value:
            scam_hits += 1
    assert scam_hits >= 8


def test_t3_04_promotion_time_travel_rewind_to_pre_promotion_and_rerun(engine, dataset_bundle) -> None:
    """T3-04: Rewinding post-promotion thread to pre-promotion checkpoint restores LLM_TEACHER without duplicate replay."""
    from core.models import ExecutionMode, NodeStatus
    from e2e_helpers import enum_str, get_node_metrics, get_node_result, get_telemetry

    for t in dataset_bundle.train_stream[:20]:
        engine.run_ticket(t, thread_id="t3-04")
    pre_cp = engine.list_checkpoints("t3-04")[0].checkpoint_id

    engine.stream_batch(dataset_bundle.train_stream[20:140], thread_id="t3-04", auto_approve=True)
    assert get_node_metrics(engine.get_metrics("t3-04"), "intent_classifier")["is_promoted"] is True

    engine.rewind_to_checkpoint("t3-04", pre_cp, restore_pre_promotion_state=True)
    assert enum_str(get_node_metrics(engine.get_metrics("t3-04"), "intent_classifier")["status"]) == NodeStatus.TEACHER_ACTIVE.value

    for t in dataset_bundle.eval_holdout[:3]:
        res = engine.run_ticket(t, thread_id="t3-04")
        assert enum_str(get_node_result(res, "intent_classifier")["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
        assert get_telemetry(res)["shadow_audits"] == []


def test_t3_05_fastapi_full_lifecycle_stream_approve_drift_metrics_rewind(api_client, dataset_bundle) -> None:
    """T3-05: FastAPI full lifecycle: stream -> checkpoint -> approve -> drift -> metrics -> rewind -> run."""
    from e2e_helpers import extract_checkpoints, get_node_metrics, ticket_to_api_payload

    assert api_client.post("/reset").status_code == 200
    sb = api_client.post("/stream-batch", json={"count": 120, "thread_id": "t3-05", "auto_approve": False})
    assert sb.status_code == 200
    assert sb.json().get("interrupted") or sb.json().get("stopped_for_interrupt")

    cps_resp = api_client.get("/checkpoints/t3-05")
    assert cps_resp.status_code == 200
    cp_list = extract_checkpoints(cps_resp.json())
    assert len(cp_list) >= 1
    pre_cp = cp_list[-1]["checkpoint_id"]

    ap = api_client.post("/approve-promotion", json={"thread_id": "t3-05", "approved": True})
    assert ap.status_code == 200
    while ap.json().get("interrupted"):
        ap = api_client.post("/approve-promotion", json={"thread_id": "t3-05", "approved": True})
        assert ap.status_code == 200
    assert api_client.post("/stream-batch", json={"count": 60, "thread_id": "t3-05", "auto_approve": True}).status_code == 200
    assert api_client.post("/inject-drift", json={"count": 40, "thread_id": "t3-05", "audit_rate": 1.0}).status_code == 200

    m_resp = api_client.get("/metrics", params={"thread_id": "t3-05"})
    assert m_resp.status_code == 200
    m = m_resp.json()
    assert float(m["cost_reduction_pct"]) > 0.0
    assert float(m["latency_reduction_pct"]) > 0.0

    rw = api_client.post(
        "/rewind",
        json={"thread_id": "t3-05", "checkpoint_id": pre_cp, "restore_pre_promotion_state": True},
    )
    assert rw.status_code == 200

    m_after = api_client.get("/metrics", params={"thread_id": "t3-05"}).json()
    assert get_node_metrics(m_after, "intent_classifier")["is_promoted"] is False

    run_after = api_client.post(
        "/run",
        json={"thread_id": "t3-05", "ticket": ticket_to_api_payload(dataset_bundle.eval_holdout[0])},
    )
    assert run_after.status_code == 200
    tel = run_after.json().get("telemetry", run_after.json())
    assert tel["intent_result"]["execution_mode"] == "LLM_TEACHER"


def test_t3_06_store_and_sqlite_checkpointer_persistence_across_reload(make_engine, dataset_bundle, temp_db_path) -> None:
    """T3-06: SqliteSaver checkpoints survive engine reload while Store tracks active traces/promotion."""
    from e2e_helpers import get_node_metrics

    eng1 = make_engine(db_name=temp_db_path.name)
    eng1.stream_batch(dataset_bundle.train_stream[:100], thread_id="persist-1", auto_approve=True)
    m1 = get_node_metrics(eng1.get_metrics("persist-1"), "intent_classifier")
    assert m1["trace_count"] == 100
    assert m1["is_promoted"] is True
    cps1 = [c.checkpoint_id for c in eng1.list_checkpoints("persist-1")]
    assert len(cps1) >= 100

    eng2 = make_engine(db_name=temp_db_path.name)
    cps2 = [c.checkpoint_id for c in eng2.list_checkpoints("persist-1")]
    assert cps2 == cps1


def test_t3_07_independent_node1_and_node2_promotion_and_demotion(engine, dataset_bundle) -> None:
    """T3-07: Approving Node 1 while rejecting Node 2 gives asymmetric modes; drift demotes affected node."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_interrupt_payload, get_node_metrics, get_node_result, to_dict

    idx = 0
    approved_node = None
    rejected_node = None
    tickets = dataset_bundle.train_stream[:140]
    while idx < len(tickets):
        res = engine.stream_batch(tickets[idx:], thread_id="t3-07", auto_approve=False)
        processed = int(res.get("processed_count", 0))
        if res.get("interrupted"):
            cur = to_dict(res)
            while cur.get("interrupted"):
                p = get_interrupt_payload(cur) or {}
                node = p.get("node") or p.get("node_name", "")
                assert node in {"intent_classifier", "policy_router"}
                if approved_node is None:
                    approved_node = node
                    cur = to_dict(engine.approve_promotion("t3-07", approved=True))
                else:
                    rejected_node = node
                    cur = to_dict(engine.approve_promotion("t3-07", approved=False))
            idx += max(processed, 1)
            if approved_node is not None and rejected_node is not None:
                break
        else:
            if processed <= 0:
                break
            idx += processed

    assert approved_node is not None and rejected_node is not None
    assert approved_node != rejected_node
    follow = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t3-07", audit_rate=0.0)
    assert enum_str(get_node_result(follow, approved_node)["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
    assert enum_str(get_node_result(follow, rejected_node)["execution_mode"]) == ExecutionMode.LLM_TEACHER.value

    engine.inject_drift(dataset_bundle.drift_wave[:25], thread_id="t3-07", audit_rate=1.0)
    metrics_after = engine.get_metrics("t3-07")
    assert get_node_metrics(metrics_after, approved_node)["demotion_count"] >= 1
    assert get_node_metrics(metrics_after, rejected_node)["demotion_count"] == 0


def test_t3_08_node3_reply_writer_always_llm_across_all_routing_states(make_engine, dataset_bundle) -> None:
    """T3-08: reply_writer stays LLM_TEACHER across cold-start, student, fallback, and demoted states."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, get_reply_text, make_ticket

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.90))
    # 1. Cold start
    r_cold = eng.run_ticket(dataset_bundle.train_stream[0], thread_id="t3-08")
    # 2. Promoted student
    eng.stream_batch(dataset_bundle.train_stream[1:140], thread_id="t3-08", auto_approve=True, audit_rate=0.0)
    r_stud = eng.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t3-08", audit_rate=0.0)
    # 3. Low-conf fallback
    r_fb = eng.run_ticket(
        make_ticket(ticket_id="T3-08-FB", ticket_text="xqzv unsure blarg"),
        thread_id="t3-08",
        audit_rate=0.0,
    )
    # 4. Drift demoted
    eng.inject_drift(dataset_bundle.drift_wave[:25], thread_id="t3-08", audit_rate=1.0)
    r_dem = eng.run_ticket(dataset_bundle.drift_wave[26], thread_id="t3-08")

    for r in (r_cold, r_stud, r_fb, r_dem):
        rr = get_node_result(r, "reply_writer")
        assert enum_str(rr["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
        assert rr["cost_inr"] > 0.0
        assert rr["latency_ms"] > 0.0
        assert len(get_reply_text(r).strip()) > 0


def test_t3_09_audit_rate_by_confidence_threshold_interaction(make_engine, dataset_bundle) -> None:
    """T3-09: High-threshold Engine A catches drift via fallbacks; low-threshold Engine B catches via shadow audits."""
    from core.distiller import DistillerConfig
    from e2e_helpers import get_node_metrics

    eng_a = make_engine(db_name="ea.db", config=DistillerConfig(confidence_threshold=0.98, shadow_audit_rate=0.05))
    eng_b = make_engine(db_name="eb.db", config=DistillerConfig(confidence_threshold=0.70, shadow_audit_rate=1.0))

    eng_a.stream_batch(dataset_bundle.train_stream[:120], thread_id="ea", auto_approve=True)
    eng_b.stream_batch(dataset_bundle.train_stream[:120], thread_id="eb", auto_approve=True)

    eng_a.inject_drift(dataset_bundle.drift_wave[:25], thread_id="ea", audit_rate=0.05)
    eng_b.inject_drift(dataset_bundle.drift_wave[:25], thread_id="eb", audit_rate=1.0)

    ma = get_node_metrics(eng_a.get_metrics("ea"), "intent_classifier")
    mb = get_node_metrics(eng_b.get_metrics("eb"), "intent_classifier")
    assert ma["low_conf_fallbacks"] >= mb["low_conf_fallbacks"]
    assert mb["shadow_audit_count"] >= ma["shadow_audit_count"]


def test_t3_10_cli_benchmark_vs_engine_metrics_consistency(tmp_path: Path) -> None:
    """T3-10: CLI benchmark.py produces BenchmarkReport consistent with USD_TO_INR and Markdown output."""
    from core.models import USD_TO_INR
    from e2e_helpers import isolated_subprocess_env

    project_root = Path(__file__).resolve().parents[2]
    out_json = tmp_path / "t3_10.json"
    out_md = tmp_path / "t3_10.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(project_root / "benchmark.py"),
            "--seed", "42",
            "--normal-count", "200",
            "--drift-count", "50",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--teacher-mode", "simulated",
        ],
        cwd=str(project_root),
        env=isolated_subprocess_env(tmp_path / "bench_checkpoints.db"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    md = out_md.read_text(encoding="utf-8")

    assert rep["baseline_total_cost_inr"] == pytest.approx(rep["baseline_total_cost_usd"] * USD_TO_INR, rel=1e-3)
    assert rep["zerotax_total_cost_inr"] < rep["baseline_total_cost_inr"]
    assert rep["zerotax_avg_latency_ms"] < rep["baseline_avg_latency_ms"]
    assert rep["cost_reduction_pct"] > 0.0
    assert rep["latency_reduction_pct"] > 0.0
    assert len(md) > 100


def test_t3_11_teacher_mode_auto_offline_and_fastapi_accounting(api_client, dataset_bundle) -> None:
    """T3-11: TeacherMode.AUTO resolves offline and FastAPI /stream-batch maintains USD_TO_INR accounting."""
    from core.models import USD_TO_INR, IntentLabel, TeacherMode
    from core.teacher import LLMTeacher

    t = LLMTeacher(mode=TeacherMode.AUTO, sleep_on_simulated=False, seed=42)
    pred = t.classify_intent(dataset_bundle.train_stream[0])
    assert pred.prediction in {e.value for e in IntentLabel}
    assert pred.cost_usd > 0.0
    assert pred.cost_inr == pytest.approx(pred.cost_usd * USD_TO_INR, rel=1e-3)

    assert api_client.post("/reset").status_code == 200
    sb = api_client.post("/stream-batch", json={"count": 40, "thread_id": "t3-11", "auto_approve": True})
    assert sb.status_code == 200
    m_resp = api_client.get("/metrics", params={"thread_id": "t3-11"})
    assert m_resp.status_code == 200
    m = m_resp.json()
    assert m["total_cost_usd"] > 0.0
    assert m["total_cost_inr"] == pytest.approx(m["total_cost_usd"] * USD_TO_INR, rel=1e-3)


def test_t3_12_dataset_split_isolation_and_holdout_generalization(engine, dataset_bundle) -> None:
    """T3-12: Online distillation via Teacher on train_stream generalizes >= 0.90 to eval_holdout."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    engine.stream_batch(dataset_bundle.train_stream, thread_id="t3-12", auto_approve=True, audit_rate=0.0)
    intent_ok = 0
    policy_ok = 0
    student_calls = 0

    for t in dataset_bundle.eval_holdout:
        res = engine.run_ticket(t, thread_id="t3-12", audit_rate=0.0)
        ir = get_node_result(res, "intent_classifier")
        pr = get_node_result(res, "policy_router")
        if enum_str(ir["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value:
            student_calls += 1
        if enum_str(ir["prediction"]) == t.ground_truth_intent.value:
            intent_ok += 1
        if enum_str(pr["prediction"]) == t.ground_truth_policy.value:
            policy_ok += 1

    assert student_calls >= 35
    assert (intent_ok / len(dataset_bundle.eval_holdout)) >= 0.90
    assert (policy_ok / len(dataset_bundle.eval_holdout)) >= 0.90


def test_t3_13_time_travel_fork_thread_isolation_via_api(api_client, dataset_bundle) -> None:
    """T3-13: Forking via /rewind creates isolated fork-hub thread leaving main-hub unchanged."""
    from e2e_helpers import ticket_to_api_payload

    assert api_client.post("/reset").status_code == 200
    for t in dataset_bundle.train_stream[:10]:
        assert api_client.post("/run", json={"thread_id": "main-hub", "ticket": ticket_to_api_payload(t)}).status_code == 200

    cps_main_early = api_client.get("/checkpoints/main-hub").json()
    early_list = cps_main_early if isinstance(cps_main_early, list) else cps_main_early.get("checkpoints", [])
    pre_cp = early_list[0]["checkpoint_id"]

    assert api_client.post("/stream-batch", json={"count": 80, "thread_id": "main-hub", "auto_approve": True}).status_code == 200
    cps_main_before = api_client.get("/checkpoints/main-hub").json()

    rw = api_client.post(
        "/rewind",
        json={
            "thread_id": "main-hub",
            "checkpoint_id": pre_cp,
            "fork_thread_id": "fork-hub",
            "restore_pre_promotion_state": True,
        },
    )
    assert rw.status_code == 200
    run_fork = api_client.post(
        "/run",
        json={"thread_id": "fork-hub", "ticket": ticket_to_api_payload(dataset_bundle.eval_holdout[0])},
    )
    assert run_fork.status_code == 200
    assert run_fork.json().get("telemetry", run_fork.json())["intent_result"]["execution_mode"] == "LLM_TEACHER"
    assert api_client.get("/checkpoints/main-hub").json() == cps_main_before


def test_t3_14_send_cv_ece_values_match_interrupt_payload_and_metrics(engine, dataset_bundle) -> None:
    """T3-14: Parallel Send() CV/ECE metrics match interrupt() payload & get_metrics()."""
    from e2e_helpers import get_interrupt_payload, get_node_metrics

    res = engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t3-14", auto_approve=False)
    assert res.get("interrupted") is True
    payload = get_interrupt_payload(res)
    assert payload is not None

    node = payload.get("node") or payload.get("node_name", "intent_classifier")
    nm = get_node_metrics(engine.get_metrics("t3-14"), node)
    acc = payload.get("accuracy") if "accuracy" in payload else payload.get("cv_accuracy")

    assert float(acc) == pytest.approx(float(nm["cv_accuracy"]), abs=1e-4)
    assert float(payload["ece"]) == pytest.approx(float(nm["ece"]), abs=1e-4)
    assert int(payload["sample_count"]) == int(nm["trace_count"])


def test_t3_15_fastapi_single_run_interrupt_and_409_conflict_guard(api_client, dataset_bundle) -> None:
    """T3-15: Single /run hits interrupt; /approve-promotion succeeds once and returns 409 before/after."""
    from e2e_helpers import ticket_to_api_payload

    assert api_client.post("/reset").status_code == 200
    assert api_client.post("/approve-promotion", json={"thread_id": "t-guard", "approved": True}).status_code == 409

    hit_interrupt = False
    for t in dataset_bundle.train_stream[:120]:
        r = api_client.post("/run", json={"thread_id": "t-guard", "ticket": ticket_to_api_payload(t)})
        assert r.status_code == 200
        if r.json().get("interrupted"):
            hit_interrupt = True
            assert r.json().get("interrupt_payload") is not None
            break

    assert hit_interrupt is True
    ap = api_client.post("/approve-promotion", json={"thread_id": "t-guard", "approved": True})
    assert ap.status_code == 200
    while ap.json().get("interrupted"):
        ap = api_client.post("/approve-promotion", json={"thread_id": "t-guard", "approved": True})
        assert ap.status_code == 200
    assert api_client.post("/approve-promotion", json={"thread_id": "t-guard", "approved": True}).status_code == 409


def test_t3_16_graph_mermaid_matches_between_engine_and_api(engine, api_client) -> None:
    """T3-16: Engine.get_graph_mermaid() and GET /graph-mermaid both expose 3-node + gate topology."""
    eng_mermaid = engine.get_graph_mermaid()
    resp = api_client.get("/graph-mermaid")
    assert resp.status_code == 200
    api_body = resp.json() if "application/json" in resp.headers.get("content-type", "") else {"mermaid": resp.text}
    api_mermaid = api_body.get("mermaid", str(api_body))

    for diagram in (eng_mermaid, api_mermaid):
        assert "intent_classifier" in diagram
        assert "policy_router" in diagram
        assert "reply_writer" in diagram
        assert ("-->" in diagram) or ("-.->" in diagram)


def test_t3_17_api_reset_clears_state_for_clean_relearning(api_client) -> None:
    """T3-17: POST /reset wipes promotion/drift history so subsequent batch relearns cleanly."""
    from e2e_helpers import get_node_metrics

    assert api_client.post("/stream-batch", json={"count": 120, "thread_id": "t3-17", "auto_approve": True}).status_code == 200
    assert api_client.post("/inject-drift", json={"count": 30, "thread_id": "t3-17", "audit_rate": 1.0}).status_code == 200

    assert api_client.post("/reset").status_code == 200
    m_resp = api_client.get("/metrics")
    assert m_resp.status_code == 200
    m = m_resp.json()
    m_intent = get_node_metrics(m, "intent_classifier")
    m_policy = get_node_metrics(m, "policy_router")
    assert m_intent["trace_count"] == 0
    assert m_intent["is_promoted"] is False
    assert m_policy["trace_count"] == 0
    assert m_policy["is_promoted"] is False

    sb2 = api_client.post("/stream-batch", json={"count": 80, "thread_id": "t3-17-new", "auto_approve": False})
    assert sb2.status_code == 200


def test_t3_18_api_rewind_404_and_metrics_thread_filtering(api_client, dataset_bundle) -> None:
    """T3-18: Thread-filtered GET /metrics tracks per-thread totals while invalid /rewind returns 404."""
    from e2e_helpers import ticket_to_api_payload

    assert api_client.post("/reset").status_code == 200
    for t in dataset_bundle.train_stream[:20]:
        assert api_client.post("/run", json={"thread_id": "thread-alpha", "ticket": ticket_to_api_payload(t)}).status_code == 200
    for t in dataset_bundle.train_stream[20:50]:
        assert api_client.post("/run", json={"thread_id": "thread-beta", "ticket": ticket_to_api_payload(t)}).status_code == 200

    resp_alpha = api_client.get("/metrics", params={"thread_id": "thread-alpha"})
    resp_beta = api_client.get("/metrics", params={"thread_id": "thread-beta"})
    assert resp_alpha.status_code == 200
    assert resp_beta.status_code == 200
    m_alpha = resp_alpha.json()
    m_beta = resp_beta.json()
    assert m_alpha["total_tickets"] == 20
    assert m_beta["total_tickets"] == 30

    rw = api_client.post("/rewind", json={"thread_id": "thread-alpha", "checkpoint_id": "00000000-dead-beef"})
    assert rw.status_code == 404
