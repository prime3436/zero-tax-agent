"""Tier 4 E2E Tests — Real-World Indian E-Commerce Operational Scenarios (T4-01 .. T4-08).

Tests: 8 realistic end-to-end operational scenarios exercising F01-F16.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_t4_01_scenario_normal_day_200_ticket_evaporation(engine, dataset_bundle) -> None:
    """T4-01: Normal-Day 200-ticket refund desk evaporates Nodes 1 & 2 while keeping Node 3 LLM."""
    from core.models import ExecutionMode, NodeStatus
    from e2e_helpers import enum_str, get_node_metrics, get_node_result, stream_until_complete

    results, interrupts = stream_until_complete(
        engine,
        dataset_bundle.train_stream,
        thread_id="t4-01-desk",
        approve=True,
        audit_rate=0.05,
    )
    assert len(interrupts) >= 2
    for payload in interrupts:
        acc = payload.get("accuracy") if "accuracy" in payload else payload.get("cv_accuracy")
        assert float(acc) >= 0.95
        assert float(payload["ece"]) <= 0.10

    metrics = engine.get_metrics("t4-01-desk")
    m_intent = get_node_metrics(metrics, "intent_classifier")
    m_policy = get_node_metrics(metrics, "policy_router")
    assert enum_str(m_intent["status"]) == NodeStatus.STUDENT_ACTIVE.value
    assert enum_str(m_policy["status"]) == NodeStatus.STUDENT_ACTIVE.value

    evap_rate = float(metrics.get("evaporation_rate_pct", 0.0))
    if evap_rate == 0.0:
        total = max(m_intent["student_calls"] + m_intent["teacher_calls"], 1)
        evap_rate = 100.0 * m_intent["student_calls"] / total
    assert evap_rate >= 50.0

    follow = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t4-01-desk", audit_rate=0.0)
    assert enum_str(get_node_result(follow, "reply_writer")["execution_mode"]) == ExecutionMode.LLM_TEACHER.value


def test_t4_02_scenario_festive_sale_empty_box_scam_wave_recovery(engine, dataset_bundle) -> None:
    """T4-02: Festive sale surge hit by EMPTY_BOX_SCAM drift wave recovers via circuit-breaker & re-promotion."""
    from core.models import ExecutionMode, IntentLabel, NodeStatus, PolicyAction
    from e2e_helpers import enum_str, get_node_metrics, get_node_result, make_ticket

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t4-02-festive", auto_approve=True, audit_rate=0.0)
    v1 = get_node_metrics(engine.get_metrics("t4-02-festive"), "intent_classifier")["model_version"]

    drift_res = engine.inject_drift(
        dataset_bundle.drift_wave,
        thread_id="t4-02-festive",
        audit_rate=1.0,
        auto_approve_repromotion=True,
    )
    assert (
        drift_res.get("drift_detected", False)
        or drift_res.get("circuit_breaker_tripped", False)
        or len(drift_res.get("demoted_nodes", [])) >= 1
    )

    m_intent = get_node_metrics(engine.get_metrics("t4-02-festive"), "intent_classifier")
    assert m_intent["demotion_count"] >= 1
    assert m_intent["model_version"] > v1
    assert enum_str(m_intent["status"]) == NodeStatus.STUDENT_ACTIVE.value

    test_wave = list(dataset_bundle.eval_holdout[:5]) + [
        make_ticket(
            ticket_id=f"T4-02-NEWSCAM-{i}",
            category="ELECTRONICS",
            item_price_inr=12500.0,
            customer_return_rate=0.45,
            days_since_delivery=4,
            ticket_text="Courier handed sealed outer carton containing only an empty box and brick",
            is_drift=True,
            ground_truth_intent=IntentLabel.EMPTY_BOX_SCAM.value,
            ground_truth_policy=PolicyAction.MANUAL_INSPECTION.value,
        )
        for i in range(10)
    ]

    correct = 0
    for probe in test_wave:
        res = engine.run_ticket(probe, thread_id="t4-02-festive", audit_rate=0.0)
        ir = get_node_result(res, "intent_classifier")
        if enum_str(ir["prediction"]) == probe.ground_truth_intent.value:
            correct += 1
        if probe.ground_truth_intent == IntentLabel.EMPTY_BOX_SCAM:
            assert enum_str(get_node_result(res, "policy_router")["prediction"]) == PolicyAction.MANUAL_INSPECTION.value
            assert enum_str(ir["execution_mode"]) in {ExecutionMode.SKLEARN_STUDENT.value, ExecutionMode.LOW_CONF_FALLBACK.value}

    assert (correct / len(test_wave)) >= 0.85


def test_t4_03_scenario_high_value_electronics_fraud_spike_manual_gating(engine, dataset_bundle) -> None:
    """T4-03: High-value Electronics fraud spike triggers LOW_CONF_FALLBACK on OOD and routes to MANUAL_INSPECTION."""
    from core.models import ExecutionMode, PolicyAction
    from e2e_helpers import enum_str, get_node_result, get_telemetry, make_ticket, to_dict

    engine.stream_batch(dataset_bundle.train_stream[:150], thread_id="t4-03-fraud", auto_approve=True, audit_rate=0.0)

    high_risk_batch = [
        make_ticket(
            ticket_id=f"T4-03-HR-{i}",
            category="ELECTRONICS",
            item_price_inr=8500.0 + i * 400.0,
            customer_return_rate=0.50 + (i % 4) * 0.10,
            days_since_delivery=16 + i,
            ticket_text="Serial mismatch high value electronics claim duplicate refund request"
            if i % 2 == 0
            else "unclear ambiguous courier box complaint blarg electronics",
            ground_truth_policy=PolicyAction.MANUAL_INSPECTION.value,
        )
        for i in range(15)
    ]

    manual_ok = 0
    low_conf_audits = 0
    for t in high_risk_batch:
        res = engine.run_ticket(t, thread_id="t4-03-fraud", audit_rate=0.0)
        pr = get_node_result(res, "policy_router")
        if enum_str(pr["prediction"]) == PolicyAction.MANUAL_INSPECTION.value:
            manual_ok += 1
        for a in get_telemetry(res).get("shadow_audits", []):
            if to_dict(a).get("audit_source") == "LOW_CONF_AUDIT":
                low_conf_audits += 1
        ir = get_node_result(res, "intent_classifier")
        if enum_str(ir["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value:
            low_conf_audits += 1

    assert (manual_ok / len(high_risk_batch)) >= 0.90
    assert low_conf_audits >= 1


def test_t4_04_scenario_bad_promotion_rollback_and_forked_recovery(api_client, dataset_bundle) -> None:
    """T4-04: Operator approves promotion, rewinds via FastAPI time-travel to pre-promotion fork, and runs safely."""
    from e2e_helpers import extract_checkpoints, ticket_to_api_payload

    assert api_client.post("/reset").status_code == 200
    sb = api_client.post("/stream-batch", json={"count": 120, "thread_id": "shift-a", "auto_approve": False})
    assert sb.status_code == 200
    assert sb.json().get("interrupted") or sb.json().get("stopped_for_interrupt")

    cps_resp = api_client.get("/checkpoints/shift-a")
    assert cps_resp.status_code == 200
    cp_list = extract_checkpoints(cps_resp.json())
    assert len(cp_list) >= 1
    pre_cp = cp_list[-1]["checkpoint_id"]

    ap = api_client.post("/approve-promotion", json={"thread_id": "shift-a", "approved": True})
    assert ap.status_code == 200
    while ap.json().get("interrupted"):
        ap = api_client.post("/approve-promotion", json={"thread_id": "shift-a", "approved": True})
        assert ap.status_code == 200
    for t in dataset_bundle.eval_holdout[:5]:
        r = api_client.post("/run", json={"thread_id": "shift-a", "ticket": ticket_to_api_payload(t)})
        assert r.status_code == 200

    rw = api_client.post(
        "/rewind",
        json={
            "thread_id": "shift-a",
            "checkpoint_id": pre_cp,
            "fork_thread_id": "shift-a-clean",
            "restore_pre_promotion_state": True,
        },
    )
    assert rw.status_code == 200

    for t in dataset_bundle.eval_holdout[5:10]:
        r_clean = api_client.post("/run", json={"thread_id": "shift-a-clean", "ticket": ticket_to_api_payload(t)})
        assert r_clean.status_code == 200
        tel = r_clean.json().get("telemetry", r_clean.json())
        assert tel["intent_result"]["execution_mode"] == "LLM_TEACHER"

    cp_after_resp = api_client.get("/checkpoints/shift-a")
    assert cp_after_resp.status_code == 200
    assert len(extract_checkpoints(cp_after_resp.json())) >= 1


def test_t4_05_scenario_governance_rejection_then_staggered_approval(engine, dataset_bundle) -> None:
    """T4-05: Governance rejects first promotion, collects extra traces, then approves both nodes staggered."""
    from core.models import NodeStatus
    from e2e_helpers import enum_str, get_interrupt_payload, get_node_metrics, to_dict

    r1 = engine.stream_batch(dataset_bundle.train_stream[:100], thread_id="t4-05-gov", auto_approve=False)
    assert r1.get("interrupted") is True
    p1 = get_interrupt_payload(r1) or {}
    first_samples = int(p1.get("sample_count", 0))
    first_node = p1.get("node") or p1.get("node_name", "intent_classifier")
    pos = int(r1.get("processed_count", 0))

    ap_rej = to_dict(engine.approve_promotion("t4-05-gov", approved=False))
    while ap_rej.get("interrupted"):
        ap_rej = to_dict(engine.approve_promotion("t4-05-gov", approved=False))
    assert get_node_metrics(engine.get_metrics("t4-05-gov"), first_node)["student_calls"] == 0

    reapproved_first = False
    while pos < len(dataset_bundle.train_stream):
        batch = engine.stream_batch(dataset_bundle.train_stream[pos:], thread_id="t4-05-gov", auto_approve=False)
        proc = int(batch.get("processed_count", 0))
        if batch.get("interrupted"):
            cur = to_dict(batch)
            while cur.get("interrupted"):
                p = get_interrupt_payload(cur) or {}
                node = p.get("node") or p.get("node_name", "")
                if node == first_node and not reapproved_first:
                    assert int(p.get("sample_count", 0)) >= first_samples + 10
                    reapproved_first = True
                cur = to_dict(engine.approve_promotion("t4-05-gov", approved=True))
            pos += max(proc, 1)
        else:
            if proc <= 0:
                break
            pos += proc

    assert reapproved_first is True
    metrics = engine.get_metrics("t4-05-gov")
    assert enum_str(get_node_metrics(metrics, "intent_classifier")["status"]) == NodeStatus.STUDENT_ACTIVE.value
    assert enum_str(get_node_metrics(metrics, "policy_router")["status"]) == NodeStatus.STUDENT_ACTIVE.value


def test_t4_06_scenario_offline_ci_reproducible_benchmark_sla_audit(tmp_path: Path) -> None:
    """T4-06: Offline CI benchmark passes SLA thresholds deterministically across 2 isolated runs."""
    from e2e_helpers import isolated_subprocess_env

    project_root = Path(__file__).resolve().parents[2]
    reports = []
    for run_idx in (1, 2):
        run_dir = tmp_path / f"run{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_json = run_dir / "benchmark_report.json"
        out_md = run_dir / "benchmark_report.md"
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
            env=isolated_subprocess_env(run_dir / "bench_checkpoints.db"),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        reports.append(json.loads(out_json.read_text(encoding="utf-8")))

    r1, r2 = reports
    for k in (
        "baseline_total_cost_inr",
        "zerotax_total_cost_inr",
        "cost_reduction_pct",
        "latency_reduction_pct",
        "intent_accuracy",
        "policy_accuracy",
        "drift_detection_ticket_idx",
        "drift_recovery_ticket_idx",
    ):
        assert r1[k] == r2[k]

    assert r1["cost_reduction_pct"] >= 25.0
    assert r1["latency_reduction_pct"] >= 25.0
    assert r1["intent_accuracy"] >= 0.90
    assert r1["policy_accuracy"] >= 0.90
    assert r1["drift_detection_ticket_idx"] is not None
    assert r1["post_recovery_accuracy"] >= 0.85


def test_t4_07_scenario_multi_hub_concurrent_fastapi_sqlite_isolation(api_client, dataset_bundle) -> None:
    """T4-07: Concurrent BLR/BOM/DEL regional hubs execute safely on FastAPI + SqliteSaver WAL."""
    from e2e_helpers import ticket_to_api_payload

    assert api_client.post("/reset").status_code == 200
    hubs = ["hub-blr", "hub-bom", "hub-del"]

    def _send_hub_tickets(hub_idx: int) -> list[int]:
        hub_id = hubs[hub_idx]
        codes = []
        for j in range(10):
            t = dataset_bundle.train_stream[hub_idx * 10 + j]
            resp = api_client.post("/run", json={"thread_id": hub_id, "ticket": ticket_to_api_payload(t)})
            codes.append(resp.status_code)
        return codes

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(_send_hub_tickets, range(3)))

    assert all(code == 200 for hub_codes in results for code in hub_codes)

    blr_cps_before = api_client.get("/checkpoints/hub-blr").json()
    bom_cps_before = api_client.get("/checkpoints/hub-bom").json()
    del_cps = api_client.get("/checkpoints/hub-del").json()
    del_list = del_cps if isinstance(del_cps, list) else del_cps.get("checkpoints", [])
    assert len(del_list) >= 10

    rw = api_client.post(
        "/rewind",
        json={"thread_id": "hub-del", "checkpoint_id": del_list[-1]["checkpoint_id"]},
    )
    assert rw.status_code == 200
    assert api_client.get("/checkpoints/hub-blr").json() == blr_cps_before
    assert api_client.get("/checkpoints/hub-bom").json() == bom_cps_before


def test_t4_08_scenario_holdout_generalization_and_checkpoint_restart_durability(
    make_engine,
    dataset_bundle,
    temp_db_path,
) -> None:
    """T4-08: Engine generalizes >=0.90 on eval_holdout and survives restart via SqliteSaver."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    eng1 = make_engine(db_name=temp_db_path.name)
    eng1.stream_batch(dataset_bundle.train_stream, thread_id="t4-08-dur", auto_approve=True, audit_rate=0.0)

    intent_ok = 0
    policy_ok = 0
    student_count = 0
    for t in dataset_bundle.eval_holdout:
        res = eng1.run_ticket(t, thread_id="t4-08-dur", audit_rate=0.0)
        ir = get_node_result(res, "intent_classifier")
        pr = get_node_result(res, "policy_router")
        if enum_str(ir["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value:
            student_count += 1
        if enum_str(ir["prediction"]) == t.ground_truth_intent.value:
            intent_ok += 1
        if enum_str(pr["prediction"]) == t.ground_truth_policy.value:
            policy_ok += 1

    assert student_count >= 35
    assert (intent_ok / len(dataset_bundle.eval_holdout)) >= 0.90
    assert (policy_ok / len(dataset_bundle.eval_holdout)) >= 0.90

    cps_before = [c.checkpoint_id for c in eng1.list_checkpoints("t4-08-dur")]
    assert len(cps_before) >= 200

    eng2 = make_engine(db_name=temp_db_path.name)
    cps_after = [c.checkpoint_id for c in eng2.list_checkpoints("t4-08-dur")]
    assert cps_after == cps_before

    follow = eng2.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t4-08-dur")
    assert len(eng2.list_checkpoints("t4-08-dur")) >= len(cps_after)
    assert get_node_result(follow, "reply_writer")["cost_usd"] > 0.0
