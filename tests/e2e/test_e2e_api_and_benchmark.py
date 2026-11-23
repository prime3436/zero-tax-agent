"""E2E Tests — F14, F15: FastAPI Service Endpoints & CLI Benchmark.

Progressive layer: M3 (api.py, benchmark.py).
Tests: 10 Tier 1 + 12 Tier 2 = 22 tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ===========================================================================
# F14: FastAPI Service Endpoints (5 T1 + 7 T2)
# ===========================================================================


def test_t1_f14_01_health_and_graph_mermaid_endpoints(api_client) -> None:
    """T1-F14-01: GET /health returns 200 ok and GET /graph-mermaid returns valid Mermaid."""
    h = api_client.get("/health")
    assert h.status_code == 200
    assert h.json().get("status") in {"ok", "healthy"}

    m = api_client.get("/graph-mermaid")
    assert m.status_code == 200
    body = m.json() if "application/json" in m.headers.get("content-type", "") else {"mermaid": m.text}
    mermaid_txt = body.get("mermaid", str(body))
    assert "intent_classifier" in mermaid_txt
    assert "policy_router" in mermaid_txt
    assert "reply_writer" in mermaid_txt


def test_t1_f14_02_run_single_ticket_and_metrics_update(api_client, dataset_bundle) -> None:
    """T1-F14-02: POST /reset -> POST /run -> GET /metrics processes 1 ticket accurately."""
    from e2e_helpers import ticket_to_api_payload

    assert api_client.post("/reset").status_code == 200
    payload = ticket_to_api_payload(dataset_bundle.train_stream[0])
    resp = api_client.post("/run", json={"ticket": payload, "thread_id": "t1-f14-02"})
    assert resp.status_code == 200
    data = resp.json()
    tel = data.get("telemetry", data)
    assert "intent_result" in tel
    assert "policy_result" in tel
    assert len(tel.get("reply_text", "")) > 0
    assert float(tel.get("total_cost_inr", 0.0)) > 0.0

    metrics = api_client.get("/metrics", params={"thread_id": "t1-f14-02"})
    assert metrics.status_code == 200


def test_t1_f14_03_stream_batch_interrupt_and_approve_promotion_cycle(api_client) -> None:
    """T1-F14-03: POST /stream-batch pauses at interrupt; POST /approve-promotion promotes node."""
    from e2e_helpers import get_node_metrics

    assert api_client.post("/reset").status_code == 200
    sb = api_client.post(
        "/stream-batch",
        json={"count": 120, "thread_id": "t1-f14-03", "auto_approve": False},
    )
    assert sb.status_code == 200
    sb_data = sb.json()
    assert sb_data.get("interrupted") or sb_data.get("stopped_for_interrupt")

    ap = api_client.post("/approve-promotion", json={"thread_id": "t1-f14-03", "approved": True})
    assert ap.status_code == 200

    m_resp = api_client.get("/metrics", params={"thread_id": "t1-f14-03"})
    assert m_resp.status_code == 200
    metrics = m_resp.json()
    m_intent = get_node_metrics(metrics, "intent_classifier")
    m_policy = get_node_metrics(metrics, "policy_router")
    assert m_intent["is_promoted"] or m_policy["is_promoted"]


def test_t1_f14_04_inject_drift_endpoint_trips_breaker(api_client) -> None:
    """T1-F14-04: POST /inject-drift triggers drift detection after promotion."""
    assert api_client.post("/reset").status_code == 200
    sb = api_client.post(
        "/stream-batch",
        json={"count": 140, "thread_id": "t1-f14-04", "auto_approve": True},
    )
    assert sb.status_code == 200

    dr = api_client.post(
        "/inject-drift",
        json={"count": 30, "thread_id": "t1-f14-04", "audit_rate": 1.0},
    )
    assert dr.status_code == 200
    dr_data = dr.json()
    assert (
        dr_data.get("drift_detected", False)
        or dr_data.get("circuit_breaker_tripped", False)
        or len(dr_data.get("demoted_nodes", [])) >= 1
    )


def test_t1_f14_05_checkpoints_and_rewind_endpoints(api_client, dataset_bundle) -> None:
    """T1-F14-05: GET /checkpoints/{thread_id} lists history and POST /rewind rewinds cleanly."""
    from e2e_helpers import extract_checkpoints, ticket_to_api_payload

    assert api_client.post("/reset").status_code == 200
    for t in dataset_bundle.train_stream[:10]:
        r = api_client.post("/run", json={"ticket": ticket_to_api_payload(t), "thread_id": "api-tt"})
        assert r.status_code == 200

    cps_resp = api_client.get("/checkpoints/api-tt")
    assert cps_resp.status_code == 200
    cps = extract_checkpoints(cps_resp.json())
    assert len(cps) >= 10
    target_cp = cps[-1]["checkpoint_id"]

    rw = api_client.post(
        "/rewind",
        json={"thread_id": "api-tt", "checkpoint_id": target_cp, "fork_thread_id": "api-tt-fork"},
    )
    assert rw.status_code == 200


def test_t2_f14_01_approve_promotion_without_interrupt_or_duplicate_returns_409(api_client) -> None:
    """T2-F14-01: Calling POST /approve-promotion with no pending interrupt or twice returns HTTP 409."""
    assert api_client.post("/reset").status_code == 200
    r_fresh = api_client.post("/approve-promotion", json={"thread_id": "fresh-none", "approved": True})
    assert r_fresh.status_code == 409

    sb = api_client.post("/stream-batch", json={"count": 120, "thread_id": "dup-409", "auto_approve": False})
    assert sb.status_code == 200
    ap = api_client.post("/approve-promotion", json={"thread_id": "dup-409", "approved": True})
    assert ap.status_code == 200
    while ap.json().get("interrupted"):
        ap = api_client.post("/approve-promotion", json={"thread_id": "dup-409", "approved": True})
        assert ap.status_code == 200
    r_dup = api_client.post("/approve-promotion", json={"thread_id": "dup-409", "approved": True})
    assert r_dup.status_code == 409


def test_t2_f14_02_approve_promotion_rejected_false_keeps_unpromoted(api_client) -> None:
    """T2-F14-02: POST /approve-promotion with approved=false returns 200 and keeps nodes unpromoted."""
    from e2e_helpers import get_node_metrics

    assert api_client.post("/reset").status_code == 200
    sb = api_client.post("/stream-batch", json={"count": 120, "thread_id": "rej-api", "auto_approve": False})
    assert sb.status_code == 200
    assert sb.json().get("interrupted") or sb.json().get("stopped_for_interrupt")

    r_missing = api_client.post("/approve-promotion", json={"thread_id": "rej-api"})
    assert r_missing.status_code == 422
    r_bad_type = api_client.post("/approve-promotion", json={"thread_id": "rej-api", "approved": "not_a_bool"})
    assert r_bad_type.status_code == 422

    ap = api_client.post("/approve-promotion", json={"thread_id": "rej-api", "approved": False})
    assert ap.status_code == 200
    while ap.json().get("interrupted"):
        ap = api_client.post("/approve-promotion", json={"thread_id": "rej-api", "approved": False})
        assert ap.status_code == 200

    m_resp = api_client.get("/metrics", params={"thread_id": "rej-api"})
    assert m_resp.status_code == 200
    metrics = m_resp.json()
    m_intent = get_node_metrics(metrics, "intent_classifier")
    m_policy = get_node_metrics(metrics, "policy_router")
    assert not m_intent["is_promoted"] and not m_policy["is_promoted"]


def test_t2_f14_03_rewind_unknown_checkpoint_returns_404(api_client) -> None:
    """T2-F14-03: POST /rewind with unknown checkpoint_id returns HTTP 404."""
    rw = api_client.post("/rewind", json={"thread_id": "api-1", "checkpoint_id": "does-not-exist-999"})
    assert rw.status_code == 404


@pytest.mark.parametrize(
    "bad_body",
    [
        {},
        {"thread_id": "bad", "ticket": {"category": "CRYPTO", "item_price_inr": 100.0, "customer_return_rate": 0.1, "days_since_delivery": 1, "ticket_text": "hi"}},
        {"thread_id": "bad", "ticket": {"category": "FOOTWEAR", "item_price_inr": -50.0, "customer_return_rate": 0.1, "days_since_delivery": 1, "ticket_text": "hi"}},
        {"thread_id": "bad", "ticket": {"category": "FOOTWEAR", "item_price_inr": 100.0, "customer_return_rate": 1.5, "days_since_delivery": 1, "ticket_text": "hi"}},
        {"thread_id": "bad", "ticket": {"category": "FOOTWEAR", "item_price_inr": 100.0, "customer_return_rate": 0.1, "days_since_delivery": -3, "ticket_text": "hi"}},
    ],
)
def test_t2_f14_04_run_malformed_or_out_of_bounds_returns_422(api_client, bad_body: dict) -> None:
    """T2-F14-04: Malformed JSON / invalid enum / negative fields on POST /run return HTTP 422."""
    resp = api_client.post("/run", json=bad_body)
    assert resp.status_code == 422


def test_t2_f14_05_run_valid_zero_boundary_ticket_returns_200(api_client) -> None:
    """T2-F14-05: POST /run accepts valid boundary ticket (price=0, rate=0, days=0, text='')."""
    resp = api_client.post(
        "/run",
        json={
            "thread_id": "boundary-200",
            "ticket": {
                "ticket_id": "BND-0",
                "customer_name": "Neha Verma",
                "city": "Pune",
                "order_id": "ORD-BND-0",
                "category": "ACCESSORIES",
                "item_price_inr": 0.0,
                "customer_return_rate": 0.0,
                "days_since_delivery": 0,
                "ticket_text": "",
            },
        },
    )
    assert resp.status_code == 200


def test_t2_f14_06_checkpoints_unknown_thread_no_500(api_client) -> None:
    """T2-F14-06: GET /checkpoints/unknown-thread returns 200 [] (or 404), never 500."""
    resp = api_client.get("/checkpoints/nonexistent-thread-xyz")
    assert resp.status_code in {200, 404}
    if resp.status_code == 200:
        body = resp.json()
        cps = body if isinstance(body, list) else body.get("checkpoints", [])
        assert cps == []


def test_t2_f14_07_stream_batch_invalid_count_or_audit_rate_returns_422(api_client) -> None:
    """T2-F14-07: Invalid count (-5) or audit_rate (2.0) on POST /stream-batch returns 422/400."""
    r1 = api_client.post("/stream-batch", json={"count": -5, "thread_id": "bad-sb"})
    r2 = api_client.post("/stream-batch", json={"count": 10, "audit_rate": 2.0, "thread_id": "bad-sb"})
    assert r1.status_code in {400, 422}
    assert r2.status_code in {400, 422}


# ===========================================================================
# F15: CLI Benchmark (benchmark.py) (5 T1 + 5 T2)
# ===========================================================================


def _run_benchmark_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    from e2e_helpers import isolated_subprocess_env

    project_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, str(project_root / "benchmark.py"), *args]
    return subprocess.run(
        cmd,
        cwd=str(project_root),
        env=isolated_subprocess_env(tmp_path / "bench_checkpoints.db"),
        capture_output=True,
        text=True,
        check=False,
    )


def test_t1_f15_01_cli_default_scale_creates_json_and_md_in_tmp(tmp_path: Path) -> None:
    """T1-F15-01: CLI benchmark.py runs 200 normal + 50 drift and outputs JSON & Markdown."""
    out_json = tmp_path / "rep.json"
    out_md = tmp_path / "rep.md"
    proc = _run_benchmark_cli(
        [
            "--seed", "42",
            "--normal-count", "200",
            "--drift-count", "50",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--teacher-mode", "simulated",
        ],
        tmp_path,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert out_json.exists() and out_json.stat().st_size > 0
    assert out_md.exists() and out_md.stat().st_size > 0


def test_t1_f15_02_benchmark_json_schema_and_savings_thresholds(tmp_path: Path) -> None:
    """T1-F15-02: BenchmarkReport JSON verifies cost/latency savings and high accuracy."""
    out_json = tmp_path / "rep2.json"
    out_md = tmp_path / "rep2.md"
    proc = _run_benchmark_cli(
        [
            "--seed", "42",
            "--normal-count", "200",
            "--drift-count", "50",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--teacher-mode", "simulated",
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(out_json.read_text(encoding="utf-8"))

    assert rep["zerotax_total_cost_inr"] < rep["baseline_total_cost_inr"]
    assert rep["cost_reduction_pct"] > 25.0
    assert rep["distillable_cost_reduction_pct"] > 50.0
    assert rep["latency_reduction_pct"] > 30.0
    assert rep["intent_accuracy"] >= 0.85
    assert rep["policy_accuracy"] >= 0.85


def test_t1_f15_03_benchmark_reports_drift_detection_and_recovery(tmp_path: Path) -> None:
    """T1-F15-03: BenchmarkReport captures drift detection index, recovery index & post-recovery accuracy."""
    out_json = tmp_path / "rep3.json"
    out_md = tmp_path / "rep3.md"
    proc = _run_benchmark_cli(
        [
            "--seed", "42",
            "--normal-count", "200",
            "--drift-count", "50",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--teacher-mode", "simulated",
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(out_json.read_text(encoding="utf-8"))

    assert rep["drift_detection_ticket_idx"] is not None
    assert rep["drift_detection_ticket_idx"] >= 200
    assert rep["drift_recovery_ticket_idx"] is not None
    assert rep["drift_recovery_ticket_idx"] > rep["drift_detection_ticket_idx"]
    assert rep["post_recovery_accuracy"] >= 0.80


def test_t1_f15_04_benchmark_markdown_matches_json_summary(tmp_path: Path) -> None:
    """T1-F15-04: Markdown benchmark report includes comparison metrics matching JSON."""
    out_json = tmp_path / "rep4.json"
    out_md = tmp_path / "rep4.md"
    proc = _run_benchmark_cli(
        [
            "--seed", "42",
            "--normal-count", "120",
            "--drift-count", "30",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--teacher-mode", "simulated",
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    md = out_md.read_text(encoding="utf-8")

    assert "Baseline" in md or "baseline" in md.lower()
    assert "Zero-Tax" in md or "zero-tax" in md.lower()
    assert f"{rep['cost_reduction_pct']:.1f}" in md or str(round(rep["cost_reduction_pct"])) in md


def test_t1_f15_05_benchmark_deterministic_reproducibility_same_seed(tmp_path: Path) -> None:
    """T1-F15-05: Identical --seed runs produce identical deterministic BenchmarkReport metrics."""
    j1 = tmp_path / "r1.json"
    m1 = tmp_path / "r1.md"
    j2 = tmp_path / "r2.json"
    m2 = tmp_path / "r2.md"

    for idx, (jp, mp) in enumerate(((j1, m1), (j2, m2)), start=1):
        run_tmp = tmp_path / f"run{idx}"
        run_tmp.mkdir(parents=True, exist_ok=True)
        proc = _run_benchmark_cli(
            [
                "--seed", "99",
                "--normal-count", "80",
                "--drift-count", "20",
                "--output-json", str(jp),
                "--output-md", str(mp),
                "--teacher-mode", "simulated",
            ],
            run_tmp,
        )
        assert proc.returncode == 0, proc.stderr

    d1 = json.loads(j1.read_text(encoding="utf-8"))
    d2 = json.loads(j2.read_text(encoding="utf-8"))
    for key in (
        "baseline_total_cost_inr",
        "zerotax_total_cost_inr",
        "intent_accuracy",
        "policy_accuracy",
        "evaporation_rate_pct",
    ):
        assert d1[key] == d2[key]


def test_t2_f15_01_benchmark_smallest_valid_batch_no_drift(tmp_path: Path) -> None:
    """T2-F15-01: Small batch (--normal-count 5 --drift-count 0) exits 0 with drift_detection=None."""
    out_json = tmp_path / "small.json"
    out_md = tmp_path / "small.md"
    proc = _run_benchmark_cli(
        [
            "--seed", "42",
            "--normal-count", "5",
            "--drift-count", "0",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--teacher-mode", "simulated",
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["drift_detection_ticket_idx"] is None


def test_t2_f15_02_benchmark_extreme_confidence_threshold_one(tmp_path: Path) -> None:
    """T2-F15-02: --confidence-threshold 1.0 drives high fallback_rate post-promotion."""
    out_json = tmp_path / "t1.json"
    out_md = tmp_path / "t1.md"
    proc = _run_benchmark_cli(
        [
            "--seed", "42",
            "--normal-count", "80",
            "--drift-count", "10",
            "--confidence-threshold", "1.0",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--teacher-mode", "simulated",
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["fallback_rate"] > 0.0


def test_t2_f15_03_benchmark_zero_audit_rates_blind_to_drift(tmp_path: Path) -> None:
    """T2-F15-03: --audit-rate 0.0 --drift-audit-rate 0.0 misses drift (drift_detection=None)."""
    out_json = tmp_path / "blind.json"
    out_md = tmp_path / "blind.md"
    proc = _run_benchmark_cli(
        [
            "--seed", "42",
            "--normal-count", "120",
            "--drift-count", "30",
            "--confidence-threshold", "0.0",
            "--audit-rate", "0.0",
            "--drift-audit-rate", "0.0",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--teacher-mode", "simulated",
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["drift_detection_ticket_idx"] is None


def test_t2_f15_04_benchmark_invalid_cli_args_exit_nonzero(tmp_path: Path) -> None:
    """T2-F15-04: Invalid --normal-count -1 or --teacher-mode badmode exits non-zero cleanly."""
    p1 = _run_benchmark_cli(
        [
            "--normal-count", "-1",
            "--output-json", str(tmp_path / "bad1.json"),
            "--output-md", str(tmp_path / "bad1.md"),
        ],
        tmp_path,
    )
    p2 = _run_benchmark_cli(
        [
            "--teacher-mode", "badmode",
            "--output-json", str(tmp_path / "bad2.json"),
            "--output-md", str(tmp_path / "bad2.md"),
        ],
        tmp_path,
    )
    assert p1.returncode != 0
    assert p2.returncode != 0


def test_t2_f15_05_benchmark_creates_nested_output_directories(tmp_path: Path) -> None:
    """T2-F15-05: Nested nonexistent parent dir for --output-json / --output-md succeeds cleanly."""
    out_json = tmp_path / "nested" / "sub" / "report.json"
    out_md = tmp_path / "nested" / "sub" / "report.md"
    proc = _run_benchmark_cli(
        [
            "--seed", "42",
            "--normal-count", "20",
            "--drift-count", "0",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--teacher-mode", "simulated",
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_json.exists() and json.loads(out_json.read_text(encoding="utf-8"))
    assert out_md.exists()
