"""Reproducible CLI benchmark comparing Naive All-LLM Baseline vs ZERO-TAX Self-Evaporating Graph."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional

from core.dataset import generate_dataset
from core.distiller import DistillerConfig
from core.graph import ZeroTaxEngine
from core.models import USD_TO_INR, BenchmarkReport, ExecutionMode, TeacherMode
from core.teacher import LLMTeacher


def _non_negative_int(value: str) -> int:
    iv = int(value)
    if iv < 0:
        raise argparse.ArgumentTypeError(f"Expected non-negative integer, got {value}")
    return iv


def run_benchmark(
    *,
    seed: int = 42,
    normal_count: int = 200,
    drift_count: int = 50,
    confidence_threshold: float = 0.90,
    audit_rate: float = 0.05,
    drift_audit_rate: float = 1.0,
    teacher_mode: str = "simulated",
    output_json: str = "reports/benchmark_report.json",
    output_md: str = "reports/benchmark_report.md",
) -> BenchmarkReport:
    mode_enum = TeacherMode(teacher_mode)
    bundle = generate_dataset(
        seed=seed,
        main_count=normal_count,
        holdout_count=50,
        drift_count=drift_count,
    )
    all_stream = [*bundle.train_stream, *bundle.drift_wave]

    # 1. Naive All-LLM Baseline
    baseline_teacher = LLMTeacher(
        mode=mode_enum,
        simulate_latency=True,
        sleep_on_simulated=False,
        seed=seed,
    )
    base_total_usd = 0.0
    base_dist_usd = 0.0
    base_total_lat = 0.0
    for ticket in all_stream:
        p1 = baseline_teacher.classify_intent(ticket)
        p2 = baseline_teacher.route_policy(ticket, p1.prediction)
        p3 = baseline_teacher.write_reply(ticket, p1.prediction, p2.prediction)
        base_dist_usd += p1.cost_usd + p2.cost_usd
        base_total_usd += p1.cost_usd + p2.cost_usd + p3.cost_usd
        base_total_lat += p1.latency_ms + p2.latency_ms + p3.latency_ms

    n_total = len(all_stream)
    base_total_inr = round(base_total_usd * USD_TO_INR, 8)
    base_dist_inr = round(base_dist_usd * USD_TO_INR, 8)
    base_avg_lat = round(base_total_lat / n_total, 4) if n_total > 0 else 0.0

    # 2. ZERO-TAX Self-Evaporating LangGraph Engine
    env_db = os.getenv("ZEROTAX_DB_PATH")
    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if env_db:
        db_path = env_db
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        db_path = str(Path(tmp_dir.name) / "bench_checkpoints.db")

    cfg = DistillerConfig(
        confidence_threshold=confidence_threshold,
        shadow_audit_rate=audit_rate,
    )
    engine = ZeroTaxEngine(
        db_path=db_path,
        teacher_mode=mode_enum,
        config=cfg,
        seed=seed,
        sleep_on_simulated=False,
    )

    try:
        thread_id = f"bench-{seed}"
        zt_total_usd = 0.0
        zt_dist_usd = 0.0
        zt_total_lat = 0.0
        intent_correct = 0
        policy_correct = 0
        dist_calls = 0
        dist_student_calls = 0
        dist_fallback_calls = 0

        drift_detection_idx: Optional[int] = None
        drift_recovery_idx: Optional[int] = None
        demoted_seen = False

        # Normal stream
        for ticket in bundle.train_stream:
            res = engine.run_ticket(
                ticket,
                thread_id=thread_id,
                auto_approve=True,
                audit_rate=audit_rate,
            )
            tel = res["telemetry"]
            ir = tel["intent_result"]
            pr = tel["policy_result"]
            rr = tel["reply_result"]
            zt_dist_usd += ir["cost_usd"] + pr["cost_usd"]
            zt_total_usd += ir["cost_usd"] + pr["cost_usd"] + rr["cost_usd"]
            zt_total_lat += ir["latency_ms"] + pr["latency_ms"] + rr["latency_ms"]

            if ticket.ground_truth_intent and ir["prediction"] == ticket.ground_truth_intent.value:
                intent_correct += 1
            if ticket.ground_truth_policy and pr["prediction"] == ticket.ground_truth_policy.value:
                policy_correct += 1

            for nr in (ir, pr):
                dist_calls += 1
                if nr["execution_mode"] == ExecutionMode.SKLEARN_STUDENT.value:
                    dist_student_calls += 1
                elif nr["execution_mode"] == ExecutionMode.LOW_CONF_FALLBACK.value:
                    dist_fallback_calls += 1

        # Drift wave stream
        for offset, ticket in enumerate(bundle.drift_wave):
            global_idx = normal_count + offset
            res = engine.run_ticket(
                ticket,
                thread_id=thread_id,
                auto_approve=True,
                audit_rate=drift_audit_rate,
            )
            tel = res["telemetry"]
            ir = tel["intent_result"]
            pr = tel["policy_result"]
            rr = tel["reply_result"]
            zt_dist_usd += ir["cost_usd"] + pr["cost_usd"]
            zt_total_usd += ir["cost_usd"] + pr["cost_usd"] + rr["cost_usd"]
            zt_total_lat += ir["latency_ms"] + pr["latency_ms"] + rr["latency_ms"]

            if ticket.ground_truth_intent and ir["prediction"] == ticket.ground_truth_intent.value:
                intent_correct += 1
            if ticket.ground_truth_policy and pr["prediction"] == ticket.ground_truth_policy.value:
                policy_correct += 1

            for nr in (ir, pr):
                dist_calls += 1
                if nr["execution_mode"] == ExecutionMode.SKLEARN_STUDENT.value:
                    dist_student_calls += 1
                elif nr["execution_mode"] == ExecutionMode.LOW_CONF_FALLBACK.value:
                    dist_fallback_calls += 1

            if (res.get("drift_demoted_nodes") or []) and drift_detection_idx is None:
                drift_detection_idx = global_idx
                demoted_seen = True

            if demoted_seen and drift_recovery_idx is None:
                decisions = res.get("promotion_decisions") or []
                if any(d.get("approved") and d.get("node_name") == "intent_classifier" for d in decisions):
                    drift_recovery_idx = global_idx

        # Post-recovery accuracy evaluation
        eval_probes = list(bundle.eval_holdout[:20])
        if drift_count > 0:
            eval_probes.extend(bundle.drift_wave[-10:])
        post_ok = 0
        for probe in eval_probes:
            r_eval = engine.run_ticket(probe, thread_id=f"{thread_id}-posteval", auto_approve=True, audit_rate=0.0)
            tel_eval = r_eval.get("telemetry")
            if (
                tel_eval
                and probe.ground_truth_intent
                and tel_eval["intent_result"]["prediction"] == probe.ground_truth_intent.value
            ):
                post_ok += 1
        post_recovery_acc = round(post_ok / len(eval_probes), 4) if eval_probes else 1.0

        metrics_snapshot = engine.get_metrics(thread_id)
        zt_total_inr = round(zt_total_usd * USD_TO_INR, 8)
        zt_dist_inr = round(zt_dist_usd * USD_TO_INR, 8)
        zt_avg_lat = round(zt_total_lat / n_total, 4) if n_total > 0 else 0.0

        cost_red_pct = (
            round(100.0 * (1.0 - (zt_total_usd / base_total_usd)), 2) if base_total_usd > 0 else 0.0
        )
        dist_cost_red_pct = (
            round(100.0 * (1.0 - (zt_dist_usd / base_dist_usd)), 2) if base_dist_usd > 0 else 0.0
        )
        lat_red_pct = (
            round(100.0 * (1.0 - (zt_avg_lat / base_avg_lat)), 2) if base_avg_lat > 0 else 0.0
        )

        report = BenchmarkReport(
            seed=seed,
            normal_count=normal_count,
            drift_count=drift_count,
            teacher_mode=mode_enum.value,
            baseline_total_cost_usd=round(base_total_usd, 8),
            baseline_total_cost_inr=base_total_inr,
            zerotax_total_cost_usd=round(zt_total_usd, 8),
            zerotax_total_cost_inr=zt_total_inr,
            cost_reduction_pct=cost_red_pct,
            distillable_baseline_cost_inr=base_dist_inr,
            distillable_zerotax_cost_inr=zt_dist_inr,
            distillable_cost_reduction_pct=dist_cost_red_pct,
            baseline_avg_latency_ms=base_avg_lat,
            zerotax_avg_latency_ms=zt_avg_lat,
            latency_reduction_pct=lat_red_pct,
            intent_accuracy=round(intent_correct / n_total, 4) if n_total > 0 else 0.0,
            policy_accuracy=round(policy_correct / n_total, 4) if n_total > 0 else 0.0,
            evaporation_rate_pct=round(100.0 * dist_student_calls / dist_calls, 2) if dist_calls > 0 else 0.0,
            fallback_rate=round(dist_fallback_calls / dist_calls, 4) if dist_calls > 0 else 0.0,
            drift_detection_ticket_idx=drift_detection_idx,
            drift_recovery_ticket_idx=drift_recovery_idx,
            post_recovery_accuracy=post_recovery_acc,
            per_node_metrics=metrics_snapshot.get("nodes", {}),
        )
    finally:
        engine.close()
        if tmp_dir is not None:
            import gc

            gc.collect()
            try:
                tmp_dir.cleanup()
            except OSError:
                pass

    json_path = Path(output_json)
    md_path = Path(output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_text = _render_markdown_report(report)
    md_path.write_text(md_text, encoding="utf-8")
    return report


def _render_markdown_report(r: BenchmarkReport) -> str:
    return (
        "# ZERO-TAX Benchmark Report\n\n"
        f"- **Seed**: `{r.seed}` | **Normal Tickets**: `{r.normal_count}` | **Drift Wave**: `{r.drift_count}` | **Teacher**: `{r.teacher_mode}`\n\n"
        "| Metric | Naive All-LLM Baseline | Zero-Tax LangGraph Engine | Improvement |\n"
        "|---|---|---|---|\n"
        f"| Full-Graph Cost (INR) | Rs. {r.baseline_total_cost_inr:.4f} | Rs. {r.zerotax_total_cost_inr:.4f} | **{r.cost_reduction_pct:.1f}% saved** |\n"
        f"| Distillable Nodes Cost (INR) | Rs. {r.distillable_baseline_cost_inr:.4f} | Rs. {r.distillable_zerotax_cost_inr:.4f} | **{r.distillable_cost_reduction_pct:.1f}% saved** |\n"
        f"| Avg Latency / Ticket | {r.baseline_avg_latency_ms:.1f} ms | {r.zerotax_avg_latency_ms:.1f} ms | **{r.latency_reduction_pct:.1f}% faster** |\n"
        f"| Intent Accuracy | - | {r.intent_accuracy*100:.1f}% | - |\n"
        f"| Policy Accuracy | - | {r.policy_accuracy*100:.1f}% | - |\n"
        f"| Student Evaporation Rate | 0.0% | {r.evaporation_rate_pct:.1f}% | - |\n"
        f"| Low-Confidence Fallback Rate | - | {r.fallback_rate*100:.2f}% | - |\n"
        f"| Drift Detected Ticket Index | - | `{r.drift_detection_ticket_idx}` | Auto-Demoted |\n"
        f"| Drift Recovery Ticket Index | - | `{r.drift_recovery_ticket_idx}` | Re-Promoted |\n"
        f"| Post-Recovery Accuracy | - | {r.post_recovery_accuracy*100:.1f}% | - |\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ZERO-TAX Reproducible Benchmark CLI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-count", type=_non_negative_int, default=200)
    parser.add_argument("--drift-count", type=_non_negative_int, default=50)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--audit-rate", type=float, default=0.05)
    parser.add_argument("--drift-audit-rate", type=float, default=1.0)
    parser.add_argument("--teacher-mode", choices=["auto", "simulated", "gemini", "openai"], default="simulated")
    parser.add_argument("--output-json", type=str, default="reports/benchmark_report.json")
    parser.add_argument("--output-md", type=str, default="reports/benchmark_report.md")

    args = parser.parse_args()
    report = run_benchmark(
        seed=args.seed,
        normal_count=args.normal_count,
        drift_count=args.drift_count,
        confidence_threshold=args.confidence_threshold,
        audit_rate=args.audit_rate,
        drift_audit_rate=args.drift_audit_rate,
        teacher_mode=args.teacher_mode,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(_render_markdown_report(report))


if __name__ == "__main__":
    main()
