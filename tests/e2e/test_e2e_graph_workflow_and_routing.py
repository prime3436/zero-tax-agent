"""E2E Tests — F01, F05, F06, F07: LangGraph 3-Node Workflow, Selective Distillation, Conditional Routing & Store.

Progressive layer: M2 (core.graph).
Tests: 20 Tier 1 + 20 Tier 2 = 40 tests.
"""

from __future__ import annotations

import pytest


# ===========================================================================
# F01: 3-Node E-Commerce Support Workflow (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f01_01_cold_start_damaged_item_ticket(engine, dataset_bundle) -> None:
    """T1-F01-01: Cold-start DAMAGED_ITEM ticket runs all 3 nodes as LLM_TEACHER."""
    from core.models import ExecutionMode, IntentLabel, PolicyAction
    from e2e_helpers import enum_str, get_node_result, get_reply_text, make_ticket

    ticket = make_ticket(
        ticket_id="F01-DAMAGED",
        category="FOOTWEAR",
        item_price_inr=2499.0,
        customer_return_rate=0.05,
        days_since_delivery=2,
        ticket_text="My sneaker sole came glued crooked and tore off immediately",
        ground_truth_intent=IntentLabel.DAMAGED_ITEM.value,
    )
    res = engine.run_ticket(ticket, thread_id="t1-f01-01")
    intent_r = get_node_result(res, "intent_classifier")
    policy_r = get_node_result(res, "policy_router")
    reply_r = get_node_result(res, "reply_writer")

    assert enum_str(intent_r["prediction"]) == IntentLabel.DAMAGED_ITEM.value
    assert enum_str(policy_r["prediction"]) in {e.value for e in PolicyAction}
    assert len(get_reply_text(res).strip()) >= 15
    assert enum_str(intent_r["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
    assert enum_str(policy_r["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
    assert enum_str(reply_r["execution_mode"]) == ExecutionMode.LLM_TEACHER.value


def test_t1_f01_02_cold_start_late_delivery_cost_inr_usd_ratio(engine) -> None:
    """T1-F01-02: Cold-start LATE_DELIVERY ticket satisfies USD_TO_INR=84.50 accounting."""
    from core.models import USD_TO_INR, IntentLabel, PolicyAction
    from e2e_helpers import enum_str, get_node_result, get_telemetry, make_ticket

    ticket = make_ticket(
        ticket_id="F01-LATE",
        category="ELECTRONICS",
        item_price_inr=1499.0,
        customer_return_rate=0.04,
        days_since_delivery=10,
        ticket_text="My parcel delivery is delayed by 8 days with no courier tracking updates",
    )
    res = engine.run_ticket(ticket, thread_id="t1-f01-02")
    tel = get_telemetry(res)
    intent_r = get_node_result(res, "intent_classifier")
    policy_r = get_node_result(res, "policy_router")

    assert enum_str(intent_r["prediction"]) in {e.value for e in IntentLabel}
    assert enum_str(policy_r["prediction"]) in {e.value for e in PolicyAction}
    assert tel["total_cost_usd"] > 0.0
    assert tel["total_cost_inr"] == pytest.approx(tel["total_cost_usd"] * USD_TO_INR, rel=1e-3)


def test_t1_f01_03_cold_start_wrong_size_latency_sum(engine) -> None:
    """T1-F01-03: Cold-start WRONG_SIZE ticket sums node latencies accurately."""
    from core.models import IntentLabel, PolicyAction
    from e2e_helpers import enum_str, get_node_result, get_telemetry, make_ticket

    ticket = make_ticket(
        ticket_id="F01-SIZE",
        category="APPAREL",
        item_price_inr=899.0,
        customer_return_rate=0.10,
        days_since_delivery=3,
        ticket_text="Ordered size M kurta but received XL which does not fit at all",
    )
    res = engine.run_ticket(ticket, thread_id="t1-f01-03")
    tel = get_telemetry(res)
    intent_r = get_node_result(res, "intent_classifier")
    policy_r = get_node_result(res, "policy_router")
    reply_r = get_node_result(res, "reply_writer")

    assert enum_str(intent_r["prediction"]) == IntentLabel.WRONG_SIZE.value
    assert enum_str(policy_r["prediction"]) in {e.value for e in PolicyAction}
    node_sum = float(intent_r["latency_ms"]) + float(policy_r["latency_ms"]) + float(reply_r["latency_ms"])
    assert tel["total_latency_ms"] >= node_sum * 0.99


def test_t1_f01_04_cold_start_fraud_suspect_high_risk_ticket(engine) -> None:
    """T1-F01-04: Cold-start FRAUD_SUSPECT high return-rate ticket routes cleanly."""
    from core.models import IntentLabel, PolicyAction
    from e2e_helpers import enum_str, get_node_result, get_reply_text, make_ticket

    ticket = make_ticket(
        ticket_id="F01-FRAUD",
        category="ELECTRONICS",
        item_price_inr=6500.0,
        customer_return_rate=0.55,
        days_since_delivery=20,
        ticket_text="Claiming item missing again after serial mismatch on high value order",
    )
    res = engine.run_ticket(ticket, thread_id="t1-f01-04")
    intent_r = get_node_result(res, "intent_classifier")
    policy_r = get_node_result(res, "policy_router")

    assert enum_str(intent_r["prediction"]) in {e.value for e in IntentLabel}
    assert enum_str(policy_r["prediction"]) == PolicyAction.MANUAL_INSPECTION.value
    assert len(get_reply_text(res).strip()) >= 20


def test_t1_f01_05_stream_batch_60_tickets_auto_approve(engine, dataset_bundle) -> None:
    """T1-F01-05: stream_batch processes 60 tickets with auto_approve=True and high accuracy."""
    from e2e_helpers import enum_str, get_node_result

    tickets = dataset_bundle.train_stream[:60]
    res = engine.stream_batch(
        tickets,
        thread_id="t1-f01-05",
        auto_approve=True,
    )
    processed = int(res["processed_count"]) if "processed_count" in res else len(res["results"])
    assert processed == 60
    if "intent_accuracy" in res and "policy_accuracy" in res:
        intent_acc = float(res["intent_accuracy"])
        policy_acc = float(res["policy_accuracy"])
    else:
        results_list = res["results"]
        assert len(results_list) == 60
        intent_acc = (
            sum(
                1
                for t, r in zip(tickets, results_list)
                if enum_str(get_node_result(r, "intent_classifier")["prediction"])
                == t.ground_truth_intent.value
            )
            / 60.0
        )
        policy_acc = (
            sum(
                1
                for t, r in zip(tickets, results_list)
                if enum_str(get_node_result(r, "policy_router")["prediction"])
                == t.ground_truth_policy.value
            )
            / 60.0
        )
    assert intent_acc >= 0.85
    assert policy_acc >= 0.85


def test_t2_f01_01_empty_batch_stream_graceful(engine) -> None:
    """T2-F01-01: stream_batch([]) returns gracefully with 0 processed and no interrupt."""
    res = engine.stream_batch([], thread_id="t2-f01-empty")
    processed = int(res["processed_count"]) if "processed_count" in res else len(res["results"])
    assert processed == 0
    assert not res.get("interrupted", False)


def test_t2_f01_02_single_ticket_batch(engine, dataset_bundle) -> None:
    """T2-F01-02: Single-ticket batch processes 1 ticket with no premature promotion."""
    from e2e_helpers import get_node_metrics

    res = engine.stream_batch([dataset_bundle.train_stream[0]], thread_id="t2-f01-one")
    processed = int(res["processed_count"]) if "processed_count" in res else len(res["results"])
    assert processed == 1
    assert not res.get("interrupted", False)
    metrics = engine.get_metrics("t2-f01-one")
    m_intent = get_node_metrics(metrics, "intent_classifier")
    m_policy = get_node_metrics(metrics, "policy_router")
    assert m_intent["is_promoted"] is False
    assert m_policy["is_promoted"] is False


def test_t2_f01_03_minimal_single_char_ticket_text(engine) -> None:
    """T2-F01-03: Minimal 1-char ticket_text '?' completes all 3 nodes without crash."""
    from core.models import IntentLabel, PolicyAction
    from e2e_helpers import enum_str, get_node_result, get_reply_text, make_ticket

    res = engine.run_ticket(make_ticket(ticket_id="T2-CHAR", ticket_text="?"), thread_id="t2-f01-char")
    assert enum_str(get_node_result(res, "intent_classifier")["prediction"]) in {e.value for e in IntentLabel}
    assert enum_str(get_node_result(res, "policy_router")["prediction"]) in {e.value for e in PolicyAction}
    assert len(get_reply_text(res).strip()) > 0


def test_t2_f01_04_very_long_5000_char_ticket_text(engine) -> None:
    """T2-F01-04: 5000+ char ticket_text runs cleanly and bills higher input_tokens."""
    from e2e_helpers import get_node_result, make_ticket

    short_res = engine.run_ticket(make_ticket(ticket_id="T2-SHORT", ticket_text="Shoe damaged"), thread_id="t2-f01-s")
    long_res = engine.run_ticket(
        make_ticket(ticket_id="T2-LONG", ticket_text="Shoe sole damaged. " * 300),
        thread_id="t2-f01-l",
    )
    assert get_node_result(long_res, "intent_classifier")["input_tokens"] > get_node_result(
        short_res, "intent_classifier"
    )["input_tokens"]


def test_t2_f01_05_invalid_payload_to_run_ticket_raises(engine) -> None:
    """T2-F01-05: Passing invalid dict/None to run_ticket raises explicit ValidationError/TypeError."""
    from pydantic import ValidationError

    for bad_input in ({"bad": "payload"}, None):
        with pytest.raises((ValidationError, TypeError, ValueError, AttributeError)):
            engine.run_ticket(bad_input, thread_id="t2-f01-bad")  # type: ignore[arg-type]


# ===========================================================================
# F05: Node 3 reply_writer Always-LLM Selective Distillation (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f05_01_reply_writer_cold_start_always_llm(engine, dataset_bundle) -> None:
    """T1-F05-01: reply_writer executes as LLM_TEACHER on cold start."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    res = engine.run_ticket(dataset_bundle.train_stream[0], thread_id="t1-f05-01")
    reply_r = get_node_result(res, "reply_writer")
    assert enum_str(reply_r["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
    assert reply_r["cost_usd"] > 0.0
    assert reply_r["latency_ms"] > 0.0


def test_t1_f05_02_reply_writer_stays_llm_after_nodes_1_and_2_promoted(engine, dataset_bundle) -> None:
    """T1-F05-02: Even after Nodes 1 & 2 promote, reply_writer runs LLM_TEACHER on 5/5 holdout tickets."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f05-02", auto_approve=True)

    for idx, ticket in enumerate(dataset_bundle.eval_holdout[:5]):
        res = engine.run_ticket(ticket, thread_id="t1-f05-02", audit_rate=0.0)
        intent_r = get_node_result(res, "intent_classifier")
        policy_r = get_node_result(res, "policy_router")
        reply_r = get_node_result(res, "reply_writer")

        assert enum_str(intent_r["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value, f"idx={idx}"
        assert enum_str(policy_r["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value, f"idx={idx}"
        assert enum_str(reply_r["execution_mode"]) == ExecutionMode.LLM_TEACHER.value, f"idx={idx}"


def test_t1_f05_03_reply_writer_never_promoted_in_metrics(engine, dataset_bundle) -> None:
    """T1-F05-03: reply_writer never promotes and runs as LLM_TEACHER with cost > 0."""
    from core.models import ExecutionMode, NodeStatus
    from e2e_helpers import enum_str, get_all_node_metrics, get_node_result

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f05-03", auto_approve=True)
    all_nodes = get_all_node_metrics(engine.get_metrics("t1-f05-03"))
    assert "intent_classifier" in all_nodes and "policy_router" in all_nodes
    if "reply_writer" in all_nodes:
        m_reply = all_nodes["reply_writer"]
        assert m_reply["is_promoted"] is False
        assert m_reply["student_calls"] == 0
        assert enum_str(m_reply["status"]) == NodeStatus.TEACHER_ACTIVE.value

    res = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t1-f05-03", audit_rate=0.0)
    reply_r = get_node_result(res, "reply_writer")
    assert enum_str(reply_r["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
    assert reply_r["cost_usd"] > 0.0


def test_t1_f05_04_reply_text_reflects_ticket_context(engine) -> None:
    """T1-F05-04: reply_writer output reflects customer/order/resolution context."""
    from e2e_helpers import get_reply_text, make_ticket

    ticket = make_ticket(
        customer_name="Aarav Sharma",
        order_id="ORD-99123",
        ticket_text="Sneaker sole came detached right out of the box",
    )
    res = engine.run_ticket(ticket, thread_id="t1-f05-04")
    reply = get_reply_text(res)
    assert len(reply.strip()) >= 20
    assert ("ORD-99123" in reply) or ("Aarav" in reply) or ("refund" in reply.lower()) or ("replacement" in reply.lower())


def test_t1_f05_05_selective_distillation_cost_invariant(engine, dataset_bundle) -> None:
    """T1-F05-05: When Nodes 1 & 2 are SKLEARN_STUDENT (audit_rate=0), total_cost == reply_writer cost."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, get_telemetry

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f05-05", auto_approve=True)
    verified_student_tickets = 0
    for ticket in dataset_bundle.eval_holdout[:5]:
        res = engine.run_ticket(ticket, thread_id="t1-f05-05", audit_rate=0.0)
        intent_r = get_node_result(res, "intent_classifier")
        policy_r = get_node_result(res, "policy_router")
        reply_r = get_node_result(res, "reply_writer")
        tel = get_telemetry(res)

        if (
            enum_str(intent_r["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
            and enum_str(policy_r["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
        ):
            assert intent_r["cost_usd"] == 0.0
            assert policy_r["cost_usd"] == 0.0
            assert tel["total_cost_usd"] == pytest.approx(reply_r["cost_usd"], rel=1e-5)
            verified_student_tickets += 1

    assert verified_student_tickets >= 1


def test_t2_f05_01_reply_writer_on_empty_ticket_text(engine) -> None:
    """T2-F05-01: Empty ticket_text still produces empathetic non-empty LLM reply."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, get_reply_text, make_ticket

    res = engine.run_ticket(make_ticket(ticket_text=""), thread_id="t2-f05-01")
    reply_r = get_node_result(res, "reply_writer")
    assert enum_str(reply_r["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
    assert len(get_reply_text(res).strip()) > 0


def test_t2_f05_02_reply_writer_remains_llm_after_drift_demotion(engine, dataset_bundle) -> None:
    """T2-F05-02: reply_writer continues executing as LLM_TEACHER after drift demotion."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t2-f05-02", auto_approve=True)
    engine.inject_drift(dataset_bundle.drift_wave[:25], thread_id="t2-f05-02", audit_rate=1.0)
    res = engine.run_ticket(dataset_bundle.drift_wave[26], thread_id="t2-f05-02")
    assert enum_str(get_node_result(res, "reply_writer")["execution_mode"]) == ExecutionMode.LLM_TEACHER.value


def test_t2_f05_03_reply_writer_distinct_replies_across_policies(engine, dataset_bundle) -> None:
    """T2-F05-03: reply_writer produces policy-aligned replies across refund/pickup/inspection."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, get_reply_text, make_ticket

    t_refund = make_ticket(ticket_id="P1", item_price_inr=350.0, customer_return_rate=0.02, days_since_delivery=2,
                           ticket_text="Torn collar on cotton shirt")
    t_pickup = make_ticket(ticket_id="P2", item_price_inr=2200.0, customer_return_rate=0.05, days_since_delivery=2,
                           ticket_text="Ordered size 8 shoes but received size 10")
    t_inspect = make_ticket(ticket_id="P3", item_price_inr=11000.0, customer_return_rate=0.65, days_since_delivery=20,
                            ticket_text="Empty brick inside electronics box scam claim")

    replies = set()
    for idx, t in enumerate((t_refund, t_pickup, t_inspect)):
        res = engine.run_ticket(t, thread_id=f"t2-f05-03-{idx}")
        assert enum_str(get_node_result(res, "reply_writer")["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
        replies.add(get_reply_text(res))
    assert len(replies) >= 2


def test_t2_f05_04_reply_writer_never_zero_cost_even_with_zero_threshold(make_engine, dataset_bundle) -> None:
    """T2-F05-04: Even with confidence_threshold=0.0, reply_writer remains LLM_TEACHER with cost > 0."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.0))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f05-04", auto_approve=True)
    res = eng.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t2-f05-04", audit_rate=0.0)

    assert enum_str(get_node_result(res, "intent_classifier")["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
    assert enum_str(get_node_result(res, "policy_router")["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
    reply_r = get_node_result(res, "reply_writer")
    assert enum_str(reply_r["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
    assert reply_r["cost_usd"] > 0.0


def test_t2_f05_05_reply_writer_zero_promotions_after_full_200_stream(engine, dataset_bundle) -> None:
    """T2-F05-05: After 200 tickets, reply_writer has 0 promotions/student calls and stays LLM_TEACHER."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_all_node_metrics, get_node_result

    engine.stream_batch(dataset_bundle.train_stream, thread_id="t2-f05-05", auto_approve=True)
    all_nodes = get_all_node_metrics(engine.get_metrics("t2-f05-05"))
    assert "intent_classifier" in all_nodes and "policy_router" in all_nodes
    if "reply_writer" in all_nodes:
        m_reply = all_nodes["reply_writer"]
        assert m_reply["promotion_count"] == 0
        assert m_reply["student_calls"] == 0

    res = engine.run_ticket(dataset_bundle.eval_holdout[1], thread_id="t2-f05-05", audit_rate=0.0)
    reply_r = get_node_result(res, "reply_writer")
    assert enum_str(reply_r["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
    assert reply_r["cost_usd"] > 0.0


# ===========================================================================
# F06: StateGraph Conditional Routing (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f06_01_pre_promotion_routes_100_pct_to_teacher_path(engine, dataset_bundle) -> None:
    """T1-F06-01: First 10 tickets route 100% through teacher_llm_path."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    for ticket in dataset_bundle.train_stream[:10]:
        res = engine.run_ticket(ticket, thread_id="t1-f06-01")
        assert enum_str(get_node_result(res, "intent_classifier")["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
        assert enum_str(get_node_result(res, "policy_router")["execution_mode"]) == ExecutionMode.LLM_TEACHER.value


def test_t1_f06_02_post_promotion_high_conf_routes_to_student_fast_path(engine, dataset_bundle) -> None:
    """T1-F06-02: Post-promotion confident ticket takes student_fast_path at $0 & < 25ms."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f06-02", auto_approve=True)
    res = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t1-f06-02", audit_rate=0.0)
    intent_r = get_node_result(res, "intent_classifier")

    assert enum_str(intent_r["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
    assert intent_r["confidence"] >= 0.90
    assert intent_r["cost_usd"] == 0.0
    assert intent_r["latency_ms"] < 25.0


def test_t1_f06_03_mixed_per_node_routing_one_promoted_one_teacher(engine, dataset_bundle) -> None:
    """T1-F06-03: Approving Node 1 while rejecting Node 2 yields mixed routing on same ticket."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_interrupt_payload, get_node_result, to_dict

    idx = 0
    approved_nodes: set[str] = set()
    rejected_nodes: set[str] = set()
    tickets = dataset_bundle.train_stream[:140]
    while idx < len(tickets):
        res = engine.stream_batch(tickets[idx:], thread_id="t1-f06-03", auto_approve=False)
        processed = int(res.get("processed_count", 0))
        if res.get("interrupted"):
            cur = to_dict(res)
            while cur.get("interrupted"):
                payload = get_interrupt_payload(cur) or {}
                node = payload.get("node") or payload.get("node_name", "")
                assert node in {"intent_classifier", "policy_router"}
                if not approved_nodes:
                    approved_nodes.add(node)
                    cur = to_dict(engine.approve_promotion("t1-f06-03", approved=True))
                else:
                    rejected_nodes.add(node)
                    cur = to_dict(engine.approve_promotion("t1-f06-03", approved=False))
            idx += max(processed, 1)
            if len(approved_nodes) == 1 and len(rejected_nodes) == 1:
                break
        else:
            if processed <= 0:
                break
            idx += processed

    assert len(approved_nodes) == 1 and len(rejected_nodes) == 1
    app_node = next(iter(approved_nodes))
    rej_node = next(iter(rejected_nodes))
    assert app_node != rej_node

    follow = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="t1-f06-03", audit_rate=0.0)
    assert enum_str(get_node_result(follow, app_node)["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
    assert enum_str(get_node_result(follow, rej_node)["execution_mode"]) == ExecutionMode.LLM_TEACHER.value


def test_t1_f06_04_graph_mermaid_reflects_3_nodes_and_edges(engine) -> None:
    """T1-F06-04: get_graph_mermaid returns valid Mermaid diagram with all 3 nodes & conditional edges."""
    mermaid = engine.get_graph_mermaid()
    assert isinstance(mermaid, str) and len(mermaid.strip()) > 30
    assert "intent_classifier" in mermaid
    assert "policy_router" in mermaid
    assert "reply_writer" in mermaid
    assert ("-->" in mermaid) or ("-.->" in mermaid)


def test_t1_f06_05_post_evaporation_cost_and_latency_drop_80_pct(engine, dataset_bundle) -> None:
    """T1-F06-05: Last 10 tickets' distillable cost & latency drop >= 80% vs first 10 cold tickets."""
    from e2e_helpers import get_node_result

    cold_costs = []
    cold_latencies = []
    for t in dataset_bundle.train_stream[:10]:
        r = engine.run_ticket(t, thread_id="t1-f06-05", audit_rate=0.0)
        ir = get_node_result(r, "intent_classifier")
        pr = get_node_result(r, "policy_router")
        cold_costs.append(float(ir["cost_usd"]) + float(pr["cost_usd"]))
        cold_latencies.append(float(ir["latency_ms"]) + float(pr["latency_ms"]))

    engine.stream_batch(dataset_bundle.train_stream[10:140], thread_id="t1-f06-05", auto_approve=True, audit_rate=0.0)

    warm_costs = []
    warm_latencies = []
    for t in dataset_bundle.train_stream[140:150]:
        r = engine.run_ticket(t, thread_id="t1-f06-05", audit_rate=0.0)
        ir = get_node_result(r, "intent_classifier")
        pr = get_node_result(r, "policy_router")
        warm_costs.append(float(ir["cost_usd"]) + float(pr["cost_usd"]))
        warm_latencies.append(float(ir["latency_ms"]) + float(pr["latency_ms"]))

    assert sum(warm_costs) <= 0.20 * sum(cold_costs)
    assert sum(warm_latencies) <= 0.20 * sum(cold_latencies)


def test_t2_f06_01_confidence_threshold_zero_always_uses_student(make_engine, dataset_bundle) -> None:
    """T2-F06-01: With confidence_threshold=0.0, promoted nodes never trigger LOW_CONF_FALLBACK."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result, make_ticket

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.0))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f06-01", auto_approve=True)

    ambiguous = make_ticket(ticket_id="AMB-0", ticket_text="maybe late or size issue not sure blarg")
    for t in [dataset_bundle.eval_holdout[0], ambiguous]:
        res = eng.run_ticket(t, thread_id="t2-f06-01", audit_rate=0.0)
        assert enum_str(get_node_result(res, "intent_classifier")["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
        assert enum_str(get_node_result(res, "policy_router")["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value


def test_t2_f06_02_confidence_threshold_090_invariant(make_engine, dataset_bundle) -> None:
    """T2-F06-02: At confidence_threshold=0.90, STUDENT calls have conf >= 0.90; FALLBACK calls < 0.90."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    eng = make_engine(config=DistillerConfig(confidence_threshold=0.90))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f06-02", auto_approve=True)

    student_hits = 0
    for t in dataset_bundle.eval_holdout[:20]:
        res = eng.run_ticket(t, thread_id="t2-f06-02", audit_rate=0.0)
        for node in ("intent_classifier", "policy_router"):
            nr = get_node_result(res, node)
            mode = enum_str(nr["execution_mode"])
            if mode == ExecutionMode.SKLEARN_STUDENT.value:
                assert nr["confidence"] >= 0.90
                student_hits += 1
            elif mode == ExecutionMode.LOW_CONF_FALLBACK.value:
                assert nr["confidence"] < 0.90
            else:
                pytest.fail(f"Unexpected execution mode {mode} for promoted node {node}")
    assert student_hits >= 1


def test_t2_f06_03_confidence_threshold_099_increases_fallback_rate(make_engine, dataset_bundle) -> None:
    """T2-F06-03: Strict threshold 0.99 yields >= fallbacks than 0.90 on identical holdout tickets."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    eng_90 = make_engine(db_name="c90.db", config=DistillerConfig(confidence_threshold=0.90))
    eng_99 = make_engine(db_name="c99.db", config=DistillerConfig(confidence_threshold=0.99))

    eng_90.stream_batch(dataset_bundle.train_stream[:140], thread_id="t90", auto_approve=True)
    eng_99.stream_batch(dataset_bundle.train_stream[:140], thread_id="t99", auto_approve=True)

    fb_90 = 0
    fb_99 = 0
    for t in dataset_bundle.eval_holdout[:20]:
        r90 = eng_90.run_ticket(t, thread_id="t90", audit_rate=0.0)
        r99 = eng_99.run_ticket(t, thread_id="t99", audit_rate=0.0)
        if enum_str(get_node_result(r90, "intent_classifier")["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value:
            fb_90 += 1
        if enum_str(get_node_result(r99, "intent_classifier")["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value:
            fb_99 += 1

    assert fb_99 >= fb_90


def test_t2_f06_04_confidence_threshold_one_forces_fallback(make_engine, dataset_bundle) -> None:
    """T2-F06-04: At confidence_threshold=1.0, probabilities < 1.0 route to LOW_CONF_FALLBACK."""
    from core.distiller import DistillerConfig
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    eng = make_engine(config=DistillerConfig(confidence_threshold=1.0))
    eng.stream_batch(dataset_bundle.train_stream[:140], thread_id="t2-f06-04", auto_approve=True)

    fallbacks = 0
    for t in dataset_bundle.eval_holdout[:10]:
        res = eng.run_ticket(t, thread_id="t2-f06-04", audit_rate=0.0)
        ir = get_node_result(res, "intent_classifier")
        if ir["confidence"] < 1.0:
            assert enum_str(ir["execution_mode"]) == ExecutionMode.LOW_CONF_FALLBACK.value
            fallbacks += 1
    assert fallbacks >= 5


def test_t2_f06_05_rejected_promotion_stays_on_teacher_llm_path(engine, dataset_bundle) -> None:
    """T2-F06-05: Rejecting promotion keeps conditional edge routing to LLM_TEACHER."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_metrics, get_node_result, to_dict

    idx = 0
    saw_interrupt = False
    tickets = dataset_bundle.train_stream[:120]
    while idx < len(tickets):
        res = engine.stream_batch(tickets[idx:], thread_id="t2-f06-05", auto_approve=False)
        processed = int(res.get("processed_count", 0))
        if res.get("interrupted"):
            saw_interrupt = True
            cur = to_dict(engine.approve_promotion("t2-f06-05", approved=False))
            while cur.get("interrupted"):
                cur = to_dict(engine.approve_promotion("t2-f06-05", approved=False))
            break
        if processed <= 0:
            break
        idx += processed

    assert saw_interrupt is True
    metrics = engine.get_metrics("t2-f06-05")
    m_i = get_node_metrics(metrics, "intent_classifier")
    m_p = get_node_metrics(metrics, "policy_router")
    assert not m_i["is_promoted"] and not m_p["is_promoted"]

    for t in dataset_bundle.eval_holdout[:10]:
        r = engine.run_ticket(t, thread_id="t2-f06-05")
        ir = get_node_result(r, "intent_classifier")
        pr = get_node_result(r, "policy_router")
        assert enum_str(ir["execution_mode"]) == ExecutionMode.LLM_TEACHER.value
        assert enum_str(pr["execution_mode"]) == ExecutionMode.LLM_TEACHER.value


# ===========================================================================
# F07: LangGraph Store Persistence (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f07_01_store_persists_per_node_traces(engine, dataset_bundle) -> None:
    """T1-F07-01: Running 15 tickets persists 15 traces per distillable node in Store/metrics."""
    from e2e_helpers import get_node_metrics

    for t in dataset_bundle.train_stream[:15]:
        engine.run_ticket(t, thread_id="t1-f07-01")

    metrics = engine.get_metrics("t1-f07-01")
    assert get_node_metrics(metrics, "intent_classifier")["trace_count"] == 15
    assert get_node_metrics(metrics, "policy_router")["trace_count"] == 15
    if hasattr(engine, "store") and engine.store is not None:
        items = list(engine.store.search(()))
        assert len(items) > 0


def test_t1_f07_02_store_shares_promotion_status_across_threads(engine, dataset_bundle) -> None:
    """T1-F07-02: Nodes promoted on thread A are immediately active in Store for thread B."""
    from core.models import ExecutionMode
    from e2e_helpers import enum_str, get_node_result

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="f07-thread-a", auto_approve=True)
    res_b = engine.run_ticket(dataset_bundle.eval_holdout[0], thread_id="f07-thread-b", audit_rate=0.0)

    assert enum_str(get_node_result(res_b, "intent_classifier")["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value
    assert enum_str(get_node_result(res_b, "policy_router")["execution_mode"]) == ExecutionMode.SKLEARN_STUDENT.value


def test_t1_f07_03_store_persists_shadow_audit_records(engine, dataset_bundle) -> None:
    """T1-F07-03: Shadow audit records and rolling agreement are persisted in Store/metrics."""
    from e2e_helpers import get_node_metrics

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f07-03", auto_approve=True, audit_rate=0.0)
    for t in dataset_bundle.eval_holdout[:20]:
        engine.run_ticket(t, thread_id="t1-f07-03", audit_rate=1.0)

    m_intent = get_node_metrics(engine.get_metrics("t1-f07-03"), "intent_classifier")
    assert m_intent["shadow_audit_count"] >= 15
    assert m_intent["rolling_shadow_agreement"] is not None


def test_t1_f07_04_store_reflects_demotion_after_drift_breaker(engine, dataset_bundle) -> None:
    """T1-F07-04: Store/metrics transition to DEMOTED_DRIFT / demotion_count >= 1 after drift."""
    from e2e_helpers import get_node_metrics

    engine.stream_batch(dataset_bundle.train_stream[:140], thread_id="t1-f07-04", auto_approve=True)
    engine.inject_drift(dataset_bundle.drift_wave[:25], thread_id="t1-f07-04", audit_rate=1.0)

    m_intent = get_node_metrics(engine.get_metrics("t1-f07-04"), "intent_classifier")
    assert m_intent["demotion_count"] >= 1


def test_t1_f07_05_engine_reset_clears_store_and_node_state(engine, dataset_bundle) -> None:
    """T1-F07-05: engine.reset() clears Store traces, promotion state, and student counters."""
    from core.models import NodeStatus
    from e2e_helpers import enum_str, get_node_metrics

    engine.stream_batch(dataset_bundle.train_stream[:120], thread_id="t1-f07-05", auto_approve=True)
    engine.reset()
    engine.run_ticket(dataset_bundle.train_stream[0], thread_id="t1-f07-05")

    m_intent = get_node_metrics(engine.get_metrics("t1-f07-05"), "intent_classifier")
    assert m_intent["trace_count"] == 1
    assert m_intent["is_promoted"] is False
    assert m_intent["student_calls"] == 0
    assert enum_str(m_intent["status"]) == NodeStatus.TEACHER_ACTIVE.value


def test_t2_f07_01_fresh_engine_zero_tickets_metrics_no_zero_division(engine) -> None:
    """T2-F07-01: Querying get_metrics() on fresh engine with 0 tickets succeeds cleanly."""
    from e2e_helpers import get_node_metrics

    metrics = engine.get_metrics()
    m_intent = get_node_metrics(metrics, "intent_classifier")
    assert m_intent["trace_count"] == 0
    assert m_intent["student_calls"] == 0
    assert m_intent["teacher_calls"] == 0


def test_t2_f07_02_metrics_on_nonexistent_thread_id_graceful(engine) -> None:
    """T2-F07-02: get_metrics(thread_id='nonexistent-thread') returns cleanly without error."""
    metrics = engine.get_metrics(thread_id="nonexistent-thread-xyz")
    assert isinstance(metrics, dict)


def test_t2_f07_03_engine_reset_is_idempotent(engine) -> None:
    """T2-F07-03: Calling engine.reset() twice consecutively succeeds cleanly."""
    from e2e_helpers import get_node_metrics

    engine.reset()
    engine.reset()
    m_intent = get_node_metrics(engine.get_metrics(), "intent_classifier")
    assert m_intent["trace_count"] == 0


def test_t2_f07_04_interleaved_threads_do_not_corrupt_store(engine, dataset_bundle) -> None:
    """T2-F07-04: Interleaving tickets across thread_A and thread_B preserves Store consistency."""
    from e2e_helpers import get_node_metrics

    for idx in range(10):
        engine.run_ticket(dataset_bundle.train_stream[idx * 2], thread_id="t2-f07-A")
        engine.run_ticket(dataset_bundle.train_stream[idx * 2 + 1], thread_id="t2-f07-B")

    m_global = get_node_metrics(engine.get_metrics(), "intent_classifier")
    assert m_global["trace_count"] == 20


def test_t2_f07_05_store_retains_all_traces_across_retrain_intervals(engine, dataset_bundle) -> None:
    """T2-F07-05: Streaming 100 tickets with retrain_interval_traces=10 retains all 100 traces."""
    from e2e_helpers import get_node_metrics

    engine.stream_batch(dataset_bundle.train_stream[:100], thread_id="t2-f07-05", auto_approve=True)
    m_intent = get_node_metrics(engine.get_metrics("t2-f07-05"), "intent_classifier")
    assert m_intent["trace_count"] == 100
