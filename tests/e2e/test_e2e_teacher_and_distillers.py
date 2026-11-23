"""E2E Tests — F02, F03, F04: Offline Teacher & Online Sklearn Distillers.

Progressive layer: M1 (core.teacher, core.distiller, core.models, core.dataset).
Tests: 15 Tier 1 + 17 Tier 2 = 32 tests.
"""

from __future__ import annotations

import random

import pytest


# ===========================================================================
# F02: Offline Simulated Teacher + INR/USD Cost & Latency Accounting (5 T1 + 6 T2)
# ===========================================================================


def test_t1_f02_01_simulated_teacher_intent_classification_and_cost_accounting(dataset_bundle) -> None:
    """T1-F02-01: TeacherMode.SIMULATED classifies intent with valid INR/USD cost & latency."""
    from core.models import USD_TO_INR, ExecutionMode, IntentLabel, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    ticket = dataset_bundle.train_stream[0]
    pred = teacher.classify_intent(ticket)

    assert pred.execution_mode == ExecutionMode.LLM_TEACHER
    assert pred.prediction in {e.value for e in IntentLabel}
    assert pred.input_tokens > 0
    assert pred.output_tokens > 0
    assert pred.cost_usd > 0.0
    assert pred.cost_inr == pytest.approx(pred.cost_usd * USD_TO_INR, rel=1e-3)
    assert pred.latency_ms > 0.0


def test_t1_f02_02_teacher_mode_auto_falls_back_to_simulated_offline(dataset_bundle) -> None:
    """T1-F02-02: TeacherMode.AUTO falls back to Simulated Teacher when no API key is set."""
    from core.models import ExecutionMode, IntentLabel, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.AUTO, sleep_on_simulated=False, seed=42)
    pred = teacher.classify_intent(dataset_bundle.train_stream[1])

    assert pred.execution_mode == ExecutionMode.LLM_TEACHER
    assert pred.prediction in {e.value for e in IntentLabel}
    assert pred.cost_inr > 0.0
    assert pred.teacher_model is not None and len(pred.teacher_model) > 0


def test_t1_f02_03_route_policy_produces_valid_policy_and_probability_simplex(dataset_bundle) -> None:
    """T1-F02-03: route_policy returns valid PolicyAction, normalized probs, and INR/USD ratio."""
    from core.models import USD_TO_INR, IntentLabel, PolicyAction, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    valid_policies = {e.value for e in PolicyAction}

    for ticket in dataset_bundle.train_stream[:4]:
        pred = teacher.route_policy(ticket, IntentLabel.DAMAGED_ITEM.value)
        assert pred.prediction in valid_policies
        assert 0.0 <= pred.confidence <= 1.0
        assert all(0.0 <= p <= 1.0 for p in pred.probabilities.values())
        assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
        assert pred.cost_inr == pytest.approx(pred.cost_usd * USD_TO_INR, rel=1e-3)


def test_t1_f02_04_write_reply_accounts_tokens_cost_latency_and_text(dataset_bundle) -> None:
    """T1-F02-04: write_reply outputs non-empty customer reply and accounts tokens/cost/latency."""
    from core.models import USD_TO_INR, ExecutionMode, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    ticket = dataset_bundle.train_stream[0]
    pred = teacher.write_reply(ticket, "DAMAGED_ITEM", "INSTANT_REFUND")

    reply = getattr(pred, "reply_text", None) or pred.prediction
    assert isinstance(reply, str) and len(reply.strip()) >= 15
    assert pred.execution_mode == ExecutionMode.LLM_TEACHER
    assert pred.output_tokens > 10
    assert pred.cost_usd > 0.0
    assert pred.cost_inr == pytest.approx(pred.cost_usd * USD_TO_INR, rel=1e-3)
    assert pred.latency_ms > 0.0


def test_t1_f02_05_teacher_never_peeks_at_ground_truth_fields(dataset_bundle) -> None:
    """T1-F02-05: Teacher outputs are identical whether ground_truth_* is set or None."""
    from core.models import TeacherMode
    from core.teacher import LLMTeacher

    t_orig = dataset_bundle.train_stream[5]
    t_masked = t_orig.model_copy(update={"ground_truth_intent": None, "ground_truth_policy": None})

    teacher_a = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    teacher_b = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)

    p_orig_intent = teacher_a.classify_intent(t_orig)
    p_mask_intent = teacher_b.classify_intent(t_masked)
    assert p_orig_intent.prediction == p_mask_intent.prediction
    assert p_orig_intent.probabilities == p_mask_intent.probabilities
    assert p_orig_intent.cost_usd == p_mask_intent.cost_usd

    p_orig_pol = teacher_a.route_policy(t_orig, p_orig_intent.prediction)
    p_mask_pol = teacher_b.route_policy(t_masked, p_mask_intent.prediction)
    assert p_orig_pol.prediction == p_mask_pol.prediction
    assert p_orig_pol.probabilities == p_mask_pol.probabilities
    assert p_orig_pol.cost_usd == p_mask_pol.cost_usd


def test_t2_f02_01_teacher_handles_empty_and_whitespace_ticket_text() -> None:
    """T2-F02-01: Empty and whitespace-only ticket_text handled cleanly without exception."""
    from core.models import IntentLabel, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import make_ticket

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    valid_intents = {e.value for e in IntentLabel}

    for text in ("", "   "):
        pred = teacher.classify_intent(make_ticket(ticket_text=text))
        assert pred.prediction in valid_intents
        assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_t2_f02_02_teacher_token_and_cost_scale_with_text_length() -> None:
    """T2-F02-02: 5000+ char text incurs strictly higher input_tokens and cost_usd than 'ok'."""
    from core.models import USD_TO_INR, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import make_ticket

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    short_pred = teacher.classify_intent(make_ticket(ticket_text="ok"))
    long_pred = teacher.classify_intent(make_ticket(ticket_text="broken sneaker sole " * 300))

    assert long_pred.input_tokens > short_pred.input_tokens
    assert long_pred.cost_usd > short_pred.cost_usd
    assert long_pred.cost_inr == pytest.approx(long_pred.cost_usd * USD_TO_INR, rel=1e-3)


def test_t2_f02_03_teacher_handles_unicode_emoji_and_hinglish_text() -> None:
    """T2-F02-03: Unicode, emojis, and Hinglish text process cleanly on Windows without encoding errors."""
    from core.models import IntentLabel, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import make_ticket

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    ticket = make_ticket(ticket_text="😡 mera parcel toot gaya aur ₹1500 ka nuksan hua!")
    intent_pred = teacher.classify_intent(ticket)
    reply_pred = teacher.write_reply(ticket, intent_pred.prediction, "REPLACEMENT_PICKUP")

    assert intent_pred.prediction in {e.value for e in IntentLabel}
    reply = getattr(reply_pred, "reply_text", None) or reply_pred.prediction
    assert isinstance(reply, str) and len(reply) > 0


def test_t2_f02_04_teacher_explicit_cloud_mode_without_api_key_fails_or_falls_back(dataset_bundle) -> None:
    """T2-F02-04: Explicit GEMINI/OPENAI mode without env key either raises explicit error or falls back cleanly."""
    from core.models import IntentLabel, TeacherMode
    from core.teacher import LLMTeacher

    ticket = dataset_bundle.train_stream[0]
    for mode in (TeacherMode.GEMINI, TeacherMode.OPENAI):
        try:
            teacher = LLMTeacher(mode=mode, sleep_on_simulated=False, seed=42)
            pred = teacher.classify_intent(ticket)
            assert pred.prediction in {e.value for e in IntentLabel}
        except (RuntimeError, ValueError, OSError) as exc:
            assert len(str(exc)) > 0


def test_t2_f02_05_teacher_simulate_latency_false_boundary(dataset_bundle) -> None:
    """T2-F02-05: simulate_latency=False runs cleanly with valid intent, cost_usd > 0, and latency_ms >= 0."""
    from core.models import USD_TO_INR, IntentLabel, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(
        mode=TeacherMode.SIMULATED,
        simulate_latency=False,
        sleep_on_simulated=False,
        seed=42,
    )
    pred = teacher.classify_intent(dataset_bundle.train_stream[0])
    assert pred.prediction in {e.value for e in IntentLabel}
    assert pred.cost_usd > 0.0
    assert pred.cost_inr == pytest.approx(pred.cost_usd * USD_TO_INR, rel=1e-3)
    assert pred.latency_ms >= 0.0


def test_t2_f02_06_teacher_invalid_mode_rejected() -> None:
    """T2-F02-06: Invalid TeacherMode string raises ValueError or ValidationError."""
    from core.teacher import LLMTeacher
    from pydantic import ValidationError

    with pytest.raises((ValueError, ValidationError)):
        LLMTeacher(mode="INVALID_MODE")  # type: ignore[arg-type]


# ===========================================================================
# F03: Node 1 intent_classifier Online Distillation (5 T1 + 5 T2)
# ===========================================================================


def test_t1_f03_01_intent_distiller_accumulates_teacher_traces(dataset_bundle) -> None:
    """T1-F03-01: Feeding 50 Teacher-labeled traces updates trace_count and stays TEACHER_ACTIVE."""
    from core.distiller import NodeDistiller
    from core.models import NodeStatus, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)

    for ticket in dataset_bundle.train_stream[:50]:
        pred = teacher.classify_intent(ticket)
        distiller.add_trace(ticket.ticket_text, pred.prediction, ticket.ticket_id)

    metrics = distiller.get_metrics()
    assert metrics.trace_count == 50
    assert metrics.traces_since_last_train == 50
    assert metrics.status == NodeStatus.TEACHER_ACTIVE
    assert metrics.is_promoted is False


def test_t1_f03_02_intent_distiller_evaluate_and_fit_returns_promotion_candidate(dataset_bundle) -> None:
    """T1-F03-02: Online 80-trace evaluation passes 5-fold CV >= 0.95 and ECE <= 0.10."""
    from core.distiller import NodeDistiller
    from core.models import TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)

    for ticket in dataset_bundle.train_stream[:80]:
        pred = teacher.classify_intent(ticket)
        distiller.add_trace(ticket.ticket_text, pred.prediction, ticket.ticket_id)

    candidate = distiller.evaluate_and_fit()
    assert candidate is not None
    assert candidate.node_name == "intent_classifier"
    assert candidate.sample_count == 80
    assert candidate.cv_accuracy >= 0.95
    assert candidate.ece <= 0.10
    assert candidate.model_version >= 1


def test_t1_f03_03_intent_distiller_holdout_generalization(dataset_bundle) -> None:
    """T1-F03-03: Promoted intent_classifier achieves >= 0.90 accuracy on unseen eval_holdout."""
    from core.distiller import NodeDistiller
    from core.models import TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)

    for ticket in dataset_bundle.train_stream[:120]:
        pred = teacher.classify_intent(ticket)
        distiller.add_trace(ticket.ticket_text, pred.prediction, ticket.ticket_id)

    cand = distiller.evaluate_and_fit()
    assert cand is not None
    distiller.approve_promotion()

    correct = 0
    for ticket in dataset_bundle.eval_holdout:
        label, conf, probs = distiller.predict(ticket.ticket_text)
        assert 0.0 <= conf <= 1.0
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
        if label == ticket.ground_truth_intent.value:
            correct += 1

    assert (correct / len(dataset_bundle.eval_holdout)) >= 0.90


def test_t1_f03_04_intent_distiller_generalizes_to_novel_paraphrased_phrasing(dataset_bundle) -> None:
    """T1-F03-04: Promoted intent_classifier classifies unseen paraphrased damaged-shoe phrasing."""
    from core.distiller import NodeDistiller
    from core.models import IntentLabel, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)

    for ticket in dataset_bundle.train_stream[:100]:
        pred = teacher.classify_intent(ticket)
        distiller.add_trace(ticket.ticket_text, pred.prediction, ticket.ticket_id)

    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    label, conf, probs = distiller.predict(
        "The rubber sole cracked and peeled away from my running shoes on day one"
    )
    assert label == IntentLabel.DAMAGED_ITEM.value
    assert 0.0 <= conf <= 1.0
    assert len(probs) >= 4


def test_t1_f03_05_intent_distiller_retrains_and_learns_empty_box_scam_on_drift(dataset_bundle) -> None:
    """T1-F03-05: Adding drift_wave traces advances model_version and learns EMPTY_BOX_SCAM."""
    from core.distiller import NodeDistiller
    from core.models import IntentLabel, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)

    for ticket in dataset_bundle.train_stream[:100]:
        pred = teacher.classify_intent(ticket)
        distiller.add_trace(ticket.ticket_text, pred.prediction, ticket.ticket_id)

    c1 = distiller.evaluate_and_fit()
    assert c1 is not None
    distiller.approve_promotion()
    v1 = distiller.get_metrics().model_version

    for ticket in dataset_bundle.drift_wave:
        pred = teacher.classify_intent(ticket)
        distiller.add_trace(ticket.ticket_text, pred.prediction, ticket.ticket_id, is_drift=True)

    c2 = distiller.evaluate_and_fit()
    assert c2 is not None
    assert c2.model_version > v1

    scam_tickets = [
        t for t in dataset_bundle.drift_wave if t.ground_truth_intent == IntentLabel.EMPTY_BOX_SCAM
    ]
    scam_hits = sum(
        1 for t in scam_tickets[:10] if distiller.predict(t.ticket_text)[0] == IntentLabel.EMPTY_BOX_SCAM.value
    )
    assert scam_hits >= 8


def test_t2_f03_01_intent_distiller_blocks_promotion_below_min_samples(dataset_bundle) -> None:
    """T2-F03-01: 44 traces (< min_samples_for_promotion=45) returns None and stays TEACHER_ACTIVE."""
    from core.distiller import NodeDistiller
    from core.models import NodeStatus, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)

    for ticket in dataset_bundle.train_stream[:44]:
        pred = teacher.classify_intent(ticket)
        distiller.add_trace(ticket.ticket_text, pred.prediction, ticket.ticket_id)

    assert distiller.evaluate_and_fit() is None
    metrics = distiller.get_metrics()
    assert metrics.is_promoted is False
    assert metrics.status == NodeStatus.TEACHER_ACTIVE


def test_t2_f03_02_intent_distiller_blocks_when_minority_class_below_min_per_class(dataset_bundle) -> None:
    """T2-F03-02: 45 traces with < 5 samples in one class returns None (stratified CV guard)."""
    from core.distiller import NodeDistiller
    from core.models import IntentLabel

    distiller = NodeDistiller("intent_classifier", seed=42)
    by_class: dict[str, list] = {
        IntentLabel.DAMAGED_ITEM.value: [],
        IntentLabel.LATE_DELIVERY.value: [],
        IntentLabel.WRONG_SIZE.value: [],
        IntentLabel.FRAUD_SUSPECT.value: [],
    }
    for t in dataset_bundle.train_stream:
        lbl = t.ground_truth_intent.value
        if lbl in by_class:
            by_class[lbl].append(t)

    subset = (
        by_class[IntentLabel.DAMAGED_ITEM.value][:15]
        + by_class[IntentLabel.LATE_DELIVERY.value][:15]
        + by_class[IntentLabel.WRONG_SIZE.value][:11]
        + by_class[IntentLabel.FRAUD_SUSPECT.value][:4]
    )
    assert len(subset) == 45
    for t in subset:
        distiller.add_trace(t.ticket_text, t.ground_truth_intent.value, t.ticket_id)

    assert distiller.evaluate_and_fit() is None


def test_t2_f03_03_intent_distiller_cold_predict_raises_explicit_error() -> None:
    """T2-F03-03: Calling predict() before model is fitted raises explicit RuntimeError/ValueError."""
    from core.distiller import NodeDistiller

    distiller = NodeDistiller("intent_classifier", seed=42)
    with pytest.raises((RuntimeError, ValueError)):
        distiller.predict("My sneaker sole came damaged")


def test_t2_f03_04_intent_distiller_handles_empty_whitespace_and_long_text(dataset_bundle) -> None:
    """T2-F03-04: Fitted intent_classifier handles '', '   ', and 6000-char inputs cleanly."""
    from core.distiller import NodeDistiller
    from core.models import TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)
    for ticket in dataset_bundle.train_stream[:80]:
        distiller.add_trace(ticket.ticket_text, teacher.classify_intent(ticket).prediction, ticket.ticket_id)

    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    for probe in ("", "   ", "delivery delayed " * 400):
        label, conf, probs = distiller.predict(probe)
        assert isinstance(label, str) and len(label) > 0
        assert 0.0 <= conf <= 1.0
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_t2_f03_05_intent_distiller_blocks_promotion_on_noisy_corrupted_labels(dataset_bundle) -> None:
    """T2-F03-05: 40% corrupted labels fail min_cv_accuracy=0.95 gate and block promotion."""
    from core.distiller import NodeDistiller
    from core.models import IntentLabel, NodeStatus

    rng = random.Random(99)
    distiller = NodeDistiller("intent_classifier", seed=42)
    normal_labels = [
        IntentLabel.DAMAGED_ITEM.value,
        IntentLabel.LATE_DELIVERY.value,
        IntentLabel.WRONG_SIZE.value,
        IntentLabel.FRAUD_SUSPECT.value,
    ]

    for idx, ticket in enumerate(dataset_bundle.train_stream[:80]):
        true_lbl = ticket.ground_truth_intent.value
        lbl = rng.choice(normal_labels) if idx % 5 < 2 else true_lbl
        distiller.add_trace(ticket.ticket_text, lbl, ticket.ticket_id)

    cand = distiller.evaluate_and_fit()
    assert cand is None
    assert distiller.get_metrics().status == NodeStatus.TEACHER_ACTIVE


# ===========================================================================
# F04: Node 2 policy_router Online Distillation (5 T1 + 6 T2)
# ===========================================================================


def _policy_features(ticket, intent_label: str) -> dict:
    return {
        "category": ticket.category.value,
        "item_price_inr": float(ticket.item_price_inr),
        "customer_return_rate": float(ticket.customer_return_rate),
        "days_since_delivery": int(ticket.days_since_delivery),
        "ticket_text": ticket.ticket_text,
        "intent": intent_label,
    }


def test_t1_f04_01_policy_distiller_ingests_structured_and_text_traces(dataset_bundle) -> None:
    """T1-F04-01: policy_router ingests mixed structured + text features from Teacher outputs."""
    from core.distiller import NodeDistiller
    from core.models import TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)

    for ticket in dataset_bundle.train_stream[:60]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)

    assert distiller.get_metrics().trace_count == 60


def test_t1_f04_02_policy_distiller_evaluate_and_fit_passes_cv_and_ece(dataset_bundle) -> None:
    """T1-F04-02: policy_router 5-fold CV >= 0.95, ECE <= 0.10, and covers all 3 PolicyActions."""
    from core.distiller import NodeDistiller
    from core.models import PolicyAction, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)

    for ticket in dataset_bundle.train_stream[:120]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)

    cand = distiller.evaluate_and_fit()
    assert cand is not None
    assert cand.node_name == "policy_router"
    assert cand.cv_accuracy >= 0.95
    assert cand.ece <= 0.10
    assert set(cand.class_distribution.keys()) == {e.value for e in PolicyAction}


def test_t1_f04_03_policy_distiller_holdout_accuracy_after_promotion(dataset_bundle) -> None:
    """T1-F04-03: Promoted policy_router achieves >= 0.88 accuracy on eval_holdout."""
    from core.distiller import NodeDistiller
    from core.models import PolicyAction, TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)

    for ticket in dataset_bundle.train_stream[:140]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)

    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    valid_policies = {e.value for e in PolicyAction}
    correct = 0
    for ticket in dataset_bundle.eval_holdout:
        intent_lbl = teacher.classify_intent(ticket).prediction
        label, conf, probs = distiller.predict(_policy_features(ticket, intent_lbl))
        assert label in valid_policies
        assert 0.0 <= conf <= 1.0
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
        if label == ticket.ground_truth_policy.value:
            correct += 1

    assert (correct / len(dataset_bundle.eval_holdout)) >= 0.88


def test_t1_f04_04_policy_distiller_sensitive_to_price_and_return_rate(dataset_bundle) -> None:
    """T1-F04-04: Promoted policy_router differentiates low-price/low-return vs high-price/high-return."""
    from core.distiller import NodeDistiller
    from core.models import IntentLabel, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import make_ticket

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)

    for ticket in dataset_bundle.train_stream[:140]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)

    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    low_risk = make_ticket(
        ticket_id="LOW-RISK",
        category="FOOTWEAR",
        item_price_inr=350.0,
        customer_return_rate=0.05,
        days_since_delivery=2,
        ticket_text="Sneaker sole is unglued on arrival",
    )
    high_risk = make_ticket(
        ticket_id="HIGH-RISK",
        category="ELECTRONICS",
        item_price_inr=12000.0,
        customer_return_rate=0.65,
        days_since_delivery=18,
        ticket_text="Sneaker sole is unglued on arrival",
    )

    t_low = teacher.route_policy(low_risk, IntentLabel.DAMAGED_ITEM.value).prediction
    t_high = teacher.route_policy(high_risk, IntentLabel.FRAUD_SUSPECT.value).prediction
    s_low, _, _ = distiller.predict(_policy_features(low_risk, IntentLabel.DAMAGED_ITEM.value))
    s_high, _, _ = distiller.predict(_policy_features(high_risk, IntentLabel.FRAUD_SUSPECT.value))

    assert s_low == t_low
    assert s_high == t_high


def test_t1_f04_05_policy_distiller_handles_all_five_product_categories(dataset_bundle) -> None:
    """T1-F04-05: Promoted policy_router encodes all 5 ProductCategory values cleanly."""
    from core.distiller import NodeDistiller
    from core.models import PolicyAction, ProductCategory, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import make_ticket

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)

    for ticket in dataset_bundle.train_stream[:120]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)

    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    valid_policies = {e.value for e in PolicyAction}
    for cat in ProductCategory:
        t = make_ticket(category=cat.value)
        label, conf, probs = distiller.predict(_policy_features(t, "DAMAGED_ITEM"))
        assert label in valid_policies
        assert 0.0 <= conf <= 1.0
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("price", [0.0, 299.0, 1199.0, 1200.0, 4999.0, 5000.0, 15000.0])
def test_t2_f04_01_policy_price_boundaries(dataset_bundle, price: float) -> None:
    """T2-F04-01: Exact INR price boundaries (0..15000) produce valid probabilities & extreme boundary sensitivity."""
    from core.distiller import NodeDistiller
    from core.models import IntentLabel, PolicyAction, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import make_ticket

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)
    for ticket in dataset_bundle.train_stream[:100]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)
    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    probe = make_ticket(
        category="FOOTWEAR",
        item_price_inr=price,
        customer_return_rate=0.10,
        days_since_delivery=2,
    )
    t_pred = teacher.route_policy(probe, "DAMAGED_ITEM")
    s_lbl, s_conf, s_probs = distiller.predict(_policy_features(probe, "DAMAGED_ITEM"))

    valid = {e.value for e in PolicyAction}
    assert t_pred.prediction in valid
    assert s_lbl in valid
    assert 0.0 <= s_conf <= 1.0
    assert sum(s_probs.values()) == pytest.approx(1.0, abs=1e-4)
    if price in (0.0, 15000.0):
        assert s_lbl == t_pred.prediction

    low_risk = make_ticket(
        category="FOOTWEAR",
        item_price_inr=0.0,
        customer_return_rate=0.0,
        days_since_delivery=0,
        ticket_text="Sneaker sole is unglued on arrival",
    )
    high_risk = make_ticket(
        category="ELECTRONICS",
        item_price_inr=15000.0,
        customer_return_rate=1.0,
        days_since_delivery=30,
        ticket_text="High value refund claim with repeated returns",
    )
    s_low, _, _ = distiller.predict(_policy_features(low_risk, IntentLabel.DAMAGED_ITEM.value))
    s_high, _, _ = distiller.predict(_policy_features(high_risk, IntentLabel.FRAUD_SUSPECT.value))
    assert s_low == teacher.route_policy(low_risk, IntentLabel.DAMAGED_ITEM.value).prediction
    assert s_high == teacher.route_policy(high_risk, IntentLabel.FRAUD_SUSPECT.value).prediction
    assert s_low != s_high


@pytest.mark.parametrize("return_rate", [0.0, 0.25, 0.40, 1.0])
def test_t2_f04_02_policy_return_rate_boundaries(dataset_bundle, return_rate: float) -> None:
    """T2-F04-02: Exact customer_return_rate boundaries (0.0, 0.25, 0.40, 1.0) are deterministic & sensitive."""
    from core.distiller import NodeDistiller
    from core.models import IntentLabel, PolicyAction, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import make_ticket

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)
    for ticket in dataset_bundle.train_stream[:100]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)
    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    probe = make_ticket(item_price_inr=1500.0, customer_return_rate=return_rate, days_since_delivery=3)
    t_pred = teacher.route_policy(probe, "DAMAGED_ITEM")
    lbl1, conf1, probs1 = distiller.predict(_policy_features(probe, "DAMAGED_ITEM"))
    lbl2, conf2, probs2 = distiller.predict(_policy_features(probe, "DAMAGED_ITEM"))

    assert lbl1 in {e.value for e in PolicyAction}
    assert lbl1 == lbl2 and conf1 == conf2 and probs1 == probs2
    assert sum(probs1.values()) == pytest.approx(1.0, abs=1e-4)
    if return_rate in (0.0, 1.0):
        assert lbl1 == t_pred.prediction

    low_risk = make_ticket(item_price_inr=0.0, customer_return_rate=0.0, days_since_delivery=0)
    high_risk = make_ticket(category="ELECTRONICS", item_price_inr=15000.0, customer_return_rate=1.0, days_since_delivery=30)
    s_low, _, _ = distiller.predict(_policy_features(low_risk, IntentLabel.DAMAGED_ITEM.value))
    s_high, _, _ = distiller.predict(_policy_features(high_risk, IntentLabel.FRAUD_SUSPECT.value))
    assert s_low == teacher.route_policy(low_risk, IntentLabel.DAMAGED_ITEM.value).prediction
    assert s_high == teacher.route_policy(high_risk, IntentLabel.FRAUD_SUSPECT.value).prediction
    assert s_low != s_high


@pytest.mark.parametrize("days", [0, 14, 15, 30])
def test_t2_f04_03_policy_days_since_delivery_boundaries(dataset_bundle, days: int) -> None:
    """T2-F04-03: Exact days_since_delivery boundaries (0, 14, 15, 30) match Teacher extremes & boundary sensitivity."""
    from core.distiller import NodeDistiller
    from core.models import IntentLabel, PolicyAction, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import make_ticket

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)
    for ticket in dataset_bundle.train_stream[:100]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)
    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    probe = make_ticket(item_price_inr=1000.0, customer_return_rate=0.10, days_since_delivery=days)
    t_pred = teacher.route_policy(probe, "LATE_DELIVERY")
    lbl, conf, probs = distiller.predict(_policy_features(probe, "LATE_DELIVERY"))
    assert lbl in {e.value for e in PolicyAction}
    assert 0.0 <= conf <= 1.0
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    if days in (0, 30):
        assert lbl == t_pred.prediction

    low_risk = make_ticket(item_price_inr=0.0, customer_return_rate=0.0, days_since_delivery=0)
    high_risk = make_ticket(category="ELECTRONICS", item_price_inr=15000.0, customer_return_rate=1.0, days_since_delivery=30)
    s_low, _, _ = distiller.predict(_policy_features(low_risk, IntentLabel.DAMAGED_ITEM.value))
    s_high, _, _ = distiller.predict(_policy_features(high_risk, IntentLabel.FRAUD_SUSPECT.value))
    assert s_low == teacher.route_policy(low_risk, IntentLabel.DAMAGED_ITEM.value).prediction
    assert s_high == teacher.route_policy(high_risk, IntentLabel.FRAUD_SUSPECT.value).prediction
    assert s_low != s_high


def test_t2_f04_04_policy_blocks_single_class_degenerate_traces(dataset_bundle) -> None:
    """T2-F04-04: 50 traces with only 1 PolicyAction label returns None (cannot promote)."""
    from core.distiller import NodeDistiller
    from core.models import PolicyAction

    distiller = NodeDistiller("policy_router", seed=42)
    for ticket in dataset_bundle.train_stream[:50]:
        distiller.add_trace(
            _policy_features(ticket, "DAMAGED_ITEM"),
            PolicyAction.INSTANT_REFUND.value,
            ticket.ticket_id,
        )

    assert distiller.evaluate_and_fit() is None


def test_t2_f04_05_policy_missing_required_feature_key_raises_error(dataset_bundle) -> None:
    """T2-F04-05: Missing required feature keys in predict() raises KeyError/ValueError explicitly."""
    from core.distiller import NodeDistiller
    from core.models import TeacherMode
    from core.teacher import LLMTeacher

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)
    for ticket in dataset_bundle.train_stream[:100]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)
    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    with pytest.raises((KeyError, ValueError, TypeError)):
        distiller.predict({"category": "FOOTWEAR"})


def test_t2_f04_06_policy_extreme_high_price_and_days_without_nan(dataset_bundle) -> None:
    """T2-F04-06: Extreme price=1_000_000 INR & days=365 yields finite probabilities without NaN."""
    import math

    from core.distiller import NodeDistiller
    from core.models import PolicyAction, TeacherMode
    from core.teacher import LLMTeacher
    from e2e_helpers import make_ticket

    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("policy_router", seed=42)
    for ticket in dataset_bundle.train_stream[:100]:
        intent_pred = teacher.classify_intent(ticket)
        pol_pred = teacher.route_policy(ticket, intent_pred.prediction)
        distiller.add_trace(_policy_features(ticket, intent_pred.prediction), pol_pred.prediction, ticket.ticket_id)
    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    extreme = make_ticket(
        category="ELECTRONICS",
        item_price_inr=1_000_000.0,
        customer_return_rate=1.0,
        days_since_delivery=365,
    )
    t_pred = teacher.route_policy(extreme, "FRAUD_SUSPECT")
    s_lbl, s_conf, s_probs = distiller.predict(_policy_features(extreme, "FRAUD_SUSPECT"))

    assert t_pred.prediction in {e.value for e in PolicyAction}
    assert s_lbl in {e.value for e in PolicyAction}
    assert math.isfinite(s_conf)
    assert all(math.isfinite(v) for v in s_probs.values())
    assert sum(s_probs.values()) == pytest.approx(1.0, abs=1e-4)
