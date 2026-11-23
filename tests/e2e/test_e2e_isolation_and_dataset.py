"""E2E Tests — F16: Dataset Public Contract, Domain Schemas & Standalone Isolation.

Progressive layer: M1 (core.models, core.dataset, repo-wide hygiene).
Tests: 5 Tier 1 + 7 Tier 2 = 12 tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_t1_f16_01_generate_dataset_default_split_sizes_and_unique_ids() -> None:
    """T1-F16-01: Default dataset produces 200 train, 50 holdout, 50 drift with unique IDs."""
    from core.dataset import generate_dataset

    bundle = generate_dataset(seed=42)
    assert len(bundle.train_stream) == 200
    assert len(bundle.eval_holdout) == 50
    assert len(bundle.drift_wave) == 50
    assert len(bundle.all_tickets) == 300

    all_ids = [t.ticket_id for t in bundle.all_tickets]
    assert len(set(all_ids)) == 300

    train_ids = {t.ticket_id for t in bundle.train_stream}
    holdout_ids = {t.ticket_id for t in bundle.eval_holdout}
    drift_ids = {t.ticket_id for t in bundle.drift_wave}
    assert train_ids.isdisjoint(holdout_ids)
    assert train_ids.isdisjoint(drift_ids)
    assert holdout_ids.isdisjoint(drift_ids)


def test_t1_f16_02_generate_dataset_deterministic_seeding() -> None:
    """T1-F16-02: Identical seeds yield identical tickets; different seeds differ."""
    from core.dataset import generate_dataset

    b1 = generate_dataset(seed=123)
    b2 = generate_dataset(seed=123)
    b3 = generate_dataset(seed=456)

    assert [t.model_dump() for t in b1.all_tickets] == [t.model_dump() for t in b2.all_tickets]
    assert [t.ticket_text for t in b1.train_stream] != [t.ticket_text for t in b3.train_stream]


def test_t1_f16_03_zero_leaked_ground_truth_enum_labels_in_ticket_text() -> None:
    """T1-F16-03: No ground-truth IntentLabel or PolicyAction enum string leaks into ticket_text."""
    from core.dataset import generate_dataset
    from core.models import IntentLabel, PolicyAction

    bundle = generate_dataset(seed=42)
    forbidden_labels = [e.value for e in IntentLabel] + [e.value for e in PolicyAction]
    for ticket in bundle.all_tickets:
        text_upper = ticket.ticket_text.upper()
        for label in forbidden_labels:
            assert label not in text_upper, f"Leaked label {label} in {ticket.ticket_id}"


def test_t1_f16_04_dataset_covers_all_categories_and_drift_empty_box_scam() -> None:
    """T1-F16-04: train_stream covers all 5 categories & 4 normal intents; drift_wave has EMPTY_BOX_SCAM."""
    from core.dataset import generate_dataset
    from core.models import IntentLabel, ProductCategory

    bundle = generate_dataset(seed=42)
    train_cats = {t.category for t in bundle.train_stream}
    assert train_cats == set(ProductCategory)

    train_intents = {t.ground_truth_intent for t in bundle.train_stream}
    expected_normal = {
        IntentLabel.DAMAGED_ITEM,
        IntentLabel.LATE_DELIVERY,
        IntentLabel.WRONG_SIZE,
        IntentLabel.FRAUD_SUSPECT,
    }
    assert expected_normal.issubset(train_intents)
    assert IntentLabel.EMPTY_BOX_SCAM not in train_intents
    assert all(not t.is_drift for t in bundle.train_stream)

    drift_intents = {t.ground_truth_intent for t in bundle.drift_wave}
    assert IntentLabel.EMPTY_BOX_SCAM in drift_intents
    assert any(t.is_drift for t in bundle.drift_wave)


def test_t1_f16_05_strict_standalone_isolation_zero_forbidden_refs() -> None:
    """T1-F16-05: Zero forbidden project references across all files in c:\\Users\\HP\\zero-tax-agent."""
    project_root = Path(__file__).resolve().parents[2]
    forbidden_terms = [
        "cynbit-" + "ai-cognicare",
        "hivemind-" + "langgraph",
        "lex" + "si",
        "agent" + "tune",
        "audit" + "kit",
    ]
    scan_exts = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml"}
    matches: list[str] = []

    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", ".git", ".pytest_cache", "venv", ".venv"} for part in path.parts):
            continue
        if path.suffix.lower() not in scan_exts:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in forbidden_terms:
            if term in content:
                matches.append(f"{path}:{term}")

    assert matches == [], f"Forbidden standalone isolation references found: {matches}"


def test_t2_f16_01_generate_dataset_zero_counts() -> None:
    """T2-F16-01: generate_dataset with all counts=0 returns empty lists cleanly."""
    from core.dataset import generate_dataset

    bundle = generate_dataset(seed=42, main_count=0, holdout_count=0, drift_count=0)
    assert bundle.train_stream == []
    assert bundle.eval_holdout == []
    assert bundle.drift_wave == []
    assert bundle.all_tickets == []


def test_t2_f16_02_support_ticket_valid_boundary_values() -> None:
    """T2-F16-02: SupportTicket accepts exact domain boundaries (price=0, rate=0/1, days=0)."""
    from e2e_helpers import make_ticket

    t_low = make_ticket(item_price_inr=0.0, customer_return_rate=0.0, days_since_delivery=0)
    t_high = make_ticket(item_price_inr=15000.0, customer_return_rate=1.0, days_since_delivery=30)

    assert t_low.item_price_inr == 0.0
    assert t_low.customer_return_rate == 0.0
    assert t_low.days_since_delivery == 0
    assert t_high.customer_return_rate == 1.0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("item_price_inr", -1.0),
        ("customer_return_rate", -0.01),
        ("customer_return_rate", 1.01),
        ("days_since_delivery", -1),
    ],
)
def test_t2_f16_03_support_ticket_rejects_out_of_range_values(field: str, bad_value: float | int) -> None:
    """T2-F16-03: SupportTicket rejects negative price/days and return_rate outside [0, 1]."""
    from pydantic import ValidationError
    from e2e_helpers import make_ticket

    with pytest.raises(ValidationError):
        make_ticket(**{field: bad_value})


def test_t2_f16_04_support_ticket_rejects_invalid_category_and_split() -> None:
    """T2-F16-04: SupportTicket rejects invalid ProductCategory and split literals."""
    from pydantic import ValidationError
    from e2e_helpers import make_ticket

    with pytest.raises((ValidationError, ValueError)):
        make_ticket(category="GADGET")

    with pytest.raises(ValidationError):
        make_ticket(split="unknown_split")


def test_t2_f16_05_e2e_suite_hygiene_and_lazy_import_self_audit() -> None:
    """T2-F16-05: tests/e2e/*.py has zero skip guards and zero top-level core/api imports."""
    e2e_dir = Path(__file__).resolve().parent
    forbidden_api_calls = ["importor" + "skip", "pytest." + "skip", "pytest.mark." + "skip"]
    forbidden_top_modules = {"core", "api", "benchmark", "pydantic", "fastapi"}

    violations: list[str] = []
    for py_file in sorted(e2e_dir.glob("*.py")):
        src = py_file.read_text(encoding="utf-8")
        for pat in forbidden_api_calls:
            if pat in src:
                violations.append(f"{py_file.name}: forbidden pattern '{pat}'")

        tree = ast.parse(src, filename=str(py_file))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in forbidden_top_modules:
                        violations.append(f"{py_file.name}: top-level import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in forbidden_top_modules:
                    violations.append(f"{py_file.name}: top-level from-import '{node.module}'")

    assert violations == [], f"E2E hygiene violations: {violations}"


def test_t2_f16_06_usd_to_inr_constant_contract() -> None:
    """T2-F16-06: USD_TO_INR equals 84.50 as a float."""
    from core.models import USD_TO_INR

    assert isinstance(USD_TO_INR, float)
    assert USD_TO_INR == 84.50


def test_t2_f16_07_node_prediction_and_metrics_schema_boundaries() -> None:
    """T2-F16-07: NodePrediction enforces confidence in [0, 1] and non-negative cost/latency."""
    from core.models import ExecutionMode, NodePrediction
    from pydantic import ValidationError

    valid = NodePrediction(
        node_name="intent_classifier",
        prediction="DAMAGED_ITEM",
        confidence=0.95,
        probabilities={"DAMAGED_ITEM": 0.95, "LATE_DELIVERY": 0.05},
        execution_mode=ExecutionMode.LLM_TEACHER,
        latency_ms=120.0,
        input_tokens=40,
        output_tokens=8,
        cost_usd=0.001,
        cost_inr=0.0845,
        teacher_model="simulated-teacher",
    )
    assert valid.confidence == 0.95

    with pytest.raises(ValidationError):
        NodePrediction(
            node_name="intent_classifier",
            prediction="DAMAGED_ITEM",
            confidence=1.2,
            probabilities={"DAMAGED_ITEM": 1.0},
            execution_mode=ExecutionMode.LLM_TEACHER,
            latency_ms=10.0,
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.0,
            cost_inr=0.0,
        )

    with pytest.raises(ValidationError):
        NodePrediction(
            node_name="intent_classifier",
            prediction="DAMAGED_ITEM",
            confidence=0.5,
            probabilities={"DAMAGED_ITEM": 1.0},
            execution_mode=ExecutionMode.LLM_TEACHER,
            latency_ms=-5.0,
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.0,
            cost_inr=0.0,
        )
