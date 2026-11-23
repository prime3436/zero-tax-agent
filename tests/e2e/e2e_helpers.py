"""Opaque-box E2E helper functions for ZERO-TAX.

Strict hygiene rules:
- Module top level imports ONLY Python stdlib.
- All core.*, api, benchmark, pydantic, and fastapi imports are lazy (inside functions).
- Zero mocks, zero dynamic skip guards, zero silent fallbacks masking missing keys.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def to_dict(obj: Any) -> dict[str, Any]:
    """Convert a Pydantic v2 model or mapping into a plain dict."""
    if obj is None:
        raise ValueError("Cannot convert None to dict")
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return dict(obj)


def enum_str(val: Any) -> str:
    """Normalize an Enum member or string to its string value."""
    if hasattr(val, "value"):
        return str(val.value)
    return str(val)


def make_ticket(
    *,
    ticket_id: str = "TKT-E2E-001",
    customer_name: str = "Aarav Sharma",
    city: str = "Bengaluru",
    order_id: str = "ORD-99123",
    category: str = "FOOTWEAR",
    item_price_inr: float = 2499.0,
    customer_return_rate: float = 0.08,
    days_since_delivery: int = 3,
    ticket_text: str = "My running sneaker sole came glued crooked and detached on day one",
    split: str = "train_stream",
    is_drift: bool = False,
    ground_truth_intent: str | None = None,
    ground_truth_policy: str | None = None,
) -> Any:
    """Construct a valid SupportTicket lazily."""
    from core.models import IntentLabel, PolicyAction, ProductCategory, SupportTicket

    return SupportTicket(
        ticket_id=ticket_id,
        customer_name=customer_name,
        city=city,
        order_id=order_id,
        category=ProductCategory(category),
        item_price_inr=item_price_inr,
        customer_return_rate=customer_return_rate,
        days_since_delivery=days_since_delivery,
        ticket_text=ticket_text,
        split=split,
        is_drift=is_drift,
        ground_truth_intent=IntentLabel(ground_truth_intent) if ground_truth_intent else None,
        ground_truth_policy=PolicyAction(ground_truth_policy) if ground_truth_policy else None,
    )


def get_telemetry(res: Any) -> dict[str, Any]:
    """Extract TicketTelemetry dictionary from run_ticket / API response strictly."""
    d = to_dict(res)
    if "telemetry" in d and d["telemetry"] is not None:
        tel = to_dict(d["telemetry"])
    else:
        tel = d
    if not any(k in tel for k in ("intent_result", "policy_result", "reply_result")):
        raise KeyError(
            f"Telemetry missing intent_result/policy_result/reply_result in keys={list(tel.keys())}"
        )
    return tel


def get_node_result(res: Any, node_name: str) -> dict[str, Any]:
    """Extract NodePrediction dict for intent_classifier, policy_router, or reply_writer."""
    tel = get_telemetry(res)
    key_map = {
        "intent_classifier": "intent_result",
        "policy_router": "policy_result",
        "reply_writer": "reply_result",
    }
    key = key_map.get(node_name, node_name)
    if key in tel and tel[key] is not None:
        return to_dict(tel[key])
    d = to_dict(res)
    if key in d and d[key] is not None:
        return to_dict(d[key])
    raise KeyError(f"Node result '{key}' not found in keys={list(tel.keys())}")


def get_reply_text(res: Any) -> str:
    """Extract customer-facing reply text from run_ticket / API response."""
    tel = get_telemetry(res)
    if tel.get("reply_text"):
        return str(tel["reply_text"])
    reply_res = tel.get("reply_result")
    if reply_res is not None:
        rd = to_dict(reply_res)
        if rd.get("prediction"):
            return str(rd["prediction"])
    d = to_dict(res)
    if d.get("reply_text"):
        return str(d["reply_text"])
    return ""


def get_interrupt_payload(res: Any) -> dict[str, Any] | None:
    """Extract promotion interrupt payload dictionary if present."""
    d = to_dict(res)
    if d.get("interrupt_payload") is not None:
        return to_dict(d["interrupt_payload"])
    for k in ("interrupts", "__interrupt__"):
        items = d.get(k)
        if items:
            first = items[0]
            if hasattr(first, "value"):
                return to_dict(first.value)
            return to_dict(first)
    return None


def get_node_metrics(metrics_res: Any, node_name: str) -> dict[str, Any]:
    """Extract per-node distillation metrics dict from engine.get_metrics() or /metrics strictly."""
    d = to_dict(metrics_res)
    for container_key in ("nodes", "per_node_metrics", "node_metrics", "distillers"):
        container = d.get(container_key)
        if isinstance(container, dict) and node_name in container:
            return to_dict(container[node_name])
    if node_name in d:
        return to_dict(d[node_name])
    raise KeyError(f"Node metrics for '{node_name}' not found in keys={list(d.keys())}")


def get_all_node_metrics(metrics_res: Any) -> dict[str, dict[str, Any]]:
    """Return mapping of node_name -> metrics dict from engine.get_metrics() or /metrics."""
    d = to_dict(metrics_res)
    for container_key in ("nodes", "per_node_metrics", "node_metrics", "distillers"):
        container = d.get(container_key)
        if isinstance(container, dict):
            return {k: to_dict(v) for k, v in container.items()}
    found = {
        k: to_dict(v)
        for k, v in d.items()
        if k in ("intent_classifier", "policy_router", "reply_writer")
    }
    if not found:
        raise KeyError(f"No node metrics container found in keys={list(d.keys())}")
    return found


def extract_checkpoints(payload: Any) -> list[dict[str, Any]]:
    """Unwrap GET /checkpoints/{thread_id} response (list or {'checkpoints': [...]}) strictly."""
    if isinstance(payload, list):
        return [to_dict(c) for c in payload]
    if isinstance(payload, dict) and "checkpoints" in payload and isinstance(payload["checkpoints"], list):
        return [to_dict(c) for c in payload["checkpoints"]]
    raise KeyError(f"Expected list or dict with 'checkpoints' list, got {type(payload).__name__}: {payload!r}")


def ticket_to_api_payload(ticket: Any) -> dict[str, Any]:
    """Convert a SupportTicket into a JSON-serializable dict for FastAPI /run."""
    d = to_dict(ticket)
    return {
        "ticket_id": d["ticket_id"],
        "customer_name": d["customer_name"],
        "city": d["city"],
        "order_id": d["order_id"],
        "category": enum_str(d["category"]),
        "item_price_inr": float(d["item_price_inr"]),
        "customer_return_rate": float(d["customer_return_rate"]),
        "days_since_delivery": int(d["days_since_delivery"]),
        "ticket_text": d["ticket_text"],
        "split": d.get("split", "train_stream"),
        "is_drift": bool(d.get("is_drift", False)),
    }


def isolated_subprocess_env(db_path: Path) -> dict[str, str]:
    """Return offline subprocess environment with isolated tmp_path SQLite DB path."""
    env = os.environ.copy()
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        env.pop(key, None)
    env["ZEROTAX_DB_PATH"] = str(db_path)
    return env


def stream_until_complete(
    engine: Any,
    tickets: list[Any],
    *,
    thread_id: str,
    approve: bool = True,
    audit_rate: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stream tickets with auto_approve=False, draining chained interrupts via approve_promotion."""
    batch_results: list[dict[str, Any]] = []
    interrupts_seen: list[dict[str, Any]] = []
    idx = 0
    while idx < len(tickets):
        res = engine.stream_batch(
            tickets[idx:],
            thread_id=thread_id,
            auto_approve=False,
            audit_rate=audit_rate,
        )
        d = to_dict(res)
        batch_results.append(d)
        processed = int(d.get("processed_count", d.get("processed", 0)))
        if d.get("interrupted"):
            cur = d
            while cur.get("interrupted"):
                payload = get_interrupt_payload(cur)
                if payload:
                    interrupts_seen.append(payload)
                approve_res = engine.approve_promotion(thread_id, approved=approve)
                cur = to_dict(approve_res)
                batch_results.append(cur)
            idx += max(processed, 1)
        else:
            if processed <= 0:
                break
            idx += processed
    return batch_results, interrupts_seen
