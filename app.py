"""ZERO-TAX Console: Black & Neon Plain-English Interface with Auto-Pilot."""

from __future__ import annotations

import html
import os
import uuid
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.dataset import generate_dataset
from core.distiller import DistillerConfig
from core.graph import ZeroTaxEngine
from core.models import (
    ExecutionMode,
    NodeStatus,
    ProductCategory,
    SupportTicket,
    TeacherMode,
)

st.set_page_config(
    page_title="ZERO-TAX // Smart AI Cost-Saver Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_VIEWS = [
    "1. Live Dashboard",
    "2. Try a Customer Ticket",
    "3. Cost & Speed Charts",
    "4. History & Undo",
]

PLAIN_NODE_NAMES = {
    "intent_classifier": "Task 1: Detect Problem",
    "policy_router": "Task 2: Pick Refund Action",
    "reply_writer": "Task 3: Write Customer Email",
}

NEON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-canvas: #05070A;
    --bg-panel: #0B0F17;
    --bg-elevated: #101724;
    --bg-input: #080C12;
    --border-subtle: #1A2332;
    --border-hover: #2B3B55;
    --text-primary: #F0F6FC;
    --text-secondary: #8B949E;
    --text-muted: #5A6577;
    --neon-cyan: #00F5D4;
    --neon-cyan-dim: rgba(0, 245, 212, 0.12);
    --neon-violet: #A855F7;
    --neon-violet-dim: rgba(168, 85, 247, 0.14);
    --neon-amber: #F59E0B;
    --neon-amber-dim: rgba(245, 158, 11, 0.14);
    --neon-crimson: #FF3B6B;
    --neon-crimson-dim: rgba(255, 59, 107, 0.14);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg-canvas) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"] {
    background: rgba(5, 7, 10, 0.88) !important;
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border-subtle);
}

#MainMenu, footer, [data-testid="stStatusWidget"] {
    visibility: hidden !important;
}

.block-container {
    padding-top: 3.1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1420px !important;
}

[data-testid="stSidebar"] {
    background-color: #070A10 !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary);
}

.zt-brand {
    padding: 0.4rem 0 1.05rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 0.9rem;
}

.zt-brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
}

.zt-brand-accent {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--neon-cyan);
    box-shadow: 0 0 10px var(--neon-cyan);
}

.zt-brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    letter-spacing: 0.12em;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

.zt-sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin: 0.95rem 0 0.45rem 0;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 6px !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    padding: 0.55rem 0.75rem !important;
    margin: 0 !important;
    width: 100% !important;
    transition: all 0.16s ease !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    border-color: var(--neon-cyan) !important;
    background: var(--bg-elevated) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    border-color: var(--neon-cyan) !important;
    background: var(--neon-cyan-dim) !important;
    box-shadow: inset 3px 0 0 var(--neon-cyan), 0 0 14px rgba(0, 245, 212, 0.12) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

.stButton > button, .stFormSubmitButton > button {
    background-color: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-hover) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.54rem 0.85rem !important;
    transition: all 0.16s ease !important;
}

.stButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    box-shadow: 0 0 14px rgba(0, 245, 212, 0.18) !important;
}

.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background-color: var(--neon-cyan) !important;
    color: #05070A !important;
    border: 1px solid var(--neon-cyan) !important;
    box-shadow: 0 0 18px rgba(0, 245, 212, 0.25) !important;
}

.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
    background-color: #33FFE2 !important;
    color: #05070A !important;
    box-shadow: 0 0 24px rgba(0, 245, 212, 0.42) !important;
}

.stButton > button:disabled,
.stButton > button[kind="primary"]:disabled,
.stFormSubmitButton > button:disabled,
.stFormSubmitButton > button[kind="primary"]:disabled {
    background-color: #080B11 !important;
    color: #4B5565 !important;
    border: 1px solid #141B26 !important;
    box-shadow: none !important;
    opacity: 0.55 !important;
}

[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
textarea {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within,
textarea:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 0 1px var(--neon-cyan) !important;
}

.zt-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    padding-bottom: 0.95rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1rem;
    gap: 1rem;
    flex-wrap: wrap;
}

.zt-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    line-height: 1.1;
}

.zt-title span {
    color: var(--neon-cyan);
}

.zt-subtitle {
    font-size: 0.86rem;
    color: var(--text-secondary);
    margin-top: 0.35rem;
}

.zt-env-pill {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    padding: 0.4rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
}

.zt-env-pill strong {
    color: var(--neon-cyan);
    font-weight: 600;
}

.zt-stepper {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.1rem;
}

.zt-step {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.62rem 0.85rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    color: var(--text-muted);
    gap: 0.5rem;
}

.zt-step.done {
    border-color: rgba(0, 245, 212, 0.35);
    color: var(--text-secondary);
}

.zt-step.active-cyan {
    border-color: var(--neon-cyan);
    background: linear-gradient(135deg, rgba(0, 245, 212, 0.11), rgba(11, 15, 23, 0.95));
    color: var(--neon-cyan);
    box-shadow: 0 0 16px rgba(0, 245, 212, 0.14);
}

.zt-step.active-amber {
    border-color: var(--neon-amber);
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.14), rgba(11, 15, 23, 0.95));
    color: var(--neon-amber);
    box-shadow: 0 0 16px rgba(245, 158, 11, 0.18);
}

.zt-step.active-crimson {
    border-color: var(--neon-crimson);
    background: linear-gradient(135deg, rgba(255, 59, 107, 0.14), rgba(11, 15, 23, 0.95));
    color: var(--neon-crimson);
    box-shadow: 0 0 16px rgba(255, 59, 107, 0.18);
}

.zt-step-tag {
    font-size: 0.64rem;
    letter-spacing: 0.06em;
    opacity: 0.92;
    white-space: nowrap;
}

.zt-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin-bottom: 1.15rem;
}

.zt-kpi-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-top: 2px solid var(--neon-cyan);
    border-radius: 8px;
    padding: 0.9rem 1.05rem;
}

.zt-kpi-card.violet {
    border-top-color: var(--neon-violet);
}

.zt-kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    letter-spacing: 0.11em;
    color: var(--text-secondary);
    text-transform: uppercase;
}

.zt-kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.82rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
    margin-top: 0.22rem;
    line-height: 1.15;
}

.zt-kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.71rem;
    font-variant-numeric: tabular-nums;
    color: var(--neon-cyan);
    margin-top: 0.35rem;
}

.zt-kpi-sub.muted {
    color: var(--text-secondary);
}

.zt-section-hdr {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin: 1.15rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.zt-section-hdr::before {
    content: "";
    display: inline-block;
    width: 14px;
    height: 2px;
    background: var(--neon-cyan);
}

.zt-node-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1.05rem 1.15rem;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: border-color 0.16s ease;
}

.zt-node-card:hover {
    border-color: var(--border-hover);
}

.zt-node-card.student {
    border-color: rgba(0, 245, 212, 0.45);
    box-shadow: 0 0 24px rgba(0, 245, 212, 0.06);
}

.zt-node-card.candidate {
    border-color: rgba(245, 158, 11, 0.55);
}

.zt-node-card.demoted {
    border-color: rgba(255, 59, 107, 0.55);
}

.zt-node-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}

.zt-node-index {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.11em;
    color: var(--text-muted);
}

.zt-node-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 0.1rem;
}

.zt-node-desc {
    font-size: 0.76rem;
    color: var(--text-secondary);
    margin-bottom: 0.65rem;
    line-height: 1.35;
}

.zt-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.24rem 0.58rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    white-space: nowrap;
}

.zt-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
}

.zt-badge-cyan {
    color: var(--neon-cyan);
    background: var(--neon-cyan-dim);
    border: 1px solid rgba(0, 245, 212, 0.4);
}
.zt-badge-cyan .zt-dot {
    background: var(--neon-cyan);
    box-shadow: 0 0 8px var(--neon-cyan);
}

.zt-badge-violet {
    color: #C084FC;
    background: var(--neon-violet-dim);
    border: 1px solid rgba(168, 85, 247, 0.4);
}
.zt-badge-violet .zt-dot {
    background: var(--neon-violet);
    box-shadow: 0 0 8px var(--neon-violet);
}

.zt-badge-amber {
    color: var(--neon-amber);
    background: var(--neon-amber-dim);
    border: 1px solid rgba(245, 158, 11, 0.45);
}
.zt-badge-amber .zt-dot {
    background: var(--neon-amber);
    box-shadow: 0 0 8px var(--neon-amber);
}

.zt-badge-crimson {
    color: var(--neon-crimson);
    background: var(--neon-crimson-dim);
    border: 1px solid rgba(255, 59, 107, 0.45);
}
.zt-badge-crimson .zt-dot {
    background: var(--neon-crimson);
    box-shadow: 0 0 8px var(--neon-crimson);
}

.zt-stat-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.5rem 0.7rem;
    margin: 0.25rem 0 0.85rem 0;
}

.zt-stat-box {
    background: var(--bg-input);
    border: 1px solid #141C28;
    border-radius: 6px;
    padding: 0.44rem 0.62rem;
}

.zt-stat-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.61rem;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    text-transform: uppercase;
}

.zt-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
    margin-top: 0.14rem;
}

.zt-prog-wrap {
    margin-top: auto;
}

.zt-prog-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

.zt-prog-track {
    width: 100%;
    height: 5px;
    background: #141C28;
    border-radius: 999px;
    overflow: hidden;
}

.zt-prog-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.25s ease;
}

.zt-hitl-box {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.10), rgba(11, 15, 23, 0.98));
    border: 1px solid var(--neon-amber);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 24px rgba(245, 158, 11, 0.12);
}

.zt-hitl-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--neon-amber);
    margin-bottom: 0.55rem;
}

.zt-hitl-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
}

.zt-spotlight {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--neon-cyan);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}

.zt-spotlight-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.55rem;
}

.zt-spotlight-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: var(--neon-cyan);
}

.zt-spotlight-msg {
    background: var(--bg-input);
    border: 1px solid #141C28;
    border-radius: 6px;
    padding: 0.65rem 0.85rem;
    font-size: 0.86rem;
    color: var(--text-primary);
    margin-bottom: 0.7rem;
}

.zt-spotlight-cols {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
}

.zt-empty-panel {
    background: var(--bg-panel);
    border: 1px dashed var(--border-hover);
    border-radius: 10px;
    padding: 1.55rem 1.4rem;
    text-align: center;
    margin-top: 0.75rem;
}

.zt-empty-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
}

.zt-empty-desc {
    font-size: 0.84rem;
    color: var(--text-secondary);
    margin-top: 0.35rem;
}

.zt-reply-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--neon-violet);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    line-height: 1.55;
    color: var(--text-primary);
    margin-top: 0.65rem;
}

.zt-sidebar-meta {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.7rem 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.71rem;
    line-height: 1.65;
    color: var(--text-secondary);
    margin-top: 0.85rem;
}

.zt-sidebar-meta strong {
    color: var(--neon-cyan);
}

@media (max-width: 960px) {
    .zt-kpi-grid, .zt-stepper, .zt-hitl-grid, .zt-spotlight-cols {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
</style>
"""


def _sync_gemini_secret() -> None:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        try:
            secret_val = str(st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or "").strip()
        except Exception:
            secret_val = ""
        if secret_val:
            os.environ["GEMINI_API_KEY"] = secret_val


def _init_session() -> None:
    _sync_gemini_secret()
    if st.session_state.get("ui_version") != 2:
        st.session_state.ui_version = 2
        if "ticket_log" in st.session_state:
            st.session_state.ticket_log = [
                r for r in st.session_state.ticket_log if isinstance(r, dict) and "customer" in r
            ]
    if "db_path" not in st.session_state:
        env_db = os.getenv("ZEROTAX_DB_PATH")
        st.session_state.db_path = env_db or f"data/sessions/{uuid.uuid4().hex[:8]}.db"
    if "engine" not in st.session_state:
        cfg = DistillerConfig(confidence_threshold=0.90, shadow_audit_rate=0.05)
        st.session_state.engine = ZeroTaxEngine(
            db_path=st.session_state.db_path,
            teacher_mode=TeacherMode.AUTO,
            config=cfg,
            seed=42,
            sleep_on_simulated=False,
        )
    if "dataset" not in st.session_state:
        st.session_state.dataset = generate_dataset(seed=42, main_count=200, holdout_count=50, drift_count=50)
    if "ticket_lookup" not in st.session_state:
        ds = st.session_state.dataset
        st.session_state.ticket_lookup = {t.ticket_id: t for t in ds.all_tickets}
    if "stream_pos" not in st.session_state:
        st.session_state.stream_pos = 0
    if "drift_pos" not in st.session_state:
        st.session_state.drift_pos = 0
    if "ticket_log" not in st.session_state:
        st.session_state.ticket_log = []
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None
    if "pending_remaining" not in st.session_state:
        st.session_state.pending_remaining = []
    if "last_probe" not in st.session_state:
        st.session_state.last_probe = None
    if "active_action" not in st.session_state:
        st.session_state.active_action = "auto_200"
    if "probe_preset" not in st.session_state:
        st.session_state.probe_preset = {
            "category": ProductCategory.FOOTWEAR.value,
            "price": 1899.0,
            "return_rate": 0.08,
            "days": 3,
            "text": "My running sneaker sole came unglued and peeled on day one.",
        }


def _human_mode_label(mode: str) -> str:
    if mode == ExecutionMode.SKLEARN_STUDENT.value:
        return "Free Local AI (Rs. 0)"
    if mode == ExecutionMode.LOW_CONF_FALLBACK.value:
        return "Cloud AI Backup"
    return "Paid Cloud AI"


def _record_result(res: dict[str, Any], ticket_obj: Optional[SupportTicket] = None) -> None:
    tel = res.get("telemetry")
    if not tel:
        return
    tid = str(tel["ticket_id"])
    if st.session_state.ticket_log and st.session_state.ticket_log[-1]["ticket_id"] == tid:
        st.session_state.ticket_log[-1]["demoted"] = bool(res.get("drift_demoted_nodes"))
        return

    resolved = ticket_obj or st.session_state.ticket_lookup.get(tid)
    customer = f"{resolved.customer_name} ({resolved.city})" if resolved else "Support Customer"
    item_info = (
        f"{resolved.category.value} | Rs. {resolved.item_price_inr:.0f} | {resolved.days_since_delivery}d"
        if resolved
        else "-"
    )
    complaint = resolved.ticket_text if resolved else "-"

    intent_mode = tel["intent_result"]["execution_mode"]
    policy_mode = tel["policy_result"]["execution_mode"]
    cost_inr = float(tel["total_cost_inr"])
    baseline_inr = float(tel["baseline_cost_inr"])

    st.session_state.ticket_log.append(
        {
            "idx": len(st.session_state.ticket_log) + 1,
            "ticket_id": tid,
            "customer": customer,
            "item_info": item_info,
            "complaint": complaint,
            "problem_tag": tel["intent_result"]["prediction"],
            "task1_ai": _human_mode_label(intent_mode),
            "task1_ms": float(tel["intent_result"]["latency_ms"]),
            "refund_action": tel["policy_result"]["prediction"],
            "task2_ai": _human_mode_label(policy_mode),
            "task2_ms": float(tel["policy_result"]["latency_ms"]),
            "reply_text": str(tel.get("reply_text") or ""),
            "intent_mode": intent_mode,
            "policy_mode": policy_mode,
            "cost_inr": cost_inr,
            "baseline_cost_inr": baseline_inr,
            "saved_inr": max(0.0, baseline_inr - cost_inr),
            "latency_ms": float(tel["total_latency_ms"]),
            "baseline_latency_ms": float(tel["baseline_latency_ms"]),
            "is_drift": bool(tel.get("is_drift", False)),
            "demoted": bool(res.get("drift_demoted_nodes")),
        }
    )


def _stream_tickets(tickets: list[SupportTicket], thread_id: str, auto_approve: bool, audit_rate: float) -> None:
    engine: ZeroTaxEngine = st.session_state.engine
    for i, ticket in enumerate(tickets):
        res = engine.run_ticket(
            ticket,
            thread_id=thread_id,
            auto_approve=auto_approve,
            audit_rate=audit_rate,
        )
        _record_result(res, ticket)
        if ticket.split == "train_stream":
            st.session_state.stream_pos = min(
                len(st.session_state.dataset.train_stream),
                st.session_state.stream_pos + 1,
            )
        if res.get("interrupted"):
            st.session_state.pending_interrupt = res["interrupt_payload"]
            st.session_state.pending_remaining = tickets[i + 1 :]
            return
    st.session_state.pending_interrupt = None
    st.session_state.pending_remaining = []


def _reset_session_state(engine: ZeroTaxEngine) -> None:
    engine.reset()
    st.session_state.ticket_log = []
    st.session_state.stream_pos = 0
    st.session_state.drift_pos = 0
    st.session_state.pending_interrupt = None
    st.session_state.pending_remaining = []
    st.session_state.last_probe = None
    st.session_state.active_action = "auto_200"


def _status_style(status: str, is_always_llm: bool = False) -> tuple[str, str, str]:
    if is_always_llm:
        return "ALWAYS PAID CLOUD AI", "zt-badge-violet", ""
    if status == NodeStatus.STUDENT_ACTIVE.value:
        return "FREE LOCAL AI (1ms)", "zt-badge-cyan", "student"
    if status == NodeStatus.DEMOTED_DRIFT.value:
        return "SCAM SHIELD: CLOUD AI", "zt-badge-crimson", "demoted"
    if status == NodeStatus.CANDIDATE_READY.value:
        return "APPROVAL READY", "zt-badge-amber", "candidate"
    return "PAID CLOUD AI", "zt-badge-violet", ""


def _render_stepper(nodes: dict[str, Any], has_interrupt: bool, has_drift_runs: bool) -> str:
    n1 = nodes["intent_classifier"]
    n2 = nodes["policy_router"]
    any_promoted = bool(n1["is_promoted"] or n2["is_promoted"])
    any_demoted = bool(n1["demotion_count"] or n2["demotion_count"])
    total_traces = int(n1["trace_count"])

    s1_cls = "done" if any_promoted or has_interrupt else "active-cyan"
    if has_interrupt:
        s2_cls = "active-amber"
    elif any_promoted:
        s2_cls = "done"
    else:
        s2_cls = ""

    if any_demoted or has_drift_runs:
        s3_cls = "done"
        is_currently_demoted = (
            n1["status"] == NodeStatus.DEMOTED_DRIFT.value
            or n2["status"] == NodeStatus.DEMOTED_DRIFT.value
        )
        s4_cls = "active-crimson" if is_currently_demoted else "active-cyan"
    elif any_promoted:
        s3_cls = "active-cyan"
        s4_cls = ""
    else:
        s3_cls = ""
        s4_cls = ""

    return (
        '<div class="zt-stepper">'
        f'<div class="zt-step {s1_cls}"><span>STAGE 1: LEARN FROM CLOUD AI</span><span class="zt-step-tag">{total_traces} EXAMPLES</span></div>'
        f'<div class="zt-step {s2_cls}"><span>STAGE 2: ACCURACY EXAM (95%+)</span><span class="zt-step-tag">{"PENDING" if has_interrupt else ("PASSED" if any_promoted else "WAITING")}</span></div>'
        f'<div class="zt-step {s3_cls}"><span>STAGE 3: FREE LOCAL AI ACTIVE</span><span class="zt-step-tag">{int(n1["is_promoted"]) + int(n2["is_promoted"])}/2 TASKS FREE</span></div>'
        f'<div class="zt-step {s4_cls}"><span>STAGE 4: SCAM SHIELD</span><span class="zt-step-tag">{int(n1["demotion_count"]) + int(n2["demotion_count"])} ALERTS</span></div>'
        "</div>"
    )


def _render_kpi_strip(metrics: dict[str, Any]) -> str:
    tickets = metrics["total_tickets"]
    cost_red = metrics["cost_reduction_pct"]
    cost_sub = f"Rs. {metrics['total_cost_inr']:.2f} paid vs Rs. {metrics['baseline_cost_inr']:.2f} all-LLM"
    lat_red = metrics["latency_reduction_pct"]
    lat_sub = f"{metrics['avg_latency_ms']:.1f} ms avg (vs {metrics['baseline_avg_latency_ms']:.0f} ms)"
    evap = metrics["evaporation_rate_pct"]
    fb_sub = f"Cloud backup safety rate: {metrics['fallback_rate_pct']:.1f}%"

    return (
        '<div class="zt-kpi-grid">'
        '<div class="zt-kpi-card">'
        '<div class="zt-kpi-label">TICKETS HANDLED</div>'
        f'<div class="zt-kpi-value">{tickets}</div>'
        '<div class="zt-kpi-sub muted">Customer support complaints</div>'
        "</div>"
        '<div class="zt-kpi-card">'
        '<div class="zt-kpi-label">MONEY SAVED</div>'
        f'<div class="zt-kpi-value">{cost_red:.1f}%</div>'
        f'<div class="zt-kpi-sub">{html.escape(cost_sub)}</div>'
        "</div>"
        '<div class="zt-kpi-card">'
        '<div class="zt-kpi-label">SPEED BOOST</div>'
        f'<div class="zt-kpi-value">{lat_red:.1f}%</div>'
        f'<div class="zt-kpi-sub">{html.escape(lat_sub)}</div>'
        "</div>"
        '<div class="zt-kpi-card violet">'
        '<div class="zt-kpi-label">FREE LOCAL AI USAGE</div>'
        f'<div class="zt-kpi-value">{evap:.1f}%</div>'
        f'<div class="zt-kpi-sub muted">{html.escape(fb_sub)}</div>'
        "</div>"
        "</div>"
    )


def _render_node_card(
    node_idx: str,
    title: str,
    desc: str,
    node_name: str,
    nm: dict[str, Any],
    engine: ZeroTaxEngine,
    always_llm: bool,
) -> str:
    label, badge_cls, card_cls = _status_style(nm["status"], is_always_llm=always_llm)
    if always_llm:
        pct = 100
        prog_color = "#A855F7"
        prog_left = "ENGINE RULE"
        prog_right = "KEPT ON PAID CLOUD AI"
        stats_html = (
            '<div class="zt-stat-grid">'
            f'<div class="zt-stat-box"><div class="zt-stat-lbl">CLOUD AI CALLS</div><div class="zt-stat-val">{nm["teacher_calls"]}</div></div>'
            f'<div class="zt-stat-box"><div class="zt-stat-lbl">AVG SPEED</div><div class="zt-stat-val">{nm["avg_latency_ms"]:.1f} ms</div></div>'
            f'<div class="zt-stat-box"><div class="zt-stat-lbl">TOTAL COST</div><div class="zt-stat-val">Rs. {nm["total_cost_inr"]:.3f}</div></div>'
            '<div class="zt-stat-box"><div class="zt-stat-lbl">WHY CLOUD AI?</div><div class="zt-stat-val">Natural Writing</div></div>'
            "</div>"
        )
    else:
        is_retrain = nm["status"] == NodeStatus.DEMOTED_DRIFT.value or int(nm["model_version"]) >= 1
        threshold = (
            engine.config.retrain_interval_traces
            if is_retrain
            else engine.config.min_samples_for_promotion + (25 if node_name == "policy_router" else 0)
        )
        since = int(nm["traces_since_last_train"])
        pct = 100 if nm["is_promoted"] else min(100, int(round(100 * since / max(1, threshold))))
        prog_color = "#FF3B6B" if nm["status"] == NodeStatus.DEMOTED_DRIFT.value else "#00F5D4"
        prog_left = f"LEARNED: {nm['trace_count']} | v{nm['model_version']}"
        pool_label = "RE-LEARNING" if is_retrain else "LEARNING"
        prog_right = "FREE LOCAL AI ACTIVE" if nm["is_promoted"] else f"{pool_label} {since}/{threshold}"
        cv_str = f"{nm['cv_accuracy'] * 100:.1f}%" if nm["cv_accuracy"] is not None else "--"
        agr_str = f"{nm['rolling_shadow_agreement'] * 100:.1f}%" if nm["rolling_shadow_agreement"] is not None else "--"
        stats_html = (
            '<div class="zt-stat-grid">'
            f'<div class="zt-stat-box"><div class="zt-stat-lbl">CLOUD / FREE CALLS</div><div class="zt-stat-val">{nm["teacher_calls"]} / {nm["student_calls"]}</div></div>'
            f'<div class="zt-stat-box"><div class="zt-stat-lbl">EXAM ACCURACY</div><div class="zt-stat-val">{cv_str}</div></div>'
            f'<div class="zt-stat-box"><div class="zt-stat-lbl">CLOUD BACKUPS</div><div class="zt-stat-val">{nm["low_conf_fallbacks"]}</div></div>'
            f'<div class="zt-stat-box"><div class="zt-stat-lbl">CHECK / ALERTS</div><div class="zt-stat-val">{agr_str} | {nm["demotion_count"]}</div></div>'
            "</div>"
        )

    return (
        f'<div class="zt-node-card {card_cls}">'
        "<div>"
        '<div class="zt-node-top">'
        "<div>"
        f'<div class="zt-node-index">{html.escape(node_idx)}</div>'
        f'<div class="zt-node-name">{html.escape(title)}</div>'
        "</div>"
        f'<span class="zt-badge {badge_cls}"><span class="zt-dot"></span>{label}</span>'
        "</div>"
        f'<div class="zt-node-desc">{html.escape(desc)}</div>'
        f"{stats_html}"
        "</div>"
        '<div class="zt-prog-wrap">'
        f'<div class="zt-prog-meta"><span>{prog_left}</span><span>{prog_right}</span></div>'
        '<div class="zt-prog-track">'
        f'<div class="zt-prog-fill" style="width: {pct}%; background: {prog_color};"></div>'
        "</div>"
        "</div>"
        "</div>"
    )


def _render_latest_spotlight(ticket_log: list[dict[str, Any]]) -> None:
    if not ticket_log:
        return
    last = ticket_log[-1]
    st.markdown('<div class="zt-section-hdr">Latest Customer Ticket Processed</div>', unsafe_allow_html=True)
    tid = html.escape(str(last.get("ticket_id", "-")))
    cust = html.escape(str(last.get("customer", "Support Customer")))
    item = html.escape(str(last.get("item_info", "-")))
    cost = float(last.get("cost_inr", 0.0))
    saved = float(last.get("saved_inr", 0.0))
    lat = float(last.get("latency_ms", 0.0))
    complaint = html.escape(str(last.get("complaint", "-")))
    prob = html.escape(str(last.get("problem_tag", last.get("intent_pred", "-"))))
    t1_ai = html.escape(str(last.get("task1_ai", _human_mode_label(str(last.get("intent_mode", ""))))))
    t1_ms = float(last.get("task1_ms", 0.0))
    act = html.escape(str(last.get("refund_action", last.get("policy_pred", "-"))))
    t2_ai = html.escape(str(last.get("task2_ai", _human_mode_label(str(last.get("policy_mode", ""))))))
    t2_ms = float(last.get("task2_ms", 0.0))
    reply = html.escape(str(last.get("reply_text", "-")))

    spotlight_html = (
        '<div class="zt-spotlight">'
        '<div class="zt-spotlight-top">'
        f'<div class="zt-spotlight-meta">{tid} // {cust} // {item}</div>'
        f'<div class="zt-spotlight-meta">PAID: Rs. {cost:.3f} (SAVED Rs. {saved:.3f}) // SPEED: {lat:.0f} ms</div>'
        "</div>"
        f'<div class="zt-spotlight-msg"><strong>Customer Complaint:</strong> "{complaint}"</div>'
        '<div class="zt-spotlight-cols">'
        f'<div class="zt-stat-box"><div class="zt-stat-lbl">TASK 1: PROBLEM DETECTED</div><div class="zt-stat-val">{prob}</div><div class="zt-node-index" style="margin-top:4px;">{t1_ai} ({t1_ms:.1f} ms)</div></div>'
        f'<div class="zt-stat-box"><div class="zt-stat-lbl">TASK 2: ACTION PICKED</div><div class="zt-stat-val">{act}</div><div class="zt-node-index" style="margin-top:4px;">{t2_ai} ({t2_ms:.1f} ms)</div></div>'
        f'<div class="zt-stat-box"><div class="zt-stat-lbl">TASK 3: EMAIL SENT TO CUSTOMER</div><div class="zt-node-desc" style="margin:4px 0 0 0; color:#F0F6FC;">{reply}</div></div>'
        "</div>"
        "</div>"
    )
    st.markdown(spotlight_html, unsafe_allow_html=True)


def _render_ticket_table(ticket_log: list[dict[str, Any]], limit: int = 25) -> None:
    rows = ticket_log[-limit:][::-1]
    display_df = pd.DataFrame(
        [
            {
                "Ticket": r.get("ticket_id", "-"),
                "Item & Price": r.get("item_info", "-"),
                "Customer Complaint": r.get("complaint", "-"),
                "Task 1: Problem": r.get("problem_tag", r.get("intent_pred", "-")),
                "Task 1 AI": r.get("task1_ai", _human_mode_label(str(r.get("intent_mode", "")))),
                "Task 2: Action": r.get("refund_action", r.get("policy_pred", "-")),
                "Task 2 AI": r.get("task2_ai", _human_mode_label(str(r.get("policy_mode", "")))),
                "Paid (Rs.)": round(float(r.get("cost_inr", 0.0)), 3),
                "Normal LLM (Rs.)": round(float(r.get("baseline_cost_inr", 0.0)), 3),
                "Time (ms)": round(float(r.get("latency_ms", 0.0)), 1),
            }
            for r in rows
        ]
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_charts(ticket_log: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(ticket_log)
    df["cum_cost_inr"] = df["cost_inr"].cumsum()
    df["cum_baseline_inr"] = df["baseline_cost_inr"].cumsum()

    fig_cost = go.Figure()
    fig_cost.add_trace(
        go.Scatter(
            x=df["idx"],
            y=df["cum_baseline_inr"],
            name="Normal All-LLM Cost (Rs.)",
            line=dict(color="#A855F7", width=2, dash="dash"),
        )
    )
    fig_cost.add_trace(
        go.Scatter(
            x=df["idx"],
            y=df["cum_cost_inr"],
            name="ZERO-TAX Cost Paid (Rs.)",
            line=dict(color="#00F5D4", width=2.8),
            fill="tonexty",
            fillcolor="rgba(0, 245, 212, 0.08)",
        )
    )
    fig_cost.update_layout(
        title=dict(text="CUMULATIVE MONEY SPENT (RS.)", font=dict(family="JetBrains Mono", size=12, color="#8B949E")),
        xaxis_title="Ticket #",
        yaxis_title="Total Cost (Rs.)",
        height=315,
        margin=dict(l=44, r=18, t=42, b=38),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#080C12",
        font=dict(color="#E6EDF3", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(gridcolor="#141C28", zerolinecolor="#1A2332"),
        yaxis=dict(gridcolor="#141C28", zerolinecolor="#1A2332"),
    )

    fig_lat = go.Figure()
    fig_lat.add_trace(
        go.Scatter(
            x=df["idx"],
            y=df["baseline_latency_ms"],
            name="Normal All-LLM Speed (ms)",
            line=dict(color="#A855F7", width=1.8, dash="dash"),
        )
    )
    fig_lat.add_trace(
        go.Scatter(
            x=df["idx"],
            y=df["latency_ms"],
            name="ZERO-TAX Speed (ms)",
            line=dict(color="#00F5D4", width=2.2),
        )
    )
    demote_rows = df[df["demoted"]]
    if not demote_rows.empty:
        fig_lat.add_trace(
            go.Scatter(
                x=demote_rows["idx"],
                y=demote_rows["latency_ms"],
                mode="markers",
                name="Scam Alert (Switched Back to Cloud AI)",
                marker=dict(color="#FF3B6B", size=9, symbol="diamond"),
            )
        )
    fig_lat.update_layout(
        title=dict(text="RESPONSE TIME PER TICKET (MS)", font=dict(family="JetBrains Mono", size=12, color="#8B949E")),
        xaxis_title="Ticket #",
        yaxis_title="Time (ms)",
        height=315,
        margin=dict(l=44, r=18, t=42, b=38),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#080C12",
        font=dict(color="#E6EDF3", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(gridcolor="#141C28", zerolinecolor="#1A2332"),
        yaxis=dict(gridcolor="#141C28", zerolinecolor="#1A2332"),
    )

    ch1, ch2 = st.columns(2)
    ch1.plotly_chart(fig_cost, use_container_width=True)
    ch2.plotly_chart(fig_lat, use_container_width=True)


def _render_hitl_gate(engine: ZeroTaxEngine, thread_id: str, auto_approve: bool, audit_rate: float) -> None:
    p = st.session_state.pending_interrupt
    if p is None:
        return
    node_key = str(p.get("node_name", ""))
    plain_title = PLAIN_NODE_NAMES.get(node_key, node_key)
    cv_pct = float(p.get("cv_accuracy", 0.0)) * 100
    ece_val = float(p.get("ece", 0.0))
    samples = p.get("sample_count", 0)
    savings_pct = p.get("estimated_cost_savings_pct", 0)
    lat_saved = p.get("estimated_latency_savings_ms", 0)

    banner_html = (
        '<div class="zt-hitl-box">'
        f'<div class="zt-hitl-title">APPROVAL NEEDED // Switch "{html.escape(plain_title)}" from Paid Cloud AI to Free Local AI?</div>'
        '<div class="zt-hitl-grid">'
        f'<div class="zt-stat-box"><div class="zt-stat-lbl">EXAM ACCURACY</div><div class="zt-stat-val">{cv_pct:.2f}%</div></div>'
        f'<div class="zt-stat-box"><div class="zt-stat-lbl">RELIABILITY ERROR</div><div class="zt-stat-val">{ece_val:.4f} (Pass &lt; 0.10)</div></div>'
        f'<div class="zt-stat-box"><div class="zt-stat-lbl">EXAMPLES LEARNED</div><div class="zt-stat-val">{samples}</div></div>'
        f'<div class="zt-stat-box"><div class="zt-stat-lbl">EXPECTED SAVINGS</div><div class="zt-stat-val">{savings_pct}% cost | {lat_saved} ms faster</div></div>'
        "</div>"
        "</div>"
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    b_app, b_rej = st.columns(2)
    if b_app.button("Approve & Switch to Free Local AI", type="primary", use_container_width=True):
        res = engine.approve_promotion(thread_id, approved=True)
        _record_result(res)
        if res.get("interrupted"):
            st.session_state.pending_interrupt = res["interrupt_payload"]
        else:
            st.session_state.pending_interrupt = None
            rem = list(st.session_state.pending_remaining)
            st.session_state.pending_remaining = []
            if rem:
                _stream_tickets(rem, thread_id, auto_approve, audit_rate)
        st.rerun()

    if b_rej.button("Keep Paid Cloud AI for Now", use_container_width=True):
        res = engine.approve_promotion(thread_id, approved=False)
        _record_result(res)
        st.session_state.pending_interrupt = res["interrupt_payload"] if res.get("interrupted") else None
        st.session_state.pending_remaining = []
        st.rerun()


def main() -> None:
    _init_session()
    st.markdown(NEON_CSS, unsafe_allow_html=True)

    engine: ZeroTaxEngine = st.session_state.engine
    dataset = st.session_state.dataset

    with st.sidebar:
        st.markdown(
            '<div class="zt-brand">'
            '<div class="zt-brand-title"><span class="zt-brand-accent"></span>ZERO-TAX</div>'
            '<div class="zt-brand-sub">AI COST-SAVER ENGINE</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="zt-sidebar-label">Menu</div>', unsafe_allow_html=True)
        active_view = st.radio(
            "Menu",
            NAV_VIEWS,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<div class="zt-sidebar-label">Simple Settings</div>', unsafe_allow_html=True)
        auto_approve = st.toggle("Auto-Switch to Free AI (Auto-Pilot)", value=True)
        teacher_choice = st.selectbox(
            "Cloud AI Mode",
            [TeacherMode.SIMULATED.value, TeacherMode.AUTO.value, TeacherMode.GEMINI.value],
            index=0,
        )
        gemini_key_input = st.text_input(
            "Gemini API Key (optional, for gemini/auto)",
            value=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
            type="password",
        )
        if gemini_key_input.strip():
            os.environ["GEMINI_API_KEY"] = gemini_key_input.strip()
        conf_thresh = st.slider("Free AI Safety Threshold", 0.50, 0.99, float(engine.config.confidence_threshold), 0.01)
        audit_rate = st.slider("Background Spot-Check Rate", 0.00, 1.00, float(engine.config.shadow_audit_rate), 0.01)
        engine.config.confidence_threshold = float(conf_thresh)
        engine.config.shadow_audit_rate = float(audit_rate)
        for dist in engine.distillers.values():
            dist.config.confidence_threshold = float(conf_thresh)
            dist.config.shadow_audit_rate = float(audit_rate)
        thread_id = st.text_input("Session Name", value="demo-desk")

        c_apply, c_reset = st.columns(2)
        if c_apply.button("Apply Settings", use_container_width=True):
            db_path = st.session_state.db_path
            chosen_mode = TeacherMode(teacher_choice)
            has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            if chosen_mode == TeacherMode.GEMINI and not has_key:
                st.session_state.key_warning = "No GEMINI_API_KEY found. Falling back safely to Auto mode."
                chosen_mode = TeacherMode.AUTO
            else:
                st.session_state.key_warning = ""
            engine.close()
            new_cfg = DistillerConfig(confidence_threshold=conf_thresh, shadow_audit_rate=audit_rate)
            st.session_state.engine = ZeroTaxEngine(
                db_path=db_path,
                teacher_mode=chosen_mode,
                config=new_cfg,
                seed=42,
                sleep_on_simulated=False,
            )
            st.session_state.ticket_log = []
            st.session_state.stream_pos = 0
            st.session_state.drift_pos = 0
            st.session_state.pending_interrupt = None
            st.session_state.pending_remaining = []
            st.session_state.last_probe = None
            st.session_state.active_action = "auto_200"
            st.rerun()

        if st.session_state.get("key_warning"):
            st.warning(st.session_state.key_warning)

        if c_reset.button("Reset All", use_container_width=True):
            _reset_session_state(engine)
            st.rerun()

        st.markdown(
            '<div class="zt-sidebar-meta">'
            f'SESSION // <strong>{html.escape(thread_id)}</strong><br>'
            f'NORMAL TICKETS // <strong>{st.session_state.stream_pos} / {len(dataset.train_stream)}</strong><br>'
            f'SCAM TEST // <strong>{st.session_state.drift_pos} / {len(dataset.drift_wave)}</strong>'
            "</div>",
            unsafe_allow_html=True,
        )

    metrics = engine.get_metrics(thread_id=thread_id)
    nodes = metrics["nodes"]

    st.markdown(
        '<div class="zt-header">'
        "<div>"
        '<div class="zt-title">ZERO-TAX // <span>Smart AI</span> Cost & Speed Engine</div>'
        '<div class="zt-subtitle">Automatically learns from Paid Cloud AI and shifts repetitive steps onto a Free 1ms Local AI - with live scam protection.</div>'
        "</div>"
        '<div class="zt-env-pill">'
        f'<span>MODE <strong>{"AUTO-PILOT" if auto_approve else "MANUAL GATE"}</strong></span>'
        f'<span>CLOUD <strong>{html.escape(engine.teacher.active_backend.value.upper())}</strong></span>'
        f'<span>SAFETY <strong>{int(conf_thresh * 100)}%</strong></span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        _render_stepper(
            nodes,
            has_interrupt=st.session_state.pending_interrupt is not None,
            has_drift_runs=st.session_state.drift_pos > 0,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(_render_kpi_strip(metrics), unsafe_allow_html=True)

    _render_hitl_gate(engine, thread_id, auto_approve, audit_rate)

    has_pending = st.session_state.pending_interrupt is not None
    train_done = st.session_state.stream_pos >= len(dataset.train_stream)
    drift_done = st.session_state.drift_pos >= len(dataset.drift_wave)

    if active_view == NAV_VIEWS[0]:
        st.markdown('<div class="zt-section-hdr">One-Click Actions</div>', unsafe_allow_html=True)
        act = st.session_state.active_action
        b_auto_type = "primary" if act == "auto_200" else "secondary"
        b_batch_type = "primary" if act == "batch_25" else "secondary"
        b_scam_type = "primary" if act == "scam_50" else "secondary"

        qa1, qa2, qa3, qa4 = st.columns(4)
        if qa1.button("Auto-Run 200 Customer Tickets", type=b_auto_type, use_container_width=True, disabled=has_pending or train_done):
            st.session_state.active_action = "auto_200"
            start = st.session_state.stream_pos
            batch = dataset.train_stream[start:]
            _stream_tickets(batch, thread_id, auto_approve, audit_rate)
            st.rerun()
        if qa2.button("Run Small Batch (25 Tickets)", type=b_batch_type, use_container_width=True, disabled=has_pending or train_done):
            st.session_state.active_action = "batch_25"
            start = st.session_state.stream_pos
            batch = dataset.train_stream[start : start + 25]
            _stream_tickets(batch, thread_id, auto_approve, audit_rate)
            st.rerun()
        if qa3.button("Simulate New Scam Attack (50)", type=b_scam_type, use_container_width=True, disabled=has_pending or drift_done):
            st.session_state.active_action = "scam_50"
            if st.session_state.stream_pos < 75:
                warm_batch = dataset.train_stream[st.session_state.stream_pos : 75]
                _stream_tickets(warm_batch, thread_id, True, audit_rate)
            drift_res = engine.inject_drift(
                dataset.drift_wave,
                thread_id=thread_id,
                audit_rate=1.0,
                auto_approve_repromotion=True,
            )
            st.session_state.drift_pos = len(dataset.drift_wave)
            for ticket, item in zip(dataset.drift_wave, drift_res["results"]):
                _record_result(item, ticket)
            st.rerun()
        if qa4.button("Reset & Start Over", type="secondary", use_container_width=True):
            _reset_session_state(engine)
            st.rerun()

        st.markdown('<div class="zt-section-hdr">How Each Ticket Is Processed (3 Tasks)</div>', unsafe_allow_html=True)
        nc1, nc2, nc3 = st.columns(3)
        nc1.markdown(
            _render_node_card(
                "TASK 1 // READ MESSAGE",
                "Detect Problem",
                "Reads complaint text and classifies problem (Damaged, Late, Wrong Size, Fraud, or Empty Box).",
                "intent_classifier",
                nodes["intent_classifier"],
                engine,
                False,
            ),
            unsafe_allow_html=True,
        )
        nc2.markdown(
            _render_node_card(
                "TASK 2 // APPLY POLICY",
                "Pick Refund Action",
                "Checks price, return history & delivery days to decide Instant Refund, Replacement, or Manual Check.",
                "policy_router",
                nodes["policy_router"],
                engine,
                False,
            ),
            unsafe_allow_html=True,
        )
        nc3.markdown(
            _render_node_card(
                "TASK 3 // WRITE RESPONSE",
                "Write Customer Email",
                "Writes a polite reply to the customer. Always kept on Paid Cloud AI for natural writing.",
                "reply_writer",
                nodes["reply_writer"],
                engine,
                True,
            ),
            unsafe_allow_html=True,
        )

        if st.session_state.ticket_log:
            _render_latest_spotlight(st.session_state.ticket_log)
            st.markdown('<div class="zt-section-hdr">Live Processed Customer Tickets (Latest 25)</div>', unsafe_allow_html=True)
            _render_ticket_table(st.session_state.ticket_log, limit=25)
            st.markdown('<div class="zt-section-hdr">Cost & Speed Telemetry</div>', unsafe_allow_html=True)
            _render_charts(st.session_state.ticket_log)
        else:
            st.markdown(
                '<div class="zt-empty-panel">'
                '<div class="zt-empty-title">Ready to Start</div>'
                '<div class="zt-empty-desc">Click "Auto-Run 200 Customer Tickets" above. You will see real customer complaints arrive, Paid Cloud AI train the Free Local AI automatically, and costs drop live.</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    elif active_view == NAV_VIEWS[1]:
        st.markdown('<div class="zt-section-hdr">Pick a Sample Complaint (or Write Your Own Below)</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        if p1.button("Example 1: Damaged Running Shoes", use_container_width=True):
            st.session_state.probe_preset = {
                "category": ProductCategory.FOOTWEAR.value,
                "price": 1899.0,
                "return_rate": 0.08,
                "days": 3,
                "text": "My running sneaker sole came unglued and peeled on day one.",
            }
            st.rerun()
        if p2.button("Example 2: Suspicious High-Value Refund", use_container_width=True):
            st.session_state.probe_preset = {
                "category": ProductCategory.ELECTRONICS.value,
                "price": 12499.0,
                "return_rate": 0.62,
                "days": 19,
                "text": "Serial number mismatch claimed by buyer after 19 days and high return ratio.",
            }
            st.rerun()
        if p3.button("Example 3: Empty Box Scam (New Fraud)", use_container_width=True):
            st.session_state.probe_preset = {
                "category": ProductCategory.ACCESSORIES.value,
                "price": 3499.0,
                "return_rate": 0.48,
                "days": 2,
                "text": "The sealed box arrived completely empty with no smart watch inside only bubble wrap.",
            }
            st.rerun()

        st.markdown('<div class="zt-section-hdr">Test Any Customer Ticket Live</div>', unsafe_allow_html=True)
        preset = st.session_state.probe_preset
        cats = [c.value for c in ProductCategory]
        cat_idx = cats.index(preset["category"]) if preset["category"] in cats else 0

        with st.form("single_ticket_form"):
            sc1, sc2, sc3, sc4 = st.columns(4)
            cat = sc1.selectbox("Product Category", cats, index=cat_idx)
            price = sc2.number_input("Item Price (Rs.)", min_value=0.0, value=float(preset["price"]), step=100.0)
            ret_rate = sc3.slider("Customer Past Return Rate", 0.0, 1.0, float(preset["return_rate"]), 0.01)
            days = sc4.number_input("Days Since Delivery", min_value=0, value=int(preset["days"]), step=1)
            text = st.text_area("Customer Complaint Message", value=str(preset["text"]), height=90)
            submitted = st.form_submit_button("Run Ticket Through 3 Tasks", type="primary", use_container_width=True)

        if submitted:
            probe = SupportTicket(
                ticket_id=f"PROBE-{len(st.session_state.ticket_log) + 1:04d}",
                category=ProductCategory(cat),
                item_price_inr=float(price),
                customer_return_rate=float(ret_rate),
                days_since_delivery=int(days),
                ticket_text=text,
            )
            res = engine.run_ticket(probe, thread_id=thread_id, auto_approve=auto_approve, audit_rate=audit_rate)
            _record_result(res, probe)
            st.session_state.last_probe = res
            if res.get("interrupted"):
                st.session_state.pending_interrupt = res["interrupt_payload"]
            st.rerun()

        if st.session_state.last_probe and st.session_state.last_probe.get("telemetry"):
            tel = st.session_state.last_probe["telemetry"]
            st.markdown('<div class="zt-section-hdr">How ZERO-TAX Handled This Ticket</div>', unsafe_allow_html=True)
            t1, t2, t3 = st.columns(3)
            for col, title, node_res in [
                (t1, "TASK 1 // DETECT PROBLEM", tel["intent_result"]),
                (t2, "TASK 2 // PICK REFUND ACTION", tel["policy_result"]),
                (t3, "TASK 3 // WRITE CUSTOMER EMAIL", tel["reply_result"]),
            ]:
                mode = node_res["execution_mode"]
                human_mode = _human_mode_label(mode).upper()
                b_cls = "zt-badge-cyan" if mode == ExecutionMode.SKLEARN_STUDENT.value else (
                    "zt-badge-amber" if mode == ExecutionMode.LOW_CONF_FALLBACK.value else "zt-badge-violet"
                )
                pred_val = "CUSTOMER REPLY GENERATED" if title.startswith("TASK 3") else html.escape(str(node_res["prediction"]))
                col.markdown(
                    '<div class="zt-node-card" style="min-height: 175px;">'
                    "<div>"
                    '<div class="zt-node-top">'
                    f'<div class="zt-node-index">{html.escape(title)}</div>'
                    f'<span class="zt-badge {b_cls}"><span class="zt-dot"></span>{html.escape(human_mode)}</span>'
                    "</div>"
                    f'<div class="zt-node-name">{pred_val}</div>'
                    "</div>"
                    '<div class="zt-stat-grid">'
                    f'<div class="zt-stat-box"><div class="zt-stat-lbl">CONFIDENCE</div><div class="zt-stat-val">{float(node_res["confidence"]) * 100:.1f}%</div></div>'
                    f'<div class="zt-stat-box"><div class="zt-stat-lbl">TIME / COST</div><div class="zt-stat-val">{float(node_res["latency_ms"]):.1f} ms | Rs. {float(node_res["cost_inr"]):.4f}</div></div>'
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="zt-reply-card"><strong>Email Reply Sent to Customer:</strong><br>{html.escape(str(tel["reply_text"]))}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Technical Details (JSON)"):
                st.json(tel)

    elif active_view == NAV_VIEWS[2]:
        st.markdown('<div class="zt-section-hdr">Spot-Check & Scam Protection Stats</div>', unsafe_allow_html=True)
        n1, n2 = nodes["intent_classifier"], nodes["policy_router"]
        total_audits = int(n1["shadow_audit_count"]) + int(n2["shadow_audit_count"])
        total_demotes = int(n1["demotion_count"]) + int(n2["demotion_count"])
        total_promos = int(n1["promotion_count"]) + int(n2["promotion_count"])
        n1_agr = f"{n1['rolling_shadow_agreement'] * 100:.1f}%" if n1["rolling_shadow_agreement"] is not None else "--"
        n2_agr = f"{n2['rolling_shadow_agreement'] * 100:.1f}%" if n2["rolling_shadow_agreement"] is not None else "--"
        st.markdown(
            '<div class="zt-kpi-grid">'
            f'<div class="zt-kpi-card"><div class="zt-kpi-label">SPOT-CHECKS RUN</div><div class="zt-kpi-value">{total_audits}</div><div class="zt-kpi-sub muted">Task 1: {n1["shadow_audit_count"]} | Task 2: {n2["shadow_audit_count"]}</div></div>'
            f'<div class="zt-kpi-card"><div class="zt-kpi-label">FREE AI ACCURACY MATCH</div><div class="zt-kpi-value">{n1_agr}</div><div class="zt-kpi-sub">Task 2 Match: {n2_agr}</div></div>'
            f'<div class="zt-kpi-card"><div class="zt-kpi-label">SCAM SHIELD ALERTS</div><div class="zt-kpi-value">{total_demotes}</div><div class="zt-kpi-sub muted">Auto-triggers below 90%</div></div>'
            f'<div class="zt-kpi-card violet"><div class="zt-kpi-label">AI UPGRADES / RECOVERY</div><div class="zt-kpi-value">{total_promos}</div><div class="zt-kpi-sub muted">Task 1 v{n1["model_version"]} | Task 2 v{n2["model_version"]}</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.ticket_log:
            _render_charts(st.session_state.ticket_log)
            st.markdown('<div class="zt-section-hdr">All Processed Customer Tickets (Latest 50)</div>', unsafe_allow_html=True)
            _render_ticket_table(st.session_state.ticket_log, limit=50)
        else:
            st.markdown(
                '<div class="zt-empty-panel">'
                '<div class="zt-empty-title">No Charts Yet</div>'
                '<div class="zt-empty-desc">Go to "1. Live Dashboard" and click "Auto-Run 200 Customer Tickets" to populate cost and speed charts.</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    else:
        st.markdown('<div class="zt-section-hdr">Saved Checkpoints (SQLite History)</div>', unsafe_allow_html=True)
        cps = engine.list_checkpoints(thread_id=thread_id, limit=50)
        if cps:
            cp_df = pd.DataFrame([c.model_dump() for c in cps])
            st.dataframe(cp_df, use_container_width=True, hide_index=True)

            st.markdown('<div class="zt-section-hdr">Undo / Rewind to an Earlier Savepoint</div>', unsafe_allow_html=True)
            rc1, rc2 = st.columns(2)
            selected_cp = rc1.selectbox("Pick Savepoint ID", [c.checkpoint_id for c in cps])
            fork_id = rc2.text_input("Optional: Copy Into a New Session Name (blank = undo current)", value="")
            restore_pre = st.toggle("Reset Free AI Back to Cloud AI After Undo", value=True)
            if st.button("Rewind to Selected Savepoint", type="primary"):
                summary = engine.rewind_to_checkpoint(
                    thread_id=thread_id,
                    checkpoint_id=selected_cp,
                    fork_thread_id=fork_id.strip() or None,
                    restore_pre_promotion_state=restore_pre,
                )
                if not fork_id.strip():
                    st.session_state.ticket_log = []
                    st.session_state.pending_interrupt = None
                    st.session_state.pending_remaining = []
                st.success(f"Restored savepoint {summary.checkpoint_id} on session '{summary.thread_id}'.")
        else:
            st.markdown(
                '<div class="zt-empty-panel">'
                '<div class="zt-empty-title">No Savepoints Recorded Yet</div>'
                f'<div class="zt-empty-desc">Run tickets on session "{html.escape(thread_id)}" first to inspect savepoints and rewind state.</div>'
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="zt-section-hdr">Under the Hood: LangGraph Flowchart</div>', unsafe_allow_html=True)
        st.code(engine.get_graph_mermaid(), language="mermaid")


if __name__ == "__main__":
    main()
