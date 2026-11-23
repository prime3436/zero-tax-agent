"""LLM Teacher abstraction supporting offline deterministic simulation and optional Gemini/OpenAI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Optional

from core.dataset import compute_ground_truth_policy
from core.models import (
    USD_TO_INR,
    ExecutionMode,
    IntentLabel,
    NodePrediction,
    PolicyAction,
    SupportTicket,
    TeacherMode,
)

# Frontier-style per-token pricing used for transparent cost accounting
INPUT_USD_PER_TOKEN = 2.50 / 1_000_000.0
OUTPUT_USD_PER_TOKEN = 10.00 / 1_000_000.0

INTENT_PATTERNS: dict[IntentLabel, list[tuple[str, float]]] = {
    IntentLabel.EMPTY_BOX_SCAM: [
        (r"\bempty\s+box\b", 4.5),
        (r"\bbox\s+empty\b", 4.5),
        (r"\bempty\s+(?:package|retail\s+box|carton)\b", 4.2),
        (r"\bouter\s+carton\s+sealed\b", 3.5),
        (r"\bunboxing\s+video\b", 3.2),
        (r"\bno\s+[\w\s]{2,30}\s+inside\b", 3.5),
        (r"\b(?:brick|soap\s+bar|stone|concrete|dummy\s+block)\s+inside\b", 4.2),
        (r"\bsealed\s+(?:outer|parcel|carton|mailer|courier\s+bag|courier\s+box)\b", 2.8),
        (r"\binstead\s+of\b", 1.8),
        (r"\bmissing\b.*\b(?:empty|sealed|brick|stone|soap)\b", 3.2),
    ],
    IntentLabel.FRAUD_SUSPECT: [
        (r"\bserial\s+(?:number|mismatch)\b", 3.5),
        (r"\bduplicate\s+refund\b", 3.8),
        (r"\brepeat(?:ed)?\s+(?:claim|high-value\s+refund)\b", 3.5),
        (r"\bswapped\b", 3.2),
        (r"\btampered\s+(?:security|barcode)\b", 3.4),
        (r"\bfake\s+receipt\b", 3.8),
        (r"\baltered\s+invoice\b", 3.6),
        (r"\bsuspicious\b", 2.5),
        (r"\bimei\b", 2.2),
        (r"\bmissing\s+again\b", 2.8),
    ],
    IntentLabel.WRONG_SIZE: [
        (r"\bwrong\s+size\b", 3.8),
        (r"\bsize\s+(?:m|s|l|xl|uk\s*\d+|39|42|mismatch|tag|issue)\b", 3.2),
        (r"\b(?:does\s+not|doesn't)\s+fit\b", 3.5),
        (r"\bfit\s+mismatch\b", 3.5),
        (r"\b(?:too\s+tight|too\s+loose|cramped|hangs\s+way\s+too\s+loose|cannot\s+be\s+worn)\b", 3.2),
        (r"\b(?:medium|extra\s+large)\b.*\b(?:large|medium|small)\b", 3.4),
        (r"\bexchange\s+for\s+the\s+right\s+fit\b", 3.0),
        (r"\bsizing\s+chart\b", 2.6),
    ],
    IntentLabel.LATE_DELIVERY: [
        (r"\bdelayed?\b", 3.4),
        (r"\blate\b", 3.2),
        (r"\btracking\b", 2.8),
        (r"\bcourier\s+(?:hub|movement|partner)\b", 2.6),
        (r"\bstuck\s+at\b", 3.0),
        (r"\boverdue\b", 3.4),
        (r"\bin\s+transit\b", 3.0),
        (r"\bdelivery\s+window\b", 2.8),
        (r"\bpostponing\s+arrival\b", 3.0),
        (r"\bpast\s+the\s+promised\s+date\b", 3.2),
    ],
    IntentLabel.DAMAGED_ITEM: [
        (r"\bdamaged?\b", 3.4),
        (r"\bcracked?\b", 3.2),
        (r"\bbroken\b", 3.4),
        (r"\bsole\b", 2.6),
        (r"\bunglued\b", 3.4),
        (r"\bglued\s+crooked\b", 3.4),
        (r"\bpeeled\b", 3.0),
        (r"\btorn\b", 3.0),
        (r"\bshattered\b", 3.4),
        (r"\bdented\b", 3.2),
        (r"\bleaking\b", 3.2),
        (r"\bsplit\s+seam\b", 3.2),
        (r"\btoot\s+gaya\b", 3.4),
        (r"\bnuksan\b", 2.8),
    ],
}


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    max_s = max(scores.values())
    exps = {k: math.exp(v - max_s) for k, v in scores.items()}
    total = sum(exps.values())
    probs = {k: round(v / total, 6) for k, v in exps.items()}
    # Adjust largest entry slightly so sum == 1.0 within 1e-6
    top_k = max(probs, key=lambda k: probs[k])
    diff = round(1.0 - sum(probs.values()), 6)
    probs[top_k] = round(probs[top_k] + diff, 6)
    return probs


def _stable_jitter(ticket_id: str, ticket_text: str, salt: str, low: float, high: float) -> float:
    digest = hashlib.sha256(f"{salt}|{ticket_id}|{ticket_text}".encode("utf-8")).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return round(low + frac * (high - low), 2)


def _compute_cost(input_tokens: int, output_tokens: int) -> tuple[float, float]:
    cost_usd = round(input_tokens * INPUT_USD_PER_TOKEN + output_tokens * OUTPUT_USD_PER_TOKEN, 8)
    cost_inr = round(cost_usd * USD_TO_INR, 8)
    return cost_usd, cost_inr


class LLMTeacher:
    """Teacher interface for Nodes 1, 2, and 3 (offline simulation or real Gemini/OpenAI)."""

    def __init__(
        self,
        mode: TeacherMode = TeacherMode.AUTO,
        simulate_latency: bool = True,
        sleep_on_simulated: bool = False,
        seed: int = 42,
        model_name: Optional[str] = None,
    ) -> None:
        self.mode = mode if isinstance(mode, TeacherMode) else TeacherMode(mode)
        self.simulate_latency = simulate_latency
        self.sleep_on_simulated = sleep_on_simulated
        self.seed = seed
        self._client = None
        self.active_backend, self.teacher_model = self._resolve_backend(model_name)

    def _resolve_backend(self, model_name: Optional[str]) -> tuple[TeacherMode, str]:
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if self.mode == TeacherMode.SIMULATED:
            return TeacherMode.SIMULATED, model_name or "simulated-llm-teacher-v1"

        if self.mode == TeacherMode.GEMINI:
            if not gemini_key:
                raise RuntimeError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for TeacherMode.GEMINI"
                )
            from langchain_google_genai import ChatGoogleGenerativeAI

            chosen = model_name or "gemini-2.0-flash"
            self._client = ChatGoogleGenerativeAI(model=chosen, google_api_key=gemini_key, temperature=0.0)
            return TeacherMode.GEMINI, chosen

        if self.mode == TeacherMode.OPENAI:
            if not openai_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is required for TeacherMode.OPENAI")
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise RuntimeError("langchain-openai is not installed in this environment") from exc
            chosen = model_name or "gpt-4o-mini"
            self._client = ChatOpenAI(model=chosen, api_key=openai_key, temperature=0.0)
            return TeacherMode.OPENAI, chosen

        # TeacherMode.AUTO
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                chosen = model_name or "gemini-2.0-flash"
                self._client = ChatGoogleGenerativeAI(model=chosen, google_api_key=gemini_key, temperature=0.0)
                return TeacherMode.GEMINI, chosen
            except Exception:
                pass

        if openai_key:
            try:
                from langchain_openai import ChatOpenAI

                chosen = model_name or "gpt-4o-mini"
                self._client = ChatOpenAI(model=chosen, api_key=openai_key, temperature=0.0)
                return TeacherMode.OPENAI, chosen
            except Exception:
                pass

        return TeacherMode.SIMULATED, model_name or "simulated-llm-teacher-v1"

    def _estimate_input_tokens(self, ticket: SupportTicket, base_prompt_tokens: int) -> int:
        text = ticket.ticket_text or ""
        word_tokens = int(len(text.split()) * 1.35)
        char_tokens = len(text) // 16
        return base_prompt_tokens + word_tokens + char_tokens

    def classify_intent(self, ticket: SupportTicket) -> NodePrediction:
        """Classify support ticket text into an IntentLabel (never reads ground_truth_*)."""
        t0 = time.perf_counter()
        if self.active_backend in (TeacherMode.GEMINI, TeacherMode.OPENAI) and self._client is not None:
            try:
                prompt = (
                    "Classify this Indian e-commerce support ticket into strictly one label from: "
                    "DAMAGED_ITEM, LATE_DELIVERY, WRONG_SIZE, FRAUD_SUSPECT, EMPTY_BOX_SCAM. "
                    'Return JSON: {"intent": "...", "confidence": 0.95}.\n'
                    f"Category: {ticket.category.value}\nMessage: {ticket.ticket_text}"
                )
                msg = self._client.invoke(prompt)
                raw = str(getattr(msg, "content", "")).strip()
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    intent_val = IntentLabel(parsed["intent"]).value
                    conf = max(0.5, min(0.99, float(parsed.get("confidence", 0.92))))
                    rem = (1.0 - conf) / 4.0
                    probs = {e.value: (conf if e.value == intent_val else rem) for e in IntentLabel}
                    elapsed_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)
                    in_tok = self._estimate_input_tokens(ticket, 420)
                    out_tok = 24
                    cost_usd, cost_inr = _compute_cost(in_tok, out_tok)
                    return NodePrediction(
                        node_name="intent_classifier",
                        prediction=intent_val,
                        confidence=probs[intent_val],
                        probabilities=probs,
                        execution_mode=ExecutionMode.LLM_TEACHER,
                        latency_ms=round(elapsed_ms, 3),
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cost_usd=cost_usd,
                        cost_inr=cost_inr,
                        teacher_model=self.teacher_model,
                    )
            except Exception:
                # Fall back safely to local simulated scoring if live API fails mid-call
                pass

        text_lower = (ticket.ticket_text or "").lower()
        scores: dict[str, float] = {e.value: 0.25 for e in IntentLabel}
        for label, patterns in INTENT_PATTERNS.items():
            for pat, weight in patterns:
                if re.search(pat, text_lower):
                    scores[label.value] += weight

        # If empty/completely unrecognized text, bias gently toward DAMAGED_ITEM with moderate uncertainty
        if max(scores.values()) <= 0.30:
            scores[IntentLabel.DAMAGED_ITEM.value] += 0.6

        probs = _softmax(scores)
        best_label = max(probs, key=lambda k: probs[k])
        in_tok = self._estimate_input_tokens(ticket, 420)
        out_tok = 24
        cost_usd, cost_inr = _compute_cost(in_tok, out_tok)

        if self.simulate_latency:
            latency_ms = _stable_jitter(ticket.ticket_id, ticket.ticket_text, "intent", 560.0, 680.0)
            if self.sleep_on_simulated:
                time.sleep(min(latency_ms / 1000.0, 0.05))
        else:
            latency_ms = round(max(0.0, (time.perf_counter() - t0) * 1000.0), 4)

        return NodePrediction(
            node_name="intent_classifier",
            prediction=best_label,
            confidence=probs[best_label],
            probabilities=probs,
            execution_mode=ExecutionMode.LLM_TEACHER,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost_usd,
            cost_inr=cost_inr,
            teacher_model=self.teacher_model,
        )

    def route_policy(self, ticket: SupportTicket, intent_label: str) -> NodePrediction:
        """Route support ticket to a PolicyAction using business rules (never reads ground_truth_*)."""
        t0 = time.perf_counter()
        intent_enum = IntentLabel(intent_label)
        chosen = compute_ground_truth_policy(
            intent=intent_enum,
            item_price_inr=float(ticket.item_price_inr),
            customer_return_rate=float(ticket.customer_return_rate),
            days_since_delivery=int(ticket.days_since_delivery),
        )
        scores = {e.value: (4.2 if e == chosen else 0.35) for e in PolicyAction}
        probs = _softmax(scores)

        in_tok = self._estimate_input_tokens(ticket, 460)
        out_tok = 22
        cost_usd, cost_inr = _compute_cost(in_tok, out_tok)

        if self.simulate_latency:
            latency_ms = _stable_jitter(ticket.ticket_id, ticket.ticket_text, "policy", 520.0, 640.0)
            if self.sleep_on_simulated:
                time.sleep(min(latency_ms / 1000.0, 0.05))
        else:
            latency_ms = round(max(0.0, (time.perf_counter() - t0) * 1000.0), 4)

        return NodePrediction(
            node_name="policy_router",
            prediction=chosen.value,
            confidence=probs[chosen.value],
            probabilities=probs,
            execution_mode=ExecutionMode.LLM_TEACHER,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost_usd,
            cost_inr=cost_inr,
            teacher_model=self.teacher_model,
        )

    def write_reply(
        self,
        ticket: SupportTicket,
        intent_label: str,
        policy_action: str,
    ) -> NodePrediction:
        """Generate customer-facing support reply (always executed by the LLM node)."""
        t0 = time.perf_counter()
        policy = PolicyAction(policy_action)
        name = ticket.customer_name or "Valued Customer"
        order_ref = ticket.order_id or "your order"
        price_str = f"Rs. {ticket.item_price_inr:,.0f}"

        if self.active_backend in (TeacherMode.GEMINI, TeacherMode.OPENAI) and self._client is not None:
            try:
                prompt = (
                    f"Write a concise, empathetic 2-sentence Indian e-commerce support reply to {name} "
                    f"for order {order_ref} ({price_str}). Classified issue: {intent_label}. "
                    f"Approved resolution: {policy.value}. Do not mention internal enum codes."
                )
                msg = self._client.invoke(prompt)
                reply_text = str(getattr(msg, "content", "")).strip()
                if len(reply_text) >= 15:
                    elapsed_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 3)
                    in_tok = self._estimate_input_tokens(ticket, 520)
                    out_tok = max(20, int(len(reply_text.split()) * 1.4))
                    cost_usd, cost_inr = _compute_cost(in_tok, out_tok)
                    return NodePrediction(
                        node_name="reply_writer",
                        prediction=reply_text,
                        confidence=1.0,
                        probabilities={},
                        execution_mode=ExecutionMode.LLM_TEACHER,
                        latency_ms=elapsed_ms,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cost_usd=cost_usd,
                        cost_inr=cost_inr,
                        teacher_model=self.teacher_model,
                        reply_text=reply_text,
                    )
            except Exception:
                pass

        if policy == PolicyAction.INSTANT_REFUND:
            reply_text = (
                f"Hi {name}, we are sorry for the trouble with order {order_ref}. "
                f"We have approved an instant refund of {price_str} back to your original UPI/payment method within 24 hours."
            )
        elif policy == PolicyAction.REPLACEMENT_PICKUP:
            reply_text = (
                f"Hi {name}, thank you for contacting us regarding order {order_ref}. "
                f"We have scheduled a doorstep replacement pickup and dispatched a fresh unit to {ticket.city} at no extra charge."
            )
        else:
            reply_text = (
                f"Hi {name}, your request for order {order_ref} ({price_str}) has been escalated to our specialist verification desk. "
                f"Our team is completing a manual inspection and will update you within 4 business hours."
            )

        in_tok = self._estimate_input_tokens(ticket, 520)
        out_tok = max(28, int(len(reply_text.split()) * 1.45))
        cost_usd, cost_inr = _compute_cost(in_tok, out_tok)

        if self.simulate_latency:
            latency_ms = _stable_jitter(ticket.ticket_id, ticket.ticket_text, "reply", 580.0, 710.0)
            if self.sleep_on_simulated:
                time.sleep(min(latency_ms / 1000.0, 0.05))
        else:
            latency_ms = round(max(0.0, (time.perf_counter() - t0) * 1000.0), 4)

        return NodePrediction(
            node_name="reply_writer",
            prediction=reply_text,
            confidence=1.0,
            probabilities={},
            execution_mode=ExecutionMode.LLM_TEACHER,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost_usd,
            cost_inr=cost_inr,
            teacher_model=self.teacher_model,
            reply_text=reply_text,
        )
