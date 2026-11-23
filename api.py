"""FastAPI service exposing the ZERO-TAX LangGraph engine, HITL promotion, drift, and time-travel."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, StrictBool

from core.dataset import generate_dataset
from core.distiller import DistillerConfig
from core.graph import ZeroTaxEngine
from core.models import SupportTicket, TeacherMode


class RunTicketRequest(BaseModel):
    ticket: SupportTicket
    thread_id: str = "default"
    auto_approve: bool = False
    audit_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class StreamBatchRequest(BaseModel):
    count: int = Field(default=50, ge=0)
    thread_id: str = "default"
    auto_approve: bool = False
    audit_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ApprovePromotionRequest(BaseModel):
    thread_id: str = "default"
    approved: StrictBool


class InjectDriftRequest(BaseModel):
    count: int = Field(default=30, ge=0)
    thread_id: str = "default"
    audit_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    auto_approve_repromotion: bool = False


class RewindRequest(BaseModel):
    thread_id: str
    checkpoint_id: str
    fork_thread_id: Optional[str] = None
    restore_pre_promotion_state: bool = False


def create_app(
    db_path: Optional[str] = None,
    teacher_mode: TeacherMode = TeacherMode.AUTO,
    config: Optional[DistillerConfig] = None,
    seed: int = 42,
    sleep_on_simulated: bool = False,
) -> FastAPI:
    resolved_db = db_path or os.getenv("ZEROTAX_DB_PATH", "data/checkpoints.db")
    engine = ZeroTaxEngine(
        db_path=resolved_db,
        teacher_mode=teacher_mode,
        config=config,
        seed=seed,
        sleep_on_simulated=sleep_on_simulated,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        engine.close()

    application = FastAPI(
        title="ZERO-TAX LangGraph Agent API",
        version="1.0.0",
        description="Online LLM-to-scikit-learn node distillation with Send() shadow audits, HITL interrupt() gates, and SqliteSaver time-travel.",
        lifespan=lifespan,
    )
    application.state.engine = engine
    application.state.stream_cursors: dict[str, int] = {}
    application.state.drift_cursors: dict[str, int] = {}
    application.state.seed = seed

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "teacher_backend": engine.teacher.active_backend.value,
            "teacher_model": engine.teacher.teacher_model,
        }

    @application.get("/graph-mermaid")
    def graph_mermaid() -> dict[str, str]:
        return {"mermaid": engine.get_graph_mermaid()}

    @application.post("/reset")
    def reset_engine() -> dict[str, str]:
        engine.reset()
        application.state.stream_cursors.clear()
        application.state.drift_cursors.clear()
        return {"status": "reset"}

    @application.post("/run")
    def run_ticket(req: RunTicketRequest) -> dict[str, Any]:
        return engine.run_ticket(
            req.ticket,
            thread_id=req.thread_id,
            auto_approve=req.auto_approve,
            audit_rate=req.audit_rate,
        )

    @application.post("/stream-batch")
    def stream_batch(req: StreamBatchRequest) -> dict[str, Any]:
        offset = application.state.stream_cursors.get(req.thread_id, 0)
        needed = max(200, offset + req.count)
        bundle = generate_dataset(seed=seed, main_count=needed, holdout_count=50, drift_count=50)
        batch = bundle.train_stream[offset : offset + req.count]
        res = engine.stream_batch(
            batch,
            thread_id=req.thread_id,
            auto_approve=req.auto_approve,
            audit_rate=req.audit_rate,
        )
        application.state.stream_cursors[req.thread_id] = offset + int(res.get("processed_count", 0))
        return res

    @application.post("/approve-promotion")
    def approve_promotion(req: ApprovePromotionRequest) -> dict[str, Any]:
        try:
            return engine.approve_promotion(req.thread_id, approved=req.approved)
        except LookupError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/inject-drift")
    def inject_drift(req: InjectDriftRequest) -> dict[str, Any]:
        offset = application.state.drift_cursors.get(req.thread_id, 0)
        needed = max(50, offset + req.count)
        bundle = generate_dataset(seed=seed, main_count=50, holdout_count=10, drift_count=needed)
        batch = bundle.drift_wave[offset : offset + req.count]
        res = engine.inject_drift(
            batch,
            thread_id=req.thread_id,
            audit_rate=req.audit_rate,
            auto_approve_repromotion=req.auto_approve_repromotion,
        )
        application.state.drift_cursors[req.thread_id] = offset + int(res.get("processed_count", 0))
        return res

    @application.get("/metrics")
    def get_metrics(thread_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
        return engine.get_metrics(thread_id=thread_id)

    @application.get("/checkpoints/{thread_id}")
    def list_checkpoints(
        thread_id: str,
        limit: Optional[int] = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        cps = engine.list_checkpoints(thread_id=thread_id, limit=limit)
        return {
            "thread_id": thread_id,
            "checkpoints": [c.model_dump() for c in cps],
        }

    @application.post("/rewind")
    def rewind_checkpoint(req: RewindRequest) -> dict[str, Any]:
        try:
            summary = engine.rewind_to_checkpoint(
                thread_id=req.thread_id,
                checkpoint_id=req.checkpoint_id,
                fork_thread_id=req.fork_thread_id,
                restore_pre_promotion_state=req.restore_pre_promotion_state,
            )
            return {
                "status": "rewound",
                "checkpoint": summary.model_dump(),
                "node_statuses": engine._current_node_statuses(),
            }
        except (KeyError, LookupError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return application


app = create_app()
