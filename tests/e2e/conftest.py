"""Shared E2E fixtures for ZERO-TAX.

Strict hygiene rules:
- Module top level imports ONLY Python stdlib, pytest, and sibling e2e_helpers.
- All core.*, api, benchmark, pydantic, and fastapi imports are lazy (inside fixtures).
- Zero mocks, zero dynamic skip guards, zero silent skips.
- Every test uses tmp_path-isolated SQLite DBs; repo data/checkpoints.db is never touched.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import pytest

E2E_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = E2E_DIR.parents[1]
if str(E2E_DIR) not in sys.path:
    sys.path.insert(0, str(E2E_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2e_helpers import (  # noqa: E402, F401
    enum_str,
    extract_checkpoints,
    get_all_node_metrics,
    get_interrupt_payload,
    get_node_metrics,
    get_node_result,
    get_reply_text,
    get_telemetry,
    isolated_subprocess_env,
    make_ticket,
    stream_until_complete,
    ticket_to_api_payload,
    to_dict,
)


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force 100% offline deterministic execution by clearing external LLM API keys."""
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def dataset_bundle() -> Any:
    """Session-scoped deterministic synthetic Indian e-commerce dataset (300 tickets)."""
    from core.dataset import generate_dataset

    return generate_dataset(seed=42, main_count=200, holdout_count=50, drift_count=50)


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Isolated per-test SQLite checkpoint path inside pytest tmp_path."""
    return tmp_path / "e2e_checkpoints.db"


def _close_engine_handles(eng: Any) -> None:
    """Release SQLite connections cleanly on Windows to avoid tmp_path PermissionError."""
    close_fn = getattr(eng, "close", None)
    if callable(close_fn):
        close_fn()
    checkpointer = getattr(eng, "checkpointer", None)
    if checkpointer is not None:
        cp_close = getattr(checkpointer, "close", None)
        if callable(cp_close):
            cp_close()
        conn = getattr(checkpointer, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


@pytest.fixture
def make_engine(tmp_path: Path):
    """Factory fixture creating tmp_path-isolated ZeroTaxEngine instances."""
    created: list[Any] = []

    def _factory(
        *,
        db_name: str = "e2e_checkpoints.db",
        teacher_mode: Any = None,
        config: Any = None,
        seed: int = 42,
        sleep_on_simulated: bool = False,
    ) -> Any:
        from core.graph import ZeroTaxEngine
        from core.models import TeacherMode

        eng = ZeroTaxEngine(
            db_path=str(tmp_path / db_name),
            teacher_mode=teacher_mode if teacher_mode is not None else TeacherMode.SIMULATED,
            config=config,
            seed=seed,
            sleep_on_simulated=sleep_on_simulated,
        )
        created.append(eng)
        return eng

    yield _factory

    for eng in reversed(created):
        _close_engine_handles(eng)
    gc.collect()


@pytest.fixture
def engine(make_engine) -> Any:
    """Default tmp_path-isolated ZeroTaxEngine running in offline SIMULATED mode."""
    return make_engine()


@pytest.fixture
def make_api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Factory fixture creating an isolated FastAPI TestClient backed by tmp_path SQLite."""
    clients: list[Any] = []
    apps: list[tuple[Any, Any, bool]] = []

    def _factory(
        *,
        db_name: str = "api_e2e.db",
        seed: int = 42,
        config: Any = None,
    ) -> Any:
        db_file = str(tmp_path / db_name)
        monkeypatch.setenv("ZEROTAX_DB_PATH", db_file)

        import api
        from core.models import TeacherMode
        from fastapi.testclient import TestClient

        if hasattr(api, "create_app"):
            app = api.create_app(
                db_path=db_file,
                teacher_mode=TeacherMode.SIMULATED,
                config=config,
                seed=seed,
                sleep_on_simulated=False,
            )
            used_create_app = True
        else:
            app = api.app
            used_create_app = False
            if hasattr(api, "engine") and hasattr(api.engine, "reset"):
                api.engine.reset()
        apps.append((api, app, used_create_app))
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _factory

    for client in reversed(clients):
        client.__exit__(None, None, None)
    for _api_mod, app, used_create_app in reversed(apps):
        if used_create_app:
            state_eng = getattr(getattr(app, "state", None), "engine", None)
            if state_eng is not None:
                _close_engine_handles(state_eng)
    gc.collect()


@pytest.fixture
def api_client(make_api_client) -> Any:
    """Default isolated FastAPI TestClient."""
    return make_api_client()
