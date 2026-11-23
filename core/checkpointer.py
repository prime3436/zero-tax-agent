"""Windows-safe stdlib sqlite3 LangGraph BaseCheckpointSaver with WAL mode and time-travel."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from core.models import CheckpointSummary


class SqliteSaver(BaseCheckpointSaver[str]):
    """Persistent SQLite checkpointer matching LangGraph's BaseCheckpointSaver protocol."""

    def __init__(self, db_path: str = "data/checkpoints.db") -> None:
        super().__init__()
        p = Path(db_path)
        if p.exists() and p.is_dir():
            raise ValueError(f"db_path points to an existing directory, not a SQLite file: {db_path}")
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(p)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._setup_schema()

    def _setup_schema(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA busy_timeout=5000;")
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB NOT NULL,
                    metadata BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );

                CREATE TABLE IF NOT EXISTS blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    type TEXT NOT NULL,
                    blob BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                );

                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    value BLOB,
                    task_path TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );
                """
            )
            self.conn.commit()

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if not versions:
            return values
        for k, v in versions.items():
            row = self.conn.execute(
                """
                SELECT type, blob FROM blobs
                WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?
                """,
                (thread_id, checkpoint_ns, k, str(v)),
            ).fetchone()
            if row is not None:
                dtype, blob = row
                if dtype != "empty":
                    values[k] = self.serde.loads_typed((dtype, blob))
        return values

    def _load_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        rows = self.conn.execute(
            """
            SELECT task_id, channel, type, value
            FROM writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ORDER BY task_id, idx
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        return [
            (task_id, channel, self.serde.loads_typed((dtype, value)))
            for task_id, channel, dtype, value in rows
        ]

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        now_iso = datetime.now(timezone.utc).isoformat()

        cp_type, cp_blob = self.serde.dumps_typed(c)
        meta_obj = get_checkpoint_metadata(config, metadata)
        meta_type, meta_blob = self.serde.dumps_typed(meta_obj)
        # Store metadata serialized with serde; type is preserved inside blobs/metadata
        _ = meta_type

        with self._lock:
            for k, v in new_versions.items():
                if k in values:
                    b_type, b_data = self.serde.dumps_typed(values[k])
                else:
                    b_type, b_data = ("empty", b"")
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO blobs (thread_id, checkpoint_ns, channel, version, type, blob)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, checkpoint_ns, k, str(v), b_type, b_data),
                )

            self.conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    parent_checkpoint_id,
                    cp_type,
                    cp_blob,
                    meta_blob,
                    now_iso,
                ),
            )
            self.conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._lock:
            for idx, (channel, value) in enumerate(writes):
                mapped_idx = WRITES_IDX_MAP.get(channel, idx)
                if mapped_idx >= 0:
                    existing = self.conn.execute(
                        """
                        SELECT 1 FROM writes
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? AND task_id = ? AND idx = ?
                        """,
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, mapped_idx),
                    ).fetchone()
                    if existing is not None:
                        continue

                w_type, w_blob = self.serde.dumps_typed(value)
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, task_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        mapped_idx,
                        channel,
                        w_type,
                        w_blob,
                        task_path,
                    ),
                )
            self.conn.commit()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self._lock:
            if checkpoint_id:
                row = self.conn.execute(
                    """
                    SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = self.conn.execute(
                    """
                    SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()

            if row is None:
                return None

            cp_id, parent_cp_id, cp_type, cp_blob, meta_blob = row
            cp_dict: Checkpoint = self.serde.loads_typed((cp_type, cp_blob))
            cp_dict["channel_values"] = self._load_blobs(
                thread_id,
                checkpoint_ns,
                cp_dict.get("channel_versions", {}),
            )
            meta_dict = self.serde.loads_typed((cp_type, meta_blob))
            pending_writes = self._load_writes(thread_id, checkpoint_ns, cp_id)

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cp_id,
                    }
                },
                checkpoint=cp_dict,
                metadata=meta_dict,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_cp_id,
                        }
                    }
                    if parent_cp_id
                    else None
                ),
                pending_writes=pending_writes,
            )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        query = """
            SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
            FROM checkpoints
        """
        clauses: list[str] = []
        params: list[Any] = []

        if config is not None:
            clauses.append("thread_id = ?")
            params.append(config["configurable"]["thread_id"])
            if "checkpoint_ns" in config["configurable"]:
                clauses.append("checkpoint_ns = ?")
                params.append(config["configurable"]["checkpoint_ns"])
            if cfg_cp_id := get_checkpoint_id(config):
                clauses.append("checkpoint_id = ?")
                params.append(cfg_cp_id)

        if before is not None and (before_id := get_checkpoint_id(before)):
            clauses.append("checkpoint_id < ?")
            params.append(before_id)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY checkpoint_id DESC"

        with self._lock:
            rows = self.conn.execute(query, params).fetchall()

        yielded = 0
        for thread_id, checkpoint_ns, cp_id, parent_cp_id, cp_type, cp_blob, meta_blob in rows:
            if limit is not None and yielded >= limit:
                break
            meta_dict = self.serde.loads_typed((cp_type, meta_blob))
            if filter and not all(meta_dict.get(k) == v for k, v in filter.items()):
                continue

            with self._lock:
                cp_dict: Checkpoint = self.serde.loads_typed((cp_type, cp_blob))
                cp_dict["channel_values"] = self._load_blobs(
                    thread_id,
                    checkpoint_ns,
                    cp_dict.get("channel_versions", {}),
                )
                pending_writes = self._load_writes(thread_id, checkpoint_ns, cp_id)

            yielded += 1
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cp_id,
                    }
                },
                checkpoint=cp_dict,
                metadata=meta_dict,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_cp_id,
                        }
                    }
                    if parent_cp_id
                    else None
                ),
                pending_writes=pending_writes,
            )

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            self.conn.execute("DELETE FROM blobs WHERE thread_id = ?", (thread_id,))
            self.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            self.conn.commit()

    def clear_all(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM writes")
            self.conn.execute("DELETE FROM blobs")
            self.conn.execute("DELETE FROM checkpoints")
            self.conn.commit()

    def list_summaries(
        self,
        thread_id: str,
        limit: Optional[int] = None,
    ) -> list[CheckpointSummary]:
        """Fast lightweight checkpoint summary list ordered newest-first."""
        with self._lock:
            sql = """
                SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ''
                ORDER BY checkpoint_id DESC
            """
            params: list[Any] = [thread_id]
            if limit is not None:
                sql += " LIMIT ?"
                params.append(int(limit))
            rows = self.conn.execute(sql, params).fetchall()

            summaries: list[CheckpointSummary] = []
            for cp_id, parent_id, cp_type, cp_blob, meta_blob, created_at in rows:
                meta = self.serde.loads_typed((cp_type, meta_blob))
                step = meta.get("step") if isinstance(meta, dict) else None
                cp_dict = self.serde.loads_typed((cp_type, cp_blob))
                versions = cp_dict.get("channel_versions", {})

                ticket_id: Optional[str] = None
                node_statuses: dict[str, str] = {}
                if "ticket" in versions:
                    t_row = self.conn.execute(
                        """
                        SELECT type, blob FROM blobs
                        WHERE thread_id = ? AND checkpoint_ns = '' AND channel = 'ticket' AND version = ?
                        """,
                        (thread_id, str(versions["ticket"])),
                    ).fetchone()
                    if t_row and t_row[0] != "empty":
                        t_obj = self.serde.loads_typed((t_row[0], t_row[1]))
                        if isinstance(t_obj, dict):
                            ticket_id = t_obj.get("ticket_id")
                if "node_statuses" in versions:
                    ns_row = self.conn.execute(
                        """
                        SELECT type, blob FROM blobs
                        WHERE thread_id = ? AND checkpoint_ns = '' AND channel = 'node_statuses' AND version = ?
                        """,
                        (thread_id, str(versions["node_statuses"])),
                    ).fetchone()
                    if ns_row and ns_row[0] != "empty":
                        ns_obj = self.serde.loads_typed((ns_row[0], ns_row[1]))
                        if isinstance(ns_obj, dict):
                            node_statuses = {str(k): str(v) for k, v in ns_obj.items()}

                summaries.append(
                    CheckpointSummary(
                        checkpoint_id=cp_id,
                        parent_checkpoint_id=parent_id,
                        thread_id=thread_id,
                        step=step,
                        created_at=created_at,
                        ticket_id=ticket_id,
                        node_statuses=node_statuses,
                    )
                )
            return summaries

    def fork_checkpoint(
        self,
        source_thread_id: str,
        checkpoint_id: str,
        target_thread_id: str,
    ) -> CheckpointSummary:
        """Copy a specific historical checkpoint cleanly into target_thread_id without mutating source."""
        with self._lock:
            row = self.conn.execute(
                """
                SELECT checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = '' AND checkpoint_id = ?
                """,
                (source_thread_id, checkpoint_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Checkpoint '{checkpoint_id}' not found for thread '{source_thread_id}'")

            cp_ns, cp_id, parent_cp_id, cp_type, cp_blob, meta_blob, created_at = row
            cp_dict = self.serde.loads_typed((cp_type, cp_blob))
            versions = cp_dict.get("channel_versions", {})

            # Clean any prior state in target thread
            self.delete_thread(target_thread_id)

            # Copy all referenced channel blobs
            for ch_name, ver in versions.items():
                b_row = self.conn.execute(
                    """
                    SELECT type, blob FROM blobs
                    WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?
                    """,
                    (source_thread_id, cp_ns, ch_name, str(ver)),
                ).fetchone()
                if b_row is not None:
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO blobs (thread_id, checkpoint_ns, channel, version, type, blob)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (target_thread_id, cp_ns, ch_name, str(ver), b_row[0], b_row[1]),
                    )

            # Insert single clean checkpoint snapshot (no pending interrupt writes)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (target_thread_id, cp_ns, cp_id, None, cp_type, cp_blob, meta_blob, created_at),
            )
            self.conn.commit()

            return CheckpointSummary(
                checkpoint_id=cp_id,
                parent_checkpoint_id=None,
                thread_id=target_thread_id,
                step=None,
                created_at=created_at,
                node_statuses={},
            )

    def rewind_thread(self, thread_id: str, checkpoint_id: str) -> CheckpointSummary:
        """Truncate checkpoints on thread_id newer than checkpoint_id so time rewinds in-place."""
        with self._lock:
            row = self.conn.execute(
                """
                SELECT parent_checkpoint_id, created_at
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = '' AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Checkpoint '{checkpoint_id}' not found for thread '{thread_id}'")

            parent_id, created_at = row
            self.conn.execute(
                "DELETE FROM writes WHERE thread_id = ? AND checkpoint_id >= ?",
                (thread_id, checkpoint_id),
            )
            self.conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id > ?",
                (thread_id, checkpoint_id),
            )
            self.conn.commit()
            return CheckpointSummary(
                checkpoint_id=checkpoint_id,
                parent_checkpoint_id=parent_id,
                thread_id=thread_id,
                created_at=created_at,
            )

    def close(self) -> None:
        with self._lock:
            if getattr(self, "conn", None) is not None:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None  # type: ignore[assignment]
