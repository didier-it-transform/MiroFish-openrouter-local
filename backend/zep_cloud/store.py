"""SQLite-backed storage for the local zep_cloud shim."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_STORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "uploads", "zep_local", "graph.db"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graphs (
    graph_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    uuid_ TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    data TEXT NOT NULL,
    data_type TEXT,
    source_description TEXT,
    metadata TEXT,
    created_at TEXT,
    processed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS nodes (
    uuid_ TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    name TEXT NOT NULL,
    labels TEXT,
    summary TEXT,
    attributes TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    uuid_ TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    name TEXT,
    fact TEXT,
    source_node_uuid TEXT NOT NULL,
    target_node_uuid TEXT NOT NULL,
    attributes TEXT,
    created_at TEXT,
    valid_at TEXT,
    invalid_at TEXT,
    expired_at TEXT
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    metadata TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_items (
    batch_id TEXT NOT NULL,
    sequence_index INTEGER NOT NULL,
    episode_uuid TEXT NOT NULL,
    item_type TEXT,
    graph_id TEXT,
    data TEXT,
    data_type TEXT,
    source_description TEXT,
    metadata TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    PRIMARY KEY (batch_id, sequence_index)
);
CREATE TABLE IF NOT EXISTS ontologies (
    graph_id TEXT PRIMARY KEY,
    entities TEXT,
    edges TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_graph ON episodes(graph_id);
CREATE INDEX IF NOT EXISTS idx_nodes_graph ON nodes(graph_id);
CREATE INDEX IF NOT EXISTS idx_edges_graph ON edges(graph_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GraphStore:
    """Thread-safe SQLite store for graphs, episodes, nodes, edges, batches."""

    def __init__(self, db_path: Optional[str] = None):
        resolved = db_path or os.environ.get("ZEP_LOCAL_STORE") or DEFAULT_STORE_PATH
        resolved = os.path.abspath(resolved)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        self.db_path = resolved
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(resolved, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "uuid_": row["uuid_"],
            "graph_id": row["graph_id"],
            "name": row["name"],
            "labels": json.loads(row["labels"] or "[]"),
            "summary": row["summary"] or "",
            "attributes": json.loads(row["attributes"] or "{}"),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "uuid_": row["uuid_"],
            "graph_id": row["graph_id"],
            "name": row["name"] or "",
            "fact": row["fact"] or "",
            "source_node_uuid": row["source_node_uuid"],
            "target_node_uuid": row["target_node_uuid"],
            "attributes": json.loads(row["attributes"] or "{}"),
            "created_at": row["created_at"],
            "valid_at": row["valid_at"],
            "invalid_at": row["invalid_at"],
            "expired_at": row["expired_at"],
        }

    def create_graph(self, graph_id: str, name: str, description: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO graphs (graph_id, name, description, created_at) "
                "VALUES (?, ?, ?, ?)",
                (graph_id, name, description, _now()),
            )

    def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM graphs WHERE graph_id = ?", (graph_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "graph_id": row["graph_id"],
            "name": row["name"],
            "description": row["description"],
            "created_at": row["created_at"],
        }

    def delete_graph(self, graph_id: str) -> None:
        with self._lock, self._conn:
            for table in ("episodes", "nodes", "edges", "ontologies"):
                self._conn.execute(f"DELETE FROM {table} WHERE graph_id = ?", (graph_id,))
            self._conn.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))

    def add_episode(
        self,
        graph_id: str,
        data: str,
        *,
        data_type: str = "text",
        source_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ) -> str:
        episode_uuid = str(uuid.uuid4())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO episodes (uuid_, graph_id, data, data_type, source_description, "
                "metadata, created_at, processed) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    episode_uuid,
                    graph_id,
                    data,
                    data_type,
                    source_description,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    created_at or _now(),
                ),
            )
        return episode_uuid

    def get_episode(self, episode_uuid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE uuid_ = ?", (episode_uuid,)
            ).fetchone()
        if row is None:
            return None
        return {
            "uuid_": row["uuid_"],
            "graph_id": row["graph_id"],
            "data": row["data"],
            "data_type": row["data_type"],
            "source_description": row["source_description"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "created_at": row["created_at"],
            "processed": bool(row["processed"]),
        }

    def mark_episode_processed(self, episode_uuid: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE episodes SET processed = 1 WHERE uuid_ = ?", (episode_uuid,)
            )

    def upsert_node(
        self,
        graph_id: str,
        name: str,
        *,
        labels: Optional[List[str]] = None,
        summary: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        key = name.strip().lower()
        with self._lock:
            row = self._conn.execute(
                "SELECT uuid_, summary, labels FROM nodes WHERE graph_id = ? AND lower(name) = ?",
                (graph_id, key),
            ).fetchone()
            if row is not None:
                merged_labels = sorted(set(json.loads(row["labels"] or "[]")) | set(labels or []))
                merged_summary = summary or row["summary"] or ""
                self._conn.execute(
                    "UPDATE nodes SET labels = ?, summary = ? WHERE uuid_ = ?",
                    (json.dumps(merged_labels), merged_summary, row["uuid_"]),
                )
                with self._conn:
                    self._conn.commit()
                return row["uuid_"]
            node_uuid = str(uuid.uuid4())
            with self._conn:
                self._conn.execute(
                    "INSERT INTO nodes (uuid_, graph_id, name, labels, summary, attributes, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        node_uuid,
                        graph_id,
                        name.strip(),
                        json.dumps(labels or [], ensure_ascii=False),
                        summary,
                        json.dumps(attributes or {}, ensure_ascii=False),
                        _now(),
                    ),
                )
        return node_uuid

    def add_edge(
        self,
        graph_id: str,
        *,
        name: str,
        fact: str,
        source_node_uuid: str,
        target_node_uuid: str,
        attributes: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ) -> str:
        edge_uuid = str(uuid.uuid4())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO edges (uuid_, graph_id, name, fact, source_node_uuid, target_node_uuid, "
                "attributes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    edge_uuid,
                    graph_id,
                    name,
                    fact,
                    source_node_uuid,
                    target_node_uuid,
                    json.dumps(attributes or {}, ensure_ascii=False),
                    created_at or _now(),
                ),
            )
        return edge_uuid

    def find_node_by_name(self, graph_id: str, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE graph_id = ? AND lower(name) = ?",
                (graph_id, name.strip().lower()),
            ).fetchone()
        return self._row_to_node(row) if row is not None else None

    def get_node(self, node_uuid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE uuid_ = ?", (node_uuid,)
            ).fetchone()
        return self._row_to_node(row) if row is not None else None

    def list_nodes(self, graph_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE graph_id = ? ORDER BY created_at, uuid_ LIMIT ? OFFSET ?",
                (graph_id, limit, offset),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def list_edges(self, graph_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE graph_id = ? ORDER BY created_at, uuid_ LIMIT ? OFFSET ?",
                (graph_id, limit, offset),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_node_uuid = ? OR target_node_uuid = ? "
                "ORDER BY created_at, uuid_",
                (node_uuid, node_uuid),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def set_ontology(
        self,
        graph_id: str,
        entities: Optional[Dict[str, Any]],
        edges: Optional[Dict[str, Any]],
    ) -> None:
        def _schema(mapping: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not mapping:
                return []
            result = []
            for type_name, model in mapping.items():
                result.append(
                    {
                        "name": type_name,
                        "doc": getattr(model, "__doc__", None) or "",
                        "fields": sorted(
                            (
                                (getattr(model, "__annotations__", None) or {}).keys()
                            )
                            - {"__"}
                        ) if getattr(model, "__annotations__", None) else [],
                    }
                )
            return result

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO ontologies (graph_id, entities, edges) VALUES (?, ?, ?)",
                (
                    graph_id,
                    json.dumps(_schema(entities), ensure_ascii=False),
                    json.dumps(_schema(edges), ensure_ascii=False),
                ),
            )

    def create_batch(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        batch_id = str(uuid.uuid4())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO batches (batch_id, metadata, status, created_at) VALUES (?, ?, 'draft', ?)",
                (batch_id, json.dumps(metadata or {}, ensure_ascii=False), _now()),
            )
        return batch_id

    def add_batch_items(self, batch_id: str, items: List[Any]) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(sequence_index), -1) AS max_idx FROM batch_items WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            next_index = (row["max_idx"] if row else -1) + 1
        for offset, item in enumerate(items):
            sequence_index = next_index + offset
            episode_uuid = str(uuid.uuid4())
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO batch_items (batch_id, sequence_index, episode_uuid, item_type, graph_id, "
                    "data, data_type, source_description, metadata, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                    (
                        batch_id,
                        sequence_index,
                        episode_uuid,
                        item.type,
                        item.graph_id,
                        item.data,
                        item.data_type,
                        item.source_description,
                        json.dumps(item.metadata or {}, ensure_ascii=False),
                    ),
                )
            details.append(
                {
                    "sequence_index": sequence_index,
                    "episode_uuid": episode_uuid,
                    "status": "pending",
                }
            )
        return details

    def list_batches(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM batches ORDER BY created_at, batch_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            {
                "batch_id": r["batch_id"],
                "metadata": json.loads(r["metadata"] or "{}"),
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                return None
            total = self._conn.execute(
                "SELECT COUNT(*) AS c, "
                "SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END) AS ok "
                "FROM batch_items WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        total_items = total["c"] or 0
        succeeded = total["ok"] or 0
        percent = int((succeeded / total_items) * 100) if total_items else 0
        return {
            "batch_id": row["batch_id"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "status": row["status"],
            "created_at": row["created_at"],
            "progress": {
                "percent_complete": percent,
                "succeeded_items": succeeded,
                "total_items": total_items,
            },
        }

    def list_batch_items(
        self, batch_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY sequence_index LIMIT ? OFFSET ?",
                (batch_id, limit, offset),
            ).fetchall()
        return [
            {
                "sequence_index": r["sequence_index"],
                "episode_uuid": r["episode_uuid"],
                "status": r["status"],
                "error": r["error"],
            }
            for r in rows
        ]

    def pending_batch_items(self, batch_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM batch_items WHERE batch_id = ? AND status = 'pending' ORDER BY sequence_index",
                (batch_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_batch_item_status(
        self, batch_id: str, sequence_index: int, status: str, error: Optional[str] = None
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE batch_items SET status = ?, error = ? WHERE batch_id = ? AND sequence_index = ?",
                (status, error, batch_id, sequence_index),
            )

    def set_batch_status(self, batch_id: str, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE batches SET status = ? WHERE batch_id = ?", (status, batch_id)
            )

    def search_nodes(self, graph_id: str, query: str, limit: int) -> List[Dict[str, Any]]:
        tokens = _tokenize(query)
        scored = []
        for node in self.list_nodes(graph_id, limit=100000):
            score = _overlap(tokens, f"{node['name']} {node['summary']}")
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda pair: -pair[0])
        return [node for _, node in scored[:limit]]

    def search_edges(self, graph_id: str, query: str, limit: int) -> List[Dict[str, Any]]:
        tokens = _tokenize(query)
        scored = []
        for edge in self.list_edges(graph_id, limit=100000):
            score = _overlap(tokens, f"{edge['name']} {edge['fact']}")
            if score > 0:
                scored.append((score, edge))
        scored.sort(key=lambda pair: -pair[0])
        return [edge for _, edge in scored[:limit]]


def _tokenize(text: str) -> List[str]:
    import re

    return [t for t in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", text.lower()) if t]


def _overlap(tokens: List[str], text: str) -> int:
    if not tokens:
        return 0
    haystack = text.lower()
    return sum(1 for token in tokens if token in haystack)
