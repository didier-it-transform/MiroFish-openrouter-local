"""Local Zep client implementing the subset of the Zep Cloud SDK used by MiroFish."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .extractor import extract_facts
from .exceptions import NotFoundError
from .store import GraphStore

_CLIENT_LOCK = threading.Lock()
_CLIENT: Optional["Zep"] = None


def _node_obj(node: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        uuid_=node["uuid_"],
        name=node["name"],
        labels=node["labels"],
        summary=node["summary"],
        attributes=node["attributes"],
        created_at=node.get("created_at"),
    )


def _edge_obj(edge: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        uuid_=edge["uuid_"],
        name=edge["name"],
        fact=edge["fact"],
        source_node_uuid=edge["source_node_uuid"],
        target_node_uuid=edge["target_node_uuid"],
        attributes=edge["attributes"],
        created_at=edge.get("created_at"),
        valid_at=edge.get("valid_at"),
        invalid_at=edge.get("invalid_at"),
        expired_at=edge.get("expired_at"),
    )


def _episode_obj(episode: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        uuid_=episode["uuid_"],
        data=episode["data"],
        data_type=episode["data_type"],
        source_description=episode["source_description"],
        metadata=episode["metadata"],
        created_at=episode["created_at"],
        processed=episode["processed"],
    )


class _GraphAPI:
    def __init__(self, client: "Zep"):
        self._client = client
        self.node = _NodeAPI(client)
        self.edge = _EdgeAPI(client)
        self.episode = _EpisodeAPI(client)

    def create(self, graph_id: str, name: str, description: str = "") -> SimpleNamespace:
        self._client.store.create_graph(graph_id, name, description)
        graph = self._client.store.get_graph(graph_id)
        assert graph is not None
        return SimpleNamespace(graph_id=graph["graph_id"], name=graph["name"])

    def get(self, graph_id: str) -> SimpleNamespace:
        graph = self._client.store.get_graph(graph_id)
        if graph is None:
            raise NotFoundError(f"Graph {graph_id} not found")
        return SimpleNamespace(**graph)

    def delete(self, graph_id: str) -> SimpleNamespace:
        self._client.store.delete_graph(graph_id)
        return SimpleNamespace(success=True)

    def add(
        self,
        graph_id: str,
        type: str,
        data: str,
        created_at: Optional[str] = None,
        source_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        episode_uuid = self._client.store.add_episode(
            graph_id,
            data,
            data_type=type,
            source_description=source_description,
            metadata=metadata,
            created_at=created_at,
        )
        self._client._process_episode(episode_uuid, graph_id, data)
        return SimpleNamespace(uuid_=episode_uuid, processed=True)

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: Optional[str] = None,
        filters: Optional[Any] = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        if scope == "nodes":
            nodes = self._client.store.search_nodes(graph_id, query, limit)
            return SimpleNamespace(nodes=[_node_obj(n) for n in nodes], edges=[], facts=[])
        edges = self._client.store.search_edges(graph_id, query, limit)
        return SimpleNamespace(
            edges=[_edge_obj(e) for e in edges],
            nodes=[],
            facts=[e["fact"] for e in edges],
        )

    def set_ontology(
        self,
        graph_ids: List[str],
        entities: Optional[Dict[str, Any]] = None,
        edges: Optional[Dict[str, Any]] = None,
    ) -> SimpleNamespace:
        for graph_id in graph_ids:
            self._client.store.set_ontology(graph_id, entities, edges)
        return SimpleNamespace(success=True)


class _NodeAPI:
    def __init__(self, client: "Zep"):
        self._client = client
        self.with_raw_response = _RawNodeAPI(client)

    def get(self, uuid_: str) -> SimpleNamespace:
        node = self._client.store.get_node(uuid_)
        if node is None:
            raise NotFoundError(f"Node {uuid_} not found")
        return _node_obj(node)

    def get_edges(self, node_uuid: str) -> List[SimpleNamespace]:
        return [_edge_obj(e) for e in self._client.store.get_node_edges(node_uuid)]


class _RawNodeAPI:
    def __init__(self, client: "Zep"):
        self._client = client

    def get_by_graph_id(self, graph_id: str, limit: int = 100, cursor: Optional[int] = None):
        offset = int(cursor or 0)
        nodes = self._client.store.list_nodes(graph_id, limit=limit, offset=offset)
        next_cursor = offset + len(nodes)
        headers: Dict[str, Any] = {}
        if len(nodes) >= limit:
            headers["zep-next-cursor"] = str(next_cursor)
        return SimpleNamespace(data=[_node_obj(n) for n in nodes], headers=headers)


class _EdgeAPI:
    def __init__(self, client: "Zep"):
        self._client = client
        self.with_raw_response = _RawEdgeAPI(client)

    def get(self, uuid_: str) -> SimpleNamespace:
        for edge in self._client.store.list_edges("", limit=1000000):
            if edge["uuid_"] == uuid_:
                return _edge_obj(edge)
        raise NotFoundError(f"Edge {uuid_} not found")


class _RawEdgeAPI:
    def __init__(self, client: "Zep"):
        self._client = client

    def get_by_graph_id(self, graph_id: str, limit: int = 100, cursor: Optional[int] = None):
        offset = int(cursor or 0)
        edges = self._client.store.list_edges(graph_id, limit=limit, offset=offset)
        next_cursor = offset + len(edges)
        headers: Dict[str, Any] = {}
        if len(edges) >= limit:
            headers["zep-next-cursor"] = str(next_cursor)
        return SimpleNamespace(data=[_edge_obj(e) for e in edges], headers=headers)


class _EpisodeAPI:
    def __init__(self, client: "Zep"):
        self._client = client

    def get(self, uuid_: str) -> SimpleNamespace:
        episode = self._client.store.get_episode(uuid_)
        if episode is None:
            raise NotFoundError(f"Episode {uuid_} not found")
        return _episode_obj(episode)


class _BatchAPI:
    def __init__(self, client: "Zep"):
        self._client = client

    def create(self, metadata: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
        batch_id = self._client.store.create_batch(metadata)
        return self.get(batch_id)

    def add(self, batch_id: str, items: List[Any]) -> List[SimpleNamespace]:
        details = self._client.store.add_batch_items(batch_id, items)
        return [SimpleNamespace(**d) for d in details]

    def process(self, batch_id: str) -> SimpleNamespace:
        for item in self._client.store.pending_batch_items(batch_id):
            try:
                self._client._process_episode(item["episode_uuid"], item["graph_id"], item["data"])
                self._client.store.set_batch_item_status(
                    batch_id, item["sequence_index"], "succeeded"
                )
            except Exception as error:
                self._client.store.set_batch_item_status(
                    batch_id, item["sequence_index"], "failed", str(error)
                )
        self._client.store.set_batch_status(batch_id, "succeeded")
        return self.get(batch_id)

    def get(self, batch_id: str) -> SimpleNamespace:
        batch = self._client.store.get_batch(batch_id)
        if batch is None:
            raise NotFoundError(f"Batch {batch_id} not found")
        progress = batch["progress"]
        return SimpleNamespace(
            batch_id=batch["batch_id"],
            metadata=batch["metadata"],
            status=batch["status"],
            created_at=batch["created_at"],
            progress=SimpleNamespace(
                percent_complete=progress["percent_complete"],
                succeeded_items=progress["succeeded_items"],
                total_items=progress["total_items"],
            ),
        )

    def list(self, limit: int = 100, cursor: Optional[int] = None) -> SimpleNamespace:
        offset = int(cursor or 0)
        batches = self._client.store.list_batches(limit=limit, offset=offset)
        next_cursor = offset + len(batches)
        return SimpleNamespace(
            batches=[
                SimpleNamespace(batch_id=b["batch_id"], metadata=b["metadata"], status=b["status"])
                for b in batches
            ],
            next_cursor=next_cursor if len(batches) >= limit else None,
        )

    def list_items(
        self, batch_id: str, limit: int = 100, cursor: Optional[int] = None
    ) -> SimpleNamespace:
        offset = int(cursor or 0)
        items = self._client.store.list_batch_items(batch_id, limit=limit, offset=offset)
        next_cursor = offset + len(items)
        return SimpleNamespace(
            items=[SimpleNamespace(**i) for i in items],
            next_cursor=next_cursor if len(items) >= limit else None,
        )


class Zep:
    """Local, SQLite-backed drop-in replacement for the Zep Cloud client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        self.store = GraphStore()
        self.graph = _GraphAPI(self)
        self.batch = _BatchAPI(self)

    def _process_episode(self, episode_uuid: str, graph_id: str, data: str) -> None:
        extraction = extract_facts(data)
        name_to_uuid: Dict[str, str] = {}
        for node in extraction["nodes"]:
            existing = self.store.find_node_by_name(graph_id, node["name"])
            if existing is not None:
                name_to_uuid[node["name"].lower()] = existing["uuid_"]
                continue
            node_uuid = self.store.upsert_node(
                graph_id,
                node["name"],
                labels=[node["label"]],
                summary=node["summary"],
            )
            name_to_uuid[node["name"].lower()] = node_uuid
        for edge in extraction["edges"]:
            source_uuid = name_to_uuid.get(edge["source"].lower())
            target_uuid = name_to_uuid.get(edge["target"].lower())
            if not source_uuid or not target_uuid:
                continue
            self.store.add_edge(
                graph_id,
                name=edge["name"],
                fact=edge["fact"],
                source_node_uuid=source_uuid,
                target_node_uuid=target_uuid,
            )
        self.store.mark_episode_processed(episode_uuid)
