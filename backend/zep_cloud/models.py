from typing import Any, Dict, Optional


class BatchAddItem:
    """A single text payload submitted to a batch."""

    def __init__(
        self,
        *,
        type: str,
        graph_id: str,
        data: str,
        data_type: str = "text",
        source_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.type = type
        self.graph_id = graph_id
        self.data = data
        self.data_type = data_type
        self.source_description = source_description
        self.metadata = metadata or {}


class EntityEdgeSourceTarget:
    """Allowed source/target entity types for an edge type."""

    def __init__(self, *, source: str, target: str):
        self.source = source
        self.target = target
