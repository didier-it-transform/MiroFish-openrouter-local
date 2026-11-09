"""Local, self-contained replacement for the Zep Cloud SDK.

This package implements the small subset of the zep-cloud SDK surface that
MiroFish uses, backed entirely by a local SQLite database. The only external
service involved is the configured LLM provider (OpenRouter), which is used
for entity/fact extraction when episodes are ingested.
"""

from .client import Zep
from .core.api_error import ApiError
from .exceptions import NotFoundError
from .models import BatchAddItem, EntityEdgeSourceTarget
from .types import SearchFilters

__all__ = [
    "Zep",
    "ApiError",
    "NotFoundError",
    "BatchAddItem",
    "EntityEdgeSourceTarget",
    "SearchFilters",
]

__version__ = "0.0.0-local"
