from typing import Any, Dict, Optional


class SearchFilters:
    """Compatibility no-op filter container."""

    def __init__(self, **kwargs: Any):
        self.kwargs: Dict[str, Any] = dict(kwargs)

    def model_dump(self) -> Dict[str, Any]:
        return dict(self.kwargs)

    def __repr__(self) -> str:
        return f"SearchFilters({self.kwargs!r})"
