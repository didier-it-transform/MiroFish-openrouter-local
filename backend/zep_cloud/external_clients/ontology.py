"""Pydantic base models and schema helpers for dynamic ontology types."""

from typing import Any, Dict, List

from pydantic import BaseModel

DEFAULT_DETAILS_PROPERTY = {
    "name": "details",
    "type": "Text",
    "description": "Additional details about this ontology type.",
}


class EntityText(str):
    """Marker string type for entity attributes."""


class EntityModel(BaseModel):
    """Base class for dynamically created entity types."""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}


class EdgeModel(BaseModel):
    """Base class for dynamically created edge types."""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}


def _field_type(model: BaseModel, field_name: str) -> str:
    annotation = model.__annotations__.get(field_name)
    if annotation is None:
        return "Text"
    return "Text"


def _model_to_api_schema(model: Any, type_name: str) -> Dict[str, Any]:
    fields = getattr(model, "model_fields", None) or {}
    properties: List[Dict[str, Any]] = []
    for field_name, field_info in fields.items():
        if field_name.startswith("model_"):
            continue
        properties.append(
            {
                "name": field_name,
                "type": "Text",
                "description": getattr(field_info, "description", None) or field_name,
            }
        )
    if not properties:
        properties.append(dict(DEFAULT_DETAILS_PROPERTY))
    return {"name": type_name, "properties": properties}


def entity_model_to_api_schema(model: Any, type_name: str) -> Dict[str, Any]:
    """Return the API schema for a dynamically built entity model."""
    return _model_to_api_schema(model, type_name)


def edge_model_to_api_schema(model: Any, type_name: str) -> Dict[str, Any]:
    """Return the API schema for a dynamically built edge model."""
    return _model_to_api_schema(model, type_name)
