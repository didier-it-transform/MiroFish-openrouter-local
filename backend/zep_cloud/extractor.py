"""LLM-based entity/fact extraction for ingested episodes.

Uses the configured OpenAI-compatible provider (OpenRouter) to turn episode
text into graph nodes and edges. Falls back to a no-op when no API key is
configured so local reads and searches keep working offline.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("zep_cloud.extractor")

_EXTRACTION_PROMPT_TEMPLATE = """You are a knowledge-graph extraction engine.
Extract entities and facts from the following text.

Return ONLY a JSON object with this exact shape:
{{
  "nodes": [
    {{"name": "<entity name>", "label": "<entity type, e.g. Person/Organization/Event>", "summary": "<one-sentence description>"}}
  ],
  "edges": [
    {{"source": "<entity name>", "target": "<entity name>", "fact": "<short factual sentence describing the relationship>", "name": "<RELATIONSHIP_TYPE>"}}
  ]
}}

Rules:
- Node names must be canonical, short entity names.
- Every edge source/target must match a node name in this same output.
- Facts must be grounded in the text; do not invent information.
- Keep output compact.

Text:
{text}"""


class _NoopCompletions:
    def create(self, **kwargs: Any):
        raise RuntimeError("LLM provider is not configured; cannot extract graph facts")


def _get_completions():
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        return _NoopCompletions()
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "120")),
        )
        return client.chat.completions
    except Exception as error:
        logger.warning("LLM client unavailable for extraction: %s", error)
        return _NoopCompletions()


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def extract_facts(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract {"nodes": [...], "edges": [...]} from free text via the LLM."""

    trimmed = (text or "").strip()
    if not trimmed:
        return {"nodes": [], "edges": []}

    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(text=trimmed[:12000])
    completions = _get_completions()
    try:
        response = completions.create(
            model=os.environ.get("LLM_MODEL_NAME", "openai/gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
    except Exception as error:
        logger.warning("LLM fact extraction failed: %s", error)
        return {"nodes": [], "edges": []}

    parsed = _parse_json_object(content)
    if not parsed:
        logger.warning("LLM fact extraction returned unparseable output")
        return {"nodes": [], "edges": []}

    nodes = parsed.get("nodes") or []
    edges = parsed.get("edges") or []
    valid_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name") or "").strip()
        if not name:
            continue
        valid_nodes.append(
            {
                "name": name,
                "label": str(node.get("label") or "Entity").strip() or "Entity",
                "summary": str(node.get("summary") or "").strip(),
            }
        )
    known_names = {n["name"].lower() for n in valid_nodes}
    valid_edges: List[Dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        fact = str(edge.get("fact") or "").strip()
        if not source or not target or not fact:
            continue
        if source.lower() not in known_names or target.lower() not in known_names:
            continue
        valid_edges.append(
            {
                "source": source,
                "target": target,
                "fact": fact,
                "name": str(edge.get("name") or "RELATED_TO").strip() or "RELATED_TO",
            }
        )

    return {"nodes": valid_nodes, "edges": valid_edges}
