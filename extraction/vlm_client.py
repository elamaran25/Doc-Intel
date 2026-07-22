"""
Async VLM client. Works against any OpenAI-compatible endpoint — local vLLM,
Groq, Together, or Fireworks — so the same code path serves both the
self-hosted and hosted-API comparisons in the cost/latency dashboard.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from extraction.schemas import ExtractedDocument, ModelRunMetadata

EXTRACTION_SYSTEM_PROMPT = """You are a document extraction engine. You will be
shown one or more page images from a single document, each preceded by a
`--- PAGE N ---` marker. Extract the requested fields and return ONLY a JSON
object matching the given schema. If a field is not present, set its value to
null and confidence to "low". Do not guess values that are not visibly present
in the document."""


@dataclass
class VLMConfig:
    provider: str  # "vllm-local" | "groq" | "together" | "fireworks"
    model_name: str
    base_url: str
    api_key: str
    top_k: int | None = None
    enable_thinking: bool = False


def build_client(config: VLMConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)


async def extract_document(
    config: VLMConfig,
    page_content: list[dict],
    schema_json: str | None = None,
) -> tuple[ExtractedDocument | None, ModelRunMetadata]:
    client = build_client(config)
    schema_hint = schema_json or ExtractedDocument.model_json_schema()

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT + f"\n\nSchema:\n{json.dumps(schema_hint)}"},
        {"role": "user", "content": page_content},
    ]

    extra_body = {}
    if config.top_k is not None:
        extra_body["top_k"] = config.top_k
    if not config.enable_thinking:
        extra_body["enable_thinking"] = False

    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=0,
        extra_body=extra_body or None,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    raw_text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    metadata = ModelRunMetadata(
        model_name=config.model_name,
        provider=config.provider,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        latency_ms=latency_ms,
        estimated_cost_usd=None,
    )

    try:
        cleaned = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = ExtractedDocument.model_validate_json(cleaned)
        return parsed, metadata
    except Exception:
        return None, metadata
