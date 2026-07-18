"""
Pydantic schemas for extracted document fields.

Swap the fields inside `ExtractedDocument` for whatever public dataset you
land on (IRS forms, SEC filings, insurance policies). Keep the surrounding
structure (confidence, page_span, validation hooks) — that's the part that's
reusable across domains.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FieldConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExtractedField(BaseModel):
    """A single extracted value plus metadata used by the validation agent."""

    value: Optional[str] = None
    confidence: FieldConfidence = FieldConfidence.MEDIUM
    source_page: Optional[int] = Field(
        default=None, description="1-indexed page the value was read from"
    )
    raw_model_output: Optional[str] = Field(
        default=None, description="unparsed model text, kept for debugging"
    )


class ExtractedDocument(BaseModel):
    """
    Example schema for a tax-form-style document (swap for your chosen domain).
    Every field is an ExtractedField so the validation agent can reason about
    confidence and flag individual fields, not just the whole document.
    """

    document_type: ExtractedField
    payer_name: ExtractedField
    recipient_name: ExtractedField
    identifier: ExtractedField  # e.g. SSN/EIN/policy number — validate format, don't store real PII
    tax_year: ExtractedField
    total_amount: ExtractedField
    line_items: list[ExtractedField] = Field(default_factory=list)

    @field_validator("total_amount")
    @classmethod
    def total_amount_is_numeric(cls, v: ExtractedField) -> ExtractedField:
        if v.value is not None:
            cleaned = v.value.replace(",", "").replace("$", "").strip()
            try:
                float(cleaned)
            except ValueError:
                v.confidence = FieldConfidence.LOW
        return v


class ValidationIssue(BaseModel):
    field_name: str
    issue: str
    severity: FieldConfidence = FieldConfidence.MEDIUM


class ValidationResult(BaseModel):
    document: ExtractedDocument
    issues: list[ValidationIssue] = Field(default_factory=list)
    needs_human_review: bool = False
    retry_count: int = 0


class ModelRunMetadata(BaseModel):
    """Attach to every extraction call — this is what feeds the cost/eval dashboard."""

    model_name: str
    provider: str  # e.g. "groq", "vllm-local", "together"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
