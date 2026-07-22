"""
Pydantic schemas for extracted document fields (W-2-style tax document).
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
    """Schema for a W-2-style synthetic tax document."""

    document_type: ExtractedField
    payer_name: ExtractedField
    recipient_name: ExtractedField
    identifier: ExtractedField  # SSN-shaped id — validate format, never store real PII
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
    """Attach to every extraction call — feeds the cost/eval dashboard."""

    model_name: str
    provider: str  # e.g. "groq", "vllm-local", "together"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
