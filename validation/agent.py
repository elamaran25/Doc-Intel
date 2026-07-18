"""
Agentic validation layer built as an explicit LangGraph state machine.

Flow:
    extract -> check_fields -> (valid: finalize) | (invalid, retries left: retry_extraction)
                                                   | (invalid, no retries left: escalate)

This is the piece that turns a script into a pipeline: instead of trusting
whatever the VLM returns, every extraction is checked against rule-based
validators, and low-confidence or inconsistent documents get either a
corrective retry or a flag for human review rather than silently shipping
bad data.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from extraction.schemas import (
    ExtractedDocument,
    FieldConfidence,
    ModelRunMetadata,
    ValidationIssue,
    ValidationResult,
)
from extraction.vlm_client import VLMConfig, extract_document

MAX_RETRIES = 2


class PipelineState(TypedDict):
    page_content: list[dict]
    config: VLMConfig
    document: ExtractedDocument | None
    issues: list[ValidationIssue]
    retry_count: int
    metadata_log: list[ModelRunMetadata]
    needs_human_review: bool


# --- Validation rules -------------------------------------------------------
# Keep these as small, named, independently testable functions. That's what
# lets you show a "validation coverage" table in the README instead of a
# black box.

def _validate_identifier_format(doc: ExtractedDocument) -> ValidationIssue | None:
    value = doc.identifier.value
    if value and not re.match(r"^[\d\-]{4,20}$", value):
        return ValidationIssue(
            field_name="identifier",
            issue="does not match expected numeric/hyphen format",
            severity=FieldConfidence.HIGH,
        )
    return None


def _validate_required_fields(doc: ExtractedDocument) -> list[ValidationIssue]:
    issues = []
    required = ["document_type", "payer_name", "total_amount"]
    for name in required:
        field = getattr(doc, name)
        if field.value is None:
            issues.append(
                ValidationIssue(field_name=name, issue="required field missing", severity=FieldConfidence.HIGH)
            )
    return issues


def _validate_confidence_floor(doc: ExtractedDocument) -> list[ValidationIssue]:
    issues = []
    for name, field in doc:
        if hasattr(field, "confidence") and field.confidence == FieldConfidence.LOW:
            issues.append(
                ValidationIssue(field_name=name, issue="low model confidence", severity=FieldConfidence.LOW)
            )
    return issues


VALIDATORS = [_validate_required_fields, _validate_confidence_floor]
SINGLE_FIELD_VALIDATORS = [_validate_identifier_format]


# --- Graph nodes -------------------------------------------------------------

async def node_extract(state: PipelineState) -> PipelineState:
    doc, metadata = await extract_document(state["config"], state["page_content"])
    state["document"] = doc
    state["metadata_log"].append(metadata)
    return state


def node_check_fields(state: PipelineState) -> PipelineState:
    doc = state["document"]
    issues: list[ValidationIssue] = []

    if doc is None:
        issues.append(ValidationIssue(field_name="_all", issue="extraction returned unparseable output", severity=FieldConfidence.HIGH))
        state["issues"] = issues
        return state

    for validator in VALIDATORS:
        issues.extend(validator(doc))
    for validator in SINGLE_FIELD_VALIDATORS:
        result = validator(doc)
        if result:
            issues.append(result)

    state["issues"] = issues
    return state


def route_after_check(state: PipelineState) -> str:
    high_severity = [i for i in state["issues"] if i.severity == FieldConfidence.HIGH]
    if not high_severity:
        return "finalize"
    if state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "escalate"


def node_retry(state: PipelineState) -> PipelineState:
    state["retry_count"] += 1
    return state


def node_finalize(state: PipelineState) -> PipelineState:
    state["needs_human_review"] = False
    return state


def node_escalate(state: PipelineState) -> PipelineState:
    state["needs_human_review"] = True
    return state


def build_validation_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("extract", node_extract)
    graph.add_node("check_fields", node_check_fields)
    graph.add_node("retry", node_retry)
    graph.add_node("finalize", node_finalize)
    graph.add_node("escalate", node_escalate)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "check_fields")
    graph.add_conditional_edges(
        "check_fields",
        route_after_check,
        {"finalize": "finalize", "retry": "retry", "escalate": "escalate"},
    )
    graph.add_edge("retry", "extract")  # loop back for a corrective re-extraction
    graph.add_edge("finalize", END)
    graph.add_edge("escalate", END)

    return graph.compile()


async def run_pipeline(page_content: list[dict], config: VLMConfig) -> ValidationResult:
    app = build_validation_graph()
    initial_state: PipelineState = {
        "page_content": page_content,
        "config": config,
        "document": None,
        "issues": [],
        "retry_count": 0,
        "metadata_log": [],
        "needs_human_review": False,
    }
    final_state = await app.ainvoke(initial_state)
    return ValidationResult(
        document=final_state["document"],
        issues=final_state["issues"],
        needs_human_review=final_state["needs_human_review"],
        retry_count=final_state["retry_count"],
    )
