"""
Evaluation harness. Enforces cross-corpus evaluation by construction — you
pass two separate corpus paths in, not a single one you split yourself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import pandas as pd

from extraction.schemas import ExtractedDocument, ModelRunMetadata
from extraction.vlm_client import VLMConfig, extract_document
from ingestion.loader import preprocess_document


@dataclass
class EvalExample:
    document_path: Path
    ground_truth: ExtractedDocument


@dataclass
class EvalRunResult:
    model_name: str
    corpus_name: str
    field_accuracy: dict[str, float] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    n_examples: int = 0


def _field_match(pred_value, true_value) -> bool:
    if pred_value is None and true_value is None:
        return True
    if pred_value is None or true_value is None:
        return False
    return str(pred_value).strip().lower() == str(true_value).strip().lower()


async def evaluate_model_on_corpus(
    config: VLMConfig,
    corpus: list[EvalExample],
    corpus_name: str,
    run_experiment: str = "doc-intel-eval",
) -> EvalRunResult:
    mlflow.set_experiment(run_experiment)
    field_hits: dict[str, int] = {}
    field_totals: dict[str, int] = {}
    latencies: list[float] = []
    metadata_log: list[ModelRunMetadata] = []

    with mlflow.start_run(run_name=f"{config.model_name}__{corpus_name}"):
        mlflow.log_params({"model": config.model_name, "provider": config.provider, "corpus": corpus_name})

        for example in corpus:
            page_content = preprocess_document(example.document_path)
            start = time.perf_counter()
            predicted, metadata = await extract_document(config, page_content)
            latencies.append((time.perf_counter() - start) * 1000)
            metadata_log.append(metadata)

            if predicted is None:
                continue

            for field_name in ExtractedDocument.model_fields:
                if field_name == "line_items":
                    continue
                pred_field = getattr(predicted, field_name)
                true_field = getattr(example.ground_truth, field_name)
                field_totals[field_name] = field_totals.get(field_name, 0) + 1
                if _field_match(pred_field.value, true_field.value):
                    field_hits[field_name] = field_hits.get(field_name, 0) + 1

        field_accuracy = {
            name: field_hits.get(name, 0) / total for name, total in field_totals.items()
        }
        overall = sum(field_hits.values()) / max(sum(field_totals.values()), 1)
        avg_latency = sum(latencies) / max(len(latencies), 1)

        for name, acc in field_accuracy.items():
            mlflow.log_metric(f"accuracy_{name}", acc)
        mlflow.log_metric("overall_accuracy", overall)
        mlflow.log_metric("avg_latency_ms", avg_latency)

        return EvalRunResult(
            model_name=config.model_name,
            corpus_name=corpus_name,
            field_accuracy=field_accuracy,
            overall_accuracy=overall,
            avg_latency_ms=avg_latency,
            n_examples=len(corpus),
        )


def summarize_runs(results: list[EvalRunResult]) -> pd.DataFrame:
    rows = [
        {
            "model": r.model_name,
            "corpus": r.corpus_name,
            "overall_accuracy": round(r.overall_accuracy, 4),
            "avg_latency_ms": round(r.avg_latency_ms, 1),
            "n_examples": r.n_examples,
        }
        for r in results
    ]
    return pd.DataFrame(rows)
