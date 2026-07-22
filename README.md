# Document Intelligence Pipeline

An end-to-end pipeline that extracts structured data from documents using
vision-language models, validates the output with an agentic LangGraph state
machine, and benchmarks accuracy/latency/cost across models.

## Why this exists

Most document-extraction demos stop at "call a VLM, print JSON." This
pipeline treats extraction as a production problem: outputs are validated
before they're trusted, evaluation is done cross-corpus (train on one
document set, test on a disjoint one) to avoid inflated accuracy numbers,
and every run is logged with cost/latency so model choice is a measured
trade-off, not a guess.

## Dataset note

Real annotated tax/insurance forms are hard to source publicly (sensitive
data — the same scarcity problem cited by the FUNSD paper). `data/generate_synthetic_w2.py`
generates unlimited W-2-style documents as PDFs with paired ground-truth
JSON and no real PII, giving the eval harness exact labels to score against.
A `--noise` flag adds layout jitter for a harder, more realistic test split.

## Architecture

```
Ingestion -> Preprocessing -> Multi-VLM extraction -> Agentic validation
                                                              |
                                                              v
                                          Structured output -> Eval + cost dashboard
```

The validation stage is a LangGraph state machine: `extract -> check_fields
-> (finalize | retry | escalate)`. Documents with high-severity issues get a
corrective re-extraction (up to `MAX_RETRIES`); documents that still fail are
flagged for human review rather than silently shipped.

## Repo structure

```
doc-intel-pipeline/
├── data/              # synthetic W-2 generator + generated corpora
├── ingestion/         # PDF/image loading, multi-page payload construction
├── extraction/        # Pydantic schemas + async VLM client (vLLM/Groq/Together)
├── validation/        # LangGraph agentic validation state machine
├── eval/               # cross-corpus evaluation harness, MLflow logging
├── serving/            # FastAPI app
├── dashboard/           # (planned) Streamlit view of eval results
├── notebooks/           # Colab-run LoRA fine-tuning notebooks
├── tests/               # pytest unit tests for validation rules
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in VLM_API_KEY etc.
```

Generate synthetic data:
```bash
python data/generate_synthetic_w2.py --n 50 --out data/synthetic_w2_train --seed 1
python data/generate_synthetic_w2.py --n 20 --out data/synthetic_w2_test --seed 2 --noise 1.0
```

Run the API:
```bash
uvicorn serving.app:app --reload --port 8000
```

Run tests:
```bash
pytest tests/ -v
```

## Compute constraints

Developed locally on a GTX 1650 (4GB VRAM) — local fine-tuning isn't
feasible on this hardware. LoRA fine-tuning runs on Colab/Kaggle free-tier
T4 GPUs with 4-bit QLoRA; inference comparisons run against quantized small
models locally and hosted APIs (Groq/Together) for larger models.

## Evaluation methodology

Training and test corpora are always disjoint (`eval/harness.py` takes two
separate corpus arguments). Results below are placeholders — fill in after
running the harness against your generated train/test splits:

| Model | Corpus | Overall accuracy | Avg latency (ms) | Notes |
|---|---|---|---|---|
| Qwen2.5-VL-3B (base) | synthetic test (noisy) | — | — | baseline, no fine-tuning |
| Qwen2.5-VL-3B (LoRA) | synthetic test (noisy) | — | — | fine-tuned on Colab T4 |
| Llama-4-Scout (Groq API) | synthetic test (noisy) | — | — | hosted comparison |

## Roadmap

- [ ] Run baseline extraction across 2-3 models on synthetic data
- [ ] LoRA fine-tune on Colab, cross-corpus eval
- [ ] Streamlit dashboard for eval/cost results
- [ ] Deploy demo to Hugging Face Spaces
- [ ] Optionally add a real noisy-scan corpus (e.g. FUNSD, downloaded from
      Colab where full internet access is available) as a secondary test set
