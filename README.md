# Document Intelligence Pipeline

An end-to-end pipeline that extracts structured data from messy real-world
documents (tax forms, insurance policies, filings) using vision-language
models, validates the output with an agentic LangGraph state machine, and
benchmarks accuracy/latency/cost across models.

## Why this exists

Most document-extraction demos stop at "call a VLM, print JSON." This
pipeline treats extraction as a production problem: outputs are validated
before they're trusted, evaluation is done cross-corpus (train on one
document set, test on a disjoint one) to avoid inflated accuracy numbers,
and every run is logged with cost/latency so model choice is a measured
trade-off, not a guess.

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
├── ingestion/       # PDF/image loading, multi-page payload construction
├── extraction/      # Pydantic schemas + async VLM client (vLLM/Groq/Together)
├── validation/       # LangGraph agentic validation state machine
├── eval/             # cross-corpus evaluation harness, MLflow logging
├── serving/          # FastAPI app
├── dashboard/         # (planned) Streamlit view of eval results
├── notebooks/         # Colab-run LoRA fine-tuning notebooks
├── tests/             # pytest unit tests for validation rules
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp  .env   # fill in VLM_API_KEY etc.
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
models locally and hosted APIs (Groq/Together) for larger models. This
mirrors a real cost-vs-capability trade-off decision, not just a hardware
limitation.

## Evaluation methodology

Training and test corpora are always disjoint (`eval/harness.py` takes two
separate corpus arguments — there's no way to accidentally evaluate on the
training set). Results below are placeholders — fill in after running
`eval/harness.py` against your chosen corpora:

| Model | Corpus | Overall accuracy | Avg latency (ms) | Notes |
|---|---|---|---|---|
| Qwen2.5-VL-3B (base) | held-out test | — | — | baseline, no fine-tuning |
| Qwen2.5-VL-3B (LoRA) | held-out test | — | — | fine-tuned on Colab T4 |
| Llama-4-Scout (Groq API) | held-out test | — | — | hosted comparison |

## Roadmap

- [ ] Wire up chosen public dataset (IRS sample forms / SEC filings)
- [ ] Run baseline extraction across 2-3 models
- [ ] LoRA fine-tune on Colab, cross-corpus eval
- [ ] Streamlit dashboard for eval/cost results
- [ ] Deploy demo to Hugging Face Spaces
