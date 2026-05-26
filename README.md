# CDR-Bench

This anonymous code release contains the core implementation for constructing and running CDR-Bench, a benchmark for compositional, order-sensitive data refinement by LLMs.

The release keeps the original project shape:

```text
configs/                  corpus, domain, and prompt-generation configs
scripts/                  end-to-end construction, prompt, inference, and scoring wrappers
src/cdrbench/             Python package
  prepare_data/           data construction and deterministic reference generation
  prompting/              prompt library and eval-file construction
  eval/                   inference output parsing and metric computation
  infer/                  OpenAI-compatible and vLLM inference backends
  release/                helper for downloading JSONL files from a dataset repo
```

The following are intentionally not included: paper drafts, model-specific experiment wrappers, analysis/plotting scripts, cached predictions, evaluation outputs, raw corpora, and the full Data-Juicer checkout.

## Environment

Use Python 3.11 if possible.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

CDR-Bench uses Data-Juicer operators to execute deterministic data-refinement recipes. For full data construction, install Data-Juicer or place a compatible checkout at `./data-juicer`:

```bash
python -m pip install py-data-juicer
```

If a local `./data-juicer` directory exists, the construction scripts prefer its `tools/process_data.py` and `tools/analyze_data.py`. Otherwise they fall back to the installed `dj-process` / `dj-analyze` commands.

## Data Layout

Raw JSONL files should be placed under `data/raw/` following `configs/corpora.yaml`:

```text
data/raw/arxiv/arxiv-4k.jsonl
data/raw/commoncrawl/cc-10k.jsonl
data/raw/enwiki/enwiki-pages-110k.jsonl
data/raw/govreport/govreport-20k.jsonl
data/raw/pii/pii-43k.jsonl
data/raw/pii/docpii-contextual-1k.jsonl
data/raw/pii/synthetic-anonymizer-8k.jsonl
```

If the raw files are hosted in a Hugging Face dataset repo, download them with:

```bash
PYTHONPATH=src python -m cdrbench.release.download_hf_jsonl \
  --repo-id <anonymous-or-camera-ready-dataset-repo> \
  --repo-root .
```

The file list used by the downloader is in `configs/release_jsonl_manifest.txt`.

## Build The Benchmark

For a quick smoke test, cap records and skip prompt generation:

```bash
PYTHONPATH=src ./scripts/run_recipe_pipeline.sh \
  --max-records 200 \
  --max-text-length 10000 \
  --skip-prompt-pipeline
```

For the full deterministic construction pipeline:

```bash
PYTHONPATH=src ./scripts/run_recipe_pipeline.sh \
  --max-text-length 10000 \
  --skip-prompt-pipeline
```

This runs:

1. domain filtering and operator tagging
2. recipe-family mining
3. deterministic recipe-library materialization
4. benchmark-instance sampling
5. deterministic reference generation

Main outputs:

```text
data/processed/domain_filtered/all.jsonl
data/processed/domain_tags/*.jsonl
data/processed/recipe_mining/
data/processed/recipe_library/
data/processed/benchmark_instances/atomic_ops.jsonl
data/processed/benchmark_instances/main.jsonl
data/processed/benchmark_instances/order_sensitivity.jsonl
```

## Build Eval-Ready Prompt Files

Prompt generation is separated from deterministic reference construction.

Template-only prompt construction, useful for reviewers without API access:

```bash
PYTHONPATH=src ./scripts/run_prompt_pipeline_all_tracks.sh \
  --benchmark-dir data/processed/benchmark_instances \
  --benchmark-output-root data/benchmark \
  --prompt-source template \
  --tracks atomic_ops,main,order_sensitivity \
  --skip-judge \
  --no-prompt-api-key
```

LLM-generated prompt variants with an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=<your_api_key>

PYTHONPATH=src ./scripts/run_prompt_pipeline_all_tracks.sh \
  --benchmark-dir data/processed/benchmark_instances \
  --benchmark-output-root data/benchmark \
  --prompt-source llm \
  --model <model_name> \
  --base-url <openai_compatible_base_url> \
  --tracks atomic_ops,main,order_sensitivity
```

Eval-ready benchmark files are written to:

```text
data/benchmark/atomic_ops/atomic_ops.jsonl
data/benchmark/main/main.jsonl
data/benchmark/order_sensitivity/order_sensitivity.jsonl
```

## Run Inference

The inference driver reads eval-ready JSONL files and writes raw model predictions. It supports remote OpenAI-compatible APIs and local vLLM servers.

Remote API example:

```bash
PYTHONPATH=src ./scripts/infer_benchmark_tracks.sh \
  --eval-root data/benchmark \
  --output-root data/evaluation \
  --tracks atomic_ops,main,order_sensitivity \
  --model-dirname my_model \
  --model <model_name> \
  --base-url <openai_compatible_base_url> \
  --api-key <your_api_key> \
  --resume
```

Local vLLM example:

```bash
PYTHONPATH=src ./scripts/infer_benchmark_tracks.sh \
  --eval-root data/benchmark \
  --output-root data/evaluation \
  --tracks atomic_ops,main,order_sensitivity \
  --model-dirname local_model \
  --model local-model \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --resume
```

For smoke tests, add `--max-samples 20`.

## Score Predictions

After inference, compute CDR-Bench metrics:

```bash
PYTHONPATH=src ./scripts/score_benchmark_tracks.sh \
  --predictions-root data/evaluation \
  --model-dirname my_model \
  --tracks atomic_ops,main,order_sensitivity
```

Each track writes:

```text
data/evaluation/<track>/<model_dirname>/predictions.jsonl
data/evaluation/<track>/<model_dirname>/predictions.summary.json
data/evaluation/<track>/<model_dirname>/score/report.txt
data/evaluation/<track>/<model_dirname>/score/paper_metrics.json
data/evaluation/<track>/<model_dirname>/score/instance_metrics.jsonl
data/evaluation/<track>/<model_dirname>/score/by_*.csv
```

Core metrics:

- `status_match`: predicted `KEEP` / `DROP` equals the deterministic reference
- `text_exact_match`: predicted clean text exactly equals the reference text
- `norm_recipe_success`: status match plus normalized text exact match
- `refinement_gain`: normalized edit-distance improvement from input toward the reference

## Important Files

- `configs/domains.yaml`: domain-to-operator plan
- `configs/recipe_prompting.yaml`: prompt styles and output contract
- `src/cdrbench/prepare_data/tag_and_assign_domains.py`: operator tagging and domain assignment
- `src/cdrbench/prepare_data/mine_domain_recipes.py`: recipe mining
- `src/cdrbench/prepare_data/materialize_domain_recipes.py`: deterministic recipe replay
- `src/cdrbench/prepare_data/materialize_benchmark_instances.py`: benchmark sampling and reference generation
- `src/cdrbench/prompting/generate_recipe_prompt_library.py`: prompt candidate generation and judging
- `src/cdrbench/prompting/build_eval_prompt_tracks.py`: eval-ready track construction
- `src/cdrbench/eval/run_benchmark_infer.py`: model inference
- `src/cdrbench/eval/run_benchmark_score.py`: metric computation

## Notes For Anonymous Review

This folder is designed for code upload during anonymous review. Dataset hosting URLs, model endpoint URLs, API keys, model-specific experiment wrappers, paper analysis code, and generated outputs should be added separately only when they are appropriate for the submission stage.
