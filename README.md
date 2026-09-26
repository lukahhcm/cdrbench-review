# CDR-Bench

Core utilities for loading benchmark JSONL files, running model inference,
scoring predictions, and summarizing results.

## Installation

Use Python 3.10 or newer. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The client uses OpenAI-compatible HTTP endpoints. For local GPU inference,
install vLLM in a separate Linux/CUDA environment and start a server there.

## Benchmark data

Download from the dataset repository supplied with the benchmark:

```bash
bash scripts/download_benchmark.sh --repo-id DATASET_REPO_ID
bash scripts/validate_benchmark.sh
```

`--repo-id` is required. Replace `DATASET_REPO_ID` with the supplied dataset
identifier. Use `--revision` to pin a dataset version. Alternatively, place
existing JSONL files in the following layout:

```text
data/benchmark_v3/
  tracks/                       evaluation-ready track files
    atomic_m.jsonl
    atomic_f.jsonl
    agnostic_m.jsonl
    order_m.jsonl
    order_f.jsonl
    semantic_*.jsonl
  tracks_all_prompts/           optional full prompt pools
```

Validate another location by passing JSONL paths explicitly:

```bash
bash scripts/validate_benchmark.sh /path/to/tracks/*.jsonl
```

## Tracks and output rules

| Track | Task | Output |
| --- | --- | --- |
| `atomic_m` | One mapper | Refined text |
| `atomic_f` | One filter | `KEEP` / `DROP` decision |
| `agnostic_m` | Multiple mappers | Refined text |
| `order_m` | Mapper order variants | Refined text |
| `order_f` | Filter placement variants | Decision and text at the stopping point |
| `semantic_pii_*` | PII redaction | Tagged text |
| `semantic_hallu_*` | Hallucination processing | JSON or tagged text |
| `semantic_rubric_*` | Rubric scoring | JSON |
| `semantic_safety_*` | Safety tagging | JSON |

Apply operations in the given order to the current intermediate text. If a
filter rejects a sample, stop immediately and return `DROP` with the text at
that point. Otherwise return `KEEP` with the final text. For tagged-text rows,
use:

```text
<status>KEEP</status><clean_text>refined text</clean_text>
```

Structured rows use the JSON fields requested by their prompt. The row fields
`output_format`, `scoring_profile`, and `reports_refinement_gain` determine the
output contract and applicable metrics. Schema validation is implemented in
`src/cdrbench_v3/schema.py`.

## Run evaluation

Hosted OpenAI-compatible endpoint:

```bash
MODEL=my-model \
MODEL_SLUG=my_model \
BASE_URL=https://api.example.com/v1 \
API_KEY=YOUR_API_KEY \
bash scripts/eval/api/eval_model.sh
```

Local vLLM endpoint:

```bash
MODEL=local-model \
MODEL_SLUG=local_model \
BASE_URL=http://127.0.0.1:8000/v1 \
API_KEY=EMPTY \
bash scripts/eval/vllm/eval_model.sh
```

The wrappers accept `infer`, `score`, or `all` (default). Set `EVAL_SUITE=semantic`
to evaluate the semantic extensions. Set `TRACKS=atomic_m` to restrict execution
to one track, and `MAX_SAMPLES=10` for a small smoke run.

Common controls:

| Variable | Default / purpose |
| --- | --- |
| `EVAL_SUITE` | `main`; also accepts `semantic` |
| `BENCHMARK_ROOT` | `data/benchmark_v3` |
| `BENCHMARK_TRACKS_SUBDIR` | `tracks` |
| `EVALUATION_ROOT` | `data/evaluation` |
| `PROMPT_MODE` | `direct`; also `few_shot`, `plan_first`, `state_aware` |
| `PROMPT_VARIANT_SAMPLE_SIZE` | `3` |
| `PROMPT_VARIANT_SAMPLING_SEED` | `0` |
| `TEMPERATURE` | `0` |
| `ENABLE_THINKING` | `false` |
| `CONCURRENCY` | `4` for API; `128` for vLLM |
| `RESUME` | `true` |

The default core track files contain three preselected prompt variants using
seed 0. For custom prompt sampling, use `tracks_all_prompts` and explicitly set
the sample size and seed. Semantic evaluation defaults to the styles `direct`,
`imperative_checklist`, and `application_context`. Keep data versions, prompt
selection, decoding settings, and scoring settings fixed when comparing runs.
The evaluation wrappers require `MAX_TOKENS=0`; configure the model/server
context length instead of truncating output in the runner.

For a direct single-file inference run:

```bash
bash scripts/run_inference.sh \
  --benchmark-path data/benchmark_v3/tracks/atomic_m.jsonl \
  --output-path data/results/atomic_m/my_model/predictions.jsonl \
  --model my-model \
  --backend api \
  --base-url https://api.example.com/v1 \
  --prompt-variant-indices 0 \
  --max-samples 10
```

Set `OPENAI_API_KEY` for this lower-level API command. API keys and generated
outputs should remain outside version control.

## Scoring and summaries

The evaluation wrappers score predictions automatically in `all` mode. To
score a separate prediction file:

```bash
bash scripts/score_predictions.sh \
  --predictions-path data/results/atomic_m/my_model/predictions.jsonl \
  --output-dir data/results/atomic_m/my_model/score \
  --rs-at-k 3 \
  --write-csv
```

Core metrics:

- **Recipe Success (RS):** matching status and normalized reference text;
  structured rows use canonical JSON comparison.
- **RS@K:** success in any of the selected K prompt variants, selected
  deterministically using the configured seed. All K predictions must be
  present and valid; incomplete sets count as unsuccessful.
- **Refinement Gain (RG):** edit-distance improvement toward the reference,
  clipped to `[0, 1]`; only reported for applicable text-output rows.
- **Order-Consistent Success (OCS):** success across all evaluated variants in
  an order-sensitive group. Evaluate complete groups for meaningful scores.

Default core evaluation outputs:

```text
data/evaluation/<track>/<model_slug>/
  predictions_direct_k3_seed0.jsonl
  score_direct_k3_seed0/
    summary.json
    metrics.json
    instance_metrics.jsonl
    instance_metrics.csv
    scored_variant_predictions.jsonl
    scored_variant_predictions.csv
```

Semantic outputs use `predictions_semantic_styles3.jsonl` and
`score_semantic_styles3/`. `metrics.json` contains the compact RS/RG/OCS summary;
`summary.json` includes aggregate breakdowns.

```bash
bash scripts/summarize_results.sh \
  --track-family main \
  --models my_model \
  --output-dir data/evaluation/reports/my_model
```

Use comma-separated model names and optionally `--base-model BASE_MODEL` for
comparisons. The command writes Markdown, JSON, and CSV reports.

## Code layout

```text
requirements.txt
src/cdrbench_v3/
  download_benchmark.py          dataset download
  schema.py                     row schema and track definitions
  validate_benchmark.py         JSONL validation
  run_inference.py               prompts, API calls, response parsing
  metrics.py                    metric definitions
  score_predictions.py          prediction scoring
  summarize_results.py          aggregate reports
  normalize_hallu_compositional.py
  io.py
scripts/
  download_benchmark.sh
  validate_benchmark.sh
  infer/                        single-file and suite inference
  score/                        single-file and suite scoring
  eval/                         shared evaluation runner and generic backends
  start_vllm.sh                  optional local serving helper
  stop_vllm.sh
```

Root-level inference and scoring scripts forward to their corresponding
subdirectories. Run `bash scripts/eval/run_model_eval.sh --help` for evaluation
controls, or append `--help` to the Python-backed entrypoints for CLI options.

## Data usage

Dataset subsets remain subject to their respective upstream licenses and terms.
Consult the documentation distributed with the benchmark data before using or
redistributing those files.
