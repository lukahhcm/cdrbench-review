from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cdrbench_v3.io import read_jsonl, write_json


CORE_TRACKS = ("atomic_m", "atomic_f", "agnostic_m", "order_m", "order_f")
SEMANTIC_DOMAINS = {
    "semantic_pii": {
        "label": "PII Redaction",
        "atomic": "semantic_pii_atomic",
        "compositional": "semantic_pii_compositional",
    },
    "semantic_hallucination": {
        "label": "Hallucination Processing",
        "atomic": "semantic_hallu_atomic",
        "compositional": "semantic_hallu_compositional",
    },
    "semantic_rubric": {
        "label": "Rubric Scoring",
        "atomic": "semantic_rubric_atomic",
        "compositional": "semantic_rubric_compositional",
    },
    "semantic_safety": {
        "label": "Safety Tagging",
        "atomic": "semantic_safety_atomic",
        "compositional": "semantic_safety_compositional",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.2f}"


def signed_pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:+.2f}"


def sort_key(value: Any) -> str:
    return str(value or "")


def discover_models(results_root: Path, score_dirname: str) -> list[str]:
    models: set[str] = set()
    for track_dir in results_root.iterdir() if results_root.exists() else []:
        if not track_dir.is_dir():
            continue
        for model_dir in track_dir.iterdir():
            if (model_dir / score_dirname / "summary.json").exists():
                models.add(model_dir.name)
    return sorted(models)


def discover_models_multi(results_root: Path, score_dirnames: list[str]) -> list[str]:
    models: set[str] = set()
    for score_dirname in score_dirnames:
        models.update(discover_models(results_root, score_dirname))
    return sorted(models)


def summary_path(results_root: Path, track: str, model: str, score_dirname: str) -> Path:
    return results_root / track / model / score_dirname / "summary.json"


def instance_metrics_path(results_root: Path, track: str, model: str, score_dirname: str) -> Path:
    return results_root / track / model / score_dirname / "instance_metrics.jsonl"


def overall_rs_at_k(path: Path) -> tuple[float | None, int | None]:
    if not path.exists():
        return None, None
    summary = read_json(path)
    overall = summary.get("overall") or {}
    return overall.get("mean_rs_at_k"), overall.get("num_instances")


def atomic_macro_rs_at_k(path: Path) -> tuple[float | None, int | None, dict[str, float]]:
    if not path.exists():
        return None, None, {}
    rows = read_jsonl(path)
    by_subtask: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        subtask = str(row.get("semantic_operator") or row.get("operator") or row.get("source_track") or "atomic")
        if row.get("rs_at_k") is None:
            continue
        by_subtask[subtask].append(1.0 if row.get("rs_at_k") else 0.0)
    subtask_scores = {key: mean(values) or 0.0 for key, values in sorted(by_subtask.items())}
    return mean(list(subtask_scores.values())), len(rows), subtask_scores


def add_group_comparisons(
    rows: list[dict[str, Any]],
    value_key: str,
    group_keys: list[str],
    base_model: str,
) -> None:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(sort_key(row.get(key)) for key in group_keys)].append(row)

    for group_rows in grouped.values():
        available = [row for row in group_rows if isinstance(row.get(value_key), (int, float))]
        if not available:
            continue
        ranked = sorted(available, key=lambda row: (-float(row[value_key]), sort_key(row.get("model"))))
        best = ranked[0]
        best_value = float(best[value_key])
        base_rows = [row for row in available if row.get("model") == base_model]
        base_value = float(base_rows[0][value_key]) if base_rows else None
        for rank, row in enumerate(ranked, start=1):
            value = float(row[value_key])
            row["rank"] = rank
            row["best_model"] = best.get("model")
            row["best_value"] = best_value
            row["best_value_pct"] = pct(best_value)
            row["delta_vs_best"] = value - best_value
            row["delta_vs_best_pp"] = signed_pct(value - best_value)
            row["shortfall_to_best"] = best_value - value
            row["shortfall_to_best_pp"] = pct(best_value - value)
            if base_value is not None:
                row["base_model"] = base_model
                row["base_value"] = base_value
                row["base_value_pct"] = pct(base_value)
                row["delta_vs_base"] = value - base_value
                row["delta_vs_base_pp"] = signed_pct(value - base_value)


def collect_core(results_root: Path, models: list[str], score_dirname: str, rs_at_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for track in CORE_TRACKS:
            value, n = overall_rs_at_k(summary_path(results_root, track, model, score_dirname))
            rows.append(
                {
                    "model": model,
                    "track_family": "core_rule",
                    "track": track,
                    "metric": f"RS@{rs_at_k}",
                    "n": n,
                    "rs_at_k": value,
                    "rs_at_k_pct": pct(value),
                    "score_path": str(summary_path(results_root, track, model, score_dirname)),
                }
            )
    return rows


def aggregate_core(core_rows: list[dict[str, Any]], rs_at_k: int) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in core_rows:
        by_model[str(row["model"])].append(row)
    rows: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        values = [float(row["rs_at_k"]) for row in model_rows if isinstance(row.get("rs_at_k"), (int, float))]
        rows.append(
            {
                "model": model,
                "track_family": "core_rule",
                "metric": f"Average RS@{rs_at_k}",
                "num_tracks": len(CORE_TRACKS),
                "available_tracks": len(values),
                "missing_tracks": len(CORE_TRACKS) - len(values),
                "avg_rs_at_k": mean(values),
                "avg_rs_at_k_pct": pct(mean(values)),
            }
        )
    return rows


def collect_semantic(results_root: Path, models: list[str], score_dirname: str, rs_at_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for domain, spec in SEMANTIC_DOMAINS.items():
            atomic_path = instance_metrics_path(results_root, spec["atomic"], model, score_dirname)
            comp_summary_path = summary_path(results_root, spec["compositional"], model, score_dirname)
            atomic_avg, atomic_n, subtask_scores = atomic_macro_rs_at_k(atomic_path)
            comp_value, comp_n = overall_rs_at_k(comp_summary_path)
            gap = (atomic_avg - comp_value) if atomic_avg is not None and comp_value is not None else None
            rows.append(
                {
                    "model": model,
                    "track_family": "semantic_extension",
                    "domain": domain,
                    "domain_label": spec["label"],
                    "metric": f"RS@{rs_at_k}",
                    "atomic_track": spec["atomic"],
                    "compositional_track": spec["compositional"],
                    "atomic_n": atomic_n,
                    "compositional_n": comp_n,
                    "atomic_avg_rs_at_k": atomic_avg,
                    "compositional_rs_at_k": comp_value,
                    "gap_atomic_avg_minus_compositional": gap,
                    "atomic_avg_rs_at_k_pct": pct(atomic_avg),
                    "compositional_rs_at_k_pct": pct(comp_value),
                    "gap_pct_points": signed_pct(gap),
                    "atomic_subtask_rs_at_k": json.dumps(subtask_scores, ensure_ascii=False, sort_keys=True),
                    "atomic_metrics_path": str(atomic_path),
                    "compositional_summary_path": str(comp_summary_path),
                }
            )
    return rows


def aggregate_semantic(semantic_rows: list[dict[str, Any]], rs_at_k: int) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in semantic_rows:
        by_model[str(row["model"])].append(row)
    rows: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        atomic_values = [
            float(row["atomic_avg_rs_at_k"]) for row in model_rows if isinstance(row.get("atomic_avg_rs_at_k"), (int, float))
        ]
        comp_values = [
            float(row["compositional_rs_at_k"])
            for row in model_rows
            if isinstance(row.get("compositional_rs_at_k"), (int, float))
        ]
        gap_values = [
            float(row["gap_atomic_avg_minus_compositional"])
            for row in model_rows
            if isinstance(row.get("gap_atomic_avg_minus_compositional"), (int, float))
        ]
        atomic_avg = mean(atomic_values)
        comp_avg = mean(comp_values)
        gap_avg = mean(gap_values)
        rows.append(
            {
                "model": model,
                "track_family": "semantic_extension",
                "metric": f"Average RS@{rs_at_k}",
                "gap_definition": f"Atomic Avg RS@{rs_at_k} - Compositional RS@{rs_at_k}",
                "num_domains": len(SEMANTIC_DOMAINS),
                "available_atomic_domains": len(atomic_values),
                "available_compositional_domains": len(comp_values),
                "missing_domains": len(SEMANTIC_DOMAINS) - min(len(atomic_values), len(comp_values)),
                "semantic_atomic_avg_rs_at_k": atomic_avg,
                "semantic_compositional_avg_rs_at_k": comp_avg,
                "semantic_gap_avg": gap_avg,
                "semantic_atomic_avg_rs_at_k_pct": pct(atomic_avg),
                "semantic_compositional_avg_rs_at_k_pct": pct(comp_avg),
                "semantic_gap_avg_pp": signed_pct(gap_avg),
            }
        )
    return rows


def build_waterline_rows(
    core_model_rows: list[dict[str, Any]],
    semantic_model_rows: list[dict[str, Any]],
    rs_at_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in core_model_rows:
        rows.append(
            {
                "model": row["model"],
                "suite": "core_rule",
                "primary_metric": f"Core Avg RS@{rs_at_k}",
                "primary_value": row.get("avg_rs_at_k"),
                "primary_value_pct": row.get("avg_rs_at_k_pct"),
                "available_units": row.get("available_tracks"),
                "missing_units": row.get("missing_tracks"),
                "rank": row.get("rank"),
                "best_model": row.get("best_model"),
                "best_value_pct": row.get("best_value_pct"),
                "delta_vs_best_pp": row.get("delta_vs_best_pp"),
                "shortfall_to_best_pp": row.get("shortfall_to_best_pp"),
                "base_model": row.get("base_model"),
                "base_value_pct": row.get("base_value_pct"),
                "delta_vs_base_pp": row.get("delta_vs_base_pp"),
            }
        )
    for row in semantic_model_rows:
        rows.append(
            {
                "model": row["model"],
                "suite": "semantic_extension",
                "primary_metric": f"Semantic Comp Avg RS@{rs_at_k}",
                "primary_value": row.get("semantic_compositional_avg_rs_at_k"),
                "primary_value_pct": row.get("semantic_compositional_avg_rs_at_k_pct"),
                "secondary_metric": f"Semantic Atomic Avg RS@{rs_at_k}",
                "secondary_value_pct": row.get("semantic_atomic_avg_rs_at_k_pct"),
                "gap_metric": f"Atomic Avg RS@{rs_at_k} - Compositional RS@{rs_at_k}",
                "gap_pp": row.get("semantic_gap_avg_pp"),
                "available_units": row.get("available_compositional_domains"),
                "missing_units": row.get("missing_domains"),
                "rank": row.get("rank"),
                "best_model": row.get("best_model"),
                "best_value_pct": row.get("best_value_pct"),
                "delta_vs_best_pp": row.get("delta_vs_best_pp"),
                "shortfall_to_best_pp": row.get("shortfall_to_best_pp"),
                "base_model": row.get("base_model"),
                "base_value_pct": row.get("base_value_pct"),
                "delta_vs_base_pp": row.get("delta_vs_base_pp"),
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    waterline_rows: list[dict[str, Any]],
    core_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    rs_at_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# CDR-Bench Result Summary", "", f"Metric: RS@{rs_at_k}", ""]
    if waterline_rows:
        lines.extend(
            [
                "## Model Waterline",
                "",
                markdown_table(
                    [
                        "Model",
                        "Suite",
                        "Primary",
                        "Rank",
                        "Best",
                        "Delta vs Best",
                        "Base",
                        "Delta vs Base",
                        "Missing",
                    ],
                    [
                        [
                            r["model"],
                            r["suite"],
                            r["primary_value_pct"],
                            r.get("rank"),
                            r.get("best_model"),
                            r.get("delta_vs_best_pp"),
                            r.get("base_model"),
                            r.get("delta_vs_base_pp"),
                            r.get("missing_units"),
                        ]
                        for r in waterline_rows
                    ],
                ),
                "",
            ]
        )
    if core_rows:
        lines.extend(
            [
                "## Core Tracks",
                "",
                markdown_table(
                    ["Model", "Track", "N", f"RS@{rs_at_k}", "Rank", "Delta vs Best", "Delta vs Base"],
                    [
                        [
                            r["model"],
                            r["track"],
                            r["n"],
                            r["rs_at_k_pct"],
                            r.get("rank"),
                            r.get("delta_vs_best_pp"),
                            r.get("delta_vs_base_pp"),
                        ]
                        for r in core_rows
                    ],
                ),
                "",
            ]
        )
    if semantic_rows:
        lines.extend(
            [
                "## Semantic Extension",
                "",
                f"Gap is defined as Atomic Avg RS@{rs_at_k} - Compositional RS@{rs_at_k}.",
                "",
                markdown_table(
                    [
                        "Model",
                        "Domain",
                        "Atomic N",
                        "Comp N",
                        f"Atomic Avg RS@{rs_at_k}",
                        f"Comp RS@{rs_at_k}",
                        "Gap",
                        "Rank",
                        "Delta vs Best",
                        "Delta vs Base",
                    ],
                    [
                        [
                            r["model"],
                            r["domain_label"],
                            r["atomic_n"],
                            r["compositional_n"],
                            r["atomic_avg_rs_at_k_pct"],
                            r["compositional_rs_at_k_pct"],
                            r["gap_pct_points"],
                            r.get("rank"),
                            r.get("delta_vs_best_pp"),
                            r.get("delta_vs_base_pp"),
                        ]
                        for r in semantic_rows
                    ],
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CDR-Bench score outputs into user-facing tables.")
    parser.add_argument("--results-root", default="data/evaluation")
    parser.add_argument("--output-dir", default="data/evaluation/reports")
    parser.add_argument("--track-family", choices=["core_rule", "main", "semantic_extension", "semantic", "all"], default="all")
    parser.add_argument("--models", default="", help="Comma-separated model directory names. Default: discover from score dirs.")
    parser.add_argument("--base-model", default="", help="Optional model directory name used for delta-vs-base columns.")
    parser.add_argument("--score-dirname", default="", help="Override both core and semantic score dir names.")
    parser.add_argument("--core-score-dirname", default="score_direct_k3_seed0")
    parser.add_argument("--semantic-score-dirname", default="score_semantic_styles3")
    parser.add_argument("--rs-at-k", type=int, default=3)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    core_score_dirname = args.score_dirname or args.core_score_dirname
    semantic_score_dirname = args.score_dirname or args.semantic_score_dirname

    include_core = args.track_family in {"all", "main", "core_rule"}
    include_semantic = args.track_family in {"all", "semantic", "semantic_extension"}
    default_score_dirs = []
    if include_core:
        default_score_dirs.append(core_score_dirname)
    if include_semantic:
        default_score_dirs.append(semantic_score_dirname)
    models = [item.strip() for item in args.models.split(",") if item.strip()] or discover_models_multi(results_root, default_score_dirs)

    core_rows = collect_core(results_root, models, core_score_dirname, args.rs_at_k) if include_core else []
    semantic_rows = collect_semantic(results_root, models, semantic_score_dirname, args.rs_at_k) if include_semantic else []
    add_group_comparisons(core_rows, "rs_at_k", ["track"], args.base_model)
    add_group_comparisons(semantic_rows, "compositional_rs_at_k", ["domain"], args.base_model)

    core_model_rows = aggregate_core(core_rows, args.rs_at_k) if core_rows else []
    semantic_model_rows = aggregate_semantic(semantic_rows, args.rs_at_k) if semantic_rows else []
    add_group_comparisons(core_model_rows, "avg_rs_at_k", ["track_family"], args.base_model)
    add_group_comparisons(
        semantic_model_rows,
        "semantic_compositional_avg_rs_at_k",
        ["track_family"],
        args.base_model,
    )
    waterline_rows = build_waterline_rows(core_model_rows, semantic_model_rows, args.rs_at_k)

    output_dir.mkdir(parents=True, exist_ok=True)
    if core_rows:
        write_csv(output_dir / "core_summary.csv", core_rows)
    if semantic_rows:
        write_csv(output_dir / "semantic_summary.csv", semantic_rows)
    if core_model_rows:
        write_csv(output_dir / "core_model_summary.csv", core_model_rows)
    if semantic_model_rows:
        write_csv(output_dir / "semantic_model_summary.csv", semantic_model_rows)
    if waterline_rows:
        write_csv(output_dir / "model_waterline.csv", waterline_rows)
    write_json(
        output_dir / "summary.json",
        {
            "metric": f"RS@{args.rs_at_k}",
            "core_score_dirname": core_score_dirname,
            "semantic_score_dirname": semantic_score_dirname,
            "models": models,
            "base_model": args.base_model or None,
            "waterline_rows": waterline_rows,
            "core_model_rows": core_model_rows,
            "semantic_model_rows": semantic_model_rows,
            "core_rows": core_rows,
            "semantic_rows": semantic_rows,
            "semantic_gap_definition": f"Atomic Avg RS@{args.rs_at_k} - Compositional RS@{args.rs_at_k}",
        },
    )
    write_markdown(output_dir / "summary.md", waterline_rows, core_rows, semantic_rows, args.rs_at_k)
    print(f"wrote summary report -> {output_dir}")


if __name__ == "__main__":
    main()
