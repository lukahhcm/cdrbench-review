from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cdrbench_v3.io import read_jsonl, write_json, write_jsonl
from cdrbench_v3.metrics import compare_json, compare_text, parse_jsonish
from cdrbench_v3.schema import is_json_reference


def _variant_predictions(row: dict[str, Any]) -> list[dict[str, Any]]:
    variants = row.get("variant_predictions")
    if isinstance(variants, list) and variants:
        return [variant for variant in variants if isinstance(variant, dict)]
    if "predicted_clean_text" in row or "parsed_response" in row or "raw_response" in row:
        return [row]
    return []


def _predicted_text(variant: dict[str, Any]) -> str:
    for key in ("predicted_clean_text", "clean_text", "text", "output"):
        if variant.get(key) is not None:
            return str(variant.get(key))
    parsed = variant.get("parsed_response")
    if isinstance(parsed, dict):
        for key in ("clean_text", "text", "output", "answer"):
            if parsed.get(key) is not None:
                return str(parsed.get(key))
    if variant.get("raw_response") is not None:
        return str(variant.get("raw_response"))
    return ""


def _predicted_status(variant: dict[str, Any]) -> str:
    if variant.get("predicted_status") is not None:
        return str(variant.get("predicted_status"))
    parsed = variant.get("parsed_response")
    if isinstance(parsed, dict) and parsed.get("status") is not None:
        return str(parsed.get("status"))
    return "KEEP"


def _valid_prediction(variant: dict[str, Any]) -> bool:
    if "valid_prediction" in variant:
        return bool(variant.get("valid_prediction"))
    if "prediction_valid_json" in variant:
        return bool(variant.get("prediction_valid_json"))
    return not bool(variant.get("prediction_error"))


def _score_variant(row: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    predicted_text = _predicted_text(variant)
    predicted_status = _predicted_status(variant)
    valid_prediction = _valid_prediction(variant)
    scoring_profile = row.get("scoring_profile")
    if not scoring_profile:
        scoring_profile = "structured_json" if is_json_reference(row.get("reference_text")) else "text_refinement"

    base = {
        "instance_id": row.get("instance_id"),
        "benchmark_track": row.get("benchmark_track"),
        "benchmark_split": row.get("benchmark_split") or ("single" if not str(row.get("benchmark_track", "")).startswith("semantic_") else None),
        "source_track": row.get("source_track") or row.get("benchmark_track"),
        "track_family": row.get("track_family") or ("semantic_extension" if str(row.get("benchmark_track", "")).startswith("semantic_") else "core_rule"),
        "domain": row.get("domain"),
        "source_domain": row.get("source_domain"),
        "operator": row.get("operator"),
        "operator_kind": row.get("operator_kind"),
        "semantic_operator": row.get("semantic_operator"),
        "reference_status": row.get("reference_status"),
        "order_family_id": row.get("order_family_id"),
        "order_slot": row.get("order_slot"),
        "order_group_instance_id": row.get("order_group_instance_id"),
        "required_slots_for_group_success": row.get("required_slots_for_group_success"),
        "scoring_profile": scoring_profile,
        "output_format": row.get("output_format"),
        "prompt_variant_index": int(variant.get("prompt_variant_index", 0) or 0),
        "prompt_style_id": variant.get("prompt_style_id"),
        "prediction_error": variant.get("prediction_error"),
        "valid_prediction": valid_prediction,
        "scorable_prediction": valid_prediction,
        "reports_refinement_gain": bool(row.get("reports_refinement_gain")),
    }
    if not valid_prediction:
        base.update({"recipe_success": False, "primary_score": 0.0})
        return base

    if scoring_profile == "structured_json":
        metrics = compare_json(row.get("reference_text"), predicted_text)
        recipe_success = bool(metrics["json_exact_match"])
        base.update(metrics)
        base.update(
            {
                "recipe_success": recipe_success,
                "primary_score": 1.0 if recipe_success else 0.0,
                "refinement_gain": None,
            }
        )
        return base

    if scoring_profile == "mixed_structured_text":
        json_metrics = compare_json(row.get("reference_json") or row.get("reference_text"), predicted_text)
        parsed_prediction = parse_jsonish(predicted_text)
        predicted_refined_text = ""
        if isinstance(parsed_prediction, dict) and parsed_prediction.get("corrected_text") is not None:
            predicted_refined_text = str(parsed_prediction.get("corrected_text"))
        text_metrics = compare_text(
            input_text=row.get("input_text"),
            reference_status=row.get("reference_status"),
            reference_text=row.get("reference_text"),
            predicted_status=predicted_status,
            predicted_text=predicted_refined_text,
            reference_text_full_run=row.get("reference_text_full_run"),
        )
        recipe_success = bool(json_metrics["json_exact_match"]) and bool(text_metrics["recipe_success"])
        base.update(json_metrics)
        base.update(text_metrics)
        base.update({"recipe_success": recipe_success, "primary_score": 1.0 if recipe_success else 0.0})
        return base

    metrics = compare_text(
        input_text=row.get("input_text"),
        reference_status=row.get("reference_status"),
        reference_text=row.get("reference_text"),
        predicted_status=predicted_status,
        predicted_text=predicted_text,
        reference_text_full_run=row.get("reference_text_full_run"),
    )
    base.update(metrics)
    base["primary_score"] = 1.0 if metrics["recipe_success"] else 0.0
    return base


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = "||".join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def _sample_variants(
    row: dict[str, Any],
    variants: list[dict[str, Any]],
    *,
    sample_size: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(variants):
        return list(variants)
    recipe_prompt_key = str(row.get("recipe_prompt_key") or row.get("workflow_prompt_key") or row.get("recipe_id") or "")
    instance_id = str(row.get("instance_id") or "")
    ranked = sorted(
        variants,
        key=lambda variant: _stable_id(
            "prompt-variant-sample",
            sample_seed,
            recipe_prompt_key,
            instance_id,
            int(variant.get("prompt_variant_index", 0) or 0),
        ),
    )
    return sorted(ranked[:sample_size], key=lambda item: int(item.get("prompt_variant_index", 0) or 0))


def score_rows(
    rows: list[dict[str, Any]],
    *,
    sample_size: int = 0,
    sample_seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    variant_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    for row in rows:
        scored_variants = [_score_variant(row, variant) for variant in _variant_predictions(row)]
        scored_variants.sort(key=lambda item: int(item.get("prompt_variant_index", 0) or 0))
        scored_for_at_k = _sample_variants(row, scored_variants, sample_size=sample_size, sample_seed=sample_seed)
        variant_rows.extend(scored_variants)
        scorable_variants = [item for item in scored_variants if bool(item.get("scorable_prediction"))]
        scorable_at_k = [item for item in scored_for_at_k if bool(item.get("scorable_prediction"))]
        required_at_k_count = len(scored_variants) if sample_size <= 0 else sample_size
        at_k_complete = len(scored_for_at_k) >= required_at_k_count and len(scorable_at_k) >= required_at_k_count
        success_values = [bool(item.get("recipe_success")) for item in scorable_variants]
        success_at_k_values = [bool(item.get("recipe_success")) for item in scorable_at_k]
        prompt0_row = next((item for item in scored_variants if int(item.get("prompt_variant_index", 0) or 0) == 0), None)
        prompt0_success = (
            bool(prompt0_row.get("recipe_success"))
            if prompt0_row is not None and bool(prompt0_row.get("scorable_prediction"))
            else None
        )
        rg_values = [
            float(item["refinement_gain"])
            for item in scorable_variants
            if item.get("refinement_gain") is not None
        ]
        instance_rows.append(
            {
                "instance_id": row.get("instance_id"),
                "benchmark_track": row.get("benchmark_track"),
                "benchmark_split": row.get("benchmark_split"),
                "source_track": row.get("source_track"),
                "track_family": row.get("track_family"),
                "domain": row.get("domain"),
                "source_domain": row.get("source_domain"),
                "operator": row.get("operator"),
                "operator_kind": row.get("operator_kind"),
                "reference_status": row.get("reference_status"),
                "order_family_id": row.get("order_family_id"),
                "order_slot": row.get("order_slot"),
                "order_group_instance_id": row.get("order_group_instance_id"),
                "required_slots_for_group_success": row.get("required_slots_for_group_success"),
                "scoring_profile": row.get("scoring_profile"),
                "num_prompt_variants": len(scored_variants),
                "num_sampled_prompt_variants": len(scored_for_at_k),
                "num_scorable_variants": len(scorable_variants),
                "num_sampled_scorable_variants": len(scorable_at_k),
                "required_at_k_prompt_variants": required_at_k_count,
                "at_k_complete": at_k_complete,
                "rs": (sum(success_values) / len(success_values) if success_values else 0.0),
                "rs_at_k": (any(success_at_k_values) if at_k_complete else None),
                "rs_prompt0": prompt0_success,
                "mean_rg": (sum(rg_values) / len(rg_values) if rg_values else None),
                "reports_refinement_gain": bool(row.get("reports_refinement_gain")),
            }
        )
    summary = aggregate(instance_rows, variant_rows, sample_size=sample_size, sample_seed=sample_seed)
    for row in rows:
        if row.get("request_model"):
            summary["model"] = row.get("request_model")
            break
    return variant_rows, instance_rows, summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: list[Any]) -> float | None:
    clean: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            clean.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(clean) / len(clean) if clean else None


def _rate_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(1.0 if bool(value) else 0.0 for value in values) / len(values)


def _canonical_order_slot(value: Any) -> str:
    slot = str(value or "").strip().lower()
    return {
        "front": "pre",
        "pre": "pre",
        "middle": "mid",
        "mid": "mid",
        "end": "post",
        "post": "post",
    }.get(slot, slot)


def _build_order_group_rows(instance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        group_id = row.get("order_group_instance_id")
        if group_id:
            grouped[str(group_id)].append(row)
    output: list[dict[str, Any]] = []
    for group_id, bucket in sorted(grouped.items()):
        required_slots: list[str] = []
        for row in bucket:
            raw_required = row.get("required_slots_for_group_success")
            if isinstance(raw_required, list):
                required_slots.extend(str(slot) for slot in raw_required if str(slot or "").strip())
            elif isinstance(raw_required, str):
                required_slots.extend(slot.strip() for slot in raw_required.split("|") if slot.strip())
        required_slots = sorted({_canonical_order_slot(slot) for slot in required_slots})
        slot_bucket = bucket
        missing_required_slots: list[str] = []
        if required_slots:
            rows_by_slot = {_canonical_order_slot(row.get("order_slot")): row for row in bucket}
            missing_required_slots = [slot for slot in required_slots if slot not in rows_by_slot]
            slot_bucket = [rows_by_slot[slot] for slot in required_slots if slot in rows_by_slot]
        complete_group = not missing_required_slots and bool(slot_bucket)
        output.append(
            {
                "order_group_instance_id": group_id,
                "slot_count": len(bucket),
                "required_slots_for_group_success": required_slots,
                "missing_required_slots": missing_required_slots,
                "ocs_at_k": (
                    all(bool(row.get("rs_at_k")) for row in slot_bucket)
                    if complete_group and all(row.get("rs_at_k") is not None for row in slot_bucket)
                    else None
                ),
                "ocs": (
                    all(bool(row.get("rs_prompt0")) for row in slot_bucket)
                    if complete_group and all(row.get("rs_prompt0") is not None for row in slot_bucket)
                    else None
                ),
            }
        )
    return output


def _slot_summary(instance_rows: list[dict[str, Any]], slot: str) -> dict[str, Any]:
    slot_rows = [row for row in instance_rows if _canonical_order_slot(row.get("order_slot")) == slot]
    return {
        f"rs_{slot}@k": _rate_optional(slot_rows, "rs_at_k"),
        f"rg_{slot}": _mean_optional([row.get("mean_rg") for row in slot_rows]),
    }


def _operator_kind_summary(instance_rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    rows = [row for row in instance_rows if str(row.get("operator_kind") or "") == kind]
    return {
        f"atomic_{kind}_rs@k": _rate_optional(rows, "rs_at_k"),
    }


def _metrics_payload(summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary.get("overall") or {}
    payload: dict[str, Any] = {
        "track": summary.get("track"),
        "model": summary.get("model"),
        "num_instances": summary.get("num_instances"),
        "prompt_variant_sample_size": summary.get("prompt_variant_sample_size"),
        "prompt_variant_sampling_seed": summary.get("prompt_variant_sampling_seed"),
        "mean_rs@k": overall.get("mean_rs_at_k"),
        "mean_rs": overall.get("mean_rs"),
        "mean_rg": overall.get("mean_rg"),
    }
    for key in (
        "ocs",
        "ocs_at_k",
        "rs_pre@k",
        "rg_pre",
        "rs_mid@k",
        "rg_mid",
        "rs_post@k",
        "rg_post",
        "atomic_mapper_rs@k",
        "atomic_filter_rs@k",
    ):
        if key in summary:
            payload[key] = summary[key]
    return payload


def _summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rs_values = [float(row.get("rs", 0.0)) for row in rows]
    rs0_values = [1.0 if row.get("rs_prompt0") else 0.0 for row in rows if row.get("rs_prompt0") is not None]
    rsk_values = [1.0 if row.get("rs_at_k") else 0.0 for row in rows if row.get("rs_at_k") is not None]
    rg_values = [float(row["mean_rg"]) for row in rows if row.get("mean_rg") is not None]
    return {
        "num_instances": len(rows),
        "mean_rs": _mean(rs_values),
        "mean_rs_prompt0": _mean(rs0_values),
        "mean_rs_at_k": _mean(rsk_values),
        "mean_rg": (_mean(rg_values) if rg_values else None),
        "num_rg_instances": len(rg_values),
    }


def aggregate(
    instance_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    *,
    sample_size: int,
    sample_seed: int,
) -> dict[str, Any]:
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scoring: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        by_track[str(row.get("benchmark_track"))].append(row)
        by_source_track[str(row.get("source_track"))].append(row)
        by_family[str(row.get("track_family"))].append(row)
        by_scoring[str(row.get("scoring_profile"))].append(row)

    variant_counts = Counter(row.get("prompt_variant_index") for row in variant_rows)
    order_group_rows = _build_order_group_rows(instance_rows)
    track_names = [str(row.get("benchmark_track") or "") for row in instance_rows]
    track = next((name for name in track_names if name), None)
    summary = {
        "track": track,
        "num_instances": len(instance_rows),
        "num_variant_predictions": len(variant_rows),
        "prompt_variant_sample_size": sample_size if sample_size > 0 else "all",
        "prompt_variant_sample_size_for_rs_at_k": sample_size if sample_size > 0 else "all",
        "prompt_variant_sampling_seed": sample_seed,
        "variant_index_counts": {str(key): value for key, value in sorted(variant_counts.items())},
        "overall": _summarize_bucket(instance_rows),
        "by_track_family": {key: _summarize_bucket(value) for key, value in sorted(by_family.items())},
        "by_benchmark_track": {key: _summarize_bucket(value) for key, value in sorted(by_track.items())},
        "by_source_track": {key: _summarize_bucket(value) for key, value in sorted(by_source_track.items())},
        "by_scoring_profile": {key: _summarize_bucket(value) for key, value in sorted(by_scoring.items())},
    }
    if order_group_rows:
        summary["num_order_groups"] = len(order_group_rows)
        summary["ocs_at_k"] = _rate_optional(order_group_rows, "ocs_at_k")
        summary["ocs"] = _rate_optional(order_group_rows, "ocs")
        for slot in ("pre", "mid", "post"):
            summary.update(_slot_summary(instance_rows, slot))
    summary.update(_operator_kind_summary(instance_rows, "mapper"))
    summary.update(_operator_kind_summary(instance_rows, "filter"))
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CDR-Bench v3 predictions.")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rs-at-k", type=int, default=0, help="Use K deterministic prompt variants for RS@K; 0 means all variants.")
    parser.add_argument("--prompt-variant-sampling-seed", type=int, default=0)
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = read_jsonl(predictions_path)
    variant_rows, instance_rows, summary = score_rows(
        rows,
        sample_size=args.rs_at_k,
        sample_seed=args.prompt_variant_sampling_seed,
    )
    summary["predictions_path"] = str(predictions_path)
    summary["output_dir"] = str(output_dir)

    write_jsonl(output_dir / "scored_variant_predictions.jsonl", variant_rows)
    write_jsonl(output_dir / "instance_metrics.jsonl", instance_rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "metrics.json", _metrics_payload(summary))
    if args.write_csv:
        _write_csv(output_dir / "instance_metrics.csv", instance_rows)
        _write_csv(output_dir / "scored_variant_predictions.csv", variant_rows)

    print(f"Scored {len(instance_rows)} instances from {predictions_path}")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
