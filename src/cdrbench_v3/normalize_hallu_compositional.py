from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cdrbench_v3.io import read_jsonl, write_json, write_jsonl
from cdrbench_v3.schema import order_v3_row


HALLU_ATOMIC_OPERATORS = (
    "hallucination_detection_mapper",
    "hallucination_span_extractor",
    "hallucination_type_classifier",
    "hallucination_correction_mapper",
)


PROMPT_VARIANTS = (
    (
        "direct",
        "Direct",
        (
            "You are a hallucination analysis and correction engine. Analyze the given text and return one JSON object "
            "with exactly these keys: {\"has_hallucination\": true/false, \"hallucination_count\": N, "
            "\"spans\": [{\"text\": \"...\", \"type\": \"...\"}], \"types\": [\"...\"], "
            "\"corrected_text\": \"...\"}. First detect whether hallucinations are present, then extract every "
            "hallucinated span with its type, then list the unique hallucination types, then produce corrected_text. "
            "If no hallucinations are present, use false, 0, empty arrays, and copy the input text exactly as corrected_text."
        ),
    ),
    (
        "imperative_checklist",
        "Imperative Checklist",
        (
            "Process the text through the full hallucination pipeline and return JSON only. Step 1: decide "
            "has_hallucination and hallucination_count. Step 2: extract each hallucinated span as an object with "
            "text and type. Step 3: output the unique types. Step 4: output corrected_text after removing or correcting "
            "hallucinated content while preserving factual content. Required JSON shape: "
            "{\"has_hallucination\": true/false, \"hallucination_count\": N, "
            "\"spans\": [{\"text\": \"...\", \"type\": \"...\"}], \"types\": [\"...\"], "
            "\"corrected_text\": \"...\"}."
        ),
    ),
    (
        "application_context",
        "Application Context",
        (
            "For an LLM output quality assurance pipeline, run detection, span extraction, type classification, "
            "and correction in one pass. Return a single JSON object with has_hallucination, hallucination_count, "
            "spans, types, and corrected_text. The corrected_text must be identical to the input when the text is clean; "
            "otherwise remove or correct unsupported, fabricated, contradictory, or unverifiable content."
        ),
    ),
)


def _json_loads(text: Any) -> Any:
    return json.loads("" if text is None else str(text))


def _atomic_by_base_sample(atomic_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_base: dict[str, dict[str, dict[str, Any]]] = {}
    for row in atomic_rows:
        base_id = str(row.get("base_sample_id") or "")
        operator = str(row.get("operator") or "")
        if operator in HALLU_ATOMIC_OPERATORS:
            by_base.setdefault(base_id, {})[operator] = row
    missing: list[str] = []
    for base_id, rows in sorted(by_base.items()):
        if any(operator not in rows for operator in HALLU_ATOMIC_OPERATORS):
            missing.append(base_id)
    if missing:
        raise ValueError(f"missing hallucination atomic operators for base_sample_id(s): {missing[:10]}")
    return by_base


def _reference_payload(atomic_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    detection = _json_loads(atomic_rows["hallucination_detection_mapper"]["reference_text"])
    spans = _json_loads(atomic_rows["hallucination_span_extractor"]["reference_text"])
    types = _json_loads(atomic_rows["hallucination_type_classifier"]["reference_text"])
    corrected_text = str(atomic_rows["hallucination_correction_mapper"]["reference_text"])
    return {
        "has_hallucination": bool(detection.get("has_hallucination")),
        "hallucination_count": int(detection.get("hallucination_count") or 0),
        "spans": spans.get("spans") or [],
        "types": types.get("types") or [],
        "corrected_text": corrected_text,
    }


def _prompt_variants() -> list[dict[str, str]]:
    return [
        {"style_id": style_id, "style_label": label, "user_requirement": requirement}
        for style_id, label, requirement in PROMPT_VARIANTS
    ]


def normalize_rows(
    compositional_rows: list[dict[str, Any]],
    atomic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_base = _atomic_by_base_sample(atomic_rows)
    output_rows: list[dict[str, Any]] = []
    for row in compositional_rows:
        base_id = str(row.get("base_sample_id") or "")
        if base_id not in by_base:
            raise ValueError(f"no atomic hallucination rows for base_sample_id={base_id}")
        payload = _reference_payload(by_base[base_id])
        next_row = dict(row)
        next_row["reference_json"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        next_row["reference_text"] = payload["corrected_text"]
        next_row["reference_text_full_run"] = payload["corrected_text"]
        next_row["output_format"] = "json"
        next_row["scoring_profile"] = "mixed_structured_text"
        next_row["reports_refinement_gain"] = True
        next_row["prompt_variants"] = _prompt_variants()
        next_row["prompt_variant_count"] = len(next_row["prompt_variants"])
        next_row["difficulty_label"] = "compositional_mixed_structured_text"
        output_rows.append(order_v3_row(next_row))
    return output_rows


def update_all_rows(all_rows: list[dict[str, Any]], normalized_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_instance = {str(row.get("instance_id")): row for row in normalized_rows}
    output_rows: list[dict[str, Any]] = []
    for row in all_rows:
        instance_id = str(row.get("instance_id") or "")
        output_rows.append(by_instance.get(instance_id, row))
    if sum(1 for row in output_rows if str(row.get("instance_id") or "") in by_instance) != len(by_instance):
        raise ValueError("not all normalized hallucination compositional rows were found in benchmark_v3_all")
    return output_rows


def update_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(manifest))
    counts = payload.setdefault("counts", {})
    scoring = counts.setdefault("by_scoring_profile", {})
    if scoring.get("text_refinement", 0) >= 300:
        scoring["text_refinement"] -= 300
    scoring["mixed_structured_text"] = scoring.get("mixed_structured_text", 0) + 300
    metric_policy = payload.setdefault("metric_policy", {})
    metric_policy["mixed_structured_text"] = (
        "Canonical JSON exact match is treated as RS; corrected_text is also used for text refinement diagnostics."
    )
    fields = payload.setdefault("public_schema_fields", [])
    if "reference_json" not in fields:
        insert_at = fields.index("reference_text") + 1 if "reference_text" in fields else len(fields)
        fields.insert(insert_at, "reference_json")
    tracks = payload.get("tracks") or {}
    hallu = tracks.get("semantic_hallu_compositional")
    if isinstance(hallu, dict):
        hallu["metric_policy"] = (
            "RS requires exact JSON over detection, span extraction, type classification, and corrected_text; "
            "corrected_text also reports refinement gain."
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize hallucination compositional rows to mixed structured-text scoring.")
    parser.add_argument("--benchmark-root", default="data/benchmark_v3")
    args = parser.parse_args()

    root = Path(args.benchmark_root)
    tracks = root / "tracks"
    atomic_path = tracks / "semantic_hallu_atomic.jsonl"
    compositional_path = tracks / "semantic_hallu_compositional.jsonl"
    all_path = root / "benchmark_v3_all.jsonl"
    manifest_path = root / "manifest.json"

    atomic_rows = read_jsonl(atomic_path)
    compositional_rows = read_jsonl(compositional_path)
    normalized_rows = normalize_rows(compositional_rows, atomic_rows)
    write_jsonl(compositional_path, normalized_rows)

    if all_path.exists():
        write_jsonl(all_path, update_all_rows(read_jsonl(all_path), normalized_rows))
    if manifest_path.exists():
        write_json(manifest_path, update_manifest(json.loads(manifest_path.read_text(encoding="utf-8"))))

    print(f"normalized {len(normalized_rows)} hallucination compositional rows -> {compositional_path}")


if __name__ == "__main__":
    main()
