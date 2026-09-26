from __future__ import annotations

from typing import Any


PUBLIC_V3_FIELD_ORDER = [
    "instance_id",
    "benchmark_track",
    "benchmark_split",
    "track_family",
    "source_track",
    "base_sample_id",
    "domain",
    "source_record_id",
    "source_benchmark",
    "input_text",
    "input_length_chars",
    "input_length_bucket",
    "operator",
    "operator_kind",
    "operator_sequence",
    "semantic_operator",
    "recipe_id",
    "recipe_prompt_key",
    "recipe_type",
    "recipe_length",
    "order_family_id",
    "order_slot",
    "order_group_instance_id",
    "group_success_rule",
    "reference_status",
    "reference_text",
    "reference_json",
    "reference_text_full_run",
    "output_format",
    "scoring_profile",
    "reports_refinement_gain",
    "prompt_variants",
    "prompt_variant_count",
    "selected_prompt_variant_indices",
    "prompt_candidate_pool_count",
    "prompt_sampling_policy",
    "prompt_sampling_seed",
    "difficulty_score",
    "difficulty_label",
    "pii_meta",
    "hallu_meta",
    "rubric_meta",
    "safety_meta",
]

FULL_V3_FIELD_ORDER = [
    "v3_schema_version",
    "benchmark_track_label",
    "track_role",
    "source_track_label",
    "evaluation_sample_policy",
    "reports_recipe_success",
    "structured_metric_profile",
    "required_output_components",
    "recipe_order_setting",
    "domain_label",
    "domain_abbr",
    "source_domain",
    "difficulty_grid_cell",
    "recipe_length_label",
    "filter_name",
    "filter_params_by_name",
    "recipe_variant_id",
    "reference_text_at_stop",
    "reference_trace",
    "recipe_prompt_key",
    "prompt_source",
    "prompt_candidate_pool_count",
    "prompt_sampling_policy",
    "prompt_sampling_seed",
    "accepted_candidate_count",
    "accepted_style_count",
    "threshold_meta",
    "agenthallu_meta",
    "v3_notes",
]

V3_FIELD_ORDER = PUBLIC_V3_FIELD_ORDER + FULL_V3_FIELD_ORDER

REQUIRED_V3_FIELDS = {
    "instance_id",
    "benchmark_track",
    "benchmark_split",
    "track_family",
    "source_track",
    "scoring_profile",
    "output_format",
    "reports_refinement_gain",
    "input_text",
    "reference_status",
    "reference_text",
}

CORE_RULE_TRACKS = {
    "atomic_m": ("Atomic-M", "atomic"),
    "atomic_f": ("Atomic-F", "atomic"),
    "agnostic_m": ("Agnostic-M", "compositional"),
    "order_m": ("Order-M", "order_sensitive"),
    "order_f": ("Order-F", "order_sensitive"),
}

SEMANTIC_SOURCE_TRACKS = {
    "semantic_pii_atomic": {
        "benchmark_track": "semantic_pii",
        "benchmark_track_label": "Semantic PII Redaction",
        "benchmark_split": "atomic",
        "structured_metric_profile": None,
    },
    "semantic_pii_compositional": {
        "benchmark_track": "semantic_pii",
        "benchmark_track_label": "Semantic PII Redaction",
        "benchmark_split": "compositional",
        "structured_metric_profile": None,
    },
    "semantic_hallu_atomic": {
        "benchmark_track": "semantic_hallucination",
        "benchmark_track_label": "Semantic Hallucination Processing",
        "benchmark_split": "atomic",
        "structured_metric_profile": "json_exact_fields",
    },
    "semantic_hallu_compositional": {
        "benchmark_track": "semantic_hallucination",
        "benchmark_track_label": "Semantic Hallucination Processing",
        "benchmark_split": "compositional",
        "structured_metric_profile": None,
    },
    "semantic_rubric_atomic": {
        "benchmark_track": "semantic_rubric",
        "benchmark_track_label": "Semantic Rubric Scoring",
        "benchmark_split": "atomic",
        "structured_metric_profile": "json_exact_fields",
    },
    "semantic_rubric_compositional": {
        "benchmark_track": "semantic_rubric",
        "benchmark_track_label": "Semantic Rubric Scoring",
        "benchmark_split": "compositional",
        "structured_metric_profile": "json_exact_fields",
    },
    "semantic_safety_atomic": {
        "benchmark_track": "semantic_safety",
        "benchmark_track_label": "Semantic Safety Tagging",
        "benchmark_split": "atomic",
        "structured_metric_profile": "json_exact_fields",
    },
    "semantic_safety_compositional": {
        "benchmark_track": "semantic_safety",
        "benchmark_track_label": "Semantic Safety Tagging",
        "benchmark_split": "compositional",
        "structured_metric_profile": "json_exact_fields",
    },
}


def is_json_reference(value: Any) -> bool:
    import json

    text = "" if value is None else str(value).strip()
    if not text or text[0] not in "{[":
        return False
    try:
        json.loads(text)
    except (TypeError, ValueError):
        return False
    return True


def order_v3_row(row: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in V3_FIELD_ORDER:
        if key in row:
            ordered[key] = row[key]
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def public_v3_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in PUBLIC_V3_FIELD_ORDER}


def validate_v3_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in sorted(REQUIRED_V3_FIELDS):
        if key not in row:
            errors.append(f"missing required field: {key}")
    if row.get("output_format") not in {"tagged_text", "json", "json_and_tagged_text"}:
        errors.append(f"invalid output_format: {row.get('output_format')!r}")
    if row.get("scoring_profile") not in {"text_refinement", "structured_json", "mixed_structured_text"}:
        errors.append(f"invalid scoring_profile: {row.get('scoring_profile')!r}")
    if row.get("track_family") not in {"core_rule", "semantic_extension"}:
        errors.append(f"invalid track_family: {row.get('track_family')!r}")
    if bool(row.get("reports_refinement_gain")) and row.get("scoring_profile") not in {"text_refinement", "mixed_structured_text"}:
        errors.append("reports_refinement_gain=true requires text_refinement scoring")
    return errors
