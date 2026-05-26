from __future__ import annotations

from typing import Any

from cdrbench.prepare_data.build_benchmark_release import RELEASE_FIELD_ORDER


PREDICTION_FIELD_ORDER = [
    'request_model',
    'request_base_url',
    'prompt_mode',
    'selected_prompt_variant_indices',
    'variant_predictions',
]


def ordered_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in RELEASE_FIELD_ORDER:
        if key in row:
            ordered[key] = row[key]
    for key in PREDICTION_FIELD_ORDER:
        if key in row and key not in ordered:
            ordered[key] = row[key]
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered
