from __future__ import annotations

import json
import re
from typing import Any

RG_EPSILON = 1e-6


def normalize_status(value: Any) -> str:
    return ("" if value is None else str(value)).strip().upper()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n(?:[ \t]*\n)+", "\n\n", text)
    return text.strip()


def normalize_text_for_match(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    try:
        import editdistance

        return int(editdistance.eval(left, right))
    except Exception:
        pass
    m, n = len(left), len(right)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def refinement_gain(
    input_text: Any,
    reference_text: Any,
    predicted_text: Any,
    reference_text_full_run: Any = None,
    reference_status: Any = None,
) -> dict[str, Any]:
    raw_input = "" if input_text is None else str(input_text)
    raw_ref = "" if reference_text is None else str(reference_text)
    raw_pred = "" if predicted_text is None else str(predicted_text)
    raw_ref_full = "" if reference_text_full_run is None else str(reference_text_full_run)
    d_input_ref = edit_distance(raw_input, raw_ref)
    d_input_pred = edit_distance(raw_input, raw_pred)
    d_pred_ref = edit_distance(raw_pred, raw_ref)
    d_pred_full = (
        edit_distance(raw_pred, raw_ref_full)
        if normalize_status(reference_status) == "DROP" and raw_ref_full
        else None
    )
    raw_rg = 1.0 if d_pred_ref == 0 else (float(d_input_ref) - float(d_pred_ref)) / max(float(d_input_ref), 1.0)
    rg = min(1.0, max(0.0, raw_rg))
    edit_calibration_denominator = max(float(d_input_pred), float(d_input_ref), 1.0)
    edit_calibration = min(1.0, max(0.0, 1.0 - (abs(float(d_input_pred) - float(d_input_ref)) / edit_calibration_denominator)))
    return {
        "edit_distance_input_to_reference": d_input_ref,
        "edit_distance_input_to_prediction": d_input_pred,
        "edit_distance_prediction_to_reference": d_pred_ref,
        "edit_distance_prediction_to_full_run_reference": d_pred_full,
        "reference_text_full_run": raw_ref_full,
        "raw_refinement_gain": raw_rg,
        "refinement_progress": rg,
        "edit_calibration": edit_calibration,
        "refinement_gain": rg,
    }


def parse_jsonish(value: Any) -> Any | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    for tag in ("clean_text", "output", "result", "answer"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (TypeError, ValueError):
                pass
    # Conservative extraction for common object/array responses. raw_decode
    # accepts trailing text without the quadratic retry loop used previously.
    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if start_candidates:
        start = min(start_candidates)
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
            return parsed
        except (TypeError, ValueError):
            pass
    return None


def canonical_json(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return sorted((canonical_json(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, dict):
        return {str(key): canonical_json(val) for key, val in sorted(value.items())}
    return value


def compare_json(reference_text: Any, predicted_text: Any) -> dict[str, Any]:
    ref = parse_jsonish(reference_text)
    pred = parse_jsonish(predicted_text)
    parse_success = pred is not None
    exact = ref is not None and pred is not None and canonical_json(ref) == canonical_json(pred)
    field_matches: dict[str, bool] = {}
    if isinstance(ref, dict) and isinstance(pred, dict):
        for key in sorted(set(ref) | set(pred)):
            field_matches[key] = canonical_json(ref.get(key)) == canonical_json(pred.get(key))
    return {
        "json_parse_success": parse_success,
        "json_exact_match": exact,
        "field_matches": field_matches,
        "field_accuracy": (sum(field_matches.values()) / len(field_matches) if field_matches else None),
    }


def compare_text(
    *,
    input_text: Any,
    reference_status: Any,
    reference_text: Any,
    predicted_status: Any,
    predicted_text: Any,
    reference_text_full_run: Any = None,
) -> dict[str, Any]:
    status_match = normalize_status(reference_status) == normalize_status(predicted_status)
    raw_ref = "" if reference_text is None else str(reference_text)
    raw_pred = "" if predicted_text is None else str(predicted_text)
    exact_ref = normalize_text(reference_text)
    exact_pred = normalize_text(predicted_text)
    norm_ref = normalize_text_for_match(reference_text)
    norm_pred = normalize_text_for_match(predicted_text)
    strict_match = raw_ref == raw_pred
    normalized_match = exact_ref == exact_pred
    norm_match = norm_ref == norm_pred
    metrics = refinement_gain(
        input_text,
        reference_text,
        predicted_text,
        reference_text_full_run=reference_text_full_run,
        reference_status=reference_status,
    )
    metrics.update(
        {
            "status_match": status_match,
            "text_exact_match": strict_match,
            "normalized_text_exact_match": normalized_match,
            "norm_text_exact_match": norm_match,
            "recipe_success_strict": status_match and strict_match,
            "normalized_recipe_success": status_match and normalized_match,
            "norm_recipe_success": status_match and norm_match,
            "recipe_success": status_match and norm_match,
        }
    )
    return metrics
