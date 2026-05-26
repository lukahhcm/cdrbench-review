#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from cdrbench.eval.metrics import compute_recipe_metrics, normalize_text_for_norm_match


ROOT = Path(__file__).resolve().parents[3]
DOMAIN_METADATA = {
    'web': {'label': 'Web Refinement', 'abbr': 'WR', 'order': 0},
    'arxiv': {'label': 'LaTeX Refinement', 'abbr': 'LR', 'order': 1},
    'knowledge_base': {'label': 'RAG Preparation', 'abbr': 'RP', 'order': 2},
    'pii': {'label': 'Privacy Redaction', 'abbr': 'PR', 'order': 3},
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    preview = line.strip()
                    if len(preview) > 200:
                        preview = preview[:200] + '...'
                    raise SystemExit(
                        f'invalid JSONL in {path} at line {line_no}: {exc}. '
                        f'Line preview: {preview}'
                    ) from exc
    return rows


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = '||'.join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    tmp_path.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    tmp_path.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
        if not text.endswith('\n'):
            handle.write('\n')
    tmp_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w',
        encoding='utf-8',
        newline='',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _safe_slug(value: Any) -> str:
    text = '' if value is None else str(value).strip().lower()
    if not text:
        return 'unknown'
    return ''.join(ch if ch.isalnum() else '_' for ch in text).strip('_') or 'unknown'


def _load_domain_metadata() -> dict[str, dict[str, Any]]:
    metadata = {key: dict(value) for key, value in DOMAIN_METADATA.items()}
    path = ROOT / 'configs' / 'domains.yaml'
    if not path.exists():
        return metadata
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    domains = payload.get('domains') if isinstance(payload, dict) else None
    if not isinstance(domains, dict):
        return metadata
    for key, value in domains.items():
        if not isinstance(key, str):
            continue
        description = value.get('description') if isinstance(value, dict) else None
        if key not in metadata:
            metadata[key] = {
                'label': description.strip() if isinstance(description, str) and description.strip() else key.replace('_', ' ').title(),
                'abbr': key.upper(),
                'order': 100 + len(metadata),
            }
    return metadata


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return _safe_mean([1.0 if bool(row.get(key)) else 0.0 for row in rows])


def _mean_optional(values: list[Any]) -> float:
    normalized: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            continue
    return _safe_mean(normalized)


def _rate_optional(rows: list[dict[str, Any]], key: str) -> float:
    return _mean_optional(
        [
            None if row.get(key) is None else (1.0 if bool(row.get(key)) else 0.0)
            for row in rows
        ]
    )


def _is_format_instability_error(error_text: Any) -> bool:
    if error_text is None:
        return False
    text = str(error_text)
    return (
        text == 'empty_response'
        or text.startswith('json_parse_error:')
        or text.startswith('tag_parse_error')
        or text.startswith('incomplete_tag_error:')
    )


def _is_request_error(error_text: Any) -> bool:
    if error_text is None:
        return False
    text = str(error_text)
    return text.startswith('request_error:') or text.startswith('non_scoring_refusal:')


def _is_incomplete_tag_error(error_text: Any) -> bool:
    if error_text is None:
        return False
    return str(error_text).startswith('incomplete_tag_error:')


def _normalize_variant_predictions(row: dict[str, Any]) -> list[dict[str, Any]]:
    variants = row.get('variant_predictions')
    if not isinstance(variants, list):
        return []
    normalized = [variant for variant in variants if isinstance(variant, dict)]
    return sorted(normalized, key=lambda item: int(item.get('prompt_variant_index', 0) or 0))


def _group_rows_by_instance_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        instance_id = str(row.get('instance_id') or '')
        if instance_id:
            grouped.setdefault(instance_id, []).append(row)
    return grouped


def _instance_row_matches_sampling_config(
    row: dict[str, Any],
    *,
    sample_size: int,
    sample_seed: int,
) -> bool:
    return (
        int(row.get('prompt_variant_sample_size', 0) or 0) == int(sample_size)
        and int(row.get('prompt_variant_sampling_seed', 0) or 0) == int(sample_seed)
    )


def _first_non_empty_str(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _canonical_order_slot(value: Any) -> str:
    slot = str(value or '').strip().lower()
    return {
        'front': 'pre',
        'pre': 'pre',
        'middle': 'mid',
        'mid': 'mid',
        'end': 'post',
        'post': 'post',
    }.get(slot, slot)


def _track_name_from_predictions_path(path: Path) -> str:
    known_tracks = {
        'atomic_ops',
        'main',
        'order_sensitivity',
        'atomic_m',
        'atomic_f',
        'agnostic_m',
        'order_m',
        'order_f',
        'atomic_mapper',
        'atomic_filter',
        'order_agnostic_m',
        'mapper_order',
        'mapper_order_sensitivity',
        'filter_order',
        'mapper_filter_order',
    }
    for part in reversed(path.parts):
        if part in known_tracks:
            return part
    if path.stem == 'predictions' and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem or 'unknown'


def _infer_labels(prediction_rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for prediction_row in prediction_rows:
        model = _first_non_empty_str(prediction_row.get('request_model'))
        base_url = _first_non_empty_str(prediction_row.get('request_base_url'))
        if model or base_url:
            return model, base_url
        for variant_prediction in _normalize_variant_predictions(prediction_row):
            model = _first_non_empty_str(variant_prediction.get('request_model'))
            base_url = _first_non_empty_str(variant_prediction.get('request_base_url'))
            if model or base_url:
                return model, base_url
    return None, None


def _base_identity(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        'instance_id',
        'benchmark_track',
        'domain',
        'source_domain',
        'order_family_id',
        'order_slot',
        'order_group_instance_id',
        'group_success_rule',
        'operator',
        'operator_kind',
        'source_record_id',
        'input_text',
        'input_length_chars',
        'input_length_bucket',
        'reference_status',
        'reference_text',
        'reference_text_full_run',
        'recipe_id',
        'recipe_variant_id',
        'recipe_type',
        'prompt_mode',
    ]
    return {key: row[key] for key in keys if key in row}


def _score_variant(prediction_row: dict[str, Any], variant_prediction: dict[str, Any]) -> dict[str, Any]:
    prediction_error = variant_prediction.get('prediction_error')
    valid_prediction = bool(
        variant_prediction.get('valid_prediction')
        if 'valid_prediction' in variant_prediction
        else variant_prediction.get('prediction_valid_json')
    )
    parsed_payload = (
        variant_prediction.get('parsed_response')
        if isinstance(variant_prediction.get('parsed_response'), dict)
        else {}
    )
    has_predicted_status = variant_prediction.get('predicted_status') is not None or parsed_payload.get('status') is not None
    has_predicted_clean_text = (
        variant_prediction.get('predicted_clean_text') is not None
        or parsed_payload.get('clean_text') is not None
    )
    predicted_status_value = (
        variant_prediction.get('predicted_status')
        if variant_prediction.get('predicted_status') is not None
        else parsed_payload.get('status')
    )
    predicted_clean_text_value = (
        variant_prediction.get('predicted_clean_text')
        if variant_prediction.get('predicted_clean_text') is not None
        else parsed_payload.get('clean_text')
    )
    predicted_status = '' if predicted_status_value is None else str(predicted_status_value)
    predicted_clean_text = '' if predicted_clean_text_value is None else str(predicted_clean_text_value)
    scorable_prediction = valid_prediction and has_predicted_status and has_predicted_clean_text

    scored = _base_identity(prediction_row)
    scored['predicted_status'] = predicted_status
    scored['predicted_clean_text'] = predicted_clean_text
    scored['prediction_has_status'] = has_predicted_status
    scored['prediction_has_clean_text'] = has_predicted_clean_text
    scored['scorable_prediction'] = scorable_prediction
    if scorable_prediction:
        scored.update(
            compute_recipe_metrics(
                input_text=prediction_row.get('input_text', ''),
                reference_status=prediction_row.get('reference_status', ''),
                reference_text=prediction_row.get('reference_text', ''),
                reference_text_full_run=prediction_row.get('reference_text_full_run', ''),
                predicted_status=predicted_status,
                predicted_clean_text=predicted_clean_text,
            )
        )
    else:
        scored.update(
            {
                'normalized_reference_status': '' if prediction_row.get('reference_status') is None else str(prediction_row.get('reference_status')).strip().upper(),
                'normalized_predicted_status': '' if predicted_status is None else str(predicted_status).strip().upper(),
                'normalized_reference_text': '',
                'normalized_predicted_text': '',
                'norm_reference_text': '',
                'norm_predicted_text': '',
                'status_match': False,
                'text_exact_match': False,
                'normalized_text_exact_match': False,
                'norm_text_exact_match': False,
                'recipe_success': False,
                'normalized_recipe_success': False,
                'norm_recipe_success': False,
                'text_match': False,
                'edit_distance_input_to_reference': None,
                'edit_distance_input_to_prediction': None,
                'edit_distance_prediction_to_reference': None,
                'edit_distance_prediction_to_full_run_reference': None,
                'reference_text_full_run': '' if prediction_row.get('reference_text_full_run') is None else str(prediction_row.get('reference_text_full_run')),
                'rg_epsilon': None,
                'raw_refinement_gain': None,
                'refinement_progress': None,
                'edit_calibration': None,
                'refinement_gain': None,
            }
        )
    dist_stop = scored.get('edit_distance_prediction_to_reference')
    dist_full = scored.get('edit_distance_prediction_to_full_run_reference')
    drop_reference_affinity = None
    if dist_stop is not None and dist_full is not None:
        try:
            dist_stop_f = float(dist_stop)
            dist_full_f = float(dist_full)
            drop_reference_affinity = (
                'stop'
                if dist_stop_f < dist_full_f
                else ('full' if dist_full_f < dist_stop_f else 'tie')
            )
        except (TypeError, ValueError):
            drop_reference_affinity = None
    scored.update(
        {
            'prompt_variant_index': int(variant_prediction.get('prompt_variant_index', 0) or 0),
            'prompt_style_id': variant_prediction.get('prompt_style_id'),
            'prompt_style_label': variant_prediction.get('prompt_style_label'),
            'user_requirement': variant_prediction.get('user_requirement'),
            'request_model': variant_prediction.get('request_model'),
            'request_base_url': variant_prediction.get('request_base_url'),
            'raw_response': variant_prediction.get('raw_response'),
            'parsed_response': variant_prediction.get('parsed_response'),
            'prediction_error': prediction_error,
            'valid_prediction': valid_prediction,
            'scorable_prediction': scorable_prediction,
            'retry_attempted': bool(variant_prediction.get('retry_attempted')),
            'format_instability_error': _is_format_instability_error(prediction_error),
            'incomplete_tag_error': _is_incomplete_tag_error(prediction_error),
            'request_error': _is_request_error(prediction_error),
            'response_usage': variant_prediction.get('response_usage'),
            'drop_reference_affinity': drop_reference_affinity,
        }
    )
    return scored


def _score_prediction_row(prediction_row: dict[str, Any]) -> list[dict[str, Any]]:
    variants = _normalize_variant_predictions(prediction_row)
    if not variants:
        variants = [prediction_row]
    return [_score_variant(prediction_row, variant) for variant in variants]


def _sample_variant_rows(
    prediction_row: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    *,
    sample_size: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(variant_rows):
        return list(variant_rows)
    recipe_prompt_key = str(
        prediction_row.get('recipe_prompt_key')
        or prediction_row.get('workflow_prompt_key')
        or ''
    )
    instance_id = str(prediction_row.get('instance_id') or '')
    ranked = sorted(
        variant_rows,
        key=lambda row: _stable_id(
            'prompt-variant-sample',
            sample_seed,
            recipe_prompt_key,
            instance_id,
            int(row.get('prompt_variant_index', 0) or 0),
        ),
    )
    return sorted(ranked[:sample_size], key=lambda row: int(row.get('prompt_variant_index', 0) or 0))


def _aggregate_instance_metrics(
    prediction_row: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    *,
    sample_size: int,
    sample_seed: int,
) -> dict[str, Any]:
    request_ok_rows = [row for row in variant_rows if not bool(row.get('request_error'))]
    scorable_rows = [row for row in variant_rows if bool(row.get('scorable_prediction'))]
    sampled_variant_rows = _sample_variant_rows(
        prediction_row,
        variant_rows,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    sampled_request_ok_rows = [row for row in sampled_variant_rows if not bool(row.get('request_error'))]
    sampled_scorable_rows = [row for row in sampled_variant_rows if bool(row.get('scorable_prediction'))]
    required_at_k_count = len(variant_rows) if sample_size <= 0 else sample_size
    at_k_complete = (
        len(sampled_variant_rows) >= required_at_k_count
        and len(sampled_scorable_rows) >= required_at_k_count
    )
    at_k_incomplete_reason = None
    if not at_k_complete:
        if len(sampled_variant_rows) < required_at_k_count:
            at_k_incomplete_reason = 'missing_prompt_variants'
        elif len(sampled_request_ok_rows) < required_at_k_count:
            at_k_incomplete_reason = 'request_error_prompt_variants'
        else:
            at_k_incomplete_reason = 'unscorable_prompt_variants'
    strict_recipe_success_values = [1.0 if bool(row.get('recipe_success')) else 0.0 for row in scorable_rows]
    normalized_recipe_success_values = [1.0 if bool(row.get('norm_recipe_success')) else 0.0 for row in scorable_rows]
    strict_recipe_success_sampled_values = [1.0 if bool(row.get('recipe_success')) else 0.0 for row in sampled_scorable_rows]
    normalized_recipe_success_sampled_values = [1.0 if bool(row.get('norm_recipe_success')) else 0.0 for row in sampled_scorable_rows]
    refinement_gain_values = [row.get('refinement_gain') for row in scorable_rows]
    refinement_progress_values = [row.get('refinement_progress') for row in scorable_rows]
    edit_calibration_values = [row.get('edit_calibration') for row in scorable_rows]
    prompt_variant_metrics = []
    strict_recipe_success_by_index: dict[int, bool | None] = {}
    normalized_recipe_success_by_index: dict[int, bool | None] = {}
    for row in variant_rows:
        index = int(row.get('prompt_variant_index', 0) or 0)
        strict_recipe_success_by_index[index] = None if not bool(row.get('scorable_prediction')) else bool(row.get('recipe_success'))
        normalized_recipe_success_by_index[index] = None if not bool(row.get('scorable_prediction')) else bool(row.get('norm_recipe_success'))
        prompt_variant_metrics.append(
            {
                'prompt_variant_index': index,
                'prompt_style_id': row.get('prompt_style_id'),
                'prompt_style_label': row.get('prompt_style_label'),
                'recipe_success_strict': bool(row.get('recipe_success')),
                'recipe_success_norm': bool(row.get('norm_recipe_success')),
                'status_match': bool(row.get('status_match')),
                'text_exact_match_strict': bool(row.get('text_exact_match')),
                'text_exact_match_norm': bool(row.get('norm_text_exact_match')),
                'refinement_gain': None if row.get('refinement_gain') is None else float(row.get('refinement_gain', 0.0)),
                'refinement_progress': None if row.get('refinement_progress') is None else float(row.get('refinement_progress', 0.0)),
                'edit_calibration': None if row.get('edit_calibration') is None else float(row.get('edit_calibration', 0.0)),
                'prediction_error': row.get('prediction_error'),
                'valid_prediction': bool(row.get('valid_prediction')),
                'scorable_prediction': bool(row.get('scorable_prediction')),
                'prediction_has_status': bool(row.get('prediction_has_status')),
                'prediction_has_clean_text': bool(row.get('prediction_has_clean_text')),
                'format_instability_error': bool(row.get('format_instability_error')),
                'incomplete_tag_error': bool(row.get('incomplete_tag_error')),
                'request_error': bool(row.get('request_error')),
            }
        )

    instance_row = _base_identity(prediction_row)
    instance_row.update(
        {
            'num_prompt_variants': len(variant_rows),
            'num_request_ok_variants': len(request_ok_rows),
            'num_scorable_variants': len(scorable_rows),
            'num_sampled_prompt_variants': len(sampled_variant_rows),
            'num_sampled_request_ok_variants': len(sampled_request_ok_rows),
            'num_sampled_scorable_variants': len(sampled_scorable_rows),
            'required_at_k_prompt_variants': required_at_k_count,
            'at_k_complete': at_k_complete,
            'at_k_incomplete_reason': at_k_incomplete_reason,
            'prompt_variant_sample_size': sample_size if sample_size > 0 else len(variant_rows),
            'prompt_variant_sampling_seed': sample_seed,
            'mean_rs': _mean_optional(normalized_recipe_success_values),
            'mean_rs@k': (any(normalized_recipe_success_sampled_values) if at_k_complete else None),
            'mean_rs_strict': _mean_optional(strict_recipe_success_values),
            'mean_rs_strict@k': (any(strict_recipe_success_sampled_values) if at_k_complete else None),
            'mean_rg': _mean_optional(refinement_gain_values),
            'mean_refinement_progress': _mean_optional(refinement_progress_values),
            'mean_edit_calibration': _mean_optional(edit_calibration_values),
            'num_valid_rg_variants': sum(1 for value in refinement_gain_values if value is not None),
            'num_invalid_variants': sum(1 for row in variant_rows if not bool(row.get('valid_prediction'))),
            'num_format_error_variants': sum(1 for row in variant_rows if bool(row.get('format_instability_error'))),
            'num_incomplete_tag_error_variants': sum(1 for row in variant_rows if bool(row.get('incomplete_tag_error'))),
            'num_request_error_variants': sum(1 for row in variant_rows if bool(row.get('request_error'))),
            'prompt_variant_metrics': prompt_variant_metrics,
            'recipe_success_prompt0': normalized_recipe_success_by_index.get(0),
            'recipe_success_prompt0_strict': strict_recipe_success_by_index.get(0),
        }
    )
    return instance_row


def _build_order_group_rows(instance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in instance_rows:
        group_id = row.get('order_group_instance_id')
        if group_id:
            grouped.setdefault(str(group_id), []).append(row)
    output = []
    for group_id, bucket in sorted(grouped.items()):
        required_slots: list[str] = []
        for row in bucket:
            raw_required = row.get('required_slots_for_group_success')
            if isinstance(raw_required, list):
                required_slots.extend(str(slot) for slot in raw_required if str(slot or '').strip())
            elif isinstance(raw_required, str):
                required_slots.extend(slot.strip() for slot in raw_required.split('|') if slot.strip())
        required_slots = sorted(set(required_slots))
        slot_bucket = bucket
        missing_required_slots: list[str] = []
        if required_slots:
            rows_by_slot = {_canonical_order_slot(row.get('order_slot')): row for row in bucket}
            missing_required_slots = [slot for slot in required_slots if _canonical_order_slot(slot) not in rows_by_slot]
            slot_bucket = [rows_by_slot[_canonical_order_slot(slot)] for slot in required_slots if _canonical_order_slot(slot) in rows_by_slot]
        complete_group = not missing_required_slots and bool(slot_bucket)
        output.append(
            {
                'order_group_instance_id': group_id,
                'slot_count': len(bucket),
                'required_slots_for_group_success': required_slots,
                'missing_required_slots': missing_required_slots,
                'ocs': (
                    all(bool(row.get('recipe_success_prompt0')) for row in slot_bucket)
                    if complete_group and all(row.get('recipe_success_prompt0') is not None for row in slot_bucket)
                    else None
                ),
                'ocs_at_k': (
                    all(bool(row.get('mean_rs@k')) for row in slot_bucket)
                    if complete_group and all(row.get('mean_rs@k') is not None for row in slot_bucket)
                    else None
                ),
                'ocs_strict': (
                    all(bool(row.get('recipe_success_prompt0_strict')) for row in slot_bucket)
                    if complete_group and all(row.get('recipe_success_prompt0_strict') is not None for row in slot_bucket)
                    else None
                ),
                'ocs_at_k_strict': (
                    all(bool(row.get('mean_rs_strict@k')) for row in slot_bucket)
                    if complete_group and all(row.get('mean_rs_strict@k') is not None for row in slot_bucket)
                    else None
                ),
            }
        )
    return output


def _instance_slice_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key) or 'UNKNOWN')
        grouped.setdefault(value, []).append(row)
    return [
        {
            key: value,
            'count': len(bucket),
            'mean_rs': _mean_optional([item.get('mean_rs') if int(item.get('num_scorable_variants', 0) or 0) > 0 else None for item in bucket]),
            'mean_rs_strict': _mean_optional([item.get('mean_rs_strict') if int(item.get('num_scorable_variants', 0) or 0) > 0 else None for item in bucket]),
            'mean_rs@k': _rate_optional(bucket, 'mean_rs@k'),
            'mean_rs_strict@k': _rate_optional(bucket, 'mean_rs_strict@k'),
            'mean_rg': _mean_optional([item.get('mean_rg') for item in bucket]),
            'mean_refinement_progress': _mean_optional([item.get('mean_refinement_progress') for item in bucket]),
            'mean_edit_calibration': _mean_optional([item.get('mean_edit_calibration') for item in bucket]),
        }
        for value, bucket in sorted(grouped.items())
    ]


def _summary_by_value(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key) or ''): row for row in _instance_slice_summary(rows, key)}


def _attach_domain_metadata(rows: list[dict[str, Any]], *, key: str, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        copied = dict(row)
        raw_value = str(row.get(key) or 'UNKNOWN')
        meta = metadata.get(
            raw_value,
            {
                'label': raw_value.replace('_', ' ').title(),
                'abbr': raw_value.upper(),
                'order': 999,
            },
        )
        copied['domain_label'] = str(meta.get('label') or raw_value)
        copied['domain_abbr'] = str(meta.get('abbr') or raw_value.upper())
        copied['domain_order'] = int(meta.get('order', 999) or 999)
        enriched.append(copied)
    return sorted(enriched, key=lambda row: (int(row.get('domain_order', 999) or 999), str(row.get(key) or '')))


def _variant_slice_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key) or 'UNKNOWN')
        grouped.setdefault(value, []).append(row)
    return [
        {
            key: value,
            'count': len(bucket),
            'recipe_success_rate': _mean_optional([None if not bool(row.get('scorable_prediction')) else (1.0 if bool(row.get('norm_recipe_success')) else 0.0) for row in bucket]),
            'recipe_success_rate_strict': _mean_optional([None if not bool(row.get('scorable_prediction')) else (1.0 if bool(row.get('recipe_success')) else 0.0) for row in bucket]),
            'status_accuracy': _mean_optional([None if not bool(row.get('scorable_prediction')) else (1.0 if bool(row.get('status_match')) else 0.0) for row in bucket]),
            'exact_text_match_rate': _mean_optional([None if not bool(row.get('scorable_prediction')) else (1.0 if bool(row.get('norm_text_exact_match')) else 0.0) for row in bucket]),
            'exact_text_match_rate_strict': _mean_optional([None if not bool(row.get('scorable_prediction')) else (1.0 if bool(row.get('text_exact_match')) else 0.0) for row in bucket]),
            'avg_refinement_gain': _mean_optional([row.get('refinement_gain') for row in bucket]),
            'avg_refinement_progress': _mean_optional([row.get('refinement_progress') for row in bucket]),
            'avg_edit_calibration': _mean_optional([row.get('edit_calibration') for row in bucket]),
            'valid_prediction_rate': _rate(bucket, 'valid_prediction'),
            'scorable_prediction_rate': _rate(bucket, 'scorable_prediction'),
            'format_error_rate': _rate(bucket, 'format_instability_error'),
            'incomplete_tag_error_rate': _rate(bucket, 'incomplete_tag_error'),
            'request_error_rate': _rate(bucket, 'request_error'),
        }
        for value, bucket in sorted(grouped.items())
    ]


def _summary_report_text(summary: dict[str, Any]) -> str:
    def fmt_float(value: Any) -> str:
        if value is None:
            return 'NA'
        try:
            return f'{float(value):.4f}'
        except (TypeError, ValueError):
            return 'NA'

    parts = [f"[summary] track={summary.get('track') or 'unknown'}"]
    if summary.get('model'):
        parts.append(f"model={summary['model']}")
    parts.append(f"num_instances={summary.get('num_instances', 0)}")
    parts.append(f"prompt_variant_sample_size={summary.get('prompt_variant_sample_size', 0)}")
    parts.append(f"prompt_variant_sampling_seed={summary.get('prompt_variant_sampling_seed', 0)}")
    parts.append(f"mean_rs={float(summary.get('mean_rs', 0.0)):.4f}")
    parts.append(f"mean_rs_strict={float(summary.get('mean_rs_strict', 0.0)):.4f}")
    parts.append(f"mean_rs@k={float(summary.get('mean_rs@k', 0.0)):.4f}")
    parts.append(f"mean_rs_strict@k={float(summary.get('mean_rs_strict@k', 0.0)):.4f}")
    parts.append(f"mean_rg={float(summary.get('mean_rg', 0.0)):.4f}")
    parts.append(f"at_k_complete_instances={int(summary.get('num_at_k_complete_instances', 0) or 0)}")
    parts.append(f"at_k_incomplete_instances={int(summary.get('num_at_k_incomplete_instances', 0) or 0)}")
    parts.append(f"at_k_complete_rate={float(summary.get('at_k_complete_rate', 0.0)):.4f}")
    parts.append(f"valid_prediction_rate={float(summary.get('valid_prediction_rate', 0.0)):.4f}")
    parts.append(f"scorable_prediction_rate={float(summary.get('scorable_prediction_rate', 0.0)):.4f}")
    parts.append(f"empty_response_rate={float(summary.get('empty_response_rate', 0.0)):.4f}")
    parts.append(f"format_error_rate={float(summary.get('format_error_rate', 0.0)):.4f}")
    parts.append(f"incomplete_tag_error_rate={float(summary.get('incomplete_tag_error_rate', 0.0)):.4f}")
    parts.append(f"request_error_rate={float(summary.get('request_error_rate', 0.0)):.4f}")
    if 'ocs' in summary:
        parts.append(f"ocs={float(summary.get('ocs', 0.0)):.4f}")
        parts.append(f"ocs_at_k={float(summary.get('ocs_at_k', 0.0)):.4f}")
        parts.append(f"ocs_strict={float(summary.get('ocs_strict', 0.0)):.4f}")
        parts.append(f"ocs_at_k_strict={float(summary.get('ocs_at_k_strict', 0.0)):.4f}")
    for key in (
        'rs_pre',
        'rs_pre@k',
        'rg_pre',
        'rs_mid',
        'rs_mid@k',
        'rg_mid',
        'rs_post',
        'rs_post@k',
        'rg_post',
    ):
        if key in summary:
            parts.append(f"{key}={float(summary.get(key, 0.0)):.4f}")
    for key in (
        'drop_affinity_count',
        'drop_affinity_count_pre',
        'drop_affinity_count_mid',
        'drop_affinity_count_post',
        'drop_full_run_closer_rate',
        'drop_stop_closer_rate',
        'drop_tie_rate',
        'drop_full_run_closer_pre',
        'drop_stop_closer_pre',
        'drop_tie_pre',
        'drop_full_run_closer_mid',
        'drop_stop_closer_mid',
        'drop_tie_mid',
        'drop_full_run_closer_post',
        'drop_stop_closer_post',
        'drop_tie_post',
    ):
        if key in summary:
            value = summary.get(key, 0.0)
            if key.startswith('drop_affinity_count'):
                parts.append(f"{key}={int(value or 0)}")
            else:
                parts.append(f"{key}={fmt_float(value)}")
    return ' '.join(parts)


def _paper_metrics_payload(summary: dict[str, Any]) -> dict[str, Any]:
    track = str(summary.get('track') or '')
    payload = {
        'track': summary.get('track'),
        'model': summary.get('model'),
        'num_instances': summary.get('num_instances'),
        'prompt_variant_sample_size': summary.get('prompt_variant_sample_size'),
        'prompt_variant_sampling_seed': summary.get('prompt_variant_sampling_seed'),
        'mean_rs': summary.get('mean_rs'),
        'mean_rs_strict': summary.get('mean_rs_strict'),
        'mean_rs@k': summary.get('mean_rs@k'),
        'mean_rs_strict@k': summary.get('mean_rs_strict@k'),
        'mean_rg': summary.get('mean_rg'),
        'num_at_k_complete_instances': summary.get('num_at_k_complete_instances'),
        'num_at_k_incomplete_instances': summary.get('num_at_k_incomplete_instances'),
        'at_k_complete_rate': summary.get('at_k_complete_rate'),
        'valid_prediction_rate': summary.get('valid_prediction_rate'),
        'scorable_prediction_rate': summary.get('scorable_prediction_rate'),
        'empty_response_rate': summary.get('empty_response_rate'),
        'format_error_rate': summary.get('format_error_rate'),
        'incomplete_tag_error_rate': summary.get('incomplete_tag_error_rate'),
        'request_error_rate': summary.get('request_error_rate'),
    }
    common_optional_keys = (
        'ocs',
        'ocs_at_k',
        'ocs_strict',
        'ocs_at_k_strict',
        'atomic_mapper_rs',
        'atomic_mapper_rs@k',
        'atomic_filter_rs',
        'atomic_filter_rs@k',
    )
    order_f_optional_keys = (
        'rs_pre',
        'rs_pre@k',
        'rg_pre',
        'rs_mid',
        'rs_mid@k',
        'rg_mid',
        'rs_post',
        'rs_post@k',
        'rg_post',
        'drop_affinity_count',
        'drop_affinity_count_pre',
        'drop_affinity_count_mid',
        'drop_affinity_count_post',
        'drop_full_run_closer_rate',
        'drop_stop_closer_rate',
        'drop_tie_rate',
        'drop_full_run_closer_pre',
        'drop_stop_closer_pre',
        'drop_tie_pre',
        'drop_full_run_closer_mid',
        'drop_stop_closer_mid',
        'drop_tie_mid',
        'drop_full_run_closer_post',
        'drop_stop_closer_post',
        'drop_tie_post',
    )
    optional_keys = common_optional_keys
    if track in {'order_sensitivity', 'order_f'}:
        optional_keys = (*optional_keys, *order_f_optional_keys)
    for key in optional_keys:
        if key in summary:
            payload[key] = summary.get(key)
    return payload


def _order_drop_distance_rows(variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in variant_rows:
        if bool(row.get('request_error')) or not bool(row.get('scorable_prediction')):
            continue
        if str(row.get('benchmark_track') or '') not in {'order_sensitivity', 'order_f'}:
            continue
        if str(row.get('reference_status') or '').strip().upper() != 'DROP':
            continue
        reference_text = '' if row.get('reference_text') is None else str(row.get('reference_text'))
        full_run_text = '' if row.get('reference_text_full_run') is None else str(row.get('reference_text_full_run'))
        if (
            not full_run_text
            or normalize_text_for_norm_match(reference_text) == normalize_text_for_norm_match(full_run_text)
        ):
            continue
        dist_stop = row.get('edit_distance_prediction_to_reference')
        dist_full = row.get('edit_distance_prediction_to_full_run_reference')
        if dist_stop is None or dist_full is None:
            continue
        try:
            dist_stop_f = float(dist_stop)
            dist_full_f = float(dist_full)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                'instance_id': row.get('instance_id'),
                'order_group_instance_id': row.get('order_group_instance_id'),
                'order_slot': _canonical_order_slot(row.get('order_slot')),
                'prompt_variant_index': row.get('prompt_variant_index'),
                'prompt_style_id': row.get('prompt_style_id'),
                'domain': row.get('domain'),
                'source_domain': row.get('source_domain'),
                'dist_to_stop': dist_stop_f,
                'dist_to_full': dist_full_f,
                'dist_gap_full_minus_stop': dist_full_f - dist_stop_f,
                'closer_to': (
                    'stop'
                    if dist_stop_f < dist_full_f
                    else ('full' if dist_full_f < dist_stop_f else 'tie')
                ),
            }
        )
    return rows


def _drop_affinity_rates(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    total = len(rows)
    if total == 0:
        return {
            'count': 0,
            'full_run_closer_rate': None,
            'stop_closer_rate': None,
            'tie_rate': None,
        }
    full_run = sum(1 for row in rows if row.get('closer_to') == 'full')
    stop = sum(1 for row in rows if row.get('closer_to') == 'stop')
    ties = sum(1 for row in rows if row.get('closer_to') == 'tie')
    return {
        'count': total,
        'full_run_closer_rate': full_run / total,
        'stop_closer_rate': stop / total,
        'tie_rate': ties / total,
    }


def _drop_affinity_summary(variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    drop_rows = _order_drop_distance_rows(variant_rows)
    overall = _drop_affinity_rates(drop_rows)
    summary: dict[str, Any] = {
        'drop_affinity_count': overall['count'],
        'drop_full_run_closer_rate': overall['full_run_closer_rate'],
        'drop_stop_closer_rate': overall['stop_closer_rate'],
        'drop_tie_rate': overall['tie_rate'],
    }
    for slot in ('pre', 'mid', 'post'):
        slot_rates = _drop_affinity_rates(
            [row for row in drop_rows if _canonical_order_slot(row.get('order_slot')) == slot]
        )
        summary[f'drop_affinity_count_{slot}'] = slot_rates['count']
        summary[f'drop_full_run_closer_{slot}'] = slot_rates['full_run_closer_rate']
        summary[f'drop_stop_closer_{slot}'] = slot_rates['stop_closer_rate']
        summary[f'drop_tie_{slot}'] = slot_rates['tie_rate']
    return summary


def _order_slot_metric_summary(rows: list[dict[str, Any]], slot: str) -> dict[str, Any]:
    slot_rows = [row for row in rows if _canonical_order_slot(row.get('order_slot')) == slot]
    return {
        f'rs_{slot}': _safe_mean([float(row.get('mean_rs', 0.0)) for row in slot_rows]),
        f'rs_{slot}@k': _rate_optional(slot_rows, 'mean_rs@k'),
        f'rg_{slot}': _mean_optional([row.get('mean_rg') for row in slot_rows]),
        f'rp_{slot}': _mean_optional([row.get('mean_refinement_progress') for row in slot_rows]),
        f'rc_{slot}': _mean_optional([row.get('mean_edit_calibration') for row in slot_rows]),
    }


def _plot_order_drop_stop_vs_full(
    *,
    output_dir: Path,
    variant_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    drop_rows = _order_drop_distance_rows(variant_rows)
    _write_csv(output_dir / 'order_drop_distance_scatter.csv', drop_rows)
    if not drop_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; skipping order-drop scatter plot.', flush=True)
        return

    slots = ['pre', 'mid', 'post']
    labels = {'pre': 'Pre', 'mid': 'Mid', 'post': 'Post'}
    colors = {'pre': '#d1495b', 'mid': '#edae49', 'post': '#00798c'}
    max_dist = max(max(float(row['dist_to_stop']), float(row['dist_to_full'])) for row in drop_rows)
    axis_max = max(1.0, max_dist * 1.05)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    any_points = False
    for ax, slot in zip(axes, slots):
        slot_rows = [row for row in drop_rows if str(row.get('order_slot') or '') == slot]
        if slot_rows:
            any_points = True
            ax.scatter(
                [float(row['dist_to_stop']) for row in slot_rows],
                [float(row['dist_to_full']) for row in slot_rows],
                s=18,
                alpha=0.65,
                color=colors[slot],
                edgecolors='none',
            )
        ax.plot([0, axis_max], [0, axis_max], linestyle='--', linewidth=1.0, color='#666666')
        ax.set_title(f"{labels[slot]} (n={len(slot_rows)})", fontsize=11)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel(r'$d(\hat{t}, t_{\mathrm{stop}})$')
    axes[0].set_ylabel(r'$d(\hat{t}, t_{\mathrm{full}})$')

    model = summary.get('model') or 'unknown-model'
    fig.suptitle(
        f'Order DROP Analysis: Stop vs Full-Run Affinity\nModel={model}',
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        'Points below the dashed diagonal are closer to the full-run reference; '
        'points above it are closer to the correct stopping reference.',
        ha='center',
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    fig.savefig(output_dir / 'order_drop_stop_vs_full_scatter.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    summary_rows = []
    for slot in slots:
        slot_rows = [row for row in drop_rows if str(row.get('order_slot') or '') == slot]
        if not slot_rows:
            continue
        total = len(slot_rows)
        closer_full = sum(1 for row in slot_rows if row['closer_to'] == 'full')
        closer_stop = sum(1 for row in slot_rows if row['closer_to'] == 'stop')
        ties = sum(1 for row in slot_rows if row['closer_to'] == 'tie')
        summary_rows.append(
            {
                'order_slot': slot,
                'count': total,
                'closer_to_full_rate': closer_full / total,
                'closer_to_stop_rate': closer_stop / total,
                'tie_rate': ties / total,
                'mean_dist_gap_full_minus_stop': _mean_optional([row['dist_gap_full_minus_stop'] for row in slot_rows]),
                'median_dist_gap_full_minus_stop': _safe_median([float(row['dist_gap_full_minus_stop']) for row in slot_rows]),
            }
        )
    _write_csv(output_dir / 'order_drop_distance_summary.csv', summary_rows)


def _build_summary(
    predictions_path: Path,
    instance_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    order_group_rows: list[dict[str, Any]],
    model_name: str | None,
    base_url: str | None,
    sample_size: int,
    sample_seed: int,
) -> dict[str, Any]:
    domain_metadata = _load_domain_metadata()
    mean_rg_values = [row.get('mean_rg') for row in instance_rows if int(row.get('num_valid_rg_variants', 0) or 0) > 0]
    mean_refinement_progress_values = [
        row.get('mean_refinement_progress')
        for row in instance_rows
        if int(row.get('num_valid_rg_variants', 0) or 0) > 0
    ]
    mean_edit_calibration_values = [
        row.get('mean_edit_calibration')
        for row in instance_rows
        if int(row.get('num_valid_rg_variants', 0) or 0) > 0
    ]
    by_domain = _attach_domain_metadata(_instance_slice_summary(instance_rows, 'domain'), key='domain', metadata=domain_metadata)
    by_source_domain = _instance_slice_summary(instance_rows, 'source_domain')
    at_k_complete_rows = [row for row in instance_rows if bool(row.get('at_k_complete'))]
    at_k_incomplete_rows = [row for row in instance_rows if not bool(row.get('at_k_complete'))]
    summary = {
        'track': _track_name_from_predictions_path(predictions_path),
        'model': model_name,
        'base_url': base_url,
        'num_instances': len(instance_rows),
        'num_variant_predictions': len(variant_rows),
        'prompt_variant_sample_size': sample_size,
        'prompt_variant_sampling_seed': sample_seed,
        'mean_rs': _mean_optional([row.get('mean_rs') if int(row.get('num_scorable_variants', 0) or 0) > 0 else None for row in instance_rows]),
        'mean_rs_strict': _mean_optional([row.get('mean_rs_strict') if int(row.get('num_scorable_variants', 0) or 0) > 0 else None for row in instance_rows]),
        'mean_rs@k': _rate_optional(instance_rows, 'mean_rs@k'),
        'mean_rs_strict@k': _rate_optional(instance_rows, 'mean_rs_strict@k'),
        'mean_rg': _mean_optional(mean_rg_values),
        'mean_refinement_progress': _mean_optional(mean_refinement_progress_values),
        'mean_edit_calibration': _mean_optional(mean_edit_calibration_values),
        'median_rg': _safe_median([float(value) for value in mean_rg_values if value is not None]),
        'num_at_k_complete_instances': len(at_k_complete_rows),
        'num_at_k_incomplete_instances': len(at_k_incomplete_rows),
        'at_k_complete_rate': len(at_k_complete_rows) / len(instance_rows) if instance_rows else 0.0,
        'by_at_k_incomplete_reason': _instance_slice_summary(at_k_incomplete_rows, 'at_k_incomplete_reason'),
        'valid_prediction_rate': _rate(variant_rows, 'valid_prediction'),
        'scorable_prediction_rate': _rate(variant_rows, 'scorable_prediction'),
        'empty_response_rate': _safe_mean([1.0 if str(row.get('prediction_error') or '') == 'empty_response' else 0.0 for row in variant_rows]),
        'format_error_rate': _rate(variant_rows, 'format_instability_error'),
        'incomplete_tag_error_rate': _rate(variant_rows, 'incomplete_tag_error'),
        'request_error_rate': _rate(variant_rows, 'request_error'),
        'by_operator': _instance_slice_summary(instance_rows, 'operator'),
        'by_operator_kind': _instance_slice_summary(instance_rows, 'operator_kind'),
        'by_domain': by_domain,
        'by_source_domain': by_source_domain,
        'by_reference_status': _instance_slice_summary(instance_rows, 'reference_status'),
        'per_prompt_variant': _variant_slice_summary(variant_rows, 'prompt_variant_index'),
    }
    by_operator_kind = _summary_by_value(instance_rows, 'operator_kind')
    mapper_kind = by_operator_kind.get('mapper')
    filter_kind = by_operator_kind.get('filter')
    if mapper_kind is not None:
        summary['atomic_mapper_rs'] = mapper_kind.get('mean_rs')
        summary['atomic_mapper_rs@k'] = mapper_kind.get('mean_rs@k')
    if filter_kind is not None:
        summary['atomic_filter_rs'] = filter_kind.get('mean_rs')
        summary['atomic_filter_rs@k'] = filter_kind.get('mean_rs@k')
    if order_group_rows:
        summary['ocs'] = _rate_optional(order_group_rows, 'ocs')
        summary['ocs_at_k'] = _rate_optional(order_group_rows, 'ocs_at_k')
        summary['ocs_strict'] = _rate_optional(order_group_rows, 'ocs_strict')
        summary['ocs_at_k_strict'] = _rate_optional(order_group_rows, 'ocs_at_k_strict')
        if str(summary.get('track') or '') in {'order_sensitivity', 'order_f'}:
            for slot in ('pre', 'mid', 'post'):
                summary.update(_order_slot_metric_summary(instance_rows, slot))
            summary.update(_drop_affinity_summary(variant_rows))
        summary['num_order_groups'] = len(order_group_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Score CDR-Bench predictions and write reports.')
    parser.add_argument('--predictions-path', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--progress-every', type=int, default=20)
    parser.add_argument('--prompt-variant-sample-size', type=int, default=3)
    parser.add_argument('--prompt-variant-sampling-seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--write-csv-slices', action='store_true')
    args = parser.parse_args()

    predictions_path = Path(args.predictions_path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (predictions_path.parent / 'score').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = _read_jsonl(predictions_path)
    if not prediction_rows:
        raise SystemExit(
            f'No prediction rows to score: predictions_path={predictions_path}. '
            'Check that inference produced non-empty predictions.'
        )
    inferred_model, inferred_base_url = _infer_labels(prediction_rows)

    existing_instance_rows_by_id: dict[str, dict[str, Any]] = {}
    existing_variant_rows_by_id: dict[str, list[dict[str, Any]]] = {}
    instance_metrics_path = output_dir / 'instance_metrics.jsonl'
    scored_variant_predictions_path = output_dir / 'scored_variant_predictions.jsonl'
    if args.resume:
        if instance_metrics_path.exists():
            for row in _read_jsonl(instance_metrics_path):
                instance_id = str(row.get('instance_id') or '')
                if instance_id:
                    existing_instance_rows_by_id[instance_id] = row
        if scored_variant_predictions_path.exists():
            existing_variant_rows_by_id = _group_rows_by_instance_id(_read_jsonl(scored_variant_predictions_path))

    scored_variant_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    reused_instance_count = 0
    new_instance_count = 0
    total_rows = len(prediction_rows)
    started = time.time()
    print(
        f'start score track={_track_name_from_predictions_path(predictions_path)} '
        f'num_instances={total_rows} progress_every={args.progress_every} resume={bool(args.resume)}',
        flush=True,
    )

    for index, prediction_row in enumerate(prediction_rows, start=1):
        instance_id = str(prediction_row.get('instance_id') or '')
        if (
            args.resume
            and instance_id in existing_instance_rows_by_id
            and _instance_row_matches_sampling_config(
                existing_instance_rows_by_id[instance_id],
                sample_size=args.prompt_variant_sample_size,
                sample_seed=args.prompt_variant_sampling_seed,
            )
        ):
            instance_rows.append(existing_instance_rows_by_id[instance_id])
            scored_variant_rows.extend(existing_variant_rows_by_id.get(instance_id, []))
            reused_instance_count += 1
        else:
            instance_variant_rows = _score_prediction_row(prediction_row)
            scored_variant_rows.extend(instance_variant_rows)
            instance_rows.append(
                _aggregate_instance_metrics(
                    prediction_row,
                    instance_variant_rows,
                    sample_size=args.prompt_variant_sample_size,
                    sample_seed=args.prompt_variant_sampling_seed,
                )
            )
            new_instance_count += 1

        if index % args.progress_every == 0 or index == total_rows:
            elapsed = time.time() - started
            print(
                f'progress score instance={index}/{total_rows} '
                f'reused_instances={reused_instance_count} new_instances={new_instance_count} '
                f'elapsed_sec={elapsed:.1f}',
                flush=True,
            )

    order_group_rows = _build_order_group_rows(instance_rows)
    summary = _build_summary(
        predictions_path=predictions_path,
        instance_rows=instance_rows,
        variant_rows=scored_variant_rows,
        order_group_rows=order_group_rows,
        model_name=inferred_model,
        base_url=inferred_base_url,
        sample_size=args.prompt_variant_sample_size,
        sample_seed=args.prompt_variant_sampling_seed,
    )

    _write_jsonl(scored_variant_predictions_path, scored_variant_rows)
    _write_jsonl(instance_metrics_path, instance_rows)
    incomplete_at_k_rows = [
        {
            'instance_id': row.get('instance_id'),
            'benchmark_track': row.get('benchmark_track'),
            'domain': row.get('domain'),
            'source_domain': row.get('source_domain'),
            'order_group_instance_id': row.get('order_group_instance_id'),
            'order_slot': row.get('order_slot'),
            'num_prompt_variants': row.get('num_prompt_variants'),
            'num_request_ok_variants': row.get('num_request_ok_variants'),
            'num_scorable_variants': row.get('num_scorable_variants'),
            'num_sampled_prompt_variants': row.get('num_sampled_prompt_variants'),
            'num_sampled_request_ok_variants': row.get('num_sampled_request_ok_variants'),
            'num_sampled_scorable_variants': row.get('num_sampled_scorable_variants'),
            'required_at_k_prompt_variants': row.get('required_at_k_prompt_variants'),
            'at_k_incomplete_reason': row.get('at_k_incomplete_reason'),
            'num_invalid_variants': row.get('num_invalid_variants'),
            'num_format_error_variants': row.get('num_format_error_variants'),
            'num_incomplete_tag_error_variants': row.get('num_incomplete_tag_error_variants'),
            'num_request_error_variants': row.get('num_request_error_variants'),
        }
        for row in instance_rows
        if not bool(row.get('at_k_complete'))
    ]
    _write_jsonl(output_dir / 'incomplete_at_k_instances.jsonl', incomplete_at_k_rows)
    unscorable_variant_rows = [
        {
            'instance_id': row.get('instance_id'),
            'benchmark_track': row.get('benchmark_track'),
            'domain': row.get('domain'),
            'source_domain': row.get('source_domain'),
            'order_group_instance_id': row.get('order_group_instance_id'),
            'order_slot': row.get('order_slot'),
            'prompt_variant_index': row.get('prompt_variant_index'),
            'prompt_style_id': row.get('prompt_style_id'),
            'valid_prediction': row.get('valid_prediction'),
            'prediction_has_status': row.get('prediction_has_status'),
            'prediction_has_clean_text': row.get('prediction_has_clean_text'),
            'prediction_error': row.get('prediction_error'),
            'raw_response': row.get('raw_response'),
            'parsed_response': row.get('parsed_response'),
        }
        for row in scored_variant_rows
        if not bool(row.get('scorable_prediction'))
    ]
    _write_jsonl(output_dir / 'unscorable_variant_predictions.jsonl', unscorable_variant_rows)
    _write_json(output_dir / 'summary.json', summary)
    _write_json(output_dir / 'paper_metrics.json', _paper_metrics_payload(summary))
    _write_text(output_dir / 'report.txt', _summary_report_text(summary))
    if summary.get('track') in {'order_sensitivity', 'order_f'}:
        _plot_order_drop_stop_vs_full(
            output_dir=output_dir,
            variant_rows=scored_variant_rows,
            summary=summary,
        )
    if args.write_csv_slices:
        _write_csv(output_dir / 'by_operator.csv', summary['by_operator'])
        _write_csv(output_dir / 'by_operator_kind.csv', summary['by_operator_kind'])
        _write_csv(output_dir / 'by_domain.csv', summary['by_domain'])
        _write_csv(output_dir / 'by_source_domain.csv', summary['by_source_domain'])
        _write_csv(output_dir / 'by_reference_status.csv', summary['by_reference_status'])
        _write_csv(output_dir / 'by_at_k_incomplete_reason.csv', summary.get('by_at_k_incomplete_reason', []))
        _write_csv(output_dir / 'incomplete_at_k_instances.csv', incomplete_at_k_rows)
        _write_csv(output_dir / 'unscorable_variant_predictions.csv', unscorable_variant_rows)
        if order_group_rows:
            _write_csv(output_dir / 'order_groups.csv', order_group_rows)
        _write_csv(output_dir / 'per_prompt_variant.csv', summary.get('per_prompt_variant', []))

    print(f'wrote scored variant predictions -> {scored_variant_predictions_path}', flush=True)
    print(f'wrote instance metrics -> {instance_metrics_path}', flush=True)
    print(f'wrote incomplete @K report -> {output_dir / "incomplete_at_k_instances.jsonl"}', flush=True)
    print(f'wrote unscorable prediction report -> {output_dir / "unscorable_variant_predictions.jsonl"}', flush=True)
    print(f'wrote summary -> {output_dir / "summary.json"}', flush=True)
    print(f'wrote paper metrics -> {output_dir / "paper_metrics.json"}', flush=True)
    print(f'wrote report -> {output_dir / "report.txt"}', flush=True)
    print(_summary_report_text(summary), flush=True)


if __name__ == '__main__':
    main()
