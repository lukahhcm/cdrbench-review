#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

from cdrbench.config import load_domains_config
from cdrbench.domain_assignment import build_domain_execution_plan
from cdrbench.prepare_data.materialize_domain_recipes import (
    FILTER_CALIBRATION_RULES,
    FILTER_STATUS_RULES,
    RATIO_THRESHOLD_KEYS,
    _format_threshold_value,
    _apply_mapper_text,
    _evaluate_filter,
    _infer_suffix,
    _labeling_meta,
    _percentile,
    iter_jsonl,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = '||'.join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


def _recipe_prompt_key(row: dict[str, Any]) -> str:
    operator_sequence = list(row.get('operator_sequence') or ([row['operator']] if row.get('operator') else []))
    return _stable_id(
        row.get('benchmark_track'),
        row.get('domain'),
        _first_present(row, 'recipe_type', 'workflow_type'),
        row.get('order_slot'),
        operator_sequence,
        row.get('filter_params_by_name') or {},
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    count = 0
    with tmp_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
    tmp_path.replace(path)
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    tmp_path.replace(path)


def _safe_cache_key(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in value)


def _cache_paths(cache_dir: Path, track: str, key: str) -> tuple[Path, Path]:
    safe_key = _safe_cache_key(key)
    track_dir = cache_dir / track
    return track_dir / f'{safe_key}.jsonl', track_dir / f'{safe_key}.summary.json'


def _load_cache(cache_dir: Path, track: str, key: str) -> tuple[list[dict[str, Any]], Any] | None:
    rows_path, summary_path = _cache_paths(cache_dir, track, key)
    if not rows_path.exists() or not summary_path.exists():
        return None
    return _read_jsonl(rows_path), json.loads(summary_path.read_text(encoding='utf-8'))


def _write_cache(cache_dir: Path, track: str, key: str, rows: list[dict[str, Any]], summary: Any) -> None:
    rows_path, summary_path = _cache_paths(cache_dir, track, key)
    _write_jsonl(rows_path, rows)
    _write_json(summary_path, summary)


def _operator_lookup(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for variant in plan['execution_variants']:
        by_name.setdefault(variant['name'], variant)
    return by_name


def _op_kind(op_name: str, operators_by_name: dict[str, dict[str, Any]]) -> str:
    variant = operators_by_name.get(op_name)
    if variant is not None:
        return str(variant['kind'])
    if op_name.endswith('_filter'):
        return 'filter'
    return 'mapper'


def _base_params(op_name: str, operators_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    variant = operators_by_name.get(op_name)
    return dict(variant.get('params', {})) if variant is not None else {}


def _mapper_names(sequence: list[str], operators_by_name: dict[str, dict[str, Any]]) -> list[str]:
    return [op_name for op_name in sequence if _op_kind(op_name, operators_by_name) == 'mapper']


def _filter_names(sequence: list[str], operators_by_name: dict[str, dict[str, Any]]) -> list[str]:
    return [op_name for op_name in sequence if _op_kind(op_name, operators_by_name) == 'filter']


def _supporting_records(
    records: list[dict[str, Any]],
    mapper_names: list[str],
    max_records: int,
    salt: str,
    max_input_chars: int,
    source_usage_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    wanted = set(mapper_names)
    supported = []
    for record in records:
        if not _within_input_char_limit(record, max_input_chars):
            continue
        active = set(_labeling_meta(record).get('active_mapper_names', []))
        if wanted.issubset(active):
            supported.append(record)
    supported.sort(key=lambda row: _record_sort_key(row, salt, source_usage_counts))
    limit = _candidate_limit(max_records)
    return supported[:limit] if limit is not None else supported


def _record_id(record: dict[str, Any]) -> str:
    for key in ('id', 'source_name', 'url'):
        value = record.get(key)
        if value:
            return str(value)
    return _stable_id(record.get('text', ''), length=20)


def _within_input_char_limit(record: dict[str, Any], max_input_chars: int) -> bool:
    return max_input_chars <= 0 or len(str(record.get('text', ''))) <= max_input_chars


def _input_length_bucket(input_length_chars: int) -> str:
    if input_length_chars <= 4_000:
        return 'short'
    if input_length_chars <= 8_000:
        return 'medium'
    return 'long'


def _candidate_limit(max_records: int) -> int | None:
    return max_records if max_records > 0 else None


def _record_sort_key(record: dict[str, Any], salt: str, source_usage_counts: dict[str, int] | None = None) -> tuple[int, str]:
    record_id = _record_id(record)
    usage = source_usage_counts.get(record_id, 0) if source_usage_counts is not None else 0
    return usage, _stable_id(salt, record_id, record.get('source_name'), record.get('url'))


def _first_filter_index(sequence: list[str], operators_by_name: dict[str, dict[str, Any]]) -> int | None:
    for idx, op_name in enumerate(sequence):
        if _op_kind(op_name, operators_by_name) == 'filter':
            return idx
    return None


def _text_before_step(
    record: dict[str, Any],
    sequence: list[str],
    step_index: int,
    operators_by_name: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    suffix = _infer_suffix(record)
    text = str(record.get('text', ''))
    trace = []
    for idx, op_name in enumerate(sequence[:step_index]):
        kind = _op_kind(op_name, operators_by_name)
        if kind == 'filter':
            continue
        before_len = len(text)
        text, result = _apply_mapper_text(op_name, text, _base_params(op_name, operators_by_name), suffix)
        trace.append(
            {
                'step_index': idx,
                'operator': op_name,
                'kind': kind,
                'input_length': before_len,
                **result,
            }
        )
    return text, trace


def _filter_value(
    record: dict[str, Any],
    sequence: list[str],
    filter_index: int,
    filter_params: dict[str, Any],
    operators_by_name: dict[str, dict[str, Any]],
) -> tuple[float | None, str | None]:
    filter_name = sequence[filter_index]
    value_key = _resolve_filter_value_key(filter_name, filter_params)
    if value_key is None:
        return None, None
    text, _trace = _text_before_step(record, sequence, filter_index, operators_by_name)
    evaluation = _evaluate_filter(filter_name, text, filter_params, _infer_suffix(record))
    value = evaluation.get('stats', {}).get(value_key)
    if isinstance(value, (int, float)):
        return float(value), value_key
    return None, value_key


def _resolve_filter_value_key(op_name: str, params: dict[str, Any]) -> str | None:
    rule = FILTER_STATUS_RULES.get(op_name)
    if rule is None:
        return None
    value_key = rule['value_key']
    return value_key(params) if callable(value_key) else value_key


def _round_float(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _calibrate_filter_params_for_target(
    op_name: str,
    base_params: dict[str, Any],
    values: list[float],
    target_drop_rate: float,
    min_positive_ratio_threshold: float,
    zero_ratio_threshold_policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rule = FILTER_STATUS_RULES.get(op_name)
    calibration = FILTER_CALIBRATION_RULES.get(op_name)
    params = dict(base_params)
    if not rule or not calibration or not values:
        return params, {
            'threshold_source': 'not_calibrated',
            'target_drop_rate': target_drop_rate,
            'threshold_value': None,
        }

    threshold_key = rule['min_key'] if calibration['direction'] == 'min' else rule['max_key']

    if calibration['direction'] == 'min':
        q = target_drop_rate
        threshold = _percentile(values, q)
        if threshold is not None:
            params[rule['min_key']] = _format_threshold_value(threshold, rule['min_key'])
            params.pop(rule['max_key'], None)
    else:
        q = 1.0 - target_drop_rate
        threshold = _percentile(values, q)
        if threshold is not None:
            params[rule['max_key']] = _format_threshold_value(threshold, rule['max_key'])
            params.pop(rule['min_key'], None)

    threshold_value = _format_threshold_value(threshold, threshold_key)
    zero_ratio_threshold = (
        threshold_key in RATIO_THRESHOLD_KEYS
        and isinstance(threshold_value, (int, float))
        and float(threshold_value) == 0.0
    )
    zero_ratio_action = None
    if zero_ratio_threshold:
        zero_ratio_action = zero_ratio_threshold_policy
        if zero_ratio_threshold_policy == 'min-positive':
            threshold_value = round(max(float(min_positive_ratio_threshold), 0.0), 6)
            params[threshold_key] = threshold_value

    return params, {
        'threshold_source': 'instance_balanced_quantile',
        'target_drop_rate': target_drop_rate,
        'threshold_quantile': q,
        'threshold_direction': calibration['direction'],
        'threshold_raw_value': _round_float(threshold),
        'threshold_value': threshold_value,
        'threshold_param_key': threshold_key,
        'zero_ratio_threshold': zero_ratio_threshold,
        'zero_ratio_action': zero_ratio_action,
    }


def _execute_recipe(
    record: dict[str, Any],
    sequence: list[str],
    operators_by_name: dict[str, dict[str, Any]],
    filter_params_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    filter_params_by_name = filter_params_by_name or {}
    suffix = _infer_suffix(record)
    text = str(record.get('text', ''))
    trace = []
    for step_index, op_name in enumerate(sequence):
        kind = _op_kind(op_name, operators_by_name)
        before_len = len(text)
        if kind == 'mapper':
            text, result = _apply_mapper_text(op_name, text, _base_params(op_name, operators_by_name), suffix)
            trace.append(
                {
                    'step_index': step_index,
                    'operator': op_name,
                    'kind': kind,
                    'input_length': before_len,
                    **result,
                }
            )
            continue

        params = dict(filter_params_by_name.get(op_name, _base_params(op_name, operators_by_name)))
        evaluation = _evaluate_filter(op_name, text, params, suffix)
        trace.append(
            {
                'step_index': step_index,
                'operator': op_name,
                'kind': kind,
                'input_length': before_len,
                'keep': evaluation['keep'],
                'status': evaluation['status'],
                'stats': evaluation.get('stats', {}),
            }
        )
        if not evaluation['keep']:
            return {
                'reference_status': 'DROP',
                'reference_text': text,
                'trace': trace,
            }

    return {
        'reference_status': 'KEEP',
        'reference_text': text,
        'trace': trace,
    }


def _execute_recipe_without_early_stop(
    record: dict[str, Any],
    sequence: list[str],
    operators_by_name: dict[str, dict[str, Any]],
    filter_params_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    filter_params_by_name = filter_params_by_name or {}
    suffix = _infer_suffix(record)
    text = str(record.get('text', ''))
    trace = []
    for step_index, op_name in enumerate(sequence):
        kind = _op_kind(op_name, operators_by_name)
        before_len = len(text)
        if kind == 'mapper':
            text, result = _apply_mapper_text(op_name, text, _base_params(op_name, operators_by_name), suffix)
            trace.append(
                {
                    'step_index': step_index,
                    'operator': op_name,
                    'kind': kind,
                    'input_length': before_len,
                    **result,
                }
            )
            continue

        params = dict(filter_params_by_name.get(op_name, _base_params(op_name, operators_by_name)))
        evaluation = _evaluate_filter(op_name, text, params, suffix)
        trace.append(
            {
                'step_index': step_index,
                'operator': op_name,
                'kind': kind,
                'input_length': before_len,
                'keep': evaluation['keep'],
                'status': evaluation['status'],
                'stats': evaluation.get('stats', {}),
            }
        )

    return {
        'reference_text': text,
        'trace': trace,
    }


def _variant_record(
    base: dict[str, Any],
    record: dict[str, Any],
    sequence: list[str],
    filter_params_by_name: dict[str, dict[str, Any]],
    execution: dict[str, Any],
    full_run_execution: dict[str, Any] | None,
    threshold_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    instance_id = _stable_id(_first_present(base, 'recipe_variant_id', 'workflow_variant_id'), _record_id(record))
    input_text = str(record.get('text', ''))
    row = {
        'instance_id': instance_id,
        'source_record_id': _record_id(record),
        'input_text': input_text,
        'input_length_chars': len(input_text),
        'input_length_bucket': _input_length_bucket(len(input_text)),
        'operator_sequence': sequence,
        'filter_params_by_name': filter_params_by_name,
        'threshold_meta': threshold_meta or {},
        'reference_status': execution['reference_status'],
        'reference_text': execution['reference_text'],
        'reference_trace': execution['trace'],
        'reference_text_full_run': None if full_run_execution is None else full_run_execution.get('reference_text'),
        'full_run_reference_trace': None if full_run_execution is None else full_run_execution.get('trace'),
        **base,
    }
    row['recipe_prompt_key'] = _recipe_prompt_key(row)
    return row


def _select_balanced(
    keep_rows: list[dict[str, Any]],
    drop_rows: list[dict[str, Any]],
    max_instances: int,
    target_drop_rate: float,
    salt: str,
    source_usage_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    target_drop = max(1, round(max_instances * target_drop_rate))
    target_keep = max_instances - target_drop
    keep_rows = _prioritize_rows(keep_rows, salt + ':keep', source_usage_counts)
    drop_rows = _prioritize_rows(drop_rows, salt + ':drop', source_usage_counts)
    take_drop = min(target_drop, len(drop_rows))
    take_keep = min(target_keep, len(keep_rows))
    return [*keep_rows[:take_keep], *drop_rows[:take_drop]]


def _prioritize_rows(
    rows: list[dict[str, Any]],
    salt: str,
    source_usage_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, str]:
        source_id = str(row.get('source_record_id') or row.get('instance_id') or '')
        usage = source_usage_counts.get(source_id, 0) if source_usage_counts is not None else 0
        return usage, _stable_id(salt, source_id, row.get('instance_id'))

    return sorted(rows, key=key)


def _mark_source_usage(rows: Iterable[dict[str, Any]], source_usage_counts: dict[str, int]) -> None:
    for row in rows:
        source_id = row.get('source_record_id')
        if source_id:
            source_usage_counts[str(source_id)] += 1


def _load_domain_recipes(recipe_library_dir: Path) -> dict[str, dict[str, Any]]:
    recipes_by_domain = {}
    paths = sorted(recipe_library_dir.glob('*/recipe_library.yaml'))
    if not paths:
        paths = sorted(recipe_library_dir.glob('*/workflow_library.yaml'))
    for path in paths:
        with path.open('r', encoding='utf-8') as f:
            payload = yaml.safe_load(f)
        if isinstance(payload, dict) and payload.get('domain'):
            recipes_by_domain[str(payload['domain'])] = payload
    return recipes_by_domain


def _active_records_for_mapper(
    records: list[dict[str, Any]],
    mapper_name: str,
    max_records: int,
    salt: str,
    max_input_chars: int,
    source_usage_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    active_records = []
    for record in records:
        if not _within_input_char_limit(record, max_input_chars):
            continue
        active = set(_labeling_meta(record).get('active_mapper_names', []))
        if mapper_name in active:
            active_records.append(record)
    active_records.sort(key=lambda row: _record_sort_key(row, salt, source_usage_counts))
    limit = _candidate_limit(max_records)
    return active_records[:limit] if limit is not None else active_records


def _atomic_record(
    *,
    op_name: str,
    op_kind: str,
    record: dict[str, Any],
    params_by_name: dict[str, dict[str, Any]],
    execution: dict[str, Any],
    threshold_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_domain = str(record.get('domain') or 'unknown')
    input_text = str(record.get('text', ''))
    row = {
        'instance_id': _stable_id('atomic', op_name, _record_id(record)),
        'benchmark_track': 'atomic',
        'domain': source_domain,
        'source_domain': source_domain,
        'operator': op_name,
        'operator_kind': op_kind,
        'source_record_id': _record_id(record),
        'input_text': input_text,
        'input_length_chars': len(input_text),
        'input_length_bucket': _input_length_bucket(len(input_text)),
        'operator_sequence': [op_name],
        'filter_params_by_name': params_by_name,
        'threshold_meta': threshold_meta or {},
        'reference_status': execution['reference_status'],
        'reference_text': execution['reference_text'],
        'reference_trace': execution['trace'],
    }
    row['recipe_prompt_key'] = _recipe_prompt_key(row)
    return row


def _materialize_atomic_mapper(
    op_name: str,
    records: list[dict[str, Any]],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    source_usage_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = _active_records_for_mapper(
        records,
        op_name,
        args.max_atomic_candidate_records,
        f'atomic:{op_name}',
        args.max_input_chars,
        source_usage_counts,
    )
    rows = []
    for record in candidates:
        try:
            execution = _execute_recipe(record, [op_name], operators_by_name)
        except Exception:
            continue
        if execution['reference_status'] == 'KEEP' and execution['reference_text'] != str(record.get('text', '')):
            rows.append(
                _atomic_record(
                    op_name=op_name,
                    op_kind='mapper',
                    record=record,
                    params_by_name={},
                    execution=execution,
                )
            )
        if len(rows) >= args.max_atomic_instances_per_op:
            break
    source_domain_counts = dict(sorted(pd.Series([row['source_domain'] for row in rows]).value_counts().items())) if rows else {}
    return rows, {
        'operator': op_name,
        'operator_kind': 'mapper',
        'status': 'kept' if rows else 'skipped_no_active_outputs',
        'candidate_count': len(candidates),
        'selected_count': len(rows),
        'selected_keep_count': len(rows),
        'selected_drop_count': 0,
        'source_domain_counts': source_domain_counts,
        'threshold_meta': {},
        'filter_params': {},
    }


def _materialize_atomic_filter(
    op_name: str,
    records: list[dict[str, Any]],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    source_usage_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = sorted(
        [record for record in records if _within_input_char_limit(record, args.max_input_chars)],
        key=lambda row: _record_sort_key(row, f'atomic:{op_name}', source_usage_counts),
    )
    limit = _candidate_limit(args.max_atomic_candidate_records)
    if limit is not None:
        candidates = candidates[:limit]
    base_params = _base_params(op_name, operators_by_name)
    values = []
    value_key = None
    value_records = []
    for record in candidates:
        try:
            value, current_value_key = _filter_value(record, [op_name], 0, base_params, operators_by_name)
        except Exception:
            continue
        value_key = value_key or current_value_key
        if value is not None:
            values.append(value)
            value_records.append(record)

    calibrated_params, threshold_meta = _calibrate_filter_params_for_target(
        op_name,
        base_params,
        values,
        args.target_drop_rate,
        args.min_positive_ratio_threshold,
        args.zero_ratio_threshold_policy,
    )
    threshold_meta = {
        **threshold_meta,
        'filter_name': op_name,
        'status_value_key': value_key,
        'value_count': len(values),
        'atomic_filter': True,
    }
    if threshold_meta.get('zero_ratio_threshold') and args.zero_ratio_threshold_policy == 'skip':
        return [], {
            'operator': op_name,
            'operator_kind': 'filter',
            'status': 'skipped_zero_ratio_threshold',
            'candidate_count': len(candidates),
            'value_count': len(values),
            'keep_count': 0,
            'drop_count': 0,
            'selected_count': 0,
            'selected_keep_count': 0,
            'selected_drop_count': 0,
            'source_domain_counts': {},
            'threshold_meta': threshold_meta,
            'filter_params': calibrated_params,
        }
    params_by_name = {op_name: calibrated_params}
    keep_rows = []
    drop_rows = []
    for record in value_records:
        try:
            execution = _execute_recipe(record, [op_name], operators_by_name, params_by_name)
        except Exception:
            continue
        row = _atomic_record(
            op_name=op_name,
            op_kind='filter',
            record=record,
            params_by_name=params_by_name,
            execution=execution,
            threshold_meta=threshold_meta,
        )
        if execution['reference_status'] == 'KEEP':
            keep_rows.append(row)
        else:
            drop_rows.append(row)

    selected = _select_balanced(
        keep_rows,
        drop_rows,
        args.max_atomic_instances_per_op,
        args.target_drop_rate,
        f'atomic:{op_name}:select',
        source_usage_counts,
    )
    has_min_balance = len(keep_rows) >= args.min_atomic_keep and len(drop_rows) >= args.min_atomic_drop
    selected_rows = selected if has_min_balance else []
    source_domain_counts = (
        dict(sorted(pd.Series([row['source_domain'] for row in selected_rows]).value_counts().items()))
        if selected_rows
        else {}
    )
    return selected_rows, {
        'operator': op_name,
        'operator_kind': 'filter',
        'status': 'kept' if selected_rows else 'skipped_unbalanced',
        'candidate_count': len(candidates),
        'value_count': len(values),
        'keep_count': len(keep_rows),
        'drop_count': len(drop_rows),
        'selected_count': len(selected_rows),
        'selected_keep_count': sum(1 for row in selected_rows if row['reference_status'] == 'KEEP'),
        'selected_drop_count': sum(1 for row in selected_rows if row['reference_status'] == 'DROP'),
        'source_domain_counts': source_domain_counts,
        'threshold_meta': threshold_meta,
        'filter_params': calibrated_params,
    }


def _materialize_atomic_ops(
    records_by_domain: dict[str, list[dict[str, Any]]],
    plan: dict[str, Any],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    source_usage_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    all_records = [record for domain in sorted(records_by_domain) for record in records_by_domain[domain]]
    mapper_names = sorted(
        {variant['name'] for variant in plan['execution_variants'] if variant['kind'] == 'mapper'}
    )
    filter_names = sorted(
        {variant['name'] for variant in plan['execution_variants'] if variant['kind'] == 'filter'}
    )
    _log(f'atomic global: {len(mapper_names)} unique mappers, {len(filter_names)} unique filters, {len(all_records)} records')

    for op_name in mapper_names:
        op_rows, summary = _materialize_atomic_mapper(op_name, all_records, operators_by_name, args, source_usage_counts)
        rows.extend(op_rows)
        _mark_source_usage(op_rows, source_usage_counts)
        summary_rows.append(summary)
        _log(f"  atomic mapper {op_name}: {summary['status']} selected={summary['selected_count']}")

    for op_name in filter_names:
        op_rows, summary = _materialize_atomic_filter(op_name, all_records, operators_by_name, args, source_usage_counts)
        rows.extend(op_rows)
        _mark_source_usage(op_rows, source_usage_counts)
        summary_rows.append(summary)
        _log(
            f"  atomic filter {op_name}: {summary['status']} "
            f"selected={summary['selected_count']} keep={summary.get('selected_keep_count', 0)} "
            f"drop={summary.get('selected_drop_count', 0)}"
        )

    return rows, summary_rows


def _materialize_main_variant(
    domain: str,
    recipe: dict[str, Any],
    variant: dict[str, Any],
    records: list[dict[str, Any]],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    source_usage_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence = list(variant['operator_sequence'])
    mapper_names = _mapper_names(sequence, operators_by_name)
    candidates = _supporting_records(
        records,
        mapper_names,
        args.max_candidate_records,
        _first_present(variant, 'recipe_variant_id', 'workflow_variant_id'),
        args.max_input_chars,
        source_usage_counts,
    )
    base = {
        'benchmark_track': 'main',
        'domain': domain,
        'recipe_id': _first_present(recipe, 'recipe_id', 'workflow_id'),
        'recipe_variant_id': _first_present(variant, 'recipe_variant_id', 'workflow_variant_id'),
        'recipe_type': _first_present(variant, 'recipe_type', 'workflow_type'),
        'order_family_id': None,
        'order_slot': None,
    }

    filter_index = _first_filter_index(sequence, operators_by_name)
    if filter_index is None:
        rows = []
        for record in candidates:
            execution = _execute_recipe(record, sequence, operators_by_name)
            if execution['reference_status'] == 'KEEP' and execution['reference_text'] != str(record.get('text', '')):
                rows.append(_variant_record(base, record, sequence, {}, execution, None, None))
            if len(rows) >= args.max_instances_per_variant:
                break
        return rows, {
            **base,
            'status': 'kept' if rows else 'skipped_no_active_outputs',
            'candidate_count': len(candidates),
            'keep_count': len(rows),
            'drop_count': 0,
            'selected_keep_count': len(rows),
            'selected_drop_count': 0,
            'selected_count': len(rows),
            'filter_name': None,
            'threshold_meta': {},
        }

    filter_name = sequence[filter_index]
    base_filter_params = dict(variant.get('filter_params') or _base_params(filter_name, operators_by_name))
    values = []
    value_key = None
    value_records = []
    for record in candidates:
        try:
            value, current_value_key = _filter_value(record, sequence, filter_index, base_filter_params, operators_by_name)
        except Exception:
            continue
        value_key = value_key or current_value_key
        if value is not None:
            values.append(value)
            value_records.append(record)

    calibrated_params, threshold_meta = _calibrate_filter_params_for_target(
        filter_name,
        base_filter_params,
        values,
        args.target_drop_rate,
        args.min_positive_ratio_threshold,
        args.zero_ratio_threshold_policy,
    )
    threshold_meta = {
        **threshold_meta,
        'filter_name': filter_name,
        'status_value_key': value_key,
        'value_count': len(values),
    }
    if threshold_meta.get('zero_ratio_threshold') and args.zero_ratio_threshold_policy == 'skip':
        return [], {
            **base,
            'status': 'skipped_zero_ratio_threshold',
            'candidate_count': len(candidates),
            'value_count': len(values),
            'keep_count': 0,
            'drop_count': 0,
            'selected_keep_count': 0,
            'selected_drop_count': 0,
            'selected_count': 0,
            'filter_name': filter_name,
            'threshold_meta': threshold_meta,
            'filter_params': calibrated_params,
        }
    filter_params_by_name = {filter_name: calibrated_params}
    keep_rows = []
    drop_rows = []
    for record in value_records:
        try:
            execution = _execute_recipe(record, sequence, operators_by_name, filter_params_by_name)
        except Exception:
            continue
        row = _variant_record(base, record, sequence, filter_params_by_name, execution, None, threshold_meta)
        if execution['reference_status'] == 'KEEP':
            keep_rows.append(row)
        else:
            drop_rows.append(row)

    selected = _select_balanced(
        keep_rows,
        drop_rows,
        args.max_instances_per_variant,
        args.target_drop_rate,
        f"{_first_present(variant, 'recipe_variant_id', 'workflow_variant_id')}:select",
        source_usage_counts,
    )
    has_min_balance = len(keep_rows) >= args.min_keep and len(drop_rows) >= args.min_drop
    selected_rows = selected if has_min_balance else []
    return selected if has_min_balance else [], {
        **base,
        'status': 'kept' if selected_rows else 'skipped_unbalanced',
        'candidate_count': len(candidates),
        'value_count': len(values),
        'keep_count': len(keep_rows),
        'drop_count': len(drop_rows),
        'selected_keep_count': sum(1 for row in selected_rows if row['reference_status'] == 'KEEP'),
        'selected_drop_count': sum(1 for row in selected_rows if row['reference_status'] == 'DROP'),
        'selected_count': len(selected_rows),
        'filter_name': filter_name,
        'threshold_meta': threshold_meta,
        'filter_params': calibrated_params,
    }


def _materialize_order_family(
    domain: str,
    recipe: dict[str, Any],
    family: dict[str, Any],
    records: list[dict[str, Any]],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    source_usage_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variants = list(family.get('variants') or [])
    variants_by_slot = {variant['order_slot']: variant for variant in variants}
    if set(variants_by_slot) != {'front', 'middle', 'end'}:
        return [], {
            'domain': domain,
            'recipe_id': _first_present(recipe, 'recipe_id', 'workflow_id'),
            'order_family_id': family.get('order_family_id'),
            'status': 'skipped_missing_slots',
            'selected_group_count': 0,
            'selected_variant_count': 0,
        }

    mapper_names = _mapper_names(variants_by_slot['end']['operator_sequence'], operators_by_name)
    candidates = _supporting_records(
        records,
        mapper_names,
        args.max_candidate_records,
        family['order_family_id'],
        args.max_input_chars,
        source_usage_counts,
    )
    filter_name = family['filter_name']
    base_params = dict(variants_by_slot['front'].get('filter_params') or _base_params(filter_name, operators_by_name))
    values = []
    value_key = None
    usable_records = []
    for record in candidates:
        record_values = []
        try:
            for slot in ('front', 'middle', 'end'):
                sequence = list(variants_by_slot[slot]['operator_sequence'])
                filter_index = _first_filter_index(sequence, operators_by_name)
                if filter_index is None:
                    continue
                value, current_value_key = _filter_value(record, sequence, filter_index, base_params, operators_by_name)
                value_key = value_key or current_value_key
                if value is not None:
                    record_values.append(value)
        except Exception:
            continue
        if record_values:
            values.extend(record_values)
            usable_records.append(record)

    calibrated_params, threshold_meta = _calibrate_filter_params_for_target(
        filter_name,
        base_params,
        values,
        args.target_drop_rate,
        args.min_positive_ratio_threshold,
        args.zero_ratio_threshold_policy,
    )
    threshold_meta = {
        **threshold_meta,
        'filter_name': filter_name,
        'status_value_key': value_key,
        'value_count': len(values),
        'shared_across_order_slots': True,
    }
    if threshold_meta.get('zero_ratio_threshold') and args.zero_ratio_threshold_policy == 'skip':
        return [], {
            'domain': domain,
            'recipe_id': _first_present(recipe, 'recipe_id', 'workflow_id'),
            'order_family_id': family.get('order_family_id'),
            'filter_name': filter_name,
            'status': 'skipped_zero_ratio_threshold',
            'candidate_count': len(candidates),
            'usable_record_count': len(usable_records),
            'value_count': len(values),
            'selected_group_count': 0,
            'selected_variant_count': 0,
            'keep_count': 0,
            'drop_count': 0,
            'threshold_meta': threshold_meta,
            'filter_params': calibrated_params,
        }
    filter_params_by_name = {filter_name: calibrated_params}
    selected_rows = []
    sensitive_group_count = 0
    for record in usable_records:
        slot_rows = []
        signatures = set()
        try:
            for slot in ('front', 'middle', 'end'):
                variant = variants_by_slot[slot]
                sequence = list(variant['operator_sequence'])
                execution = _execute_recipe(record, sequence, operators_by_name, filter_params_by_name)
                full_run_execution = _execute_recipe_without_early_stop(
                    record,
                    sequence,
                    operators_by_name,
                    filter_params_by_name,
                )
                signature = (execution['reference_status'], execution['reference_text'])
                signatures.add(signature)
                base = {
                    'benchmark_track': 'order_sensitivity',
                    'domain': domain,
                    'recipe_id': _first_present(recipe, 'recipe_id', 'workflow_id'),
                    'recipe_variant_id': _first_present(variant, 'recipe_variant_id', 'workflow_variant_id'),
                    'recipe_type': _first_present(variant, 'recipe_type', 'workflow_type'),
                    'order_family_id': family['order_family_id'],
                    'order_slot': slot,
                    'order_group_instance_id': _stable_id(family['order_family_id'], _record_id(record)),
                    'group_success_rule': family.get('group_success_rule', 'all_slots_correct'),
                }
                slot_rows.append(
                    _variant_record(
                        base,
                        record,
                        sequence,
                        filter_params_by_name,
                        execution,
                        full_run_execution,
                        threshold_meta,
                    )
                )
        except Exception:
            continue
        if len(signatures) < 2:
            continue
        selected_rows.extend(slot_rows)
        sensitive_group_count += 1
        if sensitive_group_count >= args.max_order_groups_per_family:
            break

    keep_count = sum(1 for row in selected_rows if row['reference_status'] == 'KEEP')
    drop_count = sum(1 for row in selected_rows if row['reference_status'] == 'DROP')
    enough_groups = sensitive_group_count >= args.min_order_sensitive_groups
    return selected_rows if enough_groups else [], {
        'domain': domain,
        'recipe_id': _first_present(recipe, 'recipe_id', 'workflow_id'),
        'order_family_id': family['order_family_id'],
        'filter_name': filter_name,
        'status': 'kept' if enough_groups else 'skipped_not_order_sensitive_enough',
        'candidate_count': len(candidates),
        'usable_record_count': len(usable_records),
        'value_count': len(values),
        'selected_group_count': sensitive_group_count if enough_groups else 0,
        'selected_variant_count': len(selected_rows) if enough_groups else 0,
        'keep_count': keep_count if enough_groups else 0,
        'drop_count': drop_count if enough_groups else 0,
        'threshold_meta': threshold_meta,
        'filter_params': calibrated_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Materialize benchmark instances and deterministic references without prompts.')
    parser.add_argument('--domains-config', default='configs/domains.yaml')
    parser.add_argument('--recipe-library-dir', '--workflow-library-dir', dest='recipe_library_dir', default='data/processed/recipe_library')
    parser.add_argument('--filtered-path', default='data/processed/domain_filtered/all.jsonl')
    parser.add_argument('--output-dir', default='data/processed/benchmark_instances')
    parser.add_argument('--max-candidate-records', type=int, default=0, help='Candidate cap per recipe/order family; 0 means no cap.')
    parser.add_argument('--max-instances-per-variant', type=int, default=20)
    parser.add_argument('--max-order-groups-per-family', type=int, default=10)
    parser.add_argument('--min-keep', type=int, default=5)
    parser.add_argument('--min-drop', type=int, default=5)
    parser.add_argument('--min-order-sensitive-groups', type=int, default=5)
    parser.add_argument('--target-drop-rate', type=float, default=0.5)
    parser.add_argument('--max-atomic-candidate-records', type=int, default=0, help='Candidate cap per atomic operator; 0 means no cap.')
    parser.add_argument('--max-atomic-instances-per-op', type=int, default=20)
    parser.add_argument('--min-atomic-keep', type=int, default=5)
    parser.add_argument('--min-atomic-drop', type=int, default=5)
    parser.add_argument('--skip-atomic', action='store_true', help='Only materialize main/order instances.')
    parser.add_argument(
        '--max-input-chars',
        type=int,
        default=50_000,
        help='Skip raw inputs longer than this many characters before materialization; 0 disables the limit.',
    )
    parser.add_argument(
        '--min-positive-ratio-threshold',
        type=float,
        default=0.001,
        help='Use this threshold when a calibrated ratio threshold rounds to 0 and the zero-ratio policy is min-positive.',
    )
    parser.add_argument(
        '--zero-ratio-threshold-policy',
        choices=('min-positive', 'skip'),
        default='min-positive',
        help='Handle calibrated ratio thresholds equal to 0 by trying a small positive threshold or skipping the variant.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Reuse per-variant cache shards in output-dir/_materialize_cache_v3 after an interrupted run.',
    )
    args = parser.parse_args()

    root = ROOT
    recipe_library_dir = (root / args.recipe_library_dir).resolve()
    filtered_path = (root / args.filtered_path).resolve()
    output_dir = (root / args.output_dir).resolve()
    cache_dir = output_dir / '_materialize_cache_v3'

    if not recipe_library_dir.exists():
        raise SystemExit(f'recipe library dir not found: {recipe_library_dir}')
    if not filtered_path.exists():
        raise SystemExit(f'filtered corpus not found: {filtered_path}')
    if not 0.0 < args.target_drop_rate < 1.0:
        raise SystemExit('--target-drop-rate must be in (0, 1)')
    if args.max_input_chars < 0:
        raise SystemExit('--max-input-chars must be >= 0')
    if args.max_candidate_records < 0 or args.max_atomic_candidate_records < 0:
        raise SystemExit('candidate caps must be >= 0; use 0 for no cap')
    if args.min_positive_ratio_threshold < 0:
        raise SystemExit('--min-positive-ratio-threshold must be >= 0')
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists() and not args.resume:
        shutil.rmtree(cache_dir)

    domains_cfg = load_domains_config(root / args.domains_config)
    plan = build_domain_execution_plan(domains_cfg)
    operators_by_name = _operator_lookup(plan)

    _log(f'loading filtered records -> {filtered_path}')
    records_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(filtered_path):
        domain = record.get('domain')
        if domain:
            records_by_domain[str(domain)].append(record)
    _log(
        'loaded records by domain: '
        + ', '.join(f'{domain}={len(records)}' for domain, records in sorted(records_by_domain.items()))
    )

    domain_recipes = _load_domain_recipes(recipe_library_dir)
    _log(f'loaded recipe libraries for {len(domain_recipes)} domains -> {recipe_library_dir}')

    main_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    atomic_rows: list[dict[str, Any]] = []
    main_summary_rows: list[dict[str, Any]] = []
    order_summary_rows: list[dict[str, Any]] = []
    atomic_summary_rows: list[dict[str, Any]] = []
    source_usage_counts: dict[str, int] = defaultdict(int)

    if not args.skip_atomic:
        cached = _load_cache(cache_dir, 'atomic', 'atomic_ops') if args.resume else None
        if cached is not None:
            atomic_rows, atomic_summary_rows = cached
            _mark_source_usage(atomic_rows, source_usage_counts)
            _log(f'atomic global: cached selected={len(atomic_rows)} summaries={len(atomic_summary_rows)}')
        else:
            atomic_rows, atomic_summary_rows = _materialize_atomic_ops(
                records_by_domain,
                plan,
                operators_by_name,
                args,
                source_usage_counts,
            )
            _write_cache(cache_dir, 'atomic', 'atomic_ops', atomic_rows, atomic_summary_rows)

    for domain_index, (domain, domain_yaml) in enumerate(sorted(domain_recipes.items()), start=1):
        recipes = list(domain_yaml.get('recipes') or domain_yaml.get('workflows') or [])
        records = records_by_domain.get(domain, [])
        _log(f'[{domain_index}/{len(domain_recipes)}] {domain}: {len(recipes)} recipes, {len(records)} records')
        for recipe_index, recipe in enumerate(recipes, start=1):
            recipe_id = _first_present(recipe, 'recipe_id', 'workflow_id')
            main_variants = list(recipe.get('main_recipe_variants') or recipe.get('main_workflow_variants') or [])
            order_families = list(recipe.get('order_sensitivity_families') or [])
            _log(
                f'  [{recipe_index}/{len(recipes)}] {domain}/{recipe_id}: '
                f'{len(main_variants)} main variants, {len(order_families)} order families'
            )
            for variant in main_variants:
                variant_id = _first_present(variant, 'recipe_variant_id', 'workflow_variant_id')
                cached = _load_cache(cache_dir, 'main', variant_id) if args.resume else None
                if cached is not None:
                    rows, summary = cached
                    _mark_source_usage(rows, source_usage_counts)
                else:
                    rows, summary = _materialize_main_variant(
                        domain,
                        recipe,
                        variant,
                        records,
                        operators_by_name,
                        args,
                        source_usage_counts,
                    )
                    _write_cache(cache_dir, 'main', variant_id, rows, summary)
                    _mark_source_usage(rows, source_usage_counts)
                main_rows.extend(rows)
                main_summary_rows.append(summary)
                _log(
                    f"    main {variant_id}: {summary['status']} "
                    f"selected={summary['selected_count']} keep={summary.get('keep_count', 0)} drop={summary.get('drop_count', 0)}"
                )

            for family in order_families:
                family_id = family['order_family_id']
                cached = _load_cache(cache_dir, 'order_sensitivity', family_id) if args.resume else None
                if cached is not None:
                    rows, summary = cached
                    _mark_source_usage(rows, source_usage_counts)
                else:
                    rows, summary = _materialize_order_family(
                        domain,
                        recipe,
                        family,
                        records,
                        operators_by_name,
                        args,
                        source_usage_counts,
                    )
                    _write_cache(cache_dir, 'order_sensitivity', family_id, rows, summary)
                    _mark_source_usage(rows, source_usage_counts)
                order_rows.extend(rows)
                order_summary_rows.append(summary)
                _log(
                    f"    order {family_id}: {summary['status']} "
                    f"groups={summary.get('selected_group_count', 0)} variants={summary.get('selected_variant_count', 0)}"
                )

    main_count = _write_jsonl(output_dir / 'main.jsonl', main_rows)
    order_count = _write_jsonl(output_dir / 'order_sensitivity.jsonl', order_rows)
    atomic_count = _write_jsonl(output_dir / 'atomic_ops.jsonl', atomic_rows) if not args.skip_atomic else 0
    pd.DataFrame(main_summary_rows).to_csv(output_dir / 'main_summary.csv', index=False)
    pd.DataFrame(order_summary_rows).to_csv(output_dir / 'order_sensitivity_summary.csv', index=False)
    if not args.skip_atomic:
        pd.DataFrame(atomic_summary_rows).to_csv(output_dir / 'atomic_ops_summary.csv', index=False)

    _log(f'wrote main instances: {main_count} -> {output_dir / "main.jsonl"}')
    _log(f'wrote order-sensitivity instances: {order_count} -> {output_dir / "order_sensitivity.jsonl"}')
    if not args.skip_atomic:
        _log(f'wrote atomic-op instances: {atomic_count} -> {output_dir / "atomic_ops.jsonl"}')
    _log(f'wrote summaries -> {output_dir}')


if __name__ == '__main__':
    main()
