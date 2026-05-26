#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

from cdrbench.config import load_domains_config
from cdrbench.canonical_ops import apply_canonical_mapper, has_canonical_mapper
from cdrbench.dj_operator_loader import Fields, create_operator
from cdrbench.domain_assignment import build_domain_execution_plan


FILTER_STATUS_RULES: dict[str, dict[str, Any]] = {
    'alphanumeric_filter': {
        'value_key': lambda params: 'alpha_token_ratio' if params.get('tokenization') else 'alnum_ratio',
        'min_key': 'min_ratio',
        'max_key': 'max_ratio',
    },
    'average_line_length_filter': {'value_key': 'avg_line_length', 'min_key': 'min_len', 'max_key': 'max_len'},
    'character_repetition_filter': {'value_key': 'char_rep_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'flagged_words_filter': {'value_key': 'flagged_words_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'maximum_line_length_filter': {'value_key': 'max_line_length', 'min_key': 'min_len', 'max_key': 'max_len'},
    'stopwords_filter': {'value_key': 'stopwords_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'text_length_filter': {'value_key': 'text_len', 'min_key': 'min_len', 'max_key': 'max_len'},
    'word_repetition_filter': {'value_key': 'word_rep_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'words_num_filter': {'value_key': 'num_words', 'min_key': 'min_num', 'max_key': 'max_num'},
}

FILTER_CALIBRATION_RULES: dict[str, dict[str, Any]] = {
    'alphanumeric_filter': {'direction': 'min', 'quantile': 0.20},
    'average_line_length_filter': {'direction': 'max', 'quantile': 0.80},
    'character_repetition_filter': {'direction': 'max', 'quantile': 0.80},
    'flagged_words_filter': {'direction': 'max', 'quantile': 0.80},
    'maximum_line_length_filter': {'direction': 'max', 'quantile': 0.80},
    'stopwords_filter': {'direction': 'min', 'quantile': 0.20},
    'text_length_filter': {'direction': 'min', 'quantile': 0.20},
    'word_repetition_filter': {'direction': 'max', 'quantile': 0.80},
    'words_num_filter': {'direction': 'min', 'quantile': 0.20},
}
INTEGER_THRESHOLD_KEYS = {'min_len', 'max_len', 'min_num', 'max_num'}
RATIO_THRESHOLD_KEYS = {'min_ratio', 'max_ratio'}

SAFE_MAX_CHARS_FOR_EXPENSIVE_MAPPERS = 80_000
EXPENSIVE_LONG_TEXT_MAPPERS = {
    'remove_repeat_sentences_mapper',
    'remove_words_with_incorrect_substrings_mapper',
}
DOMAIN_OUTPUT_FILES = [
    'recipe_library.yaml',
    'recipe_variants.csv',
    'filter_attachments.csv',
    'checkpoint_filter_stats.csv',
    'order_sensitivity_candidates.csv',
    'order_sensitivity_families.csv',
]


def _log(message: str) -> None:
    print(message, flush=True)


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                preview = line[:200]
                raise ValueError(
                    f'Invalid JSONL record in {path} at line {lineno}: {exc.msg}. Line preview: {preview!r}'
                ) from exc


def _call_optional_context(method, payload):
    try:
        return method(payload, context=True)
    except TypeError:
        return method(payload)


def _build_batch(text: str, suffix: str) -> dict[str, Any]:
    return {
        'text': [text],
        Fields.stats: [{}],
        Fields.context: [{}],
        Fields.meta: [{}],
        Fields.suffix: [suffix],
    }


def _build_sample(text: str, suffix: str) -> dict[str, Any]:
    return {
        'text': text,
        Fields.stats: {},
        Fields.context: {},
        Fields.meta: {},
        Fields.suffix: suffix,
    }


def _infer_suffix(record: dict[str, Any]) -> str:
    source_name = record.get('source_name')
    if isinstance(source_name, str) and source_name:
        suffix = Path(source_name).suffix
        if suffix:
            return suffix

    url = record.get('url')
    if isinstance(url, str) and url:
        suffix = Path(url).suffix
        if suffix:
            return suffix

    text = str(record.get('text', ''))
    lowered = text.lower()
    if '\\begin{document}' in lowered or '\\section' in lowered:
        return '.tex'
    if '<html' in lowered or '</html>' in lowered:
        return '.html'
    return ''


def _apply_mapper_text(op_name: str, text: str, params: dict[str, Any], suffix: str) -> tuple[str, dict[str, Any]]:
    if op_name in EXPENSIVE_LONG_TEXT_MAPPERS and len(text) > SAFE_MAX_CHARS_FOR_EXPENSIVE_MAPPERS:
        return text, {
            'active': False,
            'skipped': 'text_too_long_for_expensive_mapper',
            'output_length': len(text),
            'delta_chars': 0,
        }

    if has_canonical_mapper(op_name):
        output_text = apply_canonical_mapper(op_name, text, params, suffix)
        return output_text, {
            'active': output_text != text,
            'canonical_operator': True,
            'output_length': len(output_text),
            'delta_chars': len(output_text) - len(text),
        }

    op = create_operator(op_name, **params)
    if hasattr(op, 'process_batched') and op.is_batched_op():
        result = op.process_batched(_build_batch(text, suffix))
        output_text = result['text'][0]
    else:
        result = op.process_single(_build_sample(text, suffix))
        output_text = result['text']

    return output_text, {
        'active': output_text != text,
        'output_length': len(output_text),
        'delta_chars': len(output_text) - len(text),
    }


def _evaluate_filter(op_name: str, text: str, params: dict[str, Any], suffix: str) -> dict[str, Any]:
    op = create_operator(op_name, **params)
    if hasattr(op, 'compute_stats_batched') and op.is_batched_op():
        batch = _call_optional_context(op.compute_stats_batched, _build_batch(text, suffix))
        keep_iter = op.process_batched(batch)
        keep = list(keep_iter)[0] if not isinstance(keep_iter, list) else keep_iter[0]
        stats = batch[Fields.stats][0]
    else:
        sample = _build_sample(text, suffix)
        if hasattr(op, 'compute_stats_single'):
            sample = _call_optional_context(op.compute_stats_single, sample)
        keep = op.process_single(sample)
        stats = sample[Fields.stats]

    return {
        'keep': bool(keep),
        'status': 'KEEP' if keep else 'DROP',
        'stats': stats,
    }


def _resolve_status_value_key(op_name: str, params: dict[str, Any]) -> str | None:
    rule = FILTER_STATUS_RULES.get(op_name)
    if rule is None:
        return None
    value_key = rule['value_key']
    return value_key(params) if callable(value_key) else value_key


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _round_float(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _integer_threshold_step(value: float) -> int:
    abs_value = abs(value)
    if abs_value <= 50:
        return 5
    if abs_value <= 100:
        return 10
    if abs_value <= 1_000:
        return 50
    if abs_value <= 10_000:
        return 100
    return 1_000


def _ratio_threshold_step(value: float) -> float:
    abs_value = abs(value)
    if abs_value < 0.001:
        return 0.0001
    if abs_value < 0.01:
        return 0.001
    return 0.01


def _format_threshold_value(value: float | None, param_key: str) -> int | float | None:
    if value is None:
        return None
    if param_key in INTEGER_THRESHOLD_KEYS:
        step = _integer_threshold_step(value)
        rounded = int(round(value / step) * step)
        return max(5, rounded) if value > 0 else rounded
    if param_key in RATIO_THRESHOLD_KEYS:
        step = _ratio_threshold_step(value)
        rounded = round(value / step) * step
        if value > 0 and rounded == 0:
            rounded = step
        decimals = 4 if step == 0.0001 else 3 if step == 0.001 else 2
        return round(min(max(rounded, 0.0), 1.0), decimals)
    return _round_float(value)


def _summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        'count': len(values),
        'mean': _round_float(mean(values)) if values else None,
        'min': _round_float(min(values)) if values else None,
        'p10': _round_float(_percentile(values, 0.10)),
        'p20': _round_float(_percentile(values, 0.20)),
        'p50': _round_float(_percentile(values, 0.50)),
        'p80': _round_float(_percentile(values, 0.80)),
        'p90': _round_float(_percentile(values, 0.90)),
        'max': _round_float(max(values)) if values else None,
    }


def _calibrate_filter_params(op_name: str, base_params: dict[str, Any], values: list[float]) -> dict[str, Any]:
    rule = FILTER_STATUS_RULES.get(op_name)
    calibration = FILTER_CALIBRATION_RULES.get(op_name)
    params = dict(base_params)
    if not rule or not calibration or not values:
        return params

    threshold = _percentile(values, float(calibration['quantile']))
    if threshold is None:
        return params

    if calibration['direction'] == 'min':
        params[rule['min_key']] = _format_threshold_value(threshold, rule['min_key'])
        params.pop(rule['max_key'], None)
    else:
        params[rule['max_key']] = _format_threshold_value(threshold, rule['max_key'])
        params.pop(rule['min_key'], None)
    return params


def _threshold_rule_label(op_name: str) -> str:
    calibration = FILTER_CALIBRATION_RULES.get(op_name)
    if not calibration:
        return 'not_calibrated'
    pct = int(float(calibration['quantile']) * 100)
    return f"{calibration['direction']}_p{pct}"


def _parse_operator_set(blob: str) -> list[str]:
    return [item.strip() for item in blob.split(' | ') if item.strip()]


def _clean_optional_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return None
    return text


def _labeling_meta(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get('meta')
    if not isinstance(meta, dict):
        return {}
    payload = meta.get('cdrbench_domain_labeling')
    if not isinstance(payload, dict):
        payload = meta.get('icdrbench_domain_labeling')
    return payload if isinstance(payload, dict) else {}


def _recipe_rows_for_domain(domain_dir: Path) -> list[dict[str, Any]]:
    recipe_csv = domain_dir / 'selected_recipes.csv'
    if not recipe_csv.exists():
        recipe_csv = domain_dir / 'selected_workflows.csv'
    if not recipe_csv.exists():
        return []
    df = pd.read_csv(recipe_csv)
    if 'selection_source' in df.columns:
        df = df[df['selection_source'] != 'coverage_fallback_unassigned_signature']
    elif 'selection_reason' in df.columns:
        df = df[df['selection_reason'] != 'coverage_fallback_unassigned_signature']
    return df.to_dict(orient='records')


def _domain_outputs_complete(domain_out_dir: Path) -> bool:
    return all((domain_out_dir / filename).exists() for filename in DOMAIN_OUTPUT_FILES)


def _load_domain_yaml(domain_out_dir: Path) -> dict[str, Any] | None:
    try:
        recipe_yaml = domain_out_dir / 'recipe_library.yaml'
        legacy_workflow_yaml = domain_out_dir / 'workflow_library.yaml'
        target_yaml = recipe_yaml if recipe_yaml.exists() else legacy_workflow_yaml
        with target_yaml.open('r', encoding='utf-8') as f:
            payload = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    recipes = payload.get('recipes')
    legacy_workflows = payload.get('workflows')
    if isinstance(recipes, list) or isinstance(legacy_workflows, list):
        return payload
    return None


def _summary_rows_from_domain_yaml(domain_yaml: dict[str, Any]) -> list[dict[str, Any]]:
    domain = domain_yaml.get('domain')
    rows = []
    for recipe in domain_yaml.get('recipes') or domain_yaml.get('workflows', []):
        if not isinstance(recipe, dict):
            continue
        main_variants = recipe.get('main_recipe_variants') or recipe.get('main_workflow_variants') or []
        order_variants = recipe.get('order_sensitivity_recipe_variants') or recipe.get('order_sensitivity_variants') or []
        order_families = recipe.get('order_sensitivity_families') or []
        rows.append(
            {
                'domain': domain,
                'recipe_id': _first_present(recipe, 'recipe_id', 'workflow_id'),
                'support': int(recipe.get('support', 0) or 0),
                'mapper_length': len(recipe.get('ordered_clean_sequence') or []),
                'num_main_variants': len(main_variants),
                'num_order_sensitivity_variants': len(order_variants),
                'num_order_sensitivity_families': len(order_families),
                'num_filter_then_clean': sum(
                    1 for variant in main_variants if _first_present(variant, 'recipe_type', 'workflow_type') == 'filter-then-clean'
                ),
                'num_clean_then_filter': sum(
                    1 for variant in main_variants if _first_present(variant, 'recipe_type', 'workflow_type') == 'clean-then-filter'
                ),
            }
        )
    return rows


def _ordered_mapper_sequence(
    domain: str,
    operator_set: list[str],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    domain_profile = plan['domain_profiles'][domain]
    variants_by_key = plan['execution_variants_by_key']
    wanted = set(operator_set)
    ordered: list[dict[str, Any]] = []
    for key in domain_profile['mapper_keys']:
        variant = variants_by_key[key]
        if variant['name'] in wanted:
            ordered.append(variant)
    seen = {variant['name'] for variant in ordered}
    for op_name in operator_set:
        if op_name in seen:
            continue
        for variant in plan['execution_variants']:
            if variant['kind'] == 'mapper' and variant['name'] == op_name:
                ordered.append(variant)
                break
    return ordered


def _domain_filter_variants(domain: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    domain_profile = plan['domain_profiles'][domain]
    variants_by_key = plan['execution_variants_by_key']
    return [variants_by_key[key] for key in domain_profile['filter_keys']]


def _supporting_records(
    records: list[dict[str, Any]],
    operator_set: list[str],
    max_records: int,
) -> list[dict[str, Any]]:
    wanted = set(operator_set)
    supported = []
    for record in records:
        meta = _labeling_meta(record)
        active = set(meta.get('active_mapper_names', []))
        if wanted.issubset(active):
            supported.append(record)
    return supported[:max_records]


def _replay_mapper_checkpoints(
    record: dict[str, Any],
    ordered_mappers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    suffix = _infer_suffix(record)
    current_text = str(record.get('text', ''))
    checkpoints = [
        {
            'checkpoint_id': 'S0',
            'step_index': 0,
            'after_operator': None,
            'text': current_text,
        }
    ]
    for idx, variant in enumerate(ordered_mappers, start=1):
        current_text, mapper_result = _apply_mapper_text(variant['name'], current_text, dict(variant.get('params', {})), suffix)
        checkpoints.append(
            {
                'checkpoint_id': f'S{idx}',
                'step_index': idx,
                'after_operator': variant['name'],
                'text': current_text,
                'mapper_result': mapper_result,
            }
        )
    return checkpoints


def _collect_checkpoint_filter_stats(
    ordered_mappers: list[dict[str, Any]],
    filter_variants: list[dict[str, Any]],
    support_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    checkpoint_info: dict[str, dict[str, Any]] = {}
    filter_info: dict[str, dict[str, Any]] = {}

    for record in support_records:
        suffix = _infer_suffix(record)
        try:
            checkpoints = _replay_mapper_checkpoints(record, ordered_mappers)
        except Exception:
            continue

        for checkpoint in checkpoints:
            checkpoint_info[checkpoint['checkpoint_id']] = {
                'checkpoint_id': checkpoint['checkpoint_id'],
                'step_index': checkpoint['step_index'],
                'after_operator': checkpoint['after_operator'],
            }
            for filter_variant in filter_variants:
                op_name = filter_variant['name']
                params = dict(filter_variant.get('params', {}))
                value_key = _resolve_status_value_key(op_name, params)
                if value_key is None:
                    continue
                try:
                    evaluation = _evaluate_filter(op_name, checkpoint['text'], params, suffix)
                except Exception:
                    continue
                value = evaluation.get('stats', {}).get(value_key)
                if isinstance(value, (int, float)):
                    raw_values[(op_name, checkpoint['checkpoint_id'])].append(float(value))
                    filter_info[op_name] = {
                        'filter_name': op_name,
                        'filter_params': params,
                        'status_value_key': value_key,
                    }

    checkpoint_rows: list[dict[str, Any]] = []
    attachment_candidates: list[dict[str, Any]] = []
    previous_by_filter: dict[str, dict[str, Any]] = {}

    for filter_name in sorted(filter_info):
        checkpoints = sorted(checkpoint_info.values(), key=lambda row: row['step_index'])
        previous_by_filter[filter_name] = {}
        for checkpoint in checkpoints:
            values = raw_values.get((filter_name, checkpoint['checkpoint_id']), [])
            if not values:
                continue
            summary = _summarize_values(values)
            prev = previous_by_filter[filter_name]
            prev_mean = prev.get('mean')
            mean_value = summary['mean']
            delta = None
            rel_delta = None
            if isinstance(prev_mean, (int, float)) and isinstance(mean_value, (int, float)):
                delta = mean_value - prev_mean
                if abs(prev_mean) > 1e-12:
                    rel_delta = delta / abs(prev_mean)

            base_params = dict(filter_info[filter_name]['filter_params'])
            calibrated_params = _calibrate_filter_params(filter_name, base_params, values)
            candidate = {
                **filter_info[filter_name],
                **checkpoint,
                'support_records': summary['count'],
                'threshold_selection_rule': _threshold_rule_label(filter_name),
                'calibrated_filter_params': calibrated_params,
                'stat_mean': summary['mean'],
                'stat_min': summary['min'],
                'stat_p10': summary['p10'],
                'stat_p20': summary['p20'],
                'stat_p50': summary['p50'],
                'stat_p80': summary['p80'],
                'stat_p90': summary['p90'],
                'stat_max': summary['max'],
                'delta_from_prev_mean': _round_float(delta),
                'relative_delta_from_prev_mean': _round_float(rel_delta),
            }
            checkpoint_rows.append(candidate)
            attachment_candidates.append(
                {
                    **candidate,
                    'selection_score': (
                        summary['count'],
                        abs(rel_delta or 0.0),
                        abs(delta or 0.0),
                        checkpoint['step_index'],
                    ),
                }
            )
            previous_by_filter[filter_name] = summary

    return checkpoint_rows, attachment_candidates


def _select_stage_attachments(
    attachment_candidates: list[dict[str, Any]],
    *,
    stage: str,
    final_step_index: int,
    min_filter_support: int,
    max_filters_per_recipe: int,
) -> list[dict[str, Any]]:
    if stage == 'raw':
        candidates = [row for row in attachment_candidates if row['step_index'] == 0]
    elif stage == 'final':
        candidates = [row for row in attachment_candidates if row['step_index'] == final_step_index]
    elif stage == 'middle':
        candidates = [row for row in attachment_candidates if 0 < row['step_index'] < final_step_index]
    else:
        candidates = list(attachment_candidates)

    candidates = [row for row in candidates if row['support_records'] >= min_filter_support]
    candidates.sort(
        key=lambda row: (
            -row['selection_score'][0],
            -row['selection_score'][1],
            -row['selection_score'][2],
            row['filter_name'],
        )
    )
    selected: list[dict[str, Any]] = []
    seen_filters: set[str] = set()
    for row in candidates:
        if row['filter_name'] in seen_filters:
            continue
        seen_filters.add(row['filter_name'])
        selected.append({k: v for k, v in row.items() if k != 'selection_score'})
        if len(selected) >= max_filters_per_recipe:
            break
    return selected


def _select_order_sensitivity_families(
    recipe_id: str,
    attachment_candidates: list[dict[str, Any]],
    *,
    final_step_index: int,
    min_filter_support: int,
    max_families_per_recipe: int,
) -> list[dict[str, Any]]:
    eligible = [row for row in attachment_candidates if row['support_records'] >= min_filter_support]
    by_filter: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in eligible:
        step_index = int(row['step_index'])
        if step_index == 0:
            slot = 'front'
        elif step_index == final_step_index:
            slot = 'end'
        elif 0 < step_index < final_step_index:
            slot = 'middle'
        else:
            continue
        by_filter[row['filter_name']][slot].append(row)

    families: list[dict[str, Any]] = []
    for filter_name, slots in by_filter.items():
        if not {'front', 'middle', 'end'} <= set(slots):
            continue

        front = _best_attachment(slots['front'])
        middle = _best_attachment(slots['middle'])
        end = _best_attachment(slots['end'])
        min_support = min(front['support_records'], middle['support_records'], end['support_records'])
        sensitivity_score = max(
            abs(middle.get('relative_delta_from_prev_mean') or 0.0),
            abs(end.get('relative_delta_from_prev_mean') or 0.0),
            abs(middle.get('delta_from_prev_mean') or 0.0),
            abs(end.get('delta_from_prev_mean') or 0.0),
        )
        family = {
            'order_family_id': f'{recipe_id}__order_family__{filter_name}',
            'filter_name': filter_name,
            'front': {k: v for k, v in front.items() if k != 'selection_score'},
            'middle': {k: v for k, v in middle.items() if k != 'selection_score'},
            'end': {k: v for k, v in end.items() if k != 'selection_score'},
            'min_support_records': min_support,
            'selection_score': (min_support, sensitivity_score, filter_name),
        }
        families.append(family)

    families.sort(key=lambda row: (-row['selection_score'][0], -row['selection_score'][1], row['filter_name']))
    return [{k: v for k, v in row.items() if k != 'selection_score'} for row in families[:max_families_per_recipe]]


def _best_attachment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (
            -row['selection_score'][0],
            -row['selection_score'][1],
            -row['selection_score'][2],
            row['step_index'],
            row['filter_name'],
        ),
    )[0]


def _order_variant(
    recipe_id: str,
    order_family_id: str,
    mapper_names: list[str],
    slot: str,
    attachment: dict[str, Any],
) -> dict[str, Any]:
    if slot == 'front':
        recipe_type = 'filter-then-clean'
        operator_sequence = [attachment['filter_name'], *mapper_names]
    elif slot == 'end':
        recipe_type = 'clean-then-filter'
        operator_sequence = [*mapper_names, attachment['filter_name']]
    elif slot == 'middle':
        recipe_type = 'clean-filter-clean'
        split_at = int(attachment['step_index'])
        operator_sequence = [*mapper_names[:split_at], attachment['filter_name'], *mapper_names[split_at:]]
    else:
        raise ValueError(f'unknown order-sensitivity slot: {slot}')

    return {
        'recipe_variant_id': f'{order_family_id}__{slot}',
        'order_family_id': order_family_id,
        'order_slot': slot,
        'recipe_type': recipe_type,
        'benchmark_track': 'order_sensitivity',
        'operator_sequence': operator_sequence,
        'filter_name': attachment['filter_name'],
        'filter_checkpoint_id': attachment['checkpoint_id'],
        'filter_step_index': attachment['step_index'],
        'filter_params': attachment['calibrated_filter_params'],
        'threshold_selection_rule': attachment['threshold_selection_rule'],
    }


def _materialize_variants(
    recipe_id: str,
    ordered_mappers: list[dict[str, Any]],
    *,
    raw_attachments: list[dict[str, Any]],
    final_attachments: list[dict[str, Any]],
    order_families: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mapper_names = [variant['name'] for variant in ordered_mappers]
    main_variants = [
        {
            'recipe_variant_id': f'{recipe_id}__clean_only',
            'recipe_type': 'clean-only',
            'benchmark_track': 'main',
            'operator_sequence': mapper_names,
            'filter_name': None,
            'filter_checkpoint_id': None,
            'filter_step_index': None,
            'filter_params': None,
        }
    ]

    for idx, attachment in enumerate(raw_attachments, start=1):
        main_variants.append(
            {
                'recipe_variant_id': f'{recipe_id}__filter_then_clean_{idx:02d}',
                'recipe_type': 'filter-then-clean',
                'benchmark_track': 'main',
                'operator_sequence': [attachment['filter_name'], *mapper_names],
                'filter_name': attachment['filter_name'],
                'filter_checkpoint_id': attachment['checkpoint_id'],
                'filter_step_index': attachment['step_index'],
                'filter_params': attachment['calibrated_filter_params'],
                'threshold_selection_rule': attachment['threshold_selection_rule'],
            }
        )

    for idx, attachment in enumerate(final_attachments, start=1):
        main_variants.append(
            {
                'recipe_variant_id': f'{recipe_id}__clean_then_filter_{idx:02d}',
                'recipe_type': 'clean-then-filter',
                'benchmark_track': 'main',
                'operator_sequence': [*mapper_names, attachment['filter_name']],
                'filter_name': attachment['filter_name'],
                'filter_checkpoint_id': attachment['checkpoint_id'],
                'filter_step_index': attachment['step_index'],
                'filter_params': attachment['calibrated_filter_params'],
                'threshold_selection_rule': attachment['threshold_selection_rule'],
            }
        )

    order_variants: list[dict[str, Any]] = []
    materialized_order_families: list[dict[str, Any]] = []
    for family in order_families:
        family_variants = [
            _order_variant(recipe_id, family['order_family_id'], mapper_names, 'front', family['front']),
            _order_variant(recipe_id, family['order_family_id'], mapper_names, 'middle', family['middle']),
            _order_variant(recipe_id, family['order_family_id'], mapper_names, 'end', family['end']),
        ]
        order_variants.extend(family_variants)
        materialized_order_families.append(
            {
                'order_family_id': family['order_family_id'],
                'filter_name': family['filter_name'],
                'min_support_records': family['min_support_records'],
                'required_slots_for_group_success': ['front', 'middle', 'end'],
                'group_success_rule': 'all_slots_correct',
                'variants': family_variants,
            }
        )
    return main_variants, order_variants, materialized_order_families


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Materialize main and order-sensitivity recipe drafts from mined clean recipes.'
    )
    parser.add_argument('--domains-config', default='configs/domains.yaml')
    parser.add_argument('--recipe-mining-dir', '--workflow-mining-dir', dest='recipe_mining_dir', default='data/processed/recipe_mining')
    parser.add_argument('--filtered-path', default='data/processed/domain_filtered/all.jsonl')
    parser.add_argument('--output-dir', default='data/processed/recipe_library')
    parser.add_argument('--max-support-records', type=int, default=128)
    parser.add_argument('--min-filter-support', type=int, default=5)
    parser.add_argument('--max-filters-per-recipe', '--max-filters-per-workflow', dest='max_filters_per_recipe', type=int, default=3)
    parser.add_argument('--resume', action='store_true', help='Skip domains whose recipe-library outputs already exist.')
    args = parser.parse_args()

    root = ROOT
    domains_cfg = load_domains_config(root / args.domains_config)
    plan = build_domain_execution_plan(domains_cfg)
    recipe_mining_dir = (root / args.recipe_mining_dir).resolve()
    filtered_path = (root / args.filtered_path).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not filtered_path.exists():
        raise SystemExit(f'filtered corpus not found: {filtered_path}')
    if not recipe_mining_dir.exists():
        raise SystemExit(f'recipe mining dir not found: {recipe_mining_dir}')

    _log(f'loading filtered corpus -> {filtered_path}')
    records_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(filtered_path):
        domain = record.get('domain')
        if domain:
            records_by_domain[str(domain)].append(record)
    _log(
        'loaded records by domain: '
        + ', '.join(f'{domain}={len(records)}' for domain, records in sorted(records_by_domain.items()))
    )

    summary_rows: list[dict[str, Any]] = []
    global_yaml = {'domains': {}}
    domain_dirs = sorted(path for path in recipe_mining_dir.iterdir() if path.is_dir())
    _log(f'found {len(domain_dirs)} recipe-mining domain dirs -> {recipe_mining_dir}')

    for domain_index, domain_dir in enumerate(domain_dirs, start=1):
        domain = domain_dir.name
        domain_out_dir = output_dir / domain
        if args.resume and _domain_outputs_complete(domain_out_dir):
            existing_domain_yaml = _load_domain_yaml(domain_out_dir)
            if existing_domain_yaml is not None:
                global_yaml['domains'][domain] = existing_domain_yaml
                summary_rows.extend(_summary_rows_from_domain_yaml(existing_domain_yaml))
                _log(f'[{domain_index}/{len(domain_dirs)}] {domain}: resume skip complete outputs')
                continue
            _log(f'[{domain_index}/{len(domain_dirs)}] {domain}: resume found outputs but yaml is unreadable; recomputing')

        recipe_rows = _recipe_rows_for_domain(domain_dir)
        if not recipe_rows:
            _log(f'[{domain_index}/{len(domain_dirs)}] {domain}: no selected_recipes.csv rows; skip')
            continue

        domain_records = records_by_domain.get(domain, [])
        _log(
            f'[{domain_index}/{len(domain_dirs)}] {domain}: materializing '
            f'{len(recipe_rows)} recipes using {len(domain_records)} domain records'
        )
        ordered_filter_variants = _domain_filter_variants(domain, plan)
        domain_yaml = {'domain': domain, 'recipes': []}
        attachment_rows: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        order_candidate_rows: list[dict[str, Any]] = []
        order_family_rows: list[dict[str, Any]] = []
        variant_rows: list[dict[str, Any]] = []

        for recipe_index, row in enumerate(recipe_rows, start=1):
            recipe_id = _clean_optional_id(_first_present(row, 'recipe_id', 'workflow_id')) or f'{domain}_recipe_auto_{recipe_index:03d}'
            operator_set = _parse_operator_set(str(row['operators']))
            ordered_mappers = _ordered_mapper_sequence(domain, operator_set, plan)
            support_records = _supporting_records(
                domain_records,
                operator_set,
                max_records=args.max_support_records,
            )
            _log(
                f'  [{recipe_index}/{len(recipe_rows)}] {domain}/{recipe_id}: '
                f'{len(ordered_mappers)} clean ops, {len(support_records)} support records'
            )
            filter_checkpoint_rows, attachment_candidates = _collect_checkpoint_filter_stats(
                ordered_mappers=ordered_mappers,
                filter_variants=ordered_filter_variants,
                support_records=support_records,
            )
            final_step_index = len(ordered_mappers)
            raw_attachments = _select_stage_attachments(
                attachment_candidates,
                stage='raw',
                final_step_index=final_step_index,
                min_filter_support=args.min_filter_support,
                max_filters_per_recipe=args.max_filters_per_recipe,
            )
            final_attachments = _select_stage_attachments(
                attachment_candidates,
                stage='final',
                final_step_index=final_step_index,
                min_filter_support=args.min_filter_support,
                max_filters_per_recipe=args.max_filters_per_recipe,
            )
            order_families = _select_order_sensitivity_families(
                recipe_id,
                attachment_candidates,
                final_step_index=final_step_index,
                min_filter_support=args.min_filter_support,
                max_families_per_recipe=args.max_filters_per_recipe,
            )
            main_variants, order_variants, materialized_order_families = _materialize_variants(
                recipe_id,
                ordered_mappers,
                raw_attachments=raw_attachments,
                final_attachments=final_attachments,
                order_families=order_families,
            )
            _log(
                f'    -> checkpoint_stats={len(filter_checkpoint_rows)}, '
                f'main_variants={len(main_variants)}, '
                f'order_families={len(materialized_order_families)}, '
                f'order_variants={len(order_variants)}'
            )
            all_variants = [*main_variants, *order_variants]
            mapper_sequence = ' -> '.join(variant['name'] for variant in ordered_mappers)

            for checkpoint_row in filter_checkpoint_rows:
                checkpoint_rows.append(
                    {
                        'domain': domain,
                        'recipe_id': recipe_id,
                        'mapper_sequence': mapper_sequence,
                        **checkpoint_row,
                    }
                )

            selected_attachments = [
                *[(attachment, 'filter-then-clean', 'main') for attachment in raw_attachments],
                *[(attachment, 'clean-then-filter', 'main') for attachment in final_attachments],
            ]
            for attachment, recipe_type, benchmark_track in selected_attachments:
                attachment_row = {
                    'domain': domain,
                    'recipe_id': recipe_id,
                    'mapper_sequence': mapper_sequence,
                    'recipe_type': recipe_type,
                    'benchmark_track': benchmark_track,
                    **{k: v for k, v in attachment.items() if k != 'selection_score'},
                }
                if benchmark_track == 'order_sensitivity':
                    continue
                else:
                    attachment_rows.append(attachment_row)

            for variant in all_variants:
                variant_rows.append(
                    {
                        'domain': domain,
                        'recipe_id': recipe_id,
                        'recipe_variant_id': _first_present(variant, 'recipe_variant_id', 'workflow_variant_id'),
                        'recipe_type': _first_present(variant, 'recipe_type', 'workflow_type'),
                        'benchmark_track': variant['benchmark_track'],
                        'order_family_id': variant.get('order_family_id'),
                        'order_slot': variant.get('order_slot'),
                        'operator_sequence': ' -> '.join(variant['operator_sequence']),
                        'length': len(variant['operator_sequence']),
                        'filter_name': variant.get('filter_name'),
                        'filter_checkpoint_id': variant.get('filter_checkpoint_id'),
                        'filter_step_index': variant.get('filter_step_index'),
                        'filter_params': variant.get('filter_params'),
                    }
                )

            for family in materialized_order_families:
                variants_by_slot = {variant['order_slot']: variant for variant in family['variants']}
                order_family_rows.append(
                    {
                        'domain': domain,
                        'recipe_id': recipe_id,
                        'order_family_id': family['order_family_id'],
                        'filter_name': family['filter_name'],
                        'min_support_records': family['min_support_records'],
                        'required_slots_for_group_success': 'front | middle | end',
                        'group_success_rule': family['group_success_rule'],
                        'front_recipe_variant_id': _first_present(variants_by_slot['front'], 'recipe_variant_id', 'workflow_variant_id'),
                        'middle_recipe_variant_id': _first_present(variants_by_slot['middle'], 'recipe_variant_id', 'workflow_variant_id'),
                        'end_recipe_variant_id': _first_present(variants_by_slot['end'], 'recipe_variant_id', 'workflow_variant_id'),
                    }
                )
                for variant in family['variants']:
                    order_candidate_rows.append(
                        {
                            'domain': domain,
                            'recipe_id': recipe_id,
                            'order_family_id': family['order_family_id'],
                            'recipe_variant_id': _first_present(variant, 'recipe_variant_id', 'workflow_variant_id'),
                            'order_slot': variant['order_slot'],
                            'recipe_type': _first_present(variant, 'recipe_type', 'workflow_type'),
                            'operator_sequence': ' -> '.join(variant['operator_sequence']),
                            'length': len(variant['operator_sequence']),
                            'mapper_sequence': mapper_sequence,
                            'filter_name': variant['filter_name'],
                            'filter_checkpoint_id': variant['filter_checkpoint_id'],
                            'filter_step_index': variant['filter_step_index'],
                            'filter_params': variant['filter_params'],
                            'threshold_selection_rule': variant['threshold_selection_rule'],
                        }
                    )

            domain_yaml['recipes'].append(
                {
                    'recipe_id': recipe_id,
                    'family_id': _clean_optional_id(row.get('family_id')) or f'{domain}_auto_family',
                    'selection_source': row.get('selection_source', 'bottom_up_exact_signature'),
                    'support': int(row.get('support', 0) or 0),
                    'support_ratio': float(row.get('support_ratio', 0.0) or 0.0),
                    'mapper_operator_set': operator_set,
                    'ordered_clean_sequence': [variant['name'] for variant in ordered_mappers],
                    'support_records_used_for_filter_scan': len(support_records),
                    'main_recipe_variants': main_variants,
                    'order_sensitivity_recipe_variants': order_variants,
                    'order_sensitivity_families': materialized_order_families,
                    'selected_filter_attachments': {
                        'filter_then_clean': raw_attachments,
                        'clean_then_filter': final_attachments,
                    },
                    'checkpoint_filter_stats_file': 'checkpoint_filter_stats.csv',
                    'curation_status': 'draft_recipe_ready_for_threshold_and_prompt_curation',
                }
            )

            summary_rows.append(
                {
                    'domain': domain,
                    'recipe_id': recipe_id,
                    'support': int(row.get('support', 0) or 0),
                    'mapper_length': len(ordered_mappers),
                    'num_main_variants': len(main_variants),
                    'num_order_sensitivity_variants': len(order_variants),
                    'num_order_sensitivity_families': len(materialized_order_families),
                    'num_filter_then_clean': len(raw_attachments),
                    'num_clean_then_filter': len(final_attachments),
                }
            )

        global_yaml['domains'][domain] = domain_yaml
        domain_out_dir.mkdir(parents=True, exist_ok=True)
        with (domain_out_dir / 'recipe_library.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(domain_yaml, f, sort_keys=False, allow_unicode=True)
        pd.DataFrame(variant_rows).to_csv(domain_out_dir / 'recipe_variants.csv', index=False)
        pd.DataFrame(attachment_rows).to_csv(domain_out_dir / 'filter_attachments.csv', index=False)
        pd.DataFrame(checkpoint_rows).to_csv(domain_out_dir / 'checkpoint_filter_stats.csv', index=False)
        pd.DataFrame(order_candidate_rows).to_csv(domain_out_dir / 'order_sensitivity_candidates.csv', index=False)
        pd.DataFrame(order_family_rows).to_csv(domain_out_dir / 'order_sensitivity_families.csv', index=False)
        _log(f'[{domain_index}/{len(domain_dirs)}] {domain}: wrote outputs -> {domain_out_dir}')

    with (output_dir / 'recipe_library.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(global_yaml, f, sort_keys=False, allow_unicode=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / 'recipe_library_summary.csv', index=False)

    print(f'wrote recipe library -> {output_dir}')


if __name__ == '__main__':
    main()
