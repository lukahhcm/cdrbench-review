#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    import editdistance as _editdistance_lib
except ModuleNotFoundError:
    _editdistance_lib = None

from cdrbench.config import load_domains_config
from cdrbench.domain_assignment import build_domain_execution_plan
from cdrbench.prepare_data.build_benchmark_release import (
    DOMAIN_METADATA,
    RELEASE_FIELD_ORDER,
    _recipe_length_label,
    _task_difficulty_from_grid,
)
from cdrbench.prepare_data.materialize_benchmark_instances import (
    _base_params,
    _load_domain_recipes,
    _operator_lookup,
    _op_kind,
    _recipe_prompt_key,
)
from cdrbench.prepare_data.materialize_domain_recipes import (
    _apply_mapper_text,
    _infer_suffix,
    _labeling_meta,
    iter_jsonl,
)


ROOT = Path(__file__).resolve().parents[3]

MAPPER_ORDER_TRACK = 'mapper_order_sensitivity'
MAPPER_ORDER_TRACK_LABEL = 'Mapper-Order-Sensitive Recipe'
MAPPER_ORDER_SETTING = 'mapper_order_sensitive'
ENGLISH_LANGUAGE_VALUES = {'en', 'eng', 'english'}
CHINESE_LANGUAGE_VALUES = {'zh', 'zho', 'chi', 'chinese', 'zh-cn', 'zh-tw', 'cn'}
CJK_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')


def _log(message: str) -> None:
    print(message, flush=True)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = '||'.join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


def _record_id(record: dict[str, Any]) -> str:
    for key in ('id', 'source_name', 'url'):
        value = record.get(key)
        if value:
            return str(value)
    return _stable_id(record.get('text', ''), length=20)


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    tmp_path.replace(path)


def _load_mapper_intents(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        payload = yaml.safe_load(f) or {}
    operators = payload.get('operators') if isinstance(payload, dict) else {}
    if not isinstance(operators, dict):
        return {}
    intents = {}
    for op_name, op_cfg in operators.items():
        if isinstance(op_cfg, dict):
            intent = str(op_cfg.get('natural_language_intent') or '').strip()
            if intent:
                intents[str(op_name)] = intent
    return intents


def _preview(text: str, limit: int) -> str:
    if limit <= 0:
        return ''
    return text[:limit] if len(text) <= limit else text[:limit] + '...'


def _sequence_label(sequence: list[str]) -> str:
    return ' -> '.join(sequence)


def _first_diff_window(left: str, right: str, context_chars: int) -> dict[str, Any]:
    if left == right:
        return {
            'first_diff_index': None,
            'canonical_diff_window': '',
            'swapped_diff_window': '',
        }

    first = 0
    limit = min(len(left), len(right))
    while first < limit and left[first] == right[first]:
        first += 1

    left_end = len(left)
    right_end = len(right)
    while left_end > first and right_end > first and left[left_end - 1] == right[right_end - 1]:
        left_end -= 1
        right_end -= 1

    start = max(first - context_chars, 0)
    left_stop = min(left_end + context_chars, len(left))
    right_stop = min(right_end + context_chars, len(right))
    return {
        'first_diff_index': first,
        'canonical_diff_window': left[start:left_stop],
        'swapped_diff_window': right[start:right_stop],
    }


def _unified_diff_preview(left: str, right: str, max_lines: int, max_chars: int) -> str:
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            left_lines,
            right_lines,
            fromfile='canonical_before_swap',
            tofile='swapped_after_swap',
            lineterm='',
            n=3,
        )
    )
    if max_lines > 0 and len(diff_lines) > max_lines:
        diff_lines = [*diff_lines[:max_lines], f'... diff truncated after {max_lines} lines ...']
    diff_text = '\n'.join(diff_lines)
    if max_chars > 0 and len(diff_text) > max_chars:
        diff_text = diff_text[:max_chars] + f'\n... diff truncated after {max_chars} chars ...'
    return diff_text


def _input_length_bucket(input_length_chars: int) -> str:
    if input_length_chars <= 4_000:
        return 'short'
    if input_length_chars <= 8_000:
        return 'medium'
    return 'long'


def _ordered_release_like_row(row: dict[str, Any]) -> dict[str, Any]:
    leading_keys = ['instance_id', 'mapper_order_inspection']
    ordered = {key: row[key] for key in leading_keys if key in row}
    for key in RELEASE_FIELD_ORDER:
        if key in row and key not in ordered:
            ordered[key] = row[key]
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if _editdistance_lib is not None:
        return int(_editdistance_lib.eval(left, right))
    if len(left) * len(right) > 10_000_000:
        paired_delta = sum(1 for left_ch, right_ch in zip(left, right) if left_ch != right_ch)
        return paired_delta + abs(len(left) - len(right))
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_idx, left_ch in enumerate(left, start=1):
        current = [left_idx]
        for right_idx, right_ch in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_idx] + 1,
                    current[right_idx - 1] + 1,
                    previous[right_idx - 1] + (left_ch != right_ch),
                )
            )
        previous = current
    return previous[-1]


def _normalize_text_for_match(value: Any) -> str:
    text = '' if value is None else str(value)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n(?:[ \t]*\n)+', '\n\n', text)
    return text.strip()


def _normalize_text_for_norm_match(value: Any) -> str:
    text = '' if value is None else str(value)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +([,.;:!?])', r'\1', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n(?:[ \t]*\n)+', '\n\n', text)
    return text.strip()


def _language_value(record: dict[str, Any]) -> str:
    candidates = [
        record.get('language'),
        record.get('lang'),
    ]
    meta = record.get('meta')
    if isinstance(meta, dict):
        candidates.extend(
            [
                meta.get('language'),
                meta.get('lang'),
                meta.get('language_id'),
            ]
        )
    for value in candidates:
        if value is not None:
            text = str(value).strip().lower().replace('_', '-')
            if text:
                return text
    return ''


def _is_english_record(record: dict[str, Any]) -> bool:
    language = _language_value(record)
    if language:
        if language in ENGLISH_LANGUAGE_VALUES or language.startswith('en-'):
            return True
        if language in CHINESE_LANGUAGE_VALUES or language.startswith('zh'):
            return False
        return False

    text = str(record.get('text', ''))
    if not text.strip():
        return False
    cjk_count = len(CJK_RE.findall(text))
    if cjk_count >= 3 or cjk_count / max(len(text), 1) > 0.005:
        return False
    latin_count = sum(1 for ch in text if ('a' <= ch.lower() <= 'z'))
    alpha_count = sum(1 for ch in text if ch.isalpha())
    return latin_count >= 20 and latin_count / max(alpha_count, 1) >= 0.8


def _passes_language_filter(record: dict[str, Any], language_filter: str) -> bool:
    if language_filter == 'any':
        return True
    if language_filter == 'english':
        return _is_english_record(record)
    if language_filter == 'non-chinese':
        text = str(record.get('text', ''))
        language = _language_value(record)
        if language in CHINESE_LANGUAGE_VALUES or language.startswith('zh'):
            return False
        cjk_count = len(CJK_RE.findall(text))
        return cjk_count < 3 and cjk_count / max(len(text), 1) <= 0.005
    raise ValueError(f'unknown language filter: {language_filter}')


def _load_records_by_domain(
    filtered_path: Path,
    max_input_chars: int,
    language_filter: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = {
        'language_filter': language_filter,
        'input_records': 0,
        'kept_records': 0,
        'skipped_too_long': 0,
        'skipped_language': 0,
    }
    for record in iter_jsonl(filtered_path):
        stats['input_records'] += 1
        domain = record.get('domain')
        if not domain:
            continue
        if max_input_chars > 0 and len(str(record.get('text', ''))) > max_input_chars:
            stats['skipped_too_long'] += 1
            continue
        if not _passes_language_filter(record, language_filter):
            stats['skipped_language'] += 1
            continue
        records_by_domain[str(domain)].append(record)
        stats['kept_records'] += 1
    return records_by_domain, stats


def _supporting_records(
    records: list[dict[str, Any]],
    mapper_names: list[str],
    max_records: int,
    salt: str,
) -> list[dict[str, Any]]:
    wanted = set(mapper_names)
    supported = []
    for record in records:
        active = set(_labeling_meta(record).get('active_mapper_names', []))
        if wanted.issubset(active):
            supported.append(record)
    supported.sort(key=lambda row: _stable_id(salt, _record_id(row), row.get('source_name'), row.get('url')))
    return supported[:max_records] if max_records > 0 else supported


def _clean_only_sequences(recipe: dict[str, Any]) -> list[tuple[str, list[str]]]:
    sequences: list[tuple[str, list[str]]] = []
    variants = list(recipe.get('main_recipe_variants') or recipe.get('main_workflow_variants') or [])
    for variant in variants:
        recipe_type = str(_first_present(variant, 'recipe_type', 'workflow_type') or '')
        if recipe_type != 'clean-only':
            continue
        if variant.get('filter_name'):
            continue
        sequence = [str(op_name) for op_name in list(variant.get('operator_sequence') or []) if op_name]
        variant_id = str(_first_present(variant, 'recipe_variant_id', 'workflow_variant_id') or 'clean_only')
        if len(sequence) >= 2:
            sequences.append((variant_id, sequence))

    if not sequences:
        sequence = [str(op_name) for op_name in list(recipe.get('ordered_clean_sequence') or []) if op_name]
        if len(sequence) >= 2:
            recipe_id = str(_first_present(recipe, 'recipe_id', 'workflow_id') or 'recipe')
            sequences.append((f'{recipe_id}__ordered_clean_sequence', sequence))
    return sequences


def _swapped_sequences(sequence: list[str], mode: str) -> Iterable[tuple[str, int, int, list[str]]]:
    if mode == 'adjacent':
        pairs = [(idx, idx + 1) for idx in range(len(sequence) - 1)]
    elif mode == 'all-pairs':
        pairs = [(left, right) for left in range(len(sequence)) for right in range(left + 1, len(sequence))]
    else:
        raise ValueError(f'unknown swap mode: {mode}')

    for left, right in pairs:
        swapped = list(sequence)
        swapped[left], swapped[right] = swapped[right], swapped[left]
        yield f'swap_{left}_{right}', left, right, swapped


def _execute_mapper_sequence(
    record: dict[str, Any],
    sequence: list[str],
    operators_by_name: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    suffix = _infer_suffix(record)
    text = str(record.get('text', ''))
    trace = []
    for step_index, op_name in enumerate(sequence):
        if _op_kind(op_name, operators_by_name) != 'mapper':
            raise ValueError(f'clean-only sequence contains non-mapper operator: {op_name}')
        before_len = len(text)
        text, result = _apply_mapper_text(op_name, text, _base_params(op_name, operators_by_name), suffix)
        trace.append(
            {
                'step_index': step_index,
                'operator': op_name,
                'input_length': before_len,
                **result,
            }
        )
    return text, trace


def _operation_instruction(op_name: str, mapper_intents: dict[str, str]) -> str:
    intent = mapper_intents.get(op_name)
    return intent if intent else f'Apply `{op_name}`.'


def _build_user_requirement(sequence: list[str], mapper_intents: dict[str, str]) -> str:
    lines = [
        'Clean the input text by applying the following operations in exactly this order:',
    ]
    for index, op_name in enumerate(sequence, start=1):
        lines.append(f'{index}. {_operation_instruction(op_name, mapper_intents)}')
    lines.extend(
        [
            '',
            'Only apply the operations listed above, and preserve all remaining text exactly after the final step.',
            'Return KEEP with the final cleaned text.',
        ]
    )
    return '\n'.join(lines)


def _release_like_example_row(
    *,
    domain: str,
    recipe_id: str,
    variant_id: str,
    record: dict[str, Any],
    sequence: list[str],
    canonical_text: str,
    canonical_trace: list[dict[str, Any]],
    swap_kind: str,
    left_idx: int,
    right_idx: int,
    swapped_sequence: list[str],
    swapped_text: str,
    swapped_trace: list[dict[str, Any]],
    delta: dict[str, Any],
    preview_chars: int,
    include_full_diagnostics: bool,
    mapper_intents: dict[str, str],
) -> dict[str, Any]:
    source_record_id = _record_id(record)
    order_family_id = f'{recipe_id}__mapper_order_family__{swap_kind}'
    order_group_instance_id = _stable_id(order_family_id, source_record_id)
    instance_id = _stable_id(order_family_id, source_record_id, 'swapped')
    input_text = str(record.get('text', ''))
    input_bucket = _input_length_bucket(len(input_text))
    recipe_length = len(swapped_sequence)
    recipe_label = _recipe_length_label(recipe_length)
    difficulty_label = _task_difficulty_from_grid(recipe_label, input_bucket)
    domain_meta = DOMAIN_METADATA.get(domain, {'domain_label': domain.replace('_', ' ').title(), 'domain_abbr': domain.upper()})
    left_mapper = sequence[left_idx]
    right_mapper = sequence[right_idx]
    canonical_sequence_text = _sequence_label(sequence)
    swapped_sequence_text = _sequence_label(swapped_sequence)
    diff_window = _first_diff_window(canonical_text, swapped_text, max(80, preview_chars // 4))
    edit_distance_chars = delta['edit_distance']
    user_requirement = _build_user_requirement(swapped_sequence, mapper_intents)
    prompt_variants = [
        {
            'style_id': 'direct_ordered_mapper_steps',
            'style_label': 'Direct Ordered Mapper Steps',
            'user_requirement': user_requirement,
        }
    ]
    inspection = {
        'change_summary': (
            f'swapped mapper positions {left_idx} and {right_idx}: '
            f'{left_mapper} <-> {right_mapper}; '
            f'final_output_edit_distance_chars={edit_distance_chars}'
        ),
        'swapped_mapper_left_index': left_idx,
        'swapped_mapper_right_index': right_idx,
        'swapped_mapper_left_name': left_mapper,
        'swapped_mapper_right_name': right_mapper,
        'canonical_operator_sequence_text': canonical_sequence_text,
        'swapped_operator_sequence_text': swapped_sequence_text,
        'canonical_output_preview': _preview(canonical_text, preview_chars),
        'swapped_output_preview': _preview(swapped_text, preview_chars),
        'canonical_output_length_chars': len(canonical_text),
        'swapped_output_length_chars': len(swapped_text),
        'first_diff_index': diff_window['first_diff_index'],
        'canonical_diff_window': diff_window['canonical_diff_window'],
        'swapped_diff_window': diff_window['swapped_diff_window'],
        'unified_diff_preview': _unified_diff_preview(
            canonical_text,
            swapped_text,
            max_lines=80,
            max_chars=max(2_000, preview_chars * 4),
        ),
        'raw_text_changed': delta['raw_text_changed'],
        'normalized_text_changed': delta['normalized_text_changed'],
        'norm_text_changed': delta['norm_text_changed'],
        'edit_distance_chars': edit_distance_chars,
        'edit_distance': edit_distance_chars,
        'relative_edit_distance': delta['relative_edit_distance'],
    }

    row = {
        'instance_id': instance_id,
        'mapper_order_inspection': inspection,
        'benchmark_track': MAPPER_ORDER_TRACK,
        'benchmark_track_label': MAPPER_ORDER_TRACK_LABEL,
        'recipe_order_setting': MAPPER_ORDER_SETTING,
        'domain': domain,
        **domain_meta,
        'source_domain': domain,
        'source_record_id': source_record_id,
        'input_text': input_text,
        'input_length_chars': len(input_text),
        'input_length_bucket': input_bucket,
        'difficulty_score': {'easy': 1, 'medium': 2, 'hard': 3}[difficulty_label],
        'difficulty_label': difficulty_label,
        'difficulty_grid_cell': f'recipe_{recipe_label}__input_{input_bucket}',
        'recipe_length': recipe_length,
        'recipe_length_label': recipe_label,
        'operator': None,
        'operator_kind': 'mapper',
        'operator_sequence': swapped_sequence,
        'filter_name': None,
        'filter_params_by_name': {},
        'recipe_id': recipe_id,
        'recipe_variant_id': f'{variant_id}__{swap_kind}',
        'recipe_type': 'clean-only',
        'order_family_id': order_family_id,
        'order_slot': swap_kind,
        'order_group_instance_id': order_group_instance_id,
        'group_success_rule': 'canonical_and_swapped_outputs_correct',
        'reference_status': 'KEEP',
        'reference_text': swapped_text,
        'reference_text_at_stop': swapped_text,
        'reference_text_full_run': swapped_text,
        'reference_trace': swapped_trace,
        'prompt_source': None,
        'prompt_variants': prompt_variants,
        'prompt_variant_count': len(prompt_variants),
        'prompt_candidate_pool_count': len(prompt_variants),
        'prompt_sampling_policy': None,
        'prompt_sampling_seed': None,
        'accepted_candidate_count': len(prompt_variants),
        'accepted_style_count': len(prompt_variants),
        'threshold_meta': {},
        'user_requirement': user_requirement,
        'change_summary': inspection['change_summary'],
        'swapped_mapper_left_index': inspection['swapped_mapper_left_index'],
        'swapped_mapper_right_index': inspection['swapped_mapper_right_index'],
        'swapped_mapper_left_name': inspection['swapped_mapper_left_name'],
        'swapped_mapper_right_name': inspection['swapped_mapper_right_name'],
        'canonical_operator_sequence_text': inspection['canonical_operator_sequence_text'],
        'swapped_operator_sequence_text': inspection['swapped_operator_sequence_text'],
        'canonical_output_preview': inspection['canonical_output_preview'],
        'swapped_output_preview': inspection['swapped_output_preview'],
        'canonical_output_length_chars': inspection['canonical_output_length_chars'],
        'swapped_output_length_chars': inspection['swapped_output_length_chars'],
        'first_diff_index': inspection['first_diff_index'],
        'canonical_diff_window': inspection['canonical_diff_window'],
        'swapped_diff_window': inspection['swapped_diff_window'],
        'unified_diff_preview': inspection['unified_diff_preview'],
        'raw_text_changed': inspection['raw_text_changed'],
        'normalized_text_changed': inspection['normalized_text_changed'],
        'norm_text_changed': inspection['norm_text_changed'],
        'edit_distance_chars': inspection['edit_distance_chars'],
        'edit_distance': inspection['edit_distance'],
        'relative_edit_distance': inspection['relative_edit_distance'],
        'mapper_order_change': {
            'swap_kind': swap_kind,
            'left_index': left_idx,
            'right_index': right_idx,
            'left_mapper': left_mapper,
            'right_mapper': right_mapper,
            'canonical_operator_sequence_text': canonical_sequence_text,
            'swapped_operator_sequence_text': swapped_sequence_text,
            **delta,
        },
        'canonical_recipe_variant_id': variant_id,
        'canonical_operator_sequence': sequence,
        'canonical_reference_status': 'KEEP',
        'canonical_reference_text': canonical_text,
        'canonical_reference_trace': canonical_trace,
    }
    row['recipe_prompt_key'] = _recipe_prompt_key(row)

    if include_full_diagnostics:
        row['canonical_output_text'] = canonical_text
        row['swapped_output_text'] = swapped_text

    return _ordered_release_like_row(row)


def _changed_enough(
    canonical_text: str,
    swapped_text: str,
    *,
    compare: str,
    min_relative_delta: float,
    min_edit_distance: int,
) -> tuple[bool, dict[str, Any]]:
    normalized_text_changed = _normalize_text_for_match(canonical_text) != _normalize_text_for_match(swapped_text)
    norm_text_changed = _normalize_text_for_norm_match(canonical_text) != _normalize_text_for_norm_match(swapped_text)
    raw_text_changed = canonical_text != swapped_text
    distance = _edit_distance(canonical_text, swapped_text)
    denominator = max(len(canonical_text), len(swapped_text), 1)
    relative_delta = distance / denominator

    if compare == 'raw':
        changed = raw_text_changed
    elif compare == 'normalized':
        changed = normalized_text_changed
    elif compare == 'norm':
        changed = norm_text_changed
    else:
        raise ValueError(f'unknown compare mode: {compare}')

    changed = changed and distance >= min_edit_distance and relative_delta >= min_relative_delta
    return changed, {
        'raw_text_changed': raw_text_changed,
        'normalized_text_changed': normalized_text_changed,
        'norm_text_changed': norm_text_changed,
        'edit_distance': distance,
        'relative_edit_distance': round(relative_delta, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Scan clean-only mapper recipes for examples where swapping mapper order changes the deterministic GT.'
    )
    parser.add_argument('--domains-config', default='configs/domains.yaml')
    parser.add_argument('--recipe-library-dir', default='data/processed/recipe_library')
    parser.add_argument('--filtered-path', default='data/processed/domain_filtered/all.jsonl')
    parser.add_argument('--recipe-prompting-config', default='configs/recipe_prompting.yaml')
    parser.add_argument('--output-jsonl', default='data/processed/mapper_order_sensitivity/mapper_order_sensitivity.jsonl')
    parser.add_argument('--summary-json', default='data/processed/mapper_order_sensitivity/mapper_order_sensitivity_summary.json')
    parser.add_argument('--domains', nargs='*', default=None, help='Optional domain allowlist.')
    parser.add_argument('--swap-mode', choices=('adjacent', 'all-pairs'), default='adjacent')
    parser.add_argument('--compare', choices=('raw', 'normalized', 'norm'), default='norm')
    parser.add_argument('--max-records-per-recipe', type=int, default=128, help='0 means all supporting records.')
    parser.add_argument('--max-examples', type=int, default=200, help='0 means no global example cap.')
    parser.add_argument('--max-examples-per-recipe', type=int, default=5, help='0 means no per-recipe cap.')
    parser.add_argument('--max-input-chars', type=int, default=50_000, help='0 disables the input length cap.')
    parser.add_argument(
        '--language-filter',
        choices=('english', 'non-chinese', 'any'),
        default='english',
        help='Filter candidate records before scanning swaps. Default keeps English records only.',
    )
    parser.add_argument('--min-relative-delta', type=float, default=0.0)
    parser.add_argument('--min-edit-distance', type=int, default=1)
    parser.add_argument('--preview-chars', type=int, default=600)
    parser.add_argument('--include-full-text', action='store_true')
    args = parser.parse_args()

    root = ROOT
    recipe_library_dir = (root / args.recipe_library_dir).resolve()
    filtered_path = (root / args.filtered_path).resolve()
    output_jsonl = (root / args.output_jsonl).resolve()
    summary_json = (root / args.summary_json).resolve()
    recipe_prompting_config = (root / args.recipe_prompting_config).resolve()

    if not recipe_library_dir.exists():
        raise SystemExit(f'recipe library dir not found: {recipe_library_dir}')
    if not filtered_path.exists():
        raise SystemExit(f'filtered corpus not found: {filtered_path}')
    if args.max_records_per_recipe < 0 or args.max_examples < 0 or args.max_examples_per_recipe < 0:
        raise SystemExit('max counts must be >= 0; use 0 for no cap')
    if args.max_input_chars < 0:
        raise SystemExit('--max-input-chars must be >= 0')
    if args.min_relative_delta < 0:
        raise SystemExit('--min-relative-delta must be >= 0')
    if args.min_edit_distance < 0:
        raise SystemExit('--min-edit-distance must be >= 0')

    domains_cfg = load_domains_config(root / args.domains_config)
    plan = build_domain_execution_plan(domains_cfg)
    operators_by_name = _operator_lookup(plan)
    mapper_intents = _load_mapper_intents(recipe_prompting_config)

    _log(f'loading filtered records -> {filtered_path}')
    records_by_domain, language_stats = _load_records_by_domain(
        filtered_path,
        args.max_input_chars,
        args.language_filter,
    )
    _log(
        'loaded records by domain: '
        + ', '.join(f'{domain}={len(records)}' for domain, records in sorted(records_by_domain.items()))
    )
    _log(
        f"language filter={args.language_filter}: kept={language_stats['kept_records']} "
        f"skipped_language={language_stats['skipped_language']} "
        f"skipped_too_long={language_stats['skipped_too_long']}"
    )

    domain_recipes = _load_domain_recipes(recipe_library_dir)
    if args.domains:
        allowed = set(args.domains)
        domain_recipes = {domain: payload for domain, payload in domain_recipes.items() if domain in allowed}
    _log(f'loaded recipe libraries for {len(domain_recipes)} domains -> {recipe_library_dir}')

    examples: list[dict[str, Any]] = []
    summary = {
        'recipe_library_dir': str(recipe_library_dir),
        'filtered_path': str(filtered_path),
        'recipe_prompting_config': str(recipe_prompting_config),
        'mapper_intent_count': len(mapper_intents),
        'swap_mode': args.swap_mode,
        'compare': args.compare,
        'language_filter': args.language_filter,
        'language_filter_stats': language_stats,
        'max_records_per_recipe': args.max_records_per_recipe,
        'max_input_chars': args.max_input_chars,
        'min_relative_delta': args.min_relative_delta,
        'min_edit_distance': args.min_edit_distance,
        'domains_seen': len(domain_recipes),
        'recipes_seen': 0,
        'clean_only_recipes_seen': 0,
        'swap_variants_tested': 0,
        'record_swap_evaluations': 0,
        'changed_record_swaps': 0,
        'examples_written': 0,
        'errors': 0,
        'by_domain': {},
    }

    stop = False
    for domain, domain_yaml in sorted(domain_recipes.items()):
        domain_summary = {
            'recipes_seen': 0,
            'clean_only_recipes_seen': 0,
            'swap_variants_tested': 0,
            'record_swap_evaluations': 0,
            'changed_record_swaps': 0,
            'examples_written': 0,
            'errors': 0,
        }
        records = records_by_domain.get(domain, [])
        recipes = list(domain_yaml.get('recipes') or domain_yaml.get('workflows') or [])
        _log(f'[{domain}] scanning {len(recipes)} recipes over {len(records)} records')
        for recipe in recipes:
            summary['recipes_seen'] += 1
            domain_summary['recipes_seen'] += 1
            recipe_id = str(_first_present(recipe, 'recipe_id', 'workflow_id') or '')
            clean_sequences = _clean_only_sequences(recipe)
            if not clean_sequences:
                continue
            summary['clean_only_recipes_seen'] += 1
            domain_summary['clean_only_recipes_seen'] += 1

            recipe_examples = 0
            for variant_id, sequence in clean_sequences:
                candidates = _supporting_records(
                    records,
                    sequence,
                    args.max_records_per_recipe,
                    f'{domain}:{recipe_id}:{variant_id}',
                )
                swaps = list(_swapped_sequences(sequence, args.swap_mode))
                summary['swap_variants_tested'] += len(swaps)
                domain_summary['swap_variants_tested'] += len(swaps)
                for record in candidates:
                    try:
                        canonical_text, canonical_trace = _execute_mapper_sequence(record, sequence, operators_by_name)
                    except Exception as exc:
                        summary['errors'] += 1
                        domain_summary['errors'] += 1
                        _log(f'  error canonical {domain}/{recipe_id}/{_record_id(record)}: {type(exc).__name__}: {exc}')
                        continue
                    for swap_kind, left_idx, right_idx, swapped_sequence in swaps:
                        summary['record_swap_evaluations'] += 1
                        domain_summary['record_swap_evaluations'] += 1
                        try:
                            swapped_text, swapped_trace = _execute_mapper_sequence(record, swapped_sequence, operators_by_name)
                        except Exception as exc:
                            summary['errors'] += 1
                            domain_summary['errors'] += 1
                            _log(
                                f'  error swapped {domain}/{recipe_id}/{swap_kind}/{_record_id(record)}: '
                                f'{type(exc).__name__}: {exc}'
                            )
                            continue
                        changed, delta = _changed_enough(
                            canonical_text,
                            swapped_text,
                            compare=args.compare,
                            min_relative_delta=args.min_relative_delta,
                            min_edit_distance=args.min_edit_distance,
                        )
                        if not changed:
                            continue

                        summary['changed_record_swaps'] += 1
                        domain_summary['changed_record_swaps'] += 1
                        example = _release_like_example_row(
                            domain=domain,
                            recipe_id=recipe_id,
                            variant_id=variant_id,
                            record=record,
                            sequence=sequence,
                            canonical_text=canonical_text,
                            canonical_trace=canonical_trace,
                            swap_kind=swap_kind,
                            left_idx=left_idx,
                            right_idx=right_idx,
                            swapped_sequence=swapped_sequence,
                            swapped_text=swapped_text,
                            swapped_trace=swapped_trace,
                            delta=delta,
                            preview_chars=args.preview_chars,
                            include_full_diagnostics=args.include_full_text,
                            mapper_intents=mapper_intents,
                        )
                        examples.append(example)
                        recipe_examples += 1
                        summary['examples_written'] += 1
                        domain_summary['examples_written'] += 1

                        if args.max_examples_per_recipe > 0 and recipe_examples >= args.max_examples_per_recipe:
                            break
                        if args.max_examples > 0 and len(examples) >= args.max_examples:
                            stop = True
                            break
                    if stop or (args.max_examples_per_recipe > 0 and recipe_examples >= args.max_examples_per_recipe):
                        break
                if stop or (args.max_examples_per_recipe > 0 and recipe_examples >= args.max_examples_per_recipe):
                    break
            if stop:
                break

        summary['by_domain'][domain] = domain_summary
        _log(
            f"[{domain}] changed={domain_summary['changed_record_swaps']} "
            f"examples={domain_summary['examples_written']} evals={domain_summary['record_swap_evaluations']}"
        )
        if stop:
            break

    written = _write_jsonl(output_jsonl, examples)
    summary['examples_written'] = written
    _write_json(summary_json, summary)
    _log(f'wrote examples: {written} -> {output_jsonl}')
    _log(f'wrote summary -> {summary_json}')


if __name__ == '__main__':
    main()
