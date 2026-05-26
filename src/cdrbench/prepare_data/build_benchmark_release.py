#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]

TRACK_FILENAMES = {
    'atomic_ops': 'atomic_ops.jsonl',
    'main': 'main.jsonl',
    'order_sensitivity': 'order_sensitivity.jsonl',
    'semantic_agent_atomic': 'semantic_agent_atomic.jsonl',
    'semantic_agent_compositional': 'semantic_agent_compositional.jsonl',
}

DOMAIN_METADATA = {
    'web': {'domain_label': 'Web Refinement', 'domain_abbr': 'WR'},
    'arxiv': {'domain_label': 'LaTeX Refinement', 'domain_abbr': 'LR'},
    'knowledge_base': {'domain_label': 'RAG Preparation', 'domain_abbr': 'RP'},
    'pii': {'domain_label': 'Privacy Redaction', 'domain_abbr': 'PR'},
    'world_knowledge': {'domain_label': 'World Knowledge Agent Tasks', 'domain_abbr': 'WK'},
    'tool_use': {'domain_label': 'Tool-Use Agent Tasks', 'domain_abbr': 'TU'},
    'science': {'domain_label': 'Science Agent Tasks', 'domain_abbr': 'SCI'},
    'math': {'domain_label': 'Math Agent Tasks', 'domain_abbr': 'MATH'},
    'general_assistant': {'domain_label': 'General Assistant Agent Tasks', 'domain_abbr': 'GA'},
}

RELEASE_FIELD_ORDER = [
    'instance_id',
    'benchmark_track',
    'benchmark_track_label',
    'recipe_order_setting',
    'domain',
    'domain_label',
    'domain_abbr',
    'source_domain',
    'source_record_id',
    'input_text',
    'input_length_chars',
    'input_length_bucket',
    'difficulty_score',
    'difficulty_label',
    'difficulty_grid_cell',
    'recipe_length',
    'recipe_length_label',
    'operator',
    'operator_kind',
    'operator_sequence',
    'filter_name',
    'filter_params_by_name',
    'recipe_id',
    'recipe_variant_id',
    'recipe_type',
    'order_family_id',
    'order_slot',
    'order_group_instance_id',
    'group_success_rule',
    'reference_status',
    'reference_text',
    'reference_text_at_stop',
    'reference_text_full_run',
    'reference_trace',
    'recipe_prompt_key',
    'prompt_source',
    'prompt_variants',
    'prompt_variant_count',
    'prompt_candidate_pool_count',
    'prompt_sampling_policy',
    'prompt_sampling_seed',
    'accepted_candidate_count',
    'accepted_style_count',
    'threshold_meta',
]

TRACK_LABELS = {
    'atomic_ops': 'Atomic OPs',
    'main': 'Order-Agnostic Recipe',
    'order_sensitivity': 'Order-Sensitive Recipe',
    'semantic_agent_atomic': 'Semantic Agent Atomic',
    'semantic_agent_compositional': 'Semantic Agent Compositional',
}

TRACK_ORDER_SETTINGS = {
    'atomic_ops': 'atomic',
    'main': 'order_agnostic',
    'order_sensitivity': 'order_sensitive',
    'semantic_agent_atomic': 'semantic_atomic',
    'semantic_agent_compositional': 'semantic_compositional',
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f'invalid JSONL in {path} at line {line_no}: {exc}') from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    count = 0
    with tmp_path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    tmp_path.replace(path)
    return count


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    tmp_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _resolve_track_path(root: Path, track: str) -> Path:
    filename = TRACK_FILENAMES[track]
    nested = root / track / filename
    if nested.exists():
        return nested
    flat = root / filename
    if flat.exists():
        return flat
    raise SystemExit(f'missing benchmark file for track={track}: expected {nested} or {flat}')


def _canonical_domain(row: dict[str, Any]) -> str:
    domain = str(row.get('domain') or '').strip()
    if domain in DOMAIN_METADATA:
        return domain
    source_domain = str(row.get('source_domain') or '').strip()
    if source_domain in DOMAIN_METADATA:
        return source_domain
    recipe_id = str(row.get('recipe_id') or row.get('recipe_variant_id') or '').strip()
    for candidate in DOMAIN_METADATA:
        if recipe_id.startswith(candidate + '_'):
            return candidate
    return domain or source_domain or 'unknown'


def _ordered_row(row: dict[str, Any], *, keep_replay: bool) -> dict[str, Any]:
    release_row = dict(row)
    track = str(release_row.get('benchmark_track') or '').strip()
    if track in TRACK_LABELS:
        release_row['benchmark_track_label'] = TRACK_LABELS[track]
        release_row['recipe_order_setting'] = TRACK_ORDER_SETTINGS[track]
    domain = _canonical_domain(release_row)
    release_row['domain'] = domain
    if domain in DOMAIN_METADATA:
        release_row.update(DOMAIN_METADATA[domain])
    else:
        release_row.setdefault('domain_label', domain.replace('_', ' ').title())
        release_row.setdefault('domain_abbr', domain.upper())
    if not keep_replay:
        release_row.pop('recipe_replay', None)

    ordered = {key: release_row[key] for key in RELEASE_FIELD_ORDER if key in release_row}
    for key, value in release_row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _sequence(row: dict[str, Any]) -> list[str]:
    sequence = row.get('operator_sequence')
    if isinstance(sequence, list):
        return [str(item) for item in sequence if str(item).strip()]
    operator = str(row.get('operator') or '').strip()
    return [operator] if operator else []


def _has_filter(row: dict[str, Any]) -> bool:
    filter_name = str(row.get('filter_name') or '').strip()
    if filter_name:
        return True
    operator_kind = str(row.get('operator_kind') or '').strip().lower()
    if operator_kind == 'filter':
        return True
    return any(op_name.endswith('_filter') for op_name in _sequence(row))


def _filter_release_rows_for_track(track: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if track != 'main':
        return rows, {'input_rows': len(rows), 'kept_rows': len(rows), 'dropped_rows': 0}
    kept = [row for row in rows if not _has_filter(row)]
    return kept, {
        'input_rows': len(rows),
        'kept_rows': len(kept),
        'dropped_rows': len(rows) - len(kept),
        'filter_reason': 'main release keeps mapper-only order-agnostic recipes',
    }


def _input_length_tier(row: dict[str, Any]) -> str:
    bucket = str(row.get('input_length_bucket') or '').strip().lower()
    if bucket in {'short', 'medium', 'long'}:
        return bucket
    length = int(row.get('input_length_chars', 0) or 0)
    if length <= 4_000:
        return 'short'
    if length <= 8_000:
        return 'medium'
    return 'long'


def _recipe_length_label(length: int) -> str:
    if length <= 2:
        return 'short'
    if length <= 4:
        return 'medium'
    return 'long'


def _task_difficulty_from_grid(recipe_length_label: str, input_label: str) -> str:
    recipe_rank = {'short': 1, 'medium': 2, 'long': 3}.get(recipe_length_label, 2)
    input_rank = {'short': 1, 'medium': 2, 'long': 3}.get(input_label, 2)
    grid_rank = recipe_rank + input_rank
    if grid_rank <= 3:
        return 'easy'
    if grid_rank <= 5:
        return 'medium'
    return 'hard'


def _attach_difficulty(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored_rows = []
    for row in rows:
        sequence = _sequence(row)
        recipe_length = len(sequence)
        recipe_label = _recipe_length_label(recipe_length)
        input_label = _input_length_tier(row)
        task_label = _task_difficulty_from_grid(recipe_label, input_label)
        copied = dict(row)
        copied['recipe_length'] = recipe_length
        copied['recipe_length_label'] = recipe_label
        copied['input_length_bucket'] = input_label
        copied['difficulty_grid_cell'] = f'recipe_{recipe_label}__input_{input_label}'
        copied['difficulty_label'] = task_label
        copied['difficulty_score'] = {'easy': 1, 'medium': 2, 'hard': 3}[task_label]
        scored_rows.append(copied)
    return scored_rows


def _count_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(str(row.get(field) or 'UNKNOWN') for field in fields)
        counts[key] = counts.get(key, 0) + 1
    output = []
    total = len(rows)
    for key, count in sorted(counts.items()):
        item = {field: value for field, value in zip(fields, key)}
        item['count'] = count
        item['fraction'] = count / total if total else 0.0
        output.append(item)
    return output


def _recipe_distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    by_recipe: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        recipe_key = str(row.get('recipe_variant_id') or row.get('recipe_id') or row.get('operator') or 'UNKNOWN')
        by_recipe.setdefault(recipe_key, []).append(row)
    for recipe_key, bucket in sorted(by_recipe.items()):
        first = bucket[0]
        lengths = [int(row.get('recipe_length', len(_sequence(row))) or 0) for row in bucket]
        scores = [float(row.get('difficulty_score', 0.0) or 0.0) for row in bucket]
        median_length = sorted(lengths)[len(lengths) // 2] if lengths else 0
        output.append(
            {
                'recipe_key': recipe_key,
                'benchmark_track': first.get('benchmark_track'),
                'domain': first.get('domain'),
                'recipe_id': first.get('recipe_id'),
                'recipe_variant_id': first.get('recipe_variant_id'),
                'order_family_id': first.get('order_family_id'),
                'count': len(bucket),
                'recipe_length': median_length,
                'recipe_length_label': _recipe_length_label(median_length),
                'mean_difficulty_score': sum(scores) / len(scores) if scores else 0.0,
            }
        )
    return output


def _prompt_distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        variants = row.get('prompt_variants')
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            style_id = str(variant.get('style_id') or 'UNKNOWN')
            style_label = str(variant.get('style_label') or style_id)
            instruction = str(variant.get('user_requirement') or '')
            grouped.setdefault((style_id, style_label), []).append(len(instruction))
    output = []
    for (style_id, style_label), lengths in sorted(grouped.items()):
        output.append(
            {
                'style_id': style_id,
                'style_label': style_label,
                'count': len(lengths),
                'mean_instruction_length_chars': sum(lengths) / len(lengths) if lengths else 0.0,
                'median_instruction_length_chars': statistics.median(lengths) if lengths else 0.0,
                'min_instruction_length_chars': min(lengths) if lengths else 0,
                'max_instruction_length_chars': max(lengths) if lengths else 0,
            }
        )
    return output


def _write_release_statistics(output_dir: Path, track: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats_dir = output_dir / 'stats'
    task_track_rows = _count_rows(rows, ['benchmark_track'])
    task_order_setting_rows = _count_rows(rows, ['recipe_order_setting'])
    task_difficulty_rows = _count_rows(rows, ['difficulty_label'])
    task_difficulty_grid_rows = _count_rows(rows, ['recipe_length_label', 'input_length_bucket', 'difficulty_label'])
    task_domain_rows = _count_rows(rows, ['domain'])
    task_domain_difficulty_rows = _count_rows(rows, ['domain', 'difficulty_label'])
    recipe_length_rows = _count_rows(rows, ['recipe_length'])
    recipe_length_label_rows = _count_rows(rows, ['recipe_length_label'])
    recipe_family_rows = _count_rows(rows, ['order_family_id']) if track == 'order_sensitivity' else _count_rows(rows, ['recipe_id'])
    prompt_variant_count_rows = _count_rows(rows, ['prompt_variant_count'])
    prompt_style_rows = _prompt_distribution_rows(rows)
    recipe_rows = _recipe_distribution_rows(rows)

    _write_csv(stats_dir / f'{track}_task_track_distribution.csv', task_track_rows)
    _write_csv(stats_dir / f'{track}_task_order_setting_distribution.csv', task_order_setting_rows)
    _write_csv(stats_dir / f'{track}_task_difficulty_distribution.csv', task_difficulty_rows)
    _write_csv(stats_dir / f'{track}_task_difficulty_grid_distribution.csv', task_difficulty_grid_rows)
    _write_csv(stats_dir / f'{track}_task_domain_distribution.csv', task_domain_rows)
    _write_csv(stats_dir / f'{track}_task_domain_by_difficulty.csv', task_domain_difficulty_rows)
    _write_csv(stats_dir / f'{track}_recipe_length_distribution.csv', recipe_length_rows)
    _write_csv(stats_dir / f'{track}_recipe_length_label_distribution.csv', recipe_length_label_rows)
    _write_csv(stats_dir / f'{track}_recipe_group_distribution.csv', recipe_family_rows)
    _write_csv(stats_dir / f'{track}_recipe_distribution.csv', recipe_rows)
    _write_csv(stats_dir / f'{track}_prompt_variant_count_distribution.csv', prompt_variant_count_rows)
    _write_csv(stats_dir / f'{track}_prompt_style_distribution.csv', prompt_style_rows)

    return {
        'task': {
            'track_distribution': task_track_rows,
            'order_setting_distribution': task_order_setting_rows,
            'difficulty_distribution': task_difficulty_rows,
            'difficulty_grid_distribution': task_difficulty_grid_rows,
            'domain_distribution': task_domain_rows,
            'domain_by_difficulty': task_domain_difficulty_rows,
        },
        'recipe': {
            'length_distribution': recipe_length_rows,
            'length_label_distribution': recipe_length_label_rows,
            'group_distribution': recipe_family_rows,
            'recipe_distribution_count': len(recipe_rows),
        },
        'prompt': {
            'prompt_variant_count_distribution': prompt_variant_count_rows,
            'style_distribution_count': len(prompt_style_rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a release-friendly benchmark copy without replaying Data-Juicer operators.'
    )
    parser.add_argument('--benchmark-root', default='data/benchmark')
    parser.add_argument('--output-root', default='data/benchmark_release')
    parser.add_argument('--tracks', default='atomic_ops,main,order_sensitivity')
    parser.add_argument('--keep-replay', action='store_true', help='Preserve recipe_replay if present in the input rows.')
    parser.add_argument('--progress-every', type=int, default=1000)
    args = parser.parse_args()

    benchmark_root = (ROOT / args.benchmark_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    track_names = [item.strip() for item in str(args.tracks).split(',') if item.strip()]
    unsupported = [track for track in track_names if track not in TRACK_FILENAMES]
    if unsupported:
        raise SystemExit(f'unsupported tracks: {", ".join(unsupported)}')

    summary: dict[str, Any] = {
        'benchmark_root': str(benchmark_root),
        'output_root': str(output_root),
        'domain_metadata': DOMAIN_METADATA,
        'difficulty_definition': {
            'formula': (
                'task difficulty is assigned by a 3x3 grid over recipe_length_label '
                'and input_length_bucket. recipe_length_label is short for <=2 operators, '
                'medium for 3-4 operators, and long for >=5 operators.'
            ),
            'task_difficulty_grid': {
                'easy': ['recipe_short__input_short', 'recipe_short__input_medium', 'recipe_medium__input_short'],
                'medium': ['recipe_short__input_long', 'recipe_medium__input_medium', 'recipe_long__input_short'],
                'hard': ['recipe_medium__input_long', 'recipe_long__input_medium', 'recipe_long__input_long'],
            },
            'recipe_length_labeling': {'short': '<=2 operators', 'medium': '3-4 operators', 'long': '>=5 operators'},
        },
        'track_definition': {
            'atomic_ops': 'single-operator calibration baseline',
            'main': 'order-agnostic mapper-only compositional recipes',
            'order_sensitivity': 'order-sensitive recipes with a filter placed at pre/mid/post positions',
        },
        'tracks': {},
    }

    all_output_rows: list[dict[str, Any]] = []

    for track in track_names:
        input_path = _resolve_track_path(benchmark_root, track)
        rows = _read_jsonl(input_path)
        domain_counts: dict[str, int] = {}
        release_rows = []
        total_rows = len(rows)
        print(f'start release build track={track} rows={total_rows}', flush=True)
        for index, row in enumerate(rows, start=1):
            output_row = _ordered_row(row, keep_replay=bool(args.keep_replay))
            release_rows.append(output_row)
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == total_rows):
                print(f'progress release build track={track} row={index}/{total_rows}', flush=True)

        release_rows, filter_summary = _filter_release_rows_for_track(track, release_rows)
        if filter_summary.get('dropped_rows'):
            print(
                f'filtered release track={track}: kept={filter_summary["kept_rows"]} '
                f'dropped={filter_summary["dropped_rows"]} reason={filter_summary.get("filter_reason")}',
                flush=True,
            )

        scored_rows = _attach_difficulty(release_rows)
        output_rows = [_ordered_row(row, keep_replay=bool(args.keep_replay)) for row in scored_rows]
        for output_row in output_rows:
            domain = str(output_row.get('domain') or 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        relative_parent = input_path.parent.relative_to(benchmark_root) if input_path.parent != benchmark_root else Path()
        output_path = output_root / relative_parent / input_path.name
        count = _write_jsonl(output_path, output_rows)
        stats_summary = _write_release_statistics(output_root, track, output_rows)
        all_output_rows.extend(output_rows)
        summary['tracks'][track] = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'rows': count,
            'row_filtering': filter_summary,
            'domain_counts': dict(sorted(domain_counts.items())),
            'statistics': stats_summary,
        }
        print(f'wrote release track={track}: rows={count} -> {output_path}', flush=True)

    if all_output_rows:
        summary['all_tracks_statistics'] = _write_release_statistics(output_root, 'all', all_output_rows)

    _write_json(output_root / 'benchmark_release_summary.json', summary)
    print(f'wrote summary -> {output_root / "benchmark_release_summary.json"}', flush=True)


if __name__ == '__main__':
    main()
