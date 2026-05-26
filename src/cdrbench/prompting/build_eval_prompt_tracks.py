#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
EVAL_PROGRESS_EVERY = 200

TRACK_FILES = {
    'main': 'main.jsonl',
    'agnostic_m': 'agnostic_m.jsonl',
    'order_sensitivity': 'order_sensitivity.jsonl',
    'order_f': 'order_f.jsonl',
    'mapper_order_sensitivity': 'mapper_order_sensitivity.jsonl',
    'order_m': 'order_m.jsonl',
    'semantic_agent_atomic': 'semantic_agent_atomic.jsonl',
    'semantic_agent_compositional': 'semantic_agent_compositional.jsonl',
    'atomic_ops': 'atomic_ops.jsonl',
    'atomic_m': 'atomic_m.jsonl',
    'atomic_f': 'atomic_f.jsonl',
}

TRACK_PROMPT_LIBRARY_FILES = {
    track: f'{track}/recipe_prompt_library.jsonl' for track in TRACK_FILES
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve_prompt_library_path(base_path: Path, track: str) -> Path:
    if base_path.is_file():
        return base_path
    track_relative = TRACK_PROMPT_LIBRARY_FILES[track]
    candidate = base_path / track_relative
    if candidate.exists():
        return candidate
    fallback = base_path / 'recipe_prompt_library.jsonl'
    if fallback.exists():
        return fallback
    raise SystemExit(
        f'missing prompt library for track={track}: expected {candidate} or {fallback}'
    )


def _resolve_benchmark_track_path(base_dir: Path, track: str) -> Path:
    filename = TRACK_FILES[track]
    direct = base_dir / filename
    if direct.exists():
        return direct
    nested = base_dir / track / filename
    if nested.exists():
        return nested
    raise FileNotFoundError(f'missing benchmark track file for {track}: expected {direct} or {nested}')


def _resolve_track_output_path(base_dir: Path, track: str) -> Path:
    nested_dir = base_dir / track
    if nested_dir.exists():
        return nested_dir / TRACK_FILES[track]
    return base_dir / TRACK_FILES[track]


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


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = '||'.join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _recipe_key(row: dict[str, Any]) -> str:
    operator_sequence = list(row.get('operator_sequence') or ([row['operator']] if row.get('operator') else []))
    return _stable_id(
        row.get('benchmark_track'),
        row.get('domain'),
        _first_present(row, 'recipe_type', 'workflow_type'),
        row.get('order_slot'),
        operator_sequence,
        row.get('filter_params_by_name') or {},
    )


def _row_recipe_prompt_key(row: dict[str, Any]) -> str:
    return _recipe_key(row)


def _candidate_recipe_prompt_keys(row: dict[str, Any]) -> list[str]:
    keys = [_recipe_key(row)]
    for value in (
        _first_present(row, 'recipe_prompt_key', 'workflow_prompt_key'),
        row.get('recipe_id'),
        row.get('workflow_id'),
    ):
        if value is None:
            continue
        text = str(value)
        if text and text not in keys:
            keys.append(text)
    return keys


def _all_distinct_prompt_variants(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates_by_style: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        style_id = str(candidate.get('style_id') or '')
        if not style_id:
            continue
        candidates_by_style.setdefault(style_id, []).append(candidate)

    prompt_variants = []
    for style_id in sorted(candidates_by_style):
        style_candidates = sorted(
            candidates_by_style[style_id],
            key=lambda candidate: _stable_id(
                'prompt-candidate-canonical',
                style_id,
                candidate.get('candidate_id') or candidate.get('user_requirement') or '',
            ),
        )
        candidate = style_candidates[0]
        prompt_variants.append(
            {
                'style_id': str(candidate.get('style_id') or ''),
                'style_label': str(candidate.get('style_label') or ''),
                'user_requirement': str(candidate.get('user_requirement') or ''),
            }
        )
    return prompt_variants


def _eval_row(
    row: dict[str, Any],
    *,
    recipe_prompt_key: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_variants = _all_distinct_prompt_variants(candidates)
    keep_fields = [
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
        'operator_sequence',
        'filter_params_by_name',
        'filter_name',
    ]
    output_row = {
        'recipe_id': _first_present(row, 'recipe_id', 'workflow_id'),
        'recipe_variant_id': _first_present(row, 'recipe_variant_id', 'workflow_variant_id'),
        'recipe_type': _first_present(row, 'recipe_type', 'workflow_type'),
    }
    output_row.update({field: row[field] for field in keep_fields if field in row})
    output_row.update(
        {
            'recipe_prompt_key': recipe_prompt_key,
            'prompt_candidate_pool_count': len(candidates),
            'prompt_variant_count': len(prompt_variants),
            'prompt_sampling_policy': 'store_all_distinct_styles',
            'prompt_sampling_seed': None,
            'prompt_variants': prompt_variants,
        }
    )
    return output_row


def main() -> None:
    parser = argparse.ArgumentParser(description='Build eval-ready prompt track files from an accepted recipe prompt library.')
    parser.add_argument('--benchmark-dir', default='data/processed/benchmark_instances')
    parser.add_argument('--prompt-library', default='data/processed/prompt_library')
    parser.add_argument('--output-dir', default='data/benchmark')
    parser.add_argument('--tracks', nargs='*', default=list(TRACK_FILES), choices=sorted(TRACK_FILES))
    parser.add_argument('--prompt-variants-per-sample', type=int, default=3)
    parser.add_argument('--prompt-sampling-seed', type=int, default=0)
    parser.add_argument(
        '--min-prompt-variants-per-sample',
        type=int,
        default=3,
        help='Skip samples whose recipe prompt pool cannot supply at least this many distinct styles.',
    )
    parser.add_argument(
        '--preserve-all-benchmark-rows',
        action='store_true',
        help='Keep every benchmark row in the output even when its prompt pool is missing or has too few styles; prompt fields will still be attached with whatever count is available.',
    )
    args = parser.parse_args()

    benchmark_dir = (ROOT / args.benchmark_dir).resolve()
    prompt_library_base = (ROOT / args.prompt_library).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for track in args.tracks:
        try:
            input_path = _resolve_benchmark_track_path(benchmark_dir, track)
        except FileNotFoundError as exc:
            print(f'skip missing {track}: {exc}', flush=True)
            continue
        prompt_library_path = _resolve_prompt_library_path(prompt_library_base, track)
        library_rows = _read_jsonl(prompt_library_path)
        library_by_key = {
            str(_first_present(row, 'recipe_prompt_key', 'workflow_prompt_key')): list(row.get('candidates') or [])
            for row in library_rows
            if _first_present(row, 'recipe_prompt_key', 'workflow_prompt_key')
        }
        rows = _read_jsonl(input_path)
        total_rows = len(rows)
        output_rows = []
        missing_pool_rows = 0
        missing_pool_examples: list[dict[str, Any]] = []
        insufficient_style_rows = 0
        print(
            f'start eval track={track} input_rows={total_rows} prompt_library={prompt_library_path}',
            flush=True,
        )
        for row_index, row in enumerate(rows, start=1):
            recipe_prompt_key = _row_recipe_prompt_key(row)
            candidates = []
            for candidate_key in _candidate_recipe_prompt_keys(row):
                candidates = list(library_by_key.get(candidate_key) or [])
                if candidates:
                    recipe_prompt_key = candidate_key
                    break
            if not candidates:
                missing_pool_rows += 1
                if len(missing_pool_examples) < 5:
                    missing_pool_examples.append(
                        {
                            'row_index': row_index,
                            'instance_id': row.get('instance_id'),
                            'candidate_keys': _candidate_recipe_prompt_keys(row),
                            'benchmark_track': row.get('benchmark_track'),
                            'domain': row.get('domain'),
                            'recipe_type': _first_present(row, 'recipe_type', 'workflow_type'),
                            'order_slot': row.get('order_slot'),
                            'operator_sequence': row.get('operator_sequence'),
                        }
                    )
                if args.preserve_all_benchmark_rows:
                    output_rows.append(
                        _eval_row(
                            row,
                            recipe_prompt_key=recipe_prompt_key,
                            candidates=[],
                        )
                    )
                if row_index % EVAL_PROGRESS_EVERY == 0 or row_index == total_rows:
                    print(
                        f'progress eval track={track} row={row_index}/{total_rows} '
                        f'kept={len(output_rows)} missing_pool={missing_pool_rows} '
                        f'insufficient_styles={insufficient_style_rows}',
                        flush=True,
                    )
                continue
            distinct_style_count = len({str(candidate.get('style_id') or '') for candidate in candidates if candidate.get('style_id')})
            if distinct_style_count < args.min_prompt_variants_per_sample:
                insufficient_style_rows += 1
                if args.preserve_all_benchmark_rows:
                    output_rows.append(
                        _eval_row(
                            row,
                            recipe_prompt_key=recipe_prompt_key,
                            candidates=candidates,
                        )
                    )
                if row_index % EVAL_PROGRESS_EVERY == 0 or row_index == total_rows:
                    print(
                        f'progress eval track={track} row={row_index}/{total_rows} '
                        f'kept={len(output_rows)} missing_pool={missing_pool_rows} '
                        f'insufficient_styles={insufficient_style_rows}',
                        flush=True,
                    )
                continue
            output_rows.append(
                _eval_row(
                    row,
                    recipe_prompt_key=recipe_prompt_key,
                    candidates=candidates,
                )
            )
            if row_index % EVAL_PROGRESS_EVERY == 0 or row_index == total_rows:
                print(
                    f'progress eval track={track} row={row_index}/{total_rows} '
                    f'kept={len(output_rows)} missing_pool={missing_pool_rows} '
                    f'insufficient_styles={insufficient_style_rows}',
                    flush=True,
                )

        output_path = _resolve_track_output_path(output_dir, track)
        if rows and not output_rows and not args.preserve_all_benchmark_rows:
            if missing_pool_examples:
                debug_payload = {
                    'track': track,
                    'input_path': str(input_path),
                    'prompt_library_path': str(prompt_library_path),
                    'library_key_count': len(library_by_key),
                    'library_key_examples': list(library_by_key)[:5],
                    'missing_pool_examples': missing_pool_examples,
                }
                debug_path = output_path.parent / f'{track}_missing_prompt_pool_debug.json'
                debug_path.write_text(
                    json.dumps(debug_payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8',
                )
            raise SystemExit(
                f'No eval rows kept for track={track} from {len(rows)} input rows. '
                f'missing_pool_rows={missing_pool_rows} '
                f'insufficient_style_rows={insufficient_style_rows}. '
                f'Refusing to overwrite {output_path} with an empty file.'
            )
        count = _write_jsonl(output_path, output_rows)
        if missing_pool_examples:
            debug_payload = {
                'track': track,
                'input_path': str(input_path),
                'prompt_library_path': str(prompt_library_path),
                'library_key_count': len(library_by_key),
                'library_key_examples': list(library_by_key)[:5],
                'missing_pool_examples': missing_pool_examples,
            }
            (output_path.parent / f'{track}_missing_prompt_pool_debug.json').write_text(
                json.dumps(debug_payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                encoding='utf-8',
            )
        summary_rows.append(
            {
                'track': track,
                'input_rows': len(rows),
                'kept_rows': count,
                'missing_pool_rows': missing_pool_rows,
                'insufficient_style_rows': insufficient_style_rows,
                'prompt_variants_per_sample': args.prompt_variants_per_sample,
                'prompt_sampling_seed': args.prompt_sampling_seed,
                'min_prompt_variants_per_sample': args.min_prompt_variants_per_sample,
            }
        )
        print(f'wrote eval track {track}: {count} rows -> {output_path}', flush=True)

    _write_jsonl(output_dir / 'prompt_eval_build_summary.jsonl', summary_rows)
    print(f'wrote eval build summary -> {output_dir / "prompt_eval_build_summary.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
