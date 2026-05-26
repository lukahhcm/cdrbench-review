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
    'order_sensitivity': 'order_sensitivity.jsonl',
    'semantic_agent_atomic': 'semantic_agent_atomic.jsonl',
    'semantic_agent_compositional': 'semantic_agent_compositional.jsonl',
    'atomic_ops': 'atomic_ops.jsonl',
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
    return base_dir / track / TRACK_FILES[track]


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
    value = _first_present(row, 'recipe_prompt_key', 'workflow_prompt_key')
    if value:
        return str(value)
    return _recipe_key(row)


def _all_distinct_prompt_variants(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _output_row(
    row: dict[str, Any],
    *,
    recipe_prompt_key: str,
    prompt_library_row: dict[str, Any] | None,
) -> dict[str, Any]:
    candidates = list((prompt_library_row or {}).get('candidates') or [])
    prompt_variants = _all_distinct_prompt_variants(candidates)
    output_row = dict(row)
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
    if prompt_library_row is not None:
        output_row.setdefault('prompt_source', prompt_library_row.get('prompt_source'))
        output_row.setdefault('accepted_style_count', prompt_library_row.get('accepted_style_count'))
        output_row.setdefault('accepted_candidate_count', prompt_library_row.get('accepted_candidate_count'))
    return output_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build prompt-aware benchmark-full files by recomputing recipe_prompt_key from benchmark instances and joining existing prompt libraries.'
    )
    parser.add_argument('--benchmark-dir', default='data/processed/benchmark_instances')
    parser.add_argument('--prompt-library', default='data/processed/prompt_library')
    parser.add_argument('--output-dir', default='data/benchmark_full_prompt')
    parser.add_argument('--tracks', nargs='*', default=list(TRACK_FILES), choices=sorted(TRACK_FILES))
    parser.add_argument(
        '--min-prompt-variants-per-sample',
        type=int,
        default=1,
        help='Require at least this many distinct styles to count a row as fully matched.',
    )
    parser.add_argument(
        '--drop-unmatched-rows',
        action='store_true',
        help='Drop rows whose prompt pool is missing or has too few styles. By default rows are preserved with prompt_variant_count=0 or a smaller count.',
    )
    args = parser.parse_args()

    benchmark_dir = (ROOT / args.benchmark_dir).resolve()
    prompt_library_base = (ROOT / args.prompt_library).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for track in args.tracks:
        input_path = _resolve_benchmark_track_path(benchmark_dir, track)
        prompt_library_path = _resolve_prompt_library_path(prompt_library_base, track)
        library_rows = _read_jsonl(prompt_library_path)
        library_by_key = {
            str(_first_present(row, 'recipe_prompt_key', 'workflow_prompt_key')): row
            for row in library_rows
            if _first_present(row, 'recipe_prompt_key', 'workflow_prompt_key')
        }

        rows = _read_jsonl(input_path)
        total_rows = len(rows)
        output_rows = []
        missing_pool_rows = 0
        insufficient_style_rows = 0
        matched_rows = 0

        print(
            f'start prompt-full track={track} input_rows={total_rows} prompt_library={prompt_library_path}',
            flush=True,
        )
        for row_index, row in enumerate(rows, start=1):
            recipe_prompt_key = _row_recipe_prompt_key(row)
            prompt_library_row = library_by_key.get(recipe_prompt_key)
            candidates = list((prompt_library_row or {}).get('candidates') or [])
            distinct_style_count = len(
                {
                    str(candidate.get('style_id') or '')
                    for candidate in candidates
                    if candidate.get('style_id')
                }
            )

            if not candidates:
                missing_pool_rows += 1
                if not args.drop_unmatched_rows:
                    output_rows.append(
                        _output_row(
                            row,
                            recipe_prompt_key=recipe_prompt_key,
                            prompt_library_row=None,
                        )
                    )
            elif distinct_style_count < args.min_prompt_variants_per_sample:
                insufficient_style_rows += 1
                if not args.drop_unmatched_rows:
                    output_rows.append(
                        _output_row(
                            row,
                            recipe_prompt_key=recipe_prompt_key,
                            prompt_library_row=prompt_library_row,
                        )
                    )
            else:
                matched_rows += 1
                output_rows.append(
                    _output_row(
                        row,
                        recipe_prompt_key=recipe_prompt_key,
                        prompt_library_row=prompt_library_row,
                    )
                )

            if row_index % EVAL_PROGRESS_EVERY == 0 or row_index == total_rows:
                print(
                    f'progress prompt-full track={track} row={row_index}/{total_rows} '
                    f'kept={len(output_rows)} matched={matched_rows} '
                    f'missing_pool={missing_pool_rows} insufficient_styles={insufficient_style_rows}',
                    flush=True,
                )

        output_path = _resolve_track_output_path(output_dir, track)
        count = _write_jsonl(output_path, output_rows)
        summary_rows.append(
            {
                'track': track,
                'input_rows': total_rows,
                'kept_rows': count,
                'matched_rows': matched_rows,
                'missing_pool_rows': missing_pool_rows,
                'insufficient_style_rows': insufficient_style_rows,
                'min_prompt_variants_per_sample': args.min_prompt_variants_per_sample,
                'drop_unmatched_rows': bool(args.drop_unmatched_rows),
            }
        )
        print(f'wrote prompt-full track {track}: {count} rows -> {output_path}', flush=True)

    summary_path = output_dir / 'prompt_benchmark_full_summary.jsonl'
    _write_jsonl(summary_path, summary_rows)
    print(f'wrote prompt-full summary -> {summary_path}', flush=True)


if __name__ == '__main__':
    main()
