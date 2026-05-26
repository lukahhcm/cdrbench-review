#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    tmp_path.replace(path)
    return len(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    tmp_path.replace(path)


def _text_length(row: dict[str, Any]) -> int:
    return len(str(row.get('text', '')))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Filter domain-filtered corpus rows by text length and write a new combined/per-domain corpus.'
    )
    parser.add_argument('--input-path', default='data/processed/domain_filtered/all.jsonl')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-text-length', type=int, required=True)
    parser.add_argument('--domain-field', default='domain')
    args = parser.parse_args()

    if args.max_text_length <= 0:
        raise SystemExit('--max-text-length must be > 0')

    input_path = (ROOT / args.input_path).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    if not input_path.exists():
        raise SystemExit(f'input path not found: {input_path}')

    rows = _read_jsonl(input_path)
    kept_rows: list[dict[str, Any]] = []
    kept_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_by_domain: Counter[str] = Counter()

    for row in rows:
        domain = str(row.get(args.domain_field) or 'unknown')
        if _text_length(row) > args.max_text_length:
            skipped_by_domain[domain] += 1
            continue
        kept_rows.append(row)
        kept_by_domain[domain].append(row)

    total_kept = _write_jsonl(output_dir / 'all.jsonl', kept_rows)
    for domain, domain_rows in sorted(kept_by_domain.items()):
        _write_jsonl(output_dir / f'{domain}.jsonl', domain_rows)

    summary = {
        'input_path': str(input_path),
        'output_dir': str(output_dir),
        'max_text_length': args.max_text_length,
        'input_row_count': len(rows),
        'kept_row_count': total_kept,
        'skipped_row_count': len(rows) - total_kept,
        'domains': [
            {
                'domain': domain,
                'kept_row_count': len(kept_by_domain.get(domain, [])),
                'skipped_row_count': int(skipped_by_domain.get(domain, 0)),
            }
            for domain in sorted(set(kept_by_domain) | set(skipped_by_domain))
        ],
    }
    _write_json(output_dir / 'length_filter_summary.json', summary)
    print(f'wrote filtered combined corpus -> {output_dir / "all.jsonl"}', flush=True)
    print(f'wrote length-filter summary -> {output_dir / "length_filter_summary.json"}', flush=True)


if __name__ == '__main__':
    main()
