from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd

from cdrbench.dj_operator_loader import (
    Fields,
    create_operator,
    get_operator_execution_mode,
    get_operator_kind,
)


SAFE_MAX_CHARS_FOR_EXPENSIVE_MAPPERS = 80_000
EXPENSIVE_LONG_TEXT_MAPPERS = {
    'remove_repeat_sentences_mapper',
    'remove_words_with_incorrect_substrings_mapper',
}


# Regex-heavy table stripping can be prohibitively slow on very long documents.
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    preview = line[:200]
                    raise ValueError(
                        f'Invalid JSONL record in {path} at line {lineno}: {exc.msg}. '
                        f'Line preview: {preview!r}'
                    ) from exc
    return records


def _call_optional_context(method, payload):
    try:
        return method(payload, context=True)
    except TypeError:
        return method(payload)


def _infer_suffix(record: Dict[str, Any], raw_path: Path, field_map: Dict[str, str]) -> str:
    url_field = field_map.get('url', 'url')
    suffix_field = field_map.get('suffix', 'suffix')
    source_name_field = field_map.get('source_name', 'source_name')

    suffix = record.get(suffix_field)
    if isinstance(suffix, str) and suffix:
        return suffix

    url = record.get(url_field)
    if isinstance(url, str) and url:
        parsed = urlparse(url)
        candidate = Path(parsed.path).suffix
        if candidate:
            return candidate

    source_name = record.get(source_name_field)
    if isinstance(source_name, str) and source_name:
        candidate = Path(source_name).suffix
        if candidate:
            return candidate

    return raw_path.suffix


def normalize_record(
    record: Dict[str, Any],
    raw_path: Path,
    dataset_name: str | None = None,
    field_map: Dict[str, str] | None = None,
    defaults: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    field_map = field_map or {}
    defaults = defaults or {}

    def resolve(field_name: str, fallback: Any = None) -> Any:
        source_field = field_map.get(field_name, field_name)
        if source_field in record:
            return record[source_field]
        if field_name in defaults:
            return defaults[field_name]
        return fallback

    return {
        'id': resolve('id'),
        'dataset': resolve('domain', dataset_name or raw_path.stem),
        'source_name': resolve('source_name', raw_path.stem),
        'text': resolve('text', ''),
        'url': resolve('url'),
        'suffix': resolve('suffix', _infer_suffix(record, raw_path, field_map)),
    }


def _build_batch(text: str, suffix: str) -> Dict[str, Any]:
    return {
        'text': [text],
        Fields.stats: [{}],
        Fields.context: [{}],
        Fields.meta: [{}],
        Fields.suffix: [suffix],
    }


def _build_sample(text: str, suffix: str) -> Dict[str, Any]:
    return {
        'text': text,
        Fields.stats: {},
        Fields.context: {},
        Fields.meta: {},
        Fields.suffix: suffix,
    }


def run_mapper(op_name: str, text: str, params: Dict[str, Any], suffix: str = '') -> Dict[str, Any]:
    mode = get_operator_execution_mode(op_name)
    if op_name == 'remove_table_text_mapper':
        return {
            'kind': 'mapper',
            'execution_mode': mode,
            'active': False,
            'output_length': len(text),
            'delta_chars': 0,
            'skipped': 'temporarily_disabled_for_pipeline_runtime',
        }
    if op_name in EXPENSIVE_LONG_TEXT_MAPPERS and len(text) > SAFE_MAX_CHARS_FOR_EXPENSIVE_MAPPERS:
        return {
            'kind': 'mapper',
            'execution_mode': mode,
            'active': False,
            'output_length': len(text),
            'delta_chars': 0,
            'skipped': 'text_too_long_for_expensive_mapper',
        }
    try:
        op = create_operator(op_name, **params)
        if hasattr(op, 'process_batched') and op.is_batched_op():
            result = op.process_batched(_build_batch(text, suffix))
            output_text = result['text'][0]
        else:
            result = op.process_single(_build_sample(text, suffix))
            output_text = result['text']
        return {
            'kind': 'mapper',
            'execution_mode': mode,
            'active': output_text != text,
            'output_length': len(output_text),
            'delta_chars': len(output_text) - len(text),
        }
    except Exception as exc:
        return {
            'kind': 'mapper',
            'execution_mode': mode,
            'error': f'{type(exc).__name__}: {exc}',
        }


def run_filter(op_name: str, text: str, params: Dict[str, Any], suffix: str = '') -> Dict[str, Any]:
    mode = get_operator_execution_mode(op_name)
    try:
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
            'kind': 'filter',
            'execution_mode': mode,
            'keep': bool(keep),
            'stats': stats,
        }
    except Exception as exc:
        return {
            'kind': 'filter',
            'execution_mode': mode,
            'error': f'{type(exc).__name__}: {exc}',
            'stats': {},
        }


def _summarize_operator_results(
    labeled_name: str,
    label_field: str,
    operator_variants: List[Dict[str, Any]],
    tagged_records: List[Dict[str, Any]],
) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    for variant in operator_variants:
        key = variant['key']
        results = [row['operators'][key] for row in tagged_records]
        successful = [res for res in results if 'error' not in res]
        base_row = {
            label_field: labeled_name,
            'operator_key': key,
            'operator': variant['name'],
            'task': variant.get('task', ''),
            'kind': variant['kind'],
            'execution_mode': get_operator_execution_mode(variant['name']),
            'num_records': len(results),
            'successful_records': len(successful),
            'error_count': len(results) - len(successful),
        }
        if variant['kind'] == 'mapper':
            actives = [res['active'] for res in successful]
            deltas = [res['delta_chars'] for res in successful if res['active']]
            base_row.update(
                {
                    'active_count': sum(actives),
                    'active_rate': (sum(actives) / len(actives)) if actives else 0.0,
                    'mean_delta_chars_when_active': mean(deltas) if deltas else 0.0,
                }
            )
        else:
            keeps = [res['keep'] for res in successful]
            stat_values: List[float] = []
            for res in successful:
                stats = res.get('stats', {})
                stat_values.extend(value for value in stats.values() if isinstance(value, (int, float)))
            base_row.update(
                {
                    'keep_count': sum(1 for x in keeps if x),
                    'drop_count': sum(1 for x in keeps if not x),
                    'drop_rate': (sum(1 for x in keeps if not x) / len(keeps)) if keeps else 0.0,
                    'mean_stat_value': mean(stat_values) if stat_values else 0.0,
                }
            )
        summary_rows.append(base_row)
    return pd.DataFrame(summary_rows)


def scan_corpus_suite(
    raw_path: Path,
    operator_variants: List[Dict[str, Any]],
    tagged_path: Path,
    dataset_name: str | None = None,
    field_map: Dict[str, str] | None = None,
    defaults: Dict[str, Any] | None = None,
    label_field: str = 'corpus',
    max_records: int | None = None,
) -> pd.DataFrame:
    records = load_jsonl(raw_path)
    if max_records is not None:
        records = records[:max_records]
    labeled_name = dataset_name or raw_path.stem
    tagged_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tagged_path, 'w', encoding='utf-8') as out:
        for record in records:
            normalized = normalize_record(
                record,
                raw_path,
                dataset_name=dataset_name,
                field_map=field_map,
                defaults=defaults,
            )
            tag_payload = {
                'id': normalized['id'],
                label_field: labeled_name,
                'source_name': normalized['source_name'],
                'text_length': len(normalized['text']),
                'operators': {},
            }
            for variant in operator_variants:
                params = dict(variant.get('params', {}))
                if variant['kind'] == 'mapper':
                    res = run_mapper(variant['name'], normalized['text'], params, suffix=normalized['suffix'])
                else:
                    res = run_filter(variant['name'], normalized['text'], params, suffix=normalized['suffix'])
                tag_payload['operators'][variant['key']] = res
            out.write(json.dumps(tag_payload, ensure_ascii=False) + '\n')

    tagged_records = load_jsonl(tagged_path)
    return _summarize_operator_results(labeled_name, label_field, operator_variants, tagged_records)


def scan_domain(
    raw_path: Path,
    operators: List[Dict[str, Any]],
    tagged_path: Path,
    domain_name: str | None = None,
    field_map: Dict[str, str] | None = None,
    defaults: Dict[str, Any] | None = None,
    max_records: int | None = None,
) -> pd.DataFrame:
    variants = [
        {
            'key': op_cfg['name'],
            'name': op_cfg['name'],
            'kind': op_cfg.get('kind', get_operator_kind(op_cfg['name'])),
            'params': dict(op_cfg.get('params', {})),
        }
        for op_cfg in operators
    ]
    frame = scan_corpus_suite(
        raw_path,
        variants,
        tagged_path,
        dataset_name=domain_name,
        field_map=field_map,
        defaults=defaults,
        label_field='domain',
        max_records=max_records,
    )
    drop_cols = [col for col in ['operator_key', 'task', 'execution_mode', 'successful_records', 'error_count'] if col in frame.columns]
    return frame.drop(columns=drop_cols)
