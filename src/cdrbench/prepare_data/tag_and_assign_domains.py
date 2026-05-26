#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

from cdrbench.config import load_domains_config
from cdrbench.domain_assignment import (
    build_domain_execution_plan,
    build_filtered_record,
    domain_operator_catalog_frame,
    rank_domain_candidates,
)


FILTER_STATUS_RULES: dict[str, dict[str, Any]] = {
    'alphanumeric_filter': {'value_key': lambda params: 'alpha_token_ratio' if params.get('tokenization') else 'alnum_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'average_line_length_filter': {'value_key': 'avg_line_length', 'min_key': 'min_len', 'max_key': 'max_len'},
    'character_repetition_filter': {'value_key': 'char_rep_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'flagged_words_filter': {'value_key': 'flagged_words_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'maximum_line_length_filter': {'value_key': 'max_line_length', 'min_key': 'min_len', 'max_key': 'max_len'},
    'stopwords_filter': {'value_key': 'stopwords_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'text_length_filter': {'value_key': 'text_len', 'min_key': 'min_len', 'max_key': 'max_len'},
    'word_repetition_filter': {'value_key': 'word_rep_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'words_num_filter': {'value_key': 'num_words', 'min_key': 'min_num', 'max_key': 'max_num'},
}

TAGGING_MAPPER_RULES: dict[str, dict[str, Any]] = {
    'extract_tables_from_html_mapper': {
        'meta_key': 'html_tables',
        'is_active': lambda value: isinstance(value, list) and len(value) > 0,
    },
}

ROW_EXPANDING_MAPPERS = {
    'sentence_split_mapper',
    'text_chunk_mapper',
}


def is_supported_tagging_variant(variant: dict[str, Any]) -> bool:
    if variant['kind'] == 'mapper':
        return True
    return variant['name'] in FILTER_STATUS_RULES


def validate_variant_for_tagging(variant: dict[str, Any]) -> None:
    if variant['kind'] != 'mapper':
        return
    if variant['name'] in ROW_EXPANDING_MAPPERS:
        raise SystemExit(
            f"unsupported row-expanding mapper in tagging pipeline: {variant['name']}. "
            'Current CDR-Bench tagging assumes one input row maps to one output row. '
            'Remove this mapper from configs/domains.yaml or convert it to a one-to-one text mode first.'
        )


def is_tagging_mapper_variant(variant: dict[str, Any]) -> bool:
    if variant['kind'] != 'mapper' or variant['name'] not in TAGGING_MAPPER_RULES:
        return False
    output_mode = variant.get('params', {}).get('output_mode', 'meta')
    return output_mode == 'meta'


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                preview = line[:200]
                raise ValueError(
                    f'Invalid JSONL record in {path} at line {lineno}: {exc.msg}. Line preview: {preview!r}'
                ) from exc
    return rows


def load_cli_rows_with_recovery(
    *,
    expected_path: Path,
    expected_count: int,
    kind_label: str,
    corpus_name: str,
    op_key: str,
    cleanup_paths: list[Path],
    cmd: list[str],
    log_path: Path,
    cmd_env: dict[str, str] | None,
    aggregate_only: bool,
    resume: bool,
) -> list[dict[str, Any]]:
    rows = iter_jsonl(expected_path)
    if len(rows) == expected_count:
        return rows

    message = (
        f'{kind_label} row count mismatch for {corpus_name}/{op_key}: '
        f'expected={expected_count} actual={len(rows)} path={expected_path}'
    )
    if aggregate_only or not resume:
        raise ValueError(message)

    print(f'[{corpus_name}] recovering stale CLI output for {op_key}: expected {expected_count}, found {len(rows)}')
    for path in cleanup_paths:
        if path.exists():
            path.unlink()

    exit_code = run_command(cmd, log_path, env=cmd_env)
    if exit_code != 0:
        log_tail = read_log_tail(log_path)
        detail = f'\n--- log tail ({log_path}) ---\n{log_tail}' if log_tail else f'; see {log_path}'
        raise SystemExit(f'command failed ({exit_code}) during recovery: {" ".join(cmd)}{detail}')
    if not expected_path.exists():
        raise SystemExit(f'missing expected CLI output after recovery for {corpus_name}/{op_key}: {expected_path}')

    rows = iter_jsonl(expected_path)
    if len(rows) != expected_count:
        raise ValueError(message)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def resolve_path(root: Path, raw_path_value: str) -> Path:
    raw_path = Path(raw_path_value)
    if raw_path.is_absolute():
        return raw_path
    return root / raw_path


def count_jsonl_lines(path: Path) -> int:
    with path.open('rb') as f:
        return sum(1 for _ in f)


def ensure_head_sample(source_path: Path, sample_path: Path, max_records: int) -> Path:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_path.exists() and count_jsonl_lines(sample_path) == max_records:
        return sample_path

    with source_path.open('r', encoding='utf-8') as src, sample_path.open('w', encoding='utf-8') as dst:
        for i, line in enumerate(src):
            if i >= max_records:
                break
            dst.write(line)
    return sample_path


def resolve_bin(raw: str) -> str:
    if '/' in raw:
        return str((ROOT / raw).resolve()) if not Path(raw).is_absolute() else raw
    return raw


def resolve_repo_dir(root: Path, raw_path_value: str | None) -> Path | None:
    if not raw_path_value:
        return None
    repo_dir = resolve_path(root, raw_path_value)
    return repo_dir if repo_dir.exists() else None


def build_process_cfg(dataset_path: Path, export_path: Path, op_name: str, params: dict[str, Any], np: int, project_name: str) -> dict[str, Any]:
    return {
        'project_name': project_name,
        'dataset_path': str(dataset_path),
        'np': np,
        'export_path': str(export_path),
        'process': [{op_name: params}],
    }


def build_analyze_cfg(dataset_path: Path, export_path: Path, op_name: str, params: dict[str, Any], np: int, project_name: str) -> dict[str, Any]:
    return {
        'project_name': project_name,
        'dataset_path': str(dataset_path),
        'np': np,
        'export_path': str(export_path),
        'export_original_dataset': False,
        'save_stats_in_one_file': True,
        'process': [{op_name: params}],
    }


def stats_path_for_export(export_path: Path) -> Path:
    return export_path.with_name(f'{export_path.stem}_stats.jsonl')


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False, env=env)
    return proc.returncode


def read_log_tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ''
    with path.open('r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return ''.join(lines[-max_lines:])


def build_dj_invocation(
    *,
    explicit_bin: str,
    default_bin: str,
    dj_python: str,
    dj_repo_root: Path | None,
    repo_script_relpath: str,
) -> tuple[list[str], dict[str, str] | None, str]:
    if explicit_bin != default_bin:
        resolved_bin = resolve_bin(explicit_bin)
        return [resolved_bin], None, f'bin:{resolved_bin}'

    if dj_repo_root is not None:
        env = os.environ.copy()
        existing_pythonpath = env.get('PYTHONPATH')
        repo_pythonpath = str(dj_repo_root)
        env['PYTHONPATH'] = f'{repo_pythonpath}{os.pathsep}{existing_pythonpath}' if existing_pythonpath else repo_pythonpath
        resolved_python = resolve_bin(dj_python)
        repo_script = dj_repo_root / repo_script_relpath
        bootstrap = (
            'import runpy, sys; '
            'repo_root = sys.argv[1]; '
            'script_path = sys.argv[2]; '
            'script_args = sys.argv[3:]; '
            'sys.path.insert(0, repo_root); '
            'stale = [name for name in sys.modules if name == "data_juicer" or name.startswith("data_juicer.")]; '
            '[sys.modules.pop(name, None) for name in stale]; '
            'sys.argv = [script_path] + script_args; '
            'runpy.run_path(script_path, run_name="__main__")'
        )
        return [resolved_python, '-c', bootstrap, str(dj_repo_root), str(repo_script)], env, f'repo:{repo_script}'

    return [resolve_bin(default_bin)], None, f'bin:{default_bin}'


def resolve_record_field(record: dict[str, Any], field_name: str, field_map: dict[str, str] | None = None, defaults: dict[str, Any] | None = None, fallback: Any = None) -> Any:
    field_map = field_map or {}
    defaults = defaults or {}
    source_field = field_map.get(field_name, field_name)
    if source_field in record:
        return record[source_field]
    if field_name in defaults:
        return defaults[field_name]
    return fallback


def infer_suffix_from_record(
    record: dict[str, Any],
    *,
    raw_path: Path,
    field_map: dict[str, str] | None = None,
) -> str:
    field_map = field_map or {}

    suffix_field = field_map.get('suffix', 'suffix')
    suffix = record.get(suffix_field)
    if isinstance(suffix, str) and suffix:
        return suffix

    url_field = field_map.get('url', 'url')
    url = record.get(url_field)
    if isinstance(url, str) and url:
        candidate = Path(url).suffix
        if candidate:
            return candidate

    source_name_field = field_map.get('source_name', 'source_name')
    source_name = record.get(source_name_field)
    if isinstance(source_name, str) and source_name:
        candidate = Path(source_name).suffix
        if candidate:
            return candidate

    return raw_path.suffix


def build_cli_input_rows(
    records: list[dict[str, Any]],
    *,
    raw_path: Path,
    field_map: dict[str, str] | None = None,
    defaults: dict[str, Any] | None = None,
    corpus_name: str,
) -> list[dict[str, Any]]:
    field_map = field_map or {}
    defaults = defaults or {}

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        rows.append(
            {
                'id': resolve_record_field(record, 'id', field_map=field_map, defaults=defaults, fallback=idx),
                'text': str(resolve_record_field(record, 'text', field_map=field_map, defaults=defaults, fallback='') or ''),
                'domain': resolve_record_field(record, 'domain', field_map=field_map, defaults=defaults),
                'source_name': resolve_record_field(
                    record, 'source_name', field_map=field_map, defaults=defaults, fallback=raw_path.stem
                ),
                'url': resolve_record_field(record, 'url', field_map=field_map, defaults=defaults),
                'suffix': infer_suffix_from_record(record, raw_path=raw_path, field_map=field_map),
                'corpus': corpus_name,
            }
        )
    return rows


def filter_records_by_text_length(
    records: list[dict[str, Any]],
    *,
    field_map: dict[str, str] | None = None,
    defaults: dict[str, Any] | None = None,
    max_text_length: int,
) -> tuple[list[dict[str, Any]], int]:
    if max_text_length <= 0:
        return records, 0

    kept_records: list[dict[str, Any]] = []
    skipped_records = 0
    for record in records:
        text = str(resolve_record_field(record, 'text', field_map=field_map, defaults=defaults, fallback='') or '')
        if len(text) > max_text_length:
            skipped_records += 1
            continue
        kept_records.append(record)
    return kept_records, skipped_records


def get_keep_boolean(
    value: float | int | None,
    min_val: float | int | None,
    max_val: float | int | None,
    min_closed_interval: bool = True,
    max_closed_interval: bool = True,
    reversed_range: bool = False,
) -> bool | None:
    if value is None:
        return None
    if reversed_range:
        min_closed_interval = not min_closed_interval
        max_closed_interval = not max_closed_interval
    res = True
    if min_val is not None:
        res = res and (value >= min_val if min_closed_interval else value > min_val)
    if max_val is not None:
        res = res and (value <= max_val if max_closed_interval else value < max_val)
    if reversed_range:
        res = not res
    return res


def infer_filter_status(op_name: str, params: dict[str, Any], stats: dict[str, Any]) -> tuple[bool | None, str | None]:
    rule = FILTER_STATUS_RULES.get(op_name)
    if rule is None:
        return None, None

    value_key = rule['value_key'](params) if callable(rule['value_key']) else rule['value_key']
    value = stats.get(value_key)
    keep = get_keep_boolean(
        value=value,
        min_val=params.get(rule['min_key']),
        max_val=params.get(rule['max_key']),
        min_closed_interval=bool(params.get('min_closed_interval', True)),
        max_closed_interval=bool(params.get('max_closed_interval', True)),
        reversed_range=bool(params.get('reversed_range', False)),
    )
    return keep, value_key


def aggregate_mapper_results(
    input_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    *,
    text_field: str,
) -> list[dict[str, Any]]:
    if len(input_rows) != len(output_rows):
        raise ValueError(f'mapper output row count mismatch: input={len(input_rows)} output={len(output_rows)}')

    results: list[dict[str, Any]] = []
    for src, out in zip(input_rows, output_rows):
        src_text = src.get(text_field, '')
        out_text = out.get(text_field, '')
        results.append(
            {
                'kind': 'mapper',
                'execution_mode': 'process',
                'active': out_text != src_text,
                'output_length': len(out_text),
                'delta_chars': len(out_text) - len(src_text),
            }
        )
    return results


def aggregate_tagging_mapper_results(
    stats_rows: list[dict[str, Any]],
    *,
    op_name: str,
) -> list[dict[str, Any]]:
    rule = TAGGING_MAPPER_RULES[op_name]
    meta_key = rule['meta_key']
    is_active = rule['is_active']
    results: list[dict[str, Any]] = []
    for row in stats_rows:
        meta = row.get('__dj__meta__', {}) or {}
        meta_value = meta.get(meta_key)
        results.append(
            {
                'kind': 'mapper',
                'execution_mode': 'analyze_meta',
                'active': bool(is_active(meta_value)),
                'meta': meta,
                'meta_key': meta_key,
                'meta_value': meta_value,
            }
        )
    return results


def aggregate_filter_results(
    stats_rows: list[dict[str, Any]],
    *,
    op_name: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in stats_rows:
        stats = row.get('__dj__stats__', {}) or {}
        meta = row.get('__dj__meta__', {}) or {}
        keep, value_key = infer_filter_status(op_name, params, stats)
        results.append(
            {
                'kind': 'filter',
                'execution_mode': 'analyze',
                'keep': keep,
                'status': 'KEEP' if keep is True else 'DROP' if keep is False else 'UNKNOWN',
                'stats': stats,
                'meta': meta,
                'status_value_key': value_key,
                'status_value': stats.get(value_key) if value_key else None,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Tag corpora via Data-Juicer CLI, then aggregate per-op outputs into domain assignments and recipe-mining inputs.'
    )
    parser.add_argument('--corpora-config', default='configs/corpora.yaml')
    parser.add_argument('--domains-config', default='configs/domains.yaml')
    parser.add_argument('--corpora', nargs='*', default=None)
    parser.add_argument('--config-dir', default='data/processed/dj_cli_tagging/configs')
    parser.add_argument('--per-op-dir', default='data/processed/dj_cli_tagging/per_op')
    parser.add_argument('--sample-dir', default='data/processed/dj_cli_tagging/samples')
    parser.add_argument('--log-dir', default='data/processed/dj_cli_tagging/logs')
    parser.add_argument('--tagged-dir', default='data/processed/domain_tags')
    parser.add_argument('--filtered-dir', default='data/processed/domain_filtered')
    parser.add_argument('--combined-path', default='data/processed/domain_filtered/all.jsonl')
    parser.add_argument('--summary-path', default='data/processed/domain_labeling_summary.csv')
    parser.add_argument('--assignments-path', default='data/processed/domain_assignment_counts.csv')
    parser.add_argument('--catalog-path', default='data/processed/domain_operator_catalog.csv')
    parser.add_argument('--dj-process-bin', default='dj-process')
    parser.add_argument('--dj-analyze-bin', default='dj-analyze')
    parser.add_argument('--dj-python', default=sys.executable, help='Python executable used for repo-local Data-Juicer mode.')
    parser.add_argument(
        '--dj-repo-root',
        default='data-juicer',
        help='Repo-local Data-Juicer checkout. If this directory exists and custom CLI bins are not provided, tagging uses python -m data_juicer.tools.* with this repo injected into PYTHONPATH.',
    )
    parser.add_argument('--np', type=int, default=4)
    parser.add_argument('--min-active-mappers', type=int, default=2)
    parser.add_argument('--max-text-length', type=int, default=0, help='Drop raw records with text longer than this before tagging. 0 disables the filter.')
    parser.add_argument('--max-records', type=int, default=None)
    parser.add_argument('--resume', action='store_true', help='Skip existing per-op CLI outputs and re-aggregate.')
    parser.add_argument('--aggregate-only', action='store_true', help='Only aggregate existing CLI outputs without running dj-process/dj-analyze.')
    args = parser.parse_args()

    if args.max_text_length < 0:
        raise SystemExit('--max-text-length must be >= 0')

    root = ROOT
    corpora_cfg = load_domains_config(root / args.corpora_config)['corpora']
    domains_cfg = load_domains_config(root / args.domains_config)
    selected = set(args.corpora) if args.corpora else None

    dj_repo_root = resolve_repo_dir(root, args.dj_repo_root)
    process_cmd_prefix, process_env, process_mode = build_dj_invocation(
        explicit_bin=args.dj_process_bin,
        default_bin='dj-process',
        dj_python=args.dj_python,
        dj_repo_root=dj_repo_root,
        repo_script_relpath='tools/process_data.py',
    )
    analyze_cmd_prefix, analyze_env, analyze_mode = build_dj_invocation(
        explicit_bin=args.dj_analyze_bin,
        default_bin='dj-analyze',
        dj_python=args.dj_python,
        dj_repo_root=dj_repo_root,
        repo_script_relpath='tools/analyze_data.py',
    )
    config_dir = root / args.config_dir
    per_op_dir = root / args.per_op_dir
    sample_dir = root / args.sample_dir
    log_dir = root / args.log_dir

    if dj_repo_root is not None and args.dj_process_bin == 'dj-process' and args.dj_analyze_bin == 'dj-analyze':
        print(f'using repo-local Data-Juicer checkout -> {dj_repo_root}')
    else:
        print(f'using Data-Juicer process entry -> {process_mode}')
        print(f'using Data-Juicer analyze entry -> {analyze_mode}')

    plan = build_domain_execution_plan(domains_cfg)
    for variant in plan['execution_variants']:
        validate_variant_for_tagging(variant)
    supported_variants = [variant for variant in plan['execution_variants'] if is_supported_tagging_variant(variant)]
    skipped_variants = [variant for variant in plan['execution_variants'] if not is_supported_tagging_variant(variant)]
    if skipped_variants:
        skipped_names = ', '.join(variant['name'] for variant in skipped_variants)
        print(f'skipping unsupported tagging operators: {skipped_names}')
    catalog = domain_operator_catalog_frame(plan)
    catalog_path = root / args.catalog_path
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)
    print(f'wrote operator catalog -> {catalog_path}')

    summary_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []

    for corpus_name, corpus_cfg in corpora_cfg.items():
        if selected and corpus_name not in selected:
            continue

        raw_path = resolve_path(root, corpus_cfg['raw_path'])
        if not raw_path.exists():
            print(f'skip {corpus_name}: missing {raw_path}')
            continue

        field_map = corpus_cfg.get('field_map')
        defaults = corpus_cfg.get('defaults')
        input_source_path = raw_path
        if args.max_records is not None:
            input_source_path = ensure_head_sample(
                raw_path, sample_dir / f'{corpus_name}-head{args.max_records}.jsonl', args.max_records
            )

        raw_records = iter_jsonl(input_source_path)
        corpus_records, skipped_long_text_records = filter_records_by_text_length(
            raw_records,
            field_map=field_map,
            defaults=defaults,
            max_text_length=args.max_text_length,
        )
        cli_input_rows = build_cli_input_rows(
            corpus_records,
            raw_path=raw_path,
            field_map=field_map,
            defaults=defaults,
            corpus_name=corpus_name,
        )
        cli_input_path = sample_dir / f'{corpus_name}-djcli.jsonl'
        if args.max_records is not None:
            cli_input_path = sample_dir / f'{corpus_name}-head{args.max_records}-djcli.jsonl'
        write_jsonl(cli_input_path, cli_input_rows)

        input_path = cli_input_path
        dj_input_records = cli_input_rows
        text_field = 'text'
        per_op_results: dict[str, list[dict[str, Any]]] = {}

        if not corpus_records:
            tagged_path = root / args.tagged_dir / f'{corpus_name}.jsonl'
            filtered_path = root / args.filtered_dir / f'{corpus_name}.jsonl'
            write_jsonl(tagged_path, [])
            write_jsonl(filtered_path, [])
            print(
                f'{corpus_name}: kept 0 / 0 after raw length filter '
                f'(skipped_long_text={skipped_long_text_records}) -> {filtered_path}'
            )
            summary_rows.append(
                {
                    'corpus': corpus_name,
                    'total_records': len(raw_records),
                    'eligible_records': 0,
                    'best_candidate_records': 0,
                    'assigned_records': 0,
                    'kept_records': 0,
                    'dropped_records': 0,
                    'insufficient_mapper_records': 0,
                    'skipped_long_text_records': skipped_long_text_records,
                    'mean_active_mapper_count': 0.0,
                    'keep_rate': 0.0,
                }
            )
            continue

        for variant in supported_variants:
            op_key = variant['key']
            op_name = variant['name']
            op_kind = variant['kind']
            params = dict(variant.get('params', {}))
            use_analyze_for_mapper = is_tagging_mapper_variant(variant)

            corpus_config_dir = config_dir / corpus_name
            corpus_output_dir = per_op_dir / corpus_name
            corpus_config_dir.mkdir(parents=True, exist_ok=True)
            corpus_output_dir.mkdir(parents=True, exist_ok=True)

            export_path = corpus_output_dir / f'{op_key}.jsonl'
            cfg_path = corpus_config_dir / f'{op_key}.yaml'
            expected_path = export_path
            if op_kind == 'mapper' and not use_analyze_for_mapper:
                payload = build_process_cfg(
                    dataset_path=input_path,
                    export_path=export_path,
                    op_name=op_name,
                    params=params,
                    np=args.np,
                    project_name=f'cdrbench-{corpus_name}-{op_key}-process',
                )
                cmd = [*process_cmd_prefix, '--config', str(cfg_path)]
                cmd_env = process_env
                log_path = log_dir / corpus_name / f'{op_key}__process.log'
            else:
                payload = build_analyze_cfg(
                    dataset_path=input_path,
                    export_path=export_path,
                    op_name=op_name,
                    params=params,
                    np=args.np,
                    project_name=f'cdrbench-{corpus_name}-{op_key}-analyze',
                )
                cmd = [*analyze_cmd_prefix, '--config', str(cfg_path)]
                cmd_env = analyze_env
                expected_path = stats_path_for_export(export_path)
                suffix = 'analyze_meta' if use_analyze_for_mapper else 'analyze'
                log_path = log_dir / corpus_name / f'{op_key}__{suffix}.log'

            with cfg_path.open('w', encoding='utf-8') as f:
                yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

            if not args.aggregate_only:
                should_run = not (args.resume and expected_path.exists())
                if should_run:
                    print(f'[{corpus_name}] running {op_kind} {op_name}')
                    exit_code = run_command(cmd, log_path, env=cmd_env)
                    if exit_code != 0:
                        log_tail = read_log_tail(log_path)
                        detail = f'\n--- log tail ({log_path}) ---\n{log_tail}' if log_tail else f'; see {log_path}'
                        raise SystemExit(f'command failed ({exit_code}): {" ".join(cmd)}{detail}')

            if not expected_path.exists():
                raise SystemExit(f'missing expected CLI output for {corpus_name}/{op_key}: {expected_path}')

            if op_kind == 'mapper' and not use_analyze_for_mapper:
                output_rows = load_cli_rows_with_recovery(
                    expected_path=export_path,
                    expected_count=len(dj_input_records),
                    kind_label='mapper output',
                    corpus_name=corpus_name,
                    op_key=op_key,
                    cleanup_paths=[export_path],
                    cmd=cmd,
                    log_path=log_path,
                    cmd_env=cmd_env,
                    aggregate_only=args.aggregate_only,
                    resume=args.resume,
                )
                per_op_results[op_key] = aggregate_mapper_results(dj_input_records, output_rows, text_field=text_field)
            elif op_kind == 'mapper':
                stats_rows = load_cli_rows_with_recovery(
                    expected_path=expected_path,
                    expected_count=len(dj_input_records),
                    kind_label='tagging mapper stats',
                    corpus_name=corpus_name,
                    op_key=op_key,
                    cleanup_paths=[expected_path, export_path],
                    cmd=cmd,
                    log_path=log_path,
                    cmd_env=cmd_env,
                    aggregate_only=args.aggregate_only,
                    resume=args.resume,
                )
                per_op_results[op_key] = aggregate_tagging_mapper_results(stats_rows, op_name=op_name)
            else:
                stats_rows = load_cli_rows_with_recovery(
                    expected_path=expected_path,
                    expected_count=len(dj_input_records),
                    kind_label='filter stats',
                    corpus_name=corpus_name,
                    op_key=op_key,
                    cleanup_paths=[expected_path, export_path],
                    cmd=cmd,
                    log_path=log_path,
                    cmd_env=cmd_env,
                    aggregate_only=args.aggregate_only,
                    resume=args.resume,
                )
                per_op_results[op_key] = aggregate_filter_results(stats_rows, op_name=op_name, params=params)

        tagged_rows: list[dict[str, Any]] = []
        filtered_rows: list[dict[str, Any]] = []
        best_candidate_assignments: Counter[str] = Counter()
        kept_assignments: Counter[str] = Counter()
        total_active_mapper_count = 0

        for idx, record in enumerate(corpus_records):
            operator_results = {key: per_op_results[key][idx] for key in per_op_results}
            active_mapper_keys = [
                key for key, result in operator_results.items() if plan['execution_variants_by_key'][key]['kind'] == 'mapper' and result.get('active')
            ]
            active_mapper_names = sorted({plan['execution_variants_by_key'][key]['name'] for key in active_mapper_keys})
            unique_active_mapper_names = sorted(
                {
                    plan['execution_variants_by_key'][key]['name']
                    for profile in plan['domain_profiles'].values()
                    for key in profile.get('unique_mapper_keys', [])
                    if key in active_mapper_keys
                }
            )

            preferred_domain = resolve_record_field(record, 'domain', field_map=field_map, defaults=defaults)
            if preferred_domain not in plan['assignment_domains']:
                preferred_domain = None

            domain_candidates = rank_domain_candidates(operator_results, plan, preferred_domain=preferred_domain)
            best_domain_candidate = None
            if domain_candidates and domain_candidates[0]['matched_mapper_count'] > 0:
                best_domain_candidate = domain_candidates[0]['domain']
            assigned_domain = None
            if best_domain_candidate is not None and len(active_mapper_names) >= args.min_active_mappers:
                assigned_domain = best_domain_candidate

            tag_payload = {
                'id': resolve_record_field(record, 'id', field_map=field_map, defaults=defaults, fallback=idx),
                'corpus': corpus_name,
                'source_name': resolve_record_field(record, 'source_name', field_map=field_map, defaults=defaults, fallback=raw_path.stem),
                'original_domain': resolve_record_field(record, 'domain', field_map=field_map, defaults=defaults),
                'best_domain_candidate': best_domain_candidate,
                'assigned_domain': assigned_domain,
                'text_length': len(str(resolve_record_field(record, 'text', field_map=field_map, defaults=defaults, fallback=''))),
                'active_mapper_count': len(active_mapper_names),
                'active_mapper_names': active_mapper_names,
                'unique_active_mapper_count': len(unique_active_mapper_names),
                'unique_active_mapper_names': unique_active_mapper_names,
                'keep': assigned_domain is not None and len(active_mapper_names) >= args.min_active_mappers,
                'domain_candidates': domain_candidates,
                'operators': operator_results,
                'skipped_long_text': False,
            }
            tagged_rows.append(tag_payload)
            total_active_mapper_count += tag_payload['active_mapper_count']

            if best_domain_candidate is not None:
                best_candidate_assignments[best_domain_candidate] += 1
            if tag_payload['keep'] and assigned_domain is not None:
                filtered_record = build_filtered_record(record, corpus_name, assigned_domain, tag_payload)
                filtered_rows.append(filtered_record)
                combined_rows.append(filtered_record)
                kept_assignments[assigned_domain] += 1

        tagged_path = root / args.tagged_dir / f'{corpus_name}.jsonl'
        filtered_path = root / args.filtered_dir / f'{corpus_name}.jsonl'
        write_jsonl(tagged_path, tagged_rows)
        write_jsonl(filtered_path, filtered_rows)
        print(
            f'{corpus_name}: kept {len(filtered_rows)} / {len(tagged_rows)} '
            f'(assigned {sum(best_candidate_assignments.values())}, '
            f'skipped_long_text={skipped_long_text_records}) -> {filtered_path}'
        )

        summary_rows.append(
            {
                'corpus': corpus_name,
                'total_records': len(raw_records),
                'eligible_records': len(tagged_rows),
                'best_candidate_records': sum(best_candidate_assignments.values()),
                'assigned_records': sum(kept_assignments.values()),
                'kept_records': len(filtered_rows),
                'dropped_records': len(tagged_rows) - len(filtered_rows),
                'insufficient_mapper_records': sum(1 for row in tagged_rows if row['active_mapper_count'] < args.min_active_mappers),
                'skipped_long_text_records': skipped_long_text_records,
                'mean_active_mapper_count': (total_active_mapper_count / len(tagged_rows)) if tagged_rows else 0.0,
                'keep_rate': (len(filtered_rows) / len(tagged_rows)) if tagged_rows else 0.0,
            }
        )
        for domain_name, count in sorted(best_candidate_assignments.items()):
            assignment_rows.append({'corpus': corpus_name, 'scope': 'best_candidate', 'assigned_domain': domain_name, 'count': count})
        for domain_name, count in sorted(kept_assignments.items()):
            assignment_rows.append({'corpus': corpus_name, 'scope': 'kept', 'assigned_domain': domain_name, 'count': count})

    if not summary_rows:
        print('no corpora were processed')
        return

    combined_path = root / args.combined_path
    write_jsonl(combined_path, combined_rows)

    summary = pd.DataFrame(summary_rows)
    summary_path = root / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f'wrote summary -> {summary_path}')

    assignments = pd.DataFrame(assignment_rows, columns=['corpus', 'scope', 'assigned_domain', 'count'])
    assignments_path = root / args.assignments_path
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_path, index=False)
    print(f'wrote assignment counts -> {assignments_path}')
    print(f'wrote combined filtered corpus -> {combined_path}')


if __name__ == '__main__':
    main()
