#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import yaml


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open('rb') as f:
        return sum(1 for _ in f)


def ensure_head_sample(source_path: Path, sample_path: Path, max_records: int) -> Path:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_path.exists() and count_jsonl(sample_path) == max_records:
        return sample_path

    with source_path.open('r', encoding='utf-8') as src, sample_path.open('w', encoding='utf-8') as dst:
        for i, line in enumerate(src):
            if i >= max_records:
                break
            dst.write(line)
    return sample_path


def build_corpus_domain_map(root: Path) -> list[tuple[str, str, Path]]:
    candidates = [
        ('arxiv', 'arxiv', root / 'data/raw/arxiv/arxiv-4k.jsonl'),
        ('commoncrawl_10k', 'web', root / 'data/raw/commoncrawl/cc-10k.jsonl'),
        ('enwiki', 'web', root / 'data/raw/enwiki/enwiki-pages-110k.jsonl'),
        ('govreport', 'knowledge_base', root / 'data/raw/govreport/govreport-20k.jsonl'),
        ('pii_main', 'pii', root / 'data/raw/pii/pii-43k.jsonl'),
        ('pii_docpii', 'pii', root / 'data/raw/pii/docpii-contextual-1k.jsonl'),
        ('pii_synthetic', 'pii', root / 'data/raw/pii/synthetic-anonymizer-8k.jsonl'),
    ]
    return [(c, d, p) for (c, d, p) in candidates if p.exists()]


def load_domain_ops(domains_cfg_path: Path) -> tuple[list[dict], dict]:
    with domains_cfg_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['shared_operators'], cfg['domains']


def make_filter_analyze_cfg(dataset_path: Path, op: dict, export_path: Path, np: int, project_name: str) -> dict:
    return {
        'project_name': project_name,
        'dataset_path': str(dataset_path),
        'np': np,
        'export_path': str(export_path),
        'export_original_dataset': False,
        'process': [{op['name']: op.get('params', {})}],
    }


def make_mapper_process_cfg(dataset_path: Path, op: dict, export_path: Path, np: int, project_name: str) -> dict:
    return {
        'project_name': project_name,
        'dataset_path': str(dataset_path),
        'np': np,
        'export_path': str(export_path),
        'process': [{op['name']: op.get('params', {})}],
    }


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def write_summary(summary_csv: Path, rows: list[dict]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'corpus',
                'domain',
                'operator',
                'kind',
                'mode',
                'dataset_path',
                'input_count',
                'config_path',
                'export_path',
                'export_count',
                'stats_path',
                'stats_count',
                'mapper_output_discarded',
                'executed',
                'exit_code',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run per-operator Data-Juicer probing where each operator is evaluated against the original dataset.'
    )
    parser.add_argument('--domains-config', default='configs/domains.yaml')
    parser.add_argument('--config-dir', default='configs/dj_per_op_probe')
    parser.add_argument('--output-dir', default='data/processed/dj_per_op_probe')
    parser.add_argument('--summary-csv', default='data/processed/dj_per_op_probe/summary.csv')
    parser.add_argument('--dj-process-bin', default='dj-process')
    parser.add_argument('--dj-analyze-bin', default='dj-analyze')
    parser.add_argument('--np', type=int, default=4)
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument(
        '--mapper-max-records',
        type=int,
        default=20000,
        help='Cap mapper runs to head-N records per corpus to limit disk usage. Set <=0 to disable.',
    )
    parser.add_argument(
        '--filter-max-records',
        type=int,
        default=20000,
        help='Cap filter analyze runs to head-N records per corpus to limit disk/cache usage. Set <=0 to disable.',
    )
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--keep-mapper-output',
        action='store_true',
        help='Keep mapper process outputs on disk. Default behavior is discard-after-count to save space.',
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    domains_cfg_path = root / args.domains_config
    config_dir = root / args.config_dir
    output_dir = root / args.output_dir
    summary_csv = root / args.summary_csv
    process_bin = Path(args.dj_process_bin) if '/' in args.dj_process_bin else Path(args.dj_process_bin)
    analyze_bin = Path(args.dj_analyze_bin) if '/' in args.dj_analyze_bin else Path(args.dj_analyze_bin)

    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    shared_ops, domains = load_domain_ops(domains_cfg_path)
    mapping = build_corpus_domain_map(root)
    sample_root = root / 'data/raw/sampled'

    tasks: list[dict] = []
    for corpus, domain, dataset_path in mapping:
        domain_cfg = domains[domain]
        ops = shared_ops + domain_cfg['specific_operators']
        for op in ops:
            op_name = op['name']
            op_kind = op['kind']
            if op_kind == 'filter':
                op_dataset_path = dataset_path
                if args.filter_max_records and args.filter_max_records > 0:
                    total_records = count_jsonl(dataset_path)
                    if total_records > args.filter_max_records:
                        op_dataset_path = ensure_head_sample(
                            source_path=dataset_path,
                            sample_path=sample_root / f'{dataset_path.stem}-head{args.filter_max_records}.jsonl',
                            max_records=args.filter_max_records,
                        )
                cfg_path = config_dir / f'{corpus}__{domain}__{op_name}__analyze.yaml'
                export_path = output_dir / f'{corpus}__{domain}__{op_name}__analyze.jsonl'
                stats_path = output_dir / f'{corpus}__{domain}__{op_name}__analyze_stats.jsonl'
                payload = make_filter_analyze_cfg(
                    dataset_path=op_dataset_path,
                    op=op,
                    export_path=export_path,
                    np=args.np,
                    project_name=f'cdrbench-{corpus}-{domain}-{op_name}-analyze',
                )
                cmd = [str(analyze_bin), '--config', str(cfg_path)]
                mode = 'analyze'
            elif op_kind == 'mapper':
                mapper_dataset_path = dataset_path
                if args.mapper_max_records and args.mapper_max_records > 0:
                    total_records = count_jsonl(dataset_path)
                    if total_records > args.mapper_max_records:
                        mapper_dataset_path = ensure_head_sample(
                            source_path=dataset_path,
                            sample_path=sample_root / f'{dataset_path.stem}-head{args.mapper_max_records}.jsonl',
                            max_records=args.mapper_max_records,
                        )
                op_dataset_path = mapper_dataset_path

                cfg_path = config_dir / f'{corpus}__{domain}__{op_name}__process.yaml'
                export_path = output_dir / f'{corpus}__{domain}__{op_name}__process.jsonl'
                stats_path = output_dir / f'{corpus}__{domain}__{op_name}__process_stats.jsonl'
                payload = make_mapper_process_cfg(
                    dataset_path=mapper_dataset_path,
                    op=op,
                    export_path=export_path,
                    np=args.np,
                    project_name=f'cdrbench-{corpus}-{domain}-{op_name}-process',
                )
                cmd = [str(process_bin), '--config', str(cfg_path)]
                mode = 'process'
            else:
                continue

            with cfg_path.open('w', encoding='utf-8') as f:
                yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

            tasks.append(
                {
                    'corpus': corpus,
                    'domain': domain,
                    'operator': op_name,
                    'kind': op_kind,
                    'mode': mode,
                    'dataset_path': op_dataset_path,
                    'config_path': cfg_path,
                    'export_path': export_path,
                    'stats_path': stats_path,
                    'cmd': cmd,
                }
            )

    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    rows: list[dict] = []
    for i, task in enumerate(tasks, start=1):
        dataset_count = count_jsonl(task['dataset_path'])
        export_exists = task['export_path'].exists()
        stats_exists = task['stats_path'].exists()

        if args.execute and args.resume and (export_exists or stats_exists):
            exit_code = 0
            ran = False
        elif args.execute:
            log_path = output_dir / 'logs' / f"{task['corpus']}__{task['domain']}__{task['operator']}__{task['mode']}.log"
            print(f"[{i}/{len(tasks)}] running {task['kind']} {task['operator']} on {task['corpus']} ({task['mode']})")
            exit_code = run_command(task['cmd'], log_path)
            ran = True
        else:
            exit_code = -1
            ran = False

        export_count = count_jsonl(task['export_path']) if task['export_path'].exists() else 0
        stats_count = count_jsonl(task['stats_path']) if task['stats_path'].exists() else 0
        mapper_output_discarded = False

        if (
            args.execute
            and exit_code == 0
            and task['kind'] == 'mapper'
            and task['export_path'].exists()
            and not args.keep_mapper_output
        ):
            task['export_path'].unlink()
            mapper_output_discarded = True

        rows.append(
            {
                'corpus': task['corpus'],
                'domain': task['domain'],
                'operator': task['operator'],
                'kind': task['kind'],
                'mode': task['mode'],
                'dataset_path': str(task['dataset_path']),
                'input_count': dataset_count,
                'config_path': str(task['config_path']),
                'export_path': str(task['export_path']) if task['export_path'].exists() else '',
                'export_count': export_count,
                'stats_path': str(task['stats_path']) if task['stats_path'].exists() else '',
                'stats_count': stats_count,
                'mapper_output_discarded': mapper_output_discarded,
                'executed': ran,
                'exit_code': exit_code,
            }
        )

        # Write progress continuously so interrupted runs still leave a usable summary.
        write_summary(summary_csv, rows)

    print(f'generated tasks: {len(tasks)}')
    print(f'summary: {summary_csv}')


if __name__ == '__main__':
    main()
