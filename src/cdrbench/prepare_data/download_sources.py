#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from cdrbench.config import load_domains_config
from cdrbench.fetchers import iter_domain_records, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description='Download small bootstrap corpora for CDR-Bench domains.')
    parser.add_argument('--config', default='configs/domains.yaml')
    parser.add_argument('--out-dir', default='data/raw')
    parser.add_argument('--domains', nargs='*', default=None)
    args = parser.parse_args()

    root = ROOT
    cfg = load_domains_config(root / args.config)
    selected = set(args.domains) if args.domains else None
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain, domain_cfg in cfg['domains'].items():
        if selected and domain not in selected:
            continue
        records = []
        for source_cfg in domain_cfg['sources']:
            records.extend(iter_domain_records(domain, source_cfg))
        out_path = out_dir / f'{domain}.jsonl'
        count = write_jsonl(out_path, records)
        print(f'{domain}: wrote {count} records -> {out_path}')


if __name__ == '__main__':
    main()
