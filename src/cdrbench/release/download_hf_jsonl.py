#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def load_manifest(manifest_path: Path) -> list[Path]:
    items: list[Path] = []
    with manifest_path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            items.append(Path(line))
    return items


def to_allow_pattern(rel: Path, target_subdir: str) -> str:
    rel_s = rel.as_posix()
    if rel_s.startswith('data/raw/'):
        suffix = rel_s[len('data/raw/') :]
    else:
        suffix = rel.name
    return f"{target_subdir.rstrip('/')}/{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description='Download curated JSONL snapshot from HF dataset to local data/raw layout.')
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--manifest', default='configs/release_jsonl_manifest.txt')
    parser.add_argument('--target-subdir', default='raw')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f'manifest not found: {manifest_path}')

    entries = load_manifest(manifest_path)
    if not entries:
        raise SystemExit('manifest is empty')

    allow_patterns = [to_allow_pattern(p, args.target_subdir) for p in entries]

    token = os.environ.get('HF_TOKEN')
    local_data_dir = repo_root / 'data'
    local_data_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type='dataset',
        allow_patterns=allow_patterns,
        local_dir=str(local_data_dir),
        local_dir_use_symlinks=False,
        token=token,
    )

    print(f'Downloaded {len(allow_patterns)} files from https://huggingface.co/datasets/{args.repo_id}')


if __name__ == '__main__':
    main()
