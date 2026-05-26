from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import requests
from bs4 import BeautifulSoup
from datasets import load_dataset

USER_AGENT = 'CDRBenchBootstrap/0.1 (+https://github.com/datajuicer/data-juicer)'


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_url(url: str, timeout: int = 30) -> requests.Response:
    return requests.get(url, timeout=timeout, headers={'User-Agent': USER_AGENT})


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text('\n')
    lines = [line.strip() for line in text.splitlines()]
    return '\n'.join(line for line in lines if line)


def iter_url_source_records(domain: str, source_cfg: Dict[str, object]) -> Iterator[Dict[str, object]]:
    output_mode = str(source_cfg.get('output_mode', 'raw_html'))
    for idx, url in enumerate(source_cfg['urls']):
        resp = fetch_url(url)
        resp.raise_for_status()
        html = resp.text
        text = html if output_mode == 'raw_html' else html_to_text(html)
        yield {
            'id': f"{domain}-{source_cfg['name']}-{idx}",
            'domain': domain,
            'source_name': source_cfg['name'],
            'source_type': source_cfg['type'],
            'url': url,
            'text': text,
            'meta': {'status_code': resp.status_code, 'output_mode': output_mode},
        }


def iter_hf_dataset_records(domain: str, source_cfg: Dict[str, object]) -> Iterator[Dict[str, object]]:
    dataset_name = str(source_cfg['dataset'])
    split = str(source_cfg.get('split', 'train'))
    text_field = str(source_cfg['text_field'])
    sample_limit = int(source_cfg.get('sample_limit', 20))
    streaming = bool(source_cfg.get('streaming', True))
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    for idx, row in enumerate(ds):
        if idx >= sample_limit:
            break
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        yield {
            'id': f"{domain}-{source_cfg['name']}-{idx}",
            'domain': domain,
            'source_name': source_cfg['name'],
            'source_type': source_cfg['type'],
            'text': text,
            'meta': {'dataset': dataset_name, 'split': split, 'row_idx': idx},
        }


def choose_main_tex(tex_files: List[tuple[str, str]]) -> tuple[str, str] | None:
    if not tex_files:
        return None
    with_document = [item for item in tex_files if '\\begin{document}' in item[1]]
    candidates = with_document if with_document else tex_files
    return max(candidates, key=lambda item: len(item[1]))


def iter_arxiv_eprint_records(domain: str, source_cfg: Dict[str, object]) -> Iterator[Dict[str, object]]:
    for arxiv_id in source_cfg['arxiv_ids']:
        url = f'https://export.arxiv.org/e-print/{arxiv_id}'
        resp = fetch_url(url, timeout=60)
        resp.raise_for_status()
        tex_files: List[tuple[str, str]] = []
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode='r:*') as tf:
            for member in tf.getmembers():
                if not member.isfile() or not member.name.endswith('.tex'):
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                try:
                    content = extracted.read().decode('utf-8', errors='ignore')
                except Exception:
                    continue
                if content.strip():
                    tex_files.append((member.name, content))
        chosen = choose_main_tex(tex_files)
        if chosen is None:
            continue
        file_name, text = chosen
        yield {
            'id': f"{domain}-{source_cfg['name']}-{arxiv_id}",
            'domain': domain,
            'source_name': source_cfg['name'],
            'source_type': source_cfg['type'],
            'text': text,
            'meta': {'arxiv_id': arxiv_id, 'main_tex': file_name, 'num_tex_files': len(tex_files)},
        }


def iter_domain_records(domain: str, source_cfg: Dict[str, object]) -> Iterator[Dict[str, object]]:
    source_type = source_cfg['type']
    if source_type == 'url_list':
        yield from iter_url_source_records(domain, source_cfg)
    elif source_type == 'hf_dataset':
        yield from iter_hf_dataset_records(domain, source_cfg)
    elif source_type == 'arxiv_eprint':
        yield from iter_arxiv_eprint_records(domain, source_cfg)
    else:
        raise ValueError(f'Unsupported source type: {source_type}')


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    ensure_parent(path)
    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    return count
