#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import io
import json
import os
import re
import tarfile
import zlib
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator


CHARSET_RE = re.compile(r'charset=([^\s;]+)', re.IGNORECASE)
ARXIV_TEXT_SUFFIXES = {'.tex', '.ltx', '.sty', '.cls', '.bst', '.bib', '.bbl', '.txt'}
ARXIV_SKIP_SUFFIXES = {'.eps', '.ps', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp'}
ARXIV_MAIN_NAMES = {'main.tex', 'paper.tex', 'ms.tex', 'manuscript.tex', 'article.tex', 'source.tex'}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root / path


def write_jsonl(path: Path, records: Iterator[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, 'w', encoding='utf-8') as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    return count


def replace_jsonl(path: Path, records: Iterator[dict[str, object]]) -> int:
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    count = write_jsonl(tmp_path, records)
    os.replace(tmp_path, path)
    return count


def open_binary(path: Path) -> BinaryIO:
    if path.suffix == '.gz':
        return gzip.open(path, 'rb')
    return open(path, 'rb')


def parse_warc_headers(stream: BinaryIO) -> tuple[str, dict[str, str]] | None:
    while True:
        line = stream.readline()
        if not line:
            return None
        if line.strip():
            break

    if not line.startswith(b'WARC/'):
        return None

    version = line.decode('utf-8', errors='replace').strip()
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            break
        if line in (b'\n', b'\r\n'):
            break
        header_line = line.decode('utf-8', errors='replace').rstrip('\r\n')
        if ':' not in header_line:
            continue
        key, value = header_line.split(':', 1)
        headers[key.strip()] = value.strip()
    return version, headers


def iter_warc_records(path: Path) -> Iterator[tuple[dict[str, str], bytes]]:
    with open_binary(path) as stream:
        while True:
            parsed = parse_warc_headers(stream)
            if parsed is None:
                break
            _, headers = parsed
            try:
                content_length = int(headers.get('Content-Length', '0'))
            except ValueError:
                content_length = 0
            payload = stream.read(content_length)
            yield headers, payload


def parse_http_headers(payload: bytes) -> tuple[str, dict[str, str], bytes] | None:
    if b'\r\n\r\n' in payload:
        raw_headers, body = payload.split(b'\r\n\r\n', 1)
    elif b'\n\n' in payload:
        raw_headers, body = payload.split(b'\n\n', 1)
    else:
        return None

    lines = raw_headers.splitlines()
    if not lines:
        return None

    status_line = lines[0].decode('latin-1', errors='replace').strip()
    headers: dict[str, str] = {}
    for line in lines[1:]:
        header_line = line.decode('latin-1', errors='replace')
        if ':' not in header_line:
            continue
        key, value = header_line.split(':', 1)
        headers[key.strip()] = value.strip()
    return status_line, headers, body


def parse_status_code(status_line: str) -> int | None:
    parts = status_line.split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def dechunk_http_body(body: bytes) -> bytes:
    reader = io.BytesIO(body)
    out = bytearray()
    while True:
        size_line = reader.readline()
        if not size_line:
            break
        size_line = size_line.strip()
        if not size_line:
            continue
        try:
            chunk_size = int(size_line.split(b';', 1)[0], 16)
        except ValueError:
            return body
        if chunk_size == 0:
            break
        out.extend(reader.read(chunk_size))
        reader.readline()
    return bytes(out)


def decode_content_encoding(body: bytes, content_encoding: str) -> bytes | None:
    encoding = content_encoding.lower().strip()
    if not encoding or encoding == 'identity':
        return body
    if encoding in {'gzip', 'x-gzip'}:
        try:
            return gzip.decompress(body)
        except OSError:
            return None
    if encoding == 'deflate':
        try:
            return zlib.decompress(body)
        except zlib.error:
            try:
                return zlib.decompress(body, -zlib.MAX_WBITS)
            except zlib.error:
                return None
    if encoding == 'br':
        try:
            import brotli  # type: ignore
        except ImportError:
            return None
        try:
            return brotli.decompress(body)
        except Exception:
            return None
    return None


def detect_charset(content_type: str) -> str | None:
    match = CHARSET_RE.search(content_type)
    if match:
        return match.group(1).strip('"\'')
    return None


def decode_text(body: bytes, content_type: str) -> str | None:
    preferred = detect_charset(content_type)
    encodings = [enc for enc in [preferred, 'utf-8', 'cp1252', 'latin-1'] if enc]
    for encoding in encodings:
        try:
            return body.decode(encoding, errors='replace')
        except LookupError:
            continue
    return None


def is_probably_html(url: str, content_type: str, text: str) -> bool:
    content_type = content_type.lower()
    if 'text/html' in content_type or 'application/xhtml+xml' in content_type:
        return True
    if url.lower().endswith(('.html', '.htm', '.xhtml')):
        return True
    snippet = text[:4096].lower()
    return '<html' in snippet or '<!doctype html' in snippet


def iter_commoncrawl_records(
    inputs: list[Path],
    *,
    source_name: str,
    source_type: str,
    html_only: bool,
    status_allowlist: set[int],
    limit: int | None,
) -> Iterator[dict[str, object]]:
    record_idx = 0
    for path in inputs:
        for warc_headers, payload in iter_warc_records(path):
            if limit is not None and record_idx >= limit:
                return
            if warc_headers.get('WARC-Type') != 'response':
                continue

            parsed = parse_http_headers(payload)
            if parsed is None:
                continue
            status_line, http_headers, body = parsed
            status_code = parse_status_code(status_line)
            if status_code is None or status_code not in status_allowlist:
                continue

            transfer_encoding = http_headers.get('Transfer-Encoding', '')
            if 'chunked' in transfer_encoding.lower():
                body = dechunk_http_body(body)

            content_encoding = http_headers.get('Content-Encoding', '')
            decoded_body = decode_content_encoding(body, content_encoding)
            if decoded_body is None:
                continue

            content_type = http_headers.get('Content-Type', '')
            text = decode_text(decoded_body, content_type)
            if text is None or not text.strip():
                continue

            url = warc_headers.get('WARC-Target-URI')
            if not url:
                continue

            if html_only and not is_probably_html(url, content_type, text):
                continue

            yield {
                'id': f'{source_name}-{record_idx}',
                'source_name': source_name,
                'source_type': source_type,
                'url': url,
                'text': text,
                'meta': {
                    'status_code': status_code,
                    'output_mode': 'raw_html',
                    'source_file': path.name,
                    'warc_record_id': warc_headers.get('WARC-Record-ID'),
                    'warc_date': warc_headers.get('WARC-Date'),
                    'content_type': content_type,
                    'content_encoding': content_encoding or None,
                    'warc_ip_address': warc_headers.get('WARC-IP-Address'),
                },
            }
            record_idx += 1


def normalize_paragraph(text: str) -> str:
    return ' '.join(text.split())


def load_split_lookup(split_dir: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not split_dir.exists():
        return lookup
    for ids_path in sorted(split_dir.glob('*.ids')):
        stem = ids_path.stem
        parts = stem.split('_', 1)
        if len(parts) != 2:
            continue
        source_kind, split_name = parts
        with open(ids_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc_id = line.strip()
                if doc_id:
                    lookup[f'{source_kind}:{doc_id}'] = split_name
    return lookup


def iter_section_blocks(
    section: dict[str, object],
    *,
    include_headings: bool,
    skip_paragraphs: bool = False,
) -> Iterator[str]:
    title = section.get('section_title')
    if include_headings and isinstance(title, str):
        title = normalize_paragraph(title)
        if title:
            yield title

    if not skip_paragraphs:
        paragraphs = section.get('paragraphs', [])
        if isinstance(paragraphs, list):
            for paragraph in paragraphs:
                if isinstance(paragraph, str):
                    paragraph = normalize_paragraph(paragraph)
                    if paragraph:
                        yield paragraph

    subsections = section.get('subsections', [])
    if isinstance(subsections, list):
        for subsection in subsections:
            if isinstance(subsection, dict):
                yield from iter_section_blocks(subsection, include_headings=include_headings)


def flatten_crs_report(report_tree: dict[str, object], *, include_headings: bool) -> str:
    return '\n\n'.join(iter_section_blocks(report_tree, include_headings=include_headings)).strip()


def flatten_gao_report(
    report_sections: Iterable[dict[str, object]],
    *,
    include_headings: bool,
    drop_letter_paragraphs: bool,
) -> str:
    blocks: list[str] = []
    for section in report_sections:
        title = section.get('section_title')
        skip_paragraphs = bool(drop_letter_paragraphs and title == 'Letter')
        blocks.extend(
            iter_section_blocks(
                section,
                include_headings=include_headings,
                skip_paragraphs=skip_paragraphs,
            )
        )
    return '\n\n'.join(blocks).strip()


def build_govreport_text(
    payload: dict[str, object],
    source_kind: str,
    *,
    include_headings: bool,
    prepend_title: bool,
    drop_gao_letter_paragraphs: bool,
) -> str:
    if source_kind == 'crs':
        report_tree = payload.get('reports')
        if not isinstance(report_tree, dict):
            return ''
        body = flatten_crs_report(report_tree, include_headings=include_headings)
    else:
        report_sections = payload.get('report')
        if not isinstance(report_sections, list):
            return ''
        body = flatten_gao_report(
            report_sections,
            include_headings=include_headings,
            drop_letter_paragraphs=drop_gao_letter_paragraphs,
        )

    title = payload.get('title')
    if prepend_title and isinstance(title, str):
        title = normalize_paragraph(title)
        if title:
            return f'{title}\n\n{body}'.strip()
    return body


def iter_govreport_source_files(in_dir: Path) -> Iterator[tuple[str, Path]]:
    for source_kind in ('crs', 'gao'):
        source_dir = in_dir / source_kind
        for path in sorted(source_dir.glob('*.json')):
            yield source_kind, path


def iter_govreport_records(
    in_dir: Path,
    *,
    source_name: str,
    source_type: str,
    include_headings: bool,
    prepend_title: bool,
    drop_gao_letter_paragraphs: bool,
    split_lookup: dict[str, str],
    limit: int | None,
) -> Iterator[dict[str, object]]:
    count = 0
    for source_kind, path in iter_govreport_source_files(in_dir):
        if limit is not None and count >= limit:
            break

        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        doc_id = payload.get('id') or path.stem
        if not isinstance(doc_id, str):
            doc_id = path.stem

        text = build_govreport_text(
            payload,
            source_kind,
            include_headings=include_headings,
            prepend_title=prepend_title,
            drop_gao_letter_paragraphs=drop_gao_letter_paragraphs,
        )
        if not text:
            continue

        split_name = split_lookup.get(f'{source_kind}:{doc_id}')
        meta = {
            'dataset_dir': str(in_dir),
            'source_kind': source_kind,
            'doc_id': doc_id,
            'title': payload.get('title'),
            'url': payload.get('url'),
            'released_date': payload.get('released_date'),
            'published_date': payload.get('published_date'),
            'split': split_name,
            'output_mode': 'sectioned_plain_text',
            'include_headings': include_headings,
            'prepend_title': prepend_title,
            'drop_gao_letter_paragraphs': drop_gao_letter_paragraphs,
        }

        summary_field = 'summary' if source_kind == 'crs' else 'highlight'
        summary_value = payload.get(summary_field)
        if isinstance(summary_value, list):
            meta['summary_field'] = summary_field
            meta['summary_items'] = len(summary_value)

        yield {
            'id': f'{source_name}-{source_kind}-{doc_id}',
            'source_name': source_name,
            'source_type': source_type,
            'text': text,
            'meta': meta,
        }
        count += 1


def iter_pii_records(
    in_path: Path,
    *,
    source_name: str,
    source_type: str,
) -> Iterator[dict[str, object]]:
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            original_id = payload.get('id')
            privacy_mask = payload.get('privacy_mask')
            yield {
                'id': f'{source_name}-{original_id}',
                'source_name': source_name,
                'source_type': source_type,
                'url': None,
                'text': payload.get('source_text', ''),
                'meta': {
                    'original_id': original_id,
                    'language': payload.get('language'),
                    'split': payload.get('set'),
                    'target_text': payload.get('target_text'),
                    'privacy_mask': privacy_mask,
                    'privacy_mask_count': len(privacy_mask) if isinstance(privacy_mask, list) else None,
                    'span_labels': payload.get('span_labels'),
                    'mbert_text_tokens': payload.get('mbert_text_tokens'),
                    'mbert_bio_labels': payload.get('mbert_bio_labels'),
                    'output_mode': 'plain_text',
                    'text_field': 'source_text',
                    'target_field': 'target_text',
                },
            }


def iter_docpii_records(
    in_path: Path,
    *,
    source_name: str,
    source_type: str,
    limit: int | None,
) -> Iterator[dict[str, object]]:
    with open(in_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    uid_map = payload.get('uid', {})
    text_map = payload.get('text', {})
    entities_map = payload.get('entities', {})
    entity_count_map = payload.get('entity_count', {})
    document_type_map = payload.get('document_type', {})
    domain_map = payload.get('domain', {})
    document_description_map = payload.get('document_description', {})
    redaction_query_map = payload.get('redaction_query', {})

    if not isinstance(uid_map, dict) or not isinstance(text_map, dict):
        return

    count = 0
    for row_key in sorted(uid_map.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        if limit is not None and count >= limit:
            break
        text = text_map.get(row_key)
        if not isinstance(text, str) or not text.strip():
            continue

        uid = uid_map.get(row_key)
        if not isinstance(uid, str) or not uid:
            uid = str(row_key)

        entities = entities_map.get(row_key)
        if not isinstance(entities, list):
            entities = None

        entity_count = entity_count_map.get(row_key)
        if not isinstance(entity_count, int):
            entity_count = len(entities) if isinstance(entities, list) else None

        yield {
            'id': f'{source_name}-{uid}',
            'source_name': source_name,
            'source_type': source_type,
            'url': None,
            'text': text,
            'meta': {
                'uid': uid,
                'row_key': row_key,
                'entities': entities,
                'entity_count': entity_count,
                'document_type': document_type_map.get(row_key),
                'domain': domain_map.get(row_key),
                'document_description': document_description_map.get(row_key),
                'redaction_query': redaction_query_map.get(row_key),
                'output_mode': 'plain_text',
                'text_field': 'text',
            },
        }
        count += 1


def iter_synthetic_text_anonymizer_records(
    in_path: Path,
    *,
    source_name: str,
    source_type: str,
    limit: int | None,
) -> Iterator[dict[str, object]]:
    with open(in_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        count = 0
        for row_idx, row in enumerate(reader):
            if limit is not None and count >= limit:
                break
            text = row.get('text')
            if not isinstance(text, str) or not text.strip():
                continue

            labels_raw = row.get('labels')
            labels = None
            if isinstance(labels_raw, str) and labels_raw.strip():
                try:
                    parsed = ast.literal_eval(labels_raw)
                    if isinstance(parsed, list):
                        labels = parsed
                except (ValueError, SyntaxError):
                    labels = None

            yield {
                'id': f'{source_name}-{row_idx}',
                'source_name': source_name,
                'source_type': source_type,
                'url': None,
                'text': text,
                'meta': {
                    'row_idx': row_idx,
                    'labels': labels,
                    'label_count': len(labels) if isinstance(labels, list) else None,
                    'output_mode': 'plain_text',
                    'text_field': 'text',
                },
            }
            count += 1


def decode_text_bytes(data: bytes) -> str:
    for encoding in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return data.decode(encoding, errors='replace')
        except LookupError:
            continue
    return data.decode('utf-8', errors='replace')


def normalize_arxiv_abs_id(raw_id: str) -> str:
    if '/' in raw_id:
        return raw_id
    match = re.match(r'^([A-Za-z.\-]+)(\d{7})$', raw_id)
    if match:
        return f'{match.group(1)}/{match.group(2)}'
    return raw_id


def score_arxiv_member(name: str, text: str) -> tuple[int, int]:
    lower_name = Path(name).name.lower()
    lower_text = text[:4000].lower()
    score = 0
    if Path(lower_name).suffix in {'.tex', '.ltx'}:
        score += 100
    if lower_name in ARXIV_MAIN_NAMES:
        score += 40
    if '\\documentclass' in lower_text or '\\documentstyle' in lower_text:
        score += 80
    if '\\begin{document}' in lower_text:
        score += 60
    return score, len(text)


def extract_arxiv_archive(path: Path) -> tuple[str, dict[str, object]] | None:
    archive_id = path.stem
    try:
        with tarfile.open(path, mode='r:gz') as tf:
            candidate_texts: list[tuple[tuple[int, int], str, str]] = []
            skipped_members = 0
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                member_name = member.name
                suffix = Path(member_name).suffix.lower()
                if suffix in ARXIV_SKIP_SUFFIXES:
                    skipped_members += 1
                    continue
                if suffix not in ARXIV_TEXT_SUFFIXES:
                    skipped_members += 1
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                text = decode_text_bytes(extracted.read())
                if not text.strip():
                    continue
                candidate_texts.append((score_arxiv_member(member_name, text), member_name, text))
    except tarfile.ReadError:
        with gzip.open(path, 'rb') as f:
            text = decode_text_bytes(f.read())
        if not text.strip():
            return None
        return text, {
            'archive_file': path.name,
            'archive_kind': 'plain_gzip_text',
            'arxiv_id': archive_id,
            'arxiv_abs_id': normalize_arxiv_abs_id(archive_id),
            'selected_source_file': path.name,
            'candidate_source_files': [path.name],
            'candidate_source_file_count': 1,
            'skipped_member_count': 0,
            'output_mode': 'raw_latex_source',
        }

    if not candidate_texts:
        return None

    candidate_texts.sort(key=lambda item: (item[0][0], item[0][1], item[1]))
    _, selected_name, selected_text = candidate_texts[-1]
    candidate_files = [name for _, name, _ in candidate_texts]
    return selected_text, {
        'archive_file': path.name,
        'archive_kind': 'tar_gzip',
        'arxiv_id': archive_id,
        'arxiv_abs_id': normalize_arxiv_abs_id(archive_id),
        'selected_source_file': selected_name,
        'candidate_source_files': candidate_files,
        'candidate_source_file_count': len(candidate_files),
        'skipped_member_count': skipped_members,
        'output_mode': 'raw_latex_source',
    }


def iter_arxiv_records(
    in_dir: Path,
    *,
    source_name: str,
    source_type: str,
    limit: int | None,
) -> Iterator[dict[str, object]]:
    count = 0
    for path in sorted(in_dir.iterdir()):
        if limit is not None and count >= limit:
            break
        if path.suffix.lower() != '.gz':
            continue
        extracted = extract_arxiv_archive(path)
        if extracted is None:
            continue
        text, meta = extracted
        arxiv_id = meta['arxiv_id']
        yield {
            'id': f'{source_name}-{arxiv_id}',
            'source_name': source_name,
            'source_type': source_type,
            'url': f"https://arxiv.org/abs/{meta['arxiv_abs_id']}",
            'text': text,
            'meta': {
                **meta,
                'archive_dir': str(in_dir),
            },
        }
        count += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert local corpora into the unified CDR-Bench raw schema.')
    subparsers = parser.add_subparsers(dest='kind', required=True)

    commoncrawl = subparsers.add_parser('commoncrawl', help='Convert Common Crawl WARC files')
    commoncrawl.add_argument('inputs', nargs='+', help='Input .warc or .warc.gz paths')
    commoncrawl.add_argument('--out', default='data/raw/commoncrawl/cc-40k.jsonl')
    commoncrawl.add_argument('--source-name', default='commoncrawl')
    commoncrawl.add_argument('--source-type', default='warc')
    commoncrawl.add_argument('--keep-non-html', action='store_true')
    commoncrawl.add_argument('--status-allowlist', nargs='*', type=int, default=[200])
    commoncrawl.add_argument('--limit', type=int, default=None)

    govreport = subparsers.add_parser('govreport', help='Convert local GovReport JSON files')
    govreport.add_argument('--in-dir', default='data/raw/govreport/gov-report')
    govreport.add_argument('--out', default='data/raw/govreport/govreport-20k.jsonl')
    govreport.add_argument('--source-name', default='govreport')
    govreport.add_argument('--source-type', default='local_govreport')
    govreport.add_argument('--limit', type=int, default=None)
    govreport.add_argument('--no-headings', action='store_true')
    govreport.add_argument('--prepend-title', action='store_true')
    govreport.add_argument('--keep-gao-letter-paragraphs', action='store_true')

    pii = subparsers.add_parser('pii', help='Convert PII JSONL')
    pii.add_argument('--in', dest='in_path', default='data/raw/pii/pii-43k.jsonl')
    pii.add_argument('--out', dest='out_path', default='data/raw/pii/pii-43k.jsonl')
    pii.add_argument('--source-name', default='pii-43k')
    pii.add_argument('--source-type', default='local_pii_redaction')

    docpii = subparsers.add_parser('pii_docpii', help='Convert external DocPII benchmark JSON')
    docpii.add_argument(
        '--in',
        dest='in_path',
        default='data/raw/external_benchmarks/pii/docpii-redaction-benchmark/docPII-contextual-redaction-benchmark.json',
    )
    docpii.add_argument('--out', dest='out_path', default='data/raw/pii/docpii-contextual-1k.jsonl')
    docpii.add_argument('--source-name', default='docpii-contextual')
    docpii.add_argument('--source-type', default='external_docpii_redaction')
    docpii.add_argument('--limit', type=int, default=None)

    synthetic = subparsers.add_parser('pii_synthetic', help='Convert external synthetic text anonymizer CSV')
    synthetic.add_argument(
        '--in',
        dest='in_path',
        default='data/raw/external_benchmarks/pii/synthetic-text-anonymizer-dataset-v1/data/data.csv',
    )
    synthetic.add_argument('--out', dest='out_path', default='data/raw/pii/synthetic-anonymizer-8k.jsonl')
    synthetic.add_argument('--source-name', default='synthetic-anonymizer-v1')
    synthetic.add_argument('--source-type', default='external_synthetic_pii_ner')
    synthetic.add_argument('--limit', type=int, default=None)

    arxiv = subparsers.add_parser('arxiv', help='Convert local arXiv source archives')
    arxiv.add_argument('--in-dir', default='data/raw/arxiv/0001')
    arxiv.add_argument('--out', default='data/raw/arxiv/arxiv-0001.jsonl')
    arxiv.add_argument('--source-name', default='arxiv-0001')
    arxiv.add_argument('--source-type', default='local_arxiv_source')
    arxiv.add_argument('--limit', type=int, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = repo_root()

    if args.kind == 'commoncrawl':
        out_path = resolve_path(root, args.out)
        inputs = [resolve_path(root, raw_path) for raw_path in args.inputs]
        count = replace_jsonl(
            out_path,
            iter_commoncrawl_records(
                inputs,
                source_name=args.source_name,
                source_type=args.source_type,
                html_only=not args.keep_non_html,
                status_allowlist=set(args.status_allowlist),
                limit=args.limit,
            ),
        )
    elif args.kind == 'govreport':
        in_dir = resolve_path(root, args.in_dir)
        out_path = resolve_path(root, args.out)
        split_lookup = load_split_lookup(in_dir / 'split_ids')
        count = replace_jsonl(
            out_path,
            iter_govreport_records(
                in_dir,
                source_name=args.source_name,
                source_type=args.source_type,
                include_headings=not args.no_headings,
                prepend_title=args.prepend_title,
                drop_gao_letter_paragraphs=not args.keep_gao_letter_paragraphs,
                split_lookup=split_lookup,
                limit=args.limit,
            ),
        )
    elif args.kind == 'pii':
        in_path = resolve_path(root, args.in_path)
        out_path = resolve_path(root, args.out_path)
        count = replace_jsonl(
            out_path,
            iter_pii_records(
                in_path,
                source_name=args.source_name,
                source_type=args.source_type,
            ),
        )
    elif args.kind == 'pii_docpii':
        in_path = resolve_path(root, args.in_path)
        out_path = resolve_path(root, args.out_path)
        count = replace_jsonl(
            out_path,
            iter_docpii_records(
                in_path,
                source_name=args.source_name,
                source_type=args.source_type,
                limit=args.limit,
            ),
        )
    elif args.kind == 'pii_synthetic':
        in_path = resolve_path(root, args.in_path)
        out_path = resolve_path(root, args.out_path)
        count = replace_jsonl(
            out_path,
            iter_synthetic_text_anonymizer_records(
                in_path,
                source_name=args.source_name,
                source_type=args.source_type,
                limit=args.limit,
            ),
        )
    elif args.kind == 'arxiv':
        in_dir = resolve_path(root, args.in_dir)
        out_path = resolve_path(root, args.out)
        count = replace_jsonl(
            out_path,
            iter_arxiv_records(
                in_dir,
                source_name=args.source_name,
                source_type=args.source_type,
                limit=args.limit,
            ),
        )
    else:
        raise ValueError(f'Unsupported conversion kind: {args.kind}')

    print(f'wrote {count} records -> {out_path}')


if __name__ == '__main__':
    main()
