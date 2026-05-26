#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator
from urllib.parse import quote
import xml.etree.ElementTree as ET


def local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def direct_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem:
        if local_name(child.tag) == name:
            return child
    return None


def direct_child_text(elem: ET.Element, name: str) -> str | None:
    child = direct_child(elem, name)
    if child is None:
        return None
    return child.text


def page_url(title: str) -> str:
    return 'https://en.wikipedia.org/wiki/' + quote(title.replace(' ', '_'), safe=':_()/')


def iter_page_records(xml_path: Path, domain: str, source_name: str) -> Iterator[dict[str, object]]:
    context = ET.iterparse(xml_path, events=('start', 'end'))
    _, root = next(context)
    page_idx = 0

    for event, elem in context:
        if event != 'end' or local_name(elem.tag) != 'page':
            continue

        title = direct_child_text(elem, 'title') or ''
        namespace = direct_child_text(elem, 'ns')
        page_id = direct_child_text(elem, 'id')
        redirect = direct_child(elem, 'redirect')
        redirect_title = redirect.get('title') if redirect is not None else None

        revision = direct_child(elem, 'revision')
        revision_id = None
        timestamp = None
        model = None
        text_format = None
        text_bytes = None
        text = ''

        if revision is not None:
            revision_id = direct_child_text(revision, 'id')
            timestamp = direct_child_text(revision, 'timestamp')
            model = direct_child_text(revision, 'model')
            text_format = direct_child_text(revision, 'format')
            text_node = direct_child(revision, 'text')
            if text_node is not None:
                text = text_node.text or ''
                text_bytes = text_node.get('bytes')

        yield {
            'id': f'{domain}-{source_name}-{page_idx}',
            'domain': domain,
            'source_name': source_name,
            'source_type': 'wikipedia_xml_dump',
            'url': page_url(title) if title else None,
            'text': text,
            'meta': {
                'dump_file': xml_path.name,
                'title': title,
                'page_id': page_id,
                'revision_id': revision_id,
                'namespace': namespace,
                'redirect_title': redirect_title,
                'timestamp': timestamp,
                'model': model,
                'format': text_format,
                'text_bytes': int(text_bytes) if text_bytes and text_bytes.isdigit() else text_bytes,
                'output_mode': 'raw_wikitext',
            },
        }
        page_idx += 1

        elem.clear()
        root.clear()


def write_jsonl(out_path: Path, records: Iterator[dict[str, object]]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    return count


def default_output_name(xml_path: Path) -> str:
    return f'{xml_path.name}.jsonl'


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract individual English Wikipedia pages from XML dump parts.')
    parser.add_argument('inputs', nargs='+', help='Input XML dump part paths')
    parser.add_argument('--out-dir', default='data/raw/enwiki', help='Directory to store extracted JSONL files')
    parser.add_argument('--domain', default='enwiki', help='Domain field to emit in output records')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    out_dir = root / args.out_dir

    for raw_input in args.inputs:
        xml_path = Path(raw_input)
        if not xml_path.is_absolute():
            xml_path = root / xml_path
        source_name = xml_path.name.replace('.xml-', '_').replace('.', '_')
        out_path = out_dir / default_output_name(xml_path)
        count = write_jsonl(out_path, iter_page_records(xml_path, args.domain, source_name))
        print(f'{xml_path.name}: wrote {count} records -> {out_path}')


if __name__ == '__main__':
    main()
