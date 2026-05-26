from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd

from cdrbench.dj_operator_loader import get_operator_kind
from cdrbench.support_scan import normalize_record, run_filter, run_mapper


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    preview = line[:200]
                    raise ValueError(
                        f'Invalid JSONL record in {path} at line {lineno}: {exc.msg}. '
                        f'Line preview: {preview!r}'
                    ) from exc


def _stable_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _execution_key(op_name: str, params: Dict[str, Any]) -> str:
    if not params:
        return op_name
    params_blob = _stable_json(params)
    suffix = hashlib.sha1(params_blob.encode('utf-8')).hexdigest()[:10]
    return f'{op_name}__{suffix}'


def _domain_operator_groups(domains_cfg: Dict[str, Any], domain_cfg: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    global_shared = list(domains_cfg.get('shared_operators', []))
    domain_shared = list(domain_cfg.get('shared_operators', []))
    domain_specific = list(domain_cfg.get('specific_operators', []))

    if domain_specific or domain_shared or global_shared:
        shared = [*global_shared, *domain_shared]
        specific = domain_specific
        return shared, specific

    # Backward compatibility for older config shape.
    return [], list(domain_cfg.get('operators', []))


def build_domain_execution_plan(domains_cfg: Dict[str, Any]) -> Dict[str, Any]:
    execution_variants_by_key: Dict[str, Dict[str, Any]] = {}
    domains_by_execution_key: Dict[str, set[str]] = defaultdict(set)
    domain_profiles: Dict[str, Dict[str, Any]] = {}
    execution_order: List[str] = []
    domain_order: List[str] = []

    for domain_name, domain_cfg in domains_cfg.get('domains', {}).items():
        domain_order.append(domain_name)
        mapper_keys: List[str] = []
        filter_keys: List[str] = []
        configured_shared_mapper_keys: List[str] = []
        configured_specific_mapper_keys: List[str] = []
        seen_keys: set[str] = set()
        shared_ops, specific_ops = _domain_operator_groups(domains_cfg, domain_cfg)
        for scope, op_cfgs in (('shared', shared_ops), ('specific', specific_ops)):
            for op_cfg in op_cfgs:
                op_name = op_cfg['name']
                kind = op_cfg.get('kind') or get_operator_kind(op_name)
                params = dict(op_cfg.get('params', {}))
                key = _execution_key(op_name, params)
                if key not in execution_variants_by_key:
                    execution_variants_by_key[key] = {
                        'key': key,
                        'name': op_name,
                        'kind': kind,
                        'params': params,
                    }
                    execution_order.append(key)
                domains_by_execution_key[key].add(domain_name)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if kind == 'mapper':
                    mapper_keys.append(key)
                    if scope == 'shared':
                        configured_shared_mapper_keys.append(key)
                    else:
                        configured_specific_mapper_keys.append(key)
                else:
                    filter_keys.append(key)
        domain_profiles[domain_name] = {
            'mapper_keys': mapper_keys,
            'filter_keys': filter_keys,
            'configured_shared_mapper_keys': configured_shared_mapper_keys,
            'configured_specific_mapper_keys': configured_specific_mapper_keys,
        }

    for domain_name, profile in domain_profiles.items():
        unique_mapper_keys = [
            key for key in profile['mapper_keys'] if len(domains_by_execution_key.get(key, set())) == 1
        ]
        shared_mapper_keys = [key for key in profile['mapper_keys'] if key not in unique_mapper_keys]
        profile['unique_mapper_keys'] = unique_mapper_keys
        profile['shared_mapper_keys'] = shared_mapper_keys

    return {
        'execution_variants': [execution_variants_by_key[key] for key in execution_order],
        'execution_variants_by_key': execution_variants_by_key,
        'domain_order': domain_order,
        'assignment_domains': domain_order,
        'domains_by_execution_key': {key: sorted(values) for key, values in domains_by_execution_key.items()},
        'domain_profiles': domain_profiles,
    }


def domain_operator_catalog_frame(plan: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for variant in plan['execution_variants']:
        domains = plan['domains_by_execution_key'].get(variant['key'], [])
        for domain_name in domains:
            rows.append(
                {
                    'domain': domain_name,
                    'execution_key': variant['key'],
                    'operator': variant['name'],
                    'kind': variant['kind'],
                    'params': _stable_json(variant.get('params', {})),
                    'domain_count': len(domains),
                    'is_unique_to_domain': len(domains) == 1,
                }
            )
    return pd.DataFrame(rows)


def _rank_domain_candidates(
    operator_results: Dict[str, Dict[str, Any]],
    plan: Dict[str, Any],
    preferred_domain: str | None = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    variants_by_key = plan['execution_variants_by_key']
    domain_rank = {domain_name: idx for idx, domain_name in enumerate(plan['domain_order'])}

    for domain_name, profile in plan['domain_profiles'].items():
        matched_mapper_keys = [key for key in profile['mapper_keys'] if operator_results.get(key, {}).get('active')]
        matched_configured_specific_mapper_keys = [
            key for key in matched_mapper_keys if key in profile.get('configured_specific_mapper_keys', [])
        ]
        matched_configured_shared_mapper_keys = [
            key for key in matched_mapper_keys if key in profile.get('configured_shared_mapper_keys', [])
        ]
        matched_unique_mapper_keys = [key for key in matched_mapper_keys if key in profile['unique_mapper_keys']]
        matched_shared_mapper_keys = [key for key in matched_mapper_keys if key not in profile['unique_mapper_keys']]
        matched_configured_specific_mapper_names = sorted(
            {variants_by_key[key]['name'] for key in matched_configured_specific_mapper_keys}
        )
        matched_configured_shared_mapper_names = sorted(
            {variants_by_key[key]['name'] for key in matched_configured_shared_mapper_keys}
        )
        matched_mapper_names = sorted({variants_by_key[key]['name'] for key in matched_mapper_keys})
        matched_unique_mapper_names = sorted({variants_by_key[key]['name'] for key in matched_unique_mapper_keys})
        matched_shared_mapper_names = sorted({variants_by_key[key]['name'] for key in matched_shared_mapper_keys})
        passed_filter_keys = [key for key in profile['filter_keys'] if operator_results.get(key, {}).get('keep') is True]
        candidates.append(
            {
                'domain': domain_name,
                'matched_specific_mapper_count': len(matched_configured_specific_mapper_names),
                'matched_specific_mapper_names': matched_configured_specific_mapper_names,
                'matched_specific_mapper_keys': matched_configured_specific_mapper_keys,
                'matched_configured_shared_mapper_count': len(matched_configured_shared_mapper_names),
                'matched_configured_shared_mapper_names': matched_configured_shared_mapper_names,
                'matched_configured_shared_mapper_keys': matched_configured_shared_mapper_keys,
                'matched_mapper_count': len(matched_mapper_names),
                'matched_mapper_names': matched_mapper_names,
                'matched_mapper_keys': matched_mapper_keys,
                'matched_unique_mapper_count': len(matched_unique_mapper_names),
                'matched_unique_mapper_names': matched_unique_mapper_names,
                'matched_unique_mapper_keys': matched_unique_mapper_keys,
                'matched_shared_mapper_count': len(matched_shared_mapper_names),
                'matched_shared_mapper_names': matched_shared_mapper_names,
                'matched_shared_mapper_keys': matched_shared_mapper_keys,
                'passed_filter_count': len(passed_filter_keys),
                'passed_filter_keys': passed_filter_keys,
            }
        )

    candidates.sort(
        key=lambda row: (
            -row['matched_specific_mapper_count'],
            -row['matched_unique_mapper_count'],
            -(1 if preferred_domain and row['domain'] == preferred_domain else 0),
            -row['matched_mapper_count'],
            -row['matched_shared_mapper_count'],
            -row['passed_filter_count'],
            domain_rank.get(row['domain'], 10**9),
        )
    )
    return candidates


def _build_filtered_record(
    record: Dict[str, Any],
    corpus_name: str,
    assigned_domain: str,
    tag_payload: Dict[str, Any],
) -> Dict[str, Any]:
    filtered = dict(record)
    raw_meta = filtered.get('meta')
    if isinstance(raw_meta, dict):
        meta = dict(raw_meta)
    elif raw_meta is None:
        meta = {}
    else:
        meta = {'raw_meta': raw_meta}

    meta['cdrbench_domain_labeling'] = {
        'source_corpus': corpus_name,
        'original_domain': record.get('domain'),
        'assigned_domain': assigned_domain,
        'active_mapper_count': tag_payload['active_mapper_count'],
        'active_mapper_names': tag_payload['active_mapper_names'],
        'unique_active_mapper_count': tag_payload.get('unique_active_mapper_count', 0),
        'unique_active_mapper_names': tag_payload.get('unique_active_mapper_names', []),
        'top_domain_candidates': tag_payload['domain_candidates'][:3],
    }
    filtered['meta'] = meta
    filtered['domain'] = assigned_domain
    return filtered


def label_record(
    record: Dict[str, Any],
    raw_path: Path,
    corpus_name: str,
    plan: Dict[str, Any],
    field_map: Dict[str, str] | None = None,
    defaults: Dict[str, Any] | None = None,
    min_active_mappers: int = 2,
    max_text_length: int | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    normalized = normalize_record(
        record,
        raw_path,
        dataset_name=corpus_name,
        field_map=field_map,
        defaults=defaults,
    )

    if max_text_length is not None and len(normalized['text']) > max_text_length:
        tag_payload = {
            'id': normalized['id'],
            'corpus': corpus_name,
            'source_name': normalized['source_name'],
            'original_domain': record.get('domain'),
            'best_domain_candidate': None,
            'assigned_domain': None,
            'text_length': len(normalized['text']),
            'active_mapper_count': 0,
            'active_mapper_names': [],
            'unique_active_mapper_count': 0,
            'unique_active_mapper_names': [],
            'keep': False,
            'domain_candidates': [],
            'operators': {},
            'skipped_long_text': True,
            'skip_reason': f'text_length>{max_text_length}',
        }
        return tag_payload, None

    operator_results: Dict[str, Dict[str, Any]] = {}
    for variant in plan['execution_variants']:
        params = dict(variant.get('params', {}))
        if variant['kind'] == 'mapper':
            result = run_mapper(variant['name'], normalized['text'], params, suffix=normalized['suffix'])
        else:
            result = run_filter(variant['name'], normalized['text'], params, suffix=normalized['suffix'])
        operator_results[variant['key']] = result

    variants_by_key = plan['execution_variants_by_key']
    active_mapper_keys = [
        key
        for key, result in operator_results.items()
        if variants_by_key[key]['kind'] == 'mapper' and result.get('active')
    ]
    active_mapper_names = sorted({variants_by_key[key]['name'] for key in active_mapper_keys})
    unique_active_mapper_names = sorted(
        {
            variants_by_key[key]['name']
            for profile in plan['domain_profiles'].values()
            for key in profile.get('unique_mapper_keys', [])
            if key in active_mapper_keys
        }
    )
    preferred_domain = record.get('domain')
    if preferred_domain not in plan['assignment_domains']:
        preferred_domain = None
    domain_candidates = _rank_domain_candidates(operator_results, plan, preferred_domain=preferred_domain)
    best_domain_candidate = None
    if domain_candidates and domain_candidates[0]['matched_mapper_count'] > 0:
        best_domain_candidate = domain_candidates[0]['domain']
    assigned_domain = None
    if best_domain_candidate is not None and len(active_mapper_names) >= min_active_mappers:
        assigned_domain = best_domain_candidate

    tag_payload = {
        'id': normalized['id'],
        'corpus': corpus_name,
        'source_name': normalized['source_name'],
        'original_domain': record.get('domain'),
        'best_domain_candidate': best_domain_candidate,
        'assigned_domain': assigned_domain,
        'text_length': len(normalized['text']),
        'active_mapper_count': len(active_mapper_names),
        'active_mapper_names': active_mapper_names,
        'unique_active_mapper_count': len(unique_active_mapper_names),
        'unique_active_mapper_names': unique_active_mapper_names,
        'keep': assigned_domain is not None and len(active_mapper_names) >= min_active_mappers,
        'domain_candidates': domain_candidates,
        'operators': operator_results,
        'skipped_long_text': False,
    }

    filtered_record = None
    if tag_payload['keep']:
        filtered_record = _build_filtered_record(record, corpus_name, assigned_domain, tag_payload)
    return tag_payload, filtered_record


def process_corpus(
    corpus_name: str,
    raw_path: Path,
    tagged_path: Path,
    filtered_path: Path,
    plan: Dict[str, Any],
    field_map: Dict[str, str] | None = None,
    defaults: Dict[str, Any] | None = None,
    min_active_mappers: int = 2,
    max_text_length: int | None = None,
    max_records: int | None = None,
    progress_every: int = 0,
    total_records_hint: int | None = None,
    resume: bool = False,
    combined_handle=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    tagged_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    best_candidate_records = 0
    assigned_records = 0
    kept_records = 0
    insufficient_mapper_records = 0
    skipped_long_text_records = 0
    active_mapper_total = 0
    kept_assignments = Counter()
    best_candidate_assignments = Counter()

    resume_from_records = 0
    if resume and tagged_path.exists():
        with open(tagged_path, 'r', encoding='utf-8') as existing_tagged:
            for lineno, line in enumerate(existing_tagged, start=1):
                line = line.strip()
                if not line:
                    continue
                resume_from_records += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    preview = line[:200]
                    raise ValueError(
                        f'Invalid JSONL record in {tagged_path} at line {lineno}: {exc.msg}. '
                        f'Line preview: {preview!r}'
                    ) from exc
                active_count = int(payload.get('active_mapper_count', 0) or 0)
                active_mapper_total += active_count
                if payload.get('skipped_long_text'):
                    skipped_long_text_records += 1
                if payload.get('best_domain_candidate'):
                    best_candidate_records += 1
                    best_candidate_assignments[str(payload['best_domain_candidate'])] += 1
                if payload.get('assigned_domain'):
                    assigned_records += 1
                if not payload.get('skipped_long_text') and active_count < min_active_mappers:
                    insufficient_mapper_records += 1
                if payload.get('keep') and payload.get('assigned_domain'):
                    kept_records += 1
                    kept_assignments[str(payload['assigned_domain'])] += 1
        total_records = resume_from_records

    if resume_from_records > 0:
        print(
            f'[{corpus_name}] resume from record {resume_from_records}: '
            f'kept {kept_records}, assigned {assigned_records}',
            flush=True,
        )
    start_ts = monotonic()
    last_progress_total = -1

    def _print_progress(force: bool = False) -> None:
        nonlocal last_progress_total
        if not force and (progress_every <= 0 or total_records % progress_every != 0):
            return
        if total_records == 0:
            return
        if total_records == last_progress_total:
            return

        elapsed = max(monotonic() - start_ts, 1e-9)
        speed = total_records / elapsed
        keep_rate = kept_records / total_records
        prefix = f'[{corpus_name}] progress'
        if total_records_hint:
            pct = (total_records / total_records_hint) * 100
            remaining = max(total_records_hint - total_records, 0)
            eta_sec = (remaining / speed) if speed > 0 else float('inf')
            eta_str = f'{eta_sec / 60:.1f}m' if eta_sec != float('inf') else 'unknown'
            print(
                f"{prefix}: {total_records}/{total_records_hint} ({pct:.1f}%), "
                f"{speed:.1f} rec/s, keep_rate={keep_rate:.3f}, eta={eta_str}",
                flush=True,
            )
            last_progress_total = total_records
            return

        print(
            f"{prefix}: {total_records} records, {speed:.1f} rec/s, keep_rate={keep_rate:.3f}",
            flush=True,
        )
        last_progress_total = total_records

    tagged_mode = 'a' if resume_from_records > 0 else 'w'
    filtered_mode = 'a' if resume_from_records > 0 else 'w'

    with open(tagged_path, tagged_mode, encoding='utf-8') as tagged_out, open(filtered_path, filtered_mode, encoding='utf-8') as filtered_out:
        for record_idx, record in enumerate(iter_jsonl(raw_path)):
            if max_records is not None and record_idx >= max_records:
                break
            if record_idx < resume_from_records:
                continue
            tag_payload, filtered_record = label_record(
                record,
                raw_path,
                corpus_name,
                plan,
                field_map=field_map,
                defaults=defaults,
                min_active_mappers=min_active_mappers,
                max_text_length=max_text_length,
            )
            tagged_out.write(json.dumps(tag_payload, ensure_ascii=False) + '\n')

            total_records += 1
            active_mapper_total += tag_payload['active_mapper_count']
            if tag_payload['best_domain_candidate'] is not None:
                best_candidate_records += 1
                best_candidate_assignments[tag_payload['best_domain_candidate']] += 1
            if tag_payload['assigned_domain'] is not None:
                assigned_records += 1
            if tag_payload.get('skipped_long_text'):
                skipped_long_text_records += 1
            elif tag_payload['active_mapper_count'] < min_active_mappers:
                insufficient_mapper_records += 1
            if filtered_record is None:
                _print_progress()
                continue

            kept_records += 1
            kept_assignments[tag_payload['assigned_domain']] += 1
            serialized = json.dumps(filtered_record, ensure_ascii=False)
            filtered_out.write(serialized + '\n')
            if combined_handle is not None:
                combined_handle.write(serialized + '\n')
            _print_progress()

    _print_progress(force=True)

    summary_row = {
        'corpus': corpus_name,
        'raw_path': str(raw_path),
        'total_records': total_records,
        'best_candidate_records': best_candidate_records,
        'assigned_records': assigned_records,
        'kept_records': kept_records,
        'dropped_records': total_records - kept_records,
        'insufficient_mapper_records': insufficient_mapper_records,
        'skipped_long_text_records': skipped_long_text_records,
        'unassigned_records': total_records - assigned_records,
        'keep_rate': (kept_records / total_records) if total_records else 0.0,
        'mean_active_mapper_count': (active_mapper_total / total_records) if total_records else 0.0,
        'min_active_mappers': min_active_mappers,
    }

    assignment_rows: List[Dict[str, Any]] = []
    for scope, counter in (('best_candidate', best_candidate_assignments), ('kept', kept_assignments)):
        for domain_name, count in sorted(counter.items()):
            assignment_rows.append(
                {
                    'corpus': corpus_name,
                    'scope': scope,
                    'assigned_domain': domain_name,
                    'count': count,
                }
            )

    return summary_row, assignment_rows
