from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import pandas as pd


def stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def execution_key(op_name: str, params: dict[str, Any]) -> str:
    if not params:
        return op_name
    params_blob = stable_json(params)
    suffix = hashlib.sha1(params_blob.encode('utf-8')).hexdigest()[:10]
    return f'{op_name}__{suffix}'


def domain_operator_groups(domains_cfg: dict[str, Any], domain_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global_shared = list(domains_cfg.get('shared_operators', []))
    domain_shared = list(domain_cfg.get('shared_operators', []))
    domain_specific = list(domain_cfg.get('specific_operators', []))

    if domain_specific or domain_shared or global_shared:
        return [*global_shared, *domain_shared], domain_specific

    return [], list(domain_cfg.get('operators', []))


def build_domain_execution_plan(domains_cfg: dict[str, Any]) -> dict[str, Any]:
    execution_variants_by_key: dict[str, dict[str, Any]] = {}
    domains_by_execution_key: dict[str, set[str]] = defaultdict(set)
    domain_profiles: dict[str, dict[str, Any]] = {}
    execution_order: list[str] = []
    domain_order: list[str] = []

    for domain_name, domain_cfg in domains_cfg.get('domains', {}).items():
        domain_order.append(domain_name)
        mapper_keys: list[str] = []
        filter_keys: list[str] = []
        configured_shared_mapper_keys: list[str] = []
        configured_specific_mapper_keys: list[str] = []
        seen_keys: set[str] = set()
        shared_ops, specific_ops = domain_operator_groups(domains_cfg, domain_cfg)
        for scope, op_cfgs in (('shared', shared_ops), ('specific', specific_ops)):
            for op_cfg in op_cfgs:
                op_name = op_cfg['name']
                kind = op_cfg['kind']
                params = dict(op_cfg.get('params', {}))
                key = execution_key(op_name, params)
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


def domain_operator_catalog_frame(plan: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in plan['execution_variants']:
        domains = plan['domains_by_execution_key'].get(variant['key'], [])
        for domain_name in domains:
            rows.append(
                {
                    'domain': domain_name,
                    'execution_key': variant['key'],
                    'operator': variant['name'],
                    'kind': variant['kind'],
                    'params': stable_json(variant.get('params', {})),
                    'domain_count': len(domains),
                    'is_unique_to_domain': len(domains) == 1,
                }
            )
    return pd.DataFrame(rows)


def rank_domain_candidates(
    operator_results: dict[str, dict[str, Any]],
    plan: dict[str, Any],
    preferred_domain: str | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
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


def build_filtered_record(
    record: dict[str, Any],
    corpus_name: str,
    assigned_domain: str,
    tag_payload: dict[str, Any],
) -> dict[str, Any]:
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
