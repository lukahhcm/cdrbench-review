#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from math import ceil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

from cdrbench.config import load_domains_config


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
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


def _normalize_ops(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(values)))


def _support_threshold(num_records: int, min_support: int, min_support_ratio: float) -> int:
    ratio_threshold = ceil(num_records * min_support_ratio) if min_support_ratio > 0 else 0
    return max(min_support, ratio_threshold, 1)


def _frequent_subsets(
    operator_sets: list[tuple[str, ...]],
    min_combo_len: int,
    max_combo_len: int,
) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for ops in operator_sets:
        max_len = min(max_combo_len, len(ops))
        for size in range(min_combo_len, max_len + 1):
            counts.update(combinations(ops, size))
    return counts


def _choose_family_anchors(
    subset_counts: Counter[tuple[str, ...]],
    exact_counts: Counter[tuple[str, ...]],
    max_families: int,
    min_family_support: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for ops, subset_support in subset_counts.items():
        family_support = sum(support for sig, support in exact_counts.items() if set(ops).issubset(sig))
        if family_support < min_family_support:
            continue
        candidates.append(
            {
                'operators': ops,
                'subset_support': subset_support,
                'family_support': family_support,
            }
        )

    candidates.sort(
        key=lambda row: (
            -row['family_support'],
            -len(row['operators']),
            -row['subset_support'],
            row['operators'],
        )
    )

    selected: list[dict[str, Any]] = []
    covered_signatures: set[tuple[str, ...]] = set()
    for row in candidates:
        ops = row['operators']
        matched_signatures = [sig for sig in exact_counts if set(ops).issubset(sig)]
        new_support = sum(exact_counts[sig] for sig in matched_signatures if sig not in covered_signatures)
        if new_support <= 0 and selected:
            continue
        selected.append(
            {
                **row,
                'matched_signatures': matched_signatures,
                'new_support': new_support,
            }
        )
        covered_signatures.update(matched_signatures)
        if len(selected) >= max_families:
            break
    return selected


def _assign_signature_to_family(
    signature: tuple[str, ...],
    families: list[dict[str, Any]],
) -> int | None:
    sig_set = set(signature)
    best_idx = None
    best_score = None
    for idx, family in enumerate(families):
        anchor = tuple(family['operators'])
        anchor_set = set(anchor)
        overlap = len(sig_set & anchor_set)
        if not overlap:
            continue
        subset_bonus = 1 if anchor_set.issubset(sig_set) else 0
        score = (
            subset_bonus,
            overlap / max(len(anchor_set), 1),
            len(anchor),
            family['family_support'],
            family['subset_support'],
        )
        if best_score is None or score > best_score:
            best_idx = idx
            best_score = score
    return best_idx


def _select_cover_recipes(
    exact_counts: Counter[tuple[str, ...]],
    max_candidates: int,
) -> list[tuple[str, ...]]:
    remaining = dict(exact_counts)
    covered_ops: set[str] = set()
    selected: list[tuple[str, ...]] = []

    while remaining and len(selected) < max_candidates:
        best_sig = None
        best_score = None
        for sig, support in remaining.items():
            uncovered = len(set(sig) - covered_ops)
            score = (uncovered, len(sig), support)
            if best_score is None or score > best_score:
                best_sig = sig
                best_score = score
        if best_sig is None:
            break
        selected.append(best_sig)
        covered_ops.update(best_sig)
        remaining.pop(best_sig, None)
    return selected


def _build_domain_report(
    domain: str,
    records: list[dict[str, Any]],
    domain_cfg: dict[str, Any],
    min_support: int,
    min_recipe_support: int,
    min_support_ratio: float,
    min_combo_len: int,
    max_combo_len: int,
    top_k: int,
    max_families: int,
    max_recipes_per_family: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    operator_sets = [_normalize_ops(row.get('active_mapper_names', [])) for row in records]
    exact_counts: Counter[tuple[str, ...]] = Counter(operator_sets)
    subset_counts = _frequent_subsets(operator_sets, min_combo_len=min_combo_len, max_combo_len=max_combo_len)
    num_records = len(records)
    threshold = _support_threshold(num_records, min_support=min_support, min_support_ratio=min_support_ratio)
    families = _choose_family_anchors(
        subset_counts=subset_counts,
        exact_counts=exact_counts,
        max_families=max_families,
        min_family_support=threshold,
    )

    length_counter = Counter(len(ops) for ops in operator_sets)
    source_counter = Counter(str(row.get('corpus', 'unknown')) for row in records)
    all_active_ops = sorted({op for ops in operator_sets for op in ops})

    exact_rows = []
    for ops, support in exact_counts.most_common(top_k):
        exact_rows.append(
            {
                'domain': domain,
                'operators': ' | '.join(ops),
                'length': len(ops),
                'support': support,
                'support_ratio': support / num_records if num_records else 0.0,
            }
        )

    subset_rows = []
    for ops, support in subset_counts.most_common():
        if support < threshold:
            continue
        subset_rows.append(
            {
                'domain': domain,
                'operators': ' | '.join(ops),
                'length': len(ops),
                'support': support,
                'support_ratio': support / num_records if num_records else 0.0,
            }
        )
        if len(subset_rows) >= top_k:
            break

    family_rows = []
    recipe_rows = []
    recipe_yaml_rows = []
    assigned_signatures: set[tuple[str, ...]] = set()
    kept_families: list[dict[str, Any]] = []

    for family_idx, family in enumerate(families, start=1):
        anchor = tuple(family['operators'])
        assigned = []
        for signature, support in exact_counts.most_common():
            if signature in assigned_signatures:
                continue
            assigned_family_idx = _assign_signature_to_family(signature, families)
            if assigned_family_idx == family_idx - 1:
                assigned.append((signature, support))
                assigned_signatures.add(signature)

        if not assigned:
            continue

        assigned.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))
        valid_assigned = [(ops, support) for ops, support in assigned if support >= min_recipe_support]
        selected_recipes = valid_assigned[:max_recipes_per_family]
        if not selected_recipes:
            continue
        family_support = sum(support for _, support in assigned)
        kept_families.append(
            {
                'anchor': anchor,
                'subset_support': family['subset_support'],
                'family_support': family_support,
                'selected_recipes': selected_recipes,
            }
        )

    for kept_family_idx, kept_family in enumerate(kept_families, start=1):
        family_id = f'{domain}_family_{kept_family_idx:02d}'
        anchor = tuple(kept_family['anchor'])
        selected_recipes = list(kept_family['selected_recipes'])

        family_rows.append(
            {
                'domain': domain,
                'family_id': family_id,
                'anchor_operators': ' | '.join(anchor),
                'anchor_length': len(anchor),
                'subset_support': kept_family['subset_support'],
                'family_support': kept_family['family_support'],
                'family_support_ratio': kept_family['family_support'] / num_records if num_records else 0.0,
                'num_concrete_recipe_candidates': len(selected_recipes),
            }
        )

        yaml_recipes = []
        for recipe_rank, (ops, support) in enumerate(selected_recipes, start=1):
            recipe_id = f'{domain}_recipe_{kept_family_idx:02d}_{recipe_rank:02d}'
            recipe_rows.append(
                {
                    'domain': domain,
                    'family_id': family_id,
                    'recipe_id': recipe_id,
                    'operators': ' | '.join(ops),
                    'length': len(ops),
                    'support': support,
                    'support_ratio': support / num_records if num_records else 0.0,
                    'selection_source': 'bottom_up_exact_signature',
                }
            )
            yaml_recipes.append(
                {
                    'recipe_id': recipe_id,
                    'operator_set': list(ops),
                    'length': len(ops),
                    'support': support,
                    'support_ratio': round(support / num_records, 6) if num_records else 0.0,
                    'selection_source': 'bottom_up_exact_signature',
                    'curation_status': 'needs_ordering_and_activation_spec',
                }
            )

        recipe_yaml_rows.append(
            {
                'family_id': family_id,
                'anchor_operator_set': list(anchor),
                'anchor_length': len(anchor),
                'subset_support': kept_family['subset_support'],
                'family_support': kept_family['family_support'],
                'family_support_ratio': round(kept_family['family_support'] / num_records, 6) if num_records else 0.0,
                'concrete_recipes': yaml_recipes,
            }
        )

    leftover_exact = [
        (ops, support)
        for ops, support in exact_counts.most_common()
        if ops not in assigned_signatures and support >= min_recipe_support
    ]
    fallback_rows = []
    for rank, ops in enumerate(_select_cover_recipes(dict(leftover_exact), max_candidates=min(top_k, 12)), start=1):
        support = exact_counts[ops]
        fallback_rows.append(
            {
                'domain': domain,
                'family_id': f'{domain}_fallback_family_01',
                'recipe_id': f'{domain}_fallback_recipe_{rank:02d}',
                'rank': rank,
                'operators': ' | '.join(ops),
                'length': len(ops),
                'support': support,
                'support_ratio': support / num_records if num_records else 0.0,
                'selection_reason': 'coverage_fallback_unassigned_signature',
                'selection_source': 'coverage_fallback_unassigned_signature',
            }
        )

    report = {
        'domain': domain,
        'description': domain_cfg.get('description', ''),
        'num_records': num_records,
        'min_support_threshold': threshold,
        'min_recipe_support': min_recipe_support,
        'active_operator_inventory': all_active_ops,
        'configured_specific_operators': [op['name'] for op in domain_cfg.get('specific_operators', [])],
        'source_corpora': dict(source_counter),
        'length_distribution': dict(sorted(length_counter.items())),
        'recipe_families': recipe_yaml_rows,
        'fallback_recipe_candidates': [
            {
                'operators': row['operators'].split(' | '),
                'length': row['length'],
                'support': row['support'],
                'support_ratio': row['support_ratio'],
                'selection_reason': row['selection_reason'],
            }
            for row in fallback_rows
        ],
        'notes': [
            'Bottom-up mining is based on active operator sets from tagging outputs.',
            f'Concrete recipes are only kept if support >= {min_recipe_support}.',
            'Fallback recipe candidates are reported for inspection but excluded from selected_recipes.csv.',
            'Concrete recipe candidates still need manual ordering and activation-spec curation.',
        ],
    }

    return (
        report,
        pd.DataFrame(exact_rows),
        pd.DataFrame(subset_rows),
        pd.DataFrame(family_rows),
        pd.DataFrame(recipe_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Mine per-domain recipe families and concrete recipe candidates from tagging outputs.'
    )
    parser.add_argument('--tagged-dir', default='data/processed/domain_tags')
    parser.add_argument('--domains-config', default='configs/domains.yaml')
    parser.add_argument('--output-dir', default='data/processed/recipe_mining')
    parser.add_argument('--domain-field', choices=['assigned_domain', 'best_domain_candidate'], default='assigned_domain')
    parser.add_argument('--min-active-mappers', type=int, default=2)
    parser.add_argument('--min-support', type=int, default=5)
    parser.add_argument('--min-recipe-support', '--min-workflow-support', dest='min_recipe_support', type=int, default=5)
    parser.add_argument('--min-support-ratio', type=float, default=0.02)
    parser.add_argument('--min-combo-len', type=int, default=2)
    parser.add_argument('--max-combo-len', type=int, default=5)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--max-families-per-domain', type=int, default=6)
    parser.add_argument('--max-recipes-per-family', '--max-workflows-per-family', dest='max_recipes_per_family', type=int, default=8)
    parser.add_argument(
        '--max-text-length',
        type=int,
        default=0,
        help='Skip tagged records whose text length exceeds this threshold. 0 disables the filter.',
    )
    args = parser.parse_args()

    root = ROOT
    tagged_dir = (root / args.tagged_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tagged_dir.exists():
        raise SystemExit(f'tagged dir not found: {tagged_dir}')

    domains_cfg = load_domains_config(root / args.domains_config)
    domain_defs = domains_cfg.get('domains', {})
    records_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_for_length = 0

    for path in sorted(tagged_dir.glob('*.jsonl')):
        for row in iter_jsonl(path):
            active_count = int(row.get('active_mapper_count', 0) or 0)
            if active_count < args.min_active_mappers:
                continue
            if args.max_text_length > 0:
                text_length = row.get('text_length')
                if not isinstance(text_length, int):
                    text_length = len(str(row.get('text', '')))
                if int(text_length) > args.max_text_length:
                    skipped_for_length += 1
                    continue
            domain = row.get(args.domain_field)
            if not domain:
                continue
            records_by_domain[str(domain)].append(row)

    summary_rows = []
    global_yaml = {'domains': {}}

    for domain, records in sorted(records_by_domain.items()):
        if domain not in domain_defs:
            continue
        report, exact_df, subset_df, family_df, recipe_df = _build_domain_report(
            domain=domain,
            records=records,
            domain_cfg=domain_defs[domain],
            min_support=args.min_support,
            min_recipe_support=args.min_recipe_support,
            min_support_ratio=args.min_support_ratio,
            min_combo_len=args.min_combo_len,
            max_combo_len=args.max_combo_len,
            top_k=args.top_k,
            max_families=args.max_families_per_domain,
            max_recipes_per_family=args.max_recipes_per_family,
        )

        domain_dir = output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / 'recipe_candidates.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        (domain_dir / 'recipe_candidates.yaml').write_text(
            yaml.safe_dump(report, allow_unicode=True, sort_keys=False),
            encoding='utf-8',
        )
        exact_df.to_csv(domain_dir / 'exact_signatures.csv', index=False)
        subset_df.to_csv(domain_dir / 'frequent_operator_sets.csv', index=False)
        family_df.to_csv(domain_dir / 'recipe_families.csv', index=False)
        recipe_df.to_csv(domain_dir / 'selected_recipes.csv', index=False)

        global_yaml['domains'][domain] = {
            'description': report['description'],
            'num_records': report['num_records'],
            'recipe_families': report['recipe_families'],
            'fallback_recipe_candidates': report['fallback_recipe_candidates'],
        }

        summary_rows.append(
            {
                'domain': domain,
                'num_records': report['num_records'],
                'max_text_length': args.max_text_length,
                'active_operator_inventory_size': len(report['active_operator_inventory']),
                'num_exact_signature_candidates': len(exact_df),
                'num_frequent_operator_sets': len(subset_df),
                'num_recipe_families': len(family_df),
                'num_selected_recipes': len(recipe_df),
                'min_support_threshold': report['min_support_threshold'],
                'min_recipe_support': report['min_recipe_support'],
            }
        )

        print(
            f"{domain}: records={report['num_records']}, "
            f"families={len(family_df)}, selected_recipes={len(recipe_df)}"
        )

    if not summary_rows:
        raise SystemExit('no domain records found for recipe mining')

    pd.DataFrame(summary_rows).to_csv(output_dir / 'domain_recipe_mining_summary.csv', index=False)
    (output_dir / 'recipe_candidates.yaml').write_text(
        yaml.safe_dump(global_yaml, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )
    print(
        f'wrote recipe mining outputs -> {output_dir} '
        f'(skipped_for_length={skipped_for_length}, max_text_length={args.max_text_length})'
    )


if __name__ == '__main__':
    main()
