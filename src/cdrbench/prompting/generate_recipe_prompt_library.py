#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[3]

from cdrbench.llm_utils import build_client, chat_completion, parse_json_response, resolve_model


TRACK_FILES = {
    'main': 'main.jsonl',
    'agnostic_m': 'agnostic_m.jsonl',
    'order_sensitivity': 'order_sensitivity.jsonl',
    'order_f': 'order_f.jsonl',
    'mapper_order_sensitivity': 'mapper_order_sensitivity.jsonl',
    'order_m': 'order_m.jsonl',
    'semantic_agent_atomic': 'semantic_agent_atomic.jsonl',
    'semantic_agent_compositional': 'semantic_agent_compositional.jsonl',
    'atomic_ops': 'atomic_ops.jsonl',
    'atomic_m': 'atomic_m.jsonl',
    'atomic_f': 'atomic_f.jsonl',
}

SKIPPED_OPERATORS = {'flagged_words_filter', 'stopwords_filter'}

FILTER_THRESHOLD_EXPRESSION_GUIDE = {
    'alphanumeric_filter': 'Say this as a minimum share of alphanumeric content, e.g., "keep only text where at least 60% of the characters are letters or numbers."',
    'average_line_length_filter': 'Say this as an average line-length limit, e.g., "keep only documents whose average line length is no more than 80 characters."',
    'character_repetition_filter': 'Say this as a maximum repeated-character ratio, e.g., "reject text where repeated character patterns exceed 20%."',
    'flagged_words_filter': 'Say this as a maximum flagged-word share, e.g., "keep only text where flagged terms make up less than 0.1%."',
    'maximum_line_length_filter': 'Say this as a longest-line limit, e.g., "reject samples with any line longer than 200 characters."',
    'special_characters_filter': 'Say this as a maximum special-character share, e.g., "keep only text where special characters are below 15%."',
    'stopwords_filter': 'Say this as a minimum natural-language stopword share, e.g., "keep only text with at least 5% common function words."',
    'text_length_filter': 'Say this as a character-count constraint, e.g., "keep only text with at least 100 characters."',
    'word_repetition_filter': 'Say this as a maximum repeated-word ratio, e.g., "reject text where repeated words exceed 20%."',
    'words_num_filter': 'Say this as a word-count constraint, e.g., "keep only documents with at least 50 words."',
}

STYLE_PRESETS = [
    {
        'style_id': 'imperative_checklist',
        'label': 'Imperative Checklist',
        'definition': 'A direct command that lists the required operations explicitly and in order.',
        'template': 'Please perform the following operations on the text: {requirement}.',
        'example': 'Please clean this web page by removing the HTML, stripping links, normalizing the whitespace, and then keep it only if the cleaned text has at least 100 characters.',
        'guidance': 'Write a direct step-by-step request with explicit sequencing, like a user telling an assistant exactly what to do.',
    },
    {
        'style_id': 'goal_oriented',
        'label': 'Goal-Oriented Description',
        'definition': 'A prose description that emphasizes the intended final state rather than a numbered procedure.',
        'template': 'The goal is to make the data suitable for {use_case}; it should end up with {requirement}.',
        'example': 'The goal is to make these help documents clean enough for a support index, with links removed, repeated template sentences cleaned up, spacing made consistent, and only documents with no more than 20% repeated words retained.',
        'guidance': 'Describe the goal and desired cleanup outcome in fluent prose without sounding like code or a numbered recipe.',
    },
    {
        'style_id': 'application_context',
        'label': 'Application-Context Task',
        'definition': 'A task framed around a downstream use case such as retrieval, indexing, release, compliance, or corpus construction.',
        'template': 'For {use_case}, process the following data so that {requirement}.',
        'example': 'For downstream retrieval, process these reports so that disclaimers, table residue, and abnormal long lines are removed, then keep only reports whose longest remaining line is no more than 200 characters.',
        'guidance': 'Frame the request around a realistic downstream use case such as retrieval, indexing, release, or corpus preparation.',
    },
    {
        'style_id': 'qa_request',
        'label': 'Quality-Control Request',
        'definition': 'A request that sounds like data quality screening, emphasizing retention criteria and rejection conditions.',
        'template': 'Quality-check this sample: {requirement}. Keep it only if it still satisfies the quality condition.',
        'example': 'Quality-check this page after cleaning away links and noisy HTML. Keep it only if the remaining text has at least 50 words and looks usable as corpus content.',
        'guidance': 'Phrase it like a quality-control request that focuses on what should be retained or rejected and why.',
    },
    {
        'style_id': 'analyst_handoff',
        'label': 'Analyst Handoff',
        'definition': 'A natural teammate-to-teammate handoff written in practical workplace language.',
        'template': 'Could you take this batch and {requirement}?',
        'example': 'Could you take these LaTeX sources, remove the comments and bibliography, expand the simple macros, and then keep only sources whose final text length is at least 1,000 characters?',
        'guidance': 'Phrase it like one teammate handing a dataset-cleaning request to another teammate in normal workplace language.',
    },
    {
        'style_id': 'concise_brief',
        'label': 'Concise Brief',
        'definition': 'A short, compact instruction that keeps all necessary behavior while minimizing wording.',
        'template': '{requirement}. Return the final keep/drop decision and cleaned text.',
        'example': 'Remove private identifiers and unsafe links, normalize the remaining text, then keep only sanitized samples whose special-character share is below 15%.',
        'guidance': 'Write a compact but complete user request with minimal fluff, while still preserving all important behavior.',
    },
    {
        'style_id': 'policy_like',
        'label': 'Policy-Like Requirement',
        'definition': 'A formal processing requirement that reads like a data handling policy, not code.',
        'template': 'Before this data can be used, it must satisfy the following processing requirement: {requirement}.',
        'example': 'Before release, the text must not contain emails, IP addresses, file paths, credentials, or long secret-like tokens; after sanitization, retain it only if the result still has at least 100 words.',
        'guidance': 'Write it like a processing requirement or policy note, but still from a user-facing perspective rather than code.',
    },
    {
        'style_id': 'recipe_narrative',
        'label': 'Recipe Narrative',
        'definition': 'A scenario-style request that first explains the messy data situation and then asks for the needed processing.',
        'template': 'I have {data_situation}. Please {requirement}.',
        'example': 'I have raw crawl pages with markup, navigation links, and messy whitespace. Please turn them into clean readable text and keep only pages whose alphanumeric content makes up at least 60% of the final text.',
        'guidance': 'Describe the data situation first, then explain the requested cleanup and filtering behavior as a realistic need.',
    },
    {
        'style_id': 'end_weighted_instruction',
        'label': 'End-Weighted Instruction',
        'definition': 'The raw data appears before the concrete instruction, so the task request is weighted toward the end of the prompt.',
        'template': '[Data first] ... Above is the raw data. Please apply this processing request: {requirement}.',
        'example': 'Above is the raw document. Please remove links and contact information, normalize spacing, and then keep the document only if its final text has at least 500 characters.',
        'guidance': 'Write a requirement intended to be placed after the raw data. Make the instruction self-contained and attention-grabbing at the end.',
    },
    {
        'style_id': 'negative_constraint_driven',
        'label': 'Negative-Constraint Driven',
        'definition': 'A request centered on what must not remain in the output, useful for cleaning and filtering recipes.',
        'template': 'When processing this text, make sure the final result does not contain {noise_list}; also {requirement}.',
        'example': 'When processing this report, make sure the final result contains no disclaimers, table residue, or lines longer than 180 characters. After that, judge whether it is suitable for retrieval.',
        'guidance': 'Emphasize unwanted content that should be absent from the final output, while preserving the actual recipe order.',
    },
    {
        'style_id': 'conversational_cooperative',
        'label': 'Conversational Cooperative',
        'definition': 'A casual chat-style request that sounds like a real user asking for help, while still being complete enough to execute.',
        'template': 'Hey, could you help me with this text? I need you to {requirement}.',
        'example': 'Hey, could you help me clean up these source files? I need the LaTeX comments and references removed, the macros expanded, and then please keep only sources with at least 300 words left.',
        'guidance': 'Use natural conversational wording, but do not omit key steps, order, or filtering behavior.',
    },
]

GENERATION_SYSTEM_PROMPT = """You are an expert at understanding data processing code and operator documentation, then translating the behavior into realistic user-facing data refinement requests.

You will be given:
1. The benchmark track and domain.
2. The internal recipe sequence and filter parameters.
3. Source code and documentation snippets for the operators.
4. A list of requested style slots.

Your task:
- Generate exactly one candidate user requirement body for every requested style slot.
- The result for every slot must be FUNCTIONALLY EQUIVALENT to the recipe.
- Pretend the user has never seen the code.
- Never mention operator names, parameter names, class names, file names, YAML, Python, or implementation details.
- Preserve the exact recipe order and all essential filter semantics.
- If a recipe contains any filter/keep-drop step, you MUST express every active threshold naturally and explicitly in the user requirement.
- Express thresholds as user-style constraints such as "at least 100 characters", "no more than 200 characters per line", "below 15%", or "at least 50 words".
- Do NOT use parameter names such as min_len, max_len, min_ratio, max_ratio, min_num, max_num, or internal statistic names.
- Make the prompts stylistically diverse and realistic across different users.

The generated text should be ONLY the user requirement body.
Do not include raw data placeholders, JSON schema instructions, answer format instructions, or system/developer wording. The benchmark wrapper will add those later.

Hard requirements for every candidate requirement:
- The requested operation order must be correct.
- Filtering behavior must be correct.
- Any numeric threshold or ratio required by a filter must be explicitly grounded in natural language.
- Do not ask clarifying questions.
- Do not refer to the benchmark, hidden reference, or code.

Return JSON only, with this exact schema:
{
  "candidates": [
    {
      "request_key": "...",
      "style_id": "...",
      "style_label": "...",
      "user_requirement": "...",
      "style_notes": "short note on why this wording is stylistically distinct"
    }
  ]
}
"""

JUDGE_SYSTEM_PROMPT = """You are a strict benchmark prompt judge.

You will be given:
- The internal recipe definition and filter parameters.
- The operator code/doc evidence.
- One candidate user-facing prompt.

Judge whether the prompt is faithful to what the recipe actually does.

Mandatory keep conditions:
1. Functional equivalence: the prompt requests the same transformation and filtering behavior.
2. Order correctness: the requested order matches the internal recipe order.
3. No code leakage: the prompt does not mention operator names, parameter names, YAML, Python, hidden code, or implementation internals.
4. Threshold grounding: if the recipe has filter parameters, all active numeric thresholds must appear as natural user-facing constraints, not vague words like "long enough" or "too repetitive".
5. Wrapper compatibility: the user requirement can be safely combined with a fixed benchmark wrapper that separately supplies raw input text and the JSON output contract.

Also score:
- user_naturalness: does it sound like a plausible user request?
- threshold_grounding: are thresholds/conditions expressed naturally and correctly?
- clarity: is the request clear and executable?
- style_distinctiveness: does this candidate sound distinct from a generic template for the same style?

Return JSON only:
{
  "verdict": "keep" or "reject",
  "must_pass": {
    "functional_equivalence": true,
    "order_correct": true,
    "no_code_leakage": true,
    "thresholds_grounded": true,
    "wrapper_compatible": true
  },
  "scores": {
    "user_naturalness": 1-5,
    "threshold_grounding": 1-5,
    "clarity": 1-5,
    "style_distinctiveness": 1-5
  },
  "issues": ["..."],
  "summary": "one short sentence"
}
"""

MAX_JSON_RETRIES = 3
RAW_RESPONSE_PREVIEW_CHARS = 800
EVAL_PROGRESS_EVERY = 200
REQUEST_MAX_RETRIES = 5
REQUEST_RETRY_BASE_SECONDS = 2.0
REQUEST_RETRY_MAX_SECONDS = 20.0


class TransientLLMError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    count = 0
    with tmp_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
    tmp_path.replace(path)
    return count


def _is_retryable_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, 'status_code', None)
    if isinstance(status_code, int) and status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    response = getattr(exc, 'response', None)
    response_status = getattr(response, 'status_code', None)
    if isinstance(response_status, int) and response_status in {408, 409, 429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    retryable_signals = [
        'rate limit',
        'timeout',
        'timed out',
        'network closed',
        'connection reset',
        'server disconnected',
        'temporarily unavailable',
        'internal_server_error',
        'error code: 500',
        'error code: 502',
        'error code: 503',
        'error code: 504',
        'unavailable',
        'apiconnectionerror',
    ]
    return any(signal in message for signal in retryable_signals)


def _request_retry_sleep_seconds(attempt: int) -> float:
    return min(REQUEST_RETRY_BASE_SECONDS * (2 ** max(attempt - 1, 0)), REQUEST_RETRY_MAX_SECONDS)


def _chat_completion_with_retries(
    *,
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    request_kind: str,
    recipe_key: str,
    candidate_id: str | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_MAX_RETRIES + 2):
        try:
            return chat_completion(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except Exception as exc:
            last_error = exc
            retryable = _is_retryable_llm_error(exc)
            target = f'recipe={recipe_key}'
            if candidate_id:
                target += f' candidate={candidate_id}'
            if not retryable or attempt > REQUEST_MAX_RETRIES:
                if retryable:
                    raise TransientLLMError(
                        f'{request_kind} request failed after {attempt} attempts for {target}: {exc}'
                    ) from exc
                raise
            sleep_seconds = _request_retry_sleep_seconds(attempt)
            print(
                f'retry {request_kind} request: attempt={attempt}/{REQUEST_MAX_RETRIES + 1} '
                f'{target} sleep_sec={sleep_seconds:.1f} error={exc}',
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise TransientLLMError(
        f'{request_kind} request failed unexpectedly for recipe={recipe_key}: {last_error}'
    ) from last_error


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        payload = yaml.safe_load(f)
    return payload if isinstance(payload, dict) else {}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = '||'.join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _recipe_prompt_key(row: dict[str, Any]) -> str:
    value = _first_present(row, 'recipe_prompt_key', 'workflow_prompt_key')
    return '' if value is None else str(value)


def _find_operator_file(op_name: str, kind: str) -> Path | None:
    path = ROOT / 'data-juicer' / 'data_juicer' / 'ops' / kind / f'{op_name}.py'
    return path if path.exists() else None


def _find_operator_doc(op_name: str, kind: str) -> Path | None:
    path = ROOT / 'data-juicer' / 'docs' / 'operators' / kind / f'{op_name}.md'
    return path if path.exists() else None


def _trim_doc_text(text: str) -> str:
    markers = [
        '\n## 📊 Effect demonstration',
        '\n## Effect demonstration',
        '\n## 🔗 related links',
        '\n## Related links',
    ]
    trimmed = text
    for marker in markers:
        idx = trimmed.find(marker)
        if idx != -1:
            trimmed = trimmed[:idx]
    return trimmed.strip()


def _load_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ''
    return path.read_text(encoding='utf-8')


def _recipe_key(row: dict[str, Any]) -> str:
    operator_sequence = list(row.get('operator_sequence') or ([row['operator']] if row.get('operator') else []))
    return _stable_id(
        row.get('benchmark_track'),
        row.get('domain'),
        _first_present(row, 'recipe_type', 'workflow_type'),
        row.get('order_slot'),
        operator_sequence,
        row.get('filter_params_by_name') or {},
    )


def _should_skip_row(row: dict[str, Any], skipped_ops: set[str]) -> bool:
    operator_sequence = list(row.get('operator_sequence') or ([row['operator']] if row.get('operator') else []))
    return any(op in skipped_ops for op in operator_sequence)


def _operator_kind(op_name: str) -> str:
    return 'filter' if op_name.endswith('_filter') else 'mapper'


def _group_rows(rows: list[dict[str, Any]], skipped_ops: set[str]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    skipped = 0
    for row in rows:
        if _should_skip_row(row, skipped_ops):
            skipped += 1
            continue
        key = _recipe_key(row)
        grouped.setdefault(key, []).append(row)
    return grouped, skipped


def _style_requests(style_count: int, candidates_per_style: int) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for style in STYLE_PRESETS[:style_count]:
        for slot in range(1, candidates_per_style + 1):
            requests.append(
                {
                    **style,
                    'candidate_slot': slot,
                    'request_key': f"{style['style_id']}__{slot}",
                }
            )
    return requests


def _format_style_request(style_request: dict[str, Any]) -> str:
    return (
        f"- request_key: {style_request['request_key']}\n"
        f"  candidate_slot: {style_request['candidate_slot']}\n"
        f"  style_id: {style_request['style_id']}\n"
        f"  label: {style_request['label']}\n"
        f"  definition: {style_request['definition']}\n"
        f"  template: {style_request['template']}\n"
        f"  example: {style_request['example']}\n"
        f"  generation guidance: {style_request['guidance']}"
    )


def _recipe_bundle(
    recipe_key: str,
    rows: list[dict[str, Any]],
    prompt_cfg: dict[str, Any],
    variants_per_recipe: int,
    candidates_per_style: int,
) -> dict[str, Any]:
    row = rows[0]
    operator_sequence = list(row.get('operator_sequence') or ([row['operator']] if row.get('operator') else []))
    filter_params_by_name = row.get('filter_params_by_name') if isinstance(row.get('filter_params_by_name'), dict) else {}
    source_domain_set = sorted(
        {
            str(source_domain)
            for source_domain in (recipe_row.get('source_domain') for recipe_row in rows)
            if source_domain
        }
    )
    operators = []
    for op_name in operator_sequence:
        kind = _operator_kind(op_name)
        op_cfg = dict((prompt_cfg.get('operators') or {}).get(op_name) or {})
        filter_cfg = dict((prompt_cfg.get('filters') or {}).get(op_name) or {})
        code_path = _find_operator_file(op_name, kind)
        doc_path = _find_operator_doc(op_name, kind)
        operators.append(
            {
                'name': op_name,
                'kind': kind,
                'params': filter_params_by_name.get(op_name) if isinstance(filter_params_by_name.get(op_name), dict) else {},
                'render_hint': op_cfg.get('natural_language_intent') or filter_cfg.get('natural_language_intent'),
                'threshold_expression_guide': FILTER_THRESHOLD_EXPRESSION_GUIDE.get(op_name) if kind == 'filter' else None,
                'code_path': str(code_path) if code_path else None,
                'code_excerpt': _load_text(code_path),
                'doc_path': str(doc_path) if doc_path else None,
                'doc_excerpt': _trim_doc_text(_load_text(doc_path)),
            }
        )
    return {
        'recipe_prompt_key': recipe_key,
        'benchmark_track': row.get('benchmark_track'),
        'domain': row.get('domain'),
        'recipe_type': _first_present(row, 'recipe_type', 'workflow_type'),
        'order_slot': row.get('order_slot'),
        'operator_sequence': operator_sequence,
        'filter_params_by_name': filter_params_by_name,
        'threshold_meta': row.get('threshold_meta') or {},
        'num_instances': len(rows),
        'source_domain_set': source_domain_set,
        'style_requests': _style_requests(variants_per_recipe, candidates_per_style),
        'operators': operators,
    }


def _generation_user_prompt(bundle: dict[str, Any]) -> str:
    style_lines = [_format_style_request(style) for style in bundle['style_requests']]
    op_blocks = []
    for op in bundle['operators']:
        params_json = json.dumps(op['params'], ensure_ascii=False, sort_keys=True)
        op_blocks.append(
            '\n'.join(
                [
                    f"[Operator] {op['name']} ({op['kind']})",
                    f"Parameters: {params_json}",
                    f"Human hint: {op.get('render_hint') or ''}",
                    f"Threshold expression guide: {op.get('threshold_expression_guide') or ''}",
                    "[Documentation excerpt]",
                    op['doc_excerpt'] or '(no operator doc excerpt found)',
                    "[Source code]",
                    op['code_excerpt'] or '(no operator source found)',
                ]
            )
        )
    return (
        f"Benchmark track: {bundle['benchmark_track']}\n"
        f"Domain: {bundle['domain']}\n"
        f"Recipe type: {bundle.get('recipe_type')}\n"
        f"Order slot: {bundle.get('order_slot')}\n"
        f"Internal operator sequence: {' -> '.join(bundle['operator_sequence'])}\n"
        f"Filter params by name: {json.dumps(bundle['filter_params_by_name'], ensure_ascii=False, sort_keys=True)}\n"
        f"Threshold meta: {json.dumps(bundle['threshold_meta'], ensure_ascii=False, sort_keys=True)}\n\n"
        "Generate exactly one candidate user requirement body for every requested style slot below.\n"
        "Use the style definition, template, and example only as guidance; do not copy the example literally unless the recipe matches it.\n\n"
        f"{chr(10).join(style_lines)}\n\n"
        "Important: act as if the user has never seen code. Do not mention operator names, parameter names, class names, YAML, file paths, JSON schema, or implementation-specific terms.\n"
        "Return only the user requirement body for each candidate. The final benchmark prompt wrapper and output format will be added by code.\n\n"
        "Operator evidence:\n\n"
        f"{chr(10).join(op_blocks)}"
    )


def _judge_user_prompt(entry: dict[str, Any], candidate: dict[str, Any], *, client, model: str, temperature: float) -> dict[str, Any]:
    operator_blocks = []
    for op in entry.get('operators') or []:
        operator_blocks.append(
            '\n'.join(
                [
                    f"[Operator] {op.get('name')} ({op.get('kind')})",
                    f"Parameters: {json.dumps(op.get('params') or {}, ensure_ascii=False, sort_keys=True)}",
                    "[Documentation excerpt]",
                    str(op.get('doc_excerpt') or ''),
                    "[Source code]",
                    str(op.get('code_excerpt') or ''),
                ]
            )
        )
    user_prompt = (
        f"Benchmark track: {entry.get('benchmark_track')}\n"
        f"Domain: {entry.get('domain')}\n"
        f"Recipe type: {entry.get('recipe_type')}\n"
        f"Order slot: {entry.get('order_slot')}\n"
        f"Internal operator sequence: {' -> '.join(entry.get('operator_sequence') or [])}\n"
        f"Filter params by name: {json.dumps(entry.get('filter_params_by_name') or {}, ensure_ascii=False, sort_keys=True)}\n"
        f"Threshold meta: {json.dumps(entry.get('threshold_meta') or {}, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Candidate style: {candidate.get('style_id')} / {candidate.get('style_label')}\n"
        f"Candidate slot: {candidate.get('candidate_slot')}\n"
        f"Candidate user requirement body:\n{candidate.get('user_requirement')}\n\n"
        "Fixed wrapper that will be appended by code: raw input text plus a JSON-only output contract with status and clean_text.\n\n"
        "Operator evidence:\n"
        f"{chr(10).join(operator_blocks)}"
    )
    last_error: Exception | None = None
    last_content = ''
    for attempt in range(1, MAX_JSON_RETRIES + 1):
        content = _chat_completion_with_retries(
            client=client,
            model=model,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
            request_kind='judge',
            recipe_key=str(entry.get('recipe_prompt_key') or ''),
            candidate_id=str(candidate.get('candidate_id') or ''),
        )
        last_content = content
        try:
            payload = parse_json_response(content)
            if not isinstance(payload, dict):
                raise RuntimeError('Judge did not return a JSON object.')
            return payload
        except Exception as exc:
            last_error = exc
            print(
                f"retry judge JSON parse: attempt={attempt}/{MAX_JSON_RETRIES} "
                f"recipe={_recipe_prompt_key(entry)} candidate={candidate.get('candidate_id')} "
                f"error={exc}",
                flush=True,
            )
    preview = last_content[:RAW_RESPONSE_PREVIEW_CHARS].replace('\n', '\\n')
    raise RuntimeError(
        f"Judge JSON parse failed after {MAX_JSON_RETRIES} attempts for "
        f"recipe={_recipe_prompt_key(entry)} candidate={candidate.get('candidate_id')}. "
        f"Last error: {last_error}. Raw preview: {preview}"
    ) from last_error


def _cache_key(
    bundle: dict[str, Any],
    model: str,
    judge_model: str,
    prompt_source: str,
    min_average_score: float,
    skip_judge: bool,
) -> str:
    return _stable_id(
        model,
        judge_model,
        prompt_source,
        min_average_score,
        skip_judge,
        bundle['recipe_prompt_key'],
        bundle['style_requests'],
        bundle['operator_sequence'],
        bundle['filter_params_by_name'],
    )


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    for row in _read_jsonl(path):
        key = row.get('cache_key')
        entry = row.get('library_entry')
        if isinstance(key, str) and isinstance(entry, dict):
            accepted_candidate_count = int(entry.get('accepted_candidate_count', 0) or 0)
            judged_summary = entry.get('judged_candidate_summary')
            if accepted_candidate_count <= 0 and isinstance(judged_summary, list) and judged_summary:
                all_judge_error = True
                saw_judge_error = False
                for candidate_summary in judged_summary:
                    issues = candidate_summary.get('judge_issues') if isinstance(candidate_summary, dict) else None
                    if not isinstance(issues, list) or not issues:
                        all_judge_error = False
                        continue
                    issue_strings = [str(issue) for issue in issues]
                    has_judge_error = any(issue.startswith('judge_error:') for issue in issue_strings)
                    saw_judge_error = saw_judge_error or has_judge_error
                    if not has_judge_error:
                        all_judge_error = False
                if saw_judge_error and all_judge_error:
                    print(
                        f'skip suspect cache entry with only judge errors: '
                        f'recipe={_recipe_prompt_key(entry)} cache_key={key}',
                        flush=True,
                    )
                    continue
            cache[key] = entry
    return cache


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def _call_llm_for_candidates(
    *,
    bundle: dict[str, Any],
    client,
    model: str,
    temperature: float,
) -> list[dict[str, Any]]:
    request_map = {str(item['request_key']): item for item in bundle['style_requests']}
    last_error: Exception | None = None
    last_content = ''
    payload: dict[str, Any] | None = None
    for attempt in range(1, MAX_JSON_RETRIES + 1):
        content = _chat_completion_with_retries(
            client=client,
            model=model,
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=_generation_user_prompt(bundle),
            temperature=temperature,
            request_kind='generation',
            recipe_key=str(bundle.get('recipe_prompt_key') or ''),
        )
        last_content = content
        try:
            parsed = parse_json_response(content)
            if not isinstance(parsed, dict) or not isinstance(parsed.get('candidates'), list):
                raise RuntimeError('LLM did not return the expected JSON object with a candidates list.')
            payload = parsed
            break
        except Exception as exc:
            last_error = exc
            print(
                f"retry generation JSON parse: attempt={attempt}/{MAX_JSON_RETRIES} "
                f"recipe={_recipe_prompt_key(bundle)} error={exc}",
                flush=True,
            )
    if payload is None:
        preview = last_content[:RAW_RESPONSE_PREVIEW_CHARS].replace('\n', '\\n')
        raise RuntimeError(
            f"Generation JSON parse failed after {MAX_JSON_RETRIES} attempts for "
            f"recipe={_recipe_prompt_key(bundle)}. Last error: {last_error}. Raw preview: {preview}"
        ) from last_error
    candidates = []
    seen = set()
    for item in payload['candidates']:
        if not isinstance(item, dict):
            continue
        request_key = str(item.get('request_key') or '')
        request_meta = request_map.get(request_key)
        if request_meta is None:
            continue
        style_id = str(item.get('style_id') or request_meta['style_id'])
        user_requirement = str(item.get('user_requirement') or item.get('user_request') or '').strip()
        dedup_key = (request_key, user_requirement)
        if not request_key or not user_requirement or dedup_key in seen:
            continue
        seen.add(dedup_key)
        candidates.append(
            {
                'candidate_id': _stable_id(bundle['recipe_prompt_key'], request_key, user_requirement),
                'request_key': request_key,
                'candidate_slot': int(request_meta['candidate_slot']),
                'style_id': style_id,
                'style_label': str(item.get('style_label') or request_meta['label']),
                'style_notes': str(item.get('style_notes') or ''),
                'user_requirement': user_requirement,
            }
        )
    return candidates


def _template_candidates(bundle: dict[str, Any], prompt_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    domain_contexts = prompt_cfg.get('domain_contexts') or {}
    scenario = str(domain_contexts.get(str(bundle.get('domain'))) or 'Perform data refinement.')
    operators_cfg = prompt_cfg.get('operators') or {}
    filters_cfg = prompt_cfg.get('filters') or {}
    operator_lines = []
    for op_name in bundle['operator_sequence']:
        if op_name.endswith('_filter'):
            hint = (filters_cfg.get(op_name) or {}).get('natural_language_intent') or f'Apply {op_name}.'
        else:
            hint = (operators_cfg.get(op_name) or {}).get('natural_language_intent') or f'Apply {op_name}.'
        operator_lines.append(hint)
    joined = ' Then '.join(operator_lines)
    candidates = []
    for request_meta in bundle['style_requests']:
        if request_meta['style_id'] == 'imperative_checklist':
            text = f'{scenario} Please execute these steps in order: {joined}.'
        elif request_meta['style_id'] == 'goal_oriented':
            text = f'The goal is to refine this text for {bundle.get("domain")} use. {joined}.'
        elif request_meta['style_id'] == 'application_context':
            text = f'For downstream processing, please clean this data carefully. {joined}.'
        else:
            text = f'{scenario} {joined}.'
        candidates.append(
            {
                'candidate_id': _stable_id(bundle['recipe_prompt_key'], request_meta['request_key'], text),
                'request_key': request_meta['request_key'],
                'candidate_slot': int(request_meta['candidate_slot']),
                'style_id': request_meta['style_id'],
                'style_label': request_meta['label'],
                'style_notes': 'template fallback',
                'user_requirement': text,
            }
        )
    return candidates


def _build_library_entry(
    *,
    bundle: dict[str, Any],
    candidates: list[dict[str, Any]],
    prompt_source: str,
    generation_model: str,
    judge_model: str,
    min_average_score: float,
    client,
    judge_temperature: float,
    skip_judge: bool = False,
) -> dict[str, Any]:
    judged_candidates = []
    accepted_candidates = []
    accepted_style_ids: set[str] = set()
    for candidate in candidates:
        if skip_judge:
            verdict = {
                'verdict': 'keep',
                'must_pass': {
                    'functional_equivalence': True,
                    'order_correct': True,
                    'no_code_leakage': True,
                    'thresholds_grounded': True,
                    'wrapper_compatible': True,
                },
                'scores': {
                    'user_naturalness': 5,
                    'threshold_grounding': 5,
                    'clarity': 5,
                    'style_distinctiveness': 5,
                },
                'issues': ['judge_skipped'],
                'summary': 'Accepted without LLM judging.',
            }
        else:
            try:
                verdict = _judge_user_prompt(bundle, candidate, client=client, model=judge_model, temperature=judge_temperature)
            except TransientLLMError:
                raise
            except Exception as exc:
                print(
                    f"judge failed; rejecting candidate "
                    f"recipe={_recipe_prompt_key(bundle)} candidate={candidate.get('candidate_id')} error={exc}",
                    flush=True,
                )
                verdict = {
                    'verdict': 'reject',
                    'must_pass': {
                        'functional_equivalence': False,
                        'order_correct': False,
                        'no_code_leakage': False,
                        'thresholds_grounded': False,
                        'wrapper_compatible': False,
                    },
                    'scores': {
                        'user_naturalness': 0,
                        'threshold_grounding': 0,
                        'clarity': 0,
                        'style_distinctiveness': 0,
                    },
                    'issues': [f'judge_error: {exc}'],
                    'summary': 'judge failed to return parseable JSON',
                }
        must_pass = verdict.get('must_pass') if isinstance(verdict.get('must_pass'), dict) else {}
        scores = verdict.get('scores') if isinstance(verdict.get('scores'), dict) else {}
        average_score = sum(
            float(scores.get(key, 0))
            for key in ('user_naturalness', 'threshold_grounding', 'clarity', 'style_distinctiveness')
        ) / 4.0
        keep = (
            all(
                bool(must_pass.get(key))
                for key in (
                    'functional_equivalence',
                    'order_correct',
                    'no_code_leakage',
                    'thresholds_grounded',
                    'wrapper_compatible',
                )
            )
            and average_score >= min_average_score
            and str(verdict.get('verdict')).lower() == 'keep'
        )
        judged_candidate = {
            **candidate,
            'judge_average_score': round(average_score, 4),
            'accepted': keep,
        }
        judged_candidates.append(judged_candidate)
        if keep:
            accepted_style_ids.add(str(candidate.get('style_id') or ''))
            accepted_candidates.append(candidate)
    return {
        **bundle,
        'prompt_source': prompt_source,
        'generation_model': generation_model,
        'judge_model': judge_model,
        'requested_style_count': len({style['style_id'] for style in bundle['style_requests']}),
        'candidates_per_style': max((int(style['candidate_slot']) for style in bundle['style_requests']), default=0),
        'requested_candidate_count': len(bundle['style_requests']),
        'generated_candidate_count': len(candidates),
        'accepted_candidate_count': len(accepted_candidates),
        'accepted_style_count': len(accepted_style_ids),
        'min_average_score': min_average_score,
        'candidates': accepted_candidates,
        'judged_candidate_summary': [
            {
                'candidate_id': candidate['candidate_id'],
                'style_id': candidate['style_id'],
                'candidate_slot': candidate['candidate_slot'],
                'accepted': candidate['accepted'],
                'judge_average_score': candidate['judge_average_score'],
                'judge_issues': list(candidate.get('issues') or []),
            }
            for candidate in judged_candidates
        ],
    }


def _failed_library_entry(
    *,
    bundle: dict[str, Any],
    prompt_source: str,
    generation_model: str,
    judge_model: str,
    min_average_score: float,
    error: Exception,
) -> dict[str, Any]:
    return {
        **bundle,
        'prompt_source': prompt_source,
        'generation_model': generation_model,
        'judge_model': judge_model,
        'requested_style_count': len({style['style_id'] for style in bundle['style_requests']}),
        'candidates_per_style': max((int(style['candidate_slot']) for style in bundle['style_requests']), default=0),
        'requested_candidate_count': len(bundle['style_requests']),
        'generated_candidate_count': 0,
        'accepted_candidate_count': 0,
        'accepted_style_count': 0,
        'min_average_score': min_average_score,
        'candidates': [],
        'judged_candidate_summary': [],
        'generation_error': str(error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate and judge recipe-level CDR-Bench prompt libraries.')
    parser.add_argument('--benchmark-dir', default='data/processed/benchmark_instances')
    parser.add_argument('--output-dir', default='data/processed/prompt_library')
    parser.add_argument('--prompt-config', default='configs/recipe_prompting.yaml')
    parser.add_argument('--tracks', nargs='*', default=list(TRACK_FILES), choices=sorted(TRACK_FILES))
    parser.add_argument('--prompt-source', choices=['llm', 'template'], default='llm')
    parser.add_argument('--model', default=None, help='OpenAI-compatible generation model. Defaults to OPENAI_MODEL / LLM_MODEL env.')
    parser.add_argument('--base-url', default=None, help='OpenAI-compatible generation base URL. Defaults to OPENAI_BASE_URL / LLM_BASE_URL env.')
    parser.add_argument('--api-key', default=None, help='Generation API key. Defaults to OPENAI_API_KEY / LLM_API_KEY env.')
    parser.add_argument('--judge-model', default=None, help='OpenAI-compatible judge model. Defaults to --model or OPENAI_MODEL / LLM_MODEL env.')
    parser.add_argument('--judge-base-url', default=None, help='Judge base URL. Defaults to --base-url or OPENAI_BASE_URL / LLM_BASE_URL env.')
    parser.add_argument('--judge-api-key', default=None, help='Judge API key. Defaults to --api-key or OPENAI_API_KEY / LLM_API_KEY env.')
    parser.add_argument('--skip-judge', action='store_true', help='Accept generated/template prompt candidates without LLM judging.')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--judge-temperature', type=float, default=0.0)
    parser.add_argument('--variants-per-recipe', '--variants-per-workflow', dest='variants_per_recipe', type=int, default=11, help='How many style presets to request per recipe.')
    parser.add_argument('--candidates-per-style', type=int, default=3, help='How many prompt candidates to generate for each requested style.')
    parser.add_argument('--min-average-score', type=float, default=3.5)
    parser.add_argument(
        '--skip-operators',
        nargs='*',
        default=sorted(SKIPPED_OPERATORS),
        help='Skip recipes containing these operators when generating prompt candidates.',
    )
    parser.add_argument(
        '--cache-path',
        default='data/processed/prompt_library/recipe_prompt_library_cache.jsonl',
        help='Cache file for recipe-level prompt generation and judging.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Reuse recipe-level cache from --cache-path so interrupted runs can continue without regenerating finished recipes.',
    )
    args = parser.parse_args()

    benchmark_dir = (ROOT / args.benchmark_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_cfg = _load_yaml((ROOT / args.prompt_config).resolve())

    generation_model = resolve_model(args.model)
    judge_model = resolve_model(args.judge_model or args.model)
    generation_client = build_client(
        api_key=args.api_key,
        base_url=args.base_url,
        model=generation_model,
    ) if args.prompt_source == 'llm' else None
    judge_client = None if args.skip_judge else build_client(
        api_key=args.judge_api_key or args.api_key,
        base_url=args.judge_base_url or args.base_url,
        model=judge_model,
    )

    cache_path = (ROOT / args.cache_path).resolve()
    if args.resume:
        cache = _load_cache(cache_path)
        print(f'resume enabled: loaded {len(cache)} cached recipe prompt library entries from {cache_path}', flush=True)
    else:
        cache = {}
        if cache_path.exists():
            cache_path.unlink()
            print(f'resume disabled: cleared existing recipe prompt cache at {cache_path}', flush=True)

    prompt_library_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    skipped_ops = set(args.skip_operators)

    for track in args.tracks:
        input_path = benchmark_dir / TRACK_FILES[track]
        if not input_path.exists():
            print(f'skip missing {track}: {input_path}', flush=True)
            continue
        rows = _read_jsonl(input_path)
        if not rows:
            raise SystemExit(
                f'No benchmark rows found for track={track}: {input_path}. '
                'Refusing to overwrite the recipe prompt library with an empty run.'
            )
        grouped, skipped_count = _group_rows(rows, skipped_ops)
        track_generated_count = 0
        track_accepted_count = 0
        accepted_recipe_count = 0
        total_recipes = len(grouped)
        print(
            f"start track={track} input_rows={len(rows)} recipes={total_recipes} skipped_rows={skipped_count}",
            flush=True,
        )

        for recipe_index, (recipe_key, recipe_rows) in enumerate(grouped.items(), start=1):
            bundle = _recipe_bundle(
                recipe_key,
                recipe_rows,
                prompt_cfg,
                args.variants_per_recipe,
                args.candidates_per_style,
            )
            cache_key = _cache_key(
                bundle,
                generation_model,
                judge_model,
                args.prompt_source,
                args.min_average_score,
                args.skip_judge,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                library_entry = cached
                source = 'cache'
            else:
                try:
                    candidates = (
                        _call_llm_for_candidates(
                            bundle=bundle,
                            client=generation_client,
                            model=generation_model,
                            temperature=args.temperature,
                        )
                        if args.prompt_source == 'llm'
                        else _template_candidates(bundle, prompt_cfg)
                    )
                    library_entry = _build_library_entry(
                        bundle=bundle,
                        candidates=candidates,
                        prompt_source=args.prompt_source,
                        generation_model=generation_model,
                        judge_model=judge_model,
                        min_average_score=args.min_average_score,
                        client=judge_client,
                        judge_temperature=args.judge_temperature,
                        skip_judge=args.skip_judge,
                    )
                    cache_row = {
                        'cache_key': cache_key,
                        'recipe_prompt_key': recipe_key,
                        'library_entry': library_entry,
                    }
                    _append_jsonl(cache_path, cache_row)
                    cache[cache_key] = library_entry
                    source = args.prompt_source
                except Exception as exc:
                    print(
                        f"recipe failed; continuing track={track} recipe={recipe_key} error={exc}",
                        flush=True,
                    )
                    library_entry = _failed_library_entry(
                        bundle=bundle,
                        prompt_source=args.prompt_source,
                        generation_model=generation_model,
                        judge_model=judge_model,
                        min_average_score=args.min_average_score,
                        error=exc,
                    )
                    source = 'error'

            library_entry = {
                **library_entry,
                'prompt_source': library_entry.get('prompt_source') or source,
            }
            prompt_library_rows.append(library_entry)
            track_generated_count += int(library_entry.get('generated_candidate_count', 0) or 0)
            track_accepted_count += int(library_entry.get('accepted_candidate_count', 0) or 0)
            if int(library_entry.get('accepted_candidate_count', 0) or 0) > 0:
                accepted_recipe_count += 1
            print(
                f"progress track={track} recipe={recipe_index}/{total_recipes} "
                f"source={source} accepted={library_entry.get('accepted_candidate_count', 0)} "
                f"accepted_styles={library_entry.get('accepted_style_count', 0)} "
                f"key={recipe_key}",
                flush=True,
            )

        summary_rows.append(
            {
                'track': track,
                'input_rows': len(rows),
                'skipped_rows': skipped_count,
                'recipe_count': len(grouped),
                'accepted_recipe_count': accepted_recipe_count,
                'variants_per_recipe': args.variants_per_recipe,
                'candidates_per_style': args.candidates_per_style,
                'generated_candidate_count': track_generated_count,
                'accepted_candidate_count': track_accepted_count,
                'generation_model': generation_model,
                'judge_model': judge_model,
            }
        )

    if not prompt_library_rows:
        raise SystemExit(
            'No prompt library rows were produced. Refusing to overwrite '
            f'{output_dir / "recipe_prompt_library.jsonl"} with an empty file.'
        )
    _write_jsonl(output_dir / 'recipe_prompt_library.jsonl', prompt_library_rows)
    _write_jsonl(output_dir / 'prompt_generation_summary.jsonl', summary_rows)
    print(f'wrote recipe prompt library -> {output_dir / "recipe_prompt_library.jsonl"}', flush=True)
    print(f'wrote summary -> {output_dir / "prompt_generation_summary.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
