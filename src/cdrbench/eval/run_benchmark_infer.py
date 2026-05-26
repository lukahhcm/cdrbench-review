#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from cdrbench.eval.prediction_io import ordered_prediction_row
from cdrbench.infer.openai_infer import make_api_infer, make_vllm_infer
from cdrbench.llm_utils import parse_json_response, resolve_api_key, resolve_base_url, resolve_model
from cdrbench.prepare_data.build_benchmark_release import RELEASE_FIELD_ORDER


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_CONFIG = ROOT / 'configs' / 'recipe_prompting.yaml'
DEFAULT_FEW_SHOT_SOURCE_ROOT = ROOT / 'data' / 'benchmark_full'
DEFAULT_PROGRESS_EVERY = 20
LOCAL_HOSTS = {'127.0.0.1', '0.0.0.0', '::1', 'localhost'}
PROMPT_MODES = ('direct', 'few_shot', 'plan_first', 'state_aware')
FATAL_REQUEST_ERROR_PATTERNS = (
    re.compile(r'\b400\b'),
    re.compile(r'\b401\b'),
    re.compile(r'\b403\b'),
    re.compile(r'\b404\b'),
    re.compile(r'unauthorized', re.IGNORECASE),
    re.compile(r'forbidden', re.IGNORECASE),
    re.compile(r'invalid[^a-z0-9]+api[^a-z0-9]+key', re.IGNORECASE),
    re.compile(r'incorrect[^a-z0-9]+api[^a-z0-9]+key', re.IGNORECASE),
    re.compile(r'api[^a-z0-9]+key[^a-z0-9]+invalid', re.IGNORECASE),
    re.compile(r'permission[^a-z0-9]+denied', re.IGNORECASE),
    re.compile(r'model[^a-z0-9]+not[^a-z0-9]+found', re.IGNORECASE),
    re.compile(r'not[^a-z0-9]+found', re.IGNORECASE),
    re.compile(r'no[^a-z0-9]+such[^a-z0-9]+model', re.IGNORECASE),
)
NON_SCORING_REFUSAL_PATTERNS = (
    re.compile(r'数据不合规'),
    re.compile(r'内容不合规'),
    re.compile(r'输入不合规'),
    re.compile(r'不符合规范'),
    re.compile(r'无法处理.*合规'),
    re.compile(r'cannot comply', re.IGNORECASE),
    re.compile(r'cannot process', re.IGNORECASE),
    re.compile(r"can't assist", re.IGNORECASE),
    re.compile(r'cannot assist', re.IGNORECASE),
    re.compile(r'not compliant', re.IGNORECASE),
    re.compile(r'content policy', re.IGNORECASE),
    re.compile(r'safety policy', re.IGNORECASE),
    re.compile(r'usage policy', re.IGNORECASE),
    re.compile(r'violates? policy', re.IGNORECASE),
    re.compile(r'policy restrictions?', re.IGNORECASE),
    re.compile(r'invalid prompt.*flagged', re.IGNORECASE),
    re.compile(r'potentially violating', re.IGNORECASE),
)
def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = '||'.join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    preview = line.strip()
                    if len(preview) > 200:
                        preview = preview[:200] + '...'
                    raise SystemExit(
                        f'invalid JSONL in {path} at line {line_no}: {exc}. '
                        f'Line preview: {preview}'
                    ) from exc
    return rows


def _resolve_track_source_path(source_root: Path, track_name: str) -> Path:
    filename = f'{track_name}.jsonl'
    direct = source_root / track_name / filename
    if direct.exists():
        return direct
    fallback = source_root / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f'few-shot source file not found for track={track_name}: {direct} or {fallback}')


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(ordered_prediction_row(row), ensure_ascii=False) + '\n')
    tmp_path.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    tmp_path.replace(path)


def _write_progress_snapshot(output_path: Path, output_dir: Path, output_rows: list[dict[str, Any]], model: str, base_url: str, track_name: str) -> None:
    _write_jsonl(output_path, output_rows)
    summary = {
        'track': track_name,
        'model': model,
        'base_url': base_url,
        'num_instances': len(output_rows),
        'predictions_path': str(output_path),
    }
    _write_json(_summary_path_for_predictions(output_path), summary)


def _summary_path_for_predictions(output_path: Path) -> Path:
    return output_path.with_name(f'{output_path.stem}.summary.json')


def _raise_with_snapshot(
    *,
    output_path: Path,
    output_dir: Path,
    output_rows: list[dict[str, Any]],
    model: str,
    base_url: str,
    track_name: str,
    message: str,
) -> None:
    _write_progress_snapshot(output_path, output_dir, output_rows, model, base_url, track_name)
    raise SystemExit(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _copy_recipe_identity_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'recipe_id': _first_present(row, 'recipe_id', 'workflow_id'),
        'recipe_variant_id': _first_present(row, 'recipe_variant_id', 'workflow_variant_id'),
        'recipe_type': _first_present(row, 'recipe_type', 'workflow_type'),
    }


def _resolved_api_key(explicit_api_key: str | None, base_url: str) -> str:
    if explicit_api_key:
        return explicit_api_key
    host = (urlparse(base_url).hostname or '').strip().lower()
    if host in LOCAL_HOSTS:
        return 'EMPTY'
    return resolve_api_key(None)


def _prompt_style_config(prompt_cfg: dict[str, Any]) -> dict[str, Any]:
    prompt_styles = prompt_cfg.get('prompt_styles') or {}
    style_cfg = prompt_styles.get('user_natural_v1') or {}
    return style_cfg if isinstance(style_cfg, dict) else {}


def _default_system_prompt(prompt_cfg: dict[str, Any]) -> str:
    style_cfg = _prompt_style_config(prompt_cfg)
    prompt = style_cfg.get('system_prompt')
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return (
        'You are a careful data refinement engine. '
        'Follow the user request exactly and in order. '
        'Return only the required tagged output.'
    )


def _tagged_output_hint(prompt_cfg: dict[str, Any]) -> str:
    style_cfg = _prompt_style_config(prompt_cfg)
    output_contract = style_cfg.get('output_contract') or {}
    hint = output_contract.get('tagged_output_hint') if isinstance(output_contract, dict) else None
    if isinstance(hint, str) and hint.strip():
        return hint.strip()
    return '<status>KEEP</status><clean_text>...</clean_text> or <status>DROP</status><clean_text>...</clean_text>'


def _base_inference_row(eval_row: dict[str, Any]) -> dict[str, Any]:
    output_row = {field: eval_row[field] for field in RELEASE_FIELD_ORDER if field in eval_row}
    for key, value in _copy_recipe_identity_fields(eval_row).items():
        if value is not None:
            output_row.setdefault(key, value)
    recipe_prompt_key = _first_present(eval_row, 'recipe_prompt_key', 'workflow_prompt_key')
    if recipe_prompt_key is not None:
        output_row['recipe_prompt_key'] = recipe_prompt_key
    return output_row


def _available_prompt_variant_indices(row: dict[str, Any]) -> list[int]:
    prompt_variants = row.get('prompt_variants')
    if isinstance(prompt_variants, list) and prompt_variants:
        return list(range(len(prompt_variants)))
    if isinstance(row.get('user_requirement'), str) and str(row.get('user_requirement')).strip():
        return [0]
    raise RuntimeError(f'No prompt variants available for instance_id={row.get("instance_id")}.')


def _parse_prompt_variant_indices(value: str | None, row: dict[str, Any]) -> list[int]:
    available = _available_prompt_variant_indices(row)
    if value is None or not value.strip():
        return [0]
    if value.strip().lower() == 'all':
        return available
    requested = []
    for part in value.split(','):
        token = part.strip()
        if not token:
            continue
        index = int(token)
        if index not in available:
            raise IndexError(
                f'prompt_variant_index={index} is out of range for instance_id={row.get("instance_id")} with available={available}'
            )
        requested.append(index)
    return sorted(set(requested))


def _sample_prompt_variant_indices(
    row: dict[str, Any],
    *,
    sample_size: int,
    sample_seed: int,
) -> list[int]:
    available = _available_prompt_variant_indices(row)
    if sample_size <= 0 or sample_size >= len(available):
        return available
    recipe_prompt_key = str(_first_present(row, 'recipe_prompt_key', 'workflow_prompt_key') or '')
    instance_id = str(row.get('instance_id') or '')
    ranked = sorted(
        available,
        key=lambda index: _stable_id(
            'prompt-variant-sample',
            sample_seed,
            recipe_prompt_key,
            instance_id,
            index,
        ),
    )
    return sorted(ranked[:sample_size])


def _select_prompt_variant(row: dict[str, Any], prompt_variant_index: int) -> dict[str, Any]:
    prompt_variants = row.get('prompt_variants')
    if isinstance(prompt_variants, list) and prompt_variants:
        selected = prompt_variants[prompt_variant_index]
        if isinstance(selected, dict):
            return selected
    user_requirement = row.get('user_requirement')
    if isinstance(user_requirement, str) and user_requirement.strip():
        return {
            'style_id': str(row.get('style_id') or 'single_prompt'),
            'style_label': str(row.get('style_label') or 'Single Prompt'),
            'user_requirement': user_requirement.strip(),
        }
    raise RuntimeError(f'No usable prompt variant found for instance_id={row.get("instance_id")}.')


def _render_user_prompt(row: dict[str, Any], user_requirement: str, output_hint: str) -> str:
    return (
        f"Task:\n{user_requirement}\n\n"
        "Raw input text:\n"
        "<input>\n"
        f"{str(row.get('input_text', ''))}\n"
        "</input>\n\n"
        "Return tagged output only.\n"
        f"Use exactly this format: {output_hint}\n"
        "Rules:\n"
        "- status must be KEEP or DROP inside <status>...</status>.\n"
        "- Put the output text inside <clean_text>...</clean_text>.\n"
        "- If status is KEEP, clean_text must be the final refined text.\n"
        "- If status is DROP, clean_text must be the text state at the point where the sample is rejected.\n"
        "- Preserve backslashes exactly as plain text; do not JSON-escape them.\n"
        "- Do not output markdown, code fences, or explanations.\n"
    )


def _reference_tagged_output(row: dict[str, Any]) -> str:
    return (
        f"<status>{str(row.get('reference_status') or '').strip().upper()}</status>"
        f"<clean_text>{str(row.get('reference_text') or '')}</clean_text>"
    )


def _selection_key(row: dict[str, Any]) -> str:
    if str(row.get('benchmark_track') or '') == 'atomic':
        return str(row.get('operator') or '')
    benchmark_track = str(row.get('benchmark_track') or '')
    if benchmark_track == 'order_sensitivity':
        return str(row.get('order_family_id') or '')
    return str(row.get('recipe_variant_id') or '')


def _select_variant_for_style(row: dict[str, Any], preferred_style_id: str | None) -> dict[str, Any]:
    prompt_variants = row.get('prompt_variants')
    if isinstance(prompt_variants, list) and prompt_variants:
        if preferred_style_id:
            for variant in prompt_variants:
                if isinstance(variant, dict) and str(variant.get('style_id') or '') == preferred_style_id:
                    return variant
        for variant in prompt_variants:
            if isinstance(variant, dict):
                return variant
    user_requirement = str(row.get('user_requirement') or '').strip()
    if user_requirement:
        return {
            'style_id': str(row.get('style_id') or 'single_prompt'),
            'style_label': str(row.get('style_label') or 'Single Prompt'),
            'user_requirement': user_requirement,
        }
    return {'style_id': '', 'style_label': '', 'user_requirement': ''}


def _few_shot_examples(
    candidate_rows: list[dict[str, Any]],
    *,
    preferred_style_id: str,
    max_examples: int = 2,
) -> list[dict[str, str]]:
    def few_shot_rank(candidate: dict[str, Any]) -> tuple[int, int, str, str]:
        input_length = int(candidate.get('input_length_chars', 0) or 0)
        reference_length = len(str(candidate.get('reference_text') or ''))
        return (
            input_length if input_length > 0 else len(str(candidate.get('input_text') or '')),
            reference_length,
            str(candidate.get('source_record_id') or ''),
            str(candidate.get('instance_id') or ''),
        )

    chosen_rows = sorted(candidate_rows, key=few_shot_rank)[:max_examples]
    examples = []
    for candidate in chosen_rows:
        variant = _select_variant_for_style(candidate, preferred_style_id)
        examples.append(
            {
                'user_requirement': str(variant.get('user_requirement') or '').strip(),
                'input_text': str(candidate.get('input_text') or ''),
                'reference_output': _reference_tagged_output(candidate),
            }
        )
    return examples


def _format_few_shot_prompt(
    row: dict[str, Any],
    user_requirement: str,
    output_hint: str,
    *,
    examples: list[dict[str, str]],
) -> str:
    sections = []
    for index, example in enumerate(examples, start=1):
        sections.append(
            "\n".join(
                [
                    f"Example {index}",
                    "Task:",
                    example['user_requirement'],
                    "",
                    "Raw input text:",
                    "<input>",
                    example['input_text'],
                    "</input>",
                    "",
                    "Correct output:",
                    example['reference_output'],
                ]
            )
        )
    sections.append(_render_user_prompt(row, user_requirement, output_hint))
    return "\n\n".join(section for section in sections if section.strip())


def _format_filter_rule(op_name: str, params: Any) -> str:
    if isinstance(params, dict) and params:
        detail = ', '.join(f'{key}={value}' for key, value in sorted(params.items()))
        return f'{op_name} with {detail}'
    return op_name


def _execution_steps(row: dict[str, Any]) -> list[str]:
    operator_sequence = row.get('operator_sequence')
    if isinstance(operator_sequence, list) and operator_sequence:
        sequence = [str(item) for item in operator_sequence if str(item).strip()]
    else:
        sequence = [str(row.get('operator') or '').strip()] if str(row.get('operator') or '').strip() else []
    filter_params = row.get('filter_params_by_name') if isinstance(row.get('filter_params_by_name'), dict) else {}
    if not sequence:
        return ["1. Step 1: Apply the requested refinement.\n   Input state: S0\n   Output state: S1"]
    lines = ["Execution path:"]
    current_state_index = 0
    for index, op_name in enumerate(sequence, start=1):
        params = filter_params.get(op_name)
        if op_name.endswith('_filter'):
            input_state = f'S{current_state_index}'
            lines.append(
                "\n".join(
                    [
                        f"{index}. Step {index}: Filter at {input_state} using { _format_filter_rule(op_name, params) }",
                        f"   Input state: {input_state}",
                        f"   Output: KEEP and remain at {input_state}, or DROP at {input_state}",
                    ]
                )
            )
        else:
            input_state = f'S{current_state_index}'
            current_state_index += 1
            output_state = f'S{current_state_index}'
            lines.append(
                "\n".join(
                    [
                        f"{index}. Step {index}: Apply { _format_filter_rule(op_name, params) }",
                        f"   Input state: {input_state}",
                        f"   Output state: {output_state}",
                    ]
                )
            )
    return lines


def _format_plan_first_prompt(row: dict[str, Any], user_requirement: str, output_hint: str) -> str:
    return (
        f"Task:\n{user_requirement}\n\n"
        "Before cleaning the text, first write a short analysis block in <analyze>...</analyze>.\n"
        "Use that block to restate the requested procedure as a concise standardized execution recipe.\n"
        "List the steps in order. For each step, say what to do and include the relevant rule or threshold when there is one.\n\n"
        "Example analysis format:\n"
        "<analyze>\n"
        "Step 1: Remove repeated sentences.\n"
        "Step 2: Normalize whitespace.\n"
        "Step 3: Apply the word repetition filter with ratio <= 0.2, and drop the text if it fails.\n"
        "</analyze>\n\n"
        "Raw input text:\n"
        "<input>\n"
        f"{str(row.get('input_text', ''))}\n"
        "</input>\n\n"
        "Then execute the procedure on the text.\n"
        f"After the analysis block, output the final tagged result using exactly this format: {output_hint}\n"
        "Rules:\n"
        "- Output exactly one <analyze>...</analyze> block before the final tagged result.\n"
        "- Inside <analyze>, write an ordered step list for the intended execution recipe.\n"
        "- For each step, include the relevant rule or threshold when there is one.\n"
        "- status must be KEEP or DROP inside <status>...</status>.\n"
        "- Put the output text inside <clean_text>...</clean_text>.\n"
        "- If status is DROP, stop at the rejection point and return the current text.\n"
        "- Do not output markdown, code fences, or explanations outside <analyze>.\n"
    )


def _format_state_aware_prompt(row: dict[str, Any], user_requirement: str, output_hint: str) -> str:
    return (
        f"Task:\n{user_requirement}\n\n"
        "Before cleaning the text, first write a short analysis block in <analyze>...</analyze>.\n"
        "Use that block to identify the intermediate text states that matter for correctness.\n"
        "You may refer to them with names such as S0, S1, and S2.\n"
        "Only text-changing steps should create a new state, while filter steps should be described as operating on the relevant existing state.\n"
        "Focus on which operation or filter should be applied to which state, especially when order matters.\n\n"
        "Example analysis format:\n"
        "<analyze>\n"
        "A useful state view is S0 = raw text and S1 = text after repeated-sentence removal.\n"
        "The key risk is applying the repetition filter on S0 instead of S1; this filter should be evaluated on S1 because the cleanup step changes the statistics.\n"
        "</analyze>\n\n"
        "In that analysis:\n"
        "- refer only to the states or transitions most likely to change the result if used incorrectly;\n"
        "- emphasize the specific state where an important filter or operation must be applied;\n"
        "- do not analyze every step if most steps are unambiguous.\n\n"
        "Then apply the intended procedure to the text.\n"
        "If a filter rejects the sample, stop at the last valid state and return DROP.\n\n"
        "Raw input text:\n"
        "<input>\n"
        f"{str(row.get('input_text', ''))}\n"
        "</input>\n\n"
        "After the analysis block, return the final tagged output only.\n"
        f"Use exactly this format: {output_hint}\n"
        "Rules:\n"
        "- Output exactly one <analyze>...</analyze> block before the final tagged result.\n"
        "- In <analyze>, use state language only when it helps clarify which intermediate text a key operation or filter should use.\n"
        "- Focus on state confusions that could materially change the result, especially order-sensitive filter decisions.\n"
        "- status must be KEEP or DROP inside <status>...</status>.\n"
        "- Put the output text inside <clean_text>...</clean_text>.\n"
        "- If status is KEEP, clean_text must be the final state text.\n"
        "- If status is DROP, clean_text must be the text state at the point where the sample is rejected.\n"
        "- Do not output markdown, code fences, or explanations outside <analyze>.\n"
    )


def _is_qwen_family(model_name: str) -> bool:
    normalized = re.sub(r'[^a-z0-9]+', '', model_name.strip().lower())
    return 'qwen' in normalized


def _is_glm5_family(model_name: str) -> bool:
    normalized = re.sub(r'[^a-z0-9]+', '', model_name.strip().lower())
    return 'glm5' in normalized


def _is_deepseek_family(model_name: str) -> bool:
    normalized = re.sub(r'[^a-z0-9]+', '', model_name.strip().lower())
    return 'deepseekv4pro' in normalized or 'deepseekv4flash' in normalized


def _is_kimi_family(model_name: str) -> bool:
    normalized = re.sub(r'[^a-z0-9]+', '', model_name.strip().lower())
    return 'kimik25' in normalized or 'kimik26' in normalized


def _api_extra_body_for_model(model_name: str, *, enable_thinking: bool | None = None) -> dict[str, Any]:
    if _is_qwen_family(model_name):
        return {'enable_thinking': bool(enable_thinking) if enable_thinking is not None else False}
    if _is_kimi_family(model_name):
        return {'enable_thinking': bool(enable_thinking) if enable_thinking is not None else False}
    if _is_deepseek_family(model_name):
        return {'thinking': {'type': 'disabled'}}
    if _is_glm5_family(model_name):
        return {'thinking': {'type': 'disabled'}}
    return {}


def _final_user_prompt(
    row: dict[str, Any],
    user_requirement: str,
    output_hint: str,
    model_name: str,
    *,
    prompt_mode: str,
    few_shot_examples: list[dict[str, str]] | None = None,
) -> str:
    if prompt_mode == 'few_shot':
        return _format_few_shot_prompt(row, user_requirement, output_hint, examples=few_shot_examples or [])
    if prompt_mode == 'plan_first':
        return _format_plan_first_prompt(row, user_requirement, output_hint)
    if prompt_mode == 'state_aware':
        return _format_state_aware_prompt(row, user_requirement, output_hint)
    return _render_user_prompt(row, user_requirement, output_hint)


def _extract_tagged_prediction_payload(response_text: str) -> tuple[dict[str, Any] | None, str | None]:
    tagged_matches = list(
        re.finditer(
            r'<status>\s*(KEEP|DROP)\s*</status>\s*<clean_text>(.*?)</clean_text>',
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if tagged_matches:
        match = tagged_matches[-1]
        return {
            'status': str(match.group(1)).strip().upper(),
            'clean_text': str(match.group(2)),
            'output_format': 'tagged_v1',
        }, None

    status_matches = list(
        re.finditer(r'<status>\s*(KEEP|DROP)\s*</status>', response_text, flags=re.IGNORECASE | re.DOTALL)
    )
    if not status_matches:
        return None, 'tag_parse_error'
    status_match = status_matches[-1]
    clean_open_matches = [
        match
        for match in re.finditer(r'<clean_text\s*>', response_text, flags=re.IGNORECASE | re.DOTALL)
        if match.start() >= status_match.end()
    ]
    if not clean_open_matches:
        return None, 'tag_parse_error'
    clean_open_match = clean_open_matches[-1]
    clean_text = response_text[clean_open_match.end() :]
    return {
        'status': str(status_match.group(1)).strip().upper(),
        'clean_text': clean_text,
        'output_format': 'tagged_v1_open_ended',
    }, 'incomplete_tag_error: clean_text_missing_closing_tag'


def _extract_prediction_payload(response_text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not response_text.strip():
        return None, 'empty_response'
    tagged_payload, tagged_error = _extract_tagged_prediction_payload(response_text)
    if tagged_error is None:
        return tagged_payload, None
    if any(pattern.search(response_text) for pattern in NON_SCORING_REFUSAL_PATTERNS):
        return None, 'non_scoring_refusal: policy_or_compliance_refusal'
    try:
        payload = parse_json_response(response_text)
    except Exception as exc:
        return None, f'tag_parse_error; json_parse_error: {exc}'
    if not isinstance(payload, dict):
        return None, 'tag_parse_error; json_parse_error: response is not a JSON object'
    return payload, None


def _is_retryable_prediction_error(prediction_error: str | None) -> bool:
    if not prediction_error:
        return False
    return (
        prediction_error == 'empty_response'
        or prediction_error.startswith('json_parse_error:')
        or prediction_error.startswith('tag_parse_error')
        or prediction_error.startswith('incomplete_tag_error:')
    )


def _extract_prediction_fields(prediction_payload: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(prediction_payload, dict):
        return '', ''
    status_value = prediction_payload.get('predicted_status', prediction_payload.get('status'))
    text_value = prediction_payload.get('predicted_clean_text', prediction_payload.get('clean_text', prediction_payload.get('text')))
    return ('' if status_value is None else str(status_value), '' if text_value is None else str(text_value))


def _has_valid_prediction(prediction_payload: dict[str, Any] | None) -> bool:
    if not isinstance(prediction_payload, dict):
        return False
    has_status = prediction_payload.get('predicted_status', prediction_payload.get('status')) is not None
    has_text = (
        prediction_payload.get('predicted_clean_text') is not None
        or prediction_payload.get('clean_text') is not None
        or prediction_payload.get('text') is not None
    )
    return has_status and has_text


def _has_nonempty_prediction_fields(variant: dict[str, Any] | None) -> bool:
    if not isinstance(variant, dict):
        return False
    predicted_status = '' if variant.get('predicted_status') is None else str(variant.get('predicted_status')).strip()
    predicted_clean_text = '' if variant.get('predicted_clean_text') is None else str(variant.get('predicted_clean_text'))
    return bool(predicted_status) and bool(predicted_clean_text)


def _normalize_existing_variant_prediction(variant: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(variant)
    has_fields = _has_nonempty_prediction_fields(normalized)
    valid_prediction = bool(
        normalized.get('valid_prediction')
        if 'valid_prediction' in normalized
        else normalized.get('prediction_valid_json')
    )
    if has_fields:
        valid_prediction = True
        normalized['prediction_error'] = None
    normalized['valid_prediction'] = valid_prediction
    normalized.pop('prediction_valid_json', None)
    return normalized


def _existing_variant_prediction_map(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    variants = row.get('variant_predictions')
    if not isinstance(variants, list):
        return {}
    mapping: dict[int, dict[str, Any]] = {}
    for variant in variants:
        if isinstance(variant, dict):
            normalized_variant = _normalize_existing_variant_prediction(variant)
            mapping[int(normalized_variant.get('prompt_variant_index', 0) or 0)] = normalized_variant
    return mapping


def _normalize_existing_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    variant_map = _existing_variant_prediction_map(normalized)
    if variant_map:
        normalized['variant_predictions'] = [variant_map[key] for key in sorted(variant_map)]
    return normalized


def _variant_prediction_completed_successfully(variant: dict[str, Any] | None) -> bool:
    if not isinstance(variant, dict):
        return False
    if bool(variant.get('valid_prediction')):
        return True
    prediction_error = str(variant.get('prediction_error') or '')
    if prediction_error.startswith('non_scoring_refusal:'):
        return True
    if variant.get('prediction_error') is None:
        return True
    return _has_nonempty_prediction_fields(variant)


def _variant_prediction_should_skip_on_resume(variant: dict[str, Any] | None) -> bool:
    return _variant_prediction_completed_successfully(variant)


def _row_input_length_chars(row: dict[str, Any]) -> int:
    value = row.get('input_length_chars')
    if isinstance(value, int):
        return value
    return len(str(row.get('input_text', '')))


def _build_infer_backend(args: argparse.Namespace, model: str, base_url: str, api_key: str) -> Any:
    common_kwargs = {
        'model': model,
        'api_base': base_url,
        'concurrency': max(1, int(args.concurrency)),
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'num_runs': 1,
        'max_retries': 1,
        'retry_delay': float(args.retry_sleep_seconds),
    }
    host = (urlparse(base_url).hostname or '').strip().lower()
    if host in LOCAL_HOSTS:
        return make_vllm_infer(
            **common_kwargs,
            enable_thinking=bool(args.enable_thinking),
        )
    return make_api_infer(
        **common_kwargs,
        api_key=api_key,
        extra_body=_api_extra_body_for_model(model, enable_thinking=args.enable_thinking),
    )


def _is_fatal_request_error(error_text: str | None) -> bool:
    if not error_text:
        return False
    return any(pattern.search(error_text) for pattern in FATAL_REQUEST_ERROR_PATTERNS)


def _log_prediction_issue(
    *,
    track_name: str,
    instance_id: str,
    prompt_variant_index: int,
    prediction_error: str | None,
    response_text: str,
) -> None:
    if not prediction_error:
        return
    response_preview = response_text.strip().replace('\n', ' ')
    if len(response_preview) > 240:
        response_preview = response_preview[:240] + '...'
    print(
        f'warn infer track={track_name} instance_id={instance_id or "UNKNOWN"} '
        f'prompt_variant_index={prompt_variant_index} error={prediction_error} '
        f'response_preview={response_preview!r}',
        flush=True,
    )


def _retry_request_error(
    *,
    infer_backend: Any,
    initial_result: Any,
    messages: list[dict[str, str]],
    track_name: str,
    instance_id: str,
    prompt_variant_index: int,
    max_request_tries: int,
) -> tuple[Any, int]:
    result = initial_result
    attempts = 1
    while result.error is not None and attempts < max(1, max_request_tries):
        attempts += 1
        print(
            f'retry request track={track_name} instance_id={instance_id or "UNKNOWN"} '
            f'prompt_variant_index={prompt_variant_index} attempt={attempts}/{max(1, max_request_tries)} '
            f'error={result.error}',
            flush=True,
        )
        result = infer_backend.infer_one(messages)
    return result, attempts


def _chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    size = max(1, chunk_size)
    return [items[index : index + size] for index in range(0, len(items), size)]


def _resume_seed_output_path(output_path: Path, *, sample_size: int) -> Path | None:
    if sample_size <= 0:
        return None
    stem = output_path.stem
    match = re.match(r'^(predictions?_.+?)_k\d+_seed\d+$', stem)
    if not match:
        return None
    return output_path.with_name(match.group(1) + output_path.suffix)


def _load_existing_rows_by_id(paths: list[Path]) -> dict[str, dict[str, Any]]:
    existing_rows_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            row = _normalize_existing_prediction_row(row)
            instance_id = str(row.get('instance_id') or '')
            if instance_id:
                existing_rows_by_id[instance_id] = row
    return existing_rows_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description='Run CDR-Bench inference on eval-ready JSONL and save raw predictions.')
    parser.add_argument('--eval-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--prompt-config', default=str(DEFAULT_PROMPT_CONFIG.relative_to(ROOT)))
    parser.add_argument('--model', default=None)
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--max-tokens', type=int, default=0)
    parser.add_argument('--prompt-variant-index', type=int, default=0)
    parser.add_argument('--prompt-variant-indices', default=None)
    parser.add_argument('--prompt-variant-sample-size', type=int, default=0)
    parser.add_argument('--prompt-variant-sampling-seed', type=int, default=0)
    parser.add_argument('--prompt-mode', choices=PROMPT_MODES, default='direct')
    parser.add_argument('--few-shot-source-root', default=str(DEFAULT_FEW_SHOT_SOURCE_ROOT))
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-input-chars', type=int, default=0)
    parser.add_argument('--max-retries', type=int, default=4)
    parser.add_argument('--retry-sleep-seconds', type=float, default=2.0)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--progress-every', type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument('--stop-on-parse-error', action='store_true')
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        '--enable-thinking',
        dest='enable_thinking',
        action='store_true',
        help='Enable model thinking mode where supported, e.g. Qwen chat_template_kwargs.enable_thinking for vLLM.',
    )
    thinking_group.add_argument(
        '--disable-thinking',
        dest='enable_thinking',
        action='store_false',
        help='Disable model thinking mode where supported. This is the default.',
    )
    parser.set_defaults(enable_thinking=False)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--resume-only-existing-rows',
        action='store_true',
        help='When resuming, restrict inference to instance_ids already present in the existing output file.',
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_dir = output_path.parent
    prompt_cfg = _load_yaml((ROOT / args.prompt_config).resolve())
    system_prompt = _default_system_prompt(prompt_cfg)
    output_hint = _tagged_output_hint(prompt_cfg)
    rows = _read_jsonl(eval_path)
    few_shot_candidate_rows: list[dict[str, Any]] = []
    if args.prompt_mode == 'few_shot':
        track_name = eval_path.stem or 'unknown'
        few_shot_source_root = Path(args.few_shot_source_root).resolve()
        few_shot_source_path = _resolve_track_source_path(few_shot_source_root, track_name)
        selected_instance_ids = {
            str(row.get('instance_id') or '').strip()
            for row in rows
            if str(row.get('instance_id') or '').strip()
        }
        few_shot_candidate_rows = [
            row
            for row in _read_jsonl(few_shot_source_path)
            if str(row.get('instance_id') or '').strip() not in selected_instance_ids
        ]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if args.max_input_chars > 0:
        rows = [row for row in rows if _row_input_length_chars(row) <= args.max_input_chars]
    if not rows:
        raise SystemExit(
            f'No eval rows to infer after filtering: eval_path={eval_path} '
            f'max_samples={args.max_samples} max_input_chars={args.max_input_chars}. '
            'Check that the benchmark track JSONL is populated.'
        )

    model = resolve_model(args.model)
    base_url = resolve_base_url(args.base_url, model=model)
    api_key = _resolved_api_key(args.api_key, base_url)
    infer_backend = _build_infer_backend(args, model, base_url, api_key)

    existing_rows_by_id: dict[str, dict[str, Any]] = {}
    resume_source_paths: list[Path] = []
    fallback_output_path = None
    if args.resume:
        fallback_output_path = _resume_seed_output_path(
            output_path,
            sample_size=int(args.prompt_variant_sample_size),
        )
        if fallback_output_path is not None and fallback_output_path.exists():
            resume_source_paths.append(fallback_output_path)
        if output_path.exists():
            resume_source_paths.append(output_path)
        existing_rows_by_id = _load_existing_rows_by_id(resume_source_paths)
        if fallback_output_path is not None and fallback_output_path.exists():
            print(
                f'resume seed source detected for sampled infer: '
                f'output_path={output_path} fallback_resume_source_path={fallback_output_path} '
                f'active_resume_sources={[str(path) for path in resume_source_paths]}',
                flush=True,
            )
    if args.resume_only_existing_rows:
        if not args.resume:
            raise SystemExit('--resume-only-existing-rows requires --resume.')
        if not resume_source_paths:
            raise SystemExit(
                '--resume-only-existing-rows requires at least one existing resume source file; '
                f'checked output_path={output_path} fallback_output_path={fallback_output_path}'
            )
        existing_instance_ids = set(existing_rows_by_id)
        if not existing_instance_ids:
            raise SystemExit(
                '--resume-only-existing-rows requires at least one existing prediction row; '
                f'no instance_ids found in resume sources: {[str(path) for path in resume_source_paths]}'
            )
        original_row_count = len(rows)
        rows = [row for row in rows if str(row.get('instance_id') or '') in existing_instance_ids]
        print(
            f'resume scope limited to existing predictions rows={len(rows)}/{original_row_count} '
            f'resume_source_paths={[str(path) for path in resume_source_paths]} output_path={output_path}',
            flush=True,
        )

    output_rows = []
    variant_jobs: list[tuple[int, dict[str, Any], int, dict[str, Any], dict[int, dict[str, Any]], list[int]]] = []
    started = time.time()
    for row_index, row in enumerate(rows):
        instance_id = str(row.get('instance_id') or '')
        if args.prompt_variant_sample_size > 0:
            if args.prompt_variant_indices not in (None, '', 'all') or args.prompt_variant_index != 0:
                raise SystemExit(
                    'Do not combine --prompt-variant-sample-size with explicit prompt variant index selection. '
                    'Use either sampling or --prompt-variant-indices/--prompt-variant-index.'
                )
            selected_prompt_variant_indices = _sample_prompt_variant_indices(
                row,
                sample_size=args.prompt_variant_sample_size,
                sample_seed=args.prompt_variant_sampling_seed,
            )
        else:
            selected_prompt_variant_indices = _parse_prompt_variant_indices(
                args.prompt_variant_indices if args.prompt_variant_indices is not None else str(args.prompt_variant_index),
                row,
            )
        existing_variant_predictions = _existing_variant_prediction_map(existing_rows_by_id.get(instance_id, {}))
        variant_predictions = dict(existing_variant_predictions)
        for prompt_variant_index in selected_prompt_variant_indices:
            if args.resume and _variant_prediction_should_skip_on_resume(existing_variant_predictions.get(prompt_variant_index)):
                continue
            prompt_variant = _select_prompt_variant(row, prompt_variant_index)
            user_requirement = str(prompt_variant.get('user_requirement') or '').strip()
            few_shot_examples = None
            if args.prompt_mode == 'few_shot':
                few_shot_examples = _few_shot_examples(
                    few_shot_candidate_rows,
                    preferred_style_id=str(prompt_variant.get('style_id') or ''),
                )
            variant_jobs.append(
                (
                    row_index,
                    row,
                    prompt_variant_index,
                    {
                        'prompt_style_id': str(prompt_variant.get('style_id') or ''),
                        'prompt_style_label': str(prompt_variant.get('style_label') or ''),
                        'user_requirement': user_requirement,
                        'prompt_mode': args.prompt_mode,
                        'messages': [
                            {'role': 'system', 'content': system_prompt},
                            {
                                'role': 'user',
                                'content': _final_user_prompt(
                                    row,
                                    user_requirement,
                                    output_hint,
                                    model,
                                    prompt_mode=args.prompt_mode,
                                    few_shot_examples=few_shot_examples,
                                ),
                            },
                        ],
                    },
                    variant_predictions,
                    selected_prompt_variant_indices,
                )
            )
        output_rows.append(
            {
                **_base_inference_row(row),
                'request_model': model,
                'request_base_url': base_url,
                'request_enable_thinking': bool(args.enable_thinking),
                'prompt_mode': args.prompt_mode,
                'selected_prompt_variant_indices': selected_prompt_variant_indices,
                'variant_predictions': [existing_variant_predictions[key] for key in sorted(existing_variant_predictions)],
            }
        )

    track_name = eval_path.stem or 'unknown'
    _write_progress_snapshot(output_path, output_dir, output_rows, model, base_url, track_name)
    print(
        f'start infer track={track_name} model={model} '
        f'num_rows={len(rows)} progress_every={args.progress_every} resume={bool(args.resume)} '
        f'concurrency={max(1, int(args.concurrency))} base_url={base_url} prompt_mode={args.prompt_mode}',
        flush=True,
    )

    if variant_jobs:
        processed_variants = 0
        batch_size = max(1, int(args.concurrency))

        def process_infer_result(job: Any, infer_result: Any) -> None:
            nonlocal processed_variants
            row_index, row, prompt_variant_index, meta, variant_predictions, selected_prompt_variant_indices = job
            instance_id = str(row.get('instance_id') or '')
            infer_result, request_attempts = _retry_request_error(
                infer_backend=infer_backend,
                initial_result=infer_result,
                messages=meta['messages'],
                track_name=track_name,
                instance_id=instance_id,
                prompt_variant_index=prompt_variant_index,
                max_request_tries=max(1, int(args.max_retries) + 1),
            )
            response_text = infer_result.text
            prediction_payload = None
            prediction_error = None
            stop_message = None
            if infer_result.error is not None:
                prediction_error = f'request_error: {infer_result.error}'
                if request_attempts >= max(1, int(args.max_retries) + 1):
                    print(
                        f'stop infer track={track_name} instance_id={instance_id or "UNKNOWN"} '
                        f'prompt_variant_index={prompt_variant_index} attempts={request_attempts} '
                        f'error={infer_result.error}',
                        flush=True,
                    )
                    stop_message = (
                        'request retries exhausted; stopping at current sample '
                        f'track={track_name} instance_id={instance_id or "UNKNOWN"} '
                        f'prompt_variant_index={prompt_variant_index} '
                        f'attempts={request_attempts} model={model} base_url={base_url} '
                        f'error={infer_result.error}'
                    )
            else:
                prediction_payload, prediction_error = _extract_prediction_payload(response_text)
            retry_attempted = False
            if _is_retryable_prediction_error(prediction_error):
                retry_attempted = True
                print(
                    f'retry infer track={track_name} instance_id={instance_id or "UNKNOWN"} '
                    f'prompt_variant_index={prompt_variant_index} reason={prediction_error}',
                    flush=True,
                )
                retry_result = infer_backend.infer_one(meta['messages'])
                response_text = retry_result.text
                if retry_result.error is not None:
                    prediction_payload = None
                    prediction_error = f'request_error: {retry_result.error}'
                    stop_message = (
                        'request error encountered after parse retry; stopping at current sample '
                        f'track={track_name} instance_id={instance_id or "UNKNOWN"} '
                        f'prompt_variant_index={prompt_variant_index} '
                        f'model={model} base_url={base_url} error={retry_result.error}'
                    )
                else:
                    prediction_payload, prediction_error = _extract_prediction_payload(response_text)
            _log_prediction_issue(
                track_name=track_name,
                instance_id=instance_id,
                prompt_variant_index=prompt_variant_index,
                prediction_error=prediction_error,
                response_text=response_text,
            )
            predicted_status, predicted_clean_text = _extract_prediction_fields(prediction_payload)
            valid_prediction = prediction_error is None and _has_valid_prediction(prediction_payload)
            variant_predictions[prompt_variant_index] = {
                'prompt_variant_index': prompt_variant_index,
                'prompt_style_id': meta['prompt_style_id'],
                'prompt_style_label': meta['prompt_style_label'],
                'user_requirement': meta['user_requirement'],
                'prompt_mode': meta.get('prompt_mode') or args.prompt_mode,
                'request_model': model,
                'request_base_url': base_url,
                'request_enable_thinking': bool(args.enable_thinking),
                'raw_response': response_text,
                'parsed_response': prediction_payload,
                'prediction_error': prediction_error,
                'valid_prediction': valid_prediction,
                'retry_attempted': (retry_attempted or request_attempts > 1),
                'response_usage': {},
                'predicted_status': predicted_status,
                'predicted_clean_text': predicted_clean_text,
            }
            output_rows[row_index]['variant_predictions'] = [variant_predictions[key] for key in sorted(variant_predictions)]
            output_rows[row_index]['selected_prompt_variant_indices'] = selected_prompt_variant_indices
            _write_progress_snapshot(output_path, output_dir, output_rows, model, base_url, track_name)
            if (
                prediction_error is not None
                and stop_message is None
                and args.stop_on_parse_error
                and not str(prediction_error).startswith('non_scoring_refusal:')
            ):
                stop_message = (
                    'prediction parsing failed; stopping at current sample '
                    f'track={track_name} instance_id={instance_id or "UNKNOWN"} '
                    f'prompt_variant_index={prompt_variant_index} '
                    f'model={model} base_url={base_url} error={prediction_error}'
                )
            if stop_message is not None:
                _raise_with_snapshot(
                    output_path=output_path,
                    output_dir=output_dir,
                    output_rows=output_rows,
                    model=model,
                    base_url=base_url,
                    track_name=track_name,
                    message=stop_message,
                )
            processed_variants += 1
            if processed_variants % args.progress_every == 0 or processed_variants == len(variant_jobs):
                elapsed = time.time() - started
                print(
                    f'progress infer processed_variant={processed_variants}/{len(variant_jobs)} '
                    f'elapsed_sec={elapsed:.1f}',
                    flush=True,
                )

        first_job = variant_jobs[0]
        preflight_result = infer_backend.infer_one(first_job[3]['messages'])
        process_infer_result(first_job, preflight_result)

        for batch_start in range(1, len(variant_jobs), batch_size):
            batch_jobs = variant_jobs[batch_start : batch_start + batch_size]
            batch_results = infer_backend.infer(
                [job[3]['messages'] for job in batch_jobs],
                show_progress=False,
            )
            for job, infer_result in zip(batch_jobs, batch_results):
                process_infer_result(job, infer_result)

    _write_progress_snapshot(output_path, output_dir, output_rows, model, base_url, track_name)
    print(f'wrote predictions -> {output_path}', flush=True)
    print(f'wrote summary -> {_summary_path_for_predictions(output_path)}', flush=True)


if __name__ == '__main__':
    main()
