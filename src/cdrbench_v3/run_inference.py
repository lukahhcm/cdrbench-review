from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from cdrbench_v3.io import read_jsonl, write_jsonl

SYSTEM_PROMPT = (
    "You are a careful data refinement engine. Follow the user's data refinement request "
    "exactly and in order. Return only the required tagged output. Do not explain your reasoning."
)
PROMPT_MODES = ("direct", "few_shot", "plan_first", "state_aware")

OVERSEAS_BASE_URL = "https://eval.dashscope.aliyuncs.com/compatible-mode/v1"
DOMESTIC_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LOCAL_HOSTS = {"127.0.0.1", "0.0.0.0", "::1", "localhost"}


@dataclass(frozen=True)
class ApiModelConfig:
    model_name: str
    endpoint: str
    input_field: str = "messages"
    stream: bool = True
    top_level_system: bool = False
    aliases: tuple[str, ...] = ()


API_MODEL_CONFIGS: tuple[ApiModelConfig, ...] = (
    ApiModelConfig("openai.gpt-5.4-2026-03-05", "overseas", aliases=("gpt-5.4",)),
    ApiModelConfig("openai.gpt-5.5", "overseas", aliases=("gpt-5.5",)),
    ApiModelConfig("openai.gpt-5.4-pro-2026-03-05", "overseas", input_field="input", aliases=("gpt-5.4-pro",)),
    ApiModelConfig("mr.gpt-5.5", "domestic", input_field="input", stream=False),
    ApiModelConfig("mr.gpt-5.5-pro-2026-04-23", "domestic", input_field="input", stream=False),
    ApiModelConfig("mr.gpt-5.4", "domestic", input_field="input", stream=False),
    ApiModelConfig("mr.gpt-5.4-pro", "domestic", input_field="input", stream=False),
    ApiModelConfig("aws.claude-sonnet-4-6", "overseas", stream=False, top_level_system=True, aliases=("claude-sonnet-4-6",)),
    ApiModelConfig("aws.claude-opus-4-6", "overseas", stream=False, top_level_system=True, aliases=("claude-opus-4-6",)),
    ApiModelConfig("vertex_ai.claude-opus-4-5-20251101", "overseas", stream=False, top_level_system=True, aliases=("claude-opus-4-5-20251101",)),
    ApiModelConfig("mr.claude-opus-4-6-20260205", "domestic", stream=False, top_level_system=True),
    ApiModelConfig("mr.claude-opus-4-8", "domestic", stream=False, top_level_system=True),
    ApiModelConfig("mr.claude-opus-4-7", "domestic", stream=False, top_level_system=True),
    ApiModelConfig("mr.claude-sonnet-4-6-20260217", "domestic", stream=False, top_level_system=True),
    ApiModelConfig("mr.claude-sonnet-4-5-20250929", "domestic", stream=False, top_level_system=True),
    ApiModelConfig("vertex_ai.gemini-3.1-pro-preview", "overseas", input_field="contents", aliases=("gemini-3.1-pro-preview",)),
    ApiModelConfig("vertex_ai.gemini-3-flash-preview", "overseas", input_field="contents", aliases=("gemini-3-flash-preview",)),
    ApiModelConfig("mr.gemini-3.1-pro-preview", "domestic", input_field="contents"),
    ApiModelConfig("z_ai.glm-5", "overseas", aliases=("glm-5",)),
    ApiModelConfig("qwen3.6-max-preview", "domestic"),
    ApiModelConfig("qwen3.6-plus", "domestic"),
    ApiModelConfig("deepseek-v4-pro", "domestic", aliases=("deepseek.deepseek-v4-pro",)),
    ApiModelConfig("deepseek-v4-flash", "domestic", aliases=("deepseek.deepseek-v4-flash", "deepseek_v4_flash")),
    ApiModelConfig("kimi-k2.6", "domestic", aliases=("moonshot.kimi-k2.6",)),
    ApiModelConfig("glm-5.1", "domestic"),
)

MODEL_CONFIG_BY_NAME: dict[str, ApiModelConfig] = {}
for config in API_MODEL_CONFIGS:
    for name in (config.model_name, *config.aliases):
        MODEL_CONFIG_BY_NAME[name.casefold()] = config


def _model_config(model: str) -> ApiModelConfig | None:
    return MODEL_CONFIG_BY_NAME.get(model.strip().casefold())


def _default_base_url(model: str) -> str:
    config = _model_config(model)
    if config is None:
        return DOMESTIC_BASE_URL
    return OVERSEAS_BASE_URL if config.endpoint == "overseas" else DOMESTIC_BASE_URL


def _resolved_model_name(model: str) -> str:
    config = _model_config(model)
    return config.model_name if config is not None else model


def _normalize_request_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item["text"]))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


class OpenAICompatibleBackend:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        max_retries: int,
        retry_delay: float,
        backend: str,
        enable_thinking: bool,
    ) -> None:
        self.model = _resolved_model_name(model) if backend == "api" else model
        self.config = _model_config(model) if backend == "api" else None
        self.request_url = _normalize_request_url(base_url)
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backend = backend
        self.enable_thinking = enable_thinking

    def _split_system_messages(self, messages: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        system_parts: list[str] = []
        rest: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user").lower()
            text = _stringify_content(message.get("content")).strip()
            if role == "system":
                if text:
                    system_parts.append(text)
            else:
                rest.append(message)
        return system_parts, rest

    def _messages_payload(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {"role": str(message.get("role") or "user"), "content": _stringify_content(message.get("content"))}
            for message in messages
        ]

    def _contents_payload(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parts = []
        for message in messages:
            role = str(message.get("role") or "user").lower()
            text = _stringify_content(message.get("content")).strip()
            if text:
                parts.append(f"{role}:\n{text}")
        return [{"role": "user", "parts": [{"text": "\n\n".join(parts)}]}]

    def _extra_body(self) -> dict[str, Any]:
        normalized = re.sub(r"[^a-z0-9]+", "", self.model.lower())
        if self.backend == "vllm":
            return {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}, "do_sample": False}
        if "gpt" in normalized:
            return {"reasoning": {"effort": "low"}}
        if "qwen" in normalized or "kimi" in normalized:
            return {"enable_thinking": self.enable_thinking}
        if "deepseekv4" in normalized or "glm5" in normalized:
            return {"thinking": {"type": "enabled" if self.enable_thinking else "disabled"}}
        return {}

    def _payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        config = self.config
        input_field = config.input_field if config is not None else "messages"
        stream = config.stream if config is not None else True
        effective_messages = messages
        payload: dict[str, Any] = {"model": self.model, "stream": stream}

        if config is not None and config.top_level_system:
            system_parts, effective_messages = self._split_system_messages(messages)
            if system_parts:
                payload["system"] = "\n\n".join(system_parts)

        if input_field == "input":
            payload["input"] = self._messages_payload(effective_messages)
        elif input_field == "contents":
            payload["contents"] = self._contents_payload(effective_messages)
        else:
            payload["messages"] = self._messages_payload(effective_messages)

        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        payload["temperature"] = self.temperature
        payload.update(self._extra_body())
        return payload

    @classmethod
    def _extract_text(cls, payload: dict[str, Any]) -> str:
        response = payload.get("response")
        if isinstance(response, dict):
            text = cls._extract_text(response)
            if text:
                return text
        output = payload.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text") is not None:
                            parts.append(str(part.get("text")))
                elif item.get("text") is not None:
                    parts.append(str(item.get("text")))
            if parts:
                return "".join(parts)
        if payload.get("type") == "response.output_text.delta" and payload.get("delta") is not None:
            return _stringify_content(payload.get("delta"))
        if payload.get("type") == "response.output_text.done" and payload.get("text") is not None:
            return _stringify_content(payload.get("text"))
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get("delta")
                if isinstance(delta, dict) and delta.get("content") is not None:
                    return _stringify_content(delta.get("content"))
                message = choice.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return _stringify_content(message.get("content"))
        message = payload.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            return _stringify_content(message.get("content"))
        content = payload.get("content")
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            content_obj = candidates[0].get("content") if isinstance(candidates[0], dict) else None
            parts = content_obj.get("parts") if isinstance(content_obj, dict) else None
            if isinstance(parts, list):
                return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def _extract_stream(cls, response: requests.Response) -> str:
        parts: list[str] = []
        snapshot = ""
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            payload = json.loads(data)
            text = cls._extract_text(payload)
            if text:
                if isinstance(payload.get("message"), dict):
                    snapshot = text
                else:
                    parts.append(text)
        return "".join(parts) if parts else snapshot

    def _call_once(self, messages: list[dict[str, Any]]) -> str:
        payload = self._payload(messages)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_exc: Exception = RuntimeError("no attempts made")
        delay = self.retry_delay
        max_attempts = max(1, self.max_retries + 1)
        for attempt in range(max_attempts):
            try:
                stream = bool(payload.get("stream", True))
                host = (urlparse(self.request_url).hostname or "").lower()
                if host in LOCAL_HOSTS:
                    session = requests.Session()
                    session.trust_env = False
                    request = session.post
                else:
                    session = None
                    request = requests.post
                response_context = request(self.request_url, headers=headers, json=payload, timeout=300, stream=stream)
                try:
                    with response_context as response:
                        if response.status_code != 200:
                            raise RuntimeError(f"HTTP {response.status_code} url={self.request_url} body={response.text}")
                        if stream and "text/event-stream" in (response.headers.get("Content-Type") or "").lower():
                            return self._extract_stream(response)
                        return self._extract_text(response.json())
                finally:
                    if session is not None:
                        session.close()
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    time.sleep(delay)
                    delay *= 2
        raise last_exc


def _make_backend(
    *,
    model: str,
    backend: str,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_delay: float,
    enable_thinking: bool,
) -> tuple[OpenAICompatibleBackend, str]:
    resolved_base_url = base_url or ("http://127.0.0.1:8000/v1" if backend == "vllm" else _default_base_url(model))
    host = (urlparse(resolved_base_url).hostname or "").lower()
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    if backend == "vllm" or host in LOCAL_HOSTS:
        resolved_api_key = resolved_api_key or "EMPTY"
    if not resolved_api_key:
        raise SystemExit("No API key. Set --api-key, OPENAI_API_KEY, or DASHSCOPE_API_KEY.")
    return (
        OpenAICompatibleBackend(
            model=model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            backend=backend,
            enable_thinking=enable_thinking,
        ),
        resolved_base_url,
    )


def _prompt_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    variants = row.get("prompt_variants")
    if isinstance(variants, list) and variants:
        return [variant for variant in variants if isinstance(variant, dict)]
    requirement = row.get("user_requirement")
    if isinstance(requirement, str) and requirement.strip():
        return [{"style_id": "direct", "style_label": "Direct", "user_requirement": requirement}]
    return []


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = "||".join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def _all_indices(row: dict[str, Any], value: str) -> list[int]:
    variants = _prompt_variants(row)
    if value.strip().lower() == "all":
        return list(range(len(variants)))
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            index = int(token)
            if index < len(variants):
                out.append(index)
    return sorted(set(out)) or [0]


def _indices_for_styles(row: dict[str, Any], style_ids: str) -> list[int]:
    requested = [token.strip() for token in style_ids.split(",") if token.strip()]
    if not requested:
        return []
    variants = _prompt_variants(row)
    by_style: dict[str, list[int]] = {}
    for index, variant in enumerate(variants):
        style_id = str(variant.get("style_id") or "")
        if style_id:
            by_style.setdefault(style_id, []).append(index)
    selected: list[int] = []
    missing: list[str] = []
    for style_id in requested:
        matches = by_style.get(style_id) or []
        if matches:
            selected.append(matches[0])
        else:
            missing.append(style_id)
    if missing:
        available = ",".join(sorted(by_style)) or "none"
        raise ValueError(
            f"missing prompt styles for instance_id={row.get('instance_id')}: "
            f"requested={','.join(missing)} available={available}"
        )
    return selected


def _indices(
    row: dict[str, Any],
    value: str,
    *,
    style_ids: str,
    sample_size: int,
    sample_seed: int,
) -> list[int]:
    selected = _indices_for_styles(row, style_ids) if style_ids.strip() else _all_indices(row, value)
    if sample_size <= 0 or sample_size >= len(selected):
        return selected
    recipe_prompt_key = str(row.get("recipe_prompt_key") or row.get("workflow_prompt_key") or row.get("recipe_id") or "")
    instance_id = str(row.get("instance_id") or "")
    ranked = sorted(
        selected,
        key=lambda index: _stable_id("prompt-variant-sample", sample_seed, recipe_prompt_key, instance_id, index),
    )
    return sorted(ranked[:sample_size])


def _render_prompt(row: dict[str, Any], variant: dict[str, Any]) -> str:
    requirement = variant.get("user_requirement") or ""
    input_text = row.get("input_text") or ""
    output_format = row.get("output_format")
    if output_format == "json":
        return (
            f"Task:\n{requirement}\n\n"
            f"Input text:\n<input>\n{input_text}\n</input>\n\n"
            "Return only the required JSON object or array. Do not use markdown fences."
        )
    if output_format == "json_and_tagged_text":
        return (
            f"Task:\n{requirement}\n\n"
            f"Input text:\n<input>\n{input_text}\n</input>\n\n"
            "Return both the required JSON fields and tagged text exactly as requested. "
            "Do not add explanations."
        )
    return (
        f"Task:\n{requirement}\n\n"
        f"Raw input text:\n<input>\n{input_text}\n</input>\n\n"
        "Return tagged output only.\n"
        "Use exactly this format: "
        "<status>KEEP</status><clean_text>...</clean_text> or "
        "<status>DROP</status><clean_text>...</clean_text>\n"
        "Rules:\n"
        "- status must be KEEP or DROP inside <status>...</status>.\n"
        "- Put the output text inside <clean_text>...</clean_text>.\n"
        "- If status is KEEP, clean_text must be the final refined text.\n"
        "- If status is DROP, clean_text must be the text state at the point where the sample is rejected.\n"
        "- Preserve backslashes exactly as plain text; do not JSON-encode them.\n"
        "- Do not output markdown, code fences, or explanations.\n"
    )


def _tagged_output_hint() -> str:
    return (
        "<status>KEEP</status><clean_text>...</clean_text> "
        "or <status>DROP</status><clean_text>...</clean_text>"
    )


def _reference_tagged_output(row: dict[str, Any]) -> str:
    return (
        f"<status>{str(row.get('reference_status') or '').strip().upper()}</status>"
        f"<clean_text>{str(row.get('reference_text') or '')}</clean_text>"
    )


def _select_variant_for_style(row: dict[str, Any], preferred_style_id: str | None) -> dict[str, Any]:
    variants = _prompt_variants(row)
    if variants:
        if preferred_style_id:
            for variant in variants:
                if str(variant.get("style_id") or "") == preferred_style_id:
                    return variant
        return variants[0]
    return {"style_id": "", "style_label": "", "user_requirement": str(row.get("user_requirement") or "")}


FEW_SHOT_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "user_requirement": (
            "Remove any URLs and web links from the text, then normalize "
            "repeated whitespace into single spaces."
        ),
        "input_text": (
            "Visit https://example.com for details.   The site offers great resources."
        ),
        "reference_output": (
            "<status>KEEP</status><clean_text>Visit for details. "
            "The site offers great resources.</clean_text>"
        ),
    },
    {
        "user_requirement": (
            "Remove any URLs, then keep the text only if it contains at least 10 words."
        ),
        "input_text": "See https://example.com now.",
        "reference_output": "<status>DROP</status><clean_text>See now.</clean_text>",
    },
)


def _few_shot_examples() -> list[dict[str, str]]:
    return [dict(example) for example in FEW_SHOT_EXAMPLES]


def _format_few_shot_prompt(
    row: dict[str, Any],
    variant: dict[str, Any],
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
                    example["user_requirement"],
                    "",
                    "Raw input text:",
                    "<input>",
                    example["input_text"],
                    "</input>",
                    "",
                    "Correct output:",
                    example["reference_output"],
                ]
            )
        )
    sections.append(_render_prompt(row, variant))
    return "\n\n".join(section for section in sections if section.strip())


def _format_filter_rule(op_name: str, params: Any) -> str:
    if isinstance(params, dict) and params:
        detail = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
        return f"{op_name} with {detail}"
    return op_name


def _format_plan_first_prompt(row: dict[str, Any], variant: dict[str, Any]) -> str:
    requirement = variant.get("user_requirement") or ""
    input_text = row.get("input_text") or ""
    return (
        f"Task:\n{requirement}\n\n"
        "Before cleaning the text, first write a short analysis block in <analyze>...</analyze>.\n"
        "Use that block to restate the requested procedure as a concise standardized execution recipe.\n"
        "List the steps in order. For each step, say what to do and include the relevant rule or threshold when there is one.\n\n"
        "Example analysis format:\n"
        "<analyze>\n"
        "Step 1: Remove repeated sentences.\n"
        "Step 2: Normalize whitespace.\n"
        "Step 3: Apply the word repetition filter with ratio <= 0.2, and drop the text if it fails.\n"
        "</analyze>\n\n"
        f"Raw input text:\n<input>\n{input_text}\n</input>\n\n"
        "Then execute the procedure on the text.\n"
        f"After the analysis block, output the final tagged result using exactly this format: {_tagged_output_hint()}\n"
        "Rules:\n"
        "- Output exactly one <analyze>...</analyze> block before the final tagged result.\n"
        "- Inside <analyze>, write an ordered step list for the intended execution recipe.\n"
        "- For each step, include the relevant rule or threshold when there is one.\n"
        "- status must be KEEP or DROP inside <status>...</status>.\n"
        "- Put the output text inside <clean_text>...</clean_text>.\n"
        "- If status is DROP, stop at the rejection point and return the current text.\n"
        "- Do not output markdown, code fences, or explanations outside <analyze>.\n"
    )


def _format_state_aware_prompt(row: dict[str, Any], variant: dict[str, Any]) -> str:
    requirement = variant.get("user_requirement") or ""
    input_text = row.get("input_text") or ""
    return (
        f"Task:\n{requirement}\n\n"
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
        f"Raw input text:\n<input>\n{input_text}\n</input>\n\n"
        "After the analysis block, return the final tagged output only.\n"
        f"Use exactly this format: {_tagged_output_hint()}\n"
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


def _render_prompt_for_mode(
    row: dict[str, Any],
    variant: dict[str, Any],
    *,
    prompt_mode: str,
    few_shot_examples: list[dict[str, str]] | None = None,
) -> str:
    if row.get("output_format") in {"json", "json_and_tagged_text"}:
        return _render_prompt(row, variant)
    if prompt_mode == "few_shot":
        return _format_few_shot_prompt(row, variant, examples=few_shot_examples or [])
    if prompt_mode == "plan_first":
        return _format_plan_first_prompt(row, variant)
    if prompt_mode == "state_aware":
        return _format_state_aware_prompt(row, variant)
    return _render_prompt(row, variant)


def _parse_tagged(text: str) -> tuple[str, str]:
    status = "KEEP"
    status_match = re.search(r"<status>\s*(KEEP|DROP)\s*</status>", text, re.IGNORECASE)
    if status_match:
        status = status_match.group(1).upper()
    text_match = re.search(r"<clean_text>(.*?)</clean_text>", text, re.DOTALL | re.IGNORECASE)
    if text_match:
        return status, text_match.group(1)
    return status, text.strip()


def _infer_one(
    backend: Any,
    row: dict[str, Any],
    index: int,
    *,
    prompt_mode: str,
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    variants = _prompt_variants(row)
    variant = variants[index] if index < len(variants) else variants[0]
    examples = None
    if prompt_mode == "few_shot":
        examples = _few_shot_examples()
    prompt = _render_prompt_for_mode(row, variant, prompt_mode=prompt_mode, few_shot_examples=examples)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        raw = backend._call_once(messages)
        if row.get("output_format") == "json":
            predicted_status = "KEEP"
            predicted_clean_text = raw.strip()
        else:
            predicted_status, predicted_clean_text = _parse_tagged(raw)
        return {
            "prompt_variant_index": index,
            "prompt_style_id": variant.get("style_id"),
            "prompt_style_label": variant.get("style_label"),
            "user_requirement": variant.get("user_requirement"),
            "predicted_status": predicted_status,
            "predicted_clean_text": predicted_clean_text,
            "raw_response": raw,
            "prediction_error": None,
            "valid_prediction": True,
        }
    except Exception as exc:
        return {
            "prompt_variant_index": index,
            "prompt_style_id": variant.get("style_id"),
            "prompt_style_label": variant.get("style_label"),
            "user_requirement": variant.get("user_requirement"),
            "predicted_status": "KEEP",
            "predicted_clean_text": "",
            "raw_response": "",
            "prediction_error": str(exc),
            "valid_prediction": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified CDR-Bench v3 inference.")
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["api", "vllm"], default="api")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt-variant-indices", default="all")
    parser.add_argument("--prompt-style-ids", default="")
    parser.add_argument("--prompt-variant-sample-size", type=int, default=0)
    parser.add_argument("--prompt-variant-sampling-seed", type=int, default=0)
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="direct")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-input-chars", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=1, help="Number of retries after the first request attempt.")
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", action="store_false", dest="enable_thinking")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-only-existing-rows", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_path).resolve()
    output_path = Path(args.output_path).resolve()
    rows = read_jsonl(benchmark_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if args.max_input_chars > 0:
        rows = [row for row in rows if len(str(row.get("input_text") or "")) <= args.max_input_chars]
    candidate_rows = list(rows)

    existing: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        existing = read_jsonl(output_path)
        done_ids = {str(row.get("instance_id")) for row in existing}
    rows = [row for row in rows if str(row.get("instance_id")) not in done_ids]
    if args.resume_only_existing_rows:
        allowed_ids = {str(row.get("instance_id")) for row in existing}
        rows = [row for row in rows if str(row.get("instance_id")) in allowed_ids]

    backend, resolved_base_url = _make_backend(
        model=args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        enable_thinking=args.enable_thinking,
    )

    def process(row: dict[str, Any]) -> dict[str, Any]:
        indices = _indices(
            row,
            args.prompt_variant_indices,
            style_ids=args.prompt_style_ids,
            sample_size=args.prompt_variant_sample_size,
            sample_seed=args.prompt_variant_sampling_seed,
        )
        out = dict(row)
        out["request_model"] = args.model
        out["request_backend"] = args.backend
        out["request_base_url"] = resolved_base_url
        out["request_temperature"] = args.temperature
        out["request_enable_thinking"] = bool(args.enable_thinking)
        out["prompt_mode"] = args.prompt_mode
        out["selected_prompt_variant_indices"] = indices
        out["selected_prompt_style_ids"] = args.prompt_style_ids
        out["prompt_variant_sample_size"] = args.prompt_variant_sample_size
        out["prompt_variant_sampling_seed"] = args.prompt_variant_sampling_seed
        out["variant_predictions"] = [
            _infer_one(
                backend,
                row,
                index,
                prompt_mode=args.prompt_mode,
                candidate_rows=candidate_rows,
            )
            for index in indices
        ]
        return out

    output_rows = list(existing)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process, row): row for row in rows}
        for idx, future in enumerate(as_completed(futures), start=1):
            output_rows.append(future.result())
            if idx % args.progress_every == 0 or idx == len(rows):
                write_jsonl(output_path, output_rows)
                print(f"progress {idx}/{len(rows)}")
    write_jsonl(output_path, output_rows)
    print(f"wrote {len(output_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
