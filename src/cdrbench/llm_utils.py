from __future__ import annotations

import json
import os
import re
from typing import Any

from cdrbench.infer.api_model_config import (
    OPENAI_COMPAT_BASE_URL,
    default_base_url_for_model,
    resolve_api_model_name,
)

DEFAULT_BASE_URL = OPENAI_COMPAT_BASE_URL
DEFAULT_MODEL = "gpt-4o-mini"


def resolve_api_key(explicit: str | None = None) -> str:
    api_key = explicit or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("No API key found. Set OPENAI_API_KEY, LLM_API_KEY, or API_KEY.")
    return api_key


def resolve_base_url(explicit: str | None = None, model: str | None = None) -> str:
    return (
        explicit
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or default_base_url_for_model(model or DEFAULT_MODEL)
        or DEFAULT_BASE_URL
    )


def resolve_model(explicit: str | None = None) -> str:
    candidate = explicit or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    return resolve_api_model_name(candidate, default=DEFAULT_MODEL)


def build_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for LLM-backed prompt generation/judging. "
            "Install project dependencies with `python -m pip install -e .`."
        ) from exc
    return OpenAI(
        api_key=resolve_api_key(api_key),
        base_url=resolve_base_url(base_url, model=model),
    )


def chat_completion(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
) -> str:
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty content.")
    return content


def strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL)
    return fenced.group(1) if fenced else text


def parse_json_response(text: str) -> Any:
    cleaned = strip_code_fences(text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise
