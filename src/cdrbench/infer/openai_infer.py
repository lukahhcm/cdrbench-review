#!/usr/bin/env python3
"""
OpenAI-compatible inference backend shared by local vLLM and remote APIs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from openai import OpenAI

from .base import BaseInfer
from .api_model_config import (
    DEFAULT_COMPAT_MAX_TOKENS,
    OPENAI_COMPAT_BASE_URL,
    default_base_url_for_model,
    get_api_model_config,
)

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


class OpenAIInfer(BaseInfer):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = 'EMPTY',
        concurrency: int = 8,
        max_tokens: int = 0,
        temperature: float | None = None,
        top_p: float | None = None,
        num_runs: int = 1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_thinking: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model=model,
            concurrency=concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_runs=num_runs,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_thinking = enable_thinking
        self._extra_body: Dict[str, Any] = extra_body or {}
        self._client = OpenAI(api_key=api_key, base_url=api_base)

    def _call_once(self, messages: List[dict]) -> str:
        last_exc: Exception = RuntimeError('no attempts made')
        delay = self.retry_delay

        for attempt in range(max(1, self.max_retries)):
            try:
                collected_parts: list[str] = []
                request_kwargs: Dict[str, Any] = {
                    'model': self.model,
                    'messages': messages,
                    'stream': True,
                }
                if self.temperature is not None:
                    request_kwargs['temperature'] = self.temperature
                if self.top_p is not None and self.top_p > 0:
                    request_kwargs['top_p'] = self.top_p
                if self.max_tokens > 0:
                    request_kwargs['max_tokens'] = self.max_tokens
                if self._extra_body:
                    request_kwargs['extra_body'] = self._extra_body
                stream = self._client.chat.completions.create(**request_kwargs)
                for chunk in stream:
                    choices = getattr(chunk, 'choices', None) or []
                    if not choices:
                        continue
                    delta = getattr(choices[0], 'delta', None)
                    content = getattr(delta, 'content', None) if delta is not None else None
                    if content:
                        collected_parts.append(str(content))
                return ''.join(collected_parts)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2

        raise last_exc

    def __repr__(self) -> str:
        return (
            f'OpenAIInfer(model={self.model!r}, '
            f'concurrency={self.concurrency}, '
            f'num_runs={self.num_runs}, '
            f'enable_thinking={self.enable_thinking})'
        )


class CompatApiInfer(BaseInfer):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        concurrency: int = 8,
        max_tokens: int = 0,
        temperature: float | None = None,
        top_p: float | None = None,
        num_runs: int = 1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 300.0,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved_model = model.strip()
        self._model_config = get_api_model_config(resolved_model)
        if self._model_config is not None:
            resolved_model = self._model_config.model_name
        super().__init__(
            model=resolved_model,
            concurrency=concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_runs=num_runs,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._api_key = api_key
        self._extra_body: Dict[str, Any] = extra_body or {}
        self._request_url = self._normalize_request_url(api_base)

    @staticmethod
    def _normalize_request_url(api_base: str) -> str:
        base = api_base.rstrip('/')
        if base.endswith('/chat/completions'):
            return base
        return f'{base}/chat/completions'

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get('text')
                    if text is not None:
                        parts.append(str(text))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return ''.join(parts)
        if content is None:
            return ''
        return str(content)

    def _build_input_messages(self, messages: List[dict[str, Any]]) -> list[dict[str, str]]:
        payload_messages: list[dict[str, str]] = []
        for message in messages:
            payload_messages.append(
                {
                    'role': str(message.get('role') or 'user'),
                    'content': self._stringify_content(message.get('content')),
                }
            )
        return payload_messages

    def _split_system_messages(self, messages: List[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        system_parts: list[str] = []
        remaining_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get('role') or 'user').strip().lower()
            if role == 'system':
                text = self._stringify_content(message.get('content')).strip()
                if text:
                    system_parts.append(text)
                continue
            remaining_messages.append(message)
        return system_parts, remaining_messages

    def _build_contents_messages(self, messages: List[dict[str, Any]]) -> list[dict[str, Any]]:
        parts: list[str] = []
        for message in messages:
            role = str(message.get('role') or 'user').strip().lower()
            text = self._stringify_content(message.get('content')).strip()
            if not text:
                continue
            if role == 'system':
                parts.append(f'System instruction:\n{text}')
            elif role == 'user':
                parts.append(f'User request:\n{text}')
            else:
                parts.append(f'{role}:\n{text}')
        combined_text = '\n\n'.join(parts).strip()
        return [{'role': 'user', 'parts': [{'text': combined_text}]}]

    def _build_payload(self, messages: List[dict[str, Any]]) -> dict[str, Any]:
        input_field = self._model_config.input_field if self._model_config is not None else 'messages'
        stream = True if self._model_config is None else self._model_config.stream
        payload: dict[str, Any] = {'model': self.model, 'stream': stream}
        effective_messages = messages

        if self._model_config is not None and self._model_config.top_level_system:
            system_parts, effective_messages = self._split_system_messages(messages)
            if system_parts:
                payload['system'] = '\n\n'.join(system_parts)

        if input_field == 'input':
            payload['input'] = self._build_input_messages(effective_messages)
        elif input_field == 'contents':
            payload['contents'] = self._build_contents_messages(effective_messages)
        else:
            payload['messages'] = self._build_input_messages(effective_messages)

        if self.max_tokens > 0:
            payload['max_tokens'] = self.max_tokens
        elif self._model_config is not None and self._model_config.default_max_tokens > 0:
            payload['max_tokens'] = self._model_config.default_max_tokens
        if self._extra_body:
            payload.update(self._extra_body)
        return payload

    @classmethod
    def _extract_text_from_message_object(cls, message: dict[str, Any]) -> str:
        content = message.get('content')
        if content is None:
            return ''
        return cls._stringify_content(content)

    @classmethod
    def _extract_text(cls, response_json: dict[str, Any]) -> str:
        choices = response_json.get('choices')
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get('message')
                if isinstance(message, dict):
                    content = message.get('content')
                    return cls._stringify_content(content)

        message = response_json.get('message')
        if isinstance(message, dict):
            message_text = cls._extract_text_from_message_object(message)
            if message_text:
                return message_text

        content = response_json.get('content')
        if isinstance(content, list):
            parts = [str(item.get('text', '')) for item in content if isinstance(item, dict) and item.get('text') is not None]
            if parts:
                return ''.join(parts)

        candidates = response_json.get('candidates')
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            if isinstance(candidate, dict):
                candidate_content = candidate.get('content')
                if isinstance(candidate_content, dict):
                    parts = candidate_content.get('parts')
                    if isinstance(parts, list):
                        return ''.join(str(part.get('text', '')) for part in parts if isinstance(part, dict) and part.get('text') is not None)

        output = response_json.get('output')
        if isinstance(output, dict):
            text = output.get('text')
            if text is not None:
                return str(text)
            output_choices = output.get('choices')
            if isinstance(output_choices, list) and output_choices:
                first_choice = output_choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get('message')
                    if isinstance(message, dict):
                        return cls._stringify_content(message.get('content'))

        return json.dumps(response_json, ensure_ascii=False)

    @classmethod
    def _extract_stream_chunk_text(cls, response_json: dict[str, Any]) -> str:
        choices = response_json.get('choices')
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get('delta')
                if isinstance(delta, dict):
                    content = delta.get('content')
                    if content is not None:
                        return cls._stringify_content(content)
                message = choice.get('message')
                if isinstance(message, dict):
                    content = message.get('content')
                    if content is not None:
                        return cls._stringify_content(content)

        message = response_json.get('message')
        if isinstance(message, dict):
            message_text = cls._extract_text_from_message_object(message)
            if message_text:
                return message_text

        return cls._extract_text(response_json)

    @classmethod
    def _extract_streaming_response_text(cls, response: requests.Response) -> str:
        parts: list[str] = []
        last_snapshot_text = ''
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if not data or data == '[DONE]':
                continue
            payload = json.loads(data)
            if isinstance(payload.get('message'), dict):
                snapshot_text = cls._extract_text(payload)
                if snapshot_text:
                    last_snapshot_text = snapshot_text
                continue
            chunk_text = cls._extract_stream_chunk_text(payload)
            if chunk_text:
                parts.append(chunk_text)
        if parts:
            return ''.join(parts)
        return last_snapshot_text

    def _call_once(self, messages: List[dict]) -> str:
        last_exc: Exception = RuntimeError('no attempts made')
        delay = self.retry_delay
        payload = self._build_payload(messages)
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }

        for attempt in range(max(1, self.max_retries)):
            try:
                request_stream = bool(payload.get('stream', True))
                with requests.post(
                    self._request_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=request_stream,
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(
                            f'HTTP {response.status_code} url={self._request_url} body={response.text}'
                        )
                    if not request_stream:
                        return self._extract_text(response.json())
                    content_type = (response.headers.get('Content-Type') or '').lower()
                    if 'text/event-stream' in content_type:
                        return self._extract_streaming_response_text(response)
                    return self._extract_text(response.json())
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2

        raise last_exc

    def __repr__(self) -> str:
        host = urlparse(self._request_url).hostname or 'unknown'
        return (
            f'CompatApiInfer(model={self.model!r}, '
            f'host={host!r}, '
            f'concurrency={self.concurrency}, '
            f'num_runs={self.num_runs})'
        )


def make_vllm_infer(
    model: str,
    api_base: str = 'http://127.0.0.1:8901/v1',
    concurrency: int = 128,
    max_tokens: int = 0,
    temperature: float | None = None,
    top_p: float | None = None,
    num_runs: int = 1,
    enable_thinking: bool = False,
    max_retries: int = 2,
    retry_delay: float = 0.5,
) -> OpenAIInfer:
    return OpenAIInfer(
        model=model,
        api_base=api_base,
        api_key='EMPTY',
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        num_runs=num_runs,
        max_retries=max_retries,
        retry_delay=retry_delay,
        enable_thinking=enable_thinking,
        extra_body={'chat_template_kwargs': {'enable_thinking': enable_thinking}},
    )


def make_api_infer(
    model: str,
    api_base: str = '',
    api_key: str = '',
    concurrency: int = 8,
    max_tokens: int = 0,
    temperature: float | None = None,
    top_p: float | None = None,
    num_runs: int = 1,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    extra_body: Optional[Dict[str, Any]] = None,
) -> CompatApiInfer:
    resolved_key = api_key or os.getenv('DASHSCOPE_API_KEY', 'EMPTY')
    resolved_base = api_base or default_base_url_for_model(model) or OPENAI_COMPAT_BASE_URL
    return CompatApiInfer(
        model=model,
        api_base=resolved_base,
        api_key=resolved_key,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        num_runs=num_runs,
        max_retries=max_retries,
        retry_delay=retry_delay,
        extra_body=extra_body or {},
    )
