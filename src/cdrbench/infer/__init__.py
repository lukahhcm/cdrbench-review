from __future__ import annotations

from typing import Any

from .base import BaseInfer, InferResult

__all__ = [
    'BaseInfer',
    'InferResult',
    'OpenAIInfer',
    'make_api_infer',
    'make_vllm_infer',
]


def __getattr__(name: str) -> Any:
    if name in {'OpenAIInfer', 'make_api_infer', 'make_vllm_infer'}:
        from .openai_infer import OpenAIInfer, make_api_infer, make_vllm_infer

        exports = {
            'OpenAIInfer': OpenAIInfer,
            'make_api_infer': make_api_infer,
            'make_vllm_infer': make_vllm_infer,
        }
        return exports[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
