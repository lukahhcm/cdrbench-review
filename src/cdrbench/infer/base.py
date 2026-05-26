#!/usr/bin/env python3
"""
Inference base classes and shared result containers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

import logging

from tqdm import tqdm

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


@dataclass
class InferResult:
    contents: List[str]
    error: Optional[str] = None
    raw: Any = field(default=None, repr=False)

    @property
    def text(self) -> str:
        return self.contents[0] if self.contents else ''

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseInfer(ABC):
    def __init__(
        self,
        model: str,
        concurrency: int = 8,
        max_tokens: int = 0,
        temperature: float | None = None,
        top_p: float | None = None,
        num_runs: int = 1,
    ) -> None:
        self.model = model
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_runs = num_runs

    @abstractmethod
    def _call_once(self, messages: List[dict]) -> str:
        """Run one model call and return the response text."""

    def infer_one(self, messages: List[dict]) -> InferResult:
        if not messages:
            return InferResult(contents=[], error='empty messages')
        try:
            contents = [self._call_once(messages) for _ in range(self.num_runs)]
            return InferResult(contents=contents)
        except Exception as exc:
            return InferResult(contents=[], error=str(exc))

    def infer(
        self,
        messages_list: List[List[dict]],
        *,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> List[InferResult]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(messages_list)
        results: List[Optional[InferResult]] = [None] * total

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            future_to_idx = {
                pool.submit(self.infer_one, messages_list[idx]): idx
                for idx in range(total)
            }
            completed = as_completed(future_to_idx)
            if show_progress:
                completed = tqdm(completed, total=total, desc=progress_desc or f'infer [{self.model}]')
            for future in completed:
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = InferResult(contents=[], error=str(exc))

        return results  # type: ignore[return-value]
