"""Minimal Ollama chat client used by the determinator and the benchmark.

Only what Jobby needs: one blocking chat call with an explicit context window,
thinking disabled, and the metrics Ollama reports (token counts and timing).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Raised when Ollama cannot be reached or returns an unusable response."""


@dataclass(frozen=True)
class ChatResult:
    text: str
    wall_s: float
    metrics: dict[str, Any]

    @property
    def eval_tokens(self) -> int:
        return int(self.metrics.get("eval_count") or 0)

    @property
    def prompt_tokens(self) -> int:
        return int(self.metrics.get("prompt_eval_count") or 0)

    @property
    def tokens_per_second(self) -> float:
        duration_ns = int(self.metrics.get("eval_duration") or 0)
        if duration_ns <= 0:
            return 0.0
        return self.eval_tokens / (duration_ns / 1e9)


def chat(
    base_url: str,
    model: str,
    *,
    system: str,
    prompt: str,
    num_ctx: int,
    num_predict: int,
    temperature: float = 0,
    timeout_s: float = 900,
    think: bool = False,
    json_mode: bool = True,
) -> ChatResult:
    """Send one chat request and return the text plus Ollama's metrics."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": think,
        "keep_alive": "10m",
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": 1.0,
            "seed": 0,
        },
    }
    if json_mode:
        payload["format"] = "json"

    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc
    wall_s = time.perf_counter() - started

    if response.status_code != 200:
        raise OllamaError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:400]}"
        )
    body = response.json()
    message = body.get("message") or {}
    text = (message.get("content") or "").strip()
    if not text and message.get("thinking"):
        raise OllamaError(
            "Ollama replied with thinking tokens only; the model ignored think=false"
        )
    return ChatResult(text=text, wall_s=wall_s, metrics=body)


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the first JSON object in ``text``; return (object, error)."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return None, "no JSON object in reply"
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at {exc.pos}"
    if not isinstance(parsed, dict):
        return None, "JSON is not an object"
    return parsed, None
