"""Model clients, on the standard library only.

    make_llm("anthropic:claude-haiku-5-5")     ANTHROPIC_API_KEY (ANTHROPIC_BASE_URL optional)
    make_llm("openai:some-model")              OPENAI_API_KEY, OPENAI_BASE_URL for any server
                                               that speaks /chat/completions

A client is any object with `.name` and `.complete(system, user) -> str`.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import time
import urllib.error
import urllib.request

ANTHROPIC_VERSION = "2026-10-21"
RETRY_STATUS = {408, 409, 429}
RETRY_STREAM_ERRORS = {"overloaded_error", "api_error", "rate_limit_error", "timeout_error"}


class LLMError(RuntimeError):
    """The model could not be reached, or the request was rejected. A run stops on this."""


class LLMRefusal(RuntimeError):
    """The model answered, but declined or produced no text. The caller treats it as a
    failed attempt, not as a broken connection."""


class _Retry(Exception):
    pass


def _error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:500]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return f"{error.get('type') or error.get('code') or 'error'}: {error.get('message')}"
    return body.strip()[:500]


def _retry_after(headers) -> float | None:
    try:
        return min(60.0, max(0.0, float(headers.get("retry-after"))))
    except (TypeError, ValueError):
        return None


def _post(url: str, headers: dict, payload: dict, *, timeout: float, retries: int, backoff: float, read):
    """POST JSON and hand the open response to `read`. Retries 408/409/429/5xx, dropped
    connections and timeouts with exponential backoff."""
    data = json.dumps(payload).encode("utf-8")
    last = "no attempt made"
    for attempt in range(retries + 1):
        delay = None
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return read(response)
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}: {_error_message(exc.read().decode('utf-8', 'replace'))}"
            retryable = exc.code in RETRY_STATUS or exc.code >= 500
            hint = (exc.headers.get("x-should-retry") or "").lower()
            if hint in ("true", "false"):
                retryable = hint == "true"
            if not retryable:
                raise LLMError(f"{url}: {last}") from None
            delay = _retry_after(exc.headers)
        except _Retry as exc:
            last = str(exc)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(delay if delay is not None else min(60.0, backoff * 2**attempt) * random.uniform(0.5, 1.0))
    raise LLMError(f"{url}: gave up after {retries + 1} attempt(s): {last}")


class AnthropicLLM:
    """The Claude Messages API, streamed so long replies do not hit idle timeouts.

    Thinking and sampling are left at the model's defaults (current models reject sampling
    parameters). No fallback model is requested: the point of the experiment is that the
    policy model is fixed, so a refusal is reported instead of being answered by another model.
    """

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int | None = None,
        effort: str | None = None,
        timeout: float = 600.0,
        retries: int = 5,
        backoff: float = 2.0,
    ) -> None:
        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens or 32000
        self.effort = effort
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self.url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        self.headers = {"content-type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.headers["x-api-key"] = os.environ["ANTHROPIC_API_KEY"]
        elif os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            self.headers["authorization"] = "Bearer " + os.environ["ANTHROPIC_AUTH_TOKEN"]
        else:
            raise LLMError("ANTHROPIC_API_KEY is not set")

    def complete(self, system: str, user: str) -> str:
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": True,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            payload["system"] = system
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        text, stop = _post(
            self.url, self.headers, payload,
            timeout=self.timeout, retries=self.retries, backoff=self.backoff, read=_read_anthropic,
        )
        if stop == "refusal":
            raise LLMRefusal(f"{self.name} declined the request")
        if not text.strip():
            raise LLMRefusal(f"{self.name} returned no text (stop_reason={stop})")
        return text


def _read_anthropic(response) -> tuple[str, str | None]:
    if "text/event-stream" not in (response.headers.get("content-type") or ""):
        data = json.loads(response.read().decode("utf-8"))  # a proxy that ignored "stream"
        blocks = data.get("content") or []
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text"), data.get("stop_reason")
    parts: list[str] = []
    stop = None
    finished = False
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        event = json.loads(line[5:])
        kind = event.get("type")
        if kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                parts.append(delta.get("text", ""))
        elif kind == "message_delta":
            stop = (event.get("delta") or {}).get("stop_reason") or stop
        elif kind == "message_stop":
            finished = True
            break
        elif kind == "error":
            error = event.get("error") or {}
            message = f"{error.get('type')}: {error.get('message')}"
            if error.get("type") in RETRY_STREAM_ERRORS:
                raise _Retry(message)
            raise LLMError(message)
    if not finished and stop is None:
        raise _Retry("the response stream ended early")
    return "".join(parts), stop


class OpenAILLM:
    """Any server with an OpenAI-style /chat/completions endpoint: set OPENAI_BASE_URL.

    `max_tokens` is sent only when given; by default the server's own limit applies.
    """

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int | None = None,
        timeout: float = 600.0,
        retries: int = 5,
        backoff: float = 2.0,
        **_ignored,
    ) -> None:
        self.model = model
        self.name = f"openai:{model}"
        self.max_tokens = max_tokens
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        if not base and not key:
            raise LLMError("OPENAI_API_KEY is not set (for another server, set OPENAI_BASE_URL)")
        self.url = (base or "https://api.openai.com/v1") + "/chat/completions"
        self.headers = {"content-type": "application/json"}
        if key:
            self.headers["authorization"] = "Bearer " + key

    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        payload: dict = {"model": self.model, "messages": messages}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        data = _post(
            self.url, self.headers, payload,
            timeout=self.timeout, retries=self.retries, backoff=self.backoff,
            read=lambda response: json.loads(response.read().decode("utf-8")),
        )
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError(f"{self.url}: unexpected response: {str(data)[:300]}") from None
        if isinstance(content, list):  # some servers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if choice.get("finish_reason") == "content_filter":
            raise LLMRefusal(f"{self.name} declined the request")
        if not content or not content.strip():
            raise LLMRefusal(f"{self.name} returned no text (finish_reason={choice.get('finish_reason')})")
        return content


PROVIDERS = {"anthropic": AnthropicLLM, "openai": OpenAILLM}


def make_llm(spec, **options):
    """Build a client from "provider:model". An object that already has `.complete` is returned as is."""
    if hasattr(spec, "complete"):
        return spec
    provider, _, model = str(spec).partition(":")
    if provider not in PROVIDERS or not model:
        raise ValueError(
            f"model spec {spec!r} should look like 'provider:model', with provider one of: "
            + ", ".join(sorted(PROVIDERS))
        )
    return PROVIDERS[provider](model, **options)
