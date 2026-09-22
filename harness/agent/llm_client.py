"""Minimal OpenAI-compatible chat client for multimodal LLMs.

Design goals: no SDK dependency, explicit timeouts, retries with backoff and a
per-call log (prompt text without image bytes, response, latency, usage).
Model reasoning is switched on for every model family (``THINKING_EXTRA``).
"""

from __future__ import annotations

import base64
import json
import re
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ..core import watchdog
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BASE_URL_ENV = "LLM_API_BASE"
API_KEY_ENV = "LLM_API_KEY"
KEY_DIR = Path.home() / ".config" / "robodawn"
RATE_LIMIT_PATIENCE_S = 900.0   # keep retrying 429 / 5xx for this long before giving the episode a failed call
USER_AGENT = "curl/8.4.0"      # some gateways sit behind rules that block urllib's default agent


def load_api_key() -> str:
    """One API key: ``$LLM_API_KEY``, else the files ``~/.config/robodawn/key*`` (one key per file) or
    ``~/.config/robodawn/keys`` (one key per line). Several keys are spread over processes (chosen by
    pid), which raises the total throughput when the endpoint limits per key."""
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key.strip()
    keys: list[str] = []
    if (KEY_DIR / "keys").is_file():
        keys += [l.strip() for l in (KEY_DIR / "keys").read_text().splitlines() if l.strip() and not l.startswith("#")]
    for p in sorted(KEY_DIR.glob("key*")):
        if p.name != "keys" and p.is_file():
            keys.append(p.read_text().strip())
    keys = [k for i, k in enumerate(keys) if k and k not in keys[:i]]
    if not keys:
        raise RuntimeError(f"no API key: pass --api_key_file, set {API_KEY_ENV} or write {KEY_DIR}/key")
    return keys[os.getpid() % len(keys)]


# Request-body fields that switch reasoning ON, per model family: exact model names first, then family
# substrings. An empty dict means the model thinks by default on an OpenAI-compatible endpoint, so
# nothing is sent (sending a field can route the request differently). These are the fields the
# reported runs sent; check ``usage.completion_tokens_details.reasoning_tokens`` in ``llm_calls.jsonl``
# to confirm that your endpoint actually reasons.
THINKING_EXTRA = {
    "gemini-3.8-flash": {},
    "qwen3.8-max": {},
    "qwen3.8-27b": {"reasoning_effort": "medium"},       # a vLLM deployment: accepts low/medium/xhigh only
    "gpt-5.6-sol": {"reasoning_effort": "medium"},
    "gpt-5.6-luna": {"reasoning_effort": "medium"},
    "gpt-6-astra": {"reasoning_effort": "high"},
    "doubao": {"thinking": {"type": "enabled"}},
    "deepseek": {"thinking": {"type": "enabled"}},
    "qwen": {"enable_thinking": True},
    "gemini": {},
    "gpt": {"reasoning_effort": "medium"},
}


def _fields_for(table: dict, model: str) -> dict:
    m = model.lower()
    if m in table:
        return dict(table[m])
    for key, extra in table.items():
        if key in m:
            return dict(extra)
    return {}


def thinking_fields(model: str) -> dict:
    return _fields_for(THINKING_EXTRA, model)


@dataclass
class ChatResult:
    text: str
    latency_s: float
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    attempts: int = 1
    error: Optional[str] = None


def image_part(png_bytes: bytes, detail: Optional[str] = None) -> dict:
    url = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    part = {"type": "image_url", "image_url": {"url": url}}
    if detail:
        part["image_url"]["detail"] = detail
    return part


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def inline_system_message(messages: list[dict]) -> list[dict]:
    """The same conversation without a system role: its text leads the first user message."""
    if not messages or messages[0]["role"] != "system":
        return messages
    system, rest = messages[0]["content"], list(messages[1:])
    for i, m in enumerate(rest):
        if m["role"] == "user":
            content = m["content"] if isinstance(m["content"], list) else [text_part(m["content"])]
            rest[i] = {"role": "user", "content": [text_part(system if isinstance(system, str) else "")] + content}
            return rest
    return [{"role": "user", "content": system}] + rest


class ChatClient:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        temperature: Optional[float] = 0.0,
        max_tokens: int = 1200,
        extra_body: Optional[dict] = None,
        log_path: Optional[Path] = None,
        rpm: Optional[float] = None,
        inline_system: bool = False,
    ) -> None:
        self.model = model
        base_url = base_url or os.environ.get(BASE_URL_ENV)
        if not base_url:
            raise RuntimeError(f"no API base URL: pass --api_base or set {BASE_URL_ENV}")
        self.base_url = base_url.rstrip("/")
        self.min_interval_s = 60.0 / rpm if rpm else 0.0    # client-side rate limit (requests per minute)
        self._last_request = 0.0
        # some gateways switch Gemini's thinking off as soon as a request carries a system message; the
        # system prompt then travels as the first text of the first user message instead
        self.inline_system = inline_system
        self.api_key = api_key or load_api_key()
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        for k, v in thinking_fields(model).items():
            self.extra_body.setdefault(k, v)
        self.log_path = Path(log_path) if log_path else None
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "seconds": 0.0}

    # ------------------------------------------------------------------
    def _body(self, messages: list[dict]) -> dict:
        if self.inline_system:
            messages = inline_system_message(messages)
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra_body)
        return body

    def _post(self, body: dict, tag: str = "") -> tuple[int, Any]:
        """One attempt: (HTTP status, parsed JSON on 200 / error text otherwise); 0 = no HTTP answer at all."""
        url = self.base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "User-Agent": USER_AGENT}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return 200, json.load(resp)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(errors="replace")[:500]
        except Exception as exc:  # noqa: BLE001  (timeouts, connection resets, bad JSON)
            return 0, f"{type(exc).__name__}: {exc}"

    def chat(self, messages: list[dict], tag: str = "") -> ChatResult:
        body = self._body(messages)
        last_err = None
        t_start = time.time()
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            if self.min_interval_s:
                wait = self._last_request + self.min_interval_s - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last_request = time.time()
            t0 = time.time()
            watchdog.beat(f"llm call {tag} attempt {attempt}")
            status, payload = self._post(body, tag)
            if status == 200:
                try:
                    choice = payload["choices"][0]
                    text = choice["message"].get("content") or ""
                    if isinstance(text, list):  # some gateways return content parts
                        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                    usage = payload.get("usage") or {}
                except Exception as exc:  # noqa: BLE001  (a 200 without a usable reply)
                    last_err = f"{type(exc).__name__}: {exc}"
                else:
                    result = ChatResult(text=text, latency_s=time.time() - t0, usage=usage, raw=payload, attempts=attempt)
                    self._account(result)
                    self._log(messages, result, tag)
                    return result
            elif status:
                body_txt = str(payload)
                last_err = f"HTTP {status}: {body_txt}"
                if status == 400 and self._relax_reasoning(body_txt, body):
                    # the gateway routed us to a backend with a different reasoning-level vocabulary
                    logger.warning("400 on reasoning level; retrying with %s", {k: body.get(k) for k in ("reasoning_effort", "enable_thinking", "thinking")})
                    continue
                if status == 400 and "flagged" in body_txt:
                    # sporadic gateway-side moderation false positives; retry the same request
                    pass
                elif status in (401, 403) and any(w in body_txt.lower() for w in ("quota", "exhausted", "rate", "too many")):
                    # a per-window quota, not a bad key: some gateways answer 401 "Token quota exhausted" when many
                    # processes share one key. Treated like 429 below, so an episode is not lost to a busy minute.
                    last_err = "HTTP 429: " + body_txt
                elif status in (400, 401, 403, 404, 413, 422):
                    break
            else:
                last_err = str(payload)          # timeouts, connection resets
            sleep = min(30.0, 2.0 ** attempt)
            if last_err.startswith("HTTP 429") or last_err.startswith("HTTP 5"):
                # quota / availability: wait what the gateway asks for (or a full slot) and allow more attempts;
                # a per-minute token quota clears by itself, so giving up here would lose the whole episode
                hint = re.search(r"retry in ([0-9.]+)\s*s", last_err, re.I)
                sleep = min(120.0, max(sleep, float(hint.group(1)) + 2.0 if hint else (self.min_interval_s or 20.0)))
                if attempt >= self.max_retries and time.time() - t_start < RATE_LIMIT_PATIENCE_S:
                    attempt -= 1          # keep retrying until the patience budget is used up
            logger.warning("chat attempt %d failed (%s); retrying in %.0fs", attempt, last_err[:160], sleep)
            time.sleep(sleep)
        result = ChatResult(text="", latency_s=time.time() - t_start, attempts=self.max_retries, error=last_err)
        self._log(messages, result, tag)
        return result

    _REASONING_FALLBACKS = ["low", None]

    def _relax_reasoning(self, error_text: str, body: dict) -> bool:
        """Adjust reasoning fields after a 400 that complains about them. Returns True if a retry makes sense."""
        text = error_text.lower()
        if not any(k in text for k in ("reasoning", "thinking")):
            return False
        if "reasoning_effort" in body:
            current = body["reasoning_effort"]
            for cand in self._REASONING_FALLBACKS:
                if cand != current and (cand is None or cand not in text.split("valid levels")[0]):
                    if cand is None:
                        body.pop("reasoning_effort", None)
                    else:
                        body["reasoning_effort"] = cand
                    return True
            return False
        for k in ("enable_thinking", "thinking", "thinking_budget"):
            if k in body:
                body.pop(k, None)
                return True
        return False

    # ------------------------------------------------------------------
    def _account(self, r: ChatResult) -> None:
        u = r.usage
        self.total_usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        self.total_usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
        self.total_usage["calls"] += 1
        self.total_usage["seconds"] += r.latency_s

    def _log(self, messages: list[dict], r: ChatResult, tag: str) -> None:
        if not self.log_path:
            return
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": tag,
            "model": self.model,
            "latency_s": round(r.latency_s, 2),
            "attempts": r.attempts,
            "error": r.error,
            "usage": r.usage,
            "messages": _strip_images(messages),
            "response": r.text,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _strip_images(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if p.get("type") == "image_url":
                    parts.append({"type": "image_url", "image_url": "<png>"})
                else:
                    parts.append(p)
            out.append({"role": m["role"], "content": parts})
        else:
            out.append(m)
    return out
