"""Обёртка LLM: роли smart/simple/local, кэш, ретраи, таймауты, лог usage, стоимости и латентности.

Провайдеры (config.yaml → llm.provider):
- polza    — агрегатор Polza AI (OpenAI-совместимый API), модели GigaChat от Сбера; usage.cost_rub из ответа;
- gigachat — прямой API GigaChat через официальный SDK.

Каждый запрос пишется в logs/llm_calls.jsonl вместе с текстом сообщений — по этому логу
red-team проверяет, что маскированные ПДн не уходили в LLM.
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import diskcache
from dotenv import load_dotenv

from src.config import ROOT, cfg, path

load_dotenv(ROOT / ".env")

_cache: diskcache.Cache | None = None
_clients: dict[str, object] = {}
_log_lock = threading.Lock()
_last_usage = threading.local()
# Метка кейса для параллельных прогонов (red-team ищет по ней запросы конкретной атаки в логе)
trace: contextvars.ContextVar[str] = contextvars.ContextVar("trace", default="")


@dataclass
class LLMResult:
    text: str
    model: str
    role: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    error: str | None = None
    cost_rub: float | None = None  # фактическая стоимость от провайдера (Polza) или по цене из конфига
    meta: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _credentials() -> str:
    raw = os.environ.get("GIGACHAT_CREDENTIALS", "").strip()
    if not raw:
        raise RuntimeError("GIGACHAT_CREDENTIALS не задан в .env")
    # Сырая пара "ClientID:ClientSecret" → base64 (authorization key)
    if ":" in raw:
        raw = base64.b64encode(raw.encode()).decode()
    return raw


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(str(path("cache") / "llm"))
    return _cache


def provider() -> str:
    return cfg()["llm"].get("provider", "gigachat")


def model_for(role: str) -> str:
    roles = cfg()["llm"]["roles"][provider()]
    return roles[role if role in roles else "simple"]


def _client(model: str):
    if model not in _clients:
        from gigachat import GigaChat

        c = cfg()["llm"]
        kwargs = dict(
            credentials=_credentials(),
            scope=os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            model=model,
            timeout=c.get("timeout_s", 30),
        )
        if c.get("ca_bundle_file"):
            kwargs["ca_bundle_file"] = str(ROOT / c["ca_bundle_file"])
        else:
            kwargs["verify_ssl_certs"] = bool(c.get("verify_ssl_certs", False))
        _clients[model] = GigaChat(**kwargs)
    return _clients[model]


def _polza():
    if "polza" not in _clients:
        import httpx

        key = os.environ.get("POLZA_AI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("POLZA_AI_API_KEY не задан в .env")
        _clients["polza"] = httpx.Client(base_url=cfg()["llm"]["polza"]["base_url"],
                                         headers={"Authorization": f"Bearer {key}"},
                                         timeout=cfg()["llm"].get("timeout_s", 30))
    return _clients["polza"]


def list_models() -> list[str]:
    if provider() == "polza":
        r = _polza().get("/models")
        r.raise_for_status()
        d = r.json()
        return [m["id"] for m in (d.get("data", d) if isinstance(d, dict) else d)]
    return [m.id_ for m in _client(model_for("simple")).get_models().data]


def _call(model: str, messages: list[dict], temperature: float, max_tokens: int) -> tuple:
    """→ (text, model, prompt_tokens, completion_tokens, cost_rub | None)"""
    if provider() == "polza":
        extra = (cfg()["llm"]["polza"].get("extra_body") or {}).get(model, {})
        r = _polza().post("/chat/completions", json={"model": model, "messages": messages,
                                                      "temperature": temperature, "max_tokens": max_tokens, **extra})
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        d = r.json()
        u = d.get("usage") or {}
        _last_usage.value = u  # детали (reasoning_tokens) для отладки
        return (d["choices"][0]["message"].get("content") or "", d.get("model") or model,
                u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost_rub"))
    from gigachat.models import Chat, Messages, MessagesRole

    payload = Chat(messages=[Messages(role=MessagesRole(m["role"]), content=m["content"]) for m in messages],
                   temperature=temperature, max_tokens=max_tokens)
    resp = _client(model).chat(payload)
    return (resp.choices[0].message.content or "", resp.model or model, resp.usage.prompt_tokens,
            resp.usage.completion_tokens, None)


def _log(entry: dict) -> None:
    with _log_lock, open(path("llm_log"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def chat(
    messages: list[dict],
    role: str = "simple",
    temperature: float = 0.3,
    max_tokens: int = 512,
    use_cache: bool | None = None,
    tag: str = "",
) -> LLMResult:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}]. Текст должен быть уже маскирован."""
    from src.config import cost_rub
    # Страховка: user/assistant-сообщения прогоняются через маскирование ПДн ещё раз
    from src.pipeline.pii import mask_messages

    messages = [m if m["role"] == "system" else mask_messages([m])[0] for m in messages]

    if role == "local" and cfg()["llm"].get("local_backend") == "ollama":
        return _chat_ollama(messages, temperature, max_tokens, tag)

    model = model_for(role)
    use_cache = cfg()["llm"].get("cache_enabled", True) if use_cache is None else use_cache
    key = hashlib.sha256(
        json.dumps([model, messages, temperature, max_tokens], ensure_ascii=False).encode()
    ).hexdigest()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace": trace.get(),
        "tag": tag,
        "role": role,
        "model": model,
        "key": key,
        "messages": messages,
    }

    if use_cache:
        hit = _get_cache().get(key)
        if hit is not None:
            res = LLMResult(**{**hit, "cached": True, "latency_s": 0.0})
            _log({**entry, "cached": True, "prompt_tokens": res.prompt_tokens,
                  "completion_tokens": res.completion_tokens, "cost_rub": 0.0, "latency_s": 0.0})
            return res

    retries = cfg()["llm"].get("max_retries", 2)
    last_err = None
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            text, real_model, pt, ct, cost = _call(model, messages, temperature, max_tokens)
            res = LLMResult(text=text, model=real_model, role=role, prompt_tokens=pt, completion_tokens=ct,
                            latency_s=time.perf_counter() - t0,
                            cost_rub=cost if cost is not None else cost_rub(model, pt, ct))
            _get_cache().set(key, {k: v for k, v in asdict(res).items() if k not in ("cached",)})
            _log({**entry, "cached": False, "prompt_tokens": pt, "completion_tokens": ct,
                  "cost_rub": res.cost_rub, "latency_s": round(res.latency_s, 3), "attempt": attempt,
                  "response": text})
            return res
        except Exception as e:  # сеть, 429, 5xx
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if any(code in str(e) for code in ("401", "402", "403")):
                break  # ключ/баланс — ретраи не помогут
            time.sleep(1.5 * (attempt + 1))
    res = LLMResult(text="", model=model, role=role, latency_s=time.perf_counter() - t0, error=last_err)
    _log({**entry, "cached": False, "error": last_err, "latency_s": round(res.latency_s, 3)})
    return res


def _chat_ollama(messages, temperature, max_tokens, tag) -> LLMResult:
    import urllib.request

    o = cfg()["llm"]["ollama"]
    body = json.dumps({
        "model": o["model"], "messages": messages, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode()
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(f"{o['base_url']}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=cfg()["llm"]["timeout_s"]) as r:
            data = json.loads(r.read())
        res = LLMResult(text=data["message"]["content"], model=o["model"], role="local",
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                        latency_s=time.perf_counter() - t0)
    except Exception as e:
        res = LLMResult(text="", model=o["model"], role="local",
                        latency_s=time.perf_counter() - t0, error=str(e))
    _log({"ts": datetime.now(timezone.utc).isoformat(), "tag": tag, "role": "local",
          "model": o["model"], "messages": messages, "prompt_tokens": res.prompt_tokens,
          "completion_tokens": res.completion_tokens, "latency_s": round(res.latency_s, 3),
          "error": res.error})
    return res


def estimate_tokens(texts: list[str]) -> int:
    """Грубая оценка без вызова API: ~3,5 символа на токен для смеси RU/EN."""
    return int(sum(len(t) for t in texts) / 3.5)
