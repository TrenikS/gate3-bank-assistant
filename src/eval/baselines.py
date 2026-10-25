"""Бейзлайны для E2E.

A — «коробочный LLM-бот»: GigaChat с общим системным промптом банка, без данных клиента и шаблонов.
B — «сценарный бот»: ключевые слова + статичный FAQ без данных клиента.
Обе системы возвращают dict, совместимый по ключевым полям с Response пайплайна.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache

import yaml

from src.config import ROOT, cost_rub
from src.models import keywords
from src.pipeline import llm, pii

SYSTEM_A = (
    "Ты — вежливый виртуальный ассистент службы поддержки розничного банка. Отвечай клиенту коротко и "
    "по делу на русском языке. Помогай с вопросами по картам, переводам, платежам и приложению банка. "
    "Если вопрос требует участия сотрудника, предложи связаться с оператором."
)


def baseline_a(message: str, history: list[dict] | None = None, use_cache: bool | None = None) -> dict:
    t0 = time.perf_counter()
    masked = pii.mask(message)
    msgs = [{"role": "system", "content": SYSTEM_A}]
    for h in history or []:
        msgs.append({"role": h["role"], "content": pii.mask(h["content"]).text})
    msgs.append({"role": "user", "content": masked.text})
    t_mask = time.perf_counter() - t0
    r = llm.chat(msgs, role="simple", temperature=0.3, max_tokens=400, use_cache=use_cache, tag="baseline_a")
    text = r.text if not r.error else "Извините, сервис временно недоступен. Попробуйте позже."
    esc = bool(re.search(r"оператор|специалист|горяч\w* лини|позвоните|обратитесь в (банк|отделение)", text.lower()))
    return {"system": "A", "text": text, "escalate": esc, "route": "llm", "masked_text": masked.text,
            "latency": {"pii": round(t_mask, 4), "llm": round(r.latency_s, 4),
                        "total": round(time.perf_counter() - t0, 4)},
            "tokens": {"prompt": r.prompt_tokens, "completion": r.completion_tokens},
            "cost_rub": r.cost_rub if r.cost_rub is not None else cost_rub(r.model, r.prompt_tokens, r.completion_tokens), "model": r.model, "error": r.error}


@lru_cache(maxsize=1)
def _faq() -> dict:
    with open(ROOT / "data" / "faq_baseline.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def baseline_b(message: str, history: list[dict] | None = None, **_) -> dict:
    t0 = time.perf_counter()
    if re.search(r"оператор|человек|сотрудник", message.lower()):
        intent, esc = "operator", True
    else:
        intent = keywords.predict_one(message)
        esc = intent in ("compromised_card", "lost_or_stolen_card")
    text = _faq().get(intent, _faq()["other"])
    return {"system": "B", "text": text, "escalate": esc, "route": f"faq:{intent}", "intent": intent,
            "latency": {"total": round(time.perf_counter() - t0, 4)}, "tokens": {"prompt": 0, "completion": 0},
            "cost_rub": 0.0}
