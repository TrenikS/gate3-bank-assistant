"""Автоматическая проверка критичных фактов в ответе против DWH клиента.

critical — ответ утверждает про операцию клиента что-то, противоречащее DWH:
  * сумма в рублях, которой нет среди фактов клиента;
  * дата («5 октября», «05.10»), которой нет среди фактов;
  * статус с потерянным отрицанием: «деньги списаны», когда по данным они НЕ списаны.
unverified — общие сроки/числа без опоры на данные клиента («до 3 рабочих дней», «до 30 дней»).
"""
from __future__ import annotations

import re

from src.pipeline import dwh
from src.pipeline.guardrails import numbers

MONEY = re.compile(r"(\d[\d  ]*(?:[.,]\d+)?)\s*(?:₽|руб|р\.)", re.I)
DATE = re.compile(r"\b(\d{1,2})\s+(январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*"
                  r"|\b(\d{1,2})\.(\d{2})(?:\.\d{2,4})?\b", re.I)
TERM = re.compile(r"\d+\s*(?:-|–|до)?\s*\d*\s*(?:рабоч\w+\s+)?(?:дн|день|дня|час|минут|недел|месяц)", re.I)
# Интенты/варианты, где по данным деньги НЕ списаны
NOT_CHARGED = {"declined_card_payment", "declined_transfer", "pending_card_payment"}
CHARGED_CLAIM = re.compile(r"(?<!не )(?<!не были )(деньги|средства|сумма|оплата)\s+(уже\s+|были\s+)?(списан|снят)\w*", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[  ]", "", s).replace(",", ".")


def check(text: str, client_id: str, intent: str | None, variant: str | None, user_message: str = "") -> dict:
    facts = dwh.client_fact_strings(client_id)
    fact_nums = set()
    for f in facts:
        fact_nums.update(numbers(f))
    msg_nums = set(numbers(user_message))
    errors, unverified = [], []
    for m in MONEY.finditer(text):
        n = _norm(m.group(1)).rstrip(".")
        if n not in fact_nums and n not in msg_nums:
            errors.append(f"сумма {m.group(0).strip()} не из DWH")
    fact_text = " ".join(facts).lower()
    for m in DATE.finditer(text):
        s = m.group(0).lower()
        if m.group(1):
            key = f"{m.group(1)} {m.group(2)}"
            if key not in fact_text:
                errors.append(f"дата «{m.group(0)}» не из DWH")
        else:
            errors.append(f"дата «{m.group(0)}» не из DWH")
    if intent in NOT_CHARGED or variant == "hold":
        if CHARGED_CLAIM.search(text):
            errors.append("утверждает, что деньги списаны (по DWH — нет)")
    for m in TERM.finditer(text):
        nums = numbers(m.group(0))
        if any(n not in fact_nums for n in nums):
            unverified.append(m.group(0).strip())
    return {"critical_errors": errors, "unverified_terms": unverified,
            "has_critical": bool(errors), "has_unverified": bool(unverified)}
