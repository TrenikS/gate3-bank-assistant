"""Гардрейлы ответа.

(a) все числа и даты в ответе ⊂ фактам DWH для клиента ∪ числам, написанным в проверенном шаблоне;
(b) плейсхолдеры сохранены, новых нет, обязательные фразы (отрицания) на месте;
(c) нет данных других клиентов;
(d) нет необоснованных обещаний (стоп-фразы);
(e) текст из данных (описание операции) не показываем клиенту, если он похож на инъекцию.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from src.pipeline.templates import PLACEHOLDER

NUM = re.compile(r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?")
STOP_PHRASES = [
    "гарантир", "точно вернём", "точно вернем", "обязательно вернём", "обязательно вернем",
    "100%", "сто процентов", "без сомнени", "обещаю", "можете не переживать, деньги", "в течение часа",
    "сегодня же вернём", "деньги уже вернулись", "уже вернули", "мы компенсируем", "мы возместим",
]
MONTH_WORDS = ("январ", "феврал", "март", "апрел", "мая", "июн", "июл", "август", "сентябр", "октябр",
               "ноябр", "декабр", "вчера", "завтра", "сегодня")


# Описание операции приходит от продавца/эквайера — это недоверенный текст. Настоящие описания короткие
# (лимит платёжных систем ~25 символов) и не содержат e-mail, ссылок и командных слов.
INJECTION_WORDS = re.compile(r"\b(ignore|instruction|prompt|system|send|reveal|forget|password|"
                             r"игнорир|инструкци|забудь|промпт|отправ|пароль)", re.IGNORECASE)


def descriptor_suspicious(text: str) -> bool:
    t = text or ""
    return (len(t) > 30 or "@" in t or bool(re.search(r"https?://", t, re.I))
            or bool(INJECTION_WORDS.search(t)) or bool(re.search(r"[.!?]\s+\S", t)))


@dataclass
class GuardResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def numbers(text: str) -> list[str]:
    return [re.sub(r"[  ]", "", n).replace(",", ".") for n in NUM.findall(text)]


def check_paraphrase(template_text: str, paraphrase: str, must_keep: list[str]) -> GuardResult:
    """Проверка перефраза LLM ДО подстановки значений."""
    v = []
    if Counter(PLACEHOLDER.findall(template_text)) != Counter(PLACEHOLDER.findall(paraphrase)):
        v.append("placeholders_changed")
    if re.search(r"\{[^}]*\}", PLACEHOLDER.sub("", paraphrase)):
        v.append("placeholder_malformed")
    extra = set(numbers(PLACEHOLDER.sub("", paraphrase))) - set(numbers(PLACEHOLDER.sub("", template_text)))
    if extra:
        v.append(f"llm_added_numbers:{sorted(extra)}")
    low = paraphrase.lower()
    for phrase in must_keep:
        if phrase.lower() not in low:
            v.append(f"must_keep_missing:{phrase}")
    # Отрицание: число «не» не должно уменьшиться (LLM могла «потерять» отрицание)
    neg = lambda s: len(re.findall(r"(?<![а-яё])(?:не|нет|ни)(?![а-яё])", s.lower()))
    if neg(paraphrase) < neg(template_text):
        v.append("negation_lost")
    tmpl_low = template_text.lower()
    for s in STOP_PHRASES:
        if s in low and s not in tmpl_low:
            v.append(f"promise:{s}")
    return GuardResult(not v, v)


def check_final(text: str, allowed_values: set[str], template_text: str,
                other_clients: list[str] | None = None) -> GuardResult:
    """Проверка итогового текста после подстановки."""
    v = []
    allowed_nums = set()
    for s in allowed_values:
        allowed_nums.update(numbers(str(s)))
    allowed_nums.update(numbers(PLACEHOLDER.sub("", template_text)))
    bad = [n for n in numbers(text) if n not in allowed_nums]
    if bad:
        v.append(f"numbers_not_in_facts:{bad}")
    for ident in other_clients or []:
        if ident and ident in text:
            v.append("other_client_data")
            break
    low = text.lower()
    for s in STOP_PHRASES:
        if s in low and s not in template_text.lower():
            v.append(f"promise:{s}")
    return GuardResult(not v, v)
