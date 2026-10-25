"""Маскирование ПДн до любого вызова LLM.

mask(text) -> MaskResult(text=маскированный текст, replacements={"[CARD_1]": "4276 ...", ...})
Словарь замен остаётся в коде и в LLM не уходит.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class MaskResult:
    text: str
    replacements: dict[str, str] = field(default_factory=dict)

    @property
    def found(self) -> list[str]:
        return sorted({k.strip("[]").rsplit("_", 1)[0] for k in self.replacements})


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_SEP = r"[ \-]?"
# Порядок важен: сначала длинные/контекстные шаблоны
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("ACCOUNT", re.compile(r"(?<!\d)\d{5}" + _SEP + r"\d{3}" + _SEP + r"\d" + _SEP + r"\d{4}" + _SEP + r"\d{7}(?!\d)")),
    ("ACCOUNT", re.compile(r"(?<!\d)\d{20}(?!\d)")),
    ("CARD", re.compile(r"(?<!\d)(?:\d{4}" + _SEP + r"){3}\d{4}(?:" + _SEP + r"\d{1,3})?(?!\d)")),
    ("CARD", re.compile(r"(?<!\d)\d{13,19}(?!\d)")),
    ("PHONE", re.compile(r"(?<![\d+])(?:\+7|8|7)[ \-]?\(?\d{3}\)?[ \-]?\d{3}[ \-]?\d{2}[ \-]?\d{2}(?!\d)")),
    ("PASSPORT", re.compile(r"(?i)(?:паспорт\w*|серия)[^\d]{0,20}(\d{2}\s?\d{2}[ №N]*\d{6})")),
    ("PASSPORT", re.compile(r"(?<!\d)\d{2}\s\d{2}\s\d{6}(?!\d)")),
    ("SNILS", re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}[ -]\d{2}(?!\d)")),
    ("CVC", re.compile(r"(?i)(?:cvc2?|cvv2?|свв|свс|кода? на обороте|три цифры[^\d]{0,15})[^\d]{0,12}(\d{3})(?!\d)")),
    ("EXPIRY", re.compile(r"(?i)(?:срок\w*|действ\w*|exp\w*|до)[^\d]{0,15}((?:0[1-9]|1[0-2])\s?/\s?(?:\d{2}|20\d{2}))(?!\d)")),
    ("EXPIRY", re.compile(r"(?<![\d/.])(?:0[1-9]|1[0-2])/(?:2[4-9]|3\d)(?![\d/])")),
    ("SMS_CODE", re.compile(r"(?i)(?:код\w*|смс|sms|пароль|code|otp)[^\d]{0,25}(\d{4,6})(?!\d)")),
    ("LONGNUM", re.compile(r"(?<!\d)\d{10,}(?!\d)")),
]


@lru_cache(maxsize=1)
def _natasha():
    try:
        from natasha import Doc, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Segmenter

        emb = NewsEmbedding()
        return Doc, Segmenter(), NewsNERTagger(emb), NewsMorphTagger(emb)
    except Exception:  # natasha не установилась — работаем только на регулярках
        return None


def natasha_available() -> bool:
    return _natasha() is not None


# Слова, которые natasha иногда принимает за имена в банковском контексте
_NAME_STOP = {"сбер", "сбербанк", "тинькофф", "visa", "мир", "mastercard", "яндекс", "озон", "ozon"}


@lru_cache(maxsize=1)
def _merchants() -> set[str]:
    """Справочник мерчантов (в проде — из справочника банка): названия магазинов — не ПДн."""
    from src.data.synth_dwh import CRYPTIC, MERCHANTS

    return {m.lower() for m in MERCHANTS} | {c[1].lower() for c in CRYPTIC}


def _is_merchant(span: str) -> bool:
    s = span.lower()
    return any(s in m or m in s for m in _merchants())


def _names(text: str) -> list[tuple[int, int]]:
    nat = _natasha()
    if nat is None or not re.search(r"[А-ЯЁ][а-яё]", text):
        return []
    Doc, seg, ner, morph = nat
    doc = Doc(text)
    doc.segment(seg)
    doc.tag_ner(ner)
    doc.tag_morph(morph)
    pos = {(t.start, t.stop): t.pos for t in doc.tokens}
    out = []
    for s in doc.spans:
        if s.type != "PER" or s.text.lower() in _NAME_STOP or _is_merchant(s.text):
            continue
        # NER путает глаголы в начале фразы («Скажи», «Отправь») с именами: нужен хотя бы один PROPN
        if not any(pos.get((t.start, t.stop)) == "PROPN" for t in s.tokens):
            continue
        # Одно слово в начале предложения: заглавная буква сбивает теггер — проверяем в нижнем регистре
        sentence_start = text[:s.start].rstrip(" \t\n«\"(")[-1:] in ("", ".", "!", "?")
        if len(s.tokens) == 1 and sentence_start:
            low = Doc(text[:s.start] + s.text.lower() + text[s.stop:])
            low.segment(seg)
            low.tag_morph(morph)
            if any(t.start == s.start and t.pos == "VERB" for t in low.tokens):
                continue
        out.append((s.start, s.stop))
    return out


def mask(text: str, use_ner: bool = True) -> MaskResult:
    spans: list[tuple[int, int, str]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e, _ in spans)

    for kind, rx in PATTERNS:
        for m in rx.finditer(text):
            a, b = (m.start(1), m.end(1)) if m.groups() else (m.start(), m.end())
            value = text[a:b]
            if kind == "CARD":
                digits = re.sub(r"\D", "", value)
                if not (13 <= len(digits) <= 19):
                    continue
                if not luhn_ok(digits):
                    kind_eff = "LONGNUM"
                else:
                    kind_eff = "CARD"
            else:
                kind_eff = kind
            if free(a, b):
                spans.append((a, b, kind_eff))
    if use_ner:
        for a, b in _names(text):
            if free(a, b):
                spans.append((a, b, "NAME"))

    spans.sort()
    counters: dict[str, int] = {}
    seen: dict[tuple[str, str], str] = {}
    out, pos, repl = [], 0, {}
    for a, b, kind in spans:
        value = text[a:b]
        tag = seen.get((kind, value))
        if tag is None:
            counters[kind] = counters.get(kind, 0) + 1
            tag = f"[{kind}_{counters[kind]}]"
            seen[(kind, value)] = tag
            repl[tag] = value
        out.append(text[pos:a])
        out.append(tag)
        pos = b
    out.append(text[pos:])
    return MaskResult("".join(out), repl)


def unmask(text: str, replacements: dict[str, str]) -> str:
    for tag, value in replacements.items():
        text = text.replace(tag, value)
    return text


def mask_messages(messages: list[dict]) -> list[dict]:
    """Страховка в llm.chat: всё, что ушло бы в LLM немаскированным, маскируется."""
    return [{**m, "content": mask(m["content"]).text} for m in messages]
