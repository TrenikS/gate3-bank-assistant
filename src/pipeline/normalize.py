"""Нормализация реплики перед классификатором: русский транслит → кириллица.

Эмбеддинги плохо понимают «pochemu otklonili oplatu». Если в тексте нет кириллицы, строим
кириллический вариант; классификатор оценивает оба и берёт более уверенный (английский текст не ломается).
"""
from __future__ import annotations

import re

_MULTI = [("shch", "щ"), ("sch", "щ"), ("zh", "ж"), ("kh", "х"), ("ch", "ч"), ("sh", "ш"), ("ts", "ц"),
          ("yu", "ю"), ("ya", "я"), ("yo", "ё"), ("ye", "е"), ("iy", "ий"), ("yy", "ый"), ("y", "ы")]
_SINGLE = dict(zip("abvgdezijklmnoprstufhcwxq", "абвгдезийклмнопрстуфхцвкск"))
_CYR = re.compile(r"[а-яё]", re.I)
_MASK = re.compile(r"\[[A-Z_]+_\d+\]")


def _word(w: str) -> str:
    low = w.lower()
    out, i = [], 0
    while i < len(low):
        for lat, cyr in _MULTI:
            if low.startswith(lat, i):
                # «y» в начале слова или после гласной — «й» (moy → мой), иначе «ы»
                if lat == "y" and i > 0 and low[i - 1] in "aeiou":
                    cyr = "й"
                out.append(cyr)
                i += len(lat)
                break
        else:
            ch = low[i]
            # «i» в конце слова после гласной — «й» (kartoi → картой)
            out.append("й" if ch == "i" and i == len(low) - 1 and i > 0 and low[i - 1] in "aeiou"
                       else _SINGLE.get(ch, ch))
            i += 1
    return "".join(out).replace("'", "ь")


def translit_to_cyrillic(text: str) -> str | None:
    """None, если текст уже кириллический или латиницы почти нет."""
    if _CYR.search(text):
        return None
    words = re.findall(r"[A-Za-z']+", _MASK.sub(" ", text))
    if len(words) < 2:
        return None
    # маски ПДн ([CARD_1]) оставляем как есть, транслитерируем только текст между ними
    parts = re.split(f"({_MASK.pattern})", text)
    return "".join(p if _MASK.fullmatch(p) else re.sub(r"[A-Za-z']+", lambda m: _word(m.group(0)), p)
                   for p in parts)
