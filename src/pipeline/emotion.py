"""Детектор раздражения: score 0–1.

Основа — быстрые правила (капс, восклицания, мат, лексика раздражения). Если обучен классификатор
на эмбеддингах (src/models/emotion_train.py), итог = max(правила, модель).
"""
from __future__ import annotations

import pickle
import re
from functools import lru_cache

from src.config import path

B = r"(?<![а-яёa-z])"
MAT = re.compile(B + r"(ху[йяеёию]|пизд|п[иі]зд|[её]ба|заеб|заёб|наеб|наёб|уеб|уёб|ебан|ебуч|бля|блят|"
                 r"сука|суки|мудак|мудил|херн|нахер|нахр|охуе|охрен|долбо|говн|дерьм|жоп|задолба|"
                 r"fuck|shit|wtf)", re.IGNORECASE)
IRRITATION = re.compile(r"достали|бесит|бесите|сколько можно|безобрази|ужас|кошмар|издевательств|"
                        r"немедленно|вы что|какого|опять|задолбал|дурдом|позор|отстой|обман|разводилов|"
                        r"идиот|тупые|тупой|бред|возмутительн|верните|достало|надоел|невозможно|"
                        r"что за (фигня|хрень|бред)|где мои деньги|почему так долго|сколько ждать",
                        re.IGNORECASE)


def heuristic(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters) if len(letters) >= 8 else 0.0
    s = 0.0
    if caps >= 0.6:
        s += 0.45
    elif caps >= 0.3:
        s += 0.2
    s += min(0.3, 0.15 * len(re.findall(r"[!?]{2,}|!", text)))
    s += 0.5 if MAT.search(text) else 0.0
    s += min(0.5, 0.25 * len(IRRITATION.findall(text)))
    return round(min(1.0, s), 3)


@lru_cache(maxsize=1)
def _model():
    f = path("models") / "emotion.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def score(text: str, embedding=None) -> float:
    h = heuristic(text)
    m = _model()
    if m is None or embedding is None:
        return h
    p = float(m.predict_proba(embedding.reshape(1, -1))[0, 1])
    return round(max(h, p), 3)
