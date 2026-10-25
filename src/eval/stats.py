"""Статистика для маленьких выборок: интервалы Уилсона, бутстрэп, перцентили."""
from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prop(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "p": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def fmt_prop(d: dict) -> str:
    if not d or not d.get("n"):
        return "—"
    return f"{d['p']:.0%} [{d['ci95'][0]:.0%}–{d['ci95'][1]:.0%}] (n={d['n']})"


def mcnemar_exact(b: int, c: int) -> float:
    """Точный двусторонний тест Макнемара для парных бинарных исходов: b, c — расходящиеся пары."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_mean(x, n_boot: int = 2000, seed: int = 0) -> dict:
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x))).mean(1)
    return {"mean": round(float(x.mean()), 4), "ci95": [round(float(np.percentile(means, 2.5)), 4),
                                                     round(float(np.percentile(means, 97.5)), 4)], "n": len(x)}


def pct(x, q) -> float | None:
    x = [v for v in x if v is not None]
    return round(float(np.percentile(x, q)), 4) if x else None
