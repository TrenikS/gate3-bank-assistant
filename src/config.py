"""Загрузка config.yaml и путей проекта."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Абсолютный путь из секции paths конфига; директории создаются по требованию."""
    p = ROOT / cfg()["paths"][key]
    (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def price_per_1k(model: str) -> dict | None:
    """{"input": ₽/1k, "output": ₽/1k} или None, если цена не заполнена."""
    p = (cfg().get("price_rub_per_1k_tokens") or {}).get(model)
    if p is None:
        return None
    return p if isinstance(p, dict) else {"input": p, "output": p}


def cost_rub(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    p = price_per_1k(model)
    if p is None or p.get("input") is None or p.get("output") is None:
        return None
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1000


def fmt_rub(x: float | None) -> str:
    return "н/д" if x is None else f"{x:.2f} ₽"
