"""Шаг 7: латентность подсказок и сводка экономики.

python -m src.eval.latency
Латентность ответа по этапам берётся из E2E-прогона (src.eval.e2e run — он идёт с холодным кэшем LLM).
"""
from __future__ import annotations

import json
import random
import time

from src.config import cfg, path
from src.eval.stats import pct
from src.pipeline import dwh


def suggestions(n: int = 300) -> dict:
    from src.pipeline.assistant import suggest

    ids = [r["client_id"] for r in dwh.q("SELECT client_id FROM clients")]
    random.Random(cfg()["seed"]).shuffle(ids)
    suggest(ids[0])  # прогрев (загрузка модели)
    lat = []
    for cid in ids[:n]:
        t = time.perf_counter()
        suggest(cid)
        lat.append(time.perf_counter() - t)
    return {"n": n, "p50": pct(lat, 50), "p95": pct(lat, 95), "max": round(max(lat), 4), "target_s": 0.3}


def economics(e2e: dict) -> dict:
    """Перенос на прод: банк Gate 2 (30 млн обращений в год, оператор ≈ 100 ₽)."""
    ec = cfg()["economics"]
    ours = e2e["systems"]["ours"]
    cost = ours.get("cost_rub_mean")
    solved = ours.get("judge_gcr_in_scope") or ours.get("human_gcr_in_scope")
    no_human = 1 - ours["escalation_rate_in_scope"]["p"]
    out = {"cost_per_dialog_rub": cost, "share_without_human_in_scope": round(no_human, 4),
           "gcr_in_scope": solved, "contacts_per_year": ec["contacts_per_year"],
           "operator_cost_rub": ec["operator_cost_rub"], "chats_per_month_prod": ec["chats_per_month_prod"]}
    if cost is not None:
        out["llm_cost_per_month_rub"] = round(cost * ec["chats_per_month_prod"])
        out["tokens_per_month"] = round(ours["tokens_per_dialog_mean"] * ec["chats_per_month_prod"])
    return out


if __name__ == "__main__":
    s = suggestions()
    e2e = json.load(open(path("reports") / "step5_e2e.json", encoding="utf-8"))
    res = {"suggestions": s, "stages": e2e.get("latency_stages_ours"), "cost_by_route": e2e.get("cost_by_route_ours"),
           "economics": economics(e2e)}
    with open(path("reports") / "step7_latency_cost.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=1))
