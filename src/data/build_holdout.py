"""Отложенный E2E-набор data/eval/e2e_holdout.jsonl — честная проверка после доработок v2.

Отличия от e2e.jsonl (dev-набора, на котором разбирали ошибки v1):
- реплики не написаны мной: это реальные запросы клиентов Banking77 (переведённый RU-test). RU-test не участвовал
  ни в обучении, ни в подборе порогов и α (они подбирались на RU-train out-of-fold и синтетических сессиях);
- клиенты — реалистичная смесь: у 70% в данных есть сигнал под интент, у 30% нет (в генераторе DWH у 25% сессий
  драйвер не закладывается), поэтому DWH-априор помогает не всегда;
- набор собран один раз до прогона и больше не меняется.

python -m src.data.build_holdout
"""
from __future__ import annotations

import json
import random

import pandas as pd

from src.config import cfg, path
from src.pipeline import dwh, templates
from src.pipeline.templates import ACUTE, SCOPE

N_PER_INTENT = 10      # все 10 RU-test реплик на интент
NO_SIGNAL_PER_INTENT = 3
N_OOS = 30             # по одной реплике из 30 случайных интентов вне скоупа


def _clients(intent: str, has_signal: int, n: int, used: set, rng: random.Random) -> list[str]:
    rows = dwh.q("SELECT s.client_id FROM sessions s JOIN clients c USING(client_id) WHERE c.is_story=0 "
                 "AND s.true_intent=? AND s.has_signal=? AND s.distractor_intent IS NULL", intent, has_signal)
    ids = sorted({r["client_id"] for r in rows} - used)
    rng.shuffle(ids)
    if len(ids) < n:
        raise RuntimeError(f"Мало клиентов для {intent} (signal={has_signal}): {len(ids)}")
    used.update(ids[:n])
    return ids[:n]


def _reference(client_id: str, intent: str) -> str:
    f = dwh.facts_for(client_id, intent)
    return templates.fill(templates.get(intent, f.variant).text, f.values)


def build() -> list[dict]:
    rng = random.Random(cfg()["seed"] + 1)
    ru = pd.read_csv(path("eval") / "banking77_ru_test.csv")
    dev_clients = {json.loads(l)["client_id"] for l in open(path("eval") / "e2e.jsonl", encoding="utf-8")}
    used: set = set(dev_clients)
    cases: list[dict] = []

    def add(**kw):
        cases.append(dict(id=f"H{len(cases) + 1:03d}", history=[], **kw))

    for intent in SCOPE:
        texts = ru[ru.label == intent].text.tolist()[:N_PER_INTENT]
        n_sig = len(texts) - NO_SIGNAL_PER_INTENT
        cl = _clients(intent, 1, n_sig, used, rng) + _clients(intent, 0, NO_SIGNAL_PER_INTENT, used, rng)
        for i, (msg, cid) in enumerate(zip(texts, cl)):
            f = dwh.facts_for(cid, intent)
            add(category="in_scope", intent=intent, style="banking77", has_signal=int(i < n_sig), message=msg,
                client_id=cid, expected="answer", segment=templates.segment(intent), variant=f.variant,
                facts=f.values, reference=_reference(cid, intent))
    for intent in ACUTE:
        texts = ru[ru.label == intent].text.tolist()[:N_PER_INTENT]
        cl = _clients(intent, 1, len(texts), used, rng)
        for msg, cid in zip(texts, cl):
            add(category="escalation", intent=intent, escalation_type="fraud", style="banking77", has_signal=1,
                message=msg, client_id=cid, expected="escalate", segment="острые",
                reference=templates.system_text("fraud"))
    other_intents = sorted(set(ru.label) - set(SCOPE) - set(ACUTE))
    rng.shuffle(other_intents)
    others = sorted({r["client_id"] for r in dwh.q("SELECT client_id FROM sessions WHERE true_intent='other' "
                                                   "AND distractor_intent IS NULL")} - used)
    rng.shuffle(others)
    for intent, cid in zip(other_intents[:N_OOS], others):
        msg = ru[ru.label == intent].text.tolist()[0]
        add(category="out_of_scope", intent=None, b77_intent=intent, style="banking77", has_signal=0,
            message=msg, client_id=cid, expected="handoff", segment=None,
            reference=templates.system_text("handoff_out_of_scope"))
    return cases


def main() -> None:
    cases = build()
    with open(path("eval") / "e2e_holdout.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    df = pd.DataFrame(cases)
    print(df.groupby(["category", "has_signal"]).size().to_string(), f"\nвсего: {len(cases)}")


if __name__ == "__main__":
    main()
