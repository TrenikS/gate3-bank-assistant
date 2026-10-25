"""Шаг 3: предиктор подсказок при открытии чата (LightGBM multiclass → топ-3 интента).

python -m src.models.predictor train
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import cfg, path
from src.data.synth_dwh import SIM_NOW
from src.pipeline import dwh

SCREENS = ["Реквизиты", "Лимиты", "История операций", "Детали операции", "Выписка", "Переводы", "Карта",
           "Настройки карты", "Профиль", "Личные данные", "Заблокировать карту", "Главная", "Платежи",
           "Кешбэк", "Вклады"]
SUGGESTABLE = "other"  # «other» не показывается в подсказках


def _ts(minutes: float) -> str:
    return (SIM_NOW - timedelta(minutes=minutes)).isoformat(sep=" ")


def features(client_id: str) -> dict:
    """Признаки по DWH на момент открытия чата. Только агрегаты — без ПДн."""
    q1 = lambda sql, *a: list(dwh.con().execute(sql, (client_id, *a)).fetchone())[0] or 0
    now = SIM_NOW.isoformat(sep=" ")
    f = {
        "declined_card_15m": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND status='declined' "
                                "AND decline_reason!='fraud_suspected' AND ts>=?", _ts(15)),
        "declined_card_24h": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND status='declined' "
                                "AND decline_reason!='fraud_suspected' AND ts>=?", _ts(24 * 60)),
        "declined_card_7d": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND status='declined' AND ts>=?",
                               _ts(7 * 24 * 60)),
        "declined_wrong_pin_24h": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND "
                                     "decline_reason='wrong_pin' AND ts>=?", _ts(24 * 60)),
        "fraud_declines_2h": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND "
                                "decline_reason='fraud_suspected' AND ts>=?", _ts(120)),
        "foreign_tx_24h": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND is_foreign=1 AND ts>=?",
                             _ts(24 * 60)),
        "declined_transfer_30m": q1("SELECT COUNT(*) FROM transfers WHERE client_id=? AND status='declined' "
                                    "AND ts>=?", _ts(30)),
        "declined_transfer_7d": q1("SELECT COUNT(*) FROM transfers WHERE client_id=? AND status='declined' "
                                   "AND ts>=?", _ts(7 * 24 * 60)),
        "fee_tx_2d": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND fee>0 AND ts>=?", _ts(2 * 24 * 60)),
        "fee_tx_30d": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND fee>0 AND ts>=?",
                         _ts(30 * 24 * 60)),
        "cryptic_tx_3d": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND descriptor!=UPPER(merchant) "
                            "AND is_service_charge=0 AND ts>=?", _ts(3 * 24 * 60)),
        "duplicate_2d": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND duplicate_of IS NOT NULL "
                           "AND ts>=?", _ts(2 * 24 * 60)),
        "service_charge_7d": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND is_service_charge=1 "
                                "AND ts>=?", _ts(7 * 24 * 60)),
        "transfer_processing": q1("SELECT COUNT(*) FROM transfers WHERE client_id=? AND "
                                  "status IN ('processing','pending') AND expected_at>=?", now),
        "transfer_overdue": q1("SELECT COUNT(*) FROM transfers WHERE client_id=? AND "
                               "status IN ('processing','pending') AND expected_at<?", now),
        "transfer_completed_awaiting": q1("SELECT COUNT(*) FROM transfers WHERE client_id=? AND status='completed' "
                                          "AND expected_at>=?", now),
        "pending_card_tx": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND status='pending' "
                              "AND duplicate_of IS NULL", ),
        "pending_card_tx_old": q1("SELECT COUNT(*) FROM transactions WHERE client_id=? AND status='pending' "
                                  "AND duplicate_of IS NULL AND ts<?", _ts(24 * 60)),
        "card_in_delivery": q1("SELECT COUNT(*) FROM cards WHERE client_id=? AND status='in_delivery'"),
        "card_not_activated": q1("SELECT COUNT(*) FROM cards WHERE client_id=? AND status='not_activated'"),
        "card_blocked": q1("SELECT COUNT(*) FROM cards WHERE client_id=? AND status='blocked'"),
        "card_blocked_1h": q1("SELECT COUNT(*) FROM cards WHERE client_id=? AND status='blocked' AND blocked_at>=?",
                              _ts(60)),
        "refund_pending": q1("SELECT COUNT(*) FROM refunds WHERE client_id=? AND status='pending' AND expected_by>=?",
                             now),
        "refund_overdue": q1("SELECT COUNT(*) FROM refunds WHERE client_id=? AND status='pending' AND expected_by<?",
                             now),
        "limit_usage": q1("SELECT MAX(spent_today/daily_limit) FROM cards WHERE client_id=?"),
        # tx_30d (число операций за 30 дней) убран: генератор добавляет транзакцию при закладке драйвера,
        # и счётчик становится утечкой метки, а не поведенческим признаком.
    }
    ev = dict(dwh.con().execute("SELECT screen, COUNT(*) FROM app_events WHERE client_id=? AND ts>=? GROUP BY screen",
                                (client_id, _ts(30))).fetchall())
    for s in SCREENS:
        f[f"screen_{s}"] = ev.get(s, 0)
    return f


def build_dataset() -> pd.DataFrame:
    s = pd.DataFrame(dwh.q("SELECT s.*, c.is_story FROM sessions s JOIN clients c USING(client_id)"))
    s = s[s.is_story == 0].reset_index(drop=True)
    X = pd.DataFrame([features(c) for c in s.client_id])
    return pd.concat([s[["client_id", "true_intent", "has_signal", "distractor_intent"]], X], axis=1)


def rank_metrics(P: np.ndarray, classes: np.ndarray, y: np.ndarray, ks=(1, 3)) -> dict:
    """«other» не показываем: исключаем его из ранжирования; сессии с other — всегда промах."""
    P = P.copy()
    if SUGGESTABLE in classes:
        P[:, list(classes).index(SUGGESTABLE)] = -1
    order = np.argsort(-P, 1)
    ranks = np.array([list(classes[order[i]]).index(y[i]) + 1 if y[i] != SUGGESTABLE else np.inf
                      for i in range(len(y))])
    out = {f"hit@{k}": round(float((ranks <= k).mean()), 4) for k in ks}
    out["mrr"] = round(float(np.where(np.isfinite(ranks), 1 / ranks, 0).mean()), 4)
    return out


def train() -> dict:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    df = build_dataset()
    feats = [c for c in df.columns if c not in ("client_id", "true_intent", "has_signal", "distractor_intent")]
    X, y = df[feats].values.astype(float), df.true_intent.values
    classes = np.array(sorted(set(y)))
    P = np.zeros((len(y), len(classes)))
    params = dict(objective="multiclass", n_estimators=300, learning_rate=0.05, num_leaves=15,
                  min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, verbose=-1, random_state=cfg()["seed"])
    skf = StratifiedKFold(5, shuffle=True, random_state=cfg()["seed"])
    for tr, te in skf.split(X, y):
        m = lgb.LGBMClassifier(**params).fit(X[tr], y[tr])
        P[te] = m.predict_proba(X[te])[:, [list(m.classes_).index(c) for c in classes]]
    np.savez(path("models") / "predictor_oof.npz", P=P, classes=classes, y=y, client_id=df.client_id.values,
             has_signal=df.has_signal.values)
    res = {"n_sessions": len(y), "n_features": len(feats), "cv": "5-fold stratified",
           "model": rank_metrics(P, classes, y)}
    # Бейзлайн: топ-k самых частых интентов (без other) — по train-частотам каждого фолда ≈ глобальные
    freq = pd.Series(y[y != SUGGESTABLE]).value_counts()
    base_P = np.tile([freq.get(c, 0) for c in classes], (len(y), 1)).astype(float)
    res["baseline_top_frequent"] = {**rank_metrics(base_P, classes, y), "top3": freq.index[:3].tolist()}
    sig = df.has_signal.values == 1
    res["model_signal_only"] = rank_metrics(P[sig], classes, y[sig])
    res["model_no_signal"] = rank_metrics(P[~sig & (y != "other")], classes, y[~sig & (y != "other")])
    res["share_noise"] = round(float((~sig).mean()), 4)
    res["share_other"] = round(float((y == "other").mean()), 4)

    final = lgb.LGBMClassifier(**params).fit(X, y)
    imp = pd.Series(final.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    res["feature_importance_top15"] = {k: round(float(v), 1) for k, v in imp.head(15).items()}
    # Проверка драйверов: какой признак сильнее всего двигает каждый интент (по средним в классе)
    drivers = {}
    for c in classes:
        if c == "other":
            continue
        sd = df[feats].std().replace(0, 1)
        mu_c, mu_rest = df.loc[df.true_intent == c, feats].mean(), df.loc[df.true_intent != c, feats].mean()
        lift = ((mu_c - mu_rest) / sd).sort_values(ascending=False)
        drivers[c] = lift.index[0]
    res["top_driver_by_class"] = drivers

    with open(path("models") / "predictor_lgbm.pkl", "wb") as f:
        pickle.dump({"model": final, "features": feats}, f)
    return res


@lru_cache(maxsize=1)
def _model():
    with open(path("models") / "predictor_lgbm.pkl", "rb") as f:
        return pickle.load(f)


def proba(client_id: str) -> dict[str, float]:
    """Полное распределение предиктора по интентам (включая other) — априор для слияния с классификатором."""
    m = _model()
    f = features(client_id)
    p = m["model"].predict_proba(np.array([[f[c] for c in m["features"]]], dtype=float))[0]
    return {str(c): float(v) for c, v in zip(m["model"].classes_, p)}


def suggest(client_id: str, k: int = 3) -> list[tuple[str, float]]:
    m = _model()
    f = features(client_id)
    p = m["model"].predict_proba(np.array([[f[c] for c in m["features"]]], dtype=float))[0]
    order = [i for i in np.argsort(-p) if m["model"].classes_[i] != SUGGESTABLE][:k]
    return [(str(m["model"].classes_[i]), float(p[i])) for i in order]


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        t0 = time.perf_counter()
        r = train()
        with open(path("reports") / "step3_predictor.json", "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=2)
        from src.eval.figures import hit_at_k

        hit_at_k({1: r["model"]["hit@1"], 3: r["model"]["hit@3"]},
                 {1: r["baseline_top_frequent"]["hit@1"], 3: r["baseline_top_frequent"]["hit@3"]})
        print(json.dumps(r, ensure_ascii=False, indent=1))
        print(f"{time.perf_counter() - t0:.1f} c")
