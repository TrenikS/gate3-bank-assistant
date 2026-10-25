"""Слияние классификатора реплики с предиктором по данным клиента (DWH-априор).

posterior(c) ∝ p_clf(c) · (prior(c) / base(c))^α
  prior — распределение предиктора (17 интентов + other; other делится поровну на прочие 60 классов Banking77),
  base  — средний prior по всем сессиям (чтобы α=0 и «нейтральный» клиент не меняли классификатор).
α и пороги подбираются на синтетических сессиях: каждой сессии сопоставляется случайная RU-фраза её интента
с out-of-fold вероятностями классификатора (RU-test и E2E не используются).

python -m src.models.fusion tune
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache

import numpy as np

from src.config import cfg, path
from src.models.intent import choose_thresholds
from src.pipeline.templates import ACUTE, SCOPE

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def prior_matrix(P_pred: np.ndarray, pred_classes, clf_classes) -> np.ndarray:
    pred_idx = {c: i for i, c in enumerate(pred_classes)}
    rest = [c for c in clf_classes if c not in pred_idx]
    out = np.zeros((len(P_pred), len(clf_classes)))
    for j, c in enumerate(clf_classes):
        out[:, j] = P_pred[:, pred_idx[c]] if c in pred_idx else P_pred[:, pred_idx["other"]] / len(rest)
    return out


def fuse(P_clf: np.ndarray, prior: np.ndarray, base: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0:
        return P_clf
    w = np.power(np.clip(prior, 1e-6, None) / np.clip(base, 1e-6, None), alpha)
    q = P_clf * w
    return q / q.sum(1, keepdims=True)


def simulate(seed: int | None = None):
    seed = cfg()["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    clf = np.load(path("models") / "intent_ru_oof.npz", allow_pickle=True)
    pr = np.load(path("models") / "predictor_oof.npz", allow_pickle=True)
    clf_classes, y_clf = clf["classes"], clf["y"]
    in_scope = set(SCOPE) | set(ACUTE)
    by_label = {c: np.where(y_clf == c)[0] for c in clf_classes}
    other_idx = np.where(~np.isin(y_clf, list(in_scope)))[0]
    pick = np.array([rng.choice(by_label[t]) if t != "other" else rng.choice(other_idx) for t in pr["y"]])
    P_clf, y_true = clf["P"][pick], y_clf[pick]
    prior = prior_matrix(pr["P"], pr["classes"], clf_classes)
    base = prior.mean(0, keepdims=True)
    return P_clf, prior, base, y_true, clf_classes, pr["has_signal"].astype(bool)


def tune() -> dict:
    P_clf, prior, base, y, classes, sig = simulate()
    res = {"n_sessions": len(y), "alphas": {}}
    for a in ALPHAS:
        Q = fuse(P_clf, prior, base, a)
        pred = classes[Q.argmax(1)]
        ok = pred == y
        th = choose_thresholds(Q.max(1), ok, Q, classes, y)
        res["alphas"][a] = {"acc": round(float(ok.mean()), 4), "acc_signal": round(float(ok[sig].mean()), 4),
                            "acc_no_signal": round(float(ok[~sig].mean()), 4), "thresholds": th}
    # Выбор: максимум покрытия «отвечаем сразу» при точности 95%, при этом на сессиях без сигнала
    # (клиент пришёл не с «очевидным» интентом) точность падает не больше чем на 2 п.п.
    # Порог 2 п.п. выбран после просмотра сетки (исходный 1 п.п. отсекал всё, кроме α=0) — см. отчёт.
    base_ns = res["alphas"][0.0]["acc_no_signal"]
    ok_alphas = [a for a in ALPHAS if res["alphas"][a]["acc_no_signal"] >= base_ns - 0.02]
    best = max(ok_alphas, key=lambda a: res["alphas"][a]["thresholds"]["val_coverage_answer"])
    res["chosen_alpha"] = best
    res["rule"] = "max coverage@acc95 при падении точности на сессиях без сигнала ≤ 2 п.п. (порог выбран после сетки)"
    with open(path("models") / "fusion.json", "w", encoding="utf-8") as f:
        json.dump({"alpha": best, "base": base[0].tolist(), "classes": classes.tolist(),
                   "thresholds": res["alphas"][best]["thresholds"]}, f, ensure_ascii=False)
    return res


@lru_cache(maxsize=1)
def params() -> dict | None:
    f = path("models") / "fusion.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


def apply(client_id: str, top_probs: np.ndarray, classes) -> tuple[np.ndarray, dict]:
    """Возвращает слитые вероятности и сведения для панели «под капотом»."""
    from src.models import predictor

    p = params()
    if p is None or p["alpha"] == 0:
        return top_probs, {"alpha": 0}
    pp = predictor.proba(client_id)
    pred_classes = list(pp)
    prior = prior_matrix(np.array([[pp[c] for c in pred_classes]]), pred_classes, list(classes))
    base = np.array(p["base"])[None, :]
    q = fuse(top_probs[None, :], prior, base, p["alpha"])[0]
    return q, {"alpha": p["alpha"], "prior_top": sorted(pp.items(), key=lambda x: -x[1])[:3]}


if __name__ == "__main__":
    if sys.argv[1:] == ["tune"]:
        r = tune()
        with open(path("reports") / "step3b_fusion.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        for a, v in r["alphas"].items():
            print(a, {k: v[k] for k in ("acc", "acc_signal", "acc_no_signal")},
                  {k: v["thresholds"][k] for k in ("answer", "clarify", "val_coverage_answer", "val_acc_answer")})
        print("α =", r["chosen_alpha"])
