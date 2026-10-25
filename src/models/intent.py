"""Классификатор интента: обучение, оценка, рантайм.

python -m src.models.intent train      # шаг 1 (+ RU-оценка, если есть переводы шага 2)

Модели: majority, keywords, TF-IDF+LR, e5-эмбеддинги+LR (основная). Метрики: accuracy, macro-F1.
Рантайм-модель (для пайплайна) учится на EN train + RU train (если есть).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.config import cfg, path
from src.models import keywords
from src.pipeline.templates import ACUTE, SCOPE

EMB_MODEL = cfg()["classifier"]["embedding_model"]


# ── эмбеддинги ──
@lru_cache(maxsize=2)
def encoder(name: str = EMB_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def _prefix(name: str) -> str:
    return "query: " if "e5" in name else ""


def embed(texts: list[str], name: str = EMB_MODEL, batch_size: int = 64) -> np.ndarray:
    return encoder(name).encode([_prefix(name) + t for t in texts], batch_size=batch_size,
                                normalize_embeddings=True, show_progress_bar=len(texts) > 500)


def embed_cached(texts: list[str], tag: str, name: str = EMB_MODEL) -> np.ndarray:
    f = path("processed") / f"emb_{name.split('/')[-1]}_{tag}.npy"
    if f.exists():
        arr = np.load(f)
        if len(arr) == len(texts):
            return arr
    arr = embed(texts, name)
    np.save(f, arr)
    return arr


# ── метрики ──
def to_scope(labels) -> list[str]:
    keep = set(SCOPE) | set(ACUTE)
    return [l if l in keep else keywords.OTHER for l in labels]


def metrics(y_true, y_pred) -> dict:
    return {"n": len(y_true), "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4)}


def scope_metrics(y_true, y_pred) -> dict:
    """15 скоуп + 2 острых + other (как считает сценарный бот)."""
    return metrics(to_scope(y_true), to_scope(y_pred))


def ece(conf, correct, bins: int = 15) -> float:
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return round(float(e), 4)


def coverage_curve(conf, correct, grid=None) -> list[dict]:
    conf, correct = np.asarray(conf), np.asarray(correct)
    grid = np.round(np.arange(0.0, 1.0, 0.05), 2) if grid is None else grid
    out = []
    for t in grid:
        m = conf >= t
        out.append({"threshold": float(t), "coverage": round(float(m.mean()), 4),
                    "accuracy": round(float(correct[m].mean()), 4) if m.any() else None})
    return out


def top_confusions(y_true, y_pred, k: int = 10) -> list[dict]:
    s = pd.Series([f"{t} → {p}" for t, p in zip(y_true, y_pred) if t != p]).value_counts().head(k)
    return [{"pair": i, "count": int(c)} for i, c in s.items()]


# ── данные ──
def load_en():
    tr = pd.read_csv(path("raw") / "banking77_train.csv")
    te = pd.read_csv(path("raw") / "banking77_test.csv")
    return tr, te


def load_ru():
    ftr, fte = path("eval") / "banking77_ru_train.csv", path("eval") / "banking77_ru_test.csv"
    if not (ftr.exists() and fte.exists()):
        return None, None
    tr = pd.read_csv(ftr)
    # Продуктовая таксономия: по брифу receiving_money включает «где мои реквизиты», а в Banking77 таких
    # реплик почти нет (провал E066). Эксперимент с 25 ручными репликами (data/experiments/) закрыл E066, но
    # снизил RU macro-F1 0,866 → 0,858 и сдвинул выбор α → 0, поэтому выключен: нужна разметка реальных чатов.
    # Чтобы включить — положить файл в data/product_taxonomy_extra.csv и переобучить.
    extra = path("eval").parent / "product_taxonomy_extra.csv"
    if extra.exists():
        tr = pd.concat([tr, pd.read_csv(extra)], ignore_index=True)
    return tr, pd.read_csv(fte)


class EmbLR:
    """Эмбеддинги + логрегрессия. predict_proba принимает тексты."""

    def __init__(self, name: str = EMB_MODEL, C: float = 10.0):
        self.name, self.C = name, C
        self.lr = LogisticRegression(C=C, max_iter=3000)

    T: float = 1.0  # температура (калибровка), подбирается на валидации

    def fit_emb(self, X, y):
        self.lr.fit(X, y)
        return self

    def calibrate(self, Xva, yva):
        """Temperature scaling: минимизируем NLL на валидации."""
        from scipy.optimize import minimize_scalar

        z = self.lr.decision_function(Xva)
        idx = np.searchsorted(self.classes_, yva)

        def nll(t):
            zt = z / t
            zt = zt - zt.max(1, keepdims=True)
            return -(zt[np.arange(len(idx)), idx] - np.log(np.exp(zt).sum(1))).mean()

        self.T = float(minimize_scalar(nll, bounds=(0.05, 5), method="bounded").x)
        return self

    @property
    def classes_(self):
        return self.lr.classes_

    def proba_emb(self, X):
        z = self.lr.decision_function(X) / self.T
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict_proba(self, texts):
        return self.proba_emb(embed(list(texts), self.name))


def train_and_eval() -> dict:
    tr_all, te = load_en()
    tr_idx, va_idx = train_test_split(np.arange(len(tr_all)), test_size=0.1, stratify=tr_all.label,
                                      random_state=cfg()["seed"])
    tr_idx, va_idx = np.sort(tr_idx), np.sort(va_idx)
    tr, va = tr_all.iloc[tr_idx], tr_all.iloc[va_idx]  # порядок строк = порядок эмбеддингов
    ru_tr, ru_te = load_ru()
    res: dict = {"split": {"train": len(tr), "val": len(va), "test": len(te)}, "en": {}, "ru": {}}

    def run(name, fit_pred):
        t0 = time.perf_counter()
        out = fit_pred()
        print(f"  {name}: {time.perf_counter() - t0:.1f} c")
        return out

    # 1. majority
    maj = DummyClassifier(strategy="most_frequent").fit(tr.text, tr.label)
    res["en"]["majority"] = {**metrics(te.label, maj.predict(te.text)),
                             "scope": scope_metrics(te.label, maj.predict(te.text))}
    # 2. keywords (только скоуп + other)
    kw = keywords.predict(te.text)
    res["en"]["keywords"] = {"scope": scope_metrics(te.label, kw),
                             "note": "предсказывает только 15+2 скоуп-интента или other"}
    # 3. TF-IDF + LR
    tfidf = run("tfidf", lambda: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(C=20, max_iter=3000)).fit(tr.text, tr.label))
    p = tfidf.predict(te.text)
    res["en"]["tfidf_lr"] = {**metrics(te.label, p), "scope": scope_metrics(te.label, p)}
    # 4. e5 + LR
    Xtr = embed_cached(tr_all.text.tolist(), "b77_train")
    Xte = embed_cached(te.text.tolist(), "b77_test")
    Xtr_, Xva_ = Xtr[tr_idx], Xtr[va_idx]
    emb = run("e5+lr", lambda: EmbLR().fit_emb(Xtr_, tr.label.values))
    P = emb.proba_emb(Xte)
    p = emb.classes_[P.argmax(1)]
    res["en"]["e5_lr"] = {**metrics(te.label, p), "scope": scope_metrics(te.label, p)}

    # Калибровка и пороги — на валидации (EN val), проверка — на test
    ece_raw = ece(P.max(1), p == te.label.values)
    emb.calibrate(Xva_, va.label.values)
    P = emb.proba_emb(Xte)
    Pva = emb.proba_emb(Xva_)
    conf_va, ok_va = Pva.max(1), emb.classes_[Pva.argmax(1)] == va.label.values
    conf_te, ok_te = P.max(1), p == te.label.values
    res["calibration"] = {"temperature": round(emb.T, 3), "ece_test_before": ece_raw,
                          "ece_val": ece(conf_va, ok_va), "ece_test": ece(conf_te, ok_te)}
    res["coverage_curve_test_en"] = coverage_curve(conf_te, ok_te)
    th = choose_thresholds(conf_va, ok_va, Pva, emb.classes_, va.label.values)
    res["thresholds"] = th
    res["confusions_en"] = top_confusions(te.label, p)
    res["_cm_en"] = (te.label.tolist(), p.tolist())

    # ── RU ──
    if ru_te is not None:
        Xru_te = embed_cached(ru_te.text.tolist(), "ru_test")
        Xru_tr = embed_cached(ru_tr.text.tolist(), "ru_train")
        # EN → RU (кросс-язычный перенос)
        P_ru = emb.proba_emb(Xru_te)
        p_ru = emb.classes_[P_ru.argmax(1)]
        res["ru"]["e5_lr_en_only"] = {**metrics(ru_te.label, p_ru), "scope": scope_metrics(ru_te.label, p_ru)}
        # EN + RU-train → RU
        emb2 = EmbLR().fit_emb(np.vstack([Xtr_, Xru_tr]), np.concatenate([tr.label.values, ru_tr.label.values]))
        emb2.T = emb.T
        P2 = emb2.proba_emb(Xru_te)
        p2 = emb2.classes_[P2.argmax(1)]
        res["ru"]["e5_lr_en_ru"] = {**metrics(ru_te.label, p2), "scope": scope_metrics(ru_te.label, p2)}
        res["ru"]["coverage_curve"] = coverage_curve(P2.max(1), p2 == ru_te.label.values)
        res["ru"]["ece"] = ece(P2.max(1), p2 == ru_te.label.values)
        res["confusions_ru"] = top_confusions(ru_te.label, p2)
        # Пороги для прода (русский): out-of-fold по RU-train, RU-test не трогаем
        from sklearn.model_selection import StratifiedKFold

        P_oof = np.zeros((len(ru_tr), len(emb.classes_)))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=cfg()["seed"]).split(Xru_tr, ru_tr.label):
            m = EmbLR().fit_emb(np.vstack([Xtr_, Xru_tr[a]]), np.concatenate([tr.label.values, ru_tr.label.values[a]]))
            m.T = emb.T
            P_oof[b] = m.proba_emb(Xru_tr[b])
        y_oof = ru_tr.label.values
        ok_oof = emb.classes_[P_oof.argmax(1)] == y_oof
        th_ru = choose_thresholds(P_oof.max(1), ok_oof, P_oof, emb.classes_, y_oof)
        res["thresholds_en_val"] = th
        res["thresholds"] = th = {**th_ru, "source": "RU-train out-of-fold (5 фолдов)"}
        res["ru"]["ece_oof"] = ece(P_oof.max(1), ok_oof)
        np.savez(path("models") / "intent_ru_oof.npz", P=P_oof, y=y_oof, classes=emb.classes_)
        res["_cm_ru"] = (ru_te.label.tolist(), p2.tolist())
        # Бейзлайны на RU
        res["ru"]["majority"] = scope_metrics(ru_te.label, maj.predict(ru_te.text))
        res["ru"]["keywords"] = {"scope": scope_metrics(ru_te.label, keywords.predict(ru_te.text))}
        t_en = tfidf.predict(ru_te.text)
        res["ru"]["tfidf_lr_en_only"] = {**metrics(ru_te.label, t_en)}
        tfidf2 = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                               LogisticRegression(C=20, max_iter=3000)).fit(
            pd.concat([tr.text, ru_tr.text]), pd.concat([tr.label, ru_tr.label]))
        t2 = tfidf2.predict(ru_te.text)
        res["ru"]["tfidf_lr_en_ru"] = {**metrics(ru_te.label, t2), "scope": scope_metrics(ru_te.label, t2)}

    # ── рантайм-модель: весь EN train + RU train ──
    X_rt, y_rt = Xtr, tr_all.label.values
    if ru_tr is not None:
        X_rt = np.vstack([X_rt, embed_cached(ru_tr.text.tolist(), "ru_train")])
        y_rt = np.concatenate([y_rt, ru_tr.label.values])
    rt = EmbLR().fit_emb(X_rt, y_rt)
    rt.T = emb.T
    with open(path("models") / "intent_e5_lr.pkl", "wb") as f:
        pickle.dump({"model": rt, "thresholds": th, "trained_on": len(y_rt)}, f)
    return res


def choose_thresholds(conf, ok, P, classes, y, min_acc_answer: float = 0.95, min_top3_clarify: float = 0.85):
    """answer: минимальный порог, при котором точность среди отвеченных ≥ min_acc_answer.
    clarify: минимальный порог, при котором истинный интент в топ-3 у ≥ min_top3_clarify кейсов полосы."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    answer = next((float(t) for t in grid if (conf >= t).any() and ok[conf >= t].mean() >= min_acc_answer), 0.9)
    top3 = np.argsort(-P, 1)[:, :3]
    in_top3 = np.array([y[i] in classes[top3[i]] for i in range(len(y))])
    # Локальное правило: опускаем порог шагами по 0,05, пока в бине [t, t+0,1) истинный интент
    # попадает в топ-3 у ≥ min_top3_clarify (усреднение по всей полосе маскирует плохой низ).
    clarify = answer
    t = round(answer - 0.05, 2)
    while t > 0:
        b = (conf >= t) & (conf < min(t + 0.1, answer))
        if b.sum() < 10 or in_top3[b].mean() < min_top3_clarify:
            break
        clarify = t
        t = round(t - 0.05, 2)
    return {"answer": answer, "clarify": clarify,
            "val_coverage_answer": round(float((conf >= answer).mean()), 4),
            "val_acc_answer": round(float(ok[conf >= answer].mean()), 4),
            "val_share_clarify": round(float(((conf >= clarify) & (conf < answer)).mean()), 4),
            "val_share_handoff": round(float((conf < clarify).mean()), 4),
            "rule": f"answer: acc≥{min_acc_answer}; clarify: top3≥{min_top3_clarify} в каждом бине 0,1"}


# ── рантайм ──
@lru_cache(maxsize=1)
def runtime():
    with open(path("models") / "intent_e5_lr.pkl", "rb") as f:
        return pickle.load(f)


def proba(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (вероятности по 77 классам, классы, эмбеддинг)."""
    m = runtime()["model"]
    v = embed([text], m.name)[0]
    return m.proba_emb(v.reshape(1, -1))[0], m.classes_, v


def classify(text: str, emb_vec: np.ndarray | None = None) -> tuple[str, float, list[tuple[str, float]], np.ndarray]:
    m = runtime()["model"]
    v = embed([text], m.name)[0] if emb_vec is None else emb_vec
    p = m.proba_emb(v.reshape(1, -1))[0]
    order = np.argsort(-p)[:5]
    top = [(str(m.classes_[i]), float(p[i])) for i in order]
    return top[0][0], top[0][1], top, v


def thresholds() -> dict:
    """Пороги: от слияния с DWH-априором (если включено), иначе классификатора, иначе из config.yaml."""
    from src.models.fusion import params

    fp = params()
    if fp and fp.get("alpha", 0) > 0:
        return fp["thresholds"]
    try:
        return runtime()["thresholds"]
    except FileNotFoundError:
        return cfg()["classifier"]["thresholds"]


def save_report(res: dict) -> None:
    from src.eval.figures import confusion_scope, coverage_plot

    cm_en, cm_ru = res.pop("_cm_en", None), res.pop("_cm_ru", None)
    with open(path("reports") / "step1_classifier.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    if cm_en:
        confusion_scope(*cm_en, "confusion_scope_en.png", "Banking77 EN test, e5+LR")
    if cm_ru:
        confusion_scope(*cm_ru, "confusion_scope_ru.png", "Banking77 RU test, e5+LR (EN+RU train)")
    coverage_plot(res)


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        # импорт через пакет, чтобы pickle сохранил класс как src.models.intent.EmbLR, а не __main__.EmbLR
        from src.models import intent as _self

        r = _self.train_and_eval()
        _self.save_report(r)
        print(json.dumps({k: v for k, v in r.items() if not k.startswith("coverage")}, ensure_ascii=False, indent=1))
