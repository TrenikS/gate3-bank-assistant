"""Графики для отчёта (reports/figures)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.config import path
from src.models.keywords import OTHER
from src.pipeline.templates import ACUTE, SCOPE

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C1, C2, C3 = "#2563eb", "#f59e0b", "#9ca3af"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def confusion_scope(y_true, y_pred, fname: str, title: str) -> None:
    labels = SCOPE + ACUTE + [OTHER]
    keep = set(labels)
    yt = [y if y in keep else OTHER for y in y_true]
    yp = [y if y in keep else OTHER for y in y_pred]
    cm = confusion_matrix(yt, yp, labels=labels).astype(float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else INK)
    ax.set_xlabel("предсказано", color=MUTED)
    ax.set_ylabel("истина", color=MUTED)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path("figures") / fname, dpi=150)
    plt.close(fig)


def coverage_plot(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    curves = [("EN test", res.get("coverage_curve_test_en"), C1)]
    if res.get("ru", {}).get("coverage_curve"):
        curves.append(("RU test", res["ru"]["coverage_curve"], C2))
    for name, curve, color in curves:
        pts = [(c["coverage"], c["accuracy"]) for c in curve if c["accuracy"] is not None]
        ax.plot(*zip(*pts), marker="o", ms=3, lw=2, color=color, label=name)
    th = res.get("thresholds", {})
    if th:
        ax.axvline(th["val_coverage_answer"], color=C3, ls="--", lw=1)
        ax.text(th["val_coverage_answer"], ax.get_ylim()[0] + 0.01, f" порог «отвечать» = {th['answer']}",
                color=MUTED, fontsize=8)
    _style(ax)
    ax.set_xlabel("coverage (доля, на которую отвечаем без уточнения)", color=MUTED, fontsize=9)
    ax.set_ylabel("accuracy среди отвеченных", color=MUTED, fontsize=9)
    ax.set_title("Coverage – accuracy классификатора (e5+LR)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path("figures") / "coverage_accuracy.png", dpi=150)
    plt.close(fig)


def hit_at_k(model_hits: dict, base_hits: dict) -> None:
    ks = sorted(model_hits)
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(x - 0.18, [model_hits[k] for k in ks], 0.36, color=C1, label="LightGBM предиктор")
    ax.bar(x + 0.18, [base_hits[k] for k in ks], 0.36, color=C3, label="топ самых частых")
    for i, k in enumerate(ks):
        ax.text(i - 0.18, model_hits[k] + 0.01, f"{model_hits[k]:.0%}", ha="center", fontsize=8, color=INK)
        ax.text(i + 0.18, base_hits[k] + 0.01, f"{base_hits[k]:.0%}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x, [f"Hit@{k}" for k in ks])
    ax.set_ylim(0, 1)
    _style(ax)
    ax.set_title("Предиктор подсказок (синтетика, 5-fold CV)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path("figures") / "predictor_hit_at_k.png", dpi=150)
    plt.close(fig)


def latency_stages(stage_p50: dict, stage_p95: dict) -> None:
    names = list(stage_p50)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(y + 0.18, [stage_p50[n] for n in names], 0.36, color=C1, label="p50")
    ax.barh(y - 0.18, [stage_p95[n] for n in names], 0.36, color=C2, label="p95")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("секунды", color=MUTED, fontsize=9)
    ax.set_title("Латентность по этапам (холодный кэш LLM)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path("figures") / "latency_stages.png", dpi=150)
    plt.close(fig)
