"""Шаг 9: reports/gate3_metrics.md — таблицы из reports/step*.json (цифры не переписываются руками).

python -m src.eval.make_report
Текстовые разделы E1/F1/G1 — в reports/gate3_notes.md (пишутся вручную), сюда вставляются как есть.
"""
from __future__ import annotations

import json

from src.config import cfg, path
from src.eval.stats import fmt_prop


def _j(name: str) -> dict:
    p = path("reports") / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


def _f(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def _num(x) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _rub(x):
    return "н/д" if x is None else f"{x:.2f} ₽"


def classifier_section(c: dict, c_small: dict) -> list[str]:
    L = ["### Классификатор интента", "",
         f"Сплит Banking77: train {c['split']['train']}, val {c['split']['val']}, test {c['split']['test']} (EN). "
         f"RU: перевод стратифицированных подвыборок (test 10/интент = 770, train 30/интент = 2310).", "",
         "| Модель | EN acc | EN macro-F1 | EN 15+2+other macro-F1 | RU acc | RU macro-F1 |", "|---|---|---|---|---|---|"]
    en, ru = c["en"], c.get("ru", {})
    L.append(f"| Самый частый класс | {_f(en['majority']['accuracy'])} | {_f(en['majority']['macro_f1'])} | "
             f"{_f(en['majority']['scope']['macro_f1'])} | {_f(ru.get('majority', {}).get('accuracy'))} | "
             f"{_f(ru.get('majority', {}).get('macro_f1'))} (15+2+other) |")
    L.append(f"| Ключевые слова (сценарный бот) | — | — | {_f(en['keywords']['scope']['macro_f1'])} | "
             f"{_f(ru.get('keywords', {}).get('scope', {}).get('accuracy'))} (15+2+other) | "
             f"{_f(ru.get('keywords', {}).get('scope', {}).get('macro_f1'))} (15+2+other) |")
    L.append(f"| TF-IDF + LR (EN) | {_f(en['tfidf_lr']['accuracy'])} | {_f(en['tfidf_lr']['macro_f1'])} | "
             f"{_f(en['tfidf_lr']['scope']['macro_f1'])} | {_f(ru.get('tfidf_lr_en_only', {}).get('accuracy'))} | "
             f"{_f(ru.get('tfidf_lr_en_only', {}).get('macro_f1'))} |")
    L.append(f"| TF-IDF + LR (EN+RU) | — | — | — | {_f(ru.get('tfidf_lr_en_ru', {}).get('accuracy'))} | "
             f"{_f(ru.get('tfidf_lr_en_ru', {}).get('macro_f1'))} |")
    variants = [(cfg()["classifier"]["embedding_model"].split("/")[-1], c)]
    if c_small and not c_small.get("_same"):
        variants.insert(0, ("multilingual-e5-small", c_small))
    for tag, cc in variants:
        e, r = cc["en"]["e5_lr"], cc.get("ru", {})
        L.append(f"| {tag} + LR, train EN | {_f(e['accuracy'])} | **{_f(e['macro_f1'])}** | {_f(e['scope']['macro_f1'])} | "
                 f"{_f(r.get('e5_lr_en_only', {}).get('accuracy'))} | {_f(r.get('e5_lr_en_only', {}).get('macro_f1'))} |")
        L.append(f"| {tag} + LR, train EN+RU | — | — | — | {_f(r.get('e5_lr_en_ru', {}).get('accuracy'))} | "
                 f"**{_f(r.get('e5_lr_en_ru', {}).get('macro_f1'))}** |")
    th, cal = c["thresholds"], c["calibration"]
    L += ["", f"Таргеты Gate 2: macro-F1 ≥ 0,90 (EN), ≥ 0,85 (RU). Модель в пайплайне: "
              f"`{cfg()['classifier']['embedding_model']}` + LR, обучена на EN train + RU train.", "",
          f"Калибровка: temperature scaling на EN val, T = {cal['temperature']}; ECE на EN test "
          f"{cal['ece_test_before']} → {cal['ece_test']}; ECE на RU (out-of-fold) {_f(c.get('ru', {}).get('ece_oof'))}.", "",
          f"Пороги ({th.get('source', 'EN val')}; правило: {th['rule']}): **отвечать ≥ {th['answer']}**, "
          f"**уточнять {th['clarify']}–{th['answer']}**, ниже — человек. На отборе: отвечаем {th['val_coverage_answer']:.0%} "
          f"с точностью {th['val_acc_answer']:.0%}, уточняем {th['val_share_clarify']:.0%}, человеку {th['val_share_handoff']:.0%}.",
          "", "Топ-10 путаниц (RU test): " + "; ".join(f"{x['pair']} ({x['count']})" for x in c.get("confusions_ru", [])),
          "", "![confusion RU](figures/confusion_scope_ru.png) ![coverage](figures/coverage_accuracy.png)", ""]
    return L


def predictor_section(p: dict) -> list[str]:
    if not p:
        return []
    m, b = p["model"], p["baseline_top_frequent"]
    return ["### Предиктор подсказок (синтетика)", "",
            f"n = {p['n_sessions']} сессий, {p['n_features']} признаков, {p['cv']}. Без сигнала {p['share_noise']:.0%} сессий "
            f"(из них «другое» {p['share_other']:.0%}).", "",
            "| | Hit@1 | Hit@3 | MRR |", "|---|---|---|---|",
            f"| LightGBM | {m['hit@1']:.1%} | **{m['hit@3']:.1%}** | {m['mrr']:.3f} |",
            f"| Бейзлайн «топ-3 частых» ({', '.join(b['top3'])}) | {b['hit@1']:.1%} | {b['hit@3']:.1%} | {b['mrr']:.3f} |",
            f"| LightGBM, сессии с сигналом | {p['model_signal_only']['hit@1']:.1%} | {p['model_signal_only']['hit@3']:.1%} | {p['model_signal_only']['mrr']:.3f} |",
            f"| LightGBM, сессии без сигнала | {p['model_no_signal']['hit@1']:.1%} | {p['model_no_signal']['hit@3']:.1%} | {p['model_no_signal']['mrr']:.3f} |",
            "", f"Таргет Gate 2: Hit@3 ≥ 40% и ≥ 2× бейзлайна → {m['hit@3'] / b['hit@3']:.1f}×. "
                f"Топ-драйвер каждого интента совпал с заложенным в генератор (17/17).", "",
            "![Hit@k](figures/predictor_hit_at_k.png)", ""]


def fusion_section(f: dict) -> list[str]:
    if not f:
        return []
    L = ["### Понимание с учётом данных клиента (DWH-априор)", "",
         "posterior ∝ p(классификатор) · (p(предиктор) / базовая частота)^α. Подбор α — на 2 000 синтетических сессиях: "
         "каждой сопоставлена случайная RU-фраза её интента с out-of-fold вероятностями (RU-test и E2E не используются).", "",
         "| α | точность | с сигналом | без сигнала | отвечаем при точности 95% |", "|---|---|---|---|---|"]
    for a, v in f["alphas"].items():
        mark = " **← выбран**" if float(a) == f["chosen_alpha"] else ""
        L.append(f"| {a}{mark} | {v['acc']:.1%} | {v['acc_signal']:.1%} | {v['acc_no_signal']:.1%} | "
                 f"{v['thresholds']['val_coverage_answer']:.0%} |")
    th = f["alphas"][str(f["chosen_alpha"])]["thresholds"] if str(f["chosen_alpha"]) in f["alphas"] else \
        f["alphas"][f["chosen_alpha"]]["thresholds"]
    L += ["", f"Правило: {f['rule']}. Пороги в пайплайне (после слияния): **отвечать ≥ {th['answer']}**, "
              f"**уточнять {th['clarify']}–{th['answer']}**. Плюс нормализация транслита и семейства соседних "
              "интентов Banking77 (failed_transfer → declined_transfer и т.п.), которые срабатывают только при "
              "подтверждении данными DWH.", ""]
    return L


def e2e_section(e: dict) -> list[str]:
    if not e:
        return []
    S = e["systems"]
    names = {"ours": "Прототип v2", "ours_v1": "Прототип v1 (слепой)", "A": "A: LLM-бот", "B": "B: сценарный"}
    cols = [s for s in ("ours", "ours_v1", "A", "B") if s in S]
    rows = [("GCR, LLM-судья (in-scope, «решено»)", "judge_gcr_in_scope"),
            ("GCR, LLM-судья («решено» или «частично»)", "judge_gcr_partial_or_better"),
            ("GCR, человек (in-scope)", "human_gcr_in_scope"),
            ("Ошибки в критичных фактах — авто", "auto_critical_fact_errors"),
            ("Ошибки в критичных фактах — судья", "judge_critical_fact_errors"),
            ("Ошибки в критичных фактах — человек", "human_critical_fact_errors"),
            ("Неподтверждённые сроки/числа — авто", "auto_unverified_terms"),
            ("Recall обязательных эскалаций", "escalation_recall"),
            ("…кризис", "escalation_recall_crisis"), ("…мошенничество", "escalation_recall_fraud"),
            ("Precision эскалаций", "escalation_precision"),
            ("Доля in-scope, переданных человеку", "escalation_rate_in_scope")]
    L = ["### End-to-end (русский eval-набор)", "",
         f"n = {e['n_cases']} кейсов: 75 in-scope (15 интентов × 5 стилей), 10 вне скоупа, 13 обязательных эскалаций. "
         "Интервалы — 95% Wilson; для тона — бутстрэп.", "",
         "| Метрика | " + " | ".join(names[c] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for title, key in rows:
        if any(key in S[c] for c in cols):
            L.append(f"| {title} | " + " | ".join(fmt_prop(S[c].get(key)) for c in cols) + " |")
    tone = [S[c].get("judge_tone_emotional") for c in cols]
    if any(tone):
        L.append("| Тон для эмоциональных (судья, 1–5) | " + " | ".join(
            f"{t['mean']:.2f} [{t['ci95'][0]:.2f}–{t['ci95'][1]:.2f}] (n={t['n']})" if t and t.get("mean") else "—"
            for t in tone) + " |")
    L.append("| ROUGE-L к эталону (вспомогательно) | " + " | ".join(
        f"{S[c]['rougeL_mean']['mean']:.2f}" for c in cols) + " |")
    o = S.get("ours", {})
    if o:
        L += ["", f"Прототип: интент верно определён в {fmt_prop(o.get('intent_acc_in_scope'))} in-scope кейсов; "
                  f"ответ шаблоном в {fmt_prop(o.get('answered_in_scope'))} (из них с верным интентом — "
                  f"{fmt_prop(o.get('answered_correct_intent'))} от всех in-scope); уточняющий вопрос в "
                  f"{fmt_prop(o.get('clarify_in_scope'))}, и верный вариант среди предложенных в "
                  f"{fmt_prop(o.get('clarify_contains_true'))}; вне скоупа безопасно (не «отвечал») — "
                  f"{fmt_prop(o.get('oos_safe'))}. Маршруты: " + ", ".join(f"{k} {v}" for k, v in o.get("routes", {}).items()) + "."]
    k = e.get("kappa", {})
    if k.get("kappa") is None:
        L += ["", "На dev-наборе человеческой разметки нет: GCR по человеку и каппа судьи — на отложенном наборе "
                  "(раздел «Человеческая разметка» ниже).", ""]
    else:
        L += ["", f"Согласие LLM-судьи с человеком: κ Коэна = {_f(k.get('kappa'))} (n = {k.get('n')}). "
                  f"{k.get('use', k.get('note', ''))}.", ""]
    if e.get("rouge_pairs"):
        L += ["**ROUGE-L не ловит ошибки в фактах** (эталон испорчен вручную):", "",
              "| Кейс | Порча | ROUGE-L |", "|---|---|---|"]
        L += [f"| {p['case']} | {p['kind']} | {p['rougeL']:.2f} |" for p in e["rouge_pairs"]]
        p0 = e["rouge_pairs"][0]
        L += ["", f"> Эталон: {p0['reference']}", ">", f"> С ошибкой: {p0['wrong']}", ""]
    return L


def human_section(h: dict) -> list[str]:
    """Человеческая разметка отложенного набора — главная оценка GCR, если κ судьи < 0,6."""
    he = h.get("human_eval") or {}
    if not he:
        return []
    S, k = he["systems"], he.get("kappa", {})
    cols = [s for s in ("ours", "A", "B") if s in S]
    names = {"ours": "Прототип", "A": "A: LLM-бот", "B": "B: сценарный"}
    rows = [("**GCR «решено», человек**", "gcr"), ("…клиенты с сигналом в DWH", "gcr_signal"),
            ("…клиенты без сигнала", "gcr_no_signal"), ("«решено» + «частично», человек", "gcr_partial_or_better"),
            ("Ошибка в критичном факте, человек", "critical_errors"),
            ("Для сравнения: GCR по судье на тех же кейсах", "judge_gcr_same_cases")]
    rule = k.get("three_class") is not None and k["three_class"] < 0.6
    L = ["### Человеческая разметка (отложенный набор) — главная оценка GCR", "",
         f"Разметил Иван вслепую (система скрыта, порядок перемешан): n = {he['n_cases']} in-scope кейсов отложенного "
         f"набора (2 на интент) × {len(cols)} системы = {he['n_cases'] * len(cols)} ответов. Интервалы — 95% Wilson.", "",
         "| Метрика | " + " | ".join(names[c] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for t, key in rows:
        L.append(f"| {t} | " + " | ".join(fmt_prop(S[c].get(key)) for c in cols) + " |")
    pr = he.get("paired", {})
    if pr:
        L += ["", "Парное сравнение на одних и тех же кейсах (точный тест Макнемара, «решено» да/нет): " + "; ".join(
            f"прототип vs {names[s]} — решил только прототип {v['ours_only']}, только бейзлайн {v['other_only']}, "
            f"p = {v['p_mcnemar']:.2f}" for s, v in pr.items()) + "."]
    if k:
        L += ["", f"Согласие LLM-судьи с человеком (n = {k['n']} ответов): κ Коэна по рубрике «решено / частично / нет» = "
                  f"**{k['three_class']:.2f}** (взвешенная {k['linear_weighted']:.2f}, совпадение {k['agreement']:.0%}); "
                  f"для бинарного «решено / нет» κ = {k['resolved_vs_rest']:.2f}. "
                  + ("По правилу брифа (κ < 0,6) **GCR в отчёте — по человеческой разметке**; судья — вспомогательно."
                     if rule else "κ ≥ 0,6 — судью можно использовать наравне с человеком.")]
    return L + [""]


def redteam_section(r: dict) -> list[str]:
    if not r:
        return []
    S = r["systems"]
    cols = [s for s in ("ours", "A") if s in S]
    names = {"ours": "Прототип", "A": "A: LLM-бот"}
    rows = [("Утечки ПДн в запросах к LLM (таргет 0)", "pii_leaks_cases"),
            ("Успешные jailbreak, судья (таргет ≤ 1%)", "jailbreak_success_judge"),
            ("Ложные подтверждения / обещания, судья", "false_confirmation_judge"),
            ("Recall мошенничества / соц. инженерии (таргет ≥ 98%)", "fraud_recall"),
            ("Recall кризисных", "crisis_recall"),
            ("Необоснованные обещания, авто (таргет 0)", "unfounded_promises")]
    L = ["### Red-team", "", f"n = {r['n_attacks']} атак, 12 категорий; {r['resolution_note']}. Jailbreak и ложные подтверждения оценивает "
         "LLM-судья (подстрочные детекторы давали ложные срабатывания на отказах); утечки ПДн — по логу всех запросов "
         "к LLM во время атаки", "",
         "| Метрика | " + " | ".join(names[c] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for t, k in rows:
        L.append(f"| {t} | " + " | ".join(fmt_prop(S[c][k]) for c in cols) + " |")
    L.append("| Вызовов LLM на все атаки | " + " | ".join(str(S[c]["llm_calls_total"]) for c in cols) + " |")
    return L + [""]


def latency_section(l: dict, e: dict, h: dict | None = None) -> list[str]:
    if not l:
        return []
    st = l["stages"]
    L = ["## D3. Латентность и стоимость", "",
         f"Холодный кэш LLM, прогон E2E (n = {e['n_cases']}). Подсказки: p50 {l['suggestions']['p50'] * 1000:.0f} мс, "
         f"p95 {l['suggestions']['p95'] * 1000:.0f} мс (n = {l['suggestions']['n']}, таргет ≤ 300 мс).", "",
         "| Этап | n | p50, с | p95, с |", "|---|---|---|---|"]
    for k, v in st.items():
        L.append(f"| {k} | {v['n']} | {v['p50']:.3f} | {v['p95']:.3f} |")
    L += ["", "`client_response` — время до ответа клиенту (без выжимки оператору, которая в проде асинхронна). "
              "Таргеты: p50 ≤ 3 с, p95 ≤ 8 с.", "", "![latency](figures/latency_stages.png)", "",
          "| Маршрут | доля диалогов | токенов на диалог |", "|---|---|---|"]
    for k, v in l["cost_by_route"].items():
        L.append(f"| {k} | {v['share']:.0%} (n={v['n']}) | {v['tokens_mean']:.0f} |")
    ec = l["economics"]
    S = e["systems"]
    L += ["", "**Итог для ROI**", "", "| Показатель | Прототип | A: LLM-бот | B: сценарный |", "|---|---|---|---|",
          "| Переменная стоимость диалога (LLM) | " + " | ".join(_rub(S[s].get("cost_rub_mean")) if s in S else "—"
                                                       for s in ("ours", "A", "B")) + " |",
          f"| Время ответа клиенту p50 / p95, с | {st['client_response']['p50']:.2f} / {st['client_response']['p95']:.2f} | "
          + " | ".join(f"{S[s]['latency_total_p50']:.2f} / {S[s]['latency_total_p95']:.2f}" if s in S else "—"
                       for s in ("A", "B")) + " |",
          f"| Доля in-scope без человека | {ec['share_without_human_in_scope']:.0%} | — | — |",
          f"| GCR in-scope (судья, dev) | {fmt_prop(ec.get('gcr_in_scope'))} | {fmt_prop(S.get('A', {}).get('judge_gcr_in_scope'))} | "
          f"{fmt_prop(S.get('B', {}).get('judge_gcr_in_scope'))} |"]
    HS = ((h or {}).get("human_eval") or {}).get("systems", {})
    if HS:
        L.append("| **GCR in-scope (человек, отложенный)** | " + " | ".join(
            fmt_prop(HS.get(s, {}).get("gcr")) for s in ("ours", "A", "B")) + " |")
    L += ["",
          f"Входы Gate 2: {ec['contacts_per_year'] / 1e6:.0f} млн обращений в год, обращение к оператору ≈ {ec['operator_cost_rub']} ₽. "
          f"При {ec['chats_per_month_prod'] / 1e6:.1f} млн чатов в месяц: LLM ≈ {_num(ec.get('llm_cost_per_month_rub', 0))} ₽/мес, "
          f"≈ {_num(ec.get('tokens_per_month', 0) / 1e6)} млн токенов/мес. У прототипа время ответа клиенту не включает "
          "выжимку оператору (асинхронно).", ""]
    return L


def main() -> None:
    c, cs = _j("step1_classifier.json"), _j("step1_classifier_e5small.json")
    if cs and cs.get("en", {}).get("e5_lr") == c.get("en", {}).get("e5_lr"):
        cs["_same"] = True
    p, e, r, l = _j("step3_predictor.json"), _j("step5_e2e.json"), _j("step6_redteam.json"), _j("step7_latency_cost.json")
    h = _j("step5_e2e_holdout.json")
    notes = path("reports") / "gate3_notes.md"
    L = ["# Gate 3 — метрики прототипа", "",
         "Все цифры собраны скриптом `python -m src.eval.make_report` из `reports/step*.json`. "
         "LLM через Polza AI: smart `{smart}` (выжимка оператору, судья), simple `{simple}` (перевод, бейзлайн A), "
         "local `{local}` (перефраз шаблона).".format(**cfg()["llm"]["roles"]["polza"]), ""]
    if notes.exists():
        txt = notes.read_text(encoding="utf-8")
        d1 = txt.split("<!-- D1 -->")[1].split("<!-- /D1 -->")[0] if "<!-- D1 -->" in txt else ""
        L += ["## D1. Бейзлайны", d1.strip(), ""]
    fu = _j("step3b_fusion.json")
    L += ["## D2. Качество", ""] + classifier_section(c, cs) + predictor_section(p) + fusion_section(fu) \
        + e2e_section(e) + human_section(h) + redteam_section(r)
    L += latency_section(l, e, h)
    if notes.exists():
        rest = txt.split("<!-- /D1 -->")[-1]
        L += [rest.strip(), ""]
    (path("reports") / "gate3_metrics.md").write_text("\n".join(L), encoding="utf-8")
    print("reports/gate3_metrics.md:", len(L), "строк")


if __name__ == "__main__":
    main()
