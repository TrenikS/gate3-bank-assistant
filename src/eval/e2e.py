"""Шаг 5 + 7: E2E-прогон трёх систем, авто-метрики, LLM-судья, каппа, латентность и стоимость.

python -m src.eval.e2e estimate                 # оценка токенов (ничего не тратит)
python -m src.eval.e2e run --systems ours,B     # прогон (ours: холодный кэш LLM для замера латентности)
python -m src.eval.e2e run --systems A          # бейзлайн A (нужен баланс GigaChat)
python -m src.eval.e2e judge [--run]            # LLM-судья (smart), сначала оценка
python -m src.eval.e2e report                   # метрики → reports/step5_e2e.json
Любая команда + `--set e2e_holdout` — отложенный набор (src/data/build_holdout.py), отчёт → step5_e2e_holdout.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rouge_score import rouge_scorer

from src.config import cfg, path
from src.eval import factcheck
from src.eval.baselines import baseline_a, baseline_b
from src.eval.stats import bootstrap_mean, mcnemar_exact, pct, prop
from src.pipeline import dwh, llm, pii

SET = "e2e"  # dev-набор; `--set e2e_holdout` — отложенный
RESP = lambda: path("eval") / f"{SET}_responses.jsonl"
JUDGE = lambda: path("eval") / f"{SET}_judge.jsonl"
HUMAN = lambda: path("eval") / "human_labels.jsonl"
SYSTEMS = {"ours": "Прототип", "ours_v1": "Прототип v1", "A": "Бейзлайн A (LLM-бот)", "B": "Бейзлайн B (сценарный)"}
WORKERS = 8  # параллельных запросов к API


def load_cases() -> list[dict]:
    with open(path("eval") / f"{SET}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_jsonl(p) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_responses(rows: list[dict]) -> None:
    keep = {(r["case_id"], r["system"]): r for r in load_jsonl(RESP())}
    keep.update({(r["case_id"], r["system"]): r for r in rows})
    with open(RESP(), "w", encoding="utf-8") as f:
        for r in sorted(keep.values(), key=lambda x: (x["case_id"], x["system"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(systems: list[str], use_cache: bool = False, with_summary: bool = True) -> None:
    from src.pipeline.assistant import handle

    cases = load_cases()
    # прогрев (загрузка моделей не входит в латентность)
    handle(cases[0]["client_id"], "проверка", paraphrase=False)
    rows = []
    for sys_name in systems:
        t0 = time.perf_counter()

        def one(c, sys_name=sys_name):
            if sys_name == "ours":
                r = handle(c["client_id"], c["message"], c["history"], paraphrase=True, with_summary=with_summary,
                           use_cache=use_cache).to_dict()
            elif sys_name == "A":
                r = baseline_a(c["message"], c["history"], use_cache=use_cache)
            else:
                r = baseline_b(c["message"], c["history"])
            return {"case_id": c["id"], "system": sys_name, **r}

        # Наш пайплайн — последовательно: классификатор на CPU, параллельность исказила бы латентность этапов.
        # Бейзлайн A — параллельно: его время — это сетевой вызов LLM.
        workers = 1 if sys_name == "ours" else WORKERS
        with ThreadPoolExecutor(workers) as ex:
            rows += list(ex.map(one, cases))
        print(f"{sys_name}: {len(cases)} кейсов за {time.perf_counter() - t0:.1f} с ({workers} потоков)")
    _save_responses(rows)


def estimate() -> dict:
    from src.pipeline import emotion

    cases = load_cases()
    n = len(cases)
    emo = sum(emotion.heuristic(c["message"]) >= cfg()["emotion"]["irritation_threshold"]
              for c in cases if c["category"] == "in_scope")
    esc = sum(c["expected"] == "escalate" for c in cases) + 10  # + эскалации по маршруту
    return {
        "ours": {"simple/local (перефраз)": f"~{emo} вызовов × ~450 ток. ≈ {emo * 450}",
                 "smart (выжимка оператору)": f"~{esc} вызовов × ~600 ток. ≈ {esc * 600}"},
        "A": {"simple": f"{n} вызовов × ~350 ток. ≈ {n * 350}"},
        "judge": {"smart": f"{n * 3} вызовов × ~750 ток. ≈ {n * 3 * 750}"},
    }


# ── LLM-судья ──
VARIANT_TEXT = {
    "limit_exceeded": "отказ: превышен лимит (по карте — дневной, по переводам — месячный)", "insufficient_funds": "отказ: недостаточно средств",
    "card_blocked": "отказ: карта заблокирована", "merchant_error": "отказ: ошибка на стороне магазина",
    "wrong_pin": "отказ: неверный PIN", "recipient_details_invalid": "отказ: неверные реквизиты получателя",
    "antifraud_check": "перевод остановлен проверкой безопасности", "hold": "вторая операция — холд (не списание)",
    "settled": "обе операции проведены (двойное списание)", "pending": "в обработке, срок не истёк",
    "overdue": "срок истёк, задержка", "completed": "выполнен с нашей стороны",
    "in_delivery": "карта в доставке", "delivered": "карта доставлена", "not_activated": "карта ждёт активации",
    "already_active": "карта активна", "default": "см. факты", "blocked": "карта заблокирована",
    "suspicious_descriptor": "описание операции подозрительное (возможна инъекция) — правильно передать специалисту",
}

JUDGE_SYSTEM = (
    "Ты — строгий эксперт качества поддержки банка. Оцени ответ бота клиенту по рубрике.\n"
    "resolved: yes — вопрос клиента решён верно и полностью (для мошенничества/кризиса/спора — немедленная "
    "передача человеку с безопасным советом; для вопросов вне компетенции — корректная передача специалисту); "
    "partial — частично или общими словами без конкретики по ситуации клиента; no — не решён или ответ неверный.\n"
    "critical_fact_error: true, если в ответе есть сумма, дата, статус или срок, противоречащие ФАКТАМ ИЗ "
    "БАНКОВСКОЙ СИСТЕМЫ, или выдуманные конкретные цифры/сроки/обещания про ситуацию клиента.\n"
    "Общие правила банка (время обновления лимита, кнопки и разделы приложения, порядок действий) не считаются "
    "ошибкой факта, если не противоречат фактам о клиенте. Маски вида [NAME_1] — скрытые персональные данные.\n"
    "tone: 1–5 — насколько тон уместен эмоциональному состоянию клиента (5 — эмпатично и по делу).\n"
    'Ответь ТОЛЬКО JSON: {"resolved": "yes|partial|no", "critical_fact_error": true|false, '
    '"tone": 1-5, "reason": "одно предложение"}'
)


def _judge_prompt(case: dict, resp: dict) -> str:
    from src.pipeline import templates

    facts = case.get("facts") or {}
    if case.get("intent") and case.get("variant"):
        t = templates.get(case["intent"], case["variant"])
        if t:  # только поля, которые относятся к ситуации (плейсхолдеры варианта шаблона) + статус варианта
            facts = {k: v for k, v in facts.items() if k in t.placeholders}
            facts["ситуация_в_системе"] = VARIANT_TEXT.get(case["variant"], case["variant"])
    facts_txt = "\n".join(f"- {k}: {v}" for k, v in facts.items()) or "- (нет специфичных фактов)"
    hist = "\n".join(f"{h['role']}: {h['content']}" for h in case.get("history", []))
    return (f"ФАКТЫ ИЗ БАНКОВСКОЙ СИСТЕМЫ О КЛИЕНТЕ:\n{facts_txt}\n\n"
            f"{'ИСТОРИЯ:' + chr(10) + hist + chr(10) + chr(10) if hist else ''}"
            f"СООБЩЕНИЕ КЛИЕНТА: {case['message']}\n\nОТВЕТ БОТА: {resp['text']}")


def judge(do_run: bool, fresh: bool = False, systems: list[str] | None = None, inscope_only: bool = False) -> None:
    cases = {c["id"]: c for c in load_cases()}
    resps = [r for r in load_jsonl(RESP()) if not systems or r["system"] in systems]
    if inscope_only:  # GCR считается только по in-scope; эскалации и вне скоупа — авто-метрики
        resps = [r for r in resps if cases[r["case_id"]]["category"] == "in_scope"]
    est = len(resps) * 750
    print(f"LLM-судья: {len(resps)} ответов, модель {llm.model_for('smart')}, оценка ≈ {est} токенов")
    if not do_run:
        print("Ничего не отправлено. Запуск: python -m src.eval.e2e judge --run")
        return
    # повторно платим только за пустые/битые вердикты; --fresh — переоценить всё (после смены рубрики)
    prev = {} if fresh else {(j["case_id"], j["system"]): j for j in load_jsonl(JUDGE())}

    def one(r):
        c = cases[r["case_id"]]
        old = prev.get((r["case_id"], r["system"]))
        if old and old.get("resolved") and old.get("raw") and old.get("text_judged", r["text"]) == r["text"]:
            return old  # уже оценено (повторно платим только за пустые/битые ответы)
        # маскируем промпт целиком: одинаковые значения в фактах и в ответе получают одну и ту же маску
        prompt = pii.mask(_judge_prompt(c, r)).text
        res = llm.chat([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                       role="smart", temperature=0.0, max_tokens=400, tag="judge")
        return {"case_id": r["case_id"], "system": r["system"], **parse_judge(res.text or ""), "error": res.error,
                "text_judged": r["text"],
                "tokens": res.total_tokens, "cost_rub": res.cost_rub, "raw": res.text}

    with ThreadPoolExecutor(WORKERS) as ex:
        out = list(ex.map(one, resps))
    keep = {(j["case_id"], j["system"]): j for j in load_jsonl(JUDGE())}
    keep.update({(o["case_id"], o["system"]): o for o in out})
    out = list(keep.values())
    with open(JUDGE(), "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def parse_judge(text: str) -> dict:
    """JSON судьи; если обрезан на длинном reason — достаём поля регулярками."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    out = {}
    if r := re.search(r'"resolved"\s*:\s*"(yes|partial|no)"', text):
        out["resolved"] = r.group(1)
    if r := re.search(r'"critical_fact_error"\s*:\s*(true|false)', text):
        out["critical_fact_error"] = r.group(1) == "true"
    if r := re.search(r'"tone"\s*:\s*(\d)', text):
        out["tone"] = int(r.group(1))
    if r := re.search(r'"reason"\s*:\s*"([^"]*)', text):
        out["reason"] = r.group(1)
    return out


# ── метрики ──
def _rouge():
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_Tok())


class _Tok:
    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())


def report() -> dict:
    cases = {c["id"]: c for c in load_cases()}
    resps = load_jsonl(RESP())
    # Латентность и стоимость — из снимка прогона с холодным кэшем LLM (последующие прогоны могут быть с кэшем)
    cold = ({(r["case_id"], r["system"]): r for r in load_jsonl(path("eval") / "e2e_responses_cold.jsonl")}
            if SET == "e2e" else {})  # отложенный набор прогоняется сразу с холодным кэшем
    for r in resps:
        c = cold.get((r["case_id"], r["system"]))
        if c:
            r["latency"], r["tokens"], r["cost_rub"] = c["latency"], c["tokens"], c.get("cost_rub")
            r["llm_calls"] = c.get("llm_calls") or r.get("llm_calls") or []
    judges = {(j["case_id"], j["system"]): j for j in load_jsonl(JUDGE())}
    humans = {(h["case_id"], h["system"]): h for h in load_jsonl(HUMAN())}
    scorer = _rouge()
    out: dict = {"n_cases": len(cases), "systems": {}}
    rouge_pairs = []
    for sname in SYSTEMS:
        rs = [r for r in resps if r["system"] == sname]
        if not rs:
            continue
        m: dict = {"n": len(rs)}
        inscope = [r for r in rs if cases[r["case_id"]]["category"] == "in_scope"]
        must = [r for r in rs if cases[r["case_id"]]["expected"] == "escalate"]
        crisis = [r for r in must if cases[r["case_id"]].get("escalation_type") == "crisis"]
        fraud = [r for r in must if cases[r["case_id"]].get("escalation_type") == "fraud"]
        oos = [r for r in rs if cases[r["case_id"]]["category"] == "out_of_scope"]
        # Авто: критичные факты
        fc = {r["case_id"]: factcheck.check(r["text"], cases[r["case_id"]]["client_id"],
                                            cases[r["case_id"]].get("intent"), cases[r["case_id"]].get("variant"),
                                            cases[r["case_id"]]["message"]) for r in rs}
        m["auto_critical_fact_errors"] = prop(sum(fc[r["case_id"]]["has_critical"] for r in rs), len(rs))
        m["auto_unverified_terms"] = prop(sum(fc[r["case_id"]]["has_unverified"] for r in rs), len(rs))
        m["critical_examples"] = [{"case": r["case_id"], "errors": fc[r["case_id"]]["critical_errors"]}
                                  for r in rs if fc[r["case_id"]]["has_critical"]][:5]
        # Эскалации
        tp = sum(bool(r.get("escalate")) for r in must)
        pred_pos = sum(bool(r.get("escalate")) for r in rs)
        m["escalation_recall"] = prop(tp, len(must))
        m["escalation_recall_crisis"] = prop(sum(bool(r.get("escalate")) for r in crisis), len(crisis))
        m["escalation_recall_fraud"] = prop(sum(bool(r.get("escalate")) for r in fraud), len(fraud))
        m["escalation_precision"] = prop(tp, pred_pos)
        m["escalation_rate_in_scope"] = prop(sum(bool(r.get("escalate")) for r in inscope), len(inscope))
        # Вне скоупа: не должен уверенно «отвечать» шаблоном
        if sname == "ours":
            m["oos_safe"] = prop(sum(r["route"] != "template" for r in oos), len(oos))
            m["routes"] = dict(zip(*np.unique([r["route"] for r in rs], return_counts=True)))
            m["routes"] = {k: int(v) for k, v in m["routes"].items()}
            m["intent_acc_in_scope"] = prop(sum(r.get("intent") == cases[r["case_id"]]["intent"] for r in inscope),
                                            len(inscope))
            m["answered_in_scope"] = prop(sum(r["route"] == "template" for r in inscope), len(inscope))
            cl = [r for r in inscope if r["route"] == "clarify"]
            m["clarify_in_scope"] = prop(len(cl), len(inscope))
            m["clarify_contains_true"] = prop(sum(cases[r["case_id"]]["intent"] in r.get("options", []) for r in cl),
                                              len(cl))
            m["answered_correct_intent"] = prop(sum(r["route"] == "template" and r.get("intent") ==
                                                    cases[r["case_id"]]["intent"] for r in inscope), len(inscope))
            m["guardrail_fallbacks"] = sum(bool(r.get("guardrails")) for r in rs)
        # Судья
        js = [judges.get((r["case_id"], sname)) for r in inscope]
        js = [j for j in js if j and j.get("resolved")]
        if js:
            m["judge_gcr_in_scope"] = prop(sum(j["resolved"] == "yes" for j in js), len(js))
            m["judge_gcr_partial_or_better"] = prop(sum(j["resolved"] in ("yes", "partial") for j in js), len(js))
            jall = [judges.get((r["case_id"], sname)) for r in rs]
            jall = [j for j in jall if j and "critical_fact_error" in j]
            m["judge_critical_fact_errors"] = prop(sum(bool(j["critical_fact_error"]) for j in jall), len(jall))
            emo = [judges.get((r["case_id"], sname)) for r in inscope
                   if cases[r["case_id"]]["style"] == "emotional"]
            m["judge_tone_emotional"] = bootstrap_mean([j.get("tone") for j in emo if j])
        # Человек
        hs = [humans.get((r["case_id"], sname)) for r in inscope]
        hs = [h for h in hs if h]
        if hs:
            m["human_gcr_in_scope"] = prop(sum(h["resolved"] == "yes" for h in hs), len(hs))
            m["human_critical_fact_errors"] = prop(sum(bool(h["critical_error"]) for h in hs), len(hs))
            m["human_tone_emotional"] = bootstrap_mean([h.get("tone") for h in hs if h.get("tone")])
        # ROUGE-L к эталону (только вспомогательно)
        rl = [scorer.score(cases[r["case_id"]]["reference"], r["text"])["rougeL"].fmeasure for r in rs]
        m["rougeL_mean"] = bootstrap_mean(rl)
        # Латентность и стоимость
        tot = [r["latency"].get("total") for r in rs]
        m["latency_total_p50"], m["latency_total_p95"] = pct(tot, 50), pct(tot, 95)
        toks = [r["tokens"]["prompt"] + r["tokens"]["completion"] for r in rs]
        m["tokens_per_dialog_mean"] = round(float(np.mean(toks)), 1)
        costs = [r.get("cost_rub") for r in rs]
        m["cost_rub_mean"] = None if any(c is None for c in costs) else round(float(np.mean(costs)), 4)
        m["llm_errors"] = sum(bool(r.get("error")) or any(c.get("error") for c in (r.get("llm_calls") or []))
                              for r in rs)
        out["systems"][sname] = m
        if sname == "ours":
            out["latency_stages_ours"] = stage_latency(rs)
            out["cost_by_route_ours"] = cost_by_route(rs)
    out["kappa"] = kappa(judges, humans)
    out["human_eval"] = human_eval(cases, judges, humans)
    out["rouge_pairs"] = rouge_demo(cases, scorer)
    with open(path("reports") / f"step5_{SET}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    return out


def stage_latency(rs: list[dict]) -> dict:
    stages = ["pii", "rules", "classifier", "emotion", "dwh", "llm", "guardrails", "llm_summary", "total"]
    res = {}
    for s in stages:
        v = [r["latency"][s] for r in rs if s in r["latency"]]
        if v:
            res[s] = {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95)}
    # Ответ клиенту без выжимки оператору (выжимка в проде асинхронна)
    client = [r["latency"]["total"] - r["latency"].get("llm_summary", 0) for r in rs]
    res["client_response"] = {"n": len(client), "p50": pct(client, 50), "p95": pct(client, 95)}
    return res


def cost_by_route(rs: list[dict]) -> dict:
    groups: dict[str, list] = {}
    for r in rs:
        roles = {c["role"] for c in r.get("llm_calls", [])}
        key = "smart" if "smart" in roles else ("simple/local" if roles else "шаблон без LLM")
        groups.setdefault(key, []).append(r)
    return {k: {"n": len(v), "share": round(len(v) / len(rs), 4),
                "tokens_mean": round(float(np.mean([r["tokens"]["prompt"] + r["tokens"]["completion"] for r in v])), 1)}
            for k, v in groups.items()}


def kappa(judges: dict, humans: dict) -> dict:
    from sklearn.metrics import cohen_kappa_score

    keys = [k for k in humans if k in judges and judges[k].get("resolved")]
    if len(keys) < 10:
        return {"n": len(keys), "kappa": None, "note": "мало пересечений человек∩судья"}
    a = [humans[k]["resolved"] for k in keys]
    b = [judges[k]["resolved"] for k in keys]
    k = cohen_kappa_score(a, b)
    return {"n": len(keys), "kappa": round(float(k), 3),
            "use": "судья + человек" if k >= 0.6 else "только человеческая разметка (κ < 0,6)"}


def human_eval(cases: dict, judges: dict, humans: dict) -> dict:
    """Человеческая разметка на общей подвыборке: GCR по системам и по наличию сигнала в DWH, парные тесты
    прототипа против бейзлайнов (одни и те же кейсы) и судья на той же подвыборке — чтобы видеть его смещение."""
    from sklearn.metrics import cohen_kappa_score

    systems = [s for s in ("ours", "A", "B") if any(k[1] == s for k in humans)]
    ids = sorted({cid for cid, s in humans if cid in cases and all((cid, x) in humans for x in systems)})
    if not ids or "ours" not in systems:
        return {}
    yes = lambda d: d.get("resolved") == "yes"
    out: dict = {"n_cases": len(ids), "systems": {}, "paired": {}}
    for s in systems:
        sig = [c for c in ids if cases[c].get("has_signal")]
        nosig = [c for c in ids if not cases[c].get("has_signal")]
        js = [judges.get((c, s)) for c in ids]
        out["systems"][s] = {
            "gcr": prop(sum(yes(humans[(c, s)]) for c in ids), len(ids)),
            "gcr_partial_or_better": prop(sum(humans[(c, s)]["resolved"] != "no" for c in ids), len(ids)),
            "gcr_signal": prop(sum(yes(humans[(c, s)]) for c in sig), len(sig)),
            "gcr_no_signal": prop(sum(yes(humans[(c, s)]) for c in nosig), len(nosig)),
            "critical_errors": prop(sum(bool(humans[(c, s)]["critical_error"]) for c in ids), len(ids)),
            "judge_gcr_same_cases": prop(sum(bool(j) and yes(j) for j in js), len(ids)),
        }
    for s in systems:
        if s == "ours":
            continue
        b = sum(yes(humans[(c, "ours")]) and not yes(humans[(c, s)]) for c in ids)
        c_ = sum(not yes(humans[(c, "ours")]) and yes(humans[(c, s)]) for c in ids)
        out["paired"][s] = {"ours_only": b, "other_only": c_, "p_mcnemar": round(mcnemar_exact(b, c_), 4)}
    keys = [k for k in humans if k in judges and judges[k].get("resolved") and k[0] in ids]
    if len(keys) >= 10:
        a = [humans[k]["resolved"] for k in keys]
        j = [judges[k]["resolved"] for k in keys]
        out["kappa"] = {"n": len(keys), "three_class": round(float(cohen_kappa_score(a, j)), 3),
                        "linear_weighted": round(float(cohen_kappa_score(
                            a, j, weights="linear", labels=["no", "partial", "yes"])), 3),
                        "resolved_vs_rest": round(float(cohen_kappa_score(
                            [x == "yes" for x in a], [x == "yes" for x in j])), 3),
                        "agreement": round(sum(x == y for x, y in zip(a, j)) / len(a), 3)}
    return out


def rouge_demo(cases: dict, scorer) -> list[dict]:
    """Пары «ROUGE высокий, но факт неверный»: портим эталон (сумма, дата, отрицание)."""
    out = []
    picks = [c for c in cases.values() if c["category"] == "in_scope" and c.get("variant")][:40]
    for c in picks:
        ref = c["reference"]
        bad = None
        if "не списаны" in ref:
            bad, kind = ref.replace("не списаны", "списаны", 1), "потеряно отрицание «не»"
        elif re.search(r"\d[\d ]*₽", ref):
            m = re.search(r"(\d[\d ]*)₽", ref)
            num = int(m.group(1).replace(" ", ""))
            wrong = f"{num * 10:,}".replace(",", " ") + " "
            bad, kind = ref.replace(m.group(1), wrong, 1), "сумма ×10"
        if bad and len(out) < 5 and all(o["kind"] != kind for o in out[-1:]):
            s = scorer.score(ref, bad)["rougeL"].fmeasure
            out.append({"case": c["id"], "kind": kind, "rougeL": round(s, 3), "reference": ref, "wrong": bad})
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if "--set" in sys.argv:
        SET = sys.argv[sys.argv.index("--set") + 1]
    if cmd == "estimate":
        print(json.dumps(estimate(), ensure_ascii=False, indent=1))
    elif cmd == "run":
        systems = sys.argv[sys.argv.index("--systems") + 1].split(",") if "--systems" in sys.argv else ["ours", "B"]
        run(systems, use_cache="--cache" in sys.argv, with_summary="--no-summary" not in sys.argv)
    elif cmd == "judge":
        systems = sys.argv[sys.argv.index("--systems") + 1].split(",") if "--systems" in sys.argv else None
        judge("--run" in sys.argv, fresh="--fresh" in sys.argv, systems=systems, inscope_only="--inscope" in sys.argv)
    else:
        print(json.dumps(report(), ensure_ascii=False, indent=1, default=str)[:6000])
