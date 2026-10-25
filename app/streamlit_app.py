"""Демо Gate 3: чат + «Под капотом» + вкладка «Оператор».

uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.config import fmt_rub, path
from src.data.synth_dwh import SIM_NOW
from src.eval.baselines import baseline_a, baseline_b
from src.pipeline import dwh, templates
from src.pipeline.assistant import handle, suggest

st.set_page_config(page_title="Ассистент банка — Gate 3", page_icon="💬", layout="wide")

st.markdown("""
<style>
.phone {max-width: 440px; border: 1px solid rgba(128,128,128,.35); border-radius: 28px; padding: 14px 14px 6px;}
.bubble {padding: 9px 13px; border-radius: 16px; margin: 6px 0; line-height: 1.4; font-size: 0.93rem;
         max-width: 88%; white-space: pre-wrap;}
.user {background: #2563eb; color: white; margin-left: auto; border-bottom-right-radius: 4px;}
.bot {background: rgba(128,128,128,.15); border-bottom-left-radius: 4px;}
.row {display: flex;}
.meta {font-size: 0.75rem; opacity: .65; margin: -2px 0 6px 4px;}
.tag {display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: .75rem; margin-right: 4px;
      background: rgba(37,99,235,.12);}
.esc {background: rgba(220,38,38,.15);}
</style>""", unsafe_allow_html=True)

SYSTEMS = {"Наш пайплайн": "ours", "Бейзлайн A (LLM-бот без данных)": "A", "Бейзлайн B (сценарный бот)": "B"}


@st.cache_data
def story_clients() -> list[dict]:
    return dwh.q("SELECT client_id, full_name, story FROM clients WHERE is_story=1 ORDER BY client_id")


@st.cache_resource
def warmup():
    handle("C90001", "проверка", paraphrase=False)  # загрузка эмбеддера и моделей
    return True


def reset():
    st.session_state.history = []
    st.session_state.last = None
    st.session_state.suggest = None


# ── Сайдбар ──
with st.sidebar:
    st.header("Демо Gate 3")
    stories = story_clients()
    labels = [f"{s['client_id']} — {s['story']}" for s in stories] + ["Другой клиент (ввести ID)"]
    choice = st.selectbox("Клиент", labels, on_change=reset, key="client_choice")
    if choice.startswith("Другой"):
        client_id = st.text_input("ID клиента", "C00042", on_change=reset)
    else:
        client_id = choice.split(" — ")[0]
    system = SYSTEMS[st.radio("Система", list(SYSTEMS), key="system")]
    paraphrase = st.toggle("Перефраз LLM для раздражённых клиентов", True)
    use_cache = st.toggle("Кэш LLM", True, help="Выключите, чтобы показать реальную задержку")
    if st.button("Новый диалог"):
        reset()
    st.caption(f"Время мира: {SIM_NOW:%d.%m.%Y %H:%M}. Данные синтетические.")

if "history" not in st.session_state:
    reset()
with st.spinner("Загружаю модели…"):
    warmup()

client = dwh.client(client_id)
if client is None:
    st.error(f"Клиент {client_id} не найден")
    st.stop()


def send(text: str, forced_intent: str | None = None):
    hist = st.session_state.history
    if system == "ours":
        r = handle(client_id, text, hist, paraphrase=paraphrase, with_summary=True,
                   use_cache=use_cache, forced_intent=forced_intent).to_dict()
    elif system == "A":
        r = baseline_a(text, hist, use_cache=use_cache)
    else:
        r = baseline_b(text, hist)
    r["system"] = system
    hist.append({"role": "user", "content": text})
    hist.append({"role": "assistant", "content": r["text"], "meta": {"options": r.get("options") or []}, "resp": r})
    st.session_state.last = r


tab_chat, tab_op, tab_dwh, tab_res = st.tabs(["💬 Чат", "🎧 Оператор", "🗄️ Данные клиента (DWH)", "📊 Результаты"])

with tab_chat:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown(f"**{client['full_name']}** · {client_id}")
        # Подсказки при открытии чата
        if st.session_state.suggest is None:
            st.session_state.suggest = suggest(client_id)
        sg = st.session_state.suggest
        if not st.session_state.history:
            st.caption(f"Подсказки до первого сообщения · {sg['latency_s'] * 1000:.0f} мс")
            cols = st.columns(len(sg["suggestions"]))
            for col, s in zip(cols, sg["suggestions"]):
                if col.button(s["title"], key=f"sg_{s['intent']}", width="stretch",
                              help=f"p = {s['p']:.2f}"):
                    send(s["title"], forced_intent=s["intent"] if system == "ours" else None)
                    st.rerun()

        html = ['<div class="phone">']
        for h in st.session_state.history:
            cls = "user" if h["role"] == "user" else "bot"
            html.append(f'<div class="row"><div class="bubble {cls}">{h["content"]}</div></div>')
            if h["role"] == "assistant":
                r = h["resp"]
                tags = [f'<span class="tag">{r.get("route", "")}</span>']
                if r.get("escalate"):
                    tags.append('<span class="tag esc">передано человеку</span>')
                if r.get("followup"):
                    tags.append(f'<span class="tag">🕑 проверка {r["followup"]["at"][11:16]}</span>')
                html.append(f'<div class="meta">{"".join(tags)}</div>')
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)

        last = st.session_state.last
        if last and system == "ours":
            if last.get("options"):
                st.caption("Варианты:")
                for o in last["options"]:
                    if st.button(templates.title(o), key=f"opt_{o}_{len(st.session_state.history)}"):
                        send(templates.title(o), forced_intent=o)
                        st.rerun()
            for a in last.get("actions") or []:
                if st.button(f"▶ {a}", key=f"act_{a}_{len(st.session_state.history)}"):
                    st.toast(f"«{a}» — выполнено (имитация в прототипе)")
            if last.get("followup"):
                st.info(f"Запланирована проверка статуса {last['followup']['check']} в "
                        f"{last['followup']['at'][11:16]}: если перевод не зачислится — ассистент напишет сам.")

        msg = st.chat_input("Напишите сообщение…")
        if msg:
            send(msg)
            st.rerun()

    with right:
        st.subheader("Под капотом")
        r = st.session_state.last
        if not r:
            st.caption("Отправьте сообщение или нажмите подсказку.")
            with st.expander("Как считаются подсказки"):
                st.write("LightGBM по признакам DWH на момент открытия чата: свежие отказы, переводы в пути, "
                         "холды, экраны приложения за 30 минут. Топ-3 интента:")
                st.dataframe(pd.DataFrame(sg["suggestions"]), hide_index=True)
        elif r["system"] != "ours":
            st.write(f"**Система:** {r['system']} — без данных клиента и без шаблонов.")
            st.write("**Маршрут:**", r.get("route"), "· **эскалация:**", r.get("escalate"))
            if r.get("masked_text"):
                st.write("**Ушло в LLM (маскировано):**", r["masked_text"])
            st.write("**Латентность:**", r["latency"])
            st.write(f"**Токены:** {r['tokens']} · **стоимость:** {fmt_rub(r.get('cost_rub'))}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Интент", templates.title(r["intent"]) if r["intent"] in templates.SCOPE + templates.ACUTE
                      else (r["intent"] or "—"), f"уверенность {r['confidence']:.2f}", delta_color="off")
            c2.metric("Раздражение", f"{r['emotion']:.2f}")
            c3.metric("Маршрут", r["route"], r["route_reason"], delta_color="off")
            st.write("**Маскированный текст (только он уходит в LLM):**")
            st.code(r["masked_text"], language=None)
            if r["pii_found"]:
                st.caption("Найдены ПДн: " + ", ".join(r["pii_found"]))
            st.write("**Топ-3 интента классификатора:**",
                     ", ".join(f"{i} ({p:.2f})" for i, p in r["top_intents"]))
            if r["facts"]:
                st.write(f"**Факты из DWH** (вариант шаблона: `{r['template_variant']}`):")
                st.dataframe(pd.DataFrame(list(r["facts"].items()), columns=["поле", "значение"]),
                             hide_index=True, width="stretch")
            if r["template_text"]:
                st.write("**Шаблон с плейсхолдерами:**")
                st.code(r["template_text"], language=None, wrap_lines=True)
            if r["paraphrase"]:
                st.write("**Перефраз LLM (плейсхолдеры сохранены):**")
                st.code(r["paraphrase"], language=None, wrap_lines=True)
            if r["guardrails"]:
                st.warning("Гардрейлы сработали: " + "; ".join(r["guardrails"]))
            else:
                st.success("Гардрейлы: нарушений нет")
            lat = {k: v for k, v in r["latency"].items() if k != "total"}
            st.write(f"**Латентность:** всего {r['latency']['total']:.2f} с"
                     + (f" (из них выжимка оператору {lat.get('llm_summary', 0):.2f} с — в проде асинхронно)"
                        if lat.get("llm_summary") else ""))
            st.bar_chart(pd.Series(lat, name="секунды"), horizontal=True, height=220)
            st.write(f"**Токены:** вход {r['tokens'].get('prompt', 0)}, выход {r['tokens'].get('completion', 0)}"
                     f" · **стоимость:** {fmt_rub(r.get('cost_rub'))}")
            for c in r["llm_calls"]:
                st.caption(f"{c['role']} · {c['model']} · {c['prompt_tokens']}+{c['completion_tokens']} ток. · "
                           f"{c['latency_s']} с · {'кэш' if c['cached'] else 'живой вызов'}"
                           + (f" · ошибка: {c['error']}" if c.get("error") else ""))

with tab_op:
    esc = [h["resp"] for h in st.session_state.history if h["role"] == "assistant" and h["resp"].get("escalate")]
    if not esc:
        st.caption("Эскалаций в этом диалоге нет. Попробуйте: «Мне звонили из службы безопасности, просят код».")
    else:
        r = esc[-1]
        st.error(f"Эскалация: {r.get('route')} — {r.get('route_reason', '')}")
        if r.get("operator_summary"):
            st.markdown("**Выжимка и черновик ответа (smart-LLM по маскированному диалогу):**")
            st.text(r["operator_summary"])
        else:
            st.caption("Выжимка строится только для нашего пайплайна.")
        st.markdown("**Диалог:**")
        for h in st.session_state.history:
            st.write(("🧑 " if h["role"] == "user" else "🤖 ") + h["content"])

with tab_dwh:
    st.caption("Что видит ассистент о клиенте (синтетика).")
    st.write("**Карты**")
    st.dataframe(pd.DataFrame(dwh.q("SELECT last4, status, daily_limit, spent_today, delivery_status, "
                                    "delivery_date FROM cards WHERE client_id=?", client_id)), hide_index=True)
    st.write("**Последние операции**")
    st.dataframe(pd.DataFrame(dwh.q("SELECT ts, amount, merchant, descriptor, status, decline_reason, fee "
                                    "FROM transactions WHERE client_id=? ORDER BY ts DESC LIMIT 15", client_id)),
                 hide_index=True)
    st.write("**Переводы**")
    st.dataframe(pd.DataFrame(dwh.q("SELECT ts, amount, recipient, method, status, expected_at FROM transfers "
                                    "WHERE client_id=? ORDER BY ts DESC LIMIT 10", client_id)), hide_index=True)
    st.write("**Экраны приложения за 30 минут**")
    st.dataframe(pd.DataFrame(dwh.q("SELECT ts, screen FROM app_events WHERE client_id=? ORDER BY ts DESC",
                                    client_id)), hide_index=True)


@st.cache_data
def _report(name: str) -> dict:
    p = path("reports") / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


with tab_res:
    # Итоговые цифры из reports/*.json (make report) — чтобы в демо не открывать файлы
    hold, lat = _report("step5_e2e_holdout.json"), _report("step7_latency_cost.json")
    he, dev = hold.get("human_eval") or {}, _report("step5_e2e.json")
    if not he:
        st.info("Нет человеческой разметки: запустите `make label-holdout`, затем `make report`.")
    else:
        HS, pr, k = he["systems"], he.get("paired", {}), he.get("kappa", {})
        pct = lambda d: f"{d['p']:.0%}" if d and d.get("n") else "—"
        ci = lambda d: f"95% ДИ {d['ci95'][0]:.0%}–{d['ci95'][1]:.0%}, n = {d['n']}"
        st.subheader("Решает ли ассистент вопрос клиента (GCR)")
        st.caption(f"Слепая человеческая разметка: {he['n_cases']} реальных формулировок клиентов × 3 системы, "
                   "система скрыта, порядок перемешан. Отложенный набор — в разработке не участвовал.")
        names = {"ours": "Прототип", "A": "LLM-бот без данных", "B": "Сценарный бот"}
        for col, s in zip(st.columns(3), ("ours", "A", "B")):
            col.metric(names[s], pct(HS[s]["gcr"]), ci(HS[s]["gcr"]), delta_color="off")
        chart = pd.DataFrame({"с данными о клиенте": [HS[s]["gcr_signal"]["p"] * 100 for s in names],
                              "без данных о клиенте": [HS[s]["gcr_no_signal"]["p"] * 100 for s in names]},
                             index=list(names.values()))
        st.bar_chart(chart, stack=False, y_label="GCR, %", height=260)
        st.markdown(
            f"- Против сценарного бота прирост **значим** (парный тест, p = {pr['B']['p_mcnemar']:.2f}); "
            f"против LLM-бота — **пока нет** (p = {pr['A']['p_mcnemar']:.2f}).\n"
            f"- Вся разница — у клиентов с данными: {pct(HS['ours']['gcr_signal'])} против "
            f"{pct(HS['A']['gcr_signal'])} у LLM-бота. Без данных — {pct(HS['ours']['gcr_no_signal'])} против "
            f"{pct(HS['A']['gcr_no_signal'])}.\n"
            f"- LLM-судья согласен с человеком слабо (κ = {k.get('three_class', 0):.2f}), поэтому GCR — по человеку.")
        st.subheader("Безопасность, скорость, деньги")
        S = hold["systems"]
        m = st.columns(5)
        m[0].metric("Ошибки в цифрах", pct(S["ours"]["auto_critical_fact_errors"]),
                    f"n = {S['ours']['auto_critical_fact_errors']['n']}", delta_color="off")
        m[1].metric("Выдуманные сроки", pct(S["ours"]["auto_unverified_terms"]),
                    f"LLM-бот: {pct(S['A']['auto_unverified_terms'])}", delta_color="off")
        m[2].metric("Острые → человеку", pct(S["ours"]["escalation_recall"]),
                    f"n = {S['ours']['escalation_recall']['n']}", delta_color="off")
        cr = (lat.get("stages") or {}).get("client_response", {})
        m[3].metric("Ответ клиенту, p50", f"{cr.get('p50', 0):.2f} с", f"p95 {cr.get('p95', 0):.2f} с",
                    delta_color="off")
        m[4].metric("Стоимость диалога", fmt_rub(dev.get("systems", {}).get("ours", {}).get("cost_rub_mean")),
                    "оператор ≈ 100 ₽", delta_color="off")
