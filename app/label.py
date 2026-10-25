"""Human eval: разметка ответов трёх систем вслепую (система скрыта, порядок перемешан).

uv run streamlit run app/label.py
Разметка пишется в data/eval/human_labels.jsonl (по одной строке на (кейс, система); повторная оценка перезаписывает).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import path
from src.pipeline import templates

st.set_page_config(page_title="Разметка E2E", page_icon="🏷️", layout="centered")

# streamlit run app/label.py -- --set e2e_holdout --sample 30
# --sample N: N in-scope кейсов (поровну по интентам) × все системы — минимум для каппы судья∩человек
ARGS = sys.argv[1:]
SET = ARGS[ARGS.index("--set") + 1] if "--set" in ARGS else "e2e"
SAMPLE = int(ARGS[ARGS.index("--sample") + 1]) if "--sample" in ARGS else 0
CASES = path("eval") / f"{SET}.jsonl"
RESP = path("eval") / f"{SET}_responses.jsonl"
OUT = path("eval") / "human_labels.jsonl"


@st.cache_data
def load():
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(CASES, encoding="utf-8")}
    resps = [json.loads(l) for l in open(RESP, encoding="utf-8")]
    resps = [r for r in resps if r["system"] != "ours_v1"]
    if SAMPLE:
        rng = random.Random(11)
        by_intent: dict = {}
        for c in cases.values():
            if c["category"] == "in_scope":
                by_intent.setdefault(c["intent"], []).append(c["id"])
        picked, k = set(), 0
        pools = {i: rng.sample(v, len(v)) for i, v in sorted(by_intent.items())}
        while len(picked) < SAMPLE and any(pools.values()):
            for i in sorted(pools):
                if pools[i] and len(picked) < SAMPLE:
                    picked.add(pools[i].pop())
        resps = [r for r in resps if r["case_id"] in picked]
    items = [(r["case_id"], r["system"], r["text"]) for r in resps if r["case_id"] in cases]
    random.Random(7).shuffle(items)  # фиксированный порядок между сессиями, но системы перемешаны
    return cases, items


def labels() -> dict:
    if not OUT.exists():
        return {}
    out = {}
    for l in open(OUT, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            out[(d["case_id"], d["system"])] = d
    return out


def save(d: dict):
    cur = labels()
    cur[(d["case_id"], d["system"])] = d
    with open(OUT, "w", encoding="utf-8") as f:
        for v in cur.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")


if not RESP.exists():
    st.error(f"Нет {RESP.name} — сначала запустите python -m src.eval.e2e run --set {SET}")
    st.stop()

GUIDE = path("eval") / "labeling_guide.md"
if GUIDE.exists():
    with st.expander("Инструкция разметчику — прочитайте до начала"):
        st.markdown(GUIDE.read_text(encoding="utf-8"))

cases, items = load()
done = labels()
todo = [i for i, it in enumerate(items) if (it[0], it[1]) not in done]
st.progress(1 - len(todo) / len(items), text=f"Размечено {len(items) - len(todo)} из {len(items)}")

if "pos" not in st.session_state:
    st.session_state.pos = todo[0] if todo else 0
pos = st.session_state.pos
case_id, system, text = items[pos]
c = cases[case_id]

st.caption(f"Пункт {pos + 1} / {len(items)} · кейс {case_id} · {c['category']} · стиль: {c['style']}")
if c.get("history"):
    with st.expander("История диалога"):
        for h in c["history"]:
            st.write(("🧑 " if h["role"] == "user" else "🤖 ") + h["content"])
st.markdown(f"**Клиент:** {c['message']}")
if c.get("facts"):
    # Только поля, которые использует шаблон этого варианта: служебные заглушки (например, block_reason
    # «причина уточняется» при отказе по лимиту) сбивали разметку — так было с H004
    tpl = templates.get(c["intent"], c.get("variant")) if c.get("intent") else None
    used = set(tpl.placeholders) if tpl and tpl.variant else set(c["facts"])
    with st.expander("Факты о клиенте в DWH (правда)", expanded=True):
        if c.get("variant"):
            st.markdown(f"Ситуация в DWH: **{c['variant']}**")
        st.json({k: v for k, v in c["facts"].items() if k in used})
elif c["category"] != "in_scope":
    st.info(f"Ожидаемое поведение: **{c['expected']}**"
            + (f" ({c.get('escalation_type')})" if c.get("escalation_type") else ""))
st.markdown("**Ответ системы:**")
st.success(text)

prev = done.get((case_id, system), {})
with st.form(f"f_{pos}"):
    resolved = st.radio("Решён ли вопрос?", ["yes", "partial", "no"], horizontal=True,
                        format_func={"yes": "решено", "partial": "частично", "no": "нет"}.get,
                        index=["yes", "partial", "no"].index(prev.get("resolved", "no")))
    crit = st.radio("Ошибка в критичном факте (сумма, дата, статус, выдуманный срок)?", ["нет", "да"],
                    horizontal=True, index=1 if prev.get("critical_error") else 0)
    tone = None
    if c["style"] == "emotional":
        tone = st.slider("Тон (1–5) для эмоционального клиента", 1, 5, prev.get("tone") or 3)
    unsure = st.checkbox("Сомневаюсь (инструкция не даёт однозначной метки)", prev.get("unsure", False))
    note = st.text_input("Комментарий", prev.get("note", ""))
    ok = st.form_submit_button("Сохранить и дальше", type="primary")
if ok:
    save({"case_id": case_id, "system": system, "resolved": resolved, "critical_error": crit == "да",
          "tone": tone, "unsure": unsure, "note": note})
    rest = [i for i in todo if i != pos]
    st.session_state.pos = rest[0] if rest else pos
    st.rerun()

c1, c2 = st.columns(2)
if c1.button("← Назад") and pos > 0:
    st.session_state.pos = pos - 1
    st.rerun()
if c2.button("Вперёд →") and pos < len(items) - 1:
    st.session_state.pos = pos + 1
    st.rerun()
