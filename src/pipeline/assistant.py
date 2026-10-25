"""Ассистент: handle(client_id, message, history) -> Response, suggest(client_id), operator_summary(dialog).

Поток: маскирование → правила безопасности → классификатор + эмоция → маршрут →
факты DWH → шаблон → (перефраз LLM с плейсхолдерами, если клиент раздражён) → гардрейлы → подстановка.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from src.config import cfg, cost_rub
from src.pipeline import dwh, emotion, guardrails, llm, pii, router, templates
from src.pipeline.templates import ACUTE, SCOPE


@dataclass
class Response:
    text: str
    route: str
    route_reason: str
    escalate: bool = False
    intent: str | None = None
    confidence: float = 0.0
    top_intents: list = field(default_factory=list)
    emotion: float = 0.0
    masked_text: str = ""
    understanding: dict = field(default_factory=dict)  # классификатор до слияния, транслит, DWH-априор, семейство
    pii_found: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    template_variant: str | None = None
    template_text: str | None = None
    paraphrase: str | None = None
    guardrails: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    options: list = field(default_factory=list)
    followup: dict | None = None
    operator_summary: str | None = None
    latency: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    cost_rub: float | None = None
    llm_calls: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


PARAPHRASE_SYSTEM = (
    "Ты — редактор ответов службы поддержки банка. Перепиши готовый ответ под состояние клиента: "
    "он раздражён, поэтому начни с одной короткой фразы сочувствия, дальше спокойно и по делу.\n"
    "СТРОГИЕ ПРАВИЛА:\n"
    "1. Сохрани ВСЕ плейсхолдеры в фигурных скобках ровно как есть, например {amount}, {merchant}. "
    "Не переводи, не склоняй и не удаляй их.\n"
    "2. Не добавляй никаких цифр, сумм, дат, сроков, процентов и обещаний, которых нет в исходном ответе.\n"
    "3. Не меняй смысл. Сохрани все отрицания («не», «нет»).\n"
    "4. Не длиннее исходного ответа плюс одно короткое предложение. Без приветствий и подписи.\n"
    "Верни только текст ответа."
)

SUMMARY_SYSTEM = (
    "Ты помогаешь оператору поддержки банка. По маскированному диалогу клиента с ботом составь:\n"
    "1) ВЫЖИМКА: 2–3 предложения — что случилось, что клиент хочет, эмоциональное состояние, что бот уже ответил.\n"
    "2) ЧЕРНОВИК ОТВЕТА: короткий ответ клиенту от лица оператора.\n"
    "Не придумывай суммы, даты и статусы — их оператор увидит в блоке фактов из DWH. "
    "Маски вида [CARD_1], [PHONE_1] оставляй как есть."
)


# Соседние интенты Banking77 → продуктовый сценарий. Переносим вероятность, только если DWH подтверждает
# сценарий (есть отклонённый перевод, холд, карта в доставке и т.п.). Таксономия Banking77 дробнее продукта.
FAMILIES = {
    "failed_transfer": "declined_transfer",
    "card_not_working": "declined_card_payment",
    "cash_withdrawal_charge": "card_payment_fee_charged",
    "request_refund": "Refund_not_showing_up",
    "card_delivery_estimate": "card_arrival",
    "balance_not_updated_after_bank_transfer": "pending_transfer",
    "transfer_timing": "pending_transfer",
    "direct_debit_payment_not_recognised": "card_payment_not_recognised",
    "cash_withdrawal_not_recognised": "card_payment_not_recognised",
}


# Скоуп-интенты, которые клиенты и классификатор путают между собой. Если по выбранному интенту в DWH нет
# данных, а по соседнему есть, шаблон no_data («у вас нет переводов в обработке») стал бы ложным отрицанием
# (провалы H064, H074 на отложенном наборе) — вместо него уточняем, о чём речь.
NEIGHBORS = {
    "declined_card_payment": ["declined_transfer"],
    "declined_transfer": ["declined_card_payment"],
    "pending_card_payment": ["pending_transfer", "transfer_not_received_by_recipient", "Refund_not_showing_up"],
    "pending_transfer": ["transfer_not_received_by_recipient", "pending_card_payment"],
    "transfer_not_received_by_recipient": ["pending_transfer"],
    "card_payment_fee_charged": ["extra_charge_on_statement", "card_payment_not_recognised"],
    "extra_charge_on_statement": ["card_payment_fee_charged", "card_payment_not_recognised"],
    "card_payment_not_recognised": ["extra_charge_on_statement", "transaction_charged_twice"],
    "transaction_charged_twice": ["pending_card_payment"],
    "Refund_not_showing_up": ["pending_card_payment"],
    "card_arrival": ["activate_my_card"],
    "activate_my_card": ["card_arrival"],
}


def neighbors_with_data(client_id: str, intent: str) -> list[str]:
    return [n for n in NEIGHBORS.get(intent, []) if dwh.facts_for(client_id, n).variant is not None]


def understand(client_id: str, masked: str) -> dict:
    """Интент по реплике + транслит + DWH-априор (предиктор) + семейства, подтверждённые данными."""
    import numpy as np

    from src.models import fusion
    from src.models import intent as intent_model
    from src.pipeline.normalize import translit_to_cyrillic

    P, classes, vec = intent_model.proba(masked)
    info: dict = {"clf_top": [], "translit": None, "prior_top": [], "family": None}
    alt = translit_to_cyrillic(masked)
    if alt:
        P2, _, vec2 = intent_model.proba(alt)
        if P2.max() > P.max():
            P, vec, info["translit"] = P2, vec2, alt
    info["clf_top"] = [(str(classes[i]), round(float(P[i]), 3)) for i in np.argsort(-P)[:3]]
    P, f = fusion.apply(client_id, P, classes)
    info["prior_top"] = [(i, round(p, 3)) for i, p in f.get("prior_top", [])]
    idx = {c: i for i, c in enumerate(classes)}
    src = str(classes[P.argmax()])
    if src in FAMILIES and dwh.facts_for(client_id, FAMILIES[src]).variant is not None:
        P = P.copy()
        P[idx[FAMILIES[src]]] += P[idx[src]]
        P[idx[src]] = 0.0
        info["family"] = f"{src} → {FAMILIES[src]}"
    order = np.argsort(-P)[:5]
    top = [(str(classes[i]), float(P[i])) for i in order]
    return {"intent": top[0][0], "conf": top[0][1], "top": top, "vec": vec, **info}


def _emo_threshold() -> float:
    return cfg()["emotion"]["irritation_threshold"]


def _match_option(message: str, history: list[dict]) -> str | None:
    """Ответ на уточняющий вопрос: клиент нажал вариант (текст = заголовок интента) или прислал номер."""
    last = next((h for h in reversed(history) if h.get("role") == "assistant"), None)
    opts = (last or {}).get("meta", {}).get("options") or []
    m = message.strip().lower().rstrip(".")
    for i, o in enumerate(opts, 1):
        if m in (templates.title(o).lower(), o.lower(), str(i)):
            return o
    return None


def handle(client_id: str, message: str, history: list[dict] | None = None, paraphrase: bool = True,
           with_summary: bool = False, use_cache: bool | None = None,
           forced_intent: str | None = None) -> Response:
    """forced_intent — клиент нажал подсказку или вариант уточнения (интент известен, классификатор не нужен)."""
    history = history or []
    lat: dict[str, float] = {}
    calls: list[llm.LLMResult] = []
    t_all = time.perf_counter()

    def stage(name, fn):
        t = time.perf_counter()
        out = fn()
        lat[name] = lat.get(name, 0.0) + time.perf_counter() - t
        return out

    # 1. ПДн
    m = stage("pii", lambda: pii.mask(message))
    masked = m.text
    has_hist = any(h.get("role") == "assistant" for h in history)

    # 2. Правила безопасности
    rule = stage("rules", lambda: router.pre_rules(masked, has_hist))

    # 3. Интент + эмоция
    understanding: dict = {}
    forced = forced_intent or _match_option(message, history)
    if forced:
        intent, conf, top, vec = forced, 1.0, [(forced, 1.0)], None
    else:
        u = stage("classifier", lambda: understand(client_id, masked))
        intent, conf, top, vec = u["intent"], u["conf"], u["top"], u["vec"]
        understanding = {k: u[k] for k in ("clf_top", "translit", "prior_top", "family")}
    emo = stage("emotion", lambda: emotion.score(masked, vec))

    route = rule or router.by_confidence(intent, conf, top)
    r = Response(text="", route=route.name, route_reason=route.reason, escalate=route.escalate, intent=intent,
                 confidence=round(conf, 3), top_intents=[(i, round(p, 3)) for i, p in top[:3]],
                 emotion=emo, masked_text=masked, pii_found=m.found, understanding=understanding)

    if route.name == "escalate_acute" and intent not in ACUTE:
        intent = max((t for t in top if t[0] in ACUTE), key=lambda t: t[1])[0]
        r.intent = intent
    if route.name == "template" and not forced:
        facts0 = stage("dwh", lambda: dwh.facts_for(client_id, intent))
        alt = stage("dwh", lambda: neighbors_with_data(client_id, intent)) if facts0.variant is None else []
        if alt:
            route = router.Route("clarify", f"по {intent} нет данных, есть по {', '.join(alt)}",
                                 system_key="clarify", options=alt + [intent])
            r.route, r.route_reason = route.name, route.reason
    if route.name in ("template", "escalate_acute"):
        facts = stage("dwh", lambda: dwh.facts_for(client_id, intent))
        tpl = templates.get(intent, facts.variant)
        r.facts, r.template_variant, r.template_text = facts.values, facts.variant, tpl.text
        r.actions, r.escalate = tpl.actions, r.escalate or tpl.handoff
        if tpl.followup and facts.raw.get("followup"):
            r.followup = facts.raw["followup"]
        text_ph = tpl.text
        use_llm = (paraphrase and route.name == "template" and emo >= _emo_threshold()
                   and facts.variant is not None)
        if use_llm:
            res = stage("llm", lambda: llm.chat(
                [{"role": "system", "content": PARAPHRASE_SYSTEM},
                 {"role": "user", "content": f"Сообщение клиента: «{masked}»\n\nИсходный ответ:\n{tpl.text}"}],
                role="local", temperature=0.4, max_tokens=400, use_cache=use_cache, tag="paraphrase"))
            calls.append(res)
            if res.error or not res.text.strip():
                r.guardrails.append(f"llm_error→fallback:{res.error}")
            else:
                g = stage("guardrails", lambda: guardrails.check_paraphrase(tpl.text, res.text.strip(),
                                                                         tpl.must_keep))
                if g.ok:
                    text_ph = res.text.strip()
                    r.paraphrase = text_ph
                else:
                    r.guardrails += [f"{v}→fallback_template" for v in g.violations]
        final = templates.fill(text_ph, facts.values)
        g = stage("guardrails", lambda: guardrails.check_final(
            final, set(facts.values.values()), tpl.text, dwh.other_clients_identifiers(client_id)))
        if not g.ok and text_ph != tpl.text:
            r.guardrails += [f"{v}→fallback_template" for v in g.violations]
            final = templates.fill(tpl.text, facts.values)
            g = guardrails.check_final(final, set(facts.values.values()), tpl.text)
        if not g.ok:
            r.guardrails += [f"{v}→escalate" for v in g.violations]
            final, r.escalate, r.route = templates.system_text("handoff"), True, "handoff_guardrail"
        r.text = final
    elif route.name == "clarify":
        r.options = route.options
        r.text = templates.system_text("clarify")
    else:
        r.text = templates.system_text(route.system_key or "handoff")

    if r.escalate and with_summary:
        dialog = history + [{"role": "user", "content": message}, {"role": "assistant", "content": r.text}]
        s = stage("llm_summary", lambda: operator_summary(dialog, client_id, r, use_cache=use_cache))
        r.operator_summary = s["text"]
        calls.extend(s["calls"])

    lat["total"] = time.perf_counter() - t_all
    r.latency = {k: round(v, 4) for k, v in lat.items()}
    r.llm_calls = [{"role": c.role, "model": c.model, "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens, "cost_rub": c.cost_rub, "latency_s": round(c.latency_s, 3),
                    "cached": c.cached, "error": c.error} for c in calls]
    r.tokens = {"prompt": sum(c.prompt_tokens for c in calls),
                "completion": sum(c.completion_tokens for c in calls)}
    costs = [c.cost_rub if c.cost_rub is not None else cost_rub(c.model, c.prompt_tokens, c.completion_tokens)
             for c in calls]
    r.cost_rub = 0.0 if not calls else (None if any(x is None for x in costs) else round(sum(costs), 4))
    return r


def operator_summary(dialog: list[dict], client_id: str | None = None, resp: Response | None = None,
                     use_cache: bool | None = None) -> dict:
    """Выжимка через smart-LLM по маскированному диалогу + факты DWH, добавленные кодом (не LLM)."""
    lines = []
    for h in dialog:
        who = "Клиент" if h["role"] == "user" else "Бот"
        lines.append(f"{who}: {pii.mask(h['content']).text}")
    meta = ""
    if resp is not None:
        meta = (f"\nМаршрут: {resp.route} ({resp.route_reason}). Интент: {resp.intent} "
                f"({resp.confidence:.2f}). Раздражение: {resp.emotion:.2f}.")
    res = llm.chat([{"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": "\n".join(lines) + meta}],
                   role="smart", temperature=0.2, max_tokens=800, use_cache=use_cache, tag="operator_summary")
    text = res.text if not res.error else f"(LLM недоступна: {res.error})\n" + "\n".join(lines[-4:])
    if client_id and resp is not None and resp.intent in SCOPE + ACUTE:
        f = dwh.facts_for(client_id, resp.intent)
        if f.values:
            text += "\n\nФАКТЫ ИЗ DWH (подставлены кодом):\n" + "\n".join(f"• {k}: {v}" for k, v in f.values.items())
    return {"text": text, "calls": [res]}


def suggest(client_id: str, k: int = 3) -> dict:
    from src.models import predictor

    t = time.perf_counter()
    top = predictor.suggest(client_id, k)
    return {"suggestions": [{"intent": i, "title": templates.title(i), "p": round(p, 3)} for i, p in top],
            "latency_s": round(time.perf_counter() - t, 4)}
