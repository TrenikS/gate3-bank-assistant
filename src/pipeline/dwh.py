"""Доступ к синтетическому DWH: факты клиента под интент.

facts_for(client_id, intent) -> Facts(variant, values) — values уже отформатированы для подстановки
в шаблон. Это единственный источник цифр, дат и статусов в ответе ассистента.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.config import path
from src.data.synth_dwh import SIM_NOW

_local = threading.local()
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
          "октября", "ноября", "декабря"]
TICKET_SLA_DAYS = 3

DECLINE_TEXT = {"limit_exceeded": "превышен лимит", "insufficient_funds": "недостаточно средств",
                "merchant_error": "ошибка магазина", "card_blocked": "карта заблокирована",
                "wrong_pin": "неверный PIN", "fraud_suspected": "подозрение на мошенничество"}
TRANSFER_STATUS = {"processing": "в обработке", "pending": "ожидает зачисления",
                   "completed": "выполнен", "declined": "отклонён"}
REFUND_STATUS = {"pending": "передан в банк, ожидает зачисления", "completed": "зачислен"}
FEE_TEXT = {
    "atm_other_bank": ("снятие наличных в банкомате другого банка по тарифу карты",
                       "В банкоматах нашего банка снятие без комиссии — ближайшие есть на карте в приложении."),
    "quasi_cash": ("пополнение электронного кошелька картой по тарифу приравнивается к снятию наличных",
                   "Пополнять кошелёк без комиссии можно переводом по СБП."),
    "currency": ("конвертация валюты при оплате в иностранной валюте",
                 "Чтобы не платить за конвертацию, можно оплачивать с валютного счёта."),
}


def con() -> sqlite3.Connection:
    c = getattr(_local, "con", None)
    if c is None:
        c = sqlite3.connect(path("dwh"), check_same_thread=False)
        c.row_factory = sqlite3.Row
        _local.con = c
    return c


def q(sql: str, *args) -> list[dict]:
    return [dict(r) for r in con().execute(sql, args).fetchall()]


def q1(sql: str, *args) -> dict | None:
    rows = q(sql, *args)
    return rows[0] if rows else None


# ── форматирование ──
def fmt_money(x: float) -> str:
    s = f"{x:,.2f}".replace(",", " ").replace(".00", "")
    return s.replace(".", ",") + " ₽"


def _dt(s: str | datetime) -> datetime:
    return s if isinstance(s, datetime) else datetime.fromisoformat(s)


def fmt_dt(s, with_time: bool = True) -> str:
    d = _dt(s)
    days = (SIM_NOW.date() - d.date()).days
    day = {0: "сегодня", 1: "вчера", -1: "завтра"}.get(days, f"{d.day} {MONTHS[d.month - 1]}")
    if not with_time:
        return day
    return f"{day} в {d:%H:%M}"


def fmt_date(s) -> str:
    return fmt_dt(s, with_time=False)


def fmt_card(last4: str) -> str:
    return f"·· {last4}"


def mask_phone(p: str) -> str:
    digits = "".join(ch for ch in p if ch.isdigit())
    return f"+7 ••• •••-{digits[-4:-2]}-{digits[-2:]}"


def mask_email(e: str) -> str:
    user, dom = e.split("@")
    return f"{user[0]}•••@{dom}"


def ticket(client_id: str, intent: str) -> dict:
    tid = "T-" + hashlib.sha1(f"{client_id}:{intent}".encode()).hexdigest()[:6].upper()
    return {"ticket_id": tid, "ticket_due": fmt_date(SIM_NOW + timedelta(days=TICKET_SLA_DAYS))}


@dataclass
class Facts:
    variant: str | None  # None → no_data
    values: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)  # исходные записи DWH (для гардрейлов и панели «под капотом»)


def client(client_id: str) -> dict | None:
    return q1("SELECT * FROM clients WHERE client_id=?", client_id)


def primary_card(client_id: str) -> dict | None:
    return q1("SELECT * FROM cards WHERE client_id=? ORDER BY is_primary DESC", client_id)


def _since(minutes: float) -> str:
    return (SIM_NOW - timedelta(minutes=minutes)).isoformat(sep=" ")


def _card_of(card_id: str) -> str:
    c = q1("SELECT last4 FROM cards WHERE card_id=?", card_id)
    return fmt_card(c["last4"]) if c else ""


def facts_for(client_id: str, intent: str) -> Facts:
    cl = client(client_id)
    if cl is None:
        return Facts(None)
    fn = _FACTS.get(intent)
    return fn(client_id, cl) if fn else Facts(None)


def _declined_card_payment(cid, cl):
    t = q1("SELECT * FROM transactions WHERE client_id=? AND status='declined' AND ts>=? "
           "AND decline_reason!='fraud_suspected' ORDER BY ts DESC", cid, _since(7 * 24 * 60))
    if not t:
        return Facts(None)
    card = q1("SELECT * FROM cards WHERE card_id=?", t["card_id"])
    v = dict(amount=fmt_money(t["amount"]), merchant=t["merchant"], datetime=fmt_dt(t["ts"]),
             card=fmt_card(card["last4"]), spent_today=fmt_money(card["spent_today"]),
             daily_limit=fmt_money(card["daily_limit"]), balance=fmt_money(cl["balance"]),
             block_reason=card["block_reason"] or "причина уточняется")
    return Facts(t["decline_reason"], v, {"transaction": t, "card": card})


def _declined_transfer(cid, cl):
    t = q1("SELECT * FROM transfers WHERE client_id=? AND status='declined' AND ts>=? ORDER BY ts DESC",
           cid, _since(7 * 24 * 60))
    if not t:
        return Facts(None)
    v = dict(amount=fmt_money(t["amount"]), recipient=t["recipient"], method=t["method"],
             datetime=fmt_dt(t["ts"]), fee=fmt_money(t["fee"] or 0), balance=fmt_money(cl["balance"]),
             limit_used=fmt_money(cl["transfer_limit_used"]), transfer_limit=fmt_money(cl["transfer_limit"]))
    return Facts(t["decline_reason"], v, {"transfer": t})


def _fee(cid, cl):
    t = q1("SELECT * FROM transactions WHERE client_id=? AND fee>0 AND ts>=? ORDER BY ts DESC",
           cid, _since(30 * 24 * 60))
    if not t:
        return Facts(None)
    reason, advice = FEE_TEXT.get(t["fee_code"], ("тариф карты", ""))
    v = dict(amount=fmt_money(t["amount"]), merchant=t["merchant"], datetime=fmt_dt(t["ts"]),
             card=_card_of(t["card_id"]), fee=fmt_money(t["fee"]), fee_reason=reason, fee_advice=advice)
    return Facts("default", v, {"transaction": t})


def _not_recognised(cid, cl):
    t = q1("SELECT * FROM transactions WHERE client_id=? AND descriptor!=UPPER(merchant) AND status='ok' "
           "AND is_service_charge=0 AND ts>=? ORDER BY ts DESC", cid, _since(7 * 24 * 60))
    if not t:
        return Facts(None)
    from src.pipeline.guardrails import descriptor_suspicious

    if descriptor_suspicious(t["descriptor"]):  # недоверенный текст из данных клиенту не показываем
        v = dict(amount=fmt_money(t["amount"]), datetime=fmt_dt(t["ts"]), merchant=t["merchant"])
        return Facts("suspicious_descriptor", v, {"transaction": t})
    v = dict(amount=fmt_money(t["amount"]), datetime=fmt_dt(t["ts"]), descriptor=t["descriptor"],
             merchant=t["merchant"], merchant_category=t["category"])
    return Facts("default", v, {"transaction": t})


def _twice(cid, cl):
    d = q1("SELECT * FROM transactions WHERE client_id=? AND duplicate_of IS NOT NULL ORDER BY ts DESC", cid)
    if not d:
        return Facts(None)
    first = q1("SELECT * FROM transactions WHERE tx_id=?", d["duplicate_of"])
    v = dict(amount=fmt_money(d["amount"]), merchant=d["merchant"], datetime=fmt_dt(first["ts"]),
             datetime2=fmt_dt(d["ts"]))
    if d["status"] == "pending":
        v["release_date"] = fmt_date(d["hold_until"])
        return Facts("hold", v, {"transaction": d, "original": first})
    v.update(ticket(cid, "transaction_charged_twice"))
    return Facts("settled", v, {"transaction": d, "original": first})


def _extra_charge(cid, cl):
    from src.data.synth_dwh import SERVICE_CHARGES

    t = q1("SELECT * FROM transactions WHERE client_id=? AND is_service_charge=1 AND ts>=? ORDER BY ts DESC",
           cid, _since(30 * 24 * 60))
    if not t:
        return Facts(None)
    name, reason, action, _ = next(s for s in SERVICE_CHARGES if s[0] == t["merchant"])
    v = dict(amount=fmt_money(t["amount"]), datetime=fmt_dt(t["ts"]), charge_name=name.lower()
             if not name.startswith("Подписка") else name, charge_reason=reason, charge_action=action)
    return Facts("default", v, {"transaction": t})


def _transfer_facts(t):
    return dict(amount=fmt_money(t["amount"]), recipient=t["recipient"], method=t["method"],
                datetime=fmt_dt(t["ts"]), status_text=TRANSFER_STATUS.get(t["status"], t["status"]),
                recipient_bank=t["recipient_bank"],
                expected_at=fmt_dt(t["expected_at"]) if t["expected_at"] else "")


def _followup_at(expected_at: str) -> str:
    return fmt_dt(_dt(expected_at) + timedelta(minutes=30))


def _followup(t: dict) -> dict:
    """Запланированная проверка исхода — в то же время, что ассистент обещает в ответе (срок + 30 мин)."""
    return {"at": f"{_dt(t['expected_at']) + timedelta(minutes=30):%Y-%m-%d %H:%M:%S}", "check": t["tr_id"]}


def _pending_transfer(cid, cl):
    t = q1("SELECT * FROM transfers WHERE client_id=? AND status IN ('processing','pending') "
           "AND expected_at>=? ORDER BY ts DESC", cid, SIM_NOW.isoformat(sep=" "))
    if not t:
        return Facts(None)
    v = _transfer_facts(t)
    v["followup_at"] = _followup_at(t["expected_at"])
    return Facts("default", v, {"transfer": t, "followup": _followup(t)})


def _not_received(cid, cl):
    now = SIM_NOW.isoformat(sep=" ")
    t = q1("SELECT * FROM transfers WHERE client_id=? AND status IN ('processing','pending') AND expected_at<? "
           "ORDER BY ts DESC", cid, now)
    if t:
        v = _transfer_facts(t)
        v.update(ticket(cid, "transfer_not_received_by_recipient"))
        return Facts("overdue", v, {"transfer": t})
    t = q1("SELECT * FROM transfers WHERE client_id=? AND status='completed' AND expected_at>=? ORDER BY ts DESC",
           cid, now)
    if t:
        v = _transfer_facts(t)
        v["completed_at"] = fmt_dt(t["completed_at"])
        v["followup_at"] = _followup_at(t["expected_at"])
        return Facts("completed", v, {"transfer": t, "followup": _followup(t)})
    return Facts(None)


def _pending_card_payment(cid, cl):
    t = q1("SELECT * FROM transactions WHERE client_id=? AND status='pending' AND duplicate_of IS NULL "
           "ORDER BY ts ASC", cid)
    if not t:
        return Facts(None)
    v = dict(amount=fmt_money(t["amount"]), merchant=t["merchant"], datetime=fmt_dt(t["ts"]),
             card=_card_of(t["card_id"]), hold_until=fmt_date(t["hold_until"]))
    return Facts("default", v, {"transaction": t})


def _new_card(cid):
    return q1("SELECT * FROM cards WHERE client_id=? AND status IN ('in_delivery','not_activated') "
              "ORDER BY order_date DESC", cid)


def _card_arrival(cid, cl):
    c = _new_card(cid)
    if not c:
        return Facts(None)
    v = dict(card=fmt_card(c["last4"]), order_date=fmt_date(c["order_date"]),
             delivery_status=c["delivery_status"], delivery_date=fmt_date(c["delivery_date"]),
             delivery_method=c["delivery_method"])
    return Facts("in_delivery" if c["status"] == "in_delivery" else "delivered", v, {"card": c})


def _refund(cid, cl):
    rf = q1("SELECT * FROM refunds WHERE client_id=? AND status='pending' ORDER BY initiated_at DESC", cid)
    if not rf:
        return Facts(None)
    v = dict(amount=fmt_money(rf["amount"]), merchant=rf["merchant"], initiated_at=fmt_date(rf["initiated_at"]),
             refund_status=REFUND_STATUS[rf["status"]], expected_by=fmt_date(rf["expected_by"]),
             card=_card_of(rf["card_id"]))
    overdue = rf["expected_by"] < SIM_NOW.isoformat(sep=" ")
    if overdue:
        v.update(ticket(cid, "Refund_not_showing_up"))
    return Facts("overdue" if overdue else "pending", v, {"refund": rf})


def _activate(cid, cl):
    c = _new_card(cid)
    if c:
        v = dict(card=fmt_card(c["last4"]), delivery_date=fmt_date(c["delivery_date"]))
        return Facts("not_activated" if c["status"] == "not_activated" else "in_delivery", v, {"card": c})
    c = primary_card(cid)
    if c and c["status"] == "active":
        return Facts("already_active", dict(card=fmt_card(c["last4"])), {"card": c})
    return Facts(None)


def _change_pin(cid, cl):
    c = q1("SELECT * FROM cards WHERE client_id=? AND status='active' ORDER BY is_primary DESC", cid)
    if c:
        return Facts("default", dict(card=fmt_card(c["last4"])), {"card": c})
    b = q1("SELECT * FROM cards WHERE client_id=? AND status='blocked' ORDER BY is_primary DESC", cid)
    if b:
        return Facts("blocked", dict(card=fmt_card(b["last4"]), block_reason=b["block_reason"] or "причина уточняется"),
                     {"card": b})
    return Facts(None)


def _receiving(cid, cl):
    v = dict(full_name=cl["full_name"], account_number=cl["account_number"], bank_name="АО «Демо-Банк»",
             bik="044525000", corr_account="30101810400000000000", phone_masked=mask_phone(cl["phone"]))
    return Facts("default", v, {"client": {k: cl[k] for k in ("client_id", "account_number")}})


def _edit_details(cid, cl):
    v = dict(phone_masked=mask_phone(cl["phone"]), email_masked=mask_email(cl["email"]))
    return Facts("default", v, {})


def _acute(cid, cl):
    c = primary_card(cid)
    if not c:
        return Facts(None)
    recent = q("SELECT * FROM transactions WHERE client_id=? AND ts>=? ORDER BY ts DESC", cid, _since(24 * 60))
    return Facts("default", dict(card=fmt_card(c["last4"])), {"card": c, "recent_transactions": recent[:10]})


_FACTS = {
    "declined_card_payment": _declined_card_payment, "declined_transfer": _declined_transfer,
    "card_payment_fee_charged": _fee, "card_payment_not_recognised": _not_recognised,
    "transaction_charged_twice": _twice, "extra_charge_on_statement": _extra_charge,
    "pending_transfer": _pending_transfer, "transfer_not_received_by_recipient": _not_received,
    "pending_card_payment": _pending_card_payment, "card_arrival": _card_arrival,
    "Refund_not_showing_up": _refund, "activate_my_card": _activate, "change_pin": _change_pin,
    "receiving_money": _receiving, "edit_personal_details": _edit_details,
    "compromised_card": _acute, "lost_or_stolen_card": _acute,
}


def client_fact_strings(client_id: str) -> set[str]:
    """Все значения, которые можно законно упомянуть для клиента (для гардрейлов и авто-проверки фактов)."""
    out = set()
    for intent in _FACTS:
        out.update(str(v) for v in facts_for(client_id, intent).values.values())
    return out


def other_clients_identifiers(client_id: str) -> list[str]:
    """Имена и счета других клиентов — для гардрейла «нет чужих данных»."""
    me = client(client_id) or {}
    own = {me.get("full_name"), me.get("account_number")}  # тёзки возможны: своё имя — не чужие данные
    rows = q("SELECT full_name, account_number FROM clients WHERE client_id!=?", client_id)
    return [v for r in rows for v in (r["full_name"], r["account_number"]) if v not in own]
