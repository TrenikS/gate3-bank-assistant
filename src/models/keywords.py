"""Бейзлайн «ключевые слова» (аналог сценарного бота): правила на словарях для 15 скоуп-интентов + острые.

Всё, что не сработало → "other". Работает на EN (Banking77) и RU (русский eval). Порядок правил важен.
"""
from __future__ import annotations

import re

OTHER = "other"

RULES_EN: list[tuple[str, str]] = [
    ("lost_or_stolen_card", r"\b(lost|stolen|missing|can'?t find)\b.*\bcard\b|\bcard\b.*\b(lost|stolen|missing)\b"),
    ("compromised_card", r"fraud|hack|compromis|someone (else )?(used|is using|has used)|scam"),
    ("transaction_charged_twice", r"twice|double|two times|duplicate|charged 2|same (payment|charge)"),
    ("Refund_not_showing_up", r"refund"),
    ("declined_transfer", r"transfer.*(declin|fail|reject|not go through|didn'?t go through|didn'?t work)|"
                          r"(declin|fail|reject).*transfer"),
    ("declined_card_payment", r"declin|rejected|card (payment )?(didn'?t|did not|doesn'?t|won'?t) work|"
                              r"payment (didn'?t|did not) go through"),
    ("card_payment_not_recognised", r"recogni[sz]e|didn'?t make|did not make|unknown (payment|transaction)|"
                                    r"don'?t know (this|the|that)"),
    ("card_payment_fee_charged", r"fee.*(card|payment|purchase)|(card|payment|purchase).*fee"),
    ("extra_charge_on_statement", r"extra|statement|additional charge|charged (me )?(more|extra)|fee"),
    ("transfer_not_received_by_recipient", r"(hasn'?t|has not|didn'?t|did not|not|never) "
                                           r"(receiv|arriv|get|got|show)|recipient|beneficiary"),
    ("pending_transfer", r"transfer.*(pending|still|taking|long|wait)|(pending|waiting).*transfer"),
    ("pending_card_payment", r"pending"),
    ("activate_my_card", r"activat"),
    ("card_arrival", r"card.*(arriv|deliver|come|in the mail|post|track|ship)|when will.*card|"
                     r"(still|haven'?t).*(received|got).*card"),
    ("change_pin", r"\bpin\b"),
    ("receiving_money", r"receiv\w* (money|payment|funds|transfer)|get paid|salary|"
                        r"(someone|somebody|friend).*send (me|money)|account (details|number)|bank details"),
    ("edit_personal_details", r"(change|update|edit|modify|correct)\b.*\b(address|name|details|personal|phone|"
                              r"e-?mail|surname)"),
]

RULES_RU: list[tuple[str, str]] = [
    ("lost_or_stolen_card", r"(потерял|потеряла|украли|укр[аа]ли|пропала|не могу найти|утерян)\w*.*карт|"
                            r"карт\w*.*(потерял|украли|пропала|утеряна)"),
    ("compromised_card", r"мошен|взлом|скомпрометир|кто-то (воспользовал|использ|снял|потратил)|развод"),
    ("transaction_charged_twice", r"дважды|два раза|2 раза|двойн|повторно списал|дубл"),
    ("Refund_not_showing_up", r"возврат|вернули|рефанд|вернуть деньги за"),
    ("declined_transfer", r"перев\w*.*(отклон|не прош|не проход|отказ|ошибк)|(отклон|не прош|отказ)\w*.*перев"),
    ("declined_card_payment", r"отклон|не прош|не проход|отказ|не принима|не работает карт|карта не работает"),
    ("card_payment_not_recognised", r"не узна|не знаю (что|эту|этот|такую)|непонятн\w* (платеж|платёж|операц|оплат)|"
                                    r"что за (платеж|платёж|операц|оплат)|не (совершал|делал)"),
    ("card_payment_fee_charged", r"комисс"),
    ("extra_charge_on_statement", r"списан|выписк|лишн|доп\w* плат|непонятн\w* списан|сняли"),
    ("transfer_not_received_by_recipient", r"(не пришл|не пришёл|не пришел|не получил|не дошл|не дош[её]л|не поступ)"),
    ("pending_transfer", r"перев\w*.*(обработк|завис|висит|ожида|долго|в пути|не зачисл)|(завис|висит|долго)\w*.*перев"),
    ("pending_card_payment", r"обработк|завис|висит|ожида|заморож|холд|заблокирован\w* сумм"),
    ("activate_my_card", r"актив"),
    ("card_arrival", r"карт\w*.*(достав|прид[её]т|придет|курьер|когда получ|не пришл|ждать)|когда .*карт"),
    ("change_pin", r"пин|pin"),
    ("receiving_money", r"реквизит|получить (деньги|перевод|зарплат)|зарплат|как мне (перевести|скинуть)|"
                        r"номер сч[её]та|принять перевод"),
    ("edit_personal_details", r"(смен|измен|помен|обнов|исправ)\w*.*(адрес|фамили|имя|данные|телефон|почт|e-?mail|номер)"),
]

_EN = [(i, re.compile(p, re.IGNORECASE)) for i, p in RULES_EN]
_RU = [(i, re.compile(p, re.IGNORECASE)) for i, p in RULES_RU]
_CYR = re.compile(r"[а-яё]", re.IGNORECASE)


def predict_one(text: str) -> str:
    rules = _RU if _CYR.search(text) else _EN
    for intent, rx in rules:
        if rx.search(text):
            return intent
    return OTHER


def predict(texts) -> list[str]:
    return [predict_one(t) for t in texts]
