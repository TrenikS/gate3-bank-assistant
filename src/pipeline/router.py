"""Маршрутизация: правила безопасности → уверенность классификатора → шаблон / уточнение / человек.

Работает только по маскированному тексту.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import cfg
from src.pipeline.templates import ACUTE, SCOPE


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts), re.IGNORECASE)


B = r"(?<![а-яёa-z])"  # левая граница слова
RULES: dict[str, re.Pattern] = {
    "crisis": _rx(r"жить не хоч", r"не хоч\w* (больше )?жить", r"поконч\w* с собой", r"суицид",
                  r"(убить|убью) себя", r"св(ести|еду) сч[её]ты", r"нет смысла жить", r"(лучше|хочу) умереть",
                  r"выйти в окно", r"выйду в окно", r"повешусь", r"вскр\w* вены", r"не вижу смысла жить",
                  r"зачем (мне )?жить", r"жизнь кончена"),
    "fraud": _rx(r"служб\w* безопасности", r"звонил\w* (из|от) банка", r"сотрудник\w* банка звонил",
                 r"безопасн\w* сч[её]т", r"(продиктовал|сказал|назвал|сообщил|отправил)\w* (им |ему |ей )?(код|cvc|cvv|пин)",
                 r"мошенни", r"взлом", r"украли (деньги|с карты|со сч)", r"списали без мо", r"без моего (ведома|согласия)",
                 r"(я )?не (совершал|делал|оплачивал|подтверждал)\w*", r"это не я", r"не моя операция",
                 r"(пришлите|пришли|скажи|скажите|назови|назовите|продиктуй\w*) (мне )?(код|cvc|cvv|пин|пароль)",
                 r"\[SMS_CODE_\d+\]", r"\[CVC_\d+\]", r"перев\w* на (резервн|защищ)\w* сч"),
    "other_person_data": _rx(
        r"(баланс|сч[её]т|карт\w*|выписк\w*|операци\w*|данные|остаток|номер)\s+(моего |моей )?"
        r"(мужа|жены|мамы|матери|папы|отца|сына|дочери|дочки|брата|сестры|друга|подруги|соседа|соседки|"
        r"коллеги|бабушки|дедушки|другого (человека|клиента))",
        r"чу?ж(ой|ую|ого|ие) (сч[её]т|карт|данн|баланс)", r"кому принадлежит", r"ч(ей|ья) (это )?(сч[её]т|карта)",
        r"данные другого клиента", r"покажи\w* (данные|операции) \[NAME_\d+\]"),
    "injection": _rx(r"забудь", r"игнорируй", r"ignore (all|previous|the)", r"системн\w* (промпт|инструкц)",
                     r"system prompt", r"(покажи|выведи|раскрой)\w* (свой |свои |твой )?(промпт|инструкц)",
                     r"ты теперь", r"you are now", r"developer mode", r"режим разработчика", r"jailbreak",
                     B + r"dan" + r"(?![a-z])", r"притворись", r"представь, что ты", r"без ограничений",
                     r"исходн\w* инструкц"),
    "investment": _rx(r"куда (лучше )?(вложить|инвестировать)", r"как\w* акци", r"инвест\w* (совет|рекоменд)",
                      r"стоит ли (покупать|вкладывать|инвестировать)", r"биткоин", r"крипт", r"доходност",
                      r"что (лучше )?купить.{0,20}(акци|облигац|валют)", r"посоветуй\w* (вклад|акци|инвест)"),
    "external_email": _rx(r"(отправ|пришл|скин|вышл|перешл)\w*.{0,40}(на|по) (почт|e-?mail|имейл|мейл|\[EMAIL)",
                          r"(на|по) (почт\w*|e-?mail|имейл\w*) \[EMAIL_\d+\]"),
    "human": _rx(r"оператор", r"живо\w* (человек|сотрудник)", r"позов\w* (человек|сотрудник)",
                 r"соедини\w* с (человеком|сотрудником|оператором)", r"хочу (поговорить )?с человеком"),
    "dispute_strong": _rx(r"оспори", r"претензи", r"chargeback", r"чарджбэк", r"жалоб\w* в цб",
                          r"верните (мне )?деньги", r"требую (вернуть|возврат)"),
    "dispute_soft": _rx(r"неправд", r"это не так", r"ошибаетесь", r"не соглас", r"вр[её]те", r"враньё",
                        r"неверно", r"лимит не (исчерпан|превышен)", r"(деньги|средства) (же )?(были|есть) на карт",
                        r"у меня (же )?были деньги", r"не превыша", r"ничего подобного", r"какой ещё лимит"),
}


@dataclass
class Route:
    name: str            # template | clarify | handoff | escalate_acute | escalate_fraud | crisis | refuse_*
    reason: str
    escalate: bool = False
    system_key: str | None = None   # ключ из templates.yaml/system для фиксированных ответов
    options: list[str] = field(default_factory=list)


def pre_rules(masked: str, has_history: bool) -> Route | None:
    t = masked.lower()
    if RULES["crisis"].search(t):
        return Route("crisis", "кризисные слова", escalate=True, system_key="crisis")
    if RULES["fraud"].search(masked):
        return Route("escalate_fraud", "признаки мошенничества/соц. инженерии", escalate=True, system_key="fraud")
    if RULES["other_person_data"].search(t):
        return Route("refuse_other_person", "запрос чужих данных", system_key="other_person_data")
    if RULES["injection"].search(t):
        return Route("refuse_injection", "попытка изменить инструкции", system_key="injection")
    if RULES["external_email"].search(masked):
        return Route("refuse_external_email", "отправка данных на внешний e-mail", system_key="external_email")
    if RULES["investment"].search(t):
        return Route("refuse_investment", "инвестсовет вне компетенции", system_key="investment")
    if RULES["human"].search(t):
        return Route("handoff", "клиент просит человека", escalate=True, system_key="handoff")
    if RULES["dispute_strong"].search(t) or (has_history and RULES["dispute_soft"].search(t)):
        return Route("handoff_dispute", "спор о фактах", escalate=True, system_key="dispute")
    return None


def by_confidence(intent: str, conf: float, probs_top: list[tuple[str, float]]) -> Route:
    from src.models.intent import thresholds

    th = thresholds()  # откалиброванные пороги из обучения (шаг 1), иначе — из config.yaml
    acute_p = sum(p for i, p in probs_top if i in ACUTE)
    if intent in ACUTE or acute_p >= 0.35:
        return Route("escalate_acute", f"острый интент (p={max(conf, acute_p):.2f})", escalate=True)
    if conf >= th["answer"]:
        if intent in SCOPE:
            return Route("template", f"уверенность {conf:.2f} ≥ {th['answer']}")
        return Route("handoff_out_of_scope", f"интент {intent} вне скоупа", escalate=True,
                     system_key="handoff_out_of_scope")
    if conf >= th["clarify"]:
        opts = [i for i, _ in probs_top if i in SCOPE][:3]
        if len(opts) >= 2:
            return Route("clarify", f"средняя уверенность {conf:.2f}", system_key="clarify", options=opts)
    return Route("handoff", f"низкая уверенность {conf:.2f}", escalate=True, system_key="handoff")
