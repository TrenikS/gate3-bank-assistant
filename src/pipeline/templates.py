"""Шаблоны ответов: загрузка data/templates.yaml, выбор варианта, подстановка фактов."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from src.config import path

PLACEHOLDER = re.compile(r"\{([a-z_0-9]+)\}")
SCOPE = ["declined_card_payment", "declined_transfer", "card_payment_fee_charged",
         "card_payment_not_recognised", "transaction_charged_twice", "extra_charge_on_statement",
         "pending_transfer", "transfer_not_received_by_recipient", "pending_card_payment", "card_arrival",
         "Refund_not_showing_up", "activate_my_card", "change_pin", "receiving_money",
         "edit_personal_details"]
ACUTE = ["compromised_card", "lost_or_stolen_card"]


@lru_cache(maxsize=1)
def load() -> dict:
    with open(path("templates"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def intent_cfg(intent: str) -> dict | None:
    return load()["intents"].get(intent)


def system_text(key: str) -> str:
    return load()["system"][key]


def title(intent: str) -> str:
    c = intent_cfg(intent)
    return c["title"] if c else intent


def segment(intent: str) -> str | None:
    c = intent_cfg(intent)
    return c["segment"] if c else None


@dataclass
class Template:
    intent: str
    variant: str | None
    text: str                      # с плейсхолдерами
    must_keep: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    handoff: bool = False
    followup: bool = False

    @property
    def placeholders(self) -> list[str]:
        return PLACEHOLDER.findall(self.text)


def get(intent: str, variant: str | None) -> Template | None:
    c = intent_cfg(intent)
    if c is None:
        return None
    if variant is None:
        return Template(intent, None, c["no_data"].strip())
    v = c["variants"].get(variant) or c["variants"].get("default")
    if v is None:
        return Template(intent, None, c["no_data"].strip())
    return Template(intent, variant, v["text"].strip(), v.get("must_keep", []), v.get("actions", []),
                    v.get("handoff", False), v.get("followup", False))


def fill(text: str, values: dict) -> str:
    """Подстановка значений кодом. Неизвестный плейсхолдер → KeyError (лучше упасть, чем выдумать)."""
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)
