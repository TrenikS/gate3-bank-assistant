"""Шаг 2: русский eval-набор — перевод стратифицированных подвыборок Banking77 через GigaChat.

python -m src.data.translate            # только оценка объёма токенов (ничего не тратит)
python -m src.data.translate --run      # перевод (с кэшем; повторный запуск бесплатный)

test: 10 примеров на интент (770), train: 30 на интент (2310). Пачки по BATCH фраз в одном запросе.
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd
from tqdm import tqdm

from src.config import cfg, cost_rub, path
from src.pipeline import llm

N_TEST, N_TRAIN, BATCH = 10, 30, 20
SYSTEM = (
    "Ты переводишь сообщения клиентов банка с английского на русский. Пиши так, как реальный клиент "
    "пишет в чат поддержки российского банка: разговорный стиль, коротко, можно без заглавных букв, "
    "иногда с опечатками или без знаков препинания (примерно в каждом пятом сообщении). "
    "Сохраняй смысл и все детали запроса. Валюту и реалии адаптируй под Россию (фунты/евро → рубли, "
    "если это не важно для смысла). Не добавляй ничего от себя.\n"
    "Вход — JSON-массив строк. Выход — ТОЛЬКО JSON-массив переводов той же длины и в том же порядке."
)


def sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = cfg()["seed"]
    tr = pd.read_csv(path("raw") / "banking77_train.csv")
    te = pd.read_csv(path("raw") / "banking77_test.csv")
    # в каждом интенте test ≥ 40, train ≥ 35 примеров — хватает на N_TEST/N_TRAIN
    s_te = pd.concat([g.sample(N_TEST, random_state=seed) for _, g in te.groupby("label")])
    s_tr = pd.concat([g.sample(N_TRAIN, random_state=seed) for _, g in tr.groupby("label")])
    return s_tr.reset_index(drop=True), s_te.reset_index(drop=True)


def _messages(batch: list[str]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)}]


def estimate() -> dict:
    s_tr, s_te = sample()
    texts = s_tr.text.tolist() + s_te.text.tolist()
    n_batches = -(-len(s_tr) // BATCH) + -(-len(s_te) // BATCH)
    prompt = llm.estimate_tokens(texts) + n_batches * llm.estimate_tokens([SYSTEM])
    completion = int(llm.estimate_tokens(texts) * 1.6)  # русский текст длиннее в токенах
    return {"texts": len(texts), "requests": n_batches, "model": llm.model_for("simple"),
            "prompt_tokens_est": prompt, "completion_tokens_est": completion,
            "total_tokens_est": prompt + completion,
            "cost_rub_est": cost_rub(llm.model_for("simple"), prompt, completion)}


def _parse(text: str, n: int) -> list[str] | None:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [str(x).strip() for x in out] if isinstance(out, list) and len(out) == n else None


def _translate_batch(batch: list[str], tag: str) -> list[str | None]:
    res = llm.chat(_messages(batch), role="simple", temperature=0.7, max_tokens=2000, tag=f"translate_{tag}")
    tr = _parse(res.text, len(batch)) if not res.error else None
    if tr is not None:
        return tr
    out = []  # пачка не распарсилась — по одной фразе
    for t in batch:
        r1 = llm.chat(_messages([t]), role="simple", temperature=0.7, max_tokens=200, tag=f"translate_{tag}_single")
        p = _parse(r1.text, 1)
        out.append(p[0] if p else None)
    return out


def translate_df(df: pd.DataFrame, tag: str, workers: int = 6) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor

    batches = [df.text.iloc[i:i + BATCH].tolist() for i in range(0, len(df), BATCH)]
    with ThreadPoolExecutor(workers) as ex:
        parts = list(tqdm(ex.map(lambda b: _translate_batch(b, tag), batches), total=len(batches), desc=tag))
    out = [t for part in parts for t in part]
    res = df.assign(text_en=df.text.values, text=out)
    return res[res.text.notna() & (res.text.str.len() > 0)][["text", "label", "text_en"]]


def main() -> None:
    est = estimate()
    print("Оценка объёма:", json.dumps(est, ensure_ascii=False))
    if "--run" not in sys.argv:
        print("Ничего не отправлено. Для запуска: python -m src.data.translate --run")
        return
    s_tr, s_te = sample()
    ru_te = translate_df(s_te, "test")
    ru_tr = translate_df(s_tr, "train")
    ru_te.to_csv(path("eval") / "banking77_ru_test.csv", index=False)
    ru_tr.to_csv(path("eval") / "banking77_ru_train.csv", index=False)
    spot = ru_te.sample(50, random_state=cfg()["seed"]).assign(ok="", comment="")
    spot.to_csv(path("eval") / "translation_spotcheck.csv", index=False)
    print(f"RU test: {len(ru_te)}, RU train: {len(ru_tr)}, spot-check: 50 → data/eval/translation_spotcheck.csv")


if __name__ == "__main__":
    main()
