"""Шаг 0: скачать Banking77 и Bitext в data/raw, вывести размеры и примеры.

python -m src.data.download
"""
from __future__ import annotations

import pandas as pd

from src.config import path

B77_PARQUET = "hf://datasets/PolyAI/banking77@refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
B77_CSV = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"
BITEXT_REPOS = [
    "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
]


def _banking77_split(split: str) -> pd.DataFrame:
    # 1) load_dataset (может упасть из-за скрипта загрузки)
    try:
        from datasets import load_dataset

        ds = load_dataset("PolyAI/banking77", split=split)
        names = ds.features["label"].names
        df = ds.to_pandas()
        df["label"] = df["label"].map(lambda i: names[i])
        print(f"  banking77/{split}: load_dataset OK")
        return df[["text", "label"]]
    except Exception as e:
        print(f"  banking77/{split}: load_dataset упал ({type(e).__name__}), пробую parquet")
    # 2) GitHub CSV — там метки строками (parquet содержит только индексы)
    try:
        df = pd.read_csv(B77_CSV.format(split=split)).rename(columns={"category": "label"})
        print(f"  banking77/{split}: GitHub CSV OK")
        return df[["text", "label"]]
    except Exception as e:
        print(f"  banking77/{split}: GitHub CSV упал ({type(e).__name__})")
    raise RuntimeError("Не удалось скачать Banking77")


def banking77() -> dict[str, pd.DataFrame]:
    out = {}
    for split in ("train", "test"):
        f = path("raw") / f"banking77_{split}.csv"
        if not f.exists():
            _banking77_split(split).to_csv(f, index=False)
        out[split] = pd.read_csv(f)
    return out


def bitext() -> pd.DataFrame:
    f = path("raw") / "bitext.csv"
    if f.exists():
        return pd.read_csv(f)
    from datasets import load_dataset

    for repo in BITEXT_REPOS:
        try:
            df = load_dataset(repo, split="train").to_pandas()
            df["source"] = repo
            df.to_csv(f, index=False)
            print(f"  bitext: взят {repo}")
            return df
        except Exception as e:
            print(f"  bitext: {repo} недоступен ({type(e).__name__}: {e})")
    raise RuntimeError("Не удалось скачать Bitext")


def main() -> None:
    b = banking77()
    for split, df in b.items():
        print(f"\nBanking77 {split}: {len(df)} строк, {df.label.nunique()} интентов")
        print(df.sample(3, random_state=0).to_string(index=False))
    labels = sorted(b["train"].label.unique())
    print("\nМетки с заглавной буквы:", [l for l in labels if l[0].isupper()])

    bt = bitext()
    print(f"\nBitext ({bt.source.iloc[0]}): {len(bt)} строк, колонки {list(bt.columns)}")
    print(f"интентов: {bt.intent.nunique()}, категорий: {bt.category.nunique()}")
    print(bt.intent.value_counts().head(10).to_string())
    for _, r in bt.sample(2, random_state=0).iterrows():
        print(f"\n[{r.intent}] Q: {r.instruction}\nA: {r.response[:400]}")


if __name__ == "__main__":
    main()
