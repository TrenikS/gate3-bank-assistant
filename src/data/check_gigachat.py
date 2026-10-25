"""Шаг 0: проверка доступа к GigaChat — токен, список моделей, один тестовый запрос.

python -m src.data.check_gigachat
"""
from src.pipeline import llm


def main() -> None:
    models = llm.list_models()
    print("Доступные модели:", ", ".join(models))
    for role in ("smart", "simple"):
        m = llm.model_for(role)
        print(f"  роль {role}: {m} {'OK' if m in models else '!!! нет в списке — поправь config.yaml'}")
    r = llm.chat([{"role": "user", "content": "Ответь одним словом: столица России?"}],
                 role="simple", max_tokens=10, use_cache=False, tag="healthcheck")
    if r.error:
        print("Ошибка запроса:", r.error)
    else:
        print(f"Ответ: {r.text!r} | модель {r.model} | токены {r.prompt_tokens}+{r.completion_tokens}"
              f" | {r.latency_s:.2f} с")


if __name__ == "__main__":
    main()
