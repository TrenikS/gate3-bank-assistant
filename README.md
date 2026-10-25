# Gate 3: предиктивный ассистент поддержки банка — прототип

Ассистент в чате розничного банка: подсказки при открытии чата → маскирование ПДн → интент + эмоция →
маршрут → **проверенный шаблон + факты из DWH** (цифры подставляет код, не LLM) → гардрейлы → эскалация с
выжимкой для оператора. Подробный бриф — в [CLAUDE.md](CLAUDE.md).

## Результаты

- **Демо-видео:** https://www.youtube.com/watch?v=8ssncw_1xLw — то же самое вживую: `make demo`, вкладка «📊 Результаты».
- **[reports/gate3_metrics.md](reports/gate3_metrics.md)** — все таблицы для документа (D1–D3, E1, F1, G1), собираются `make report`.
- [reports/figures/](reports/figures/) — confusion, coverage–accuracy, Hit@k, латентность.
- На ревью Ивану: [data/templates.yaml](data/templates.yaml) (шаблоны), [data/eval/e2e_review.csv](data/eval/e2e_review.csv)
  (E2E-набор), [data/eval/redteam_review.csv](data/eval/redteam_review.csv), [data/eval/translation_spotcheck.csv](data/eval/translation_spotcheck.csv),
  разметка ответов — `make label`.

## Запуск с нуля

```bash
# 1. Окружение (uv; если uv нет — pip install uv в любой venv)
uv python install 3.11
uv sync                                   # Python 3.11, зависимости из pyproject.toml
./scripts/fix_lightgbm_macos.sh           # только macOS без Homebrew-libomp (см. ниже)
cp .env.example .env                      # и вписать POLZA_AI_API_KEY

# 2. Демо (всё уже обучено и посчитано)
make demo                                 # = .venv/bin/streamlit run app/streamlit_app.py
make label                                # страница human eval

# 3. Пересчёт с нуля
make data                                 # шаг 0 + синтетический DWH (без LLM)
python -m src.data.check_gigachat         # проверка LLM-провайдера: модели, тестовый запрос
make estimate                             # оценка токенов и рублей ДО платных прогонов
python -m src.data.translate --run        # шаг 2: перевод Banking77 → RU (~28 ₽)
make train                                # шаги 1, 3 и слияние (без LLM)
python -m src.data.build_e2e              # E2E-набор
python -m src.eval.e2e run --systems ours,A,B   # прототип — последовательно (замер латентности), A — 8 потоков
python -m src.eval.e2e judge --run        # LLM-судья, 8 потоков
python -m src.eval.redteam build && python -m src.eval.redteam run && python -m src.eval.redteam judge --run
python -m src.eval.latency
make report
make test
```

Все вызовы LLM кэшируются (`cache/llm`), поэтому повторный прогон с теми же запросами бесплатный. Флаг `--fresh`
у судьи переоценивает всё (нужен после смены рубрики).

## LLM: провайдер и роли

**Отступление от брифа (решение Ивана):** вместо GigaChat используются модели через агрегатор
**Polza AI** (`https://polza.ai/api/v1`, OpenAI-совместимый API). Причины: у прямого ключа GigaChat нулевой
баланс (402), GigaChat через Polza дорогой. Переключение обратно — `llm.provider: gigachat` в config.yaml.

| Роль | Модель | Для чего |
|---|---|---|
| smart | `anthropic/claude-sonnet-5.5` | выжимка оператору, LLM-судья |
| simple | `google/gemini-3.8-flash` | перевод, бейзлайн A |
| local | `google/gemini-3.5-flash-lite` | перефраз шаблона (маскированные данные клиента) |

- У Flash и Sonnet через Polza **обязательное скрытое reasoning**: без настроек оно съедало весь `max_tokens` и
  возвращало пустой или обрезанный текст. Настройки под каждую модель лежат в `llm.polza.extra_body`
  (Sonnet: `effort: low`). Для перефраза взят Flash-Lite: 5 из 5 перефразов прошли гардрейлы, 2,3 с, 0,03 ₽
  (у Flash — 1 из 5, 4,8 с, 0,17 ₽).
- Цены (вход и выход отдельно) — `price_rub_per_1k_tokens`. Фактическая стоимость каждого вызова берётся из
  `usage.cost_rub` ответа Polza и пишется в `logs/llm_calls.jsonl` вместе с текстом запроса и ответа.
- **ПДн:** модели Polza не соответствуют 152-ФЗ, Gemini и Claude хранят данные вне РФ. В LLM уходит только
  маскированный текст. Для продакшена нужен РФ-контур или on-prem (см. E1 и F1 в отчёте).
- Расход на весь прототип: ~455 ₽. Основные статьи: выжимки на Sonnet, три прогона LLM-судьи, перевод.

## Окружение: принятые решения

- **Python 3.11 через uv.** На машине был только 3.13/3.14; uv поставлен в локальный `.tools/` через pip,
  Python 3.11 скачан uv.
- **torch 2.2.2 + numpy<2.** Это последняя сборка torch для Intel-macOS (x86_64). На Apple Silicon/Linux
  ограничение можно снять.
- **LightGBM без Homebrew.** Колесо LightGBM ищет `libomp` из Homebrew. `scripts/fix_lightgbm_macos.sh`
  копирует `libomp`, который уже лежит в колесе scikit-learn, и перенаправляет ссылку
  (`install_name_tool`). Меняет только `.venv`.
- **Сертификат GigaChat** (если вернуть `provider: gigachat`): в прототипе `verify_ssl_certs: false` (config.yaml). Для продакшена — установить
  корневой сертификат Минцифры и указать путь в `llm.ca_bundle_file`.
- **Ключ GigaChat:** `GIGACHAT_CREDENTIALS` — authorization key (base64 от `ClientID:ClientSecret`). Если в `.env`
  лежит сырая пара `ClientID:ClientSecret`, код сам кодирует её в base64. Ключ нигде не логируется.
- **natasha** установилась — имена маскируются NER (тип PER), остальное — регулярками.
- **Banking77:** `load_dataset("PolyAI/banking77")` падает (скрипт загрузки больше не поддерживается) —
  берём CSV из GitHub PolyAI (`task-specific-datasets/banking_data`). 10 003 train + 3 080 test.
- **Bitext:** взят банковский вариант `bitext/Bitext-retail-banking-llm-chatbot-training-dataset`
  (25 545 пар, 26 интентов). Ответы там — длинные пошаговые инструкции; наши шаблоны сознательно короче
  и дают «результат, а не инструкцию».
- **Twitter-набор (P2)** пропущен: нет `~/.kaggle/kaggle.json`.

## LLM: кэш, логи, параллельность

`src/pipeline/llm.py`. Все вызовы кэшируются (diskcache, ключ — хэш запроса). Каждый вызов пишется в
`logs/llm_calls.jsonl`: текст сообщений, ответ, usage, стоимость, латентность и `trace`-метка кейса. По этому
логу red-team проверяет, что ПДн не уходили в LLM, — даже при параллельных запросах. Перед отправкой
user- и assistant-сообщения ещё раз прогоняются через маскирование (страховка). Массовые прогоны (бейзлайн A,
судья, red-team) идут в 8 потоков; прототип на E2E — последовательно, чтобы не искажать замер латентности CPU-этапов.

## Синтетический DWH: драйверы и шум

`src/data/synth_dwh.py`, seed 42, время мира фиксировано: **4 октября 2026, 14:00** (все сессии
открываются в этот момент). 2 000 клиентов + 6 сюжетных для демо (C90001–C90006).

Для каждой сессии выбирается истинный интент (приор: отклонённая оплата 12%, «другое» 16%, …). Дальше:

| Интент | Заложенный драйвер |
|---|---|
| declined_card_payment | отклонённая оплата картой за последние 15 мин (причины: лимит 40%, нет денег 30%, магазин 10%, блок 10%, PIN 10%); при лимите — потрачено почти всё, 50% — открыт экран «Лимиты» |
| declined_transfer | отклонённый перевод за 30 мин |
| card_payment_fee_charged | операция с комиссией за 2 дня (банкомат другого банка / кошелёк / конвертация) |
| card_payment_not_recognised | операция с «непонятным» дескриптором за 3 дня + экраны «История операций», «Детали операции» |
| transaction_charged_twice | две операции одной суммы в одном магазине с разницей в минуты (60% — вторая холд, 40% — проведена) |
| extra_charge_on_statement | служебное списание банка за 7 дней (SMS, обслуживание, подписка) |
| pending_transfer | перевод в обработке моложе суток, срок зачисления в будущем |
| transfer_not_received_by_recipient | перевод старше суток: 60% просрочен (pending), 40% выполнен, ждём банк получателя |
| pending_card_payment | холд по карте старше суток |
| card_arrival | новая карта в доставке + экран «Карта» |
| Refund_not_showing_up | возврат от магазина в статусе pending (30% — просрочен) |
| activate_my_card | карта доставлена, но не активирована |
| change_pin | экран «Настройки карты» 1–3 раза; 30% — отказ в банкомате из-за PIN |
| receiving_money | экран «Реквизиты» открыт 2–4 раза за 30 мин |
| edit_personal_details | экраны «Профиль» / «Личные данные» |
| compromised_card | 2–5 отклонённых зарубежных операций «подозрение на мошенничество» за 2 часа |
| lost_or_stolen_card | экран «Заблокировать карту»; 60% — карта заблокирована клиентом за последний час |

**Шум:**
- 25% сессий — драйвер не закладывается (клиент пришёл «не с очевидным» интентом);
- 16% сессий — интент «другое» (никакого драйвера);
- итого без сигнала **37%** сессий;
- 10% — дополнительно закладывается драйвер **другого** интента (ложный след);
- у всех — фоновые декои вне окна: старый отказ 1–6 дней назад (12%), свежий холд (10%), старая комиссия
  (8%), однократный экран «Реквизиты» (15%), «Лимиты» (8%), завершённый возврат (6%).

**Найденный и исправленный баг генератора:** признак `tx_30d` (число операций за 30 дней) сначала оказался
«главным драйвером» для 5 интентов — закладка драйвера добавляет транзакцию, и счётчик протекал меткой.
Признак убран; после этого топ-драйвер каждого из 17 интентов совпадает с заложенным.

**Важно для документа:** предиктор проверен только на синтетике, где драйверы заложены нами. Цифры
показывают, что модель умеет их находить, но не то, насколько такие драйверы есть в реальных данных банка.

## Структура

```
src/data/       загрузка, перевод, синтетический DWH, e2e-набор
src/models/     intent.py (классификаторы), keywords.py (бейзлайн), predictor.py (подсказки),
                fusion.py (DWH-априор: слияние классификатора с предиктором)
src/pipeline/   pii, normalize (транслит), router, templates, llm, guardrails, emotion, dwh, assistant
src/eval/       e2e (метрики, судья, каппа), redteam, baselines (A, B), factcheck, stats, latency,
                figures, make_report
app/            streamlit_app.py (демо), label.py (разметка)
data/templates.yaml   шаблоны ответов — на проверку эксперту поддержки
reports/        gate3_metrics.md, figures/, step*.json
```
