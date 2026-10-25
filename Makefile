# Короткие команды. Python-окружение — .venv (uv sync).
PY = .venv/bin/python
export TOKENIZERS_PARALLELISM = false

demo:            ## демо в браузере
	.venv/bin/streamlit run app/streamlit_app.py

label:           ## страница human eval
	.venv/bin/streamlit run app/label.py

label-holdout:   ## разметка 30 кейсов отложенного набора × 3 системы (~1 час) → каппа судьи
	.venv/bin/streamlit run app/label.py -- --set e2e_holdout --sample 30

test:
	$(PY) -m pytest -q

data:            ## шаг 0 + синтетика (без LLM)
	$(PY) -m src.data.download
	$(PY) -m src.data.synth_dwh

train:           ## шаги 1–3 (без LLM; RU-метрики — если есть перевод шага 2)
	$(PY) -m src.models.intent train
	$(PY) -m src.models.predictor train
	$(PY) -m src.models.fusion tune

estimate:        ## оценка токенов перед платными прогонами
	$(PY) -m src.data.translate
	$(PY) -m src.eval.e2e estimate

report:          ## пересобрать метрики и reports/gate3_metrics.md (без LLM, судья берётся из файлов)
	$(PY) -m src.eval.e2e report > /dev/null
	$(PY) -m src.eval.e2e report --set e2e_holdout > /dev/null
	$(PY) -m src.eval.redteam report > /dev/null
	$(PY) -m src.eval.make_report

.PHONY: demo label label-holdout test data train estimate report
