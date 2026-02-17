.PHONY: install test api retrain-worker mlops-flow mlops-monitor dvc-repro

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e .

test:
	pytest -q

api:
	uvicorn contract_ai.api.main:app --host 0.0.0.0 --port 8000 --reload

retrain-worker:
	python scripts/retrain_worker.py

mlops-flow:
	python scripts/mlops/run_prefect_flow.py

mlops-monitor:
	python scripts/mlops/summarize_monitoring.py

dvc-repro:
	dvc repro
