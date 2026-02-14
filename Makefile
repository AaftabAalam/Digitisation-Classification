.PHONY: install test api retrain-worker

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e .

test:
	pytest -q

api:
	uvicorn contract_ai.api.main:app --host 0.0.0.0 --port 8000 --reload

retrain-worker:
	python scripts/retrain_worker.py
