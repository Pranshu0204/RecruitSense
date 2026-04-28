.PHONY: install install-frontend install-finetune ingest run frontend finetune evaluate test lint format docker-up docker-down clean

install:
	pip install -r backend/requirements.txt

install-frontend:
	pip install -r frontend/requirements.txt

install-finetune:
	pip install -r finetune/requirements.txt

ingest:
	python -m backend.rag.ingest

run:
	uvicorn backend.api.main:app --reload --port 8000

frontend:
	streamlit run frontend/app.py

finetune:
	python -m finetune.train

evaluate:
	python -m finetune.evaluate

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
