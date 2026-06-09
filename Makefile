.PHONY: install install-frontend install-finetune ingest run frontend finetune evaluate test lint format docker-up docker-down infra stop clean

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

# Local-dev infra only: Qdrant + Redis in Docker, NOT the app backend/frontend.
# Use this with `make run` + `make frontend` so the app always runs your local
# (uncontainerized) code and never a stale image.
infra:
	docker compose up -d qdrant redis

# Stop any local backend/frontend and free their ports. Run this if you ever see
# a stale process answering on :8000 (e.g. an old `docker compose up` backend).
stop:
	-docker compose stop backend frontend 2>/dev/null
	-pkill -9 -f "uvicorn backend.api.main" 2>/dev/null
	-pkill -9 -f "streamlit run frontend/app.py" 2>/dev/null
	-lsof -ti :8000 | xargs kill -9 2>/dev/null
	-lsof -ti :8501 | xargs kill -9 2>/dev/null
	@echo "Stopped local + containerized backend/frontend; freed ports 8000 and 8501."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
