.PHONY: install seed serve eval eval-compare test lint up down frontend

install:
	pip install -e ".[dev,frontend]"

seed:
	python -m scripts.seed

serve:
	uvicorn curarag.api.main:app --host 0.0.0.0 --port 8000 --reload

eval:
	python -m eval.runner

eval-compare:
	python -m eval.compare_chunking

test:
	pytest -q

lint:
	ruff check src eval tests scripts

up:
	docker compose up --build -d
	@echo "Waiting for API, then seeding demo corpus..."
	docker compose run --rm api python -m scripts.seed

down:
	docker compose down

frontend:
	docker compose --profile frontend up --build -d
