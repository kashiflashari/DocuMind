.PHONY: install dev test lint serve web docker-up docker-down eval

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

serve:
	uvicorn documind.api.main:app --reload --port 8000

web:
	cd web && npm install && npm run dev

# Run the eval harness on the bundled sample, ingesting the sample docs first.
eval:
	documind eval datasets/sample.json --version dev --ingest examples/sample_docs/*.md

docker-up:
	docker compose up --build

docker-down:
	docker compose down
