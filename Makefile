.PHONY: bootstrap lint format format-check test run

bootstrap:
	./scripts/dev/bootstrap_venv.sh

lint:
	.venv/bin/ruff check src tests

format:
	.venv/bin/black src tests

format-check:
	.venv/bin/black --check src tests

test:
	.venv/bin/pytest

run:
	.venv/bin/python -m src.main
