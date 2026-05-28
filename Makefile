.PHONY: check fix

check:
	ruff check agents/core tests
	ruff format --check agents/core tests
	pytest

fix:
	ruff check --fix agents/core tests
	ruff format agents/core tests
