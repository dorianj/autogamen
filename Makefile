test:
	@echo "→ running tests"
	@uv run python -m unittest test.test_board

lint:
	@echo "→ running ruff check"
	@uv run ruff check .
	@echo "→ running mypy type check"
	@uv run mypy --explicit-package-bases autogamen/

lint-fix:
	@echo "→ running ruff check --fix"
	@uv run ruff check --fix .
	@echo "→ running mypy type check"
	@uv run mypy --explicit-package-bases autogamen/

typecheck:
	@echo "→ running mypy type check"
	@uv run mypy --explicit-package-bases autogamen/

.PHONY: test lint lint-fix typecheck
