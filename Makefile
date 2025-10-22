lint:
	@echo "→ running ruff check"
	@uv run ruff check .
	@echo "→ running mypy type check"
	@uv run mypy autogamen/

lint-fix:
	@echo "→ running ruff check --fix"
	@uv run ruff check --fix .
	@echo "→ running mypy type check"
	@uv run mypy autogamen/

typecheck:
	@echo "→ running mypy type check"
	@uv run mypy autogamen/
