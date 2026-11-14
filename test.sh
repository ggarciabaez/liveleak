uv run pytest -q
uv run --no-project --with ruff ruff check --isolated tests
uv run --no-project mypy backend/backend.py
uv run scripts/smoke_test.py  # only works if the backend is active
