# basedpyright gets explicit dirs: exclude = ["throwaway"] hangs it (see pyproject.toml).
default: lint typecheck test

lint:
    uv run ruff check .

typecheck:
    uv run basedpyright canvit_pytorch tests demos

test:
    uv run pytest -q
