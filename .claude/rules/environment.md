# Environment Management

This project uses **uv** for dependency and environment management.

## Running Commands

**CRITICAL**: All commands should be run using `uv run`:

```bash
uv run <command>

# Examples
uv run pytest
uv run ruff check .
uv run python script.py
```

## Initial Setup

```bash
uv sync --all-extras
```

## Adding Dependencies

### Runtime dependencies:
1. Add to `pyproject.toml` under `dependencies`
2. Run: `uv sync --all-extras`

### Development dependencies:
1. Add to `pyproject.toml` under `[project.optional-dependencies]`
2. Run: `uv sync --all-extras`

## Updating Dependencies

```bash
# Update all
uv sync --upgrade --all-extras

# Update specific package
uv add --upgrade <package-name>
```

## Troubleshooting

### "Command not found" errors
Use `uv run <command>` or activate: `source .venv/bin/activate`

### Import errors after adding dependencies
Run `uv sync --all-extras`

### Stale environment
```bash
rm -rf .venv
uv sync --all-extras
```
