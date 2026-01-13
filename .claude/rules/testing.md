# Testing and Code Quality

## Running Tests

```bash
uv run pytest
uv run pytest -v                    # verbose
uv run pytest tests/test_main.py    # specific file
uv run pytest -k "test_name"        # by name pattern
```

## Code Quality Checks

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/

# Pre-commit hooks (runs all checks)
uv run pre-commit run --all-files
```

## Test Data Location

Test fixtures are in `tests/data/`:
- `Result_straight_2d_1.vtk` - 2D test mesh (VTK format)
- `Result_straight_3d_1.vtk` - 3D test mesh (VTK format)
- `Result_2D_100.vts` - 2D test mesh (VTS format)
- `Result_3D_100.vts` - 3D test mesh (VTS format)
- `user_options_*.py` - Test configuration files
- `output_straight/` - Pre-generated test output

## Writing Tests

- Test files go in `tests/` directory
- Follow naming convention: `test_*.py`
- Use pytest fixtures from `tests/support.py`
