# Version Management

Uses `bump-my-version` to update version across files:
- `pyproject.toml`
- `src/fluvial_particle/__init__.py`
- `.bumpversion.cfg`
- `code.json`
- `meta.yaml`

## Bumping Versions

```bash
# Patch: 0.0.3 -> 0.0.4
uv run bump-my-version bump patch

# Minor: 0.0.4 -> 0.1.0
uv run bump-my-version bump minor

# Major: 0.1.0 -> 1.0.0
uv run bump-my-version bump major
```

**Note**: Automatically commits changes and creates git tags.
