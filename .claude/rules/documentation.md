# Documentation

## Building Locally

```bash
# Build HTML docs
uv run sphinx-build docs docs/_build/html

# Auto-rebuild on changes (dev mode)
uv run sphinx-autobuild docs docs/_build/html

# Clean previous build
rm -rf docs/_build
```

## ReadTheDocs

Documentation auto-builds on ReadTheDocs from GitHub.

**URL**: https://fluvial-particle.readthedocs.io/en/latest/

**Config files:**
- `.readthedocs.yml` - RTD build config
- `docs/conf.py` - Sphinx config
- `docs/requirements.txt` - Doc dependencies

## Key Documentation Files

- `docs/optionsfile.rst` - User options file reference
- `docs/example.rst` - Usage examples
- `docs/api.rst` - API reference
