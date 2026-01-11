"""Nox sessions using uv for package management."""

import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox


package = "fluvial_particle"
python_versions = ["3.13"]
nox.needs_version = ">= 2024.3.2"  # Version with uv support
nox.options.sessions = (
    "pre-commit",
    "ruff_check",
    "ruff_format",
    "safety",
    "mypy",
    "tests",
    "typeguard",
    "xdoctest",
    "docs-build",
)
# Use uv as the default backend for faster, more reliable package installation
nox.options.default_venv_backend = "uv"


def activate_virtualenv_in_precommit_hooks(session: nox.Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        session: The Session object.
    """
    if session.bin is None:
        return

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bindir = repr(session.bin)[1:-1]  # strip quotes
        if not ((Path("A") == Path("a") and bindir.lower() in text.lower()) or bindir in text):
            continue

        lines = text.splitlines()
        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@nox.session(name="pre-commit", python="3.13")
def precommit(session: nox.Session) -> None:
    """Lint using pre-commit (includes ruff, mypy, security checks, etc.)."""
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.install("-e", ".[dev]")
    session.install("ruff", "pre-commit", "pre-commit-hooks")
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@nox.session(python=python_versions)
def ruff_check(session: nox.Session) -> None:
    """Run ruff linter."""
    session.install("-e", ".")
    session.install("ruff")
    args = session.posargs or ["check", "src", "tests", "noxfile.py"]
    session.run("ruff", *args)


@nox.session(python=python_versions)
def ruff_format(session: nox.Session) -> None:
    """Check ruff formatting."""
    session.install("-e", ".")
    session.install("ruff")
    args = session.posargs or ["format", "--check", "src", "tests", "noxfile.py"]
    session.run("ruff", *args)


@nox.session(python="3.13")
def safety(session: nox.Session) -> None:
    """Scan dependencies for insecure packages."""
    session.install("-e", ".[dev]")
    session.install("safety")
    session.run("safety", "check", "--full-report")


@nox.session(python=python_versions)
def mypy(session: nox.Session) -> None:
    """Type-check using mypy."""
    session.install("-e", ".[dev]")
    session.install("mypy", "pytest")
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@nox.session(python=python_versions)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install("-e", ".[dev]")
    session.install("coverage[toml]", "pytest", "pygments")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@nox.session(python=python_versions)
def coverage(session: nox.Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]
    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@nox.session(python=python_versions)
def typeguard(session: nox.Session) -> None:
    """Runtime type checking using Typeguard."""
    session.install("-e", ".[dev]")
    session.install("pytest", "typeguard", "pygments")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@nox.session(python=python_versions)
def xdoctest(session: nox.Session) -> None:
    """Run examples with xdoctest."""
    session.install("-e", ".[dev]")
    session.install("xdoctest[colors]")
    args = session.posargs or ["all"]
    session.run("python", "-m", "xdoctest", package, *args)


@nox.session(name="docs-build", python="3.13")
def docs_build(session: nox.Session) -> None:
    """Build the documentation."""
    session.install("-e", ".[dev]")
    session.install("sphinx", "sphinx-click", "sphinx-rtd-theme", "myst-parser", "sphinx-autobuild")
    args = session.posargs or ["docs", "docs/_build/html"]

    build_dir = Path("docs", "_build/html")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@nox.session(python="3.13")
def docs(session: nox.Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    session.install("-e", ".[dev]")
    session.install("sphinx", "sphinx-autobuild", "sphinx-click", "sphinx-rtd-theme")
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
