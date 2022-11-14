# Contributor Guide

Thank you for your interest in improving this project. This project is
open-source under the [CCO
license](https://creativecommons.org/publicdomain/zero/1.0/legalcode)
and welcomes contributions in the form of bug reports, feature requests,
and pull requests.

Here is a list of important resources for contributors:

- [Source Code](https://code.usgs.gov/wma/nhgf/fluvparticle)
- [Documentation](https://fluvial-particle.readthedocs.io/)
- [Issue Tracker](https://code.usgs.gov/wma/nhgf/fluvparticle/-/issues)
- [Code of Conduct](https://code.usgs.gov/wma/nhgf/fluvparticle/-/blob/main/CODE_OF_CONDUCT.md)

## How to report a bug

Report bugs on the [Issue Tracker](https://code.usgs.gov/wma/nhgf/fluvparticle/-/issues).

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or
steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker\_]{.title-ref}.

## How to submit changes

Open a [pull request](https://code.usgs.gov/wma/nhgf/fluvparticle/-/merge_requests)
to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation
  accordingly.

Feel free to submit early, though---we can always iterate on this.

To run linting and code formatting checks before commiting your change,
you can install pre-commit as a Git hook by running the following
command:

```shell
nox --session=pre-commit -- install
```

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate
your approach.

## Development requirements

To set up your development environment do the following.

### Fork the repository

Code edits must be made on your personal fork of
the [main repository](https://code.usgs.gov/wma/nhgf/fluvparticle/-/tree/main).
Read the [Gitlab forking workflow documentation](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html)
for forking instructions and additional information.

### Clone the forked repository

To remind yourself that you're working on a fork of the main repository,
we suggest using the name _fluvial-particle-fork_ for your local repo:

```shell
git clone git@code.usgs.gov:<user_id>/fluvparticle.git fluvial-particle-fork
```

### Setup development environment

```shell
conda env create -f environment.yml
conda develop -n fluvial-particle src
conda activate fluvial-particle
pip install -r requirements.dev
```

It is important to get [preccommit](https://pre-commit.com/) enabled on
the project, to ensure that certain standards are always met on a git
commit. With several of these, it might fail if files are changed, but
it will change them, and trying the commit a second time will actually
work.

### Git hook configuration

```shell
pre-commit install --install-hooks
```

### Testing

[Nox](https://nox.thea.codes/) is used for testing everything, with
several sessions built-in. To run the full suite of tests, simply use:

```shell
nox
```

The different sessions are:

- `pre-commit` -- validates that the
  [preccommit](https://pre-commit.com/) checks all come back clean.
- `safety` -- validates the [Safety](https://github.com/pyupio/safety)
  of all production dependencies.
- `mypy` -- validates the type-hints for the application using
  [mypy](http://mypy-lang.org/).
- `tests` -- runs all [pytest](https://docs.pytest.org/en/latest/)
  tests.
- `typeguard` -- runs all [pytest](https://docs.pytest.org/en/latest/)
  tests, validates with
  [Typeguard](https://github.com/agronholm/typeguard).
- `xdoctest` -- runs any and all documentation examples with
  [xdoctest](https://github.com/Erotemic/xdoctest).
- `docs-build` -- builds the full set of generated API docs with
  [Sphinx](http://www.sphinx-doc.org/).

These can be run individually with the following command:

```shell
nox -s <session>
```

Replace `<session>` with the name of the session give above, i.e.:

```shell
nox -s mypy
```

You can also simply run [pytest](https://docs.pytest.org/en/latest/)
tests, by using the command:

```shell
pytest tests
```

### Dependencies

Production dependencies are duplicated, in both `requirements.txt` and
`environment.yml` due to how [conda](https://www.anaconda.com) does not
work with the `requirements.txt` file. It is necessary for both files to
be updated as dependencies are added.

Development dependencies are contained in `requirements.dev`.

### Version Management

The projects made by this cookiecutter use
[Bump2version](https://github.com/c4urself/bump2version) for version
management. The default version that the project starts with is a
developmental version, `0.0.1-dev0`. In github, this should be
auto-incremented on each commit to the next dev build number. To manage
the version changes yourself, you can use the
[Bump2version](https://github.com/c4urself/bump2version) command:

```shell
bump2version <part>
```

Where `<part>` is one of:

- `major`
- `minor`
- `patch`
- `build`

Note:
: This makes a `dev` version, which does not write a tag into git. It is just useful for development purposes and not the version that is recommended for a release version. The version string will be formatted as: `<major>.<minor>.<patch>-dev<build>`

To do a production release, use the command:

```shell
bump2version --tag release
```

This will add a tag in the git repository noting the version.

Note:
: The version string for this will be: `<major>.<minor>.<patch>`
