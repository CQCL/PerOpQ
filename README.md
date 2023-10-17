# PerOpQ

Perturbative Optimization for Hamiltonian Simulation

## Development

This project uses [Poetry](https://python-poetry.org/) for packaging and dependency management and
[Nox](https://nox.thea.codes/en/stable/) for task automation.

### Recommended development setup

It is recommended to do these steps in a fresh python virtual environment

Install development tools:

```shell
pip install -r dev-tool-requirements.txt
```

### Local development with Nox (recommended)

[Nox](https://nox.thea.codes/en/stable/) can be used to automate various development tasks running in isolated python environments.
The following Nox sessions are provided:

- `pre-commit`: run the configured pre-commit hooks within [.pre-commit-config.yaml](.pre-commit-config.yaml), this includes linting with black and pylint
- `mypy`: run type checks using mypy
- `tests`: run the unit tests
- `docs-build`: build the documentation

To run a session use:

```shell
nox -s <session_name>
```

To save time, reuse the session virtual environment using the `-r` option, i.e. `nox -rs <session_name>` (may cause errors after a dependency update).

[Pre-commit](https://pre-commit.com/) can be used to run the pre-commit hooks before each commit. This is recommended.
To set up the pre-commit hooks to run automatically on each commit run:

```shell
nox -s pre-commit -- install
```

Afterward, the [pre-configured hooks](.pre-commit-config.yaml) will run on all changed files in a commit and the commit will be
rejected if the hooks find errors. Some hooks will correct formatting issues automatically (but will still reject the commit, so that
the `git commit` command will need to be repeated).

<!-- github-only -->
