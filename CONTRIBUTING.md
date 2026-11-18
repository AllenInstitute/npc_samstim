# Contributing

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

## Environment setup

1. Fork and clone the repository:
```bash
git clone https://github.com/AllenInstitute/npc_samstim
cd npc_samstim
```

2. Create a new virtual environment:
- conda is convenient for getting a specific Python version, but adds additional packages
- it's preferable to use a completely clean environment to properly test the project's specified dependencies in isolation
- where possible, use the lowest supported Python version (specified in `pyproject.toml` `project/requires-python`) 
```bash
python3 -m venv .venv
```

3. Activate the environment:
- Windows
  ```bash
  .venv\scripts\activate
  ```

- Unix
  ```bash
  source .venv/bin/scripts/activate
  ```

4. Install [uv](https://docs.astral.sh/uv/) to manage the project's dependencies and run development tasks:
```bash
uv sync
```

You now have an editable pip install of the project, with all dev dependencies.
The following should work:
```bash
python -c "import npc_samstim; print(npc_samstim.__version__)"
```

### Using uv

The project uses [uv](https://docs.astral.sh/uv/) for reproducible dev environments, with pre-defined `pyproject.toml` configuration for tools.
While working on the project, use uv to manage dependencies:
- add dependencies: `uv add numpy pandas`
  - add development dependencies: `uv add --group testing mypy`
- remove dependencies: `uv remove numpy`
- update the environment and lockfile: `uv lock`
Always commit `uv.lock` to share the up-to-date development environment.


## Development (internal contributors)

1. Edit the code and/or the documentation on the main branch

2. Add simple doctests to functions or more elaborate tests to modules in `tests`

3. If you updated the project's dependencies (or you pulled changes):
  - run `uv lock`
  - if it fails due to dependencies you added, follow any error messages to resolve dependency version conflicts
  - when it doesn't fail, commit any changes to `uv.lock` along with the changes to `pyproject.toml`

4. Run tests with `uv run task test`
  - mypy will check all functions that contain type annotations in their signature
  - pytest will run doctests and any tests in the `tests` dir

5. If you updated the documentation or the project dependencies:
  - run `uv run task docs`
  - go to http://localhost:8000 and check that everything looks good
 
- if you are unsure about how to fix a test, just push your changes - the continuous integration will fail on Github and someone else can have a look

- don't update the changelog, it will be taken care of automatically

- link to any related issue number in the Commit message: `Fix variable name #13` 

- pull changes with `git pull --rebase` to keep the commit history easy to read

## Updating from the original template
With a clean working directory, run `pipx run copier update --defaults`.

See the [uv documentation](https://docs.astral.sh/uv/) for more info.