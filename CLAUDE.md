Use the `bd` tool for task tracking from now on. Run `bd quickstart` before doing anything else to get a handle of things.

Be explicit with dependency tracking, and be thorough with your issue descriptions and reasons when closing an issue

Always uv-managed python, e.g. `uv venv --python-preference only-managed` or similar for other commands.

Never use raw pip, always use `uv run` or `uv build`. In the worst case, `uv pip`.

Always use `uv sync` instead of `uv pip install .` Use `--editable` if appropriate.

Always prefer using `--frozen` and `--locked` when possible, e.g. to `uv run` or `uv sync`.

If you need pip, it should either be a declared dependency so `uv run` works or it can be seeded into the venv with `uv venv --seed`.
