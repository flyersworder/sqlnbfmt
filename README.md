# sqlnbfmt

[![PyPI](https://img.shields.io/pypi/v/sqlnbfmt.svg)](https://pypi.org/project/sqlnbfmt/)
[![Tests](https://github.com/flyersworder/sqlnbfmt/actions/workflows/ci-and-publish.yml/badge.svg)](https://github.com/flyersworder/sqlnbfmt/actions/workflows/ci-and-publish.yml)
[![License](https://img.shields.io/pypi/l/sqlnbfmt.svg)](https://github.com/flyersworder/sqlnbfmt/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/sqlnbfmt.svg)](https://pypi.org/project/sqlnbfmt/)

> **Quick start:** `pip install sqlnbfmt && sqlnbfmt notebook.ipynb`

A SQL formatter for Jupyter Notebooks and Marimo notebooks. `sqlnbfmt` automatically formats SQL queries embedded in notebook cells — Python strings, SQL magic cells (`%%sql`), and Marimo `mo.sql()` calls — helping you maintain clean and consistent code.

## Features

- **Zero-config**: Works out of the box with sensible defaults — no config file needed
- **Jupyter + Marimo**: Formats SQL in `.ipynb` notebooks and Marimo `.py` notebooks
- **Smart SQL Detection**: Automatically identifies and formats SQL queries in code cells, magic SQL cells, and `mo.sql()` calls
- **AST-Powered**: Uses Abstract Syntax Tree parsing for accurate SQL string identification
- **Safe Formatting**: Preserves Python comments, query parameters (e.g., `%s`, `?`), f-string placeholders, and SQL comments
- **CI-Friendly**: `--check` mode exits non-zero when formatting is needed; `--diff` shows what would change
- **Skip Hints**: Add `# sqlnbfmt: skip` to any cell to exclude it from formatting
- **Pre-commit Ready**: Seamlessly integrates with pre-commit hooks
- **JupyterLab Plugin**: Works with `jupyterlab-code-formatter` for interactive format-on-save
- **Lightweight**: Only three runtime dependencies (sqlglot, nbformat, pyyaml)

## Installation

```bash
pip install sqlnbfmt
```

## Usage

### Command Line

Format Jupyter notebooks:
```bash
sqlnbfmt path/to/your_notebook.ipynb
```

Format Marimo notebooks (or any Python file with SQL strings):
```bash
sqlnbfmt path/to/your_notebook.py
```

Mix both in a single invocation:
```bash
sqlnbfmt *.ipynb marimo_app.py
```

Check formatting without modifying files (useful in CI):
```bash
sqlnbfmt --check path/to/your_notebook.ipynb path/to/marimo_app.py
```

Show a diff of what would change:
```bash
sqlnbfmt --diff path/to/your_notebook.ipynb
```

### Skipping Cells

Add a `# sqlnbfmt: skip` comment to skip formatting. In Jupyter notebooks this applies per-cell; in Python files it applies to the entire file.

```python
# sqlnbfmt: skip
query = "select * from my_special_table where id = 1"
```

### JupyterLab Integration

Format SQL cells interactively with [jupyterlab-code-formatter](https://github.com/jupyterlab-contrib/jupyterlab_code_formatter):

```bash
pip install sqlnbfmt jupyterlab-code-formatter
```

That's it — sqlnbfmt auto-registers as a Jupyter server extension. In JupyterLab, use **Edit > Format Cell** or configure format-on-save in Settings > Code Formatter, and select "Apply SQL Notebook Formatter".

The extension uses `[tool.sqlnbfmt]` (including `dialect`) from the nearest `pyproject.toml` of the directory you start Jupyter in. To set the dialect explicitly instead, add this to `~/.jupyter/jupyter_server_config.py`:
```python
from sqlnbfmt.jupyterlab_integration import register
register(dialect="postgres")
```

### Pre-commit Integration

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Add to `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/flyersworder/sqlnbfmt
    rev: v0.6.0
    hooks:
      - id: sqlnbfmt
        types: [jupyter]
```

To also format Marimo notebooks (Python files), add the Python hook:
```yaml
      - id: sqlnbfmt-py
        files: '\.py$'  # optionally narrow with: files: 'marimo_.*\.py$'
```

All arguments are optional. Settings in `[tool.sqlnbfmt]` in your `pyproject.toml` are picked up automatically (see [Configuration](#configuration)); alternatively pass them as arguments:
```yaml
        args: [--dialect, postgres, --config, config.yaml]
```

3. Install the hook:
```bash
pre-commit install
```

4. (Optional) Run on all files:
```bash
pre-commit run --all-files
```

### CI Usage

Use `--check` in GitHub Actions to enforce formatting:

```yaml
- name: Check SQL notebook formatting
  run: |
    pip install sqlnbfmt
    sqlnbfmt --check **/*.ipynb
    sqlnbfmt --check marimo_*.py  # if using Marimo
```

## Configuration

`sqlnbfmt` works without any configuration. To override defaults, use a `[tool.sqlnbfmt]` table in `pyproject.toml` or a separate YAML file.

### pyproject.toml

```toml
[tool.sqlnbfmt]
dialect = "postgres"
extend_function_names = ["run_query"]  # recognized in addition to the defaults

[tool.sqlnbfmt.formatting_options]
indent_width = 2
```

`sqlnbfmt` searches the current directory and its parents for the nearest `pyproject.toml` with a `[tool.sqlnbfmt]` table, stopping at the repository root (the directory containing `.git`). A `pyproject.toml` without that table is skipped, so a sub-package's file doesn't hide the repository-level configuration. The JupyterLab integration does the same, starting from the Jupyter server's working directory.

### YAML file

Pass a YAML file with `--config config.yaml`. [Here](https://github.com/flyersworder/sqlnbfmt/blob/main/config.yaml) is a template. `--config` also accepts a `.toml` path, which is read from its `[tool.sqlnbfmt]` table.

When `--config` is given, `pyproject.toml` discovery is skipped. Options you don't set keep their defaults, and unknown option names produce a warning.

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `dialect` | SQL dialect passed to sqlglot (e.g. `postgres`, `mysql`, `bigquery`); `--dialect` overrides it | none (generic SQL) |
| `sql_keywords` | SQL keywords to recognize and format | Common SQL keywords |
| `function_names` | Python functions containing SQL code (see [matching](#how-function-and-decorator-names-match)) | `read_sql`, `execute`, `sql` (Marimo's `mo.sql`), etc. |
| `sql_decorators` | Decorators whose SQL is formatted: string arguments of `@query("SELECT ...")`, and strings returned by a function decorated with `@query` | `query`, `db_query`, `sql_query`, `sql_decorator` |
| `extend_sql_keywords`, `extend_function_names`, `extend_sql_decorators` | Entries added to the corresponding list instead of replacing it | none |
| `formatting_options.single_line_threshold` | **Deprecated:** never had any effect and will be removed. Line breaks are decided by sqlglot | |
| `formatting_options.indent_width` | Number of spaces for indentation | 4 |

To add to the built-in lists, use the `extend_*` options. `sql_keywords`, `function_names` and `sql_decorators` **replace** the built-in lists, so setting `function_names = ["run_query"]` drops `sql`, and Marimo's `mo.sql()` calls stop being formatted. If you do replace a list, copy the defaults you still rely on from the [template](https://github.com/flyersworder/sqlnbfmt/blob/main/config.yaml). You can set both: the `extend_*` entries are added to your replacement list.

### How function and decorator names match

A call counts as SQL-bearing when a configured name appears **anywhere** in its full dotted name. Matching is case-sensitive. This is deliberately permissive, so that wrappers and variants are covered without configuration:

- `execute` matches `cursor.execute`, `cursor.executemany` and `execute_sql`
- `sql` matches `mo.sql`, `spark.sql`, pandasql's `sqldf`/`pysqldf` and `sqlalchemy.text`
- `sql` does **not** match `runSQLQuery`, because matching is case-sensitive; add `SQL` if you need it

Decorator names in `sql_decorators` match the same way, against the decorator's dotted name (`query` matches `@query`, `@db_query` and `@app.query(...)`).

A matched call's string arguments are only reformatted if they look like SQL. Still, prefer specific names: `run` would also match `rerun(...)` and `subprocess.run(...)`, so use `run_query` rather than `run`.

## Examples

### Jupyter Notebook

Before formatting:
```python
execute_sql("""SELECT a.col1, b.col2 FROM table_a a JOIN table_b b ON a.id = b.a_id WHERE a.status = 'active' ORDER BY a.created_at DESC""")
```

After formatting:
```python
execute_sql(
    """
    SELECT
      a.col1,
      b.col2
    FROM table_a AS a
    JOIN table_b AS b
      ON a.id = b.a_id
    WHERE
      a.status = 'active'
    ORDER BY
      a.created_at DESC
    """
)
```

### Marimo Notebook

Before formatting:
```python
@app.cell
def _(mo):
    _df = mo.sql(f"select id, name from users where active = 1 order by name")
    return
```

After formatting:
```python
@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        SELECT
          id,
          name
        FROM users
        WHERE
          active = 1
        ORDER BY
          name
        """
    )
    return
```

## Troubleshooting

**SQL not being formatted?**
- Ensure the string contains at least 2 SQL keywords or a recognizable pattern like `SELECT...FROM`
- Check that the function name is in the recognized list (add custom ones via `extend_function_names` in `[tool.sqlnbfmt]` or your `--config` file)

**Comments being modified?**
- Python comments (`#`) are preserved. SQL comments (`--`) inside strings are converted to `/* */` block comments by sqlglot.

**Pre-commit hook fails?**
- Make sure the `rev` matches the installed version
- Run `pre-commit autoupdate` to get the latest version

## Contributing

We welcome contributions! Here's how to get started:

1. Clone the repository:
```bash
git clone https://github.com/flyersworder/sqlnbfmt.git
cd sqlnbfmt
```

2. Use `uv` to sync the environment:
```bash
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Run tests:
```bash
pytest
```

   Before a release, also run the live end-to-end tests. They build the wheel, install it into a fresh virtualenv with Jupyter Server and jupyterlab-code-formatter, and exercise the CLI, the pre-commit hooks and the Jupyter extension against a scratch project. They need `uv`, `git` and network access, so the default `pytest` run skips them:
```bash
uv run pytest -m live
```

4. Add eval cases: see `tests/eval/generate_fixtures.py` for examples. Run `python tests/eval/generate_fixtures.py` to regenerate fixtures.

5. Install dev pre-commit hooks:
```bash
pre-commit install
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [sqlglot](https://github.com/tobymao/sqlglot) - SQL parsing and formatting engine
- All contributors and early adopters who helped shape this tool

---
Made with ♥️ by the sqlnbfmt team
