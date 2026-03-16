# sqlnbfmt

[![PyPI](https://img.shields.io/pypi/v/sqlnbfmt.svg)](https://pypi.org/project/sqlnbfmt/)
[![Tests](https://github.com/flyersworder/sqlnbfmt/actions/workflows/ci-and-publish.yml/badge.svg)](https://github.com/flyersworder/sqlnbfmt/actions/workflows/ci-and-publish.yml)
[![License](https://img.shields.io/pypi/l/sqlnbfmt.svg)](https://github.com/flyersworder/sqlnbfmt/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/sqlnbfmt.svg)](https://pypi.org/project/sqlnbfmt/)

> **Quick start:** `pip install sqlnbfmt && sqlnbfmt notebook.ipynb`

A SQL formatter designed specifically for Jupyter Notebooks. `sqlnbfmt` automatically formats SQL queries embedded in code cells, including both Python strings and SQL magic cells (`%%sql`), helping you maintain clean and consistent code.

## Features

- **Zero-config**: Works out of the box with sensible defaults — no config file needed
- **Smart SQL Detection**: Automatically identifies and formats SQL queries in code cells and magic SQL cells
- **AST-Powered**: Uses Abstract Syntax Tree parsing for accurate SQL string identification
- **Safe Formatting**: Preserves Python comments, query parameters (e.g., `%s`, `?`), and SQL comments
- **CI-Friendly**: `--check` mode exits non-zero when formatting is needed; `--diff` shows what would change
- **Skip Hints**: Add `# sqlnbfmt: skip` to any cell to exclude it from formatting
- **Pre-commit Ready**: Seamlessly integrates with pre-commit hooks
- **Lightweight**: Only three runtime dependencies (sqlglot, nbformat, pyyaml)

## Installation

```bash
pip install sqlnbfmt
```

## Usage

### Command Line

Format a single notebook:
```bash
sqlnbfmt path/to/your_notebook.ipynb
```

Check formatting without modifying files (useful in CI):
```bash
sqlnbfmt --check path/to/your_notebook.ipynb
```

Show a diff of what would change:
```bash
sqlnbfmt --diff path/to/your_notebook.ipynb
```

### Skipping Cells

Add a `# sqlnbfmt: skip` comment anywhere in a cell to skip formatting:

```python
# sqlnbfmt: skip
query = "select * from my_special_table where id = 1"
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
    rev: v0.3.0
    hooks:
      - id: sqlnbfmt
        types: [jupyter]
```

All arguments are optional. To specify a dialect or custom config:
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
```

## Configuration

`sqlnbfmt` works without any configuration file. A `config.yaml` is only needed to override defaults.

Create a `config.yaml` file to customize formatting behavior. [Here](https://github.com/flyersworder/sqlnbfmt/blob/main/config.yaml) is a template.

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `sql_keywords` | SQL keywords to recognize and format | Common SQL keywords |
| `function_names` | Python functions containing SQL code | `read_sql`, `execute`, etc. |
| `sql_decorators` | Decorators indicating SQL code | `query`, `sql_query`, etc. |
| `single_line_threshold` | Maximum length before splitting SQL | 80 |
| `indent_width` | Number of spaces for indentation | 4 |

## Example

Before formatting:
```python
execute_sql("""SELECT a.col1, b.col2 FROM table_a a JOIN table_b b ON a.id = b.a_id WHERE a.status = 'active' ORDER BY a.created_at DESC""")
```

After formatting:
```python
execute_sql("""
SELECT
  a.col1,
  b.col2
FROM
  table_a AS a
JOIN
  table_b AS b
  ON a.id = b.a_id
WHERE
  a.status = 'active'
ORDER BY
  a.created_at DESC
""")
```

## Troubleshooting

**SQL not being formatted?**
- Ensure the string contains at least 2 SQL keywords or a recognizable pattern like `SELECT...FROM`
- Check that the function name is in the recognized list (use `--config` to add custom ones)

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
