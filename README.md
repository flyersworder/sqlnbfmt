# sqlnbfmt

[![PyPI](https://img.shields.io/pypi/v/sqlnbfmt.svg)](https://pypi.org/project/sqlnbfmt/)
[![License](https://img.shields.io/pypi/l/sqlnbfmt.svg)](https://github.com/flyersworder/sqlnbfmt/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/sqlnbfmt.svg)](https://pypi.org/project/sqlnbfmt/)

A SQL formatter designed specifically for Jupyter Notebooks. `sqlnbfmt` automatically formats SQL queries embedded in code cells, including both Python strings and SQL magic cells (`%%sql`), helping you maintain clean and consistent code.

## Features

- 🎯 **Smart SQL Detection**: Automatically identifies and formats SQL queries in code cells and magic SQL cells
- 🌳 **AST-Powered**: Uses Abstract Syntax Tree parsing for accurate SQL string identification
- 🔒 **Safe Formatting**: Preserves query parameters (e.g., `%s`, `?`) and SQL comments
- ⚙️ **Highly Configurable**: Customize formatting through YAML configuration
- 🔄 **Pre-commit Ready**: Seamlessly integrates with pre-commit hooks
- 📦 **Zero Dependencies**: Minimal installation footprint

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

Format all notebooks in a directory:
```bash
sqlnbfmt path/to/notebooks/
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
    rev: v0.1.1
    hooks:
      - id: sqlnbfmt
        name: sqlnbfmt
        types: [jupyter]
        args: [--config, config.yaml, --dialect, postgres]
```
Please fun the following command in your CMD for help:

```bash
sqlnbfmt -h
```

3. Install the hook:
```bash
pre-commit install
```

4. (Optional) Run on all files:
```bash
pre-commit run --all-files
```

## Configuration

Create a `config.yaml` file to customize formatting behavior. [Here](https://github.com/flyersworder/sqlnbfmt/blob/main/config.yaml) is a template.

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `sql_keywords` | SQL keywords to recognize and format | Common SQL keywords |
| `function_names` | Python functions containing SQL code | `[]` |
| `sql_decorators` | Decorators indicating SQL code | `[]` |
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [sqlglot](https://github.com/tobymao/sqlglot) - SQL parsing and formatting engine
- All contributors and early adopters who helped shape this tool

---
Made with ♥️ by the sqlnbfmt team
