# Eval Test Suite

Fixture-based evaluation tests for `sqlnbfmt`. Each test case is a directory containing paired files:

```
tests/eval/
├── basic_select/
│   ├── input.ipynb      # Unformatted Jupyter notebook
│   └── expected.ipynb   # Expected formatter output
├── marimo_basic_sql/
│   ├── input.py         # Unformatted Marimo/Python file
│   └── expected.py      # Expected formatter output
└── ...
```

## How it works

`test_eval.py` auto-discovers all `tests/eval/*/` directories. For each case it:

- **Jupyter cases** (`input.ipynb` + `expected.ipynb`): copies `input.ipynb` to a temp directory, runs `process_notebook()`, compares cell-by-cell against `expected.ipynb`
- **Marimo/Python cases** (`input.py` + `expected.py`): copies `input.py` to a temp directory, runs `process_python_file()`, compares full file content against `expected.py`

## Adding a new eval case

1. Edit `generate_fixtures.py` — add a function that returns an input notebook:

```python
def my_new_case():
    """Description of what this tests."""
    nb = nbf.new_notebook()
    nb.cells = [
        nbf.new_code_cell('query = "select * from my_table"'),
    ]
    return nb
```

2. Register it in the `CASES` dict:

```python
CASES = {
    ...
    "my_new_case": my_new_case,
}
```

3. Regenerate fixtures:

```bash
python tests/eval/generate_fixtures.py
```

4. Verify:

```bash
uv run pytest tests/test_eval.py -v -k my_new_case
```

## Regenerating all fixtures

If the formatter behavior changes (e.g., new sqlglot version), regenerate all expected outputs:

```bash
python tests/eval/generate_fixtures.py
uv run pytest tests/test_eval.py -v
```

## Test cases

### SQL Pattern Breadth
| Case | Description |
|------|-------------|
| `basic_select` | SELECT/WHERE/ORDER BY |
| `joins` | INNER JOIN, LEFT JOIN |
| `cte` | WITH ... AS common table expressions |
| `subquery` | Nested SELECT in WHERE/FROM |
| `window_function` | ROW_NUMBER OVER (PARTITION BY ...) |
| `case_expression` | CASE WHEN...THEN...ELSE...END |
| `union` | UNION ALL between SELECTs |
| `group_by_having` | GROUP BY + HAVING + aggregates |

### Notebook Scenarios
| Case | Description |
|------|-------------|
| `comment_preservation` | Python comments survive formatting |
| `multi_cell_mixed` | Only SQL cells modified; markdown/Python untouched |
| `magic_commands` | `%sql` (line) and `%%sql` (cell) magic |
| `fstring_variables` | F-string placeholders preserved |
| `in_function_sql` | SQL inside `pd.read_sql()`, `execute()` |
| `idempotency` | Already-formatted input unchanged |
| `non_sql_unchanged` | Pure Python cells untouched |

### Marimo Scenarios
| Case | Description |
|------|-------------|
| `marimo_basic_sql` | `mo.sql()` with simple SELECT |
| `marimo_fstring_interpolation` | `mo.sql()` with f-string `{expr}` placeholders preserved |
| `marimo_complex_sql` | CTE + JOIN + GROUP BY + window function in `mo.sql()` |
| `marimo_multi_cell_mixed` | SQL and non-SQL cells mixed; only SQL cells touched |
| `marimo_comment_preservation` | Python comments in cells survive formatting |
| `marimo_multi_statement` | Multiple SQL statements in one `mo.sql()` call |
| `marimo_skip_hint` | `# sqlnbfmt: skip` prevents formatting of entire file |
| `marimo_non_sql_unchanged` | Pure Python cells produce no changes |
| `marimo_idempotency` | Already-formatted output unchanged on re-run |
