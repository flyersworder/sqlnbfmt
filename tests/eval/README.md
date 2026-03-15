# Eval Test Suite

Fixture-based evaluation tests for `sqlnbfmt`. Each test case is a directory containing paired notebooks:

```
tests/eval/
├── basic_select/
│   ├── input.ipynb      # Unformatted notebook
│   └── expected.ipynb   # Expected formatter output
├── joins/
│   ├── input.ipynb
│   └── expected.ipynb
└── ...
```

## How it works

`test_eval.py` auto-discovers all `tests/eval/*/` directories. For each case it:

1. Copies `input.ipynb` to a temp directory
2. Runs `process_notebook()` on the copy
3. Compares the result cell-by-cell against `expected.ipynb`

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
