import logging
import subprocess

import pytest

from sqlnbfmt.formatter import (
    _format_python_source,
    diff_python_file,
    load_config,
    process_python_file,
)


@pytest.fixture
def logger():
    log = logging.getLogger("test_marimo")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(handler)
    return log


# ---------------------------------------------------------------------------
# _format_python_source unit tests
# ---------------------------------------------------------------------------


def test_format_mo_sql_basic(logger):
    """mo.sql() with a simple SELECT is formatted."""
    source = """import marimo as mo
_df = mo.sql(f"select id, name from users where active = 1")
"""
    config = load_config()
    result, changed = _format_python_source(source, config, "mysql", logger)
    assert changed
    assert "SELECT" in result
    assert "FROM users" in result


def test_format_mo_sql_fstring_placeholder(logger):
    """mo.sql() with f-string {expr} placeholders preserved."""
    source = """import marimo as mo
table = "users"
_df = mo.sql(f"select * from {table} where active = 1")
"""
    config = load_config()
    result, changed = _format_python_source(source, config, "mysql", logger)
    assert changed
    assert "{table}" in result
    assert "SELECT" in result


def test_format_non_sql_unchanged(logger):
    """Non-SQL Python code is not modified."""
    source = """x = 1
y = "hello world"
print(x + y)
"""
    config = load_config()
    result, changed = _format_python_source(source, config, "mysql", logger)
    assert not changed
    assert result == source


def test_format_idempotent(logger):
    """Formatting already-formatted output produces no changes."""
    source = """import marimo as mo
_df = mo.sql(f"select id, name from users where active = 1")
"""
    config = load_config()
    result1, changed1 = _format_python_source(source, config, "mysql", logger)
    assert changed1
    result2, changed2 = _format_python_source(result1, config, "mysql", logger)
    assert not changed2
    assert result2 == result1


def test_format_idempotent_fstring(logger):
    """Idempotency holds for f-strings with placeholders."""
    source = """import marimo as mo
table = "users"
_df = mo.sql(f"select * from {table} where active = 1")
"""
    config = load_config()
    result1, _ = _format_python_source(source, config, "mysql", logger)
    result2, changed2 = _format_python_source(result1, config, "mysql", logger)
    assert not changed2


def test_format_skip_hint(logger):
    """File with skip hint is not modified."""
    source = """# sqlnbfmt: skip
import marimo as mo
_df = mo.sql(f"select id, name from users where active = 1")
"""
    config = load_config()
    result, changed = _format_python_source(source, config, "mysql", logger)
    assert not changed


def test_format_empty_source(logger):
    """Empty source returns unchanged."""
    config = load_config()
    result, changed = _format_python_source("", config, "mysql", logger)
    assert not changed
    assert result == ""


def test_format_syntax_error(logger):
    """Invalid Python source returns unchanged."""
    source = "def foo(:\n"
    config = load_config()
    result, changed = _format_python_source(source, config, "mysql", logger)
    assert not changed
    assert result == source


def test_format_plain_python_with_sql(logger):
    """Non-Marimo Python file with cursor.execute() is also formatted."""
    source = """import sqlite3
conn = sqlite3.connect("db.sqlite")
cursor = conn.cursor()
cursor.execute("select id, name from users where active = 1")
"""
    config = load_config()
    result, changed = _format_python_source(source, config, "mysql", logger)
    assert changed
    assert "SELECT" in result
    assert "FROM users" in result


# ---------------------------------------------------------------------------
# process_python_file / diff_python_file tests
# ---------------------------------------------------------------------------


def test_process_python_file(tmp_path, logger):
    """process_python_file formats and writes back."""
    source = """import marimo as mo
_df = mo.sql(f"select id, name from users where active = 1")
"""
    py_path = tmp_path / "notebook.py"
    py_path.write_text(source)

    config = load_config()
    changed = process_python_file(py_path, config, "mysql", logger)
    assert changed

    result = py_path.read_text()
    assert "SELECT" in result
    assert "FROM users" in result


def test_process_python_file_check_only(tmp_path, logger):
    """check_only=True detects changes without writing."""
    source = """import marimo as mo
_df = mo.sql(f"select id, name from users where active = 1")
"""
    py_path = tmp_path / "notebook.py"
    py_path.write_text(source)

    config = load_config()
    changed = process_python_file(py_path, config, "mysql", logger, check_only=True)
    assert changed
    # File unchanged
    assert py_path.read_text() == source


def test_diff_python_file(tmp_path, logger):
    """diff_python_file returns unified diff."""
    source = """import marimo as mo
_df = mo.sql(f"select id, name from users where active = 1")
"""
    py_path = tmp_path / "notebook.py"
    py_path.write_text(source)

    config = load_config()
    diff = diff_python_file(py_path, config, "mysql", logger)
    assert diff
    assert "---" in diff
    assert "+++" in diff


def test_diff_python_file_no_changes(tmp_path, logger):
    """diff_python_file returns empty string when no changes needed."""
    source = """x = 1
y = "hello"
"""
    py_path = tmp_path / "notebook.py"
    py_path.write_text(source)

    config = load_config()
    diff = diff_python_file(py_path, config, "mysql", logger)
    assert diff == ""


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_cli_marimo_format(tmp_path):
    """CLI formats a .py file."""
    source = """import marimo as mo
_df = mo.sql(f"select id, name from users where active = 1")
"""
    py_path = tmp_path / "notebook.py"
    py_path.write_text(source)

    result = subprocess.run(
        ["sqlnbfmt", str(py_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    formatted = py_path.read_text()
    assert "SELECT" in formatted


def test_cli_marimo_check(tmp_path):
    """CLI --check on unformatted .py exits 1."""
    source = """import marimo as mo
_df = mo.sql(f"select id, name from users where active = 1")
"""
    py_path = tmp_path / "notebook.py"
    py_path.write_text(source)

    result = subprocess.run(
        ["sqlnbfmt", "--check", str(py_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_cli_marimo_diff(tmp_path):
    """CLI --diff on .py prints diff and exits 0."""
    source = """import marimo as mo
_df = mo.sql(f"select id, name from users where active = 1")
"""
    py_path = tmp_path / "notebook.py"
    py_path.write_text(source)

    result = subprocess.run(
        ["sqlnbfmt", "--diff", str(py_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "---" in result.stdout


def test_cli_mixed_files(tmp_path):
    """CLI handles .ipynb and .py files in the same invocation."""
    import nbformat
    from nbformat import v4 as nbf

    # Create .ipynb
    nb = nbf.new_notebook()
    nb.cells = [nbf.new_code_cell('query = "select id from users where active = 1"')]
    nb_path = tmp_path / "notebook.ipynb"
    nbformat.write(nb, nb_path)

    # Create .py
    py_source = """import marimo as mo
_df = mo.sql(f"select id from users where active = 1")
"""
    py_path = tmp_path / "notebook.py"
    py_path.write_text(py_source)

    result = subprocess.run(
        ["sqlnbfmt", str(nb_path), str(py_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
