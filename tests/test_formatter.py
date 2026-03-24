import pytest
import nbformat
from nbformat import v4 as nbf
import logging

from sqlnbfmt.formatter import (
    format_cell_source,
    process_notebook,
    load_config,
    diff_notebook,
)


# Fixture for temporary notebook path
@pytest.fixture
def temp_nb_path(tmp_path):
    return tmp_path / "temp_test_notebook.ipynb"


# Fixture for logger
@pytest.fixture
def logger():
    # Set up a logger for testing
    logger = logging.getLogger("test_formatter")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# Test function with updated expected outputs
@pytest.mark.parametrize(
    "input_cells, expected_cells",
    [
        # Test Case 1: Simple SELECT Query
        (
            [
                nbf.new_code_cell(
                    """query = "select id, name from users where active = 1" """
                )
            ],
            [
                nbf.new_code_cell(
                    '''query = """\nSELECT\n  id,\n  name\nFROM users\nWHERE\n  active = 1\n""" '''
                )
            ],
        ),
        # Test Case 2: SELECT Query with F-String and Variable
        (
            [
                nbf.new_code_cell(
                    '''table_name = "users"\nquery = f"select * from {table_name}"'''
                )
            ],
            [
                nbf.new_code_cell(
                    '''table_name = "users"\nquery = f"""\nSELECT\n  *\nFROM {table_name}\n"""'''
                )
            ],
        ),
        # Test Case 3: Multi-line SELECT Query with Variables
        (
            [
                nbf.new_code_cell(
                    '''columns = "id, name"\nquery = f"select {columns} from users where active = 1"'''
                )
            ],
            [
                nbf.new_code_cell(
                    '''columns = "id, name"\nquery = f"""\nSELECT\n  {columns}\nFROM users\nWHERE\n  active = 1\n"""'''
                )
            ],
        ),
        # Test Case 4: SQL Query with Line Magic Command
        (
            [nbf.new_code_cell("""%sql select * from users where active = 1""")],
            [nbf.new_code_cell("""%sql SELECT * FROM users WHERE active = 1""")],
        ),
        # Test Case 5: SQL Query with Cell Magic Command
        (
            [nbf.new_code_cell("""%%sql\nselect * from users where active = 1""")],
            [
                nbf.new_code_cell(
                    """%%sql\nSELECT\n  *\nFROM users\nWHERE\n  active = 1"""
                )
            ],
        ),
        # Test Case 6: INSERT Query with Variables (Remains Unchanged)
        (
            [
                nbf.new_code_cell(
                    '''table_name = "users"\ncolumns = "(id, name)"\nvalues = "(1, 'Alice')"\nquery = f"insert into {table_name} {columns} values {values}"'''
                )
            ],
            [
                nbf.new_code_cell(
                    '''table_name = "users"\ncolumns = "(id, name)"\nvalues = "(1, 'Alice')"\nquery = f"insert into {table_name} {columns} values {values}"'''
                )
            ],
        ),
        # Test Case 7: UPDATE Query with Variables
        (
            [
                nbf.new_code_cell(
                    '''table_name = "users"\nquery = f"update {table_name} set active = 0 where id = 1"'''
                )
            ],
            [
                nbf.new_code_cell(
                    '''table_name = "users"\nquery = f"""\nUPDATE {table_name} SET active = 0\nWHERE\n  id = 1\n"""'''
                )
            ],
        ),
        # Test Case 8: DELETE Query
        (
            [nbf.new_code_cell("""query = "delete from users where active = 0" """)],
            [
                nbf.new_code_cell(
                    '''query = """\nDELETE FROM users\nWHERE\n  active = 0\n""" '''
                )
            ],
        ),
        # Test Case 9: Query with SQL Comments
        (
            [
                nbf.new_code_cell(
                    '''query = "-- Select active users\\nselect id, name from users where active = 1"'''
                )
            ],
            [
                nbf.new_code_cell(
                    '''query = """\n/* Select active users */\nSELECT\n  id,\n  name\nFROM users\nWHERE\n  active = 1\n"""'''
                )
            ],
        ),
        # Test Case 10: Non-SQL String (Should Remain Unchanged)
        (
            [
                nbf.new_code_cell(
                    '''message = "This is not an SQL query: select * from users;"'''
                )
            ],
            [
                nbf.new_code_cell(
                    '''message = "This is not an SQL query: select * from users;"'''
                )
            ],
        ),
        # Test Case 11: F-string without variables
        (
            [
                nbf.new_code_cell(
                    '''query = f"select id, name from users where active = 1"'''
                )
            ],
            [
                nbf.new_code_cell(
                    '''query = f"""\nSELECT\n  id,\n  name\nFROM users\nWHERE\n  active = 1\n"""'''
                )
            ],
        ),
        # Test Case 12: F-string with only string constants
        (
            [nbf.new_code_cell('''query = f"select " f"id, name " f"from users"''')],
            [
                nbf.new_code_cell(
                    '''query = f"""\nSELECT\n  id,\n  name\nFROM users\n"""'''
                )
            ],
        ),
        # Test Case 13: In-function SQL Query
        (
            [
                nbf.new_code_cell(
                    '''pd.read_sql("""select id, name from users where active = 1""")'''
                )
            ],
            [
                nbf.new_code_cell(
                    '''pd.read_sql(\n    """\n    SELECT\n      id,\n      name\n    FROM users\n    WHERE\n      active = 1\n    """\n)'''
                )
            ],
        ),
        # Test Case 14: In-function SQL Query with F-String
        (
            [
                nbf.new_code_cell(
                    """table_name = "users"\npd.read_sql(f"select * from {table_name}")"""
                )
            ],
            [
                nbf.new_code_cell(
                    '''table_name = "users"\npd.read_sql(\n    f"""\n    SELECT\n      *\n    FROM {table_name}\n    """\n)'''
                )
            ],
        ),
        # Test Case 15: Python comment preservation
        (
            [
                nbf.new_code_cell(
                    '''# This query fetches active users\nquery = "select id, name from users where active = 1"'''
                )
            ],
            [
                nbf.new_code_cell(
                    '''# This query fetches active users\nquery = """\nSELECT\n  id,\n  name\nFROM users\nWHERE\n  active = 1\n"""'''
                )
            ],
        ),
    ],
)
def test_sql_formatter(temp_nb_path, logger, input_cells, expected_cells):
    # Create a new notebook
    nb = nbf.new_notebook()
    nb.cells = input_cells

    # Write the notebook to the temporary path
    nbformat.write(nb, temp_nb_path)

    # Load configuration
    config = load_config()
    dialect = "mysql"  # or your preferred SQL dialect

    # Run the formatter on the notebook
    process_notebook(temp_nb_path, config, dialect, logger=logger)

    # Read the formatted notebook
    formatted_nb = nbformat.read(temp_nb_path, as_version=4)

    # Compare the cells
    for formatted_cell, expected_cell in zip(formatted_nb.cells, expected_cells):
        formatted_code = formatted_cell.source.strip()
        expected_code = expected_cell.source.strip()
        assert formatted_code == expected_code, (
            f"Formatted code does not match expected output.\nExpected:\n{expected_code}\nActual:\n{formatted_code}"
        )

    # Clean up
    temp_nb_path.unlink()


def test_skip_hint(tmp_path, logger):
    """Cell with # sqlnbfmt: skip should remain untouched."""
    nb = nbf.new_notebook()
    original_source = (
        '# sqlnbfmt: skip\nquery = "select id, name from users where active = 1"'
    )
    nb.cells = [nbf.new_code_cell(original_source)]
    nb_path = tmp_path / "skip_test.ipynb"
    nbformat.write(nb, nb_path)

    config = load_config()
    changed = process_notebook(nb_path, config, "mysql", logger)
    assert not changed

    result_nb = nbformat.read(nb_path, as_version=4)
    assert result_nb.cells[0].source == original_source


def test_check_only_unformatted(tmp_path, logger):
    """--check on unformatted notebook returns True but file unchanged."""
    nb = nbf.new_notebook()
    original_source = 'query = "select id, name from users where active = 1"'
    nb.cells = [nbf.new_code_cell(original_source)]
    nb_path = tmp_path / "check_test.ipynb"
    nbformat.write(nb, nb_path)

    config = load_config()
    changed = process_notebook(nb_path, config, "mysql", logger, check_only=True)
    assert changed

    # File should be unchanged
    result_nb = nbformat.read(nb_path, as_version=4)
    assert result_nb.cells[0].source == original_source


def test_check_only_formatted(tmp_path, logger):
    """--check on already-formatted notebook returns False."""
    nb = nbf.new_notebook()
    nb.cells = [
        nbf.new_code_cell(
            'query = """\nSELECT\n  id,\n  name\nFROM users\nWHERE\n  active = 1\n"""'
        )
    ]
    nb_path = tmp_path / "check_formatted.ipynb"
    nbformat.write(nb, nb_path)

    config = load_config()
    changed = process_notebook(nb_path, config, "mysql", logger, check_only=True)
    assert not changed


def test_diff_notebook_unformatted(tmp_path, logger):
    """diff_notebook returns non-empty diff for unformatted notebook."""
    nb = nbf.new_notebook()
    nb.cells = [
        nbf.new_code_cell('query = "select id, name from users where active = 1"')
    ]
    nb_path = tmp_path / "diff_test.ipynb"
    nbformat.write(nb, nb_path)

    config = load_config()
    diff_output = diff_notebook(nb_path, config, "mysql", logger)
    assert diff_output  # non-empty
    assert "---" in diff_output
    assert "+++" in diff_output


def test_diff_notebook_formatted(tmp_path, logger):
    """diff_notebook returns empty string for already-formatted notebook."""
    nb = nbf.new_notebook()
    nb.cells = [
        nbf.new_code_cell(
            'query = """\nSELECT\n  id,\n  name\nFROM users\nWHERE\n  active = 1\n"""'
        )
    ]
    nb_path = tmp_path / "diff_formatted.ipynb"
    nbformat.write(nb, nb_path)

    config = load_config()
    diff_output = diff_notebook(nb_path, config, "mysql", logger)
    assert diff_output == ""


def test_load_config_defaults():
    """load_config() without args returns defaults."""
    config = load_config()
    assert "SELECT" in config.sql_keywords
    assert "read_sql" in config.function_names
    assert config.indent_width == 4


def test_load_config_missing_file():
    """load_config with missing path raises FileNotFoundError."""
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_config_empty_file(tmp_path):
    """load_config with empty YAML file returns defaults (not AttributeError)."""
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("")
    config = load_config(empty_config)
    assert "SELECT" in config.sql_keywords
    assert "read_sql" in config.function_names


def test_load_config_partial_override(tmp_path):
    """Config file with only indent_width still gets default keywords."""
    partial_config = tmp_path / "partial.yaml"
    partial_config.write_text("formatting_options:\n  indent_width: 2\n")
    config = load_config(partial_config)
    assert config.indent_width == 2
    assert "SELECT" in config.sql_keywords
    assert "read_sql" in config.function_names


def test_skip_hint_in_string_not_triggered(tmp_path, logger):
    """Skip hint inside a string literal should NOT skip the cell."""
    nb = nbf.new_notebook()
    original_source = (
        'msg = "Use # sqlnbfmt: skip to disable"\n'
        'query = "select id, name from users where active = 1"'
    )
    nb.cells = [nbf.new_code_cell(original_source)]
    nb_path = tmp_path / "skip_in_string.ipynb"
    nbformat.write(nb, nb_path)

    config = load_config()
    changed = process_notebook(nb_path, config, "mysql", logger)
    assert changed  # SQL should still be formatted


def test_skip_hint_magic_cell(tmp_path, logger):
    """Skip hint in a magic command cell should skip formatting."""
    nb = nbf.new_notebook()
    original_source = "# sqlnbfmt: skip\n%%sql\nselect * from users where active = 1"
    nb.cells = [nbf.new_code_cell(original_source)]
    nb_path = tmp_path / "skip_magic.ipynb"
    nbformat.write(nb, nb_path)

    config = load_config()
    changed = process_notebook(nb_path, config, "mysql", logger)
    assert not changed

    result_nb = nbformat.read(nb_path, as_version=4)
    assert result_nb.cells[0].source == original_source


def test_check_diff_combined(tmp_path, logger):
    """--check --diff combined: no duplicate files, diff is printed."""
    nb = nbf.new_notebook()
    nb.cells = [
        nbf.new_code_cell('query = "select id, name from users where active = 1"')
    ]
    nb_path = tmp_path / "combined.ipynb"
    nbformat.write(nb, nb_path)

    config = load_config()
    # Simulate the combined logic from main()
    diff_output = diff_notebook(nb_path, config, "mysql", logger)
    assert diff_output  # has diff
    # In the refactored main(), diff_notebook is used as the single source
    # of truth for both --check and --diff, so no duplication possible.


def test_main_diff_only_exits_zero(tmp_path):
    """--diff alone always exits 0 even when changes are needed."""
    import subprocess

    nb = nbf.new_notebook()
    nb.cells = [
        nbf.new_code_cell('query = "select id, name from users where active = 1"')
    ]
    nb_path = tmp_path / "diff_exit.ipynb"
    nbformat.write(nb, nb_path)

    result = subprocess.run(
        ["sqlnbfmt", "--diff", str(nb_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "---" in result.stdout  # diff was printed


def test_main_check_exits_one(tmp_path):
    """--check exits 1 when formatting is needed."""
    import subprocess

    nb = nbf.new_notebook()
    nb.cells = [
        nbf.new_code_cell('query = "select id, name from users where active = 1"')
    ]
    nb_path = tmp_path / "check_exit.ipynb"
    nbformat.write(nb, nb_path)

    result = subprocess.run(
        ["sqlnbfmt", "--check", str(nb_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_main_check_diff_exits_one(tmp_path):
    """--check --diff combined exits 1 and prints diff."""
    import subprocess

    nb = nbf.new_notebook()
    nb.cells = [
        nbf.new_code_cell('query = "select id, name from users where active = 1"')
    ]
    nb_path = tmp_path / "check_diff_exit.ipynb"
    nbformat.write(nb, nb_path)

    result = subprocess.run(
        ["sqlnbfmt", "--check", "--diff", str(nb_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "---" in result.stdout  # diff was printed


# ---------------------------------------------------------------------------
# format_cell_source tests
# ---------------------------------------------------------------------------


def test_format_cell_source_magic(logger):
    """format_cell_source handles %%sql magic cells."""
    config = load_config()
    code = "%%sql\nselect id, name from users where active = 1"
    result = format_cell_source(code, config, "mysql", logger)
    assert result.startswith("%%sql\n")
    assert "SELECT" in result


def test_format_cell_source_line_magic(logger):
    """format_cell_source handles %sql line magic."""
    config = load_config()
    code = "%sql select * from users where active = 1"
    result = format_cell_source(code, config, "mysql", logger)
    assert result.startswith("%sql ")
    assert "SELECT" in result


def test_format_cell_source_python(logger):
    """format_cell_source handles regular Python code with SQL."""
    config = load_config()
    code = 'query = "select id, name from users where active = 1"'
    result = format_cell_source(code, config, "mysql", logger)
    assert "SELECT" in result


def test_format_cell_source_non_sql(logger):
    """format_cell_source returns non-SQL code unchanged."""
    config = load_config()
    code = "x = 42\nprint(x)"
    result = format_cell_source(code, config, "mysql", logger)
    assert result == code


def test_format_cell_source_skip_hint(logger):
    """format_cell_source respects skip hints."""
    config = load_config()
    code = '# sqlnbfmt: skip\nquery = "select * from users"'
    result = format_cell_source(code, config, "mysql", logger)
    assert result == code
