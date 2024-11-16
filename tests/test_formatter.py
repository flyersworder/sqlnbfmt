import pytest
import nbformat
from nbformat import v4 as nbf
from pathlib import Path
import logging
import sys

# Add the parent directory to sys.path to import the formatter module
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlnbfmt.formatter import process_notebook, FormattingConfig, load_config

# Fixture for temporary notebook path
@pytest.fixture
def temp_nb_path(tmp_path):
    return tmp_path / "temp_test_notebook.ipynb"

# Fixture for logger
@pytest.fixture
def logger():
    # Set up a logger for testing
    logger = logging.getLogger('test_formatter')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Test function with updated expected outputs
@pytest.mark.parametrize("input_cells, expected_cells", [
    # Test Case 1: Simple SELECT Query
    (
        [nbf.new_code_cell("""query = "select id, name from users where active = 1" """)],
        [nbf.new_code_cell('''query = """
SELECT
  id,
  name
FROM users
WHERE
  active = 1
"""''')],
    ),
    # Test Case 2: SELECT Query with F-String and Variable
    (
        [nbf.new_code_cell('''table_name = "users"\nquery = f"select * from {table_name}"''')],
        [nbf.new_code_cell('''table_name = "users"
query = f"""
SELECT
  *
FROM {table_name}
"""''')],
    ),
    # Test Case 3: Multi-line SELECT Query with Variables
    (
        [nbf.new_code_cell('''columns = "id, name"\nquery = f"select {columns} from users where active = 1"''')],
        [nbf.new_code_cell('''columns = "id, name"
query = f"""
SELECT
  {columns}
FROM users
WHERE
  active = 1
"""''')],
    ),
    # Test Case 4: SQL Query with Line Magic Command
    (
        [nbf.new_code_cell("""%sql select * from users where active = 1""")],
        [nbf.new_code_cell("""%sql SELECT * FROM users WHERE active = 1""")],
    ),
    # Test Case 5: SQL Query with Cell Magic Command
    (
        [nbf.new_code_cell('''%%sql\nselect * from users where active = 1''')],
        [nbf.new_code_cell('''%%sql
SELECT
  *
FROM users
WHERE
  active = 1''')],
    ),
    # Test Case 6: INSERT Query with Variables (Remains Unchanged)
    (
        [nbf.new_code_cell('''table_name = "users"\ncolumns = "(id, name)"\nvalues = "(1, 'Alice')"\nquery = f"insert into {table_name} {columns} values {values}"''')],
        [nbf.new_code_cell('''table_name = "users"
columns = "(id, name)"
values = "(1, 'Alice')"
query = f"insert into {table_name} {columns} values {values}"''')],
    ),
    # Test Case 7: UPDATE Query with Variables
    (
        [nbf.new_code_cell('''table_name = "users"\nquery = f"update {table_name} set active = 0 where id = 1"''')],
        [nbf.new_code_cell('''table_name = "users"
query = f"""
UPDATE {table_name} SET active = 0
WHERE
  id = 1
"""''')],
    ),
    # Test Case 8: DELETE Query
    (
        [nbf.new_code_cell("""query = "delete from users where active = 0" """)],
        [nbf.new_code_cell('''query = """
DELETE FROM users
WHERE
  active = 0
"""''')],
    ),
    # Test Case 9: Query with Comments
    (
        [nbf.new_code_cell('''query = "-- Select active users\\nselect id, name from users where active = 1"''')],
        [nbf.new_code_cell('''query = """
/* Select active users */
SELECT
  id,
  name
FROM users
WHERE
  active = 1
"""''')],
    ),
    # Test Case 10: Non-SQL String (Should Remain Unchanged)
    (
        [nbf.new_code_cell('''message = "This is not an SQL query: select * from users;"''')],
        [nbf.new_code_cell('''message = "This is not an SQL query: select * from users;"''')],
    ),
    # Test Case 11: F-string without variables
    (
        [nbf.new_code_cell('''query = f"select id, name from users where active = 1"''')],
        [nbf.new_code_cell('''query = f"""
SELECT
  id,
  name
FROM users
WHERE
  active = 1
"""''')],
    ),
    # Test Case 12: F-string with only string constants
    (
        [nbf.new_code_cell('''query = f"select " f"id, name " f"from users"''')],
        [nbf.new_code_cell('''query = f"""
SELECT
  id,
  name
FROM users
"""''')],
    ),
    # Test Case 13: In-function SQL Query
    (
        [nbf.new_code_cell('''pd.read_sql("""select id, name from users where active = 1""")''')],
        [nbf.new_code_cell('''pd.read_sql(
    """
    SELECT
      id,
      name
    FROM users
    WHERE
      active = 1
    """
)''')],
    ),
    # Test Case 14: In-function SQL Query with F-String
    (
        [nbf.new_code_cell('''table_name = "users"\npd.read_sql(f"select * from {table_name}")''')],
        [nbf.new_code_cell('''table_name = "users"
pd.read_sql(
    f"""
    SELECT
      *
    FROM {table_name}
    """
)''')],
    ),
])
def test_sql_formatter(temp_nb_path, logger, input_cells, expected_cells):
    # Create a new notebook
    nb = nbf.new_notebook()
    nb.cells = input_cells

    # Write the notebook to the temporary path
    nbformat.write(nb, temp_nb_path)

    # Load configuration
    config = load_config('config.yaml')
    dialect = 'mysql'  # or your preferred SQL dialect

    # Run the formatter on the notebook
    process_notebook(temp_nb_path, config, dialect, logger=logger)

    # Read the formatted notebook
    formatted_nb = nbformat.read(temp_nb_path, as_version=4)

    # Compare the cells
    for formatted_cell, expected_cell in zip(formatted_nb.cells, expected_cells):
        formatted_code = formatted_cell.source.strip()
        expected_code = expected_cell.source.strip()
        assert formatted_code == expected_code, f"Formatted code does not match expected output.\nExpected:\n{expected_code}\nActual:\n{formatted_code}"

    # Clean up
    temp_nb_path.unlink()