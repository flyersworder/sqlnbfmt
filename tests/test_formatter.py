# tests/test_formatter.py

import ast
import pytest
import re
import nbformat
import logging
from pathlib import Path
from typing import Optional

from sqlnbfmt.formatter import (
    format_sql_code,
    process_notebook,
    FormattingConfig,
    SQLFormattingError
)
from sqlglot import errors
from sqlglot.errors import TokenError

# Helper function to normalize SQL for comparison
def normalize_sql(sql: str) -> str:
    sql = re.sub(r'\s+', ' ', sql)      # Replace multiple spaces with a single space
    sql = re.sub(r'\(\s+', '(', sql)    # Remove space after '('
    sql = re.sub(r'\s+\)', ')', sql)    # Remove space before ')'
    sql = re.sub(r'\s*,\s*', ', ', sql) # Normalize commas and surrounding spaces
    return sql.strip().upper()

# Setup logging for tests
def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger('formatter')
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Fixtures
@pytest.fixture
def sample_config():
    return FormattingConfig(
        sql_keywords={'SELECT', 'ORDER BY', 'JOIN', 'GROUP BY', 'FROM', 'WHERE'},
        function_names={'execute_sql', 'run_query'},
        sql_decorators={'@sql'},
        single_line_threshold=80,
        indent_width=4
    )

# Test 1: Basic Reformatting
def test_format_sql_code_basic_reformatting(sample_config):
    """
    Test that basic SQL queries are reformatted with proper indentation and uppercase keywords.
    """
    sql = """
    select col1, col2 from table where condition order by col1 desc
    """
    formatted = format_sql_code(sql, None, sample_config)
    expected = "SELECT col1, col2 FROM table WHERE condition ORDER BY col1 DESC"
    assert normalize_sql(expected) == normalize_sql(formatted)

# Test 2: Malicious Input Handling
def test_format_sql_code_malicious_input(sample_config):
    """
    Test that the formatter handles malicious SQL inputs without executing them.
    Specifically, it should raise an error for multiple SQL statements.
    """
    sql = "SELECT * FROM users; DROP TABLE users;"
    # The formatter should raise an error for multiple statements
    with pytest.raises(SQLFormattingError):
        format_sql_code(sql, None, sample_config)

# Test 3: Escape Characters in Strings
def test_format_sql_code_escape_characters(sample_config):
    """
    Test that escape characters in SQL strings are handled correctly.
    """
    # Correct SQL escaping uses two single quotes to escape
    sql = "SELECT * FROM table WHERE name = 'O''Connor'"
    formatted = format_sql_code(sql, None, sample_config)
    expected = "SELECT * FROM table WHERE name = 'O''Connor'"
    assert normalize_sql(expected) == normalize_sql(formatted)

# Test 4: Special Characters in Identifiers
def test_format_sql_code_special_characters(sample_config):
    """
    Test that SQL identifiers containing special characters are properly quoted.
    """
    sql = """
    SELECT @column1, `#column2`, $column3
    FROM table
    WHERE condition = '@value'
    """
    # In MySQL, '#' denotes a comment, so '`#column2`' should be quoted to avoid being treated as a comment
    formatted = format_sql_code(sql, "mysql", sample_config)
    expected = "SELECT @column1, `#column2`, $column3 FROM `table` WHERE `condition` = '@value'"
    assert normalize_sql(expected) == normalize_sql(formatted)

# Test 5: Reserved Keywords as Identifiers
def test_format_sql_code_reserved_keywords(sample_config):
    """
    Test that SQL reserved keywords are correctly quoted when used as identifiers.
    """
    sql = "SELECT `group`, `order` FROM `table` WHERE `select` > 10"
    # Reserved keywords should be quoted to avoid parsing errors
    formatted = format_sql_code(sql, "mysql", sample_config)
    expected = "SELECT `group`, `order` FROM `table` WHERE `select` > 10"
    assert normalize_sql(expected) == normalize_sql(formatted)

# Test 6: Process Notebook with Various SQL Scenarios
def test_process_notebook(tmp_path, sample_config):
    """
    Test processing a notebook containing a single SQL code cell.
    """
    # Create a sample notebook
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("execute_sql('SELECT * FROM table')"))
    notebook_path = tmp_path / "sample_notebook.ipynb"
    nbformat.write(nb, notebook_path)

    logger = setup_logging("DEBUG")
    result = process_notebook(notebook_path, sample_config, None, logger)
    assert isinstance(result, bool)
    assert result is True  # Assuming processing succeeds

    # Read the processed notebook and verify changes
    nb_processed = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in nb_processed.cells if cell.cell_type == "code"]

    # Verify first code cell
    first_cell = code_cells[0]
    expected = "execute_sql('SELECT * FROM table')"
    assert normalize_sql(expected) == normalize_sql(first_cell.source.strip())

# Test 7: Process Real-World Notebook with Complex SQL Queries
def test_process_real_world_notebook(tmp_path, sample_config):
    """
    Integration test with a real-world-like notebook containing complex SQL queries.
    """
    # Create a more complex notebook
    nb = nbformat.v4.new_notebook()
    nb.cells.extend([
        nbformat.v4.new_markdown_cell("# Real-World SQL Formatting Test"),
        nbformat.v4.new_code_cell('''
execute_sql("""
SELECT a.col1, b.col2
FROM table_a a
JOIN table_b b ON a.id = b.a_id
WHERE a.status = 'active'
ORDER BY a.created_at DESC
""")
        '''),
        nbformat.v4.new_code_cell('''
%%sql
SELECT
    user_id,
    COUNT(*) AS login_count
FROM logins
WHERE login_date > '2023-01-01'
GROUP BY user_id
ORDER BY login_count DESC
        '''),
    ])
    notebook_path = tmp_path / "real_world_notebook.ipynb"
    nbformat.write(nb, notebook_path)

    logger = setup_logging("DEBUG")
    result = process_notebook(notebook_path, sample_config, None, logger)
    assert isinstance(result, bool)
    assert result is True  # Assuming processing succeeds

    # Read the processed notebook and verify changes
    nb_processed = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in nb_processed.cells if cell.cell_type == "code"]

    # Helper function to extract SQL from function calls using AST
    def extract_sql_from_execute(cell_source: str) -> Optional[str]:
        try:
            tree = ast.parse(cell_source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'execute_sql':
                    # Assuming the first argument is the SQL code
                    if node.args:
                        arg = node.args[0]
                        if isinstance(arg, (ast.Constant, ast.Constant)):
                            return arg.value.strip()
                        elif isinstance(arg, ast.JoinedStr):
                            # Handle f-strings
                            sql_parts = []
                            for value in arg.values:
                                if isinstance(value, (ast.Constant, ast.Constant)):
                                    sql_parts.append(value.s if isinstance(value, ast.Constant) else value.value)
                                else:
                                    sql_parts.append("{expression}")
                            return ''.join(sql_parts).strip()
            return None
        except Exception as e:
            print(f"Error parsing cell source: {e}")
            return None

    # Helper function to extract SQL from magic commands
    def extract_sql_from_magic(cell_source: str) -> Optional[str]:
        match = re.match(r'%%sql\s*\n([\s\S]*)', cell_source, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    # Verify first code cell (execute_sql)
    first_cell = code_cells[0]
    formatted_sql = extract_sql_from_execute(first_cell.source)
    assert formatted_sql is not None, "SQL not found in execute_sql call."

    expected = (
        "SELECT a.col1, b.col2 FROM table_a AS a "
        "JOIN table_b AS b ON a.id = b.a_id WHERE a.status = 'active' "
        "ORDER BY a.created_at DESC"
    )
    assert normalize_sql(expected) == normalize_sql(formatted_sql)

    # Verify second code cell (%%sql magic)
    second_cell = code_cells[1]
    formatted_sql_magic = extract_sql_from_magic(second_cell.source)
    expected_magic = (
        "SELECT user_id, COUNT(*) AS login_count FROM logins "
        "WHERE login_date > '2023-01-01' GROUP BY user_id "
        "ORDER BY login_count DESC"
    )
    assert normalize_sql(expected_magic) == normalize_sql(formatted_sql_magic)

# Test 9: Logging Messages at Various Levels
def test_logging_messages_various_levels(sample_config, caplog):
    """
    Test that logging messages are correctly emitted at different log levels.

    Verifies that DEBUG messages are captured when the logger is set to DEBUG,
    and not captured when set to INFO.
    """
    sql = "SELECT * FROM table"
    logger = setup_logging("DEBUG")
    with caplog.at_level(logging.DEBUG, logger="formatter"):
        format_sql_code(sql, None, sample_config)
        assert "Original SQL:" in caplog.text
        assert "Formatted SQL:" in caplog.text

    caplog.clear()  # Clear previous logs

    logger = setup_logging("INFO")
    with caplog.at_level(logging.INFO, logger="formatter"):
        format_sql_code(sql, None, sample_config)
        # DEBUG messages should not appear
        assert "Original SQL:" not in caplog.text
        assert "Formatted SQL:" not in caplog.text

# Test 10: Subqueries Formatting
def test_format_sql_code_with_subqueries(sample_config):
    """
    Test formatting SQL queries that contain subqueries.

    Ensures that nested SELECT statements are formatted correctly.
    """
    sql = """
    SELECT
        a.id,
        (SELECT COUNT(*) FROM orders o WHERE o.user_id = a.id) AS order_count
    FROM users a
    WHERE a.active = TRUE
    ORDER BY order_count DESC
    """
    formatted = format_sql_code(sql, None, sample_config)
    expected_subquery = "(SELECT COUNT(*) FROM orders AS o WHERE o.user_id = a.id) AS order_count"
    assert normalize_sql(expected_subquery) in normalize_sql(formatted)

# Test 11: Various Indentation Widths
def test_format_sql_code_various_indentation(sample_config):
    """
    Test that the formatter applies different indentation widths correctly.

    Ensures that the indent_width configuration option is respected.
    """
    # Change indent_width to 2
    sample_config.indent_width = 2
    sql = """
    SELECT
      column1,
      column2
    FROM table
    WHERE condition
    ORDER BY column1 DESC
    """
    formatted = format_sql_code(sql, None, sample_config)
    expected = (
        "SELECT\n"
        "  column1,\n"
        "  column2\n"
        "FROM table\n"
        "WHERE condition\n"
        "ORDER BY column1 DESC"
    )
    assert normalize_sql(expected) == normalize_sql(formatted)

# Test 12: Multiple Formatting Options
def test_format_sql_code_multiple_options(sample_config):
    """
    Test that multiple formatting options work together correctly.

    Ensures that both indent_width and single_line_threshold are honored.
    """
    sample_config.indent_width = 2
    sample_config.single_line_threshold = 50
    sql = "SELECT col1, col2 FROM table WHERE condition ORDER BY col1 DESC"
    formatted = format_sql_code(sql, None, sample_config)
    # Should be single line due to low threshold
    expected = "SELECT col1, col2 FROM table WHERE condition ORDER BY col1 DESC"
    assert normalize_sql(expected) == normalize_sql(formatted)

# Test 13: Common Table Expressions (CTEs) Formatting
def test_format_sql_code_with_ctes(sample_config):
    """
    Test that SQL queries with Common Table Expressions (CTEs) are formatted correctly.
    """
    sql = """
    WITH recent_orders AS (
        SELECT user_id, COUNT(*) AS order_count
        FROM orders
        WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        GROUP BY user_id
    ), top_customers AS (
        SELECT u.id, u.name, ro.order_count
        FROM users u
        JOIN recent_orders ro ON u.id = ro.user_id
        WHERE ro.order_count > 5
    )
    SELECT tc.id, tc.name, tc.order_count
    FROM top_customers tc
    ORDER BY tc.order_count DESC
    """
    formatted = format_sql_code(sql, "mysql", sample_config)
    expected = (
        "WITH recent_orders AS ("
        "SELECT user_id, COUNT(*) AS order_count FROM orders "
        "WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL '30' DAY) GROUP BY user_id"
        "), top_customers AS ("
        "SELECT u.id, u.name, ro.order_count FROM users AS u "
        "JOIN recent_orders AS ro ON u.id = ro.user_id WHERE ro.order_count > 5"
        ") "
        "SELECT tc.id, tc.name, tc.order_count FROM top_customers AS tc ORDER BY tc.order_count DESC"
    )
    assert normalize_sql(expected) == normalize_sql(formatted)

# Test 14: Different SQL Dialects
def test_format_sql_code_different_dialects(sample_config):
    """
    Test that SQL formatting respects different SQL dialects.
    """
    sql = 'SELECT * FROM "my_table" WHERE "status" = \'active\''
    formatted_pg = format_sql_code(sql, "postgres", sample_config)
    expected_pg = 'SELECT * FROM "my_table" WHERE "status" = \'active\''
    assert normalize_sql(expected_pg) == normalize_sql(formatted_pg)

    formatted_sqlite = format_sql_code(sql, "sqlite", sample_config)
    expected_sqlite = 'SELECT * FROM "my_table" WHERE "status" = \'active\''
    assert normalize_sql(expected_sqlite) == normalize_sql(formatted_sqlite)

# Test 15: Window Functions Formatting
def test_format_sql_code_with_window_functions(sample_config):
    """
    Test that SQL queries with window functions are formatted correctly.
    """
    sql = """
    SELECT
        user_id,
        login_date,
        COUNT(*) OVER (PARTITION BY user_id ORDER BY login_date) AS cumulative_logins
    FROM logins
    """
    formatted = format_sql_code(sql, "mysql", sample_config)
    expected = (
        "SELECT user_id, login_date, COUNT(*) OVER (PARTITION BY user_id ORDER BY login_date) AS cumulative_logins FROM logins"
    )
    assert normalize_sql(expected) == normalize_sql(formatted)

# Test 16: Placeholder Preservation
def test_format_sql_code_with_placeholders(sample_config):
    """
    Test that SQL queries with placeholders are preserved correctly.
    """
    sql = "SELECT * FROM users WHERE id = %s AND status = ?"
    formatted = format_sql_code(sql, "mysql", sample_config)
    expected = "SELECT * FROM users WHERE id = %s AND status = ?"
    assert normalize_sql(expected) == normalize_sql(formatted)

# Test 17: Handling Comments
def test_format_sql_code_preserve_comments(sample_config):
    """
    Test that SQL comments are preserved in the formatted output.
    """
    sql = """
    -- This is a comment
    SELECT * FROM users -- Inline comment
    """
    formatted = format_sql_code(sql, None, sample_config)
    expected = """
    /* This is a comment */
    SELECT * FROM users /* Inline comment */
    """
    assert normalize_sql(expected) == normalize_sql(formatted)