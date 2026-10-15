"""Which calls count as SQL-bearing for a configured function name (#28)."""

import logging

import pytest

from sqlnbfmt.formatter import (
    FormattingConfig,
    _format_python_source,
    matches_function_name,
)


@pytest.mark.parametrize(
    ("call_name", "name"),
    [
        # Plain names match whole words of the last name component
        ("pd.read_sql", "read_sql"),
        ("pd.read_sql_query", "read_sql"),
        ("cursor.execute", "execute"),
        ("execute_sql", "execute"),
        ("execute_sql", "sql"),
        ("extras.execute_values", "execute"),
        ("conn.exec_driver_sql", "sql"),
        ("mo.sql", "sql"),
        ("spark.sql", "sql"),
        ("conn.executescript", "executescript"),
        ("self.db.run_query", "run_query"),
        ("execute", "execute"),  # e.g. get_conn().execute(...)
        # camelCase boundaries split words too; matching ignores case
        ("stmt.executeQuery", "execute"),
        ("stmt.executeQuery", "execute_query"),
        ("runSql", "sql"),
        # Dotted names match the end of the call path
        ("mo.sql", "mo.sql"),
        ("app.mo.sql", "mo.sql"),
        ("db.run", "db.run"),
    ],
)
def test_matches(call_name, name):
    assert matches_function_name(call_name, name)


@pytest.mark.parametrize(
    ("call_name", "name"),
    [
        ("rerun", "run"),  # not a whole word
        ("df.query", "q"),
        ("requests.post", "q"),
        ("mysql_connector.connect", "sql"),  # module path doesn't count
        ("pyspark.sql.functions.expr", "sql"),
        ("cursor.executemany", "execute"),  # covered by its own default name
        ("sqldf", "sql"),
        ("subprocess.run", "db.run"),  # dotted names need the full suffix
        ("memo.sql", "mo.sql"),
        ("mo.sql", "app.mo.sql"),
    ],
)
def test_does_not_match(call_name, name):
    assert not matches_function_name(call_name, name)


def fmt(source, **config):
    result, _ = _format_python_source(
        source, FormattingConfig(**config), None, logging.getLogger("test")
    )
    return result


SQL = "select id, name from users where active = 1"


@pytest.mark.parametrize(
    "call",
    [
        "pd.read_sql",
        "pd.read_sql_query",
        "cursor.execute",
        "cursor.executemany",
        "conn.executescript",
        "mo.sql",
        "spark.sql",
        "duckdb.sql",
        "sqldf",
        "execute_sql",
    ],
)
def test_default_names_cover_common_calls(call):
    assert "SELECT" in fmt(f'{call}("{SQL}")\n')


def test_generic_name_no_longer_matches_unrelated_calls():
    source = f'df.query("{SQL}")\nrerun("{SQL}")\n'
    assert fmt(source, function_names={"q", "run"}) == source


def test_dotted_name_targets_one_call():
    source = f'db.run("{SQL}")\nsubprocess.run("{SQL}")\n'
    formatted = fmt(source, function_names={"db.run"})
    assert formatted.count("SELECT") == 1
    assert formatted.endswith(f'subprocess.run("{SQL}")\n')
