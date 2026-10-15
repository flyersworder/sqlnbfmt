"""Which calls count as SQL-bearing for a configured function name.

Names match as case-sensitive substrings of the call's full dotted name. This
is deliberately permissive (#28): a false positive still has to pass the
looks-like-SQL check, while a missed call silently leaves SQL unformatted.
These tests pin that behavior, including calls that only match by substring.
"""

import logging

import pytest

from sqlnbfmt.formatter import FormattingConfig, _format_python_source

SQL = "select id, name from users where active = 1"


def fmt(source, **config):
    result, _ = _format_python_source(
        source, FormattingConfig(**config), None, logging.getLogger("test")
    )
    return result


@pytest.mark.parametrize(
    "call",
    [
        "pd.read_sql",
        "pd.read_sql_query",
        "cursor.execute",
        "cursor.executemany",  # via "execute"
        "client.execute2",  # via "execute"
        "conn.executescript",
        "extras.execute_values",  # psycopg2, via "execute"
        "execute_sql",
        "mo.sql",
        "spark.sql",
        "duckdb.sql",
        "sqldf",  # pandasql, via "sql"
        "pysqldf",  # the usual pandasql wrapper, via "sql"
        "sqlalchemy.text",  # via "sql" in the module path
    ],
)
def test_default_names_match_common_calls(call):
    assert "SELECT" in fmt(f'{call}("{SQL}")\n')


def test_matching_is_case_sensitive():
    source = f'session.runSQLQuery("{SQL}")\n'
    assert fmt(source) == source
    assert "SELECT" in fmt(source, function_names={"SQL"})


def test_names_match_anywhere_in_the_dotted_name():
    """Short names match broadly; prefer specific names like run_query."""
    source = f'rerun("{SQL}")\nsubprocess.run("{SQL}")\n'
    assert fmt(source, function_names={"run"}).count("SELECT") == 2
    assert fmt(source, function_names={"run_query"}) == source


def test_non_sql_strings_are_left_alone_even_when_the_name_matches():
    source = 'subprocess.run("echo hello world")\n'
    assert fmt(source, function_names={"run"}) == source
