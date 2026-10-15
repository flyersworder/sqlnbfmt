"""sql_decorators: SQL in decorator arguments and in decorated functions' returns (#30)."""

import ast
import logging

import pytest

from sqlnbfmt.formatter import FormattingConfig, _format_python_source, load_config

SQL = "select id, name from users where active = 1"
FORMATTED = "SELECT\n  id,\n  name\nFROM users\nWHERE\n  active = 1"


def fmt(source, config=None):
    result, _ = _format_python_source(
        source, config or FormattingConfig(), None, logging.getLogger("test")
    )
    return result


@pytest.mark.parametrize(
    "decorator", ["query", "db_query", "sql_decorator", "app.query"]
)
def test_decorator_argument(decorator):
    source = f'@{decorator}("{SQL}")\ndef get_users(): ...\n'
    assert fmt(source) == (
        f'@{decorator}(\n    """\n'
        + "".join(f"    {line}\n" for line in FORMATTED.split("\n"))
        + '    """\n)\ndef get_users(): ...\n'
    )


def test_decorator_keyword_argument():
    formatted = fmt(f'@query(sql="{SQL}", timeout=5)\ndef get_users(): ...\n')
    assert formatted.count("SELECT") == 1 and "timeout=5" in formatted


def test_decorator_matching_both_lists_is_formatted_once():
    """sql_query is a decorator and also contains the function name `sql`."""
    formatted = fmt(f'@sql_query("{SQL}")\ndef get_users(): ...\n')
    assert formatted.count("SELECT") == 1
    ast.parse(formatted)


@pytest.mark.parametrize("decorator", ["@query", "@query()", "@db.query"])
@pytest.mark.parametrize("prefix", ["def", "async def"])
def test_decorated_function_return(decorator, prefix):
    source = f'{decorator}\n{prefix} get_users():\n    return "{SQL}"\n'
    assert fmt(source) == (
        f'{decorator}\n{prefix} get_users():\n    return """\n{FORMATTED}\n"""\n'
    )


def test_decorated_function_fstring_return():
    source = '@query\ndef get(table):\n    return f"select id from {table} where active = 1"\n'
    assert "FROM {table}" in fmt(source)


def test_decorated_class_argument():
    assert "SELECT" in fmt(f'@query("{SQL}")\nclass Users: ...\n')


@pytest.mark.parametrize(
    "source",
    [
        # Undecorated, or decorated with something unrelated
        f'def get_users():\n    return "{SQL}"\n',
        f'@functools.cache\ndef get_users():\n    return "{SQL}"\n',
        f'@cache("{SQL}")\ndef get_users(): ...\n',
        # Docstrings aren't returns
        f'@query\ndef get_users():\n    """{SQL}"""\n',
        # Non-SQL return values
        '@query\ndef name():\n    return "hello world, nice to meet you"\n',
        # Returns of nested functions belong to the nested function
        f'@query\ndef outer():\n    def inner():\n        return "{SQL}"\n    return inner\n',
        f'@query\ndef outer():\n    return lambda: "{SQL}"\n',
    ],
    ids=[
        "undecorated",
        "unrelated-decorator",
        "unrelated-decorator-call",
        "docstring",
        "non-sql-return",
        "nested-function",
        "lambda",
    ],
)
def test_left_alone(source):
    assert fmt(source) == source


@pytest.mark.parametrize(
    "source",
    [
        f'@query("{SQL}")\ndef get_users(): ...\n',
        f'@query\ndef get_users():\n    return "{SQL}"\n',
        f'class Repo:\n    @db_query("{SQL}")\n    def users(self): ...\n',
    ],
)
def test_valid_and_idempotent(source):
    formatted = fmt(source)
    ast.parse(formatted)
    assert formatted != source
    assert fmt(formatted) == formatted


@pytest.mark.parametrize("config_format", ["toml", "yaml"])
def test_custom_decorators_via_config(tmp_path, config_format):
    source = f'@warehouse("{SQL}")\ndef get_users(): ...\n'
    assert fmt(source) == source  # not a default decorator

    if config_format == "toml":
        path = tmp_path / "pyproject.toml"
        path.write_text('[tool.sqlnbfmt]\nextend_sql_decorators = ["warehouse"]\n')
    else:
        path = tmp_path / "config.yaml"
        path.write_text("sql_decorators:\n  - warehouse\n")
    assert "SELECT" in fmt(source, load_config(path))
