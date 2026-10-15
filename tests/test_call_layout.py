"""Layout of multi-line SQL passed to function calls.

The reformatted string must be indented relative to the line the call starts on,
and code after it (a closing paren, further arguments) must not end up at column 0.
"""

import ast
import logging

import pytest

from sqlnbfmt.formatter import FormattingConfig, _format_python_source

SQL = "select id from users limit 10"
CONFIG = FormattingConfig(function_names={"pull_rows", "read_sql", "execute", "sql"})


def fmt(source: str, config: FormattingConfig = CONFIG) -> str:
    result, _ = _format_python_source(source, config, None, logging.getLogger("test"))
    return result


CASES = {
    "sole_argument_nested": (
        f'def _():\n    pull_rows("{SQL}")\n    return\n',
        (
            "def _():\n"
            "    pull_rows(\n"
            '        """\n'
            "        SELECT\n"
            "          id\n"
            "        FROM users\n"
            "        LIMIT 10\n"
            '        """\n'
            "    )\n"
            "    return\n"
        ),
    ),
    "first_argument_with_more_arguments": (
        f'df = pd.read_sql("{SQL}", conn)\n',
        (
            "df = pd.read_sql(\n"
            '    """\n'
            "    SELECT\n"
            "      id\n"
            "    FROM users\n"
            "    LIMIT 10\n"
            '    """, conn)\n'
        ),
    ),
    "keyword_argument_hugs": (
        f'def run(cursor):\n    cursor.execute(sql="{SQL}")\n',
        (
            "def run(cursor):\n"
            '    cursor.execute(sql="""\n'
            "        SELECT\n"
            "          id\n"
            "        FROM users\n"
            "        LIMIT 10\n"
            '        """)\n'
        ),
    ),
    "later_positional_argument_hugs": (
        f'execute(conn, "{SQL}")\n',
        (
            'execute(conn, """\n'
            "    SELECT\n"
            "      id\n"
            "    FROM users\n"
            "    LIMIT 10\n"
            '    """)\n'
        ),
    ),
    "argument_on_its_own_line": (
        f'def _():\n    pull_rows(\n            "{SQL}"\n    )\n',
        (
            "def _():\n"
            "    pull_rows(\n"
            '        """\n'
            "        SELECT\n"
            "          id\n"
            "        FROM users\n"
            "        LIMIT 10\n"
            '        """\n'
            "    )\n"
        ),
    ),
    "marimo_fstring": (
        (
            "@app.cell\n"
            "def _(mo, tbl):\n"
            '    _df = mo.sql(f"""select id from {tbl} limit 10""")\n'
            "    return\n"
        ),
        (
            "@app.cell\n"
            "def _(mo, tbl):\n"
            "    _df = mo.sql(\n"
            '        f"""\n'
            "        SELECT\n"
            "          id\n"
            "        FROM {tbl}\n"
            "        LIMIT 10\n"
            '        """\n'
            "    )\n"
            "    return\n"
        ),
    ),
    "call_on_continuation_line": (
        f'result = (\n    pull_rows("{SQL}")\n)\n',
        (
            "result = (\n"
            "    pull_rows(\n"
            '        """\n'
            "        SELECT\n"
            "          id\n"
            "        FROM users\n"
            "        LIMIT 10\n"
            '        """\n'
            "    )\n"
            ")\n"
        ),
    ),
}


@pytest.mark.parametrize(("source", "expected"), CASES.values(), ids=CASES.keys())
def test_call_layout(source, expected):
    assert fmt(source) == expected


@pytest.mark.parametrize(("source", "expected"), CASES.values(), ids=CASES.keys())
def test_call_layout_is_valid_and_idempotent(source, expected):
    formatted = fmt(source)
    ast.parse(formatted)
    assert fmt(formatted) == formatted


def test_call_layout_respects_indent_width():
    config = FormattingConfig(function_names={"pull_rows"}, indent_width=2)
    assert fmt(f'def _():\n    pull_rows("{SQL}")\n', config) == (
        "def _():\n"
        "    pull_rows(\n"
        '      """\n'
        "      SELECT\n"
        "        id\n"
        "      FROM users\n"
        "      LIMIT 10\n"
        '      """\n'
        "    )\n"
    )
