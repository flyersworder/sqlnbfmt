"""Rewritten SQL literals must be valid Python with the intended value.

The formatter writes the formatted SQL back as a new string literal. It has to
keep backslashes and raw prefixes intact and pick quotes that don't clash with
the SQL; when that isn't possible the string is left alone rather than corrupted.
"""

import ast
import logging
import warnings

import pytest

from sqlnbfmt.formatter import FormattingConfig, _format_python_source


def fmt(source, dialect="postgres"):
    result, _ = _format_python_source(
        source, FormattingConfig(), dialect, logging.getLogger("test")
    )
    return result


def strict_parse(source):
    """Parses source, failing on SyntaxWarnings such as invalid escapes."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return ast.parse(source)


def string_values(source):
    return [
        node.value
        for node in ast.walk(strict_parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_backslash_escapes_keep_their_value():
    # The SQL compares c with a backslash followed by "n", not with a newline
    source = "q = \"select a, b from t where c = '\\\\n' and d = 1\"\n"
    formatted = fmt(source)
    assert formatted != source
    (value,) = string_values(formatted)
    assert "c = '\\n'" in value
    assert "\n  c" not in value.split("c = ")[1][:3]


def test_raw_strings_keep_their_prefix():
    source = "q = r\"select a, b from t where c like '%\\_%' and d = 1\"\n"
    formatted = fmt(source)
    assert formatted.startswith('q = r"""')
    (value,) = string_values(formatted)
    assert "LIKE '%\\_%'" in value


def test_single_line_sql_with_double_quotes():
    source = "cursor.execute('delete from \"users\"')\n"
    formatted = fmt(source)
    assert string_values(formatted) == ['DELETE FROM "users"']


def test_single_line_sql_with_both_quote_kinds():
    source = "cursor.execute('update \"t\" set a = \\'x\\'')\n"
    formatted = fmt(source)
    assert string_values(formatted) == ["UPDATE \"t\" SET a = 'x'"]


def test_sql_containing_triple_quotes_is_left_alone():
    source = 'q = \'select a, b from t where c = 1 -- note """ here\'\n'
    formatted = fmt(source)
    strict_parse(formatted)
    assert formatted == source or "'''" in formatted


def test_fstring_literal_braces_are_not_turned_into_placeholders():
    source = "q = f\"select '{{}}'::json as j, a from {table} where b = 1\"\n"
    formatted = fmt(source)
    tree = strict_parse(formatted)
    (fstring,) = [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]
    placeholders = [v for v in fstring.values if isinstance(v, ast.FormattedValue)]
    assert len(placeholders) == 1


@pytest.mark.parametrize(
    "source",
    [
        "q = \"select a, b from t where c = '\\\\n' and d = 1\"\n",
        "q = r\"select a, b from t where c like '%\\_%' and d = 1\"\n",
        "cursor.execute('delete from \"users\"')\n",
        "cursor.execute('update \"t\" set a = \\'x\\'')\n",
        'q = \'select a, b from t where c = 1 -- note """ here\'\n',
        "q = f\"select '{{}}'::json as j, a from {table} where b = 1\"\n",
        "def _():\n    pd.read_sql(r\"select a from t where b like '\\_%' and c = 1\", conn)\n",
    ],
)
def test_output_is_valid_and_idempotent(source):
    formatted = fmt(source)
    strict_parse(formatted)
    assert fmt(formatted) == formatted


# Awkward SQL fragments, written as they'd appear inside the original literal's
# source (so "\\\\" is one backslash in a non-raw literal, two in a raw one).
TRICKY_CONDITIONS = [
    "c = '\\\\n'",
    "c like '%\\\\_%'",
    "c = 'it''s'",
    '"Col" = 1',
    "c = '{{}}'",
    "c = 'a\\\\\\\\b'",
]


@pytest.mark.parametrize("prefix", ["", "r", "f", "rf"])
@pytest.mark.parametrize("quote", ['"', "'''"])
@pytest.mark.parametrize("condition", TRICKY_CONDITIONS)
def test_tricky_literals_stay_valid(prefix, quote, condition):
    if quote == '"' and '"' in condition:
        pytest.skip("not representable in this quote style")
    if "f" not in prefix and "{{" in condition:
        condition = condition.replace("{{", "{").replace("}}", "}")
    sql = f"select a, b from t where {condition} and d = 1"
    source = f"q = {prefix}{quote}{sql}{quote}\n"
    try:
        strict_parse(source)
    except (SyntaxError, SyntaxWarning):
        pytest.skip("input itself isn't valid Python")
    formatted = fmt(source)
    strict_parse(formatted)
    assert fmt(formatted) == formatted
