"""Formatting must never change what a SQL string means, or break the code.

Covers non-ASCII source (AST column offsets are UTF-8 bytes, not characters),
f-string round-tripping (quotes around placeholders, literal braces, conversions
and format specs, backslashes), and the whole-source safety net.
"""

import ast
import logging
import warnings

import pytest
import sqlglot

from sqlnbfmt.formatter import (
    FormattingConfig,
    SQLStringFormatter,
    _format_python_source,
    format_cell_source,
)

LOGGER = logging.getLogger("test")


def fmt(source, dialect=None):
    result, _ = _format_python_source(source, FormattingConfig(), dialect, LOGGER)
    return result


def parse_strict(source):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return ast.parse(source)


def string_literals(tree):
    """Top-level string literals (plain and f-strings) in source order."""
    found = []

    def visit(node):
        if isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.Constant) and isinstance(node.value, str)
        ):
            found.append(node)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def sql_template(node):
    """The literal's SQL with stable markers for f-string fields and DB-API
    parameters, plus its sorted fields. Markers are named after the expression,
    so dialect rewrites that reorder clauses (LIMIT -> TOP) still compare equal."""
    if isinstance(node, ast.Constant):
        text, exprs = node.value, []
    else:
        dumps = sorted(
            {ast.dump(p) for p in node.values if not isinstance(p, ast.Constant)}
        )
        text = "".join(
            p.value if isinstance(p, ast.Constant) else f"PH_{dumps.index(ast.dump(p))}"
            for p in node.values
        )
        exprs = sorted(
            ast.dump(p) for p in node.values if not isinstance(p, ast.Constant)
        )
    text = text.replace("%s", "PARAM_").replace("?", "PARAM_")
    return text, exprs


def canonical(sql, dialect=None):
    return sqlglot.transpile(sql, read=dialect, write=dialect, pretty=False)


def assert_same_meaning(original_source, formatted_source, dialect=None):
    """Valid Python, same literals, and each SQL literal means the same thing."""
    before = string_literals(ast.parse(original_source))
    after = string_literals(parse_strict(formatted_source))
    assert len(before) == len(after)
    for old, new in zip(before, after):
        (old_sql, old_exprs), (new_sql, new_exprs) = (
            sql_template(old),
            sql_template(new),
        )
        assert old_exprs == new_exprs
        if old_sql != new_sql:
            assert canonical(old_sql, dialect) == canonical(new_sql, dialect), (
                old_sql,
                new_sql,
            )


NON_ASCII = ["é", "日本語", "🚀"]


@pytest.mark.parametrize("text", NON_ASCII)
def test_non_ascii_inside_sql_argument(text):
    source = f"cursor.execute(\"select a, b from t where name = '{text}' and c = 1\")\n"
    formatted = fmt(source)
    assert formatted != source
    assert_same_meaning(source, formatted)
    assert formatted.rstrip().endswith(")")


@pytest.mark.parametrize("text", NON_ASCII)
def test_non_ascii_before_raw_string(text):
    source = f'x = "{text}"; q = r"select a from t where b like \'x\\_%\' and c = 1"\n'
    formatted = fmt(source)
    assert formatted != source
    assert_same_meaning(source, formatted)


@pytest.mark.parametrize("text", NON_ASCII)
def test_non_ascii_in_notebook_cell(text):
    source = (
        f"# {text}\n"
        f"label = '{text}'; df = pd.read_sql(\"select a from t where n = '{text}' and b = 1\", conn)"
    )
    formatted = format_cell_source(source, FormattingConfig(), None, LOGGER)
    assert formatted != source
    assert_same_meaning(source, formatted)


@pytest.mark.parametrize("text", NON_ASCII)
def test_non_ascii_multiline_call_layout(text):
    source = f"def _():\n    x = '{text}'; pull(\"select id from users where n = '{text}' limit 5\")\n"
    formatted, _ = _format_python_source(
        source, FormattingConfig(function_names={"pull"}), None, LOGGER
    )
    assert formatted != source
    assert_same_meaning(source, formatted)


@pytest.mark.parametrize(
    "source",
    [
        # Quotes around a placeholder are part of the SQL
        "q = f\"select * from t where a = {x} and b = '{y}'\"\n",
        'q = f"select * from t where a = {x} and b = \\"{y}\\""\n',
        # Escaped braces, with and without placeholders
        "q = f\"select a from t where b = '{{{{' and c = 1\"\n",
        "q = f\"select a from t where b = '{{x}}' and c = {y}\"\n",
        # Conversions and format specs
        'q = f"select a from t where b = {x!r} and c = {y:>3} and d = 1"\n',
        # Backslashes in the text of an f-string (and a raw f-string)
        "q = f\"select a from t where b like '\\\\_' and c = {y}\"\n",
        "q = rf\"select a from t where b like '\\_' and c = {y}\"\n",
        # Many placeholders: PLACEHOLDER_1 must not clash with PLACEHOLDER_10
        'q = f"select {a0}, {a1}, {a2}, {a3}, {a4}, {a5}, {a6}, {a7}, {a8}, {a9}, {a10}, {a11} from t"\n',
    ],
    ids=[
        "quoted-placeholder",
        "double-quoted-placeholder",
        "escaped-braces-only",
        "escaped-braces-and-placeholder",
        "conversion-and-format-spec",
        "backslash",
        "raw-fstring-backslash",
        "many-placeholders",
    ],
)
def test_fstrings_keep_their_meaning(source):
    formatted = fmt(source)
    assert formatted != source, "should be formatted, not skipped"
    assert_same_meaning(source, formatted)
    assert fmt(formatted) == formatted


def test_safety_net_rejects_a_broken_rewrite(monkeypatch, caplog):
    """If a rewrite would break the code, the source is returned unchanged."""
    caplog.set_level(logging.WARNING)

    def bad_layout(self, start, end, block, base_indent, block_indent):
        return start, end + 1, block  # also swallows the closing paren

    monkeypatch.setattr(SQLStringFormatter, "_layout_call_argument", bad_layout)
    source = 'pd.read_sql("select a, b from t where c = 1 and d = 2")\n'
    result, changed = _format_python_source(source, FormattingConfig(), None, LOGGER)
    assert (result, changed) == (source, False)
    assert "left unchanged" in caplog.text


def test_safety_net_in_notebook_cells(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    def bad_layout(self, start, end, block, base_indent, block_indent):
        return start, end + 1, block

    monkeypatch.setattr(SQLStringFormatter, "_layout_call_argument", bad_layout)
    source = 'pd.read_sql("select a, b from t where c = 1 and d = 2")'
    assert format_cell_source(source, FormattingConfig(), None, LOGGER) == source
    assert "left unchanged" in caplog.text


def test_skipped_string_is_reported_with_its_line(caplog):
    """A string that can't be written back is reported, not silently skipped."""
    caplog.set_level(logging.WARNING)
    # The comment contains both kinds of triple quote, so no quoting works
    source = 'x = 1\nq = "select a, b from t where c = 1 -- has \\"\\"\\" and \'\'\'"\n'
    assert fmt(source) == source
    assert "line 2: SQL formatting skipped" in caplog.text


DIALECTS = [
    None,
    "postgres",
    "mysql",
    "duckdb",
    "snowflake",
    "bigquery",
    "tsql",
    "sqlite",
]


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize(
    "source",
    [
        'cursor.execute("select a, b from users where id = %s and c = 1", (1,))\n',
        'cursor.execute("select a, b from users where id = ? and c = 1", (1,))\n',
        'q = f"select a, b from {table} where c = {value} limit {n}"\n',
        "q = f\"select a from {schema}.{table} where name = '{name}' and c = 1\"\n",
        # 12 parameters: AUTO_PLACEHOLDER_1_ must not clash with _10_
        'cursor.execute("insert into t values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)\n',
    ],
    ids=["percent-s", "qmark", "fstring", "fstring-qualified", "many-qmarks"],
)
def test_parameters_survive_every_dialect(source, dialect):
    """Named-parameter stand-ins used to be rewritten per dialect (%(x)s, $x, @x)."""
    formatted = fmt(source, dialect)
    assert formatted != source
    assert_same_meaning(source, formatted, dialect)
    for marker in ("%s", "?", "{table}", "{value}", "{n}", "{schema}", "'{name}'"):
        assert formatted.count(marker) == source.count(marker), marker
