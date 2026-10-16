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
    so dialect rewrites that reorder clauses (LIMIT -> TOP) still compare equal.

    Positional parameters (%s, ?) are indistinguishable after the fact, so this
    can't see them being reordered; the formatter refuses to move them, which
    test_moved_or_rewrapped_placeholders_are_left_alone checks directly."""
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
        'q = f"select a, b from {table} where c = {value} and d = {n}"\n',
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


# --- Placeholders must keep their place (#34 review) ----------------------


@pytest.mark.parametrize(
    ("source", "dialect"),
    [
        # A field holding a clause would get an alias: FROM t AS {where_clause}
        ('q = f"SELECT * FROM t {where_clause}"\n', None),
        ('q = f"SELECT * FROM t {join_clause} WHERE a = 1"\n', None),
        # Positional parameters must not be reordered (they bind by position)
        (
            'cursor.execute("SELECT a FROM t WHERE x = %s OFFSET %s LIMIT %s", p)\n',
            None,
        ),
        ('cursor.execute("SELECT a FROM t WHERE x = ? LIMIT ?", p)\n', "tsql"),
    ],
    ids=["clause-field", "join-field", "offset-limit", "tsql-top"],
)
def test_moved_or_rewrapped_placeholders_are_left_alone(source, dialect, caplog):
    caplog.set_level(logging.WARNING)
    assert fmt(source, dialect) == source
    assert "SQL formatting skipped" in caplog.text


@pytest.mark.parametrize(
    ("dialect", "param"),
    [
        ("postgres", ":name"),
        ("bigquery", ":name"),
        ("duckdb", "%(name)s"),
        ("postgres", "$1"),
        (None, ":user_id"),
    ],
)
def test_user_written_parameters_are_kept(dialect, param):
    source = f'cursor.execute("select a, b from t where x = {param} and y = 1", p)\n'
    formatted = fmt(source, dialect)
    assert formatted != source
    assert formatted.count(param) == 1


@pytest.mark.parametrize(
    "text", ["PLACEHOLDER_0_", "AUTO_PLACEHOLDER_0_", "SQLNBFMT_F0_", "SQLNBFMT_P0_"]
)
def test_sql_text_resembling_stand_ins_is_untouched(text):
    for source in (
        f"q = f\"select a from t where x = {{v}} and y = '{text}'\"\n",
        f"cursor.execute(\"select a from t where note = '{text}' and x = ?\", p)\n",
    ):
        formatted = fmt(source)
        assert_same_meaning(source, formatted)
        assert formatted.count(text) == 1


@pytest.mark.parametrize(
    "field",
    ['{d["k"]}', "{x if y else 'z'}", "{ {1: 2}[1] }", "{v!r:>3}", "{obj.attr}"],
)
def test_fstring_fields_keep_their_exact_source(field):
    source = f'q = f"select a from t where b = {field} and c = 1"\n'
    formatted = fmt(source)
    assert formatted != source
    assert field in formatted


def test_not_equal_with_parameter_is_still_formatted():
    source = 'cursor.execute("select a from t where b != %s and c = 1", p)\n'
    formatted = fmt(source)
    assert formatted != source
    assert "%s" in formatted


# --- Second #34 review: no false skips, data inside strings, robustness ------


@pytest.mark.parametrize(
    "source",
    [
        'q = "SELECT a FROM t -- why?\\nWHERE b = 1"\n',
        'cursor.execute("SELECT * FROM t WHERE a = %s -- the id\\n AND b = 2", p)\n',
        'cursor.execute("SELECT * FROM t WHERE a = %s;", p)\n',
        'cursor.execute("SELECT * FROM t WHERE id = %s::int AND b = 1", p)\n',
        "cursor.execute(\"SELECT * FROM t WHERE note LIKE '50%s?' AND a = %s\", p)\n",
        'q = f"select a as {alias}, b from t where c = 1"\n',
    ],
    ids=[
        "comment-with-qmark",
        "param-before-comment",
        "trailing-semicolon",
        "cast",
        "param-chars-in-string",
        "explicit-alias-field",
    ],
)
def test_harmless_rewrites_around_parameters_are_formatted(source, caplog):
    caplog.set_level(logging.WARNING)
    formatted = fmt(source)
    assert formatted != source
    assert "skipped" not in caplog.text
    for marker in ("%s", "?", "{alias}"):
        assert formatted.count(marker) == source.count(marker), marker


def test_multiline_string_inside_sql_keeps_its_value():
    source = (
        "def _():\n"
        '    cursor.execute("""SELECT a, \'multi\nline\' AS s FROM t WHERE c = 1""")\n'
    )
    formatted = fmt(source)
    assert formatted != source
    assert "'multi\nline'" in formatted
    assert_same_meaning(source, formatted)


@pytest.mark.parametrize("newline", ["\r", "\r\n"])
def test_carriage_return_line_breaks(newline):
    source = (
        f'x = "é"{newline}cursor.execute("SELECT a, b FROM t WHERE c = 1"){newline}'
    )
    formatted = format_cell_source(source, FormattingConfig(), None, LOGGER)
    parse_strict(formatted)
    assert "SELECT" in formatted and "FROM t" in formatted


def test_oracle_numeric_parameters_are_kept():
    source = 'cursor.execute("select a, b from t where x = :1 and y = :2", p)\n'
    formatted = fmt(source, "oracle")
    assert formatted != source
    assert formatted.count(":1") == 1 and formatted.count(":2") == 1


def test_one_bad_rewrite_does_not_discard_the_others(monkeypatch, caplog):
    """The safety net rejects only the faulty rewrite and names its line."""
    caplog.set_level(logging.WARNING)
    original = SQLStringFormatter._layout_call_argument

    def bad_layout(self, start, end, block, base_indent, block_indent):
        if "bad" in self.source[start:end]:
            return start, end + 1, block  # swallows the closing paren
        return original(self, start, end, block, base_indent, block_indent)

    monkeypatch.setattr(SQLStringFormatter, "_layout_call_argument", bad_layout)
    source = (
        'pd.read_sql("select a, b from good where c = 1 and d = 2")\n'
        'pd.read_sql("select a, b from bad where c = 1 and d = 2")\n'
    )
    result, changed = _format_python_source(source, FormattingConfig(), None, LOGGER)
    assert changed
    assert "FROM good" in result
    assert 'pd.read_sql("select a, b from bad where c = 1 and d = 2")' in result
    assert "line 2" in caplog.text


# --- Final #34 review: the dialect's tokenizer decides what is a string -----


@pytest.mark.parametrize(
    ("source", "dialect"),
    [
        # Moved into a string: INTERVAL %s DAY -> INTERVAL '%s DAY'
        (
            'cursor.execute("SELECT * FROM t WHERE ts > now() - INTERVAL %s DAY", p)\n',
            "postgres",
        ),
        ('q = f"SELECT * FROM t WHERE ts > now() - INTERVAL {span}"\n', "postgres"),
        # Parenthesized: breaks psycopg2's execute_values
        ('execute_values(cur, "INSERT INTO t (a, b) VALUES %s", rows)\n', "postgres"),
        ('q = f"INSERT INTO t (a, b) VALUES {values_sql}"\n', None),
        # Aliased: SELECT {distinct} a -> SELECT {distinct} AS a
        ('q = f"SELECT {distinct} a, b FROM t WHERE x = 1"\n', None),
        # The scanner used to misread '\' and -- don't, hiding these from checks
        (
            (
                "cursor.execute(\"SELECT replace(p, '\\\\', '/') AS q FROM t "
                "WHERE k = ? AND c = 'x' ORDER BY q LIMIT ?\", p)\n"
            ),
            "tsql",
        ),
        (
            'cursor.execute("SELECT a FROM t OFFSET %s LIMIT %s -- don\'t", p)\n',
            "postgres",
        ),
    ],
    ids=[
        "interval-param",
        "interval-field",
        "values-param",
        "values-field",
        "distinct-field",
        "backslash-string-tsql-top",
        "quote-in-comment-offset-limit",
    ],
)
def test_placeholders_are_not_quoted_parenthesized_or_aliased(source, dialect, caplog):
    caplog.set_level(logging.WARNING)
    assert fmt(source, dialect) == source
    assert "SQL formatting skipped" in caplog.text


@pytest.mark.parametrize(
    ("dialect", "sql"),
    [
        (
            "postgres",
            "SELECT replace(p, '\\\\', '/') AS q FROM t WHERE id = :id AND c = 'x'",
        ),
        (
            "duckdb",
            "SELECT replace(p, '\\\\', '/') AS q FROM t WHERE id = :id AND c = 'x'",
        ),
        ("postgres", "SELECT $$it's$$ AS q, b FROM t WHERE id = :id AND c = 'x'"),
        ("bigquery", "SELECT a FROM t # don't\\nWHERE b = :id AND c = 'x'"),
    ],
    ids=["postgres-backslash", "duckdb-backslash", "dollar-quoted", "hash-comment"],
)
def test_parameters_after_dialect_specific_quoting_are_kept(dialect, sql):
    source = f'cursor.execute("{sql}", p)\n'
    formatted = fmt(source, dialect)
    assert formatted.count(":id") == 1
    assert_same_meaning(source, formatted, dialect)


@pytest.mark.parametrize(
    ("dialect", "sql", "value"),
    [
        (
            "postgres",
            "SELECT * FROM t WHERE p LIKE 'a\\\\_%' ESCAPE '\\\\' AND note = 'l1\nl2'",
            "'l1\nl2'",
        ),
        ("mysql", "SELECT a AS `x\ny` FROM t WHERE b = 1", "`x\ny`"),
        ("tsql", "SELECT a AS [x\ny] FROM t WHERE b = 1", "[x\ny]"),
    ],
    ids=["after-backslash-string", "backtick-name", "bracket-name"],
)
def test_multiline_values_in_every_dialect_keep_their_value(dialect, sql, value):
    source = f'def _():\n    cursor.execute("""{sql}""")\n'
    formatted = fmt(source, dialect)
    # Formatted with current sqlglot; older versions indent quoted names, which
    # the re-parse check catches, leaving the string as is
    assert value in formatted


def test_dollar_quoted_function_body_is_kept_verbatim():
    body = (
        'rows = plpy.execute("SELECT name FROM users")\n'
        "# don't fail when empty\n"
        "if not rows:\n"
        "    return None\n"
        "return rows[0]['name']\n"
    )
    source = (
        "def setup(cur):\n"
        '    cur.execute("""CREATE FUNCTION first_name() RETURNS text AS $$\n'
        f'{body}$$ LANGUAGE plpython3u""")\n'
    )
    formatted = fmt(source, "postgres")
    assert formatted != source
    assert f"$$\n{body}$$" in formatted


def test_skip_hint_with_carriage_returns():
    source = "x = 1\r# sqlnbfmt: skip\rq = 'SELECT a, b FROM t WHERE c = 1'"
    assert format_cell_source(source, FormattingConfig(), None, LOGGER) == source


def test_cell_magic_with_carriage_return_keeps_its_sql():
    source = "%%sql\rSELECT a, b FROM t WHERE c = 1"
    formatted = format_cell_source(source, FormattingConfig(), None, LOGGER)
    assert formatted.startswith("%%sql\rSELECT\n")
    assert "FROM t" in formatted


def test_line_magic_keeps_the_lines_after_it():
    source = "# setup\n%sql select a, b from t where c = 1\nx = _\nprint(x)"
    formatted = format_cell_source(source, FormattingConfig(), None, LOGGER)
    assert formatted == "# setup\n%sql SELECT a, b FROM t WHERE c = 1\nx = _\nprint(x)"


# --- Fourth #34 review: literals around fields, comments, magics ------------


@pytest.mark.parametrize(
    ("source", "dialect"),
    [
        # sqlglot rewrites the quotes around a field; a value with ' then breaks
        ("q = f'SELECT a FROM t WHERE name = \"{name}\" AND b = 1'\n", "mysql"),
        ('q = f"SELECT a FROM t WHERE body = $${body}$$ AND b = 1"\n', "postgres"),
        (
            "q = f\"SELECT a FROM t WHERE REGEXP_CONTAINS(x, r'{pat}') AND b = 1\"\n",
            "bigquery",
        ),
        # Drivers fill in %s inside comments too; sqlglot moves or drops comments
        (
            (
                'cursor.execute("""SELECT a FROM t\nWHERE a = %s\n-- AND b = %s\n'
                'LIMIT %s""", p)\n'
            ),
            None,
        ),
        ('cursor.execute("SELECT a FROM t WHERE NOT /* %s */ a = %s", p)\n', None),
    ],
    ids=[
        "double-quoted-field",
        "dollar-quoted-field",
        "raw-string-field",
        "param-in-moved-comment",
        "param-in-dropped-comment",
    ],
)
def test_unsafe_rewrites_are_left_alone(source, dialect, caplog):
    caplog.set_level(logging.WARNING)
    assert fmt(source, dialect) == source
    assert "SQL formatting skipped" in caplog.text


def test_escaped_percent_is_not_a_parameter():
    source = 'cursor.execute("SELECT a %%s FROM t WHERE x = %s AND y = 1", p)\n'
    formatted = fmt(source)
    assert formatted.count("%%s") == 1
    assert formatted.count("%s") == source.count("%s")


@pytest.mark.parametrize(
    ("source", "dialect", "kept"),
    [
        (
            'spark.sql("SELECT * FROM {df} WHERE x > {v} AND y = 1", df=df, v=1)\n',
            "spark",
            ["{df}", "{v}"],
        ),
        ('con.sql("SELECT a FROM t WHERE x = $name AND y = 1")\n', "duckdb", ["$name"]),
    ],
    ids=["pyspark-args", "duckdb-named"],
)
def test_placeholders_printed_as_written_are_formatted(source, dialect, kept):
    try:
        sqlglot.parse(source.split('"')[1], read=dialect)
    except sqlglot.errors.ParseError:
        pytest.skip("this sqlglot version can't parse the syntax")
    formatted = fmt(source, dialect)
    assert formatted != source
    for placeholder in kept:
        assert formatted.count(placeholder) == 1


def test_block_comment_in_call_argument_is_stable():
    source = (
        'def _():\n    pd.read_sql("""SELECT a /* x\n  y */ FROM t WHERE b = 1""", c)\n'
    )
    once = fmt(source)
    assert once != source
    assert "/* x\n  y */" in once
    assert fmt(once) == once


def test_printer_bugs_are_caught_by_reparsing(monkeypatch, caplog):
    """A wrong but valid rewrite from sqlglot leaves the string as is."""
    caplog.set_level(logging.WARNING)
    original_sql = sqlglot.exp.Expression.sql

    def buggy_sql(self, *args, **kwargs):
        return original_sql(self, *args, **kwargs).replace("FROM t", "FROM u")

    monkeypatch.setattr(sqlglot.exp.Expression, "sql", buggy_sql)
    source = 'pd.read_sql("select a, b from t where c = 1 and d = 2")\n'
    assert fmt(source) == source
    assert "would read differently" in caplog.text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Magic arguments (an assignment, a connection) are kept
        ("%%sql df <<\nselect a from t", "%%sql df <<\nSELECT\n  a\nFROM t"),
        (
            "%%sql postgresql://u@h/db\r\nselect a from t",
            "%%sql postgresql://u@h/db\r\nSELECT\n  a\nFROM t",
        ),
        # Line breaks around the magic are kept
        ("# c\r\n%sql SELECT a FROM t", "# c\r\n%sql SELECT a FROM t"),
        ("%sql select a from t\r\nx = _", "%sql SELECT a FROM t\r\nx = _"),
        # String values in a line magic are kept
        ("%sql select 'a  b' from t", "%sql SELECT 'a  b' FROM t"),
        # A bare %sql has no SQL to format
        ("%sql\nselect a from t", "%sql\nselect a from t"),
    ],
    ids=[
        "assignment",
        "connection-crlf",
        "crlf-before",
        "crlf-after",
        "string-spaces",
        "bare-line-magic",
    ],
)
def test_magic_cells_keep_what_they_dont_format(source, expected):
    assert format_cell_source(source, FormattingConfig(), None, LOGGER) == expected
