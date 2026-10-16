import ast
import copy
import difflib
import logging
import re
import sys
import textwrap
import tomllib
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
import yaml
from sqlglot import errors, exp, parse
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import Token, TokenType

DEFAULT_SQL_KEYWORDS = frozenset(
    {
        "SELECT",
        "FROM",
        "WHERE",
        "INSERT",
        "UPDATE",
        "DELETE",
        "WITH",
        "JOIN",
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "UNION",
        "EXCEPT",
        "INTERSECT",
        "GROUP BY",
        "ORDER BY",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "DISTINCT",
    }
)

DEFAULT_FUNCTION_NAMES = frozenset(
    {
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "execute",
        "executescript",
        "fetchall",
        "fetchone",
        "fetchmany",
        "execute_query",
        "sql",
    }
)

DEFAULT_SQL_DECORATORS = frozenset(
    {
        "sql_decorator",
        "query",
        "sql_query",
        "db_query",
    }
)


@dataclass
class SQLReplacement:
    """A text replacement to apply to the source code."""

    start_offset: int
    end_offset: int
    new_text: str
    # The string literal being replaced and the template it must evaluate to
    node: ast.AST | None = None
    intended: tuple | None = None


@dataclass
class FormattingConfig:
    """Configuration for SQL formatting."""

    sql_keywords: set[str] = field(default_factory=lambda: set(DEFAULT_SQL_KEYWORDS))
    function_names: set[str] = field(
        default_factory=lambda: set(DEFAULT_FUNCTION_NAMES)
    )
    sql_decorators: set[str] = field(
        default_factory=lambda: set(DEFAULT_SQL_DECORATORS)
    )
    # Deprecated: never had any effect (sqlglot decides line breaks). Kept so
    # existing configs and FormattingConfig(...) calls keep working.
    single_line_threshold: int = 80
    indent_width: int = 4
    dialect: str | None = None


class SQLFormattingError(Exception):
    """Custom exception for SQL formatting errors."""


class UnsafeFormattingError(SQLFormattingError):
    """The SQL parses, but formatting it would change what it means; the string
    is left as is, with a warning."""


class PlaceholderMovedError(UnsafeFormattingError):
    """sqlglot would move or rewrap a query parameter or f-string field."""


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or has invalid values."""


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Sets up logging with the specified level."""
    logger = logging.getLogger("formatter")
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _string_set(config: dict, key: str, default: frozenset[str]) -> set[str]:
    value = config.get(key, default)
    if not isinstance(value, list | tuple | set | frozenset) or not all(
        isinstance(item, str) for item in value
    ):
        raise ConfigError(f"'{key}' must be a list of strings, got {value!r}")
    if any(not item.strip() for item in value):
        # Names match as substrings, so a blank entry would match everything
        raise ConfigError(f"'{key}' must not contain blank entries, got {value!r}")
    return set(value)


def _extended_string_set(config: dict, key: str, default: frozenset[str]) -> set[str]:
    """``key`` (or the defaults) plus anything listed under ``extend_<key>``."""
    return _string_set(config, key, default) | _string_set(
        config, f"extend_{key}", frozenset()
    )


def _integer(options: dict, key: str, default: int) -> int:
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer, got {value!r}")
    return value


def validate_dialect(dialect: str) -> str:
    """Returns ``dialect`` unchanged if sqlglot knows it, else raises ValueError.

    Without this check an unknown dialect makes every SQL string fail to parse,
    and since per-string failures are only warnings, ``--check`` would pass.
    """
    Dialect.get_or_raise(dialect)
    return dialect


def _optional_dialect(config: dict) -> str | None:
    value = config.get("dialect")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"'dialect' must be a string, got {value!r}")
    try:
        return validate_dialect(value)
    except ValueError as e:
        raise ConfigError(str(e)) from e


_LIST_KEYS = ("sql_keywords", "function_names", "sql_decorators")
_TOP_LEVEL_KEYS = {
    "dialect",
    "formatting_options",
    *_LIST_KEYS,
    *(f"extend_{key}" for key in _LIST_KEYS),
}
_FORMATTING_KEYS = {"single_line_threshold", "indent_width"}


def _suggest_key(key: str) -> str | None:
    """The full name of a known option that ``key`` likely meant, if any.

    Catches kebab-case spellings and formatting options placed at the wrong level.
    """
    normalized = key.replace("-", "_")
    if normalized in _TOP_LEVEL_KEYS:
        return normalized
    if normalized in _FORMATTING_KEYS:
        return f"formatting_options.{normalized}"
    return None


def _warn_unknown_keys(config: dict, options: dict) -> None:
    """Warns about unrecognized options, which would otherwise be silently ignored."""
    logger = logging.getLogger("formatter")
    unknown = [(key, key) for key in config.keys() - _TOP_LEVEL_KEYS]
    unknown += [
        (key, f"formatting_options.{key}") for key in options.keys() - _FORMATTING_KEYS
    ]
    for key, full_name in unknown:
        suggestion = _suggest_key(key)
        hint = f" (did you mean '{suggestion}'?)" if suggestion else ""
        logger.warning(f"Ignoring unknown configuration option '{full_name}'{hint}")


def _config_from_dict(config: dict) -> FormattingConfig:
    """Builds a FormattingConfig from a parsed YAML document or [tool.sqlnbfmt] table."""
    if not isinstance(config, dict):
        raise ConfigError(f"configuration must be a mapping, got {config!r}")
    options = config.get("formatting_options", {})
    if not isinstance(options, dict):
        raise ConfigError(f"'formatting_options' must be a mapping, got {options!r}")
    _warn_unknown_keys(config, options)
    if "single_line_threshold" in options:
        logging.getLogger("formatter").warning(
            "'formatting_options.single_line_threshold' is deprecated and has no "
            "effect; remove it from your configuration"
        )
    return FormattingConfig(
        # is_likely_sql matches keywords against upper-cased code
        sql_keywords={
            keyword.upper()
            for keyword in _extended_string_set(
                config, "sql_keywords", DEFAULT_SQL_KEYWORDS
            )
        },
        function_names=_extended_string_set(
            config, "function_names", DEFAULT_FUNCTION_NAMES
        ),
        sql_decorators=_extended_string_set(
            config, "sql_decorators", DEFAULT_SQL_DECORATORS
        ),
        single_line_threshold=_integer(options, "single_line_threshold", 80),
        indent_width=_integer(options, "indent_width", 4),
        dialect=_optional_dialect(config),
    )


def _read_pyproject_table(pyproject: Path) -> dict | None:
    """Returns the [tool.sqlnbfmt] table of a pyproject.toml, or None if absent."""
    try:
        with open(pyproject, "rb") as file:
            data = tomllib.load(file)
    except OSError as e:
        raise ConfigError(f"Could not read {pyproject}: {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Could not parse {pyproject}: {e}") from e
    return data.get("tool", {}).get("sqlnbfmt")


def find_pyproject_config(start: str | Path | None = None) -> Path | None:
    """Finds the nearest pyproject.toml with a [tool.sqlnbfmt] table.

    Searches ``start`` (default: the current directory) and its parents, stopping
    after the repository root (a directory containing ``.git``) or the filesystem
    root. Files without a [tool.sqlnbfmt] table are skipped, so a sub-package's
    pyproject.toml doesn't hide the repository-level configuration.
    """
    directory = Path(start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and _read_pyproject_table(pyproject) is not None:
            return pyproject
        if (candidate / ".git").exists():
            break
    return None


def load_config(config_path: str | Path | None = None) -> FormattingConfig:
    """Loads configuration from a YAML file or a pyproject.toml.

    A ``.toml`` path is read from its [tool.sqlnbfmt] table; any other path is
    parsed as YAML. Returns defaults if no path is given.
    """
    if config_path is None:
        return FormattingConfig()

    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    if config_file.suffix == ".toml":
        table = _read_pyproject_table(config_file)
        if table is None:
            raise ConfigError(f"{config_file} has no [tool.sqlnbfmt] table")
        return _config_from_dict(table)

    try:
        with open(config_file) as file:
            config = yaml.safe_load(file)
    except OSError as e:
        raise ConfigError(f"Could not read {config_file}: {e}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing the configuration file: {e}")
    return _config_from_dict(config or {})


# Line breaks as Python's tokenizer sees them (a lone "\r" counts too)
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def build_line_offsets(source: str) -> list:
    """Returns list where offsets[i] is the character offset of line i+1 in source."""
    offsets = [0]
    offsets.extend(match.end() for match in _LINE_BREAK.finditer(source))
    return offsets


def _char_column(source: str, line_start: int, byte_column: int) -> int:
    """Converts an AST column (UTF-8 bytes) to a character column."""
    # A line's first n characters encode to at least n bytes, so this slice
    # covers the byte column; the column always falls on a character boundary.
    prefix = source[line_start : line_start + byte_column]
    if prefix.isascii():
        return byte_column
    return len(prefix.encode("utf-8")[:byte_column].decode("utf-8"))


def node_offsets(node: ast.AST, line_offsets: list, source: str) -> tuple[int, int]:
    """Converts AST node line/col positions to (start, end) character offsets."""
    start_line = line_offsets[node.lineno - 1]
    end_line = line_offsets[node.end_lineno - 1]
    return (
        start_line + _char_column(source, start_line, node.col_offset),
        end_line + _char_column(source, end_line, node.end_col_offset),
    )


# Query parameter styles: %(name)s (pyformat), %s (format), ? (qmark),
# :name and :1 (named, numeric), $1. Matches inside SQL strings and comments
# are text, not parameters; sqlglot's tokenizer tells them apart.
_PARAMETER = re.compile(r"%%|%\(\w+\)s|%s|\?|(?<![:\w]):\w+|\$\d+")
# Drivers fill these in by position anywhere in the text, even in comments
_POSITIONAL = re.compile(r"%%|%s|\?")

# Tokens whose text is a value (string literals, quoted names), as opposed to
# SQL code. Comments aren't tokens: sqlglot attaches them to a neighbour.
# Looked up by name: older sqlglot versions lack some (NATIONAL_RAW_STRING).
_LITERAL_TOKENS = {
    getattr(TokenType, name)
    for name in (
        "STRING",
        "IDENTIFIER",
        "BIT_STRING",
        "HEX_STRING",
        "BYTE_STRING",
        "NATIONAL_STRING",
        "NATIONAL_RAW_STRING",
        "RAW_STRING",
        "HEREDOC_STRING",
        "UNICODE_STRING",
    )
    if hasattr(TokenType, name)
}


def _tokenize(sql: str, dialect: str | None) -> list[Token]:
    """sqlglot's tokens for ``sql``, so quoting follows the dialect's rules
    (e.g. ``'\\'`` is a complete string in postgres, $$...$$ is one string)."""
    try:
        return Dialect.get_or_raise(dialect).tokenize(sql)
    except errors.TokenError as e:
        raise SQLFormattingError(f"Failed to tokenize SQL code: {e}") from e


def _unique_prefix(text: str, base: str) -> str:
    """``base``, extended until it doesn't occur in ``text`` (case-insensitively)."""
    lowered = text.lower()
    prefix = base
    while prefix.lower() in lowered:
        prefix += "X"
    return prefix


_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _newlines_in_values(sql: str, dialect: str | None) -> set[int]:
    """Positions of line breaks inside a SQL string, quoted name or block comment.

    Re-indenting those would change a string's value, or make a comment grow
    on every run (sqlglot keeps comment text as written).
    """
    positions: set[int] = set()

    def add(start: int, text: str) -> None:
        positions.update(start + i for i, ch in enumerate(text) if ch == "\n")

    gap_start = 0  # text between tokens is whitespace and comments
    for token in [*_tokenize(sql, dialect), None]:
        gap_end = token.start if token else len(sql)
        for comment in _BLOCK_COMMENT.finditer(sql, gap_start, gap_end):
            add(comment.start(), comment.group(0))
        if token is None:
            break
        if token.token_type in _LITERAL_TOKENS:
            add(token.start, sql[token.start : token.end + 1])
        gap_start = token.end + 1
    return positions


def _stand_in_uses(
    sql: str, tokens: list[Token], stand_ins: set[str]
) -> tuple[dict[str, list[bool]], list[str]]:
    """Per stand-in, each use as a code token: whether it's wrapped in parens;
    and the source of each literal containing one (e.g. '%{v}%')."""
    uses: dict[str, list[bool]] = {token: [] for token in stand_ins}
    literals = []
    for i, token in enumerate(tokens):
        if token.token_type in _LITERAL_TOKENS:
            source = sql[token.start : token.end + 1]
            if any(stand_in in source for stand_in in stand_ins):
                literals.append(source)
            continue
        if token.text not in stand_ins:
            continue
        before = tokens[i - 1].token_type if i > 0 else None
        after = tokens[i + 1].token_type if i + 1 < len(tokens) else None
        uses[token.text].append(
            before == TokenType.L_PAREN and after == TokenType.R_PAREN
        )
    return {token: sorted(found) for token, found in uses.items()}, sorted(literals)


def _unquoted(statements: list) -> list:
    """The statements with identifiers unquoted, for comparing meaning: sqlglot
    may quote a name the dialect reserves (rank -> `rank`), which is harmless."""

    def unquote(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Identifier) and node.quoted:
            return exp.Identifier(this=node.this, quoted=False)
        return node

    return [stmt.transform(unquote) if stmt else stmt for stmt in statements]


def _check_placeholders(
    statements: list,
    original_sql: str,
    formatted_sql: str,
    dialect: str | None,
    stand_ins: list[str],
) -> None:
    """Raises PlaceholderMovedError if formatting moved or reinterpreted one.

    Stand-ins hold code we can't see (a parameter's value, an f-string field
    that may be a whole clause), so formatting must leave each one where it was:

    - Each one is still a code token, as often as before: not moved into a
      string (INTERVAL %s DAY -> INTERVAL '%s DAY') or a comment. A string
      holding one is written exactly as before: a field's value may contain
      quotes or backslashes, so "{name}" -> '{name}' or $${v}$$ -> '{v}' breaks.
    - No parentheses added around it (VALUES %s -> VALUES (%s) breaks
      psycopg2's execute_values) and no alias made of it (SELECT {distinct} a
      -> SELECT {distinct} AS a).
    - A stand-in isn't turned into an alias, or into the expression an alias
      names, unless the original wrote AS there: in f"FROM t {where_clause}"
      sqlglot reads the field as a table alias and prints FROM t AS {where_clause}.
    """
    if not stand_ins:
        return
    names = set(stand_ins)
    original_tokens = _tokenize(original_sql, dialect)
    formatted_tokens = _tokenize(formatted_sql, dialect)
    for token in stand_ins:
        if formatted_sql.count(token) != original_sql.count(token):
            raise PlaceholderMovedError(f"formatting would alter placeholder {token}")
    if _stand_in_uses(formatted_sql, formatted_tokens, names) != _stand_in_uses(
        original_sql, original_tokens, names
    ):
        raise PlaceholderMovedError(
            "formatting would quote or parenthesize a query parameter or f-string field"
        )
    upper_stand_ins = {token.upper() for token in stand_ins}
    for statement in statements:
        for alias in statement.find_all(exp.Alias, exp.TableAlias):
            name = alias.alias if isinstance(alias, exp.Alias) else alias.name
            if name.upper() in upper_stand_ins and not re.search(
                rf"\bAS\s+{re.escape(name)}\b", original_sql, re.IGNORECASE
            ):
                raise PlaceholderMovedError(
                    "formatting would turn an f-string field or parameter into an alias"
                )
            # SELECT {distinct} a: the field isn't the aliased expression
            aliased = alias.this
            if (
                isinstance(alias, exp.Alias)
                and isinstance(aliased, exp.Column)
                and not aliased.table
                and aliased.name.upper() in upper_stand_ins
                and not re.search(
                    rf"\b{re.escape(aliased.name)}\s+AS\b", original_sql, re.IGNORECASE
                )
            ):
                raise PlaceholderMovedError(
                    "formatting would alias an f-string field or parameter"
                )


def format_sql_code(
    sql_code: str,
    dialect: str | None,
    config: FormattingConfig,
    placeholders: Iterable[str] | None = None,
    force_single_line: bool = False,
    is_magic_command: bool = False,
    is_cell_magic: bool = False,
) -> str:
    """
    Formats SQL code using sqlglot's native formatting capabilities.

    Args:
        sql_code (str): The original SQL code to format.
        dialect (Optional[str]): The SQL dialect to use (e.g., 'postgres', 'mysql').
        config (FormattingConfig): The formatting configuration.
        placeholders (Optional[Iterable[str]]): Stand-in tokens for f-string fields
            that are already present in ``sql_code`` (see extract_fstring_parts).
            They come back unchanged; the caller restores the fields.
        force_single_line (bool): Whether to force the formatted SQL into a single line.
        is_magic_command (bool): Whether the SQL code comes from a magic command.
        is_cell_magic (bool): Whether the SQL code comes from a cell magic command.

    Returns:
        str: The formatted SQL code.
    """
    try:
        logger = logging.getLogger("formatter")

        if not sql_code.strip():
            return sql_code

        temp_sql = sql_code

        # Query parameters (%s, ?, %(name)s, :name, $1) and f-string fields
        # (already replaced by the caller) are swapped for plain identifiers:
        # every dialect prints those unchanged, whereas sqlglot rewrites
        # parameters per dialect (%(name)s, $name, @name). The prefix never
        # occurs in the SQL, so stand-ins can't collide with the user's text.
        prefix = _unique_prefix(temp_sql, "SQLNBFMT_P")
        parameters: dict[str, str] = {}
        is_code = bytearray(len(temp_sql))  # 1 where a code token is
        for token in _tokenize(temp_sql, dialect):
            if token.token_type not in _LITERAL_TOKENS:
                is_code[token.start : token.end + 1] = b"\x01" * (
                    token.end + 1 - token.start
                )

        def stand_in(match: re.Match) -> str:
            if match.group(0) == "%%" or not all(is_code[match.start() : match.end()]):
                return match.group(0)  # in a string or comment: text, keep as is
            token = f"{prefix}{len(parameters)}_"
            parameters[token] = match.group(0)
            return token

        temp_sql = _PARAMETER.sub(stand_in, temp_sql)
        stand_ins = [*(placeholders or ()), *parameters]

        temp_sql = temp_sql.strip()

        # Parse and format SQL
        statements = parse(temp_sql, read=dialect)
        # Placeholders left in the tree are styles we don't swap (PySpark's
        # {name}, duckdb's $name); fine if printed back as written, otherwise
        # sqlglot would rewrite them for the dialect
        for stmt in statements:
            for placeholder in stmt.find_all(exp.Placeholder) if stmt else ():
                if placeholder.sql(dialect=dialect) not in temp_sql:
                    raise PlaceholderMovedError(
                        "the SQL has a query parameter that formatting could rewrite"
                    )

        # A line magic must stay on one line; collapsing whitespace instead
        # would change strings ('a  b' -> 'a b')
        single_line = force_single_line or (is_magic_command and not is_cell_magic)
        formatted_parts = []
        for stmt in statements:
            formatted_parts.append(
                stmt.sql(
                    pretty=not single_line,
                    indent=config.indent_width,
                    dialect=dialect,
                )
            )
        separator = "; " if single_line else ";\n"
        formatted_sql = separator.join(formatted_parts).strip()
        if single_line and "\n" in formatted_sql:
            raise SQLFormattingError("the SQL doesn't fit on one line")

        _check_placeholders(statements, temp_sql, formatted_sql, dialect, stand_ins)

        # The output must mean what the input did: guards against printer bugs
        # in the sqlglot versions we support (e.g. before 30.23, pretty output
        # indented the second line of a quoted name like `x\ny`)
        if _unquoted(parse(formatted_sql, read=dialect)) != _unquoted(statements):
            raise UnsafeFormattingError("the formatted SQL would read differently")

        # %s and ? bind by position, wherever they are (drivers fill them in
        # even inside comments), so their order must not change: sqlglot
        # reorders OFFSET/LIMIT, turns LIMIT ? into T-SQL TOP (?), and moves
        # or drops comments. Stand-ins make each parameter distinguishable.
        positional = [t for t, p in parameters.items() if p in ("%s", "?")]
        order = re.compile("|".join([*map(re.escape, positional), _POSITIONAL.pattern]))

        def positional_order(text: str) -> list[str]:
            return [m for m in order.findall(text) if m != "%%"]

        if positional_order(formatted_sql) != positional_order(temp_sql):
            raise PlaceholderMovedError(
                "formatting would reorder positional query parameters"
            )

        # Restore query parameters; f-string fields are restored by the caller
        for token, original in parameters.items():
            formatted_sql = formatted_sql.replace(token, original)

        # Logging for debugging purposes
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        return formatted_sql

    except SQLFormattingError:
        raise  # our own errors (e.g. UnsafeFormattingError) keep their type
    except errors.ParseError as e:
        raise SQLFormattingError(f"Failed to parse SQL code: {e}")
    except Exception as e:  # noqa: BLE001 - sqlglot raises many types; normalize them
        raise SQLFormattingError(f"Unexpected error during SQL formatting: {e}")


_STRING_PREFIX = re.compile(r"[A-Za-z]*")


def _normalize_template(parts: list[tuple[str, str]]) -> tuple:
    merged: list[tuple[str, str]] = []
    for kind, content in parts:
        if kind == "text" and not content:
            continue
        if kind == "text" and merged and merged[-1][0] == "text":
            merged[-1] = ("text", merged[-1][1] + content)
        else:
            merged.append((kind, content))
    return ("fstring", tuple(merged))


def literal_template(node: ast.AST) -> tuple | None:
    """What a string literal evaluates to: its value, or for an f-string its text
    and replacement fields. Used to check that rewrites keep the meaning."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ("str", node.value)
    if isinstance(node, ast.JoinedStr):
        return _normalize_template(
            [
                ("text", part.value)
                if isinstance(part, ast.Constant)
                else ("expr", ast.dump(part))
                for part in node.values
            ]
        )
    return None


def _string_literals(tree: ast.AST) -> list[ast.AST]:
    """Top-level string literals (not the parts inside f-strings), in source order."""
    found = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if literal_template(node) is not None:
            found.append(node)
        else:
            stack.extend(ast.iter_child_nodes(node))
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def _splice(source: str, replacements: list[SQLReplacement]) -> str:
    result = source
    for rep in sorted(replacements, key=lambda r: r.start_offset, reverse=True):
        result = result[: rep.start_offset] + rep.new_text + result[rep.end_offset :]
    return result


def _is_faithful(
    tree: ast.AST, result: str, replacements: list[SQLReplacement]
) -> bool:
    """The edited source parses, with the same string literals in the same order,
    untouched ones unchanged and rewritten ones equal to their intended template."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_tree = ast.parse(result)
    except SyntaxError:
        return False
    intended = {id(rep.node): rep.intended for rep in replacements}
    old_literals = _string_literals(tree)
    new_literals = _string_literals(new_tree)
    return len(old_literals) == len(new_literals) and all(
        literal_template(new) == intended.get(id(old), literal_template(old))
        for old, new in zip(old_literals, new_literals)
    )


def _apply_replacements(
    source: str,
    tree: ast.AST,
    replacements: list[SQLReplacement],
    logger: logging.Logger,
) -> str:
    """Applies the replacements that keep the code's meaning (safety net).

    If the combined result fails the check, each rewrite is checked on its own;
    failing ones are reported by line and dropped, the others still applied.
    """
    result = _splice(source, replacements)
    if _is_faithful(tree, result, replacements):
        return result

    good = []
    for rep in replacements:
        if _is_faithful(tree, _splice(source, [rep]), [rep]):
            good.append(rep)
        else:
            logger.warning(
                f"line {rep.node.lineno}: SQL formatting skipped: the rewrite failed "
                "the safety check, so this string was left unchanged. Please report "
                "this at https://github.com/flyersworder/sqlnbfmt/issues"
            )
    if good and len(good) < len(replacements):
        result = _splice(source, good)
        if _is_faithful(tree, result, good):
            return result
    if good:
        logger.warning(
            "SQL formatting failed the safety check, so the code was left unchanged. "
            "Please report this at https://github.com/flyersworder/sqlnbfmt/issues"
        )
    return source


class SQLStringFormatter(ast.NodeVisitor):
    """AST NodeVisitor that collects SQL string replacements."""

    def __init__(
        self,
        config: FormattingConfig,
        dialect: str | None,
        logger: logging.Logger,
        line_offsets: list,
        source: str,
    ):
        super().__init__()
        self.config = config
        self.dialect = dialect
        self.logger = logger
        self.line_offsets = line_offsets
        self.source = source
        self.replacements: list = []
        # ids of decorator calls matching sql_decorators, e.g. @query("SELECT ..."),
        # recorded by visit_FunctionDef/visit_ClassDef before generic_visit
        # reaches them. visit_Call then formats their arguments, so each call is
        # handled in one place even if it also matches function_names.
        self._sql_decorator_calls: set[int] = set()

    def is_likely_sql(self, code: str) -> bool:
        """Enhanced SQL detection with better heuristics."""
        if not code or len(code.strip()) < 10:
            return False

        if re.match(r"^\s*(/|[a-zA-Z]:\\|https?://|<!DOCTYPE|<html)", code.strip()):
            return False

        upper_code = code.upper()
        keyword_count = sum(
            1
            for keyword in self.config.sql_keywords
            if re.search(rf"\b{re.escape(keyword)}\b", upper_code)
        )

        has_sql_pattern = bool(
            re.search(
                r"\bSELECT\b.*\bFROM\b|\bUPDATE\b.*\bSET\b|\bINSERT\b.*\bINTO\b|\bDELETE\b.*\bFROM\b",
                upper_code,
                re.DOTALL,
            )
        )

        return keyword_count >= 2 or has_sql_pattern

    @staticmethod
    def extract_fstring_parts(
        node: ast.JoinedStr,
    ) -> tuple[str, dict[str, ast.FormattedValue]]:
        """The f-string's SQL with a token for each ``{...}``, and token -> node.

        Tokens are plain identifiers with a prefix that doesn't occur in the
        f-string's text, and a trailing "_" so _1_ is not a prefix of _10_.
        """
        text = "".join(v.value for v in node.values if isinstance(v, ast.Constant))
        prefix = _unique_prefix(text, "SQLNBFMT_F")
        sql_parts = []
        placeholders = {}
        for value in node.values:
            if isinstance(value, ast.Constant):
                sql_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                placeholder = f"{prefix}{len(placeholders)}_"
                sql_parts.append(placeholder)
                placeholders[placeholder] = value
        return "".join(sql_parts), placeholders

    def format_sql_node(
        self,
        node: ast.Constant | ast.JoinedStr,
        force_single_line: bool = False,
        block_indent: str | None = None,
    ) -> tuple[str, tuple] | None:
        """Formats SQL in a string literal: (replacement literal, template) or None."""
        try:
            placeholders = {}

            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if not self.is_likely_sql(node.value):
                    return None
                sql_str = node.value
            elif isinstance(node, ast.JoinedStr):
                sql_str, placeholders = self.extract_fstring_parts(node)
                if not sql_str or not self.is_likely_sql(sql_str):
                    return None
            else:
                return None

            try:
                formatted_sql = format_sql_code(
                    sql_str,
                    self.dialect,
                    self.config,
                    placeholders=placeholders or None,
                    force_single_line=force_single_line,
                )
            except UnsafeFormattingError as e:
                self.logger.warning(f"line {node.lineno}: SQL formatting skipped: {e}")
                return None
            except SQLFormattingError:
                return None

            if formatted_sql == textwrap.dedent(sql_str).strip():
                return None

            replacement = self._build_replacement_text(
                formatted_sql, node, block_indent, placeholders
            )
            if replacement is None:
                self.logger.warning(
                    f"line {node.lineno}: SQL formatting skipped: the formatted SQL "
                    "can't be written back as an equivalent string literal"
                )
                return None
            return replacement

        except SQLFormattingError as e:
            self.logger.warning(f"SQL formatting skipped: {e}")
            return None
        except Exception as e:  # noqa: BLE001 - one bad string must not abort the file
            self.logger.warning(f"Unexpected error during SQL formatting: {e}")
            return None

    def _build_replacement_text(
        self,
        formatted_sql: str,
        node: ast.Constant | ast.JoinedStr,
        block_indent: str | None,
        placeholders: dict[str, ast.FormattedValue],
    ) -> tuple[str, tuple] | None:
        """Constructs the replacement literal and its template, or None if unsafe.

        With ``block_indent`` (SQL passed to a call), the SQL lines and closing
        quotes are indented by it; where the block starts and what follows it is
        decided by the caller, see ``_layout_call_argument``.
        """
        if block_indent is not None and "\n" in formatted_sql:
            # Lines continuing a multi-line SQL string are part of its value
            try:
                in_string = _newlines_in_values(formatted_sql, self.dialect)
            except SQLFormattingError:
                return None
            lines, position = [], 0
            for i, line in enumerate(formatted_sql.split("\n")):
                continues_string = i > 0 and position - 1 in in_string
                indent = "" if continues_string or not line.strip() else block_indent
                lines.append(indent + line)
                position += len(line) + 1
            indented_sql = "\n".join(lines)
            return self._to_literal(
                f"\n{indented_sql}\n{block_indent}", node, placeholders
            )
        if "\n" in formatted_sql:
            return self._to_literal(f"\n{formatted_sql}\n", node, placeholders)
        return self._to_literal(formatted_sql, node, placeholders)

    def _to_literal(
        self,
        text: str,
        node: ast.Constant | ast.JoinedStr,
        placeholders: dict[str, ast.FormattedValue],
    ) -> tuple[str, tuple] | None:
        """Writes ``text`` as a literal replacing ``node``: (literal, template).

        ``text`` is the intended value, with f-string placeholder tokens in place
        of ``{...}``. Keeps the original raw (``r``) prefix, escapes backslashes
        otherwise, doubles literal braces in f-strings, and picks quotes that
        don't clash. The literal is parsed back and must match the intended
        template (see ``literal_template``); otherwise None.
        """
        start = self._offsets(node)[0]
        raw = "r" in _STRING_PREFIX.match(self.source, start).group().lower()
        is_fstring = isinstance(node, ast.JoinedStr)
        prefix = ("r" if raw else "") + ("f" if is_fstring else "")

        def escape(part: str) -> str:
            if not raw:
                part = part.replace("\\", "\\\\")
            if is_fstring:
                part = part.replace("{", "{{").replace("}", "}}")
            return part

        body_parts, template_parts = [], []
        if placeholders:
            token = re.compile("(" + "|".join(map(re.escape, placeholders)) + ")")
            pieces = token.split(text)
        else:
            pieces = [text]
        for piece in pieces:
            if piece in placeholders:
                # The field's own source, e.g. {d["k"]!r:>3}, copied verbatim
                start, end = self._offsets(placeholders[piece])
                body_parts.append(self.source[start:end])
                template_parts.append(("expr", ast.dump(placeholders[piece])))
            else:
                body_parts.append(escape(piece))
                template_parts.append(("text", piece))
        body = "".join(body_parts)
        intended = _normalize_template(template_parts) if is_fstring else ("str", text)
        if raw and body.endswith("\\"):
            return None

        quotes = ('"""', "'''") if "\n" in body else ('"', "'", '"""', "'''")
        for quote in quotes:
            if quote in body or body.endswith(quote[0]):
                continue
            literal = f"{prefix}{quote}{body}{quote}"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # e.g. invalid escape sequences
                    parsed = ast.parse(literal, mode="eval").body
            except (SyntaxError, SyntaxWarning):
                continue
            if literal_template(parsed) == intended:
                return literal, intended
        return None

    def _offsets(self, node: ast.AST) -> tuple[int, int]:
        return node_offsets(node, self.line_offsets, self.source)

    def _replace_string(self, value: ast.expr | None) -> None:
        """Formats a string literal in place (assigned or returned SQL)."""
        if isinstance(value, (ast.Constant, ast.JoinedStr)):
            replacement = self.format_sql_node(value)
            if replacement:
                literal, intended = replacement
                start, end = self._offsets(value)
                self.replacements.append(
                    SQLReplacement(start, end, literal, value, intended)
                )

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handles assignments."""
        self._replace_string(node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Handles function calls, including SQL decorators with arguments."""
        func_name = self.get_full_func_name(node.func)
        if id(node) in self._sql_decorator_calls or any(
            name in func_name for name in self.config.function_names
        ):
            values = [*node.args, *(keyword.value for keyword in node.keywords)]
            for value in values:
                if isinstance(value, (ast.Constant, ast.JoinedStr)):
                    self._replace_call_argument(node, value)
        self.generic_visit(node)

    def _is_sql_decorator(self, decorator: ast.expr) -> bool:
        """Whether a decorator (``@query`` or ``@query(...)``) is in sql_decorators.

        Names match as substrings of the dotted name, like function_names.
        """
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = self.get_full_func_name(target)
        return any(
            decorator_name in name for decorator_name in self.config.sql_decorators
        )

    def _register_sql_decorators(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> bool:
        """Marks SQL decorator calls for visit_Call; returns whether any matched."""
        matched = False
        for decorator in node.decorator_list:
            if self._is_sql_decorator(decorator):
                matched = True
                if isinstance(decorator, ast.Call):
                    self._sql_decorator_calls.add(id(decorator))
        return matched

    @staticmethod
    def _own_returns(func: ast.FunctionDef | ast.AsyncFunctionDef):
        """Return statements of ``func`` itself, not of nested functions or classes."""
        stack = list(func.body)
        while stack:
            node = stack.pop()
            if isinstance(node, ast.Return):
                yield node
            elif not isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
            ):
                stack.extend(ast.iter_child_nodes(node))

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Formats SQL returned by functions decorated with an SQL decorator."""
        if self._register_sql_decorators(node):
            for ret in self._own_returns(node):
                self._replace_string(ret.value)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Registers SQL decorator calls on classes, e.g. @query("SELECT ...")."""
        self._register_sql_decorators(node)
        self.generic_visit(node)

    def _replace_call_argument(
        self, call: ast.Call, arg: ast.Constant | ast.JoinedStr
    ) -> None:
        """Formats a SQL argument, indented relative to the line the call starts on."""
        call_line_start = self.line_offsets[call.lineno - 1]
        call_line = _LINE_BREAK.split(self.source[call_line_start:], maxsplit=1)[0]
        base_indent = call_line[: len(call_line) - len(call_line.lstrip())]
        block_indent = base_indent + " " * self.config.indent_width

        replacement = self.format_sql_node(arg, block_indent=block_indent)
        if not replacement:
            return
        literal, intended = replacement
        start, end = self._offsets(arg)
        if "\n" in literal:
            start, end, literal = self._layout_call_argument(
                start, end, literal, base_indent, block_indent
            )
        self.replacements.append(SQLReplacement(start, end, literal, arg, intended))

    def _layout_call_argument(
        self,
        start: int,
        end: int,
        block: str,
        base_indent: str,
        block_indent: str,
    ) -> tuple[int, int, str]:
        """Places a multi-line SQL block within its call.

        - Already on its own line: normalize that line's indentation.
        - First thing after ``(``: break after the paren.
        - Otherwise (after a keyword or another argument): open the quotes in
          place and leave what follows (e.g. the closing paren) after the
          closing quotes.

        In the first two cases a directly following ``)`` moves to its own line at
        the call's indentation; other arguments stay after the closing quotes, so
        code outside the string is never rewritten beyond whitespace.
        """
        line_start = (
            max(self.source.rfind("\n", 0, start), self.source.rfind("\r", 0, start))
            + 1
        )
        before = self.source[line_start:start]
        if not before.strip():
            start, block = line_start, block_indent + block
        elif before.rstrip().endswith("("):
            start -= len(before) - len(before.rstrip())
            block = "\n" + block_indent + block
        else:
            return start, end, block

        after = self.source[end:]
        rest = after.lstrip(" \t")
        if rest.startswith(")"):
            end += len(after) - len(rest)
            block += "\n" + base_indent
        return start, end, block

    @staticmethod
    def get_full_func_name(node: ast.AST) -> str:
        """Gets the full function name from an AST node."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))


def _has_skip_hint(source: str) -> bool:
    """Check if any comment line contains the sqlnbfmt: skip directive.

    Only dedicated comment lines are recognised (the line must start with ``#``
    after stripping whitespace).  Inline trailing comments like
    ``x = 1  # sqlnbfmt: skip`` are intentionally ignored so that the directive
    cannot be triggered accidentally from within an expression or string.
    """
    return any(
        line.strip().startswith("#") and "sqlnbfmt: skip" in line
        for line in _LINE_BREAK.split(source)
    )


def format_cell_source(
    code: str,
    config: FormattingConfig,
    dialect: str | None,
    logger: logging.Logger,
) -> str:
    """Format SQL in a single cell's source code.

    Handles both magic commands (%%sql, %sql) and regular Python code
    with embedded SQL strings. Returns the formatted source.
    """
    if not code.strip():
        return code

    if _has_skip_hint(code):
        return code

    lines = _LINE_BREAK.split(code)
    line_starts = build_line_offsets(code)

    # Look for magic commands
    magic_cmd = None
    magic_cmd_index = None

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith(("%%sql", "%sql")):
            magic_cmd = stripped.split()[0]
            magic_cmd_index = idx
            break
        else:
            break

    if magic_cmd:
        is_cell_magic = magic_cmd.startswith("%%sql")
        magic_line = lines[magic_cmd_index]
        line_start = line_starts[magic_cmd_index]
        line_end = line_start + len(magic_line)
        # The %%sql line (with arguments such as "df <<" or a connection) and
        # everything before it, line breaks included, are kept as written
        body_start = (
            line_starts[magic_cmd_index + 1]
            if magic_cmd_index + 1 < len(line_starts)
            else len(code)
        )

        if is_cell_magic:
            sql_code = code[body_start:].strip()
        else:
            sql_code = magic_line.strip()[len(magic_cmd) :].strip()
        if not sql_code:
            return code

        try:
            formatted_sql = format_sql_code(
                sql_code,
                dialect,
                config,
                is_magic_command=True,
                is_cell_magic=is_cell_magic,
            )

            if is_cell_magic:
                if body_start == line_end:  # no line break after the magic line
                    return f"{code[:line_end]}\n{formatted_sql}"
                return code[:body_start] + formatted_sql
            # Keep the lines before and after the line magic as they are
            indent = magic_line[: len(magic_line) - len(magic_line.lstrip())]
            return (
                f"{code[:line_start]}{indent}{magic_cmd} {formatted_sql}"
                + code[line_end:]
            )

        except SQLFormattingError as e:
            logger.warning(f"SQL magic formatting skipped: {e}")
            return code

    # Handle regular Python code with surgical string replacement
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.warning(f"Failed to parse cell:\n{code}")
        return code

    line_offsets = build_line_offsets(code)
    formatter = SQLStringFormatter(config, dialect, logger, line_offsets, code)
    try:
        formatter.visit(tree)
    except Exception as e:  # noqa: BLE001 - never abort a whole run over one cell or file
        logger.warning(f"SQL formatting skipped: unexpected error: {e}")
        return code

    if not formatter.replacements:
        return code

    return _apply_replacements(code, tree, formatter.replacements, logger)


def _format_cells(
    notebook: nbformat.NotebookNode,
    config: FormattingConfig,
    dialect: str | None,
    logger: logging.Logger,
) -> bool:
    """Format cells in-place. Returns True if any cell was modified."""
    changed = False

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        original_code = cell.source
        formatted = format_cell_source(original_code, config, dialect, logger)
        if formatted != original_code:
            cell.source = formatted
            changed = True

    return changed


def _format_python_source(
    source: str,
    config: FormattingConfig,
    dialect: str | None,
    logger: logging.Logger,
) -> tuple[str, bool]:
    """Format SQL strings in Python source code.

    Returns (formatted_source, changed) tuple.
    """
    if not source.strip():
        return source, False

    if _has_skip_hint(source):
        return source, False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning("Failed to parse Python source")
        return source, False

    line_offsets = build_line_offsets(source)
    formatter = SQLStringFormatter(config, dialect, logger, line_offsets, source)
    try:
        formatter.visit(tree)
    except Exception as e:  # noqa: BLE001 - never abort a whole run over one cell or file
        logger.warning(f"SQL formatting skipped: unexpected error: {e}")
        return source, False

    if not formatter.replacements:
        return source, False

    result = _apply_replacements(source, tree, formatter.replacements, logger)
    return result, result != source


def process_notebook(
    notebook_path: str | Path,
    config: FormattingConfig,
    dialect: str | None,
    logger: logging.Logger,
    check_only: bool = False,
) -> bool:
    """Processes a Jupyter notebook.

    Args:
        check_only: If True, detect changes but don't write the file.

    Returns:
        True if changes were made (or would be made in check mode).
    """
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
        changed = _format_cells(notebook, config, dialect, logger)

        if changed and not check_only:
            nbformat.write(notebook, notebook_path)
            logger.info(f"Updated notebook: {notebook_path}")

        return changed

    except Exception:
        logger.exception(f"Failed to process notebook {notebook_path}")
        raise


def diff_notebook(
    notebook_path: str | Path,
    config: FormattingConfig,
    dialect: str | None,
    logger: logging.Logger,
) -> str:
    """Returns a unified diff of formatting changes for a notebook."""
    notebook_path = Path(notebook_path)
    original_nb = nbformat.read(notebook_path, as_version=4)
    formatted_nb = copy.deepcopy(original_nb)

    _format_cells(formatted_nb, config, dialect, logger)

    # Build unified diff cell-by-cell
    diff_lines = []
    for i, (orig_cell, fmt_cell) in enumerate(
        zip(original_nb.cells, formatted_nb.cells)
    ):
        if orig_cell.source != fmt_cell.source:
            diff_lines.extend(
                difflib.unified_diff(
                    orig_cell.source.splitlines(keepends=True),
                    fmt_cell.source.splitlines(keepends=True),
                    fromfile=f"{notebook_path} Cell [{i}] original",
                    tofile=f"{notebook_path} Cell [{i}] formatted",
                )
            )
    return "".join(diff_lines)


def process_python_file(
    file_path: str | Path,
    config: FormattingConfig,
    dialect: str | None,
    logger: logging.Logger,
    check_only: bool = False,
) -> bool:
    """Processes a Python file (e.g. Marimo notebook).

    Returns True if changes were made (or would be made in check mode).
    """
    try:
        file_path = Path(file_path)
        source = file_path.read_text(encoding="utf-8")
        formatted, changed = _format_python_source(source, config, dialect, logger)

        if changed and not check_only:
            file_path.write_text(formatted, encoding="utf-8")
            logger.info(f"Updated file: {file_path}")

        return changed

    except Exception:
        logger.exception(f"Failed to process file {file_path}")
        raise


def diff_python_file(
    file_path: str | Path,
    config: FormattingConfig,
    dialect: str | None,
    logger: logging.Logger,
) -> str:
    """Returns a unified diff of formatting changes for a Python file."""
    file_path = Path(file_path)
    source = file_path.read_text(encoding="utf-8")
    formatted, _ = _format_python_source(source, config, dialect, logger)

    if source == formatted:
        return ""

    return "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            formatted.splitlines(keepends=True),
            fromfile=f"{file_path} original",
            tofile=f"{file_path} formatted",
        )
    )


def _dialect_arg(value: str) -> str:
    """argparse type for --dialect: rejects unknown dialects up front."""
    import argparse

    try:
        return validate_dialect(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def main():
    """Main entry point for the SQL formatter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Format SQL code in Jupyter notebooks and Python files"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Notebook (.ipynb) or Python (.py) file paths to process",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Configuration file path: YAML, or a pyproject.toml with a "
        "[tool.sqlnbfmt] table (optional; by default the nearest pyproject.toml "
        "with that table is used)",
    )
    parser.add_argument(
        "--dialect",
        type=_dialect_arg,
        help="SQL dialect (e.g., mysql, postgres); overrides the configured dialect",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check formatting without modifying files (exit code 1 if changes needed)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Print a unified diff of formatting changes",
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    config_path = args.config
    try:
        if config_path is None:
            config_path = find_pyproject_config()
        config = load_config(config_path)
    except FileNotFoundError as e:
        logger.error(
            f"{e}\nHint: --config is optional; omit it to use built-in defaults."
        )
        sys.exit(1)
    except (ConfigError, yaml.YAMLError) as e:
        location = f" in {config_path}" if config_path else ""
        logger.error(f"Invalid configuration{location}: {e}")
        sys.exit(1)
    logger.debug(f"Using configuration from {config_path or 'built-in defaults'}")
    dialect = args.dialect or config.dialect

    try:
        changed_files = []

        for file_path in args.files:
            is_python = file_path.suffix == ".py"

            if args.check or args.diff:
                if is_python:
                    diff_output = diff_python_file(file_path, config, dialect, logger)
                else:
                    diff_output = diff_notebook(file_path, config, dialect, logger)
                needs_formatting = bool(diff_output)

                if args.diff and diff_output:
                    print(diff_output, end="")

                if needs_formatting:
                    changed_files.append(file_path)
            else:
                if is_python:
                    changed = process_python_file(file_path, config, dialect, logger)
                else:
                    changed = process_notebook(file_path, config, dialect, logger)
                if changed:
                    changed_files.append(file_path)

        if args.check:
            if changed_files:
                logger.error(f"{len(changed_files)} file(s) would be reformatted:")
                for f in changed_files:
                    logger.error(f"  - {f}")
                sys.exit(1)
            else:
                logger.info("All files are properly formatted.")
                sys.exit(0)

        # Print summary for normal mode
        if not args.diff:
            if changed_files:
                logger.info("Changes made to the following files:")
                for file in changed_files:
                    logger.info(f"  - {file}")
            else:
                logger.info("No changes needed. All files are properly formatted.")

        sys.exit(0)

    except Exception as e:  # noqa: BLE001 - CLI boundary: report and exit non-zero
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
