import ast
import copy
import difflib
import logging
import re
import sys
import textwrap
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
import yaml
from sqlglot import errors, parse
from sqlglot.dialects.dialect import Dialect

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
    single_line_threshold: int = 80
    indent_width: int = 4
    dialect: str | None = None


class SQLFormattingError(Exception):
    """Custom exception for SQL formatting errors."""


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
    return set(value)


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


_TOP_LEVEL_KEYS = {
    "dialect",
    "sql_keywords",
    "function_names",
    "sql_decorators",
    "formatting_options",
}
_FORMATTING_KEYS = {"single_line_threshold", "indent_width"}


def _warn_unknown_keys(config: dict, options: dict) -> None:
    """Warns about unrecognized options, which would otherwise be silently ignored."""
    logger = logging.getLogger("formatter")
    for key in config.keys() - _TOP_LEVEL_KEYS:
        hint = (
            f" (did you mean 'formatting_options.{key}'?)"
            if key in _FORMATTING_KEYS
            else ""
        )
        logger.warning(f"Ignoring unknown configuration option '{key}'{hint}")
    for key in options.keys() - _FORMATTING_KEYS:
        logger.warning(
            f"Ignoring unknown configuration option 'formatting_options.{key}'"
        )


def _config_from_dict(config: dict) -> FormattingConfig:
    """Builds a FormattingConfig from a parsed YAML document or [tool.sqlnbfmt] table."""
    if not isinstance(config, dict):
        raise ConfigError(f"configuration must be a mapping, got {config!r}")
    options = config.get("formatting_options", {})
    if not isinstance(options, dict):
        raise ConfigError(f"'formatting_options' must be a mapping, got {options!r}")
    _warn_unknown_keys(config, options)
    return FormattingConfig(
        sql_keywords=_string_set(config, "sql_keywords", DEFAULT_SQL_KEYWORDS),
        function_names=_string_set(config, "function_names", DEFAULT_FUNCTION_NAMES),
        sql_decorators=_string_set(config, "sql_decorators", DEFAULT_SQL_DECORATORS),
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


def build_line_offsets(source: str) -> list:
    """Returns list where offsets[i] is the character offset of line i+1 in source."""
    offsets = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def node_offsets(node: ast.AST, line_offsets: list) -> tuple[int, int]:
    """Converts AST node line/col positions to (start, end) character offsets."""
    start = line_offsets[node.lineno - 1] + node.col_offset
    end = line_offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def format_sql_code(
    sql_code: str,
    dialect: str | None,
    config: FormattingConfig,
    placeholders: dict[str, str] | None = None,
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
        placeholders (Optional[Dict[str, str]]): A mapping of placeholders to their expressions.
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

        # Handle placeholders in f-strings
        placeholder_mapping = {}
        if placeholders:
            for placeholder in placeholders:
                # Replace placeholder with a valid SQL parameter
                temp_placeholder = f":{placeholder}"
                temp_sql = temp_sql.replace(placeholder, temp_placeholder)
                placeholder_mapping[temp_placeholder] = placeholder

        # Handle automatic placeholders (%s, ?)
        auto_placeholder_pattern = re.compile(r"%s|\?")
        auto_placeholders = auto_placeholder_pattern.findall(temp_sql)
        auto_placeholder_mapping = {}
        for idx, ph in enumerate(auto_placeholders):
            temp_placeholder = f":AUTO_PLACEHOLDER_{idx}"
            temp_sql = temp_sql.replace(ph, temp_placeholder, 1)
            auto_placeholder_mapping[temp_placeholder] = ph

        temp_sql = temp_sql.strip()

        # Parse and format SQL
        statements = parse(temp_sql, read=dialect)
        formatted_parts = []
        for stmt in statements:
            formatted_parts.append(
                stmt.sql(
                    pretty=not force_single_line,
                    indent=config.indent_width,
                    dialect=dialect,
                )
            )
        separator = "; " if force_single_line else ";\n"
        formatted_sql = separator.join(formatted_parts)

        # Apply formatting based on context
        if is_magic_command and not is_cell_magic:
            # Line magic: single line
            formatted_sql = " ".join(formatted_sql.split())
        elif force_single_line:
            formatted_sql = " ".join(formatted_sql.split())
        else:
            formatted_sql = formatted_sql.strip()

        # Restore placeholders in f-strings
        if placeholders:
            for temp_placeholder, original_placeholder in placeholder_mapping.items():
                # Remove quotes around placeholders if any
                formatted_sql = formatted_sql.replace(
                    f"'{temp_placeholder}'", temp_placeholder
                )
                formatted_sql = formatted_sql.replace(
                    f'"{temp_placeholder}"', temp_placeholder
                )
                # Replace temp placeholders with original placeholders
                formatted_sql = formatted_sql.replace(
                    temp_placeholder, original_placeholder
                )

        # Restore automatic placeholders
        for temp_placeholder, original_placeholder in auto_placeholder_mapping.items():
            formatted_sql = formatted_sql.replace(
                temp_placeholder, original_placeholder
            )

        # Logging for debugging purposes
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        return formatted_sql

    except errors.ParseError as e:
        raise SQLFormattingError(f"Failed to parse SQL code: {e}")
    except Exception as e:  # noqa: BLE001 - sqlglot raises many types; normalize them
        raise SQLFormattingError(f"Unexpected error during SQL formatting: {e}")


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
        self.changed = False

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

    def extract_fstring_parts(self, node: ast.JoinedStr) -> tuple[str, dict[str, str]]:
        """Extracts parts of an f-string, preserving expressions."""
        sql_parts = []
        placeholders = {}
        placeholder_counter = 0

        # Handle empty f-strings or f-strings with only constants
        if all(isinstance(value, ast.Constant) for value in node.values):
            return "".join(value.value for value in node.values), {}

        for value in node.values:
            if isinstance(value, ast.Constant):
                sql_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                expr = ast.unparse(value.value)
                placeholder = f"PLACEHOLDER_{placeholder_counter}"
                sql_parts.append(placeholder)
                placeholders[placeholder] = expr
                placeholder_counter += 1

        return "".join(sql_parts), placeholders

    def format_sql_node(
        self,
        node: ast.Constant | ast.JoinedStr,
        force_single_line: bool = False,
        block_indent: str | None = None,
    ) -> str | None:
        """Formats SQL code in AST nodes. Returns replacement text or None."""
        try:
            is_fstring = isinstance(node, ast.JoinedStr)
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
                    placeholders=placeholders if placeholders else None,
                    force_single_line=force_single_line,
                )
            except SQLFormattingError:
                return None

            if formatted_sql == textwrap.dedent(sql_str).strip():
                return None

            # Restore placeholders as f-string expressions
            if placeholders:
                for ph in placeholders:
                    formatted_sql = formatted_sql.replace(f"'{ph}'", ph)
                    formatted_sql = formatted_sql.replace(f'"{ph}"', ph)
                for ph, expr in placeholders.items():
                    formatted_sql = formatted_sql.replace(ph, f"{{{expr}}}")

            self.changed = True
            return self._build_replacement_text(formatted_sql, is_fstring, block_indent)

        except SQLFormattingError as e:
            self.logger.warning(f"SQL formatting skipped: {e}")
            return None
        except Exception as e:  # noqa: BLE001 - one bad string must not abort the file
            self.logger.warning(f"Unexpected error during SQL formatting: {e}")
            return None

    def _build_replacement_text(
        self, formatted_sql: str, is_fstring: bool, block_indent: str | None
    ) -> str:
        """Constructs the replacement string literal text.

        With ``block_indent`` (SQL passed to a call), the SQL lines and closing
        quotes are indented by it; where the block starts and what follows it is
        decided by the caller, see ``_layout_call_argument``.
        """
        prefix = "f" if is_fstring else ""
        if block_indent is not None and "\n" in formatted_sql:
            lines = formatted_sql.split("\n")
            indented_sql = "\n".join(
                block_indent + line if line.strip() else line for line in lines
            )
            return f'{prefix}"""\n{indented_sql}\n{block_indent}"""'
        elif "\n" in formatted_sql:
            return f'{prefix}"""\n{formatted_sql}\n"""'
        else:
            return f'{prefix}"{formatted_sql}"'

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handles assignments."""
        if isinstance(node.value, (ast.Constant, ast.JoinedStr)):
            replacement_text = self.format_sql_node(node.value)
            if replacement_text:
                start, end = node_offsets(node.value, self.line_offsets)
                self.replacements.append(SQLReplacement(start, end, replacement_text))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Handles function calls."""
        func_name = self.get_full_func_name(node.func)
        if any(name in func_name for name in self.config.function_names):
            values = [*node.args, *(keyword.value for keyword in node.keywords)]
            for value in values:
                if isinstance(value, (ast.Constant, ast.JoinedStr)):
                    self._replace_call_argument(node, value)
        self.generic_visit(node)

    def _replace_call_argument(
        self, call: ast.Call, arg: ast.Constant | ast.JoinedStr
    ) -> None:
        """Formats a SQL argument, indented relative to the line the call starts on."""
        call_line_start = self.line_offsets[call.lineno - 1]
        call_line = self.source[call_line_start:].partition("\n")[0]
        base_indent = call_line[: len(call_line) - len(call_line.lstrip())]
        block_indent = base_indent + " " * self.config.indent_width

        replacement = self.format_sql_node(arg, block_indent=block_indent)
        if not replacement:
            return
        start, end = node_offsets(arg, self.line_offsets)
        if "\n" in replacement:
            start, end, replacement = self._layout_call_argument(
                start, end, replacement, base_indent, block_indent
            )
        self.replacements.append(SQLReplacement(start, end, replacement))

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
        line_start = self.source.rfind("\n", 0, start) + 1
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
        for line in source.split("\n")
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

    lines = code.split("\n")

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

        if is_cell_magic:
            sql_code = "\n".join(lines[magic_cmd_index + 1 :]).strip()
        else:
            sql_code = lines[magic_cmd_index][len(magic_cmd) :].strip()

        try:
            formatted_sql = format_sql_code(
                sql_code,
                dialect,
                config,
                is_magic_command=True,
                is_cell_magic=is_cell_magic,
            )

            if is_cell_magic:
                pre_magic = "\n".join(lines[:magic_cmd_index])
                if pre_magic:
                    return f"{pre_magic}\n{magic_cmd}\n{formatted_sql}"
                return f"{magic_cmd}\n{formatted_sql}"
            else:
                pre_magic = "\n".join(lines[:magic_cmd_index])
                if pre_magic:
                    return f"{pre_magic}\n{magic_cmd} {formatted_sql}"
                return f"{magic_cmd} {formatted_sql}"

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
    formatter.visit(tree)

    if not formatter.replacements:
        return code

    result = code
    for rep in sorted(
        formatter.replacements,
        key=lambda r: r.start_offset,
        reverse=True,
    ):
        result = result[: rep.start_offset] + rep.new_text + result[rep.end_offset :]

    return result


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
    formatter.visit(tree)

    if not formatter.replacements:
        return source, False

    result = source
    for rep in sorted(
        formatter.replacements,
        key=lambda r: r.start_offset,
        reverse=True,
    ):
        result = result[: rep.start_offset] + rep.new_text + result[rep.end_offset :]

    changed = result != source
    return result, changed


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
