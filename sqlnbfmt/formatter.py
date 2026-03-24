import ast
import copy
import difflib
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Optional, Dict, Union, Tuple

import nbformat
import yaml
from sqlglot import parse, errors

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

    sql_keywords: Set[str] = field(default_factory=lambda: set(DEFAULT_SQL_KEYWORDS))
    function_names: Set[str] = field(
        default_factory=lambda: set(DEFAULT_FUNCTION_NAMES)
    )
    sql_decorators: Set[str] = field(
        default_factory=lambda: set(DEFAULT_SQL_DECORATORS)
    )
    single_line_threshold: int = 80
    indent_width: int = 4


class SQLFormattingError(Exception):
    """Custom exception for SQL formatting errors."""

    pass


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


def load_config(config_path: Optional[Union[str, Path]] = None) -> FormattingConfig:
    """Loads configuration from a YAML file, or returns defaults if no path given."""
    if config_path is None:
        return FormattingConfig()

    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    try:
        with open(config_file) as file:
            config = yaml.safe_load(file)
            if config is None:
                return FormattingConfig()
            return FormattingConfig(
                sql_keywords=set(config.get("sql_keywords", DEFAULT_SQL_KEYWORDS)),
                function_names=set(
                    config.get("function_names", DEFAULT_FUNCTION_NAMES)
                ),
                sql_decorators=set(
                    config.get("sql_decorators", DEFAULT_SQL_DECORATORS)
                ),
                single_line_threshold=config.get("formatting_options", {}).get(
                    "single_line_threshold", 80
                ),
                indent_width=config.get("formatting_options", {}).get(
                    "indent_width", 4
                ),
            )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing the configuration file: {e}")


def build_line_offsets(source: str) -> list:
    """Returns list where offsets[i] is the character offset of line i+1 in source."""
    offsets = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def node_offsets(node: ast.AST, line_offsets: list) -> Tuple[int, int]:
    """Converts AST node line/col positions to (start, end) character offsets."""
    start = line_offsets[node.lineno - 1] + node.col_offset
    end = line_offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def format_sql_code(
    sql_code: str,
    dialect: Optional[str],
    config: FormattingConfig,
    placeholders: Optional[Dict[str, str]] = None,
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
            for placeholder in placeholders.keys():
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
    except Exception as e:
        raise SQLFormattingError(f"Unexpected error during SQL formatting: {e}")


class SQLStringFormatter(ast.NodeVisitor):
    """AST NodeVisitor that collects SQL string replacements."""

    def __init__(
        self,
        config: FormattingConfig,
        dialect: Optional[str],
        logger: logging.Logger,
        line_offsets: list,
    ):
        super().__init__()
        self.config = config
        self.dialect = dialect
        self.logger = logger
        self.line_offsets = line_offsets
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

    def extract_fstring_parts(self, node: ast.JoinedStr) -> Tuple[str, Dict[str, str]]:
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
        node: Union[ast.Constant, ast.JoinedStr],
        force_single_line: bool = False,
        in_function: bool = False,
    ) -> Optional[str]:
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
            return self._build_replacement_text(formatted_sql, is_fstring, in_function)

        except SQLFormattingError as e:
            self.logger.warning(f"SQL formatting skipped: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Unexpected error during SQL formatting: {e}")
            return None

    def _build_replacement_text(
        self, formatted_sql: str, is_fstring: bool, in_function: bool
    ) -> str:
        """Constructs the replacement string literal text."""
        prefix = "f" if is_fstring else ""
        if in_function and "\n" in formatted_sql:
            indent = " " * self.config.indent_width
            lines = formatted_sql.split("\n")
            indented_lines = [indent + line if line.strip() else line for line in lines]
            indented_sql = "\n".join(indented_lines)
            return f'\n{indent}{prefix}"""\n{indented_sql}\n{indent}"""\n'
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
            for arg in node.args:
                if isinstance(arg, (ast.Constant, ast.JoinedStr)):
                    replacement_text = self.format_sql_node(arg, in_function=True)
                    if replacement_text:
                        start, end = node_offsets(arg, self.line_offsets)
                        self.replacements.append(
                            SQLReplacement(start, end, replacement_text)
                        )
            for keyword in node.keywords:
                if isinstance(keyword.value, (ast.Constant, ast.JoinedStr)):
                    replacement_text = self.format_sql_node(
                        keyword.value, in_function=True
                    )
                    if replacement_text:
                        start, end = node_offsets(keyword.value, self.line_offsets)
                        self.replacements.append(
                            SQLReplacement(start, end, replacement_text)
                        )
        self.generic_visit(node)

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
    dialect: Optional[str],
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
        if stripped.startswith("%%sql") or stripped.startswith("%sql"):
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
    formatter = SQLStringFormatter(config, dialect, logger, line_offsets)
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
    dialect: Optional[str],
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
    dialect: Optional[str],
    logger: logging.Logger,
) -> Tuple[str, bool]:
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
    formatter = SQLStringFormatter(config, dialect, logger, line_offsets)
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
    notebook_path: Union[str, Path],
    config: FormattingConfig,
    dialect: Optional[str],
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

    except Exception as e:
        logger.error(f"Failed to process notebook {notebook_path}: {e}", exc_info=True)
        raise


def diff_notebook(
    notebook_path: Union[str, Path],
    config: FormattingConfig,
    dialect: Optional[str],
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
    file_path: Union[str, Path],
    config: FormattingConfig,
    dialect: Optional[str],
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

    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}", exc_info=True)
        raise


def diff_python_file(
    file_path: Union[str, Path],
    config: FormattingConfig,
    dialect: Optional[str],
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
        "--config", type=Path, default=None, help="Configuration file path (optional)"
    )
    parser.add_argument(
        "--dialect", type=str, help="SQL dialect (e.g., mysql, postgres)"
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

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(
            f"{e}\nHint: --config is optional; omit it to use built-in defaults."
        )
        sys.exit(1)

    try:
        changed_files = []

        for file_path in args.files:
            is_python = file_path.suffix == ".py"

            if args.check or args.diff:
                if is_python:
                    diff_output = diff_python_file(
                        file_path, config, args.dialect, logger
                    )
                else:
                    diff_output = diff_notebook(file_path, config, args.dialect, logger)
                needs_formatting = bool(diff_output)

                if args.diff and diff_output:
                    print(diff_output, end="")

                if needs_formatting:
                    changed_files.append(file_path)
            else:
                if is_python:
                    changed = process_python_file(
                        file_path, config, args.dialect, logger
                    )
                else:
                    changed = process_notebook(file_path, config, args.dialect, logger)
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

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
