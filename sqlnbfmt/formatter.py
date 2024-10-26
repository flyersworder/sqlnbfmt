import ast
import logging
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Optional, Dict, Union, Tuple, List

import nbformat
import yaml
from sqlglot import parse_one, parse, errors
from sqlglot.errors import TokenError


@dataclass
class FormattingConfig:
    """Configuration for SQL formatting."""

    sql_keywords: Set[str]
    function_names: Set[str]
    sql_decorators: Set[str]
    single_line_threshold: int = 80
    indent_width: int = 4


class SQLFormattingError(Exception):
    """Custom exception for SQL formatting errors."""
    pass


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Sets up logging with the specified level."""
    logger = logging.getLogger('formatter')
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_path: Union[str, Path] = "config.yaml") -> FormattingConfig:
    """Loads configuration from a YAML file."""
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    try:
        with open(config_file) as file:
            config = yaml.safe_load(file)
            return FormattingConfig(
                sql_keywords=set(config.get("sql_keywords", [])),
                function_names=set(config.get("function_names", [])),
                sql_decorators=set(config.get("sql_decorators", [])),
                single_line_threshold=config.get("formatting_options", {}).get(
                    "single_line_threshold", 80
                ),
                indent_width=config.get("formatting_options", {}).get(
                    "indent_width", 4
                ),
            )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing the configuration file: {e}")


def format_sql_code(
    sql_code: str,
    dialect: Optional[str],
    config: FormattingConfig,
    placeholders: Optional[Dict[str, str]] = None,
    force_single_line: bool = False,
    ignore_placeholders: bool = False,
) -> str:
    """Formats SQL code using sqlglot's native formatting capabilities while preserving placeholders."""
    try:
        logger = logging.getLogger('formatter')

        if not sql_code.strip():
            return sql_code

        temp_sql = sql_code
        placeholder_mapping = {}

        # Handle placeholders for expressions
        if placeholders and not ignore_placeholders:
            for expr_placeholder, expr in placeholders.items():
                marker = f"'__EXPR_PLACEHOLDER_{uuid.uuid4()}__'"
                temp_sql = temp_sql.replace(expr_placeholder, marker)
                placeholder_mapping[marker.strip("'")] = expr_placeholder
                logger.debug(f"Replaced placeholder {expr_placeholder} with marker {marker}")

        # Automatically replace common placeholders (%s and ?) with unique markers
        if not ignore_placeholders:
            auto_placeholders = {'%s', '?'}
            for placeholder in auto_placeholders:
                unique_id = str(uuid.uuid4())
                marker = f"'__AUTO_PLACEHOLDER_{unique_id}__'"
                temp_sql = temp_sql.replace(placeholder, marker)
                placeholder_mapping[marker.strip("'")] = placeholder
                logger.debug(f"Auto-replaced placeholder {placeholder} with marker {marker}")

        temp_sql = temp_sql.strip()

        # Prevent multiple statements
        statements = parse(temp_sql, read=dialect)
        if len(statements) > 1:
            raise SQLFormattingError("Multiple SQL statements are not supported.")

        # Parse SQL
        parsed = statements[0]

        # Generate formatted SQL with uppercase keywords and proper quoting
        formatted_sql = parsed.sql(
            pretty=not force_single_line,
            indent=config.indent_width,
            dialect=dialect,
        )

        # Condense to single line if necessary
        if force_single_line or (
            config.single_line_threshold
            and len(formatted_sql.replace('\n', ' ')) <= config.single_line_threshold
        ):
            formatted_sql = " ".join(formatted_sql.split())

        # Restore placeholders
        if placeholder_mapping:
            for marker, original_placeholder in placeholder_mapping.items():
                formatted_sql = formatted_sql.replace(f"'{marker}'", original_placeholder)
                logger.debug(f"Restored marker '{marker}' to placeholder {original_placeholder}")

        # Logging
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        return formatted_sql

    except errors.ParseError as e:
        logger.warning(f"SQL parsing failed: {e}. Returning original SQL code.")
        return sql_code  # Return the original SQL code unmodified
    except TokenError as te:
        logger.warning(f"Tokenization failed: {te}. Returning original SQL code.")
        return sql_code  # Return the original SQL code unmodified
    except Exception as e:
        raise SQLFormattingError(f"Unexpected error during SQL formatting: {e}") from e


class SQLStringFormatter(ast.NodeTransformer):
    """AST NodeTransformer that formats SQL strings."""

    def __init__(
        self, config: FormattingConfig, dialect: Optional[str], logger: logging.Logger
    ):
        super().__init__()
        self.config = config
        self.dialect = dialect
        self.logger = logger
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

        for value in node.values:
            if isinstance(value, ast.Constant):
                sql_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                expr = ast.unparse(value.value).strip()
                placeholder = f"__EXPR_PLACEHOLDER_{uuid.uuid4()}__"
                sql_parts.append(placeholder)
                placeholders[placeholder] = f"{{{expr}}}"

        return "".join(sql_parts), placeholders

    def format_sql_node(
        self, node: Union[ast.Constant, ast.JoinedStr], force_single_line: bool = False
    ) -> Optional[ast.AST]:
        """Formats SQL code in AST nodes."""
        try:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if not self.is_likely_sql(node.value):
                    return None
                formatted_sql = format_sql_code(
                    node.value,
                    self.dialect,
                    self.config,
                    force_single_line=force_single_line,
                )
                if formatted_sql != node.value:
                    self.changed = True
                    return ast.Constant(value=formatted_sql)

            elif isinstance(node, ast.JoinedStr):
                sql_str, placeholders = self.extract_fstring_parts(node)
                if not self.is_likely_sql(sql_str):
                    return None
                formatted_sql = format_sql_code(
                    sql_str,
                    self.dialect,
                    self.config,
                    placeholders=placeholders,
                    force_single_line=force_single_line,
                )
                if formatted_sql != sql_str:
                    self.changed = True
                    # Reconstruct the f-string
                    new_values = []
                    pattern = re.compile('|'.join(re.escape(k) for k in placeholders.keys()))
                    idx = 0
                    while idx < len(formatted_sql):
                        match = pattern.search(formatted_sql, idx)
                        if match:
                            if match.start() > idx:
                                new_values.append(ast.Constant(value=formatted_sql[idx:match.start()]))
                            expr_placeholder = match.group()
                            expr_str = placeholders[expr_placeholder]
                            expr_ast = ast.parse(expr_str).body[0].value
                            new_values.append(ast.FormattedValue(
                                value=expr_ast,
                                conversion=-1,
                                format_spec=None
                            ))
                            idx = match.end()
                        else:
                            new_values.append(ast.Constant(value=formatted_sql[idx:]))
                            break
                    return ast.JoinedStr(values=new_values)
            return None
        except SQLFormattingError as e:
            self.logger.warning(f"SQL formatting skipped: {e}")
            return None

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Handles string constants."""
        formatted_node = self.format_sql_node(node)
        return formatted_node or node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        """Handles f-strings."""
        formatted_node = self.format_sql_node(node)
        return formatted_node or node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Handles assignments."""
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Handles function calls."""
        func_name = self.get_full_func_name(node.func)
        if any(name in func_name for name in self.config.function_names):
            for idx, arg in enumerate(node.args):
                if isinstance(arg, (ast.Constant, ast.JoinedStr)):
                    formatted_node = self.format_sql_node(arg)
                    if formatted_node:
                        node.args[idx] = formatted_node
            for keyword in node.keywords:
                if isinstance(keyword.value, (ast.Constant, ast.JoinedStr)):
                    formatted_node = self.format_sql_node(keyword.value)
                    if formatted_node:
                        keyword.value = formatted_node
        self.generic_visit(node)
        return node

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


def process_notebook(
    notebook_path: Path,
    config: FormattingConfig,
    dialect: Optional[str],
    logger: logging.Logger
) -> bool:
    """Processes a Jupyter notebook, formatting SQL code within code cells."""
    try:
        nb = nbformat.read(notebook_path, as_version=4)
        changed = False
        failed = False  # Flag to track failures

        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            original_source = cell.source

            # Handle magic SQL cells
            if cell.source.lstrip().startswith("%%sql"):
                stripped_source = cell.source.lstrip()
                lines = stripped_source.split('\n', 1)
                magic_command = lines[0]
                sql_code = lines[1] if len(lines) > 1 else ''
                try:
                    formatted_sql = format_sql_code(
                        sql_code, dialect, config, ignore_placeholders=True
                    )
                    cell.source = f"{magic_command}\n{formatted_sql}"
                    if cell.source != original_source:
                        changed = True
                except SQLFormattingError as e:
                    logger.warning(f"Failed to format magic SQL in cell {idx}: {e}")
                    failed = True
                continue

            # Process regular code cells
            try:
                tree = ast.parse(cell.source)
                formatter = SQLStringFormatter(config, dialect, logger)
                new_tree = formatter.visit(tree)
                if formatter.changed:
                    cell.source = ast.unparse(new_tree)
                    changed = True
            except SyntaxError as e:
                logger.warning(f"Syntax error in cell {idx}: {e}")
                failed = True
            except Exception as e:
                logger.error(f"Error processing cell {idx}: {e}")
                failed = True

        if changed and not failed:
            nbformat.write(nb, notebook_path)
            logger.info(f"Updated notebook: {notebook_path}")
            return True
        elif failed:
            logger.error(f"Notebook processing failed for: {notebook_path}")
            return False
        else:
            logger.info(f"No changes made to notebook: {notebook_path}")
            return True

    except Exception as e:
        logger.error(f"Error processing notebook {notebook_path}: {e}")
        return False


def main():
    """Main entry point for the SQL formatter."""
    import argparse

    parser = argparse.ArgumentParser(description="Format SQL code in Jupyter notebooks")
    parser.add_argument(
        "notebooks", nargs="+", type=Path, help="Notebook paths to process"
    )
    parser.add_argument(
        "--config", type=Path, default="config.yaml", help="Configuration file path"
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

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    try:
        config = load_config(args.config)
        all_succeeded = True

        for notebook in args.notebooks:
            success = process_notebook(notebook, config, args.dialect, logger)
            if not success:
                all_succeeded = False

        sys.exit(0 if all_succeeded else 1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)