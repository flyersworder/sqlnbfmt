import ast
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Optional, Dict, Union, Tuple, List

import nbformat
import yaml
from sqlglot.tokens import Token, Tokenizer
from sqlglot import parse_one, errors, exp
from sqlglot.errors import TokenError


@dataclass
class FormattingConfig:
    """Configuration for SQL formatting."""

    sql_keywords: Set[str]
    function_names: Set[str]
    sql_decorators: Set[str]
    single_line_threshold: int = 80
    preserve_comments: bool = True  # Although we remove comment handling, keeping the flag for potential future use
    indent_width: int = 4


class SQLFormattingError(Exception):
    """Custom exception for SQL formatting errors."""

    pass


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Sets up logging with the specified level."""
    logger = logging.getLogger(__name__)
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
                preserve_comments=config.get("formatting_options", {}).get(
                    "preserve_comments", True
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
    is_magic_command: bool = False,
    is_cell_magic: bool = False,
) -> str:
    """Formats SQL code using sqlglot's native formatting capabilities while preserving placeholders."""
    try:
        logger = logging.getLogger(__name__)

        if not sql_code.strip():
            return sql_code

        temp_sql = sql_code
        placeholder_mapping = {}

        # Handle placeholders for expressions
        if placeholders:
            for idx, (placeholder, value) in enumerate(placeholders.items()):
                marker = f"__PLACEHOLDER_{idx}__"
                temp_sql = temp_sql.replace(placeholder, marker)
                placeholder_mapping[marker] = placeholder
                logger.debug(f"Replaced placeholder {placeholder} with marker {marker}")

        temp_sql = temp_sql.strip()

        # Remove comments if not preserving them
        if not config.preserve_comments:
            temp_sql = re.sub(r'--.*', '', temp_sql)
            temp_sql = re.sub(r'/\*[\s\S]*?\*/', '', temp_sql)
            logger.debug("Comments removed from SQL.")

        # Prevent multiple statements
        if ';' in temp_sql.strip().rstrip(';'):
            raise SQLFormattingError("Multiple SQL statements are not supported.")

        # Parse SQL
        parsed = parse_one(temp_sql, read=dialect)

        # Generate formatted SQL with uppercase keywords and proper quoting
        formatted_sql = parsed.sql(
            pretty=not force_single_line,
            indent=config.indent_width,
            dialect=dialect,
        )

        # Condense to single line if necessary
        if force_single_line or (config.single_line_threshold and len(formatted_sql.replace('\n', ' ')) <= config.single_line_threshold):
            formatted_sql = " ".join(formatted_sql.split())

        # Restore placeholders
        if placeholders:
            for marker, original_placeholder in placeholder_mapping.items():
                formatted_sql = formatted_sql.replace(marker, original_placeholder)
                logger.debug(f"Restored marker {marker} to placeholder {original_placeholder}")

        # Logging
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        return formatted_sql

    except errors.ParseError as e:
        raise SQLFormattingError(f"Failed to parse SQL code: {e}") from e
    except TokenError as te:
        raise SQLFormattingError(f"Tokenization failed: {te}") from te
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
        current_pos = 0

        for value in node.values:
            if isinstance(value, ast.Constant):
                sql_parts.append(value.value)
                current_pos += len(value.value)
            elif isinstance(value, ast.FormattedValue):
                expr = ast.unparse(value.value).strip()  # Using ast.unparse instead of astor.to_source
                placeholder = f"__EXPR_PLACEHOLDER_{expr}__"
                sql_parts.append(placeholder)
                placeholders[placeholder] = expr
                current_pos += len(placeholder)

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
                    # Check if the string is multi-line
                    if "\n" in formatted_sql:
                        # Reconstruct the multi-line string with newlines after opening and before closing quotes
                        reconstructed_str = f'"""\n{formatted_sql}\n"""'
                        formatted_node = ast.parse(reconstructed_str).body[0].value
                    else:
                        # Single-line string
                        reconstructed_str = f'"""{formatted_sql}"""'
                        formatted_node = ast.parse(reconstructed_str).body[0].value
                    return formatted_node

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
                    # Reconstruct the f-string with newlines after opening and before closing quotes
                    if "\n" in formatted_sql:
                        reconstructed_fstring = f'f"""\n{formatted_sql}\n"""'
                    else:
                        reconstructed_fstring = f'f"""{formatted_sql}"""'
                    formatted_node = ast.parse(reconstructed_fstring).body[0].value
                    return formatted_node
            return None
        except SQLFormattingError as e:
            self.logger.warning(f"SQL formatting skipped: {e}")
            return None

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Handles assignments."""
        if isinstance(node.value, (ast.Constant, ast.JoinedStr)):
            formatted_node = self.format_sql_node(node.value)
            if formatted_node:
                node.value = formatted_node
        return self.generic_visit(node)

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
        return self.generic_visit(node)

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

        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            original_source = cell.source
            formatted_source = original_source

            # Identify and format SQL within function calls or magic commands
            for func in config.function_names:
                pattern = rf'{func}\s*\("""(.*?)"""'
                matches = re.findall(pattern, formatted_source, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        formatted_sql = format_sql_code(match, dialect, config)
                        # Replace the SQL string within the function call
                        formatted_source = formatted_source.replace(match, formatted_sql)
                        changed = True
                    except SQLFormattingError as e:
                        logger.warning(f"Failed to parse cell:\n{match}\nError: {e}")
                        failed = True  # Mark as failed

            # Handle SQL magic commands (e.g., %%sql)
            magic_pattern = r'%%sql\s*\n(.*)'
            magic_matches = re.findall(magic_pattern, formatted_source, re.DOTALL | re.IGNORECASE)
            for magic_sql in magic_matches:
                try:
                    formatted_sql_magic = format_sql_code(magic_sql.strip(), dialect, config)
                    # Replace the SQL string within the magic command
                    formatted_source = formatted_source.replace(magic_sql, formatted_sql_magic)
                    changed = True
                except SQLFormattingError as e:
                    logger.warning(f"Failed to parse magic cell:\n{magic_sql}\nError: {e}")
                    failed = True  # Mark as failed

            # Update cell source if changes were made
            if formatted_source != original_source:
                cell.source = formatted_source

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
        changed = False

        for notebook in args.notebooks:
            if process_notebook(notebook, config, args.dialect, logger):
                changed = True

        sys.exit(1 if changed else 0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(2)