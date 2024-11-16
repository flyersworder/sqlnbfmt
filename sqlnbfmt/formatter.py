import ast
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Optional, Dict, Union, Tuple

import astor
import black
import nbformat
import yaml
from sqlglot import parse_one, errors


@dataclass
class FormattingConfig:
    """Configuration for SQL formatting."""

    sql_keywords: Set[str]
    function_names: Set[str]
    sql_decorators: Set[str]
    single_line_threshold: int = 80
    preserve_comments: bool = True
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
        logger = logging.getLogger('formatter')

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
        auto_placeholder_pattern = re.compile(r'%s|\?')
        auto_placeholders = auto_placeholder_pattern.findall(temp_sql)
        auto_placeholder_mapping = {}
        for idx, ph in enumerate(auto_placeholders):
            temp_placeholder = f":AUTO_PLACEHOLDER_{idx}"
            temp_sql = temp_sql.replace(ph, temp_placeholder, 1)
            auto_placeholder_mapping[temp_placeholder] = ph

        temp_sql = temp_sql.strip()

        # Parse and format SQL
        parsed = parse_one(temp_sql, read=dialect)
        formatted_sql = parsed.sql(
            pretty=not force_single_line,
            indent=config.indent_width,
            dialect=dialect
        )

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
                formatted_sql = formatted_sql.replace(f"'{temp_placeholder}'", temp_placeholder)
                formatted_sql = formatted_sql.replace(f'"{temp_placeholder}"', temp_placeholder)
                # Replace temp placeholders with original placeholders
                formatted_sql = formatted_sql.replace(temp_placeholder, original_placeholder)

        # Restore automatic placeholders
        for temp_placeholder, original_placeholder in auto_placeholder_mapping.items():
            formatted_sql = formatted_sql.replace(temp_placeholder, original_placeholder)

        # Logging for debugging purposes
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        return formatted_sql

    except errors.ParseError as e:
        raise SQLFormattingError(f"Failed to parse SQL code: {e}")
    except Exception as e:
        raise SQLFormattingError(f"Unexpected error during SQL formatting: {e}")


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
        placeholder_counter = 0

        # Handle empty f-strings or f-strings with only constants
        if all(isinstance(value, ast.Constant) for value in node.values):
            return "".join(value.value for value in node.values), {}

        for value in node.values:
            if isinstance(value, ast.Constant):
                sql_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                expr = astor.to_source(value.value).strip()
                placeholder = f"PLACEHOLDER_{placeholder_counter}"
                sql_parts.append(placeholder)
                placeholders[placeholder] = expr
                placeholder_counter += 1

        return "".join(sql_parts), placeholders

    def format_sql_node(
        self, 
        node: Union[ast.Constant, ast.JoinedStr], 
        force_single_line: bool = False,
        in_function: bool = False
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
                    if in_function and '\n' in formatted_sql:
                        # Apply Black-style indentation for function arguments
                        lines = formatted_sql.split('\n')
                        # Indent all lines including the first SQL line
                        indented_lines = [
                            " " * 4 + line if line.strip() else line
                            for line in lines
                        ]
                        formatted_sql = '\n'.join(indented_lines)
                        
                        # Reconstruct the string with appropriate quotes
                        formatted_str = f'"""\n{formatted_sql}\n{" " * 4}"""'
                    else:
                        # Ensure newlines are added around the formatted SQL
                        formatted_str = f'"""\n{formatted_sql}\n"""'
                    
                    formatted_node = ast.parse(formatted_str).body[0].value
                    return formatted_node

            elif isinstance(node, ast.JoinedStr):
                sql_str, placeholders = self.extract_fstring_parts(node)
                if not sql_str or not self.is_likely_sql(sql_str):
                    return None

                try:
                    formatted_sql = format_sql_code(
                        sql_str,
                        self.dialect,
                        self.config,
                        placeholders=placeholders,
                        force_single_line=force_single_line,
                    )
                except SQLFormattingError:
                    return None

                if placeholders:
                    for placeholder in placeholders.keys():
                        if not placeholder:
                            continue
                        formatted_sql = formatted_sql.replace(f"'{placeholder}'", placeholder)
                        formatted_sql = formatted_sql.replace(f'"{placeholder}"', placeholder)

                if formatted_sql != sql_str:
                    self.changed = True
                    # Apply Black-style indentation for function arguments
                    if in_function and '\n' in formatted_sql:
                        lines = formatted_sql.split('\n')
                        # Indent all lines including the first SQL line
                        indented_lines = [
                            " " * 4 + line if line.strip() else line
                            for line in lines
                        ]
                        formatted_sql = '\n'.join(indented_lines)

                    # Reconstruct the f-string
                    new_values = []
                    if not placeholders:
                        new_values.append(ast.Constant(value=formatted_sql))
                    else:
                        pattern = re.compile('|'.join(re.escape(k) for k in placeholders.keys() if k))
                        idx = 0
                        while idx < len(formatted_sql):
                            match = pattern.search(formatted_sql, idx)
                            if match:
                                if match.start() > idx:
                                    new_values.append(ast.Constant(value=formatted_sql[idx:match.start()]))
                                placeholder = match.group()
                                expr_str = placeholders[placeholder].strip()
                                try:
                                    expr_ast = ast.parse(expr_str, mode='eval').body
                                except SyntaxError as e:
                                    self.logger.warning(f"Failed to parse expression '{expr_str}': {e}")
                                    return None
                                new_values.append(ast.FormattedValue(
                                    value=expr_ast,
                                    conversion=-1,
                                    format_spec=None
                                ))
                                idx = match.end()
                            else:
                                new_values.append(ast.Constant(value=formatted_sql[idx:]))
                                break

                    new_node = ast.JoinedStr(values=new_values)
                    if '\n' in formatted_sql:
                        if in_function:
                            formatted_fstring = ast.JoinedStr(values=[
                                ast.Constant(value='\n'),
                                *new_node.values,
                                ast.Constant(value=f'\n{" " * 4}')
                            ])
                        else:
                            formatted_fstring = ast.JoinedStr(values=[
                                ast.Constant(value='\n'),
                                *new_node.values,
                                ast.Constant(value='\n')
                            ])
                    else:
                        formatted_fstring = new_node
                    return formatted_fstring

            return None
        except SQLFormattingError as e:
            self.logger.warning(f"SQL formatting skipped: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Unexpected error during SQL formatting: {e}")
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
                    # Format with Black-style indentation for function arguments
                    formatted_node = self.format_sql_node(arg, in_function=True)
                    if formatted_node:
                        node.args[idx] = formatted_node
            for keyword in node.keywords:
                if isinstance(keyword.value, (ast.Constant, ast.JoinedStr)):
                    formatted_node = self.format_sql_node(keyword.value, in_function=True)
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
    notebook_path: Union[str, Path],
    config: FormattingConfig,
    dialect: Optional[str],
    logger: logging.Logger,
) -> bool:
    """Processes a Jupyter notebook."""
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
        changed = False

        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue

            original_code = cell.source
            if not original_code.strip():
                continue

            lines = original_code.split("\n")

            # Initialize variables to track magic commands
            magic_cmd = None
            magic_cmd_index = None

            # Iterate through lines to find the first non-comment magic command
            for idx, line in enumerate(lines):
                stripped = line.strip()

                if not stripped:
                    continue  # Skip empty lines

                if stripped.startswith("#"):
                    continue  # Skip comment lines

                if stripped.startswith("%%sql") or stripped.startswith("%sql"):
                    magic_cmd = stripped.split()[0]
                    magic_cmd_index = idx
                    break  # Magic command found

                else:
                    break  # Non-magic, non-comment line found

            if magic_cmd:
                is_cell_magic = magic_cmd.startswith("%%sql")

                if is_cell_magic:
                    # Cell magic: SQL code starts from the next line
                    sql_code = "\n".join(lines[magic_cmd_index + 1 :]).strip()
                else:
                    # Line magic: SQL code is on the same line after the magic command
                    sql_code = lines[magic_cmd_index][len(magic_cmd) :].strip()

                try:
                    formatted_sql = format_sql_code(
                        sql_code,
                        dialect,
                        config,
                        is_magic_command=True,
                        is_cell_magic=is_cell_magic,
                    )

                    # Reconstruct the cell content
                    if is_cell_magic:
                        # Preserve comments before the magic command
                        pre_magic = "\n".join(lines[:magic_cmd_index])
                        if pre_magic:
                            new_content = f"{pre_magic}\n{magic_cmd}\n{formatted_sql}"
                        else:
                            new_content = f"{magic_cmd}\n{formatted_sql}"
                    else:
                        # Preserve comments before the magic command
                        pre_magic = "\n".join(lines[:magic_cmd_index])
                        if pre_magic:
                            new_content = f"{pre_magic}\n{magic_cmd} {formatted_sql}"
                        else:
                            new_content = f"{magic_cmd} {formatted_sql}"

                    if new_content != original_code:
                        cell.source = new_content
                        changed = True

                except SQLFormattingError as e:
                    logger.warning(f"SQL magic formatting skipped: {e}")

                continue  # Move to the next cell after handling magic command

            # Handle regular Python code
            try:
                tree = ast.parse(original_code)
                formatter = SQLStringFormatter(config, dialect, logger)
                new_tree = formatter.visit(tree)
                if formatter.changed:
                    # Use astor to convert AST back to source code
                    formatted_code = astor.to_source(new_tree)
                    # Now format with black
                    formatted_code = black.format_str(formatted_code, mode=black.FileMode())
                    
                    if formatted_code != original_code:
                        cell.source = formatted_code
                        changed = True
            except SyntaxError:
                logger.warning(f"Failed to parse cell:\n{original_code}")
                continue

        if changed:
            nbformat.write(notebook, notebook_path)
            logger.info(f"Updated notebook: {notebook_path}")

        return changed

    except Exception as e:
        logger.error(f"Failed to process notebook {notebook_path}: {e}", exc_info=True)
        raise


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
        changed_files = []

        for notebook in args.notebooks:
            if process_notebook(notebook, config, args.dialect, logger):
                changed = True
                changed_files.append(notebook)

        # Print summary
        if changed:
            logger.info("Changes made to the following notebooks:")
            for file in changed_files:
                logger.info(f"  - {file}")
        else:
            logger.info("No changes needed. All notebooks are properly formatted.")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()