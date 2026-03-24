"""JupyterLab code formatter integration.

Provides a formatter plugin for jupyterlab-code-formatter that formats
SQL in notebook cells (both magic commands and Python strings).

Setup: add this to your ``jupyter_server_config.py``::

    from sqlnbfmt.jupyterlab_integration import register
    register()

Then select "Apply SQL Notebook Formatter" in JupyterLab's
Code > Format Cell menu or configure it for format-on-save.
"""

from __future__ import annotations

import logging
from typing import Optional


def _get_formatter_class():
    """Build the formatter class lazily to avoid hard dependency."""
    from jupyterlab_code_formatter.formatters import BaseFormatter

    from sqlnbfmt.formatter import FormattingConfig, format_cell_source, load_config

    class SQLNotebookFormatter(BaseFormatter):
        label = "Apply SQL Notebook Formatter"

        def __init__(
            self,
            config: Optional[FormattingConfig] = None,
            dialect: Optional[str] = None,
        ):
            self._config = config or load_config()
            self._dialect = dialect
            self._logger = logging.getLogger("sqlnbfmt.jupyterlab")

        @property
        def importable(self) -> bool:
            return True

        def format_code(self, code: str, notebook: bool, **options) -> str:
            dialect = options.pop("dialect", self._dialect)
            return format_cell_source(code, self._config, dialect, self._logger)

    return SQLNotebookFormatter


def register(
    name: str = "sqlnbfmt",
    dialect: Optional[str] = None,
    config_path: Optional[str] = None,
) -> None:
    """Register sqlnbfmt with jupyterlab-code-formatter.

    Call this in ``jupyter_server_config.py``::

        from sqlnbfmt.jupyterlab_integration import register
        register()                       # defaults
        register(dialect="postgres")     # with options

    Args:
        name: Key used in JupyterLab settings to reference this formatter.
        dialect: SQL dialect (e.g. ``"postgres"``, ``"mysql"``).
        config_path: Path to a sqlnbfmt YAML config file.
    """
    from jupyterlab_code_formatter.formatters import SERVER_FORMATTERS

    from sqlnbfmt.formatter import load_config

    config = load_config(config_path)
    cls = _get_formatter_class()
    SERVER_FORMATTERS[name] = cls(config=config, dialect=dialect)
