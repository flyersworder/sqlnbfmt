"""Tests for the JupyterLab code formatter integration.

Since jupyterlab-code-formatter is not a dependency, these tests
mock the necessary imports and verify the adapter works correctly.
"""

from unittest.mock import MagicMock, patch


def test_register_adds_to_server_formatters():
    """register() inserts an instance into SERVER_FORMATTERS."""
    mock_formatters = {}
    mock_base = type("BaseFormatter", (), {})

    with (
        patch.dict(
            "sys.modules",
            {
                "jupyterlab_code_formatter": MagicMock(),
                "jupyterlab_code_formatter.formatters": MagicMock(
                    BaseFormatter=mock_base,
                    SERVER_FORMATTERS=mock_formatters,
                ),
            },
        ),
    ):
        from sqlnbfmt.jupyterlab_integration import register

        register()
        assert "sqlnbfmt" in mock_formatters

        register(name="custom_name", dialect="postgres")
        assert "custom_name" in mock_formatters


def test_formatter_class_format_code():
    """The formatter class delegates to format_cell_source."""
    mock_base = type("BaseFormatter", (), {})

    with patch.dict(
        "sys.modules",
        {
            "jupyterlab_code_formatter": MagicMock(),
            "jupyterlab_code_formatter.formatters": MagicMock(
                BaseFormatter=mock_base,
                SERVER_FORMATTERS={},
            ),
        },
    ):
        from sqlnbfmt.jupyterlab_integration import _get_formatter_class

        cls = _get_formatter_class()
        formatter = cls()

        assert formatter.label == "Apply SQL Notebook Formatter"
        assert formatter.importable is True

        # Format a cell with SQL
        result = formatter.format_code(
            'query = "select id, name from users where active = 1"',
            notebook=True,
        )
        assert "SELECT" in result

        # Format a %%sql magic cell
        result = formatter.format_code(
            "%%sql\nselect * from users where active = 1",
            notebook=True,
        )
        assert "%%sql" in result
        assert "SELECT" in result

        # Non-SQL code passes through unchanged
        code = "x = 42\nprint(x)"
        result = formatter.format_code(code, notebook=True)
        assert result == code
