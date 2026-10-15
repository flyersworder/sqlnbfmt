"""Tests for the JupyterLab code formatter integration.

Since jupyterlab-code-formatter is not a dependency, these tests
mock the necessary imports and verify the adapter works correctly.
"""

from unittest.mock import MagicMock, patch

import pytest


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


def test_register_discovers_pyproject_config(tmp_path, monkeypatch):
    """register() without config_path uses [tool.sqlnbfmt] from pyproject.toml."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.sqlnbfmt.formatting_options]\nindent_width = 2\n"
    )
    monkeypatch.chdir(tmp_path)
    mock_formatters = {}
    mock_base = type("BaseFormatter", (), {})

    with patch.dict(
        "sys.modules",
        {
            "jupyterlab_code_formatter": MagicMock(),
            "jupyterlab_code_formatter.formatters": MagicMock(
                BaseFormatter=mock_base,
                SERVER_FORMATTERS=mock_formatters,
            ),
        },
    ):
        from sqlnbfmt.jupyterlab_integration import register

        register()
        assert mock_formatters["sqlnbfmt"]._config.indent_width == 2


def test_register_uses_dialect_from_config(tmp_path, monkeypatch):
    """register() falls back to the config's dialect; an explicit one wins."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text('[tool.sqlnbfmt]\ndialect = "tsql"\n')
    monkeypatch.chdir(tmp_path)
    mock_formatters = {}
    mock_base = type("BaseFormatter", (), {})

    with patch.dict(
        "sys.modules",
        {
            "jupyterlab_code_formatter": MagicMock(),
            "jupyterlab_code_formatter.formatters": MagicMock(
                BaseFormatter=mock_base,
                SERVER_FORMATTERS=mock_formatters,
            ),
        },
    ):
        from sqlnbfmt.jupyterlab_integration import register

        register()
        assert mock_formatters["sqlnbfmt"]._dialect == "tsql"

        register(dialect="postgres")
        assert mock_formatters["sqlnbfmt"]._dialect == "postgres"


@pytest.fixture
def code_formatter():
    """A stand-in jupyterlab_code_formatter module; yields its SERVER_FORMATTERS."""
    formatters = {}
    module = MagicMock(
        BaseFormatter=type("BaseFormatter", (), {}), SERVER_FORMATTERS=formatters
    )
    with patch.dict(
        "sys.modules",
        {
            "jupyterlab_code_formatter": MagicMock(),
            "jupyterlab_code_formatter.formatters": module,
        },
    ):
        yield formatters


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_register_rejects_unknown_dialect(code_formatter, isolated_cwd):
    from sqlnbfmt.jupyterlab_integration import register

    with pytest.raises(ValueError, match="Did you mean postgres"):
        register(dialect="postgress")
    assert "sqlnbfmt" not in code_formatter


def test_format_code_rejects_unknown_dialect_option(code_formatter, isolated_cwd):
    """A bad per-request dialect surfaces as a cell error, not a silent no-op."""
    from sqlnbfmt.jupyterlab_integration import register

    register()
    with pytest.raises(ValueError, match="Did you mean postgres"):
        code_formatter["sqlnbfmt"].format_code(
            "%%sql\nselect 1", notebook=True, dialect="postgress"
        )


def test_extension_keeps_user_registration(code_formatter, isolated_cwd):
    """register(dialect=...) in jupyter_server_config.py runs before the extension loads."""
    from sqlnbfmt import _load_jupyter_server_extension
    from sqlnbfmt.jupyterlab_integration import register

    register(dialect="postgres")
    user_formatter = code_formatter["sqlnbfmt"]
    _load_jupyter_server_extension(MagicMock())
    assert code_formatter["sqlnbfmt"] is user_formatter


def test_extension_reports_config_errors(code_formatter, isolated_cwd):
    from sqlnbfmt import _load_jupyter_server_extension

    (isolated_cwd / "pyproject.toml").write_text(
        '[tool.sqlnbfmt]\ndialect = "postgress"\n'
    )
    server_app = MagicMock()
    _load_jupyter_server_extension(server_app)
    assert "sqlnbfmt" not in code_formatter
    server_app.log.error.assert_called_once()
    assert "Did you mean postgres" in server_app.log.error.call_args.args[0]


def test_extension_quiet_without_code_formatter():
    from sqlnbfmt import _load_jupyter_server_extension

    server_app = MagicMock()
    with patch.dict("sys.modules", {"jupyterlab_code_formatter": None}):
        _load_jupyter_server_extension(server_app)
    server_app.log.error.assert_not_called()
    server_app.log.debug.assert_called_once()
