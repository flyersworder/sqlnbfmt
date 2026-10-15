import os
import subprocess
import tomllib

import pytest

from sqlnbfmt.formatter import find_pyproject_config, load_config

PYPROJECT_WITH_TOOL = """\
[project]
name = "demo"

[tool.sqlnbfmt]
function_names = ["pull_rows"]
sql_decorators = ["my_query"]

[tool.sqlnbfmt.formatting_options]
indent_width = 2
"""

PYPROJECT_WITHOUT_TOOL = """\
[project]
name = "demo"
"""

# `pull_rows` is not a default function name, so this only gets formatted
# when the [tool.sqlnbfmt] config above has actually been loaded.
UNFORMATTED_SOURCE = 'pull_rows("select id, name from users where active = 1")\n'


@pytest.fixture
def repo(tmp_path):
    """A fake repo root, so discovery never walks above tmp_path."""
    (tmp_path / ".git").mkdir()
    return tmp_path


def test_load_config_from_pyproject(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT_WITH_TOOL)
    config = load_config(pyproject)
    assert config.function_names == {"pull_rows"}
    assert config.sql_decorators == {"my_query"}
    assert config.indent_width == 2
    # Unset options keep their defaults
    assert "SELECT" in config.sql_keywords
    assert config.single_line_threshold == 80


def test_load_config_explicit_toml_without_tool_table(tmp_path):
    """An explicitly passed .toml without [tool.sqlnbfmt] is a mistake, not defaults."""
    pyproject = tmp_path / "sqlnbfmt.toml"
    pyproject.write_text('dialect = "postgres"\n')
    with pytest.raises(ValueError, match=r"no \[tool\.sqlnbfmt\] table"):
        load_config(pyproject)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
@pytest.mark.parametrize("name", ["pyproject.toml", "config.yaml"])
def test_load_config_unreadable_file(tmp_path, name):
    config_file = tmp_path / name
    config_file.write_text("[tool.sqlnbfmt]\n")
    config_file.chmod(0)
    try:
        with pytest.raises(ValueError, match="Could not read"):
            load_config(config_file)
    finally:
        config_file.chmod(0o644)


def test_load_config_malformed_pyproject(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.sqlnbfmt\n")
    with pytest.raises(ValueError, match="pyproject.toml") as excinfo:
        load_config(pyproject)
    assert isinstance(excinfo.value.__cause__, tomllib.TOMLDecodeError)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('sql_keywords = "SELECT"', "sql_keywords"),
        ("function_names = [1, 2]", "function_names"),
        ('formatting_options = { indent_width = "2" }', "indent_width"),
    ],
)
def test_load_config_rejects_wrong_types(tmp_path, body, message):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f"[tool.sqlnbfmt]\n{body}\n")
    with pytest.raises(ValueError, match=message):
        load_config(pyproject)


def test_find_pyproject_in_start_dir(repo):
    (repo / "pyproject.toml").write_text(PYPROJECT_WITH_TOOL)
    assert find_pyproject_config(repo) == repo / "pyproject.toml"


def test_find_pyproject_walks_up(repo):
    (repo / "pyproject.toml").write_text(PYPROJECT_WITH_TOOL)
    nested = repo / "notebooks" / "analysis"
    nested.mkdir(parents=True)
    assert find_pyproject_config(nested) == repo / "pyproject.toml"


def test_find_pyproject_skips_files_without_tool_table(repo):
    """A sub-package pyproject.toml without [tool.sqlnbfmt] doesn't hide the root one."""
    (repo / "pyproject.toml").write_text(PYPROJECT_WITH_TOOL)
    package = repo / "packages" / "etl"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(PYPROJECT_WITHOUT_TOOL)
    assert find_pyproject_config(package) == repo / "pyproject.toml"


def test_find_pyproject_stops_at_repo_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT_WITH_TOOL)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    assert find_pyproject_config(repo) is None


def test_find_pyproject_none_found(repo):
    assert find_pyproject_config(repo) is None


def _run_check(cwd, *extra_args):
    return subprocess.run(
        ["sqlnbfmt", "--check", *extra_args, "query.py"],
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
    )


def test_cli_uses_discovered_pyproject(repo):
    (repo / "query.py").write_text(UNFORMATTED_SOURCE)
    assert _run_check(repo).returncode == 0  # pull_rows unknown -> untouched

    (repo / "pyproject.toml").write_text(PYPROJECT_WITH_TOOL)
    result = _run_check(repo, "--log-level", "DEBUG")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Using configuration from" in result.stdout
    assert "pyproject.toml" in result.stdout


def test_cli_config_flag_overrides_pyproject(repo):
    (repo / "query.py").write_text(UNFORMATTED_SOURCE)
    (repo / "pyproject.toml").write_text(PYPROJECT_WITH_TOOL)
    (repo / "config.yaml").write_text("formatting_options:\n  indent_width: 2\n")
    result = _run_check(repo, "--config", "config.yaml")
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_reports_invalid_pyproject(repo):
    (repo / "query.py").write_text(UNFORMATTED_SOURCE)
    (repo / "pyproject.toml").write_text('[tool.sqlnbfmt]\nsql_keywords = "SELECT"\n')
    result = _run_check(repo)
    assert result.returncode == 1
    assert "sql_keywords" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_unknown_keys_warn(tmp_path, caplog):
    """Misplaced or misspelled options are reported instead of silently ignored."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.sqlnbfmt]\n"
        "indent_width = 2\n"
        "function_name = ['pull_rows']\n"
        "[tool.sqlnbfmt.formatting_options]\n"
        "indent = 2\n"
    )
    config = load_config(pyproject)
    assert config.indent_width == 4
    assert "did you mean 'formatting_options.indent_width'" in caplog.text
    assert "'function_name'" in caplog.text
    assert "'formatting_options.indent'" in caplog.text


# T-SQL rewrites LIMIT to TOP, so the output reveals which dialect was applied.
LIMIT_SOURCE = 'pull_rows("select id from users limit 10")\n'


def _format_file(cwd, *extra_args):
    result = subprocess.run(
        ["sqlnbfmt", *extra_args, "query.py"],
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
    )
    return result, (cwd / "query.py").read_text()


def test_load_config_dialect(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.sqlnbfmt]\ndialect = "postgres"\n')
    assert load_config(pyproject).dialect == "postgres"


def test_load_config_dialect_defaults_to_none(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.sqlnbfmt]\n")
    assert load_config(pyproject).dialect is None


@pytest.mark.parametrize(
    ("value", "message"),
    [('"postgress"', "Did you mean postgres"), ("5", "dialect")],
)
def test_load_config_rejects_invalid_dialect(tmp_path, value, message):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f"[tool.sqlnbfmt]\ndialect = {value}\n")
    with pytest.raises(ValueError, match=message):
        load_config(pyproject)


def test_cli_uses_dialect_from_pyproject(repo):
    (repo / "query.py").write_text(LIMIT_SOURCE)
    (repo / "pyproject.toml").write_text(
        '[tool.sqlnbfmt]\nfunction_names = ["pull_rows"]\ndialect = "tsql"\n'
    )
    result, formatted = _format_file(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TOP 10" in formatted


def test_cli_dialect_flag_overrides_pyproject(repo):
    (repo / "query.py").write_text(LIMIT_SOURCE)
    (repo / "pyproject.toml").write_text(
        '[tool.sqlnbfmt]\nfunction_names = ["pull_rows"]\ndialect = "tsql"\n'
    )
    result, formatted = _format_file(repo, "--dialect", "postgres")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LIMIT 10" in formatted


def test_cli_rejects_unknown_dialect_flag(repo):
    """A typo'd dialect must fail loudly, not report everything as formatted."""
    (repo / "query.py").write_text(LIMIT_SOURCE)
    result = _run_check(repo, "--dialect", "postgress")
    assert result.returncode == 2
    assert "Did you mean postgres" in result.stderr
