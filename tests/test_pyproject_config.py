import json
import logging
import os
import subprocess
import tomllib

import pytest

from sqlnbfmt.formatter import (
    DEFAULT_FUNCTION_NAMES,
    DEFAULT_SQL_DECORATORS,
    DEFAULT_SQL_KEYWORDS,
    _format_python_source,
    find_pyproject_config,
    load_config,
)

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


# --- extend_* options (#26) ------------------------------------------------


def _write_config(tmp_path, fmt, options):
    """Writes ``options`` (key -> list of strings) as a pyproject.toml or YAML config."""
    if fmt == "toml":
        lines = [f"{key} = {json.dumps(value)}" for key, value in options.items()]
        path = tmp_path / "pyproject.toml"
        path.write_text("[tool.sqlnbfmt]\n" + "\n".join(lines) + "\n")
    else:
        lines = [f"{key}: {json.dumps(value)}" for key, value in options.items()]
        path = tmp_path / "config.yaml"
        path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(params=["toml", "yaml"])
def config_format(request):
    return request.param


def test_extend_adds_to_defaults(tmp_path, config_format):
    path = _write_config(
        tmp_path,
        config_format,
        {
            "extend_function_names": ["run_query"],
            "extend_sql_decorators": ["warehouse_query"],
            "extend_sql_keywords": ["QUALIFY"],
        },
    )
    config = load_config(path)
    assert config.function_names == set(DEFAULT_FUNCTION_NAMES) | {"run_query"}
    assert config.sql_decorators == set(DEFAULT_SQL_DECORATORS) | {"warehouse_query"}
    assert config.sql_keywords == set(DEFAULT_SQL_KEYWORDS) | {"QUALIFY"}


def test_extend_adds_to_replaced_list(tmp_path, config_format):
    path = _write_config(
        tmp_path,
        config_format,
        {"function_names": ["read_sql"], "extend_function_names": ["run_query"]},
    )
    assert load_config(path).function_names == {"read_sql", "run_query"}


def test_extend_rejects_wrong_type(tmp_path, config_format):
    path = _write_config(
        tmp_path, config_format, {"extend_function_names": "run_query"}
    )
    with pytest.raises(ValueError, match="extend_function_names"):
        load_config(path)


def test_extend_keys_are_not_reported_as_unknown(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="formatter")
    path = _write_config(
        tmp_path, "toml", {"extend_function_names": ["run_query"], "bogus": ["x"]}
    )
    load_config(path)
    assert "'bogus'" in caplog.text  # warnings are captured...
    assert "extend_function_names" not in caplog.text  # ...but not for extend_*


@pytest.mark.parametrize(
    ("toml", "suggestion"),
    [
        ('extend-function-names = ["run_query"]', "extend_function_names"),
        ("indent-width = 2", "formatting_options.indent_width"),
        (
            "[tool.sqlnbfmt.formatting_options]\nindent-width = 2",
            "formatting_options.indent_width",
        ),
    ],
)
def test_kebab_case_key_suggests_snake_case(tmp_path, caplog, toml, suggestion):
    caplog.set_level(logging.WARNING, logger="formatter")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f"[tool.sqlnbfmt]\n{toml}\n")
    config = load_config(pyproject)
    assert config.indent_width == 4 and "run_query" not in config.function_names
    assert f"did you mean '{suggestion}'" in caplog.text


def test_extend_keeps_marimo_formatting(tmp_path):
    """The point of #26: adding a function doesn't drop `sql` (mo.sql)."""
    path = _write_config(tmp_path, "toml", {"extend_function_names": ["pull_rows"]})
    source = (
        'def _(mo):\n    _df = mo.sql(f"select id from users limit 5")\n'
        'pull_rows("select id from users limit 5")\n'
    )
    formatted, _ = _format_python_source(
        source, load_config(path), None, logging.getLogger("test")
    )
    assert formatted.count("LIMIT 5") == 2


@pytest.mark.parametrize("key", ["sql_keywords", "extend_sql_keywords"])
def test_sql_keywords_are_case_insensitive(tmp_path, config_format, key):
    """Detection compares against upper-cased code, so keywords are stored upper-case."""
    path = _write_config(tmp_path, config_format, {key: ["merge", "Using"]})
    config = load_config(path)
    assert {"MERGE", "USING"} <= config.sql_keywords
    assert "merge" not in config.sql_keywords


@pytest.mark.parametrize("key", ["function_names", "extend_function_names"])
@pytest.mark.parametrize("value", ["", "  "])
def test_lists_reject_blank_entries(tmp_path, key, value):
    """A blank function name would match every call (names match as substrings)."""
    path = _write_config(tmp_path, "toml", {key: ["read_sql", value]})
    with pytest.raises(ValueError, match=key):
        load_config(path)


@pytest.mark.parametrize(
    ("config_format", "text"),
    [
        ("toml", "[tool.sqlnbfmt.formatting_options]\nsingle_line_threshold = 120\n"),
        ("yaml", "formatting_options:\n  single_line_threshold: 120\n"),
    ],
)
def test_single_line_threshold_is_deprecated(tmp_path, caplog, config_format, text):
    """It never had any effect; setting it warns but still loads."""
    caplog.set_level(logging.WARNING, logger="formatter")
    path = tmp_path / ("pyproject.toml" if config_format == "toml" else "config.yaml")
    path.write_text(text)
    load_config(path)
    assert "single_line_threshold' is deprecated and has no effect" in caplog.text
    assert "unknown" not in caplog.text


def test_no_deprecation_warning_when_unset(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="formatter")
    path = tmp_path / "pyproject.toml"
    path.write_text("[tool.sqlnbfmt.formatting_options]\nindent_width = 2\n")
    load_config(path)
    assert caplog.text == ""


def test_config_template_does_not_use_deprecated_options(caplog):
    from pathlib import Path

    caplog.set_level(logging.WARNING, logger="formatter")
    load_config(Path(__file__).parent.parent / "config.yaml")
    assert caplog.text == ""
