"""Live end-to-end tests against the built wheel.

These build sqlnbfmt, install it into a fresh virtualenv together with Jupyter
Server and jupyterlab-code-formatter, and drive every real entry point against a
scratch project configured only through pyproject.toml:

- the ``sqlnbfmt`` CLI
- the pre-commit hooks from ``.pre-commit-hooks.yaml`` (via ``pre-commit try-repo``)
- the Jupyter server extension, through jupyterlab-code-formatter's HTTP API

They need ``uv``, ``git`` and network access, and take a minute or so, so they
are excluded from the default test run. Run them before a release with::

    uv run pytest -m live

The scratch project uses ``dialect = "tsql"`` because T-SQL rewrites ``LIMIT``
to ``TOP``, which makes the applied dialect visible in the output.
"""

import ast
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import nbformat
import pytest
from nbformat import v4 as nbf

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (shutil.which("uv") and shutil.which("git")),
        reason="live tests need uv and git on PATH",
    ),
]

REPO_ROOT = Path(__file__).resolve().parent.parent

PYPROJECT = """\
[project]
name = "live-demo"
version = "0.0.0"

[tool.sqlnbfmt]
dialect = "tsql"
"""

MARIMO_APP = '''\
import marimo

app = marimo.App()


@app.cell
def _(mo):
    _df = mo.sql(f"""select id, name from users where active = 1 order by name limit 5""")
    return
'''

NOTEBOOK_CELLS = [
    (
        'df = pd.read_sql("select a.id, b.total from accounts a join billing b '
        'on a.id = b.account_id where b.total > 100 limit 20", conn)'
    ),
    "%%sql\nselect count(*) as n from events where kind = 'click' limit 3",
    'print("hello world")',
]


def run(*args, cwd=None, env=None, check=True):
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture(scope="module")
def venv_bin(tmp_path_factory):
    """Bin directory of a fresh venv with the built wheel and Jupyter installed."""
    root = tmp_path_factory.mktemp("live-env")
    run("uv", "build", "--wheel", "--no-sources", "-o", root / "dist", cwd=REPO_ROOT)
    (wheel,) = (root / "dist").glob("sqlnbfmt-*.whl")
    venv = root / "venv"
    run("uv", "venv", "-q", "--python", sys.executable, venv)
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    run(
        "uv", "pip", "install", "-q", "--python", bin_dir / "python",
        wheel, "jupyter-server", "jupyterlab-code-formatter",
    )  # fmt: skip
    return bin_dir


@pytest.fixture
def project(tmp_path):
    """A git repo with unformatted notebook and Marimo files, configured via pyproject."""
    project = tmp_path / "project"
    notebooks = project / "notebooks"
    notebooks.mkdir(parents=True)
    (project / "pyproject.toml").write_text(PYPROJECT)
    (notebooks / "marimo_app.py").write_text(MARIMO_APP)
    nb = nbf.new_notebook()
    nb.cells = [nbf.new_code_cell(source) for source in NOTEBOOK_CELLS]
    nbformat.write(nb, notebooks / "analysis.ipynb")
    run("git", "init", "-q", cwd=project)
    run("git", "add", "-A", cwd=project)
    return project


def notebook_sources(path):
    return [cell.source for cell in nbformat.read(path, as_version=4).cells]


def test_cli_formats_with_discovered_pyproject_config(venv_bin, project):
    sqlnbfmt = venv_bin / "sqlnbfmt"
    notebooks = project / "notebooks"
    files = ["analysis.ipynb", "marimo_app.py"]

    # Run from a subdirectory: the config is found by walking up.
    check = run(sqlnbfmt, "--check", "--log-level", "DEBUG", *files, cwd=notebooks, check=False)  # fmt: skip
    assert check.returncode == 1, check.stdout + check.stderr
    assert f"Using configuration from {project / 'pyproject.toml'}" in check.stdout

    run(sqlnbfmt, *files, cwd=notebooks)
    marimo = (notebooks / "marimo_app.py").read_text()
    cells = notebook_sources(notebooks / "analysis.ipynb")

    assert "TOP 5" in marimo and "TOP 20" in cells[0] and "TOP 3" in cells[1]
    assert cells[2] == NOTEBOOK_CELLS[2]  # non-SQL untouched
    ast.parse(marimo)
    # The SQL block is indented under mo.sql( and the paren returns to the call's level
    assert '    _df = mo.sql(\n        f"""\n        SELECT\n' in marimo
    assert '        """\n    )\n    return\n' in marimo

    recheck = run(sqlnbfmt, "--check", *files, cwd=notebooks, check=False)
    assert recheck.returncode == 0, recheck.stdout + recheck.stderr


def test_cli_rejects_unknown_dialect(venv_bin, project):
    result = run(
        venv_bin / "sqlnbfmt", "--check", "--dialect", "postgress", "notebooks/marimo_app.py",
        cwd=project, check=False,
    )  # fmt: skip
    assert result.returncode == 2
    assert "Did you mean postgres" in result.stderr


def try_repo(project, hook):
    return run(
        "uvx", "pre-commit", "try-repo", REPO_ROOT, hook, "--all-files",
        cwd=project, check=False,
    )  # fmt: skip


def test_pre_commit_hooks(project):
    for hook in ("sqlnbfmt-check", "sqlnbfmt-py-check"):
        result = try_repo(project, hook)
        assert result.returncode == 1, f"{hook} should fail:\n{result.stdout}"

    for hook in ("sqlnbfmt", "sqlnbfmt-py"):
        result = try_repo(project, hook)
        assert "files were modified by this hook" in result.stdout, result.stdout

    # The hooks read [tool.sqlnbfmt] without any args
    assert "TOP 5" in (project / "notebooks" / "marimo_app.py").read_text()

    for hook in ("sqlnbfmt-check", "sqlnbfmt-py-check"):
        result = try_repo(project, hook)
        assert result.returncode == 0, f"{hook} should pass:\n{result.stdout}"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def jupyter_server(venv_bin, cwd, config_dir):
    """Runs a Jupyter server; yields (request, log_path)."""
    port, token = free_port(), "live-test"
    log_path = cwd.parent / "jupyter.log"
    env = {**os.environ, "JUPYTER_CONFIG_DIR": str(config_dir)}
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [
                venv_bin / "jupyter", "server", "--no-browser",
                f"--port={port}", "--port-retries=0", "--ServerApp.ip=127.0.0.1",
                f"--ServerApp.token={token}",
            ],
            cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
        )  # fmt: skip

    def request(path, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"token {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                request("/api/status")
                break
            except (urllib.error.URLError, ConnectionError):
                if server.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(
                        f"Jupyter server did not start:\n{log_path.read_text()}"
                    )
                time.sleep(0.5)
        yield request, log_path
    finally:
        server.terminate()
        server.wait(timeout=10)


def format_cells(request, cells, **options):
    body = {"formatter": "sqlnbfmt", "notebook": True, "code": cells}
    if options:
        body["options"] = options
    return [result["code"] for result in request("/jupyterlab_code_formatter/format", body)["code"]]  # fmt: skip


def test_jupyter_extension_uses_pyproject_config(venv_bin, project, tmp_path):
    config_dir = tmp_path / "jupyter-config"  # isolate from the user's own config
    config_dir.mkdir()
    with jupyter_server(venv_bin, project, config_dir) as (request, _):
        formatters = request("/jupyterlab_code_formatter/formatters")["formatters"]
        assert formatters["sqlnbfmt"]["enabled"] is True

        formatted = format_cells(request, NOTEBOOK_CELLS)
        assert "TOP 20" in formatted[0] and "TOP 3" in formatted[1]
        assert formatted[2] == NOTEBOOK_CELLS[2]

        # A per-request dialect (the JupyterLab setting) overrides the config
        assert "LIMIT 3" in format_cells(request, [NOTEBOOK_CELLS[1]], dialect="postgres")[0]  # fmt: skip


def test_jupyter_extension_keeps_user_registration(venv_bin, project, tmp_path):
    """register() in jupyter_server_config.py wins over the auto-registration."""
    config_dir = tmp_path / "jupyter-config"
    config_dir.mkdir()
    (config_dir / "jupyter_server_config.py").write_text(
        "from sqlnbfmt.jupyterlab_integration import register\n"
        'register(dialect="postgres")\n'
    )
    with jupyter_server(venv_bin, project, config_dir) as (request, _):
        assert "LIMIT 3" in format_cells(request, [NOTEBOOK_CELLS[1]])[0]


def test_jupyter_extension_reports_config_errors(venv_bin, project, tmp_path):
    config_dir = tmp_path / "jupyter-config"
    config_dir.mkdir()
    (project / "pyproject.toml").write_text('[tool.sqlnbfmt]\ndialect = "postgress"\n')
    with jupyter_server(venv_bin, project, config_dir) as (request, log_path):
        formatters = request("/jupyterlab_code_formatter/formatters")["formatters"]
        assert "sqlnbfmt" not in formatters
        log = log_path.read_text()
        assert "sqlnbfmt: SQL formatter not registered" in log
        assert "Did you mean postgres" in log
