"""Fixture-based eval test suite for sqlnbfmt.

Auto-discovers test cases from tests/eval/*/ directories.
Jupyter cases: input.ipynb + expected.ipynb.
Marimo/Python cases: input.py + expected.py.
"""

import logging
import shutil
from pathlib import Path

import nbformat
import pytest

from sqlnbfmt.formatter import load_config, process_notebook, process_python_file

EVAL_DIR = Path(__file__).resolve().parent / "eval"
DIALECT = "mysql"


def discover_eval_cases():
    """Find all eval case directories containing input.ipynb + expected.ipynb."""
    if not EVAL_DIR.is_dir():
        return []
    cases = []
    for case_dir in sorted(EVAL_DIR.iterdir()):
        if (
            case_dir.is_dir()
            and (case_dir / "input.ipynb").exists()
            and (case_dir / "expected.ipynb").exists()
        ):
            cases.append(case_dir.name)
    return cases


@pytest.fixture
def logger():
    log = logging.getLogger("test_eval")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(handler)
    return log


@pytest.mark.parametrize("case_name", discover_eval_cases(), ids=lambda c: c)
def test_eval(case_name, logger, tmp_path):
    """Run formatter on input.ipynb and compare cell-by-cell with expected.ipynb."""
    case_dir = EVAL_DIR / case_name

    # Copy input to temp directory so we don't modify fixtures
    work_path = tmp_path / "notebook.ipynb"
    shutil.copy2(case_dir / "input.ipynb", work_path)

    # Load config and run formatter
    config = load_config()
    process_notebook(work_path, config, DIALECT, logger)

    # Read formatted and expected
    formatted_nb = nbformat.read(work_path, as_version=4)
    expected_nb = nbformat.read(case_dir / "expected.ipynb", as_version=4)

    # Compare cell-by-cell
    assert len(formatted_nb.cells) == len(expected_nb.cells), (
        f"[{case_name}] Cell count mismatch: "
        f"got {len(formatted_nb.cells)}, expected {len(expected_nb.cells)}"
    )

    for i, (fmt_cell, exp_cell) in enumerate(
        zip(formatted_nb.cells, expected_nb.cells)
    ):
        assert fmt_cell.cell_type == exp_cell.cell_type, (
            f"[{case_name}] Cell {i} type mismatch: "
            f"got {fmt_cell.cell_type}, expected {exp_cell.cell_type}"
        )
        assert fmt_cell.source == exp_cell.source, (
            f"[{case_name}] Cell {i} content mismatch.\n"
            f"Expected:\n{exp_cell.source}\n"
            f"Actual:\n{fmt_cell.source}"
        )


def discover_marimo_eval_cases():
    """Find eval case directories containing input.py + expected.py."""
    if not EVAL_DIR.is_dir():
        return []
    cases = []
    for case_dir in sorted(EVAL_DIR.iterdir()):
        if (
            case_dir.is_dir()
            and (case_dir / "input.py").exists()
            and (case_dir / "expected.py").exists()
        ):
            cases.append(case_dir.name)
    return cases


@pytest.mark.parametrize("case_name", discover_marimo_eval_cases(), ids=lambda c: c)
def test_marimo_eval(case_name, logger, tmp_path):
    """Run formatter on input.py and compare with expected.py."""
    case_dir = EVAL_DIR / case_name
    work_path = tmp_path / "notebook.py"
    shutil.copy2(case_dir / "input.py", work_path)

    config = load_config()
    process_python_file(work_path, config, DIALECT, logger)

    formatted = work_path.read_text()
    expected = (case_dir / "expected.py").read_text()
    assert formatted == expected, (
        f"[{case_name}] Content mismatch.\nExpected:\n{expected}\nActual:\n{formatted}"
    )
