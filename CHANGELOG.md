# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - Unreleased

### Added

- **Jupyter server extension**: `pip install sqlnbfmt` now auto-registers with jupyterlab-code-formatter — no manual config needed

### Changed

- Switched build system from setuptools to hatchling
- Updated JupyterLab docs to use `jupyter_server_config.py` (modern) instead of `jupyter_notebook_config.py` (legacy)

## [0.4.0] - 2026-03-24

### Added

- **Marimo notebook support**: format SQL in Marimo `.py` notebooks via `mo.sql()` calls — same `sqlnbfmt` command, auto-detected by file extension
- **Python file support**: `sqlnbfmt` now accepts `.py` files alongside `.ipynb` — works on any Python file containing SQL strings
- **JupyterLab integration**: `sqlnbfmt.jupyterlab_integration.register()` for use with `jupyterlab-code-formatter`
- `format_cell_source()` public API for formatting a single cell's source code
- `_format_python_source()`, `process_python_file()`, `diff_python_file()` public API for programmatic use
- `sql` added to default recognized function names (matches `mo.sql()`, etc.)
- `sqlnbfmt-py` and `sqlnbfmt-py-check` pre-commit hooks for Python files
- Marimo eval fixtures: `marimo_basic_sql`, `marimo_fstring_interpolation`, `marimo_complex_sql`, `marimo_multi_cell_mixed`, `marimo_comment_preservation`, `marimo_multi_statement`, `marimo_skip_hint`, `marimo_non_sql_unchanged`, `marimo_idempotency`

### Fixed

- Idempotency: already-formatted SQL in triple-quoted strings (with leading whitespace/indentation) is no longer re-formatted on every run

### Changed

- CLI description and help text updated to reflect multi-format support
- Project description and keywords updated for Marimo discoverability

## [0.3.2] - 2026-03-16

### Fixed

- Multiple SQL statements in a single cell are no longer concatenated on one line — each statement now starts on its own line after the `;` separator ([#3](https://github.com/flyersworder/sqlnbfmt/issues/3))

### Changed

- SQL parsing switched from `sqlglot.parse_one()` to `sqlglot.parse()` to correctly handle multi-statement inputs

### Added

- `multi_statement` eval fixture case covering `DROP TABLE` + `CREATE TABLE` sequences

## [0.3.1] - 2026-03-15

### Fixed

- `diff_notebook()` no longer duplicates cell-processing logic — extracted shared `_format_cells()` helper to prevent future divergence
- `--check --diff` combined mode no longer double-counts notebooks or processes them twice
- Skip hint (`# sqlnbfmt: skip`) no longer triggers on string literals containing the directive — only dedicated comment lines are recognised
- `load_config()` with a partial config file (missing keys) now falls back to built-in defaults instead of empty sets
- `load_config()` with an empty YAML file no longer raises `AttributeError`
- In-function SQL indentation now respects `indent_width` config instead of hardcoded 4 spaces

### Added

- `main()` integration tests for `--diff`, `--check`, and `--check --diff` exit code behavior
- Docstring on `_has_skip_hint()` clarifying that inline trailing comments are intentionally ignored

## [0.3.0] - 2026-03-15

### Added

- **Zero-config defaults**: sqlnbfmt now works out of the box without a `config.yaml` file — built-in defaults for SQL keywords, function names, and decorators
- **`--check` mode**: verify formatting without modifying files; exits with code 1 if changes are needed (CI-friendly)
- **`--diff` mode**: print a unified diff of formatting changes without modifying files
- **Skip hints**: add `# sqlnbfmt: skip` to any cell to skip formatting
- **`sqlnbfmt-check` pre-commit hook**: a read-only hook variant for CI pipelines
- Cell index in all warning/debug messages for easier debugging
- PyPI keywords and project URLs in package metadata
- `skip_hint` eval fixture case

### Changed

- `--config` argument is now optional (defaults to built-in settings instead of requiring `config.yaml`)
- `load_config()` accepts `None` to return defaults; passing a missing path still raises `FileNotFoundError`
- Version bumped to 0.3.0

## [0.2.0] - 2026-03-15

### Fixed

- Python comments in code cells are no longer stripped during formatting ([#2](https://github.com/flyersworder/sqlnbfmt/issues/2))

### Changed

- Replaced AST-mutation approach (`ast.NodeTransformer` + `astor` + `black`) with surgical string replacement (`ast.NodeVisitor` + offset-based text splicing), preserving all non-SQL content exactly
- Removed `astor` and `black` from dependencies — fewer install requirements
- CI publish now uses PyPI Trusted Publishers (OIDC) instead of API tokens
- Migrated `tool.uv.dev-dependencies` to `dependency-groups.dev` (PEP 735)
- Updated `actions/checkout` from v3 to v4

### Added

- Python 3.13 classifier
- Comment-preservation test case

### Fixed

- Typo in config.yaml: `exeucte_query` → `execute_query`

## [0.1.9] - 2025-01-01

### Changed

- Updated dependencies (attrs, black, referencing, sqlglot, typing-extensions)

## [0.1.8] - 2024-12-01

### Fixed

- First-line indentation error for in-function SQL
- New-line handling issues in formatted output
- Introduced `black` formatter to resolve in-function SQL formatting edge cases

## [0.1.7] - 2024-11-01

### Added

- Pre-commit hook support
- F-string SQL formatting (with and without variables)
- In-function SQL formatting (e.g., `pd.read_sql(...)`)

## [0.1.0] - 2024-10-01

### Added

- Initial release
- SQL formatting in Jupyter notebook code cells using sqlglot
- Support for `%sql` / `%%sql` magic commands
- Configurable SQL keywords, function names, and formatting options
- GitHub Actions CI pipeline
