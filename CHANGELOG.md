# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `extend_sql_keywords`, `extend_function_names` and `extend_sql_decorators` config options, which add entries to the built-in lists instead of replacing them, so adding one function name no longer drops defaults like `sql` (Marimo's `mo.sql()`) ([#26](https://github.com/flyersworder/sqlnbfmt/issues/26))
- The unknown-option warning suggests the intended option for kebab-case keys (e.g. `extend-function-names`, `indent-width`), at the top level and under `formatting_options`

### Fixed

- **`sql_decorators` now works.** The option was parsed and documented but never used by the formatter in any version ([#30](https://github.com/flyersworder/sqlnbfmt/issues/30)). SQL in decorator arguments (`@query("SELECT ...")`) and string literals returned from decorated functions (`@query` above `def f(): return "SELECT ..."`) are now formatted. Decorator names match like function names, and docstrings and returns of nested functions are left alone
- **Rewritten SQL strings could corrupt the code.** When writing formatted SQL back, sqlnbfmt now keeps the value of the string exactly. In particular:
  - backslash escapes are preserved (`'\\n'` used to become a real newline)
  - the raw `r` prefix is kept (dropping it turned `\_` into an invalid escape)
  - quotes are chosen so they don't clash with the SQL (`cursor.execute('delete from "users"')` produced broken Python, as did `"""` inside the SQL)
  - literal `{{ }}` braces in f-strings are no longer turned into placeholders

  Each rewritten literal is parsed back and checked; a string that can't be written back faithfully is left unchanged
- `sql_keywords` entries are now case-insensitive. SQL detection compares against upper-cased code, so a lowercase entry such as `"merge"` was silently never matched
- Blank entries in `sql_keywords`, `function_names` and `sql_decorators` (and their `extend_*` forms) are rejected. Function names match as substrings, so `""` matched every call

## [0.5.0] - 2026-10-15

### Added

- **`pyproject.toml` configuration**: set options in a `[tool.sqlnbfmt]` table instead of a separate YAML file. Without `--config`, sqlnbfmt uses the nearest `pyproject.toml` that has the table, searching up to the repository root. The JupyterLab integration uses the same discovery ([#22](https://github.com/flyersworder/sqlnbfmt/issues/22))
- `--config` accepts a `.toml` path as well as YAML
- `dialect` config option (in `[tool.sqlnbfmt]` or YAML), so the dialect no longer has to be passed as `--dialect` on every run; `--dialect` still overrides it

### Changed

- Config values are now type-checked: a wrong type (e.g. a string where a list is expected) is reported as an error naming the file. Previously such values were silently mis-parsed or crashed with a traceback
- Unknown config options now produce a warning instead of being silently ignored

### Fixed

- SQL passed to a function call is now indented relative to the line the call is on, and a closing `)` or remaining arguments no longer land at column 0. This was most visible in Marimo cells, where every `mo.sql(...)` sits inside a function. It had regressed in 0.3.0, when the `black` post-processing pass was removed. Code already formatted by an earlier version keeps its layout until its SQL changes
- The Jupyter server extension no longer overrides a formatter registered in `jupyter_server_config.py` (e.g. `register(dialect="postgres")`), and it now logs configuration errors instead of silently not registering
- An unknown dialect (e.g. `--dialect postgress`) is now rejected with a suggestion. Previously every SQL string silently failed to parse, so `--check` reported all files as properly formatted
- The `config.yaml` template was missing the `sql` function name (default since 0.4.0). Because config lists replace the defaults, copying the template silently disabled Marimo `mo.sql()` formatting. The README now states the replace semantics explicitly
- Upgraded locked dependencies, including sqlglot 29 → 30 (formatting output unchanged across the eval suite)
- A skipped Jupyter extension registration is now logged at debug level instead of being silently swallowed

## [0.4.2] - 2026-03-24

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
