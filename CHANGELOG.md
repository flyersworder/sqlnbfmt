# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
