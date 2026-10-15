def _jupyter_server_extension_points():
    return [{"module": "sqlnbfmt"}]


def _load_jupyter_server_extension(server_app):
    try:
        from jupyterlab_code_formatter.formatters import SERVER_FORMATTERS
    except ImportError:
        server_app.log.debug(
            "sqlnbfmt: jupyterlab-code-formatter not installed; skipping registration"
        )
        return

    # jupyter_server_config.py runs before extensions load; keep a formatter the
    # user registered there (e.g. register(dialect="postgres")).
    if "sqlnbfmt" in SERVER_FORMATTERS:
        server_app.log.info("sqlnbfmt: keeping the formatter registered in config")
        return

    try:
        from sqlnbfmt.jupyterlab_integration import register

        register()
    except Exception as e:  # noqa: BLE001 - must never break Jupyter server startup
        server_app.log.error(f"sqlnbfmt: SQL formatter not registered: {e}")
        return
    server_app.log.info(
        "sqlnbfmt: registered SQL formatter with jupyterlab-code-formatter"
    )
