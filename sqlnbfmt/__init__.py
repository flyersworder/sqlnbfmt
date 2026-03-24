def _jupyter_server_extension_points():
    return [{"module": "sqlnbfmt"}]


def _load_jupyter_server_extension(server_app):
    try:
        from sqlnbfmt.jupyterlab_integration import register

        register()
        server_app.log.info(
            "sqlnbfmt: registered SQL formatter with jupyterlab-code-formatter"
        )
    except Exception:
        # jupyterlab-code-formatter not installed — silently skip
        pass
