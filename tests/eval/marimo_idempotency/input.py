import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    _df = mo.sql(
    f"""
    SELECT
      id,
      name
    FROM users
    WHERE
      active = 1
    ORDER BY
      name
    """
)
    return


if __name__ == "__main__":
    app.run()
