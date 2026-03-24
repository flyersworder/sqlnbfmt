import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    # Fetch active users for the dashboard
    # This query powers the main reporting view
    _df = mo.sql(
    f"""
    SELECT
      id,
      name,
      email
    FROM users
    WHERE
      active = 1
    ORDER BY
      name
    """
)
    return


@app.cell
def _(mo):
    threshold = 100
    # Orders above threshold
    result = mo.sql(
    f"""
    SELECT
      *
    FROM orders
    WHERE
      total > {threshold}
    """
)  # important query
    return (result,)


if __name__ == "__main__":
    app.run()
