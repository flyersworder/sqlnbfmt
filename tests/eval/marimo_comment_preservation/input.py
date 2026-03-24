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
    _df = mo.sql(f"select id, name, email from users where active = 1 order by name")
    return


@app.cell
def _(mo):
    threshold = 100
    # Orders above threshold
    result = mo.sql(f"select * from orders where total > {threshold}")  # important query
    return (result,)


if __name__ == "__main__":
    app.run()
