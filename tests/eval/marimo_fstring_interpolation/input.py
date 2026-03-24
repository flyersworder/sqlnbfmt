import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    table = "users"
    _df = mo.sql(f"select * from {table} where active = 1")
    return


@app.cell
def _(mo):
    min_age = 18
    _df = mo.sql(f"select id, name from users where age >= {min_age} order by name")
    return


if __name__ == "__main__":
    app.run()
