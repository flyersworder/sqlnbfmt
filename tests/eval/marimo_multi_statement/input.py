import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    _df = mo.sql(f"drop table if exists temp_users; create table temp_users as select id, name from users where active = 1; select * from temp_users")
    return


if __name__ == "__main__":
    app.run()
