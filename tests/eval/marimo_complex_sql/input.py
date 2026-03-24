import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    _df = mo.sql(f"with active_users as (select id, name from users where active = 1) select a.id, a.name, count(o.id) as order_count from active_users a left join orders o on a.id = o.user_id group by a.id, a.name having count(o.id) > 0 order by order_count desc")
    return


@app.cell
def _(mo):
    dept = "engineering"
    _df = mo.sql(f"select id, name, salary, row_number() over (partition by department order by salary desc) as rank from employees where department = {dept}")
    return


if __name__ == "__main__":
    app.run()
