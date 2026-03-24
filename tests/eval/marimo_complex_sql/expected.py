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
    WITH active_users AS (
        SELECT
          id,
          name
        FROM users
        WHERE
          active = 1
    )
    SELECT
      a.id,
      a.name,
      COUNT(o.id) AS order_count
    FROM active_users AS a
    LEFT JOIN orders AS o
      ON a.id = o.user_id
    GROUP BY
      a.id,
      a.name
    HAVING
      COUNT(o.id) > 0
    ORDER BY
      order_count DESC
    """
)
    return


@app.cell
def _(mo):
    dept = "engineering"
    _df = mo.sql(
    f"""
    SELECT
      id,
      name,
      salary,
      ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS `rank`
    FROM employees
    WHERE
      department = {dept}
    """
)
    return


if __name__ == "__main__":
    app.run()
