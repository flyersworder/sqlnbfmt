import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return (mo, pd)


@app.cell
def _(pd):
    # Pure Python cell — should not be touched
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    total = df["a"].sum()
    return (df, total)


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


@app.cell
def _():
    # Another non-SQL cell
    x = 42
    y = x ** 2
    print(f"Result: {y}")
    return (x, y)


@app.cell
def _(mo):
    result = mo.sql(
    f"""
    SELECT
      department,
      COUNT(*) AS cnt
    FROM employees
    GROUP BY
      department
    HAVING
      COUNT(*) > 5
    """
)
    return (result,)


if __name__ == "__main__":
    app.run()
