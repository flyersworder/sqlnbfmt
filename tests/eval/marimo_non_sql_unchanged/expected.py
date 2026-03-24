import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    x = 42
    y = x ** 2
    print(f"Result: {y}")
    return (x, y)


@app.cell
def _():
    import os
    path = os.path.join("/tmp", "data.csv")
    return (path,)


@app.cell
def _():
    def fibonacci(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    return (fibonacci,)


if __name__ == "__main__":
    app.run()
