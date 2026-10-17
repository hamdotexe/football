"""Convert a percent-format .py file into a Jupyter .ipynb notebook.

Cells are separated by lines starting with "# %%". A marker of "# %% [markdown]"
makes a markdown cell; its lines have the leading "# " stripped.

Usage: python scripts/build_notebook.py notebooks/01_baseline/baseline.py
       (writes baseline.ipynb next to it)
"""
import json
import sys
from pathlib import Path


def parse_cells(text):
    cells, kind, lines = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
        elif kind is not None:
            lines.append(line)
    flush()
    return cells


def to_notebook(cells):
    def source(body):
        parts = body.splitlines(keepends=True)
        return parts

    nb_cells = []
    for kind, body in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": source(body)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(to_notebook(parse_cells(src.read_text(encoding="utf-8"))), indent=1), encoding="utf-8")
    print(f"wrote {out}")
