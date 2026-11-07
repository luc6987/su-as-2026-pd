"""Build the submission notebook from its source, and optionally run it.

    python tools/notebook/build.py notebook/pd_challenge.py notebook/pd_challenge.ipynb
    python tools/notebook/build.py notebook/pd_challenge.py notebook/pd_challenge.ipynb --execute

The source is a plain ``.py`` file in the jupytext *percent* format: ``# %%`` starts a code
cell, ``# %% [markdown]`` a markdown cell whose lines lose one leading ``#``. The output is
nbformat 4.5 JSON, written deterministically — the same source built twice is byte-identical,
so the notebook can be committed and diffed.

``--execute`` runs every code cell **in this process**, in one shared namespace, and records
stdout, stderr, the value of a trailing expression and any matplotlib figures the cell left
open. It exists because this machine has neither the notebook stack (nbformat, nbclient,
ipykernel, IPython) nor local sockets, so a Jupyter kernel cannot run — and because a
notebook that only uses plain Python cannot tell the difference. The notebook is written to
avoid anything Jupyter-specific: no magics, no ``display``, no ``get_ipython``.

Execution is deterministic in the sense the notebook needs: no cell draws from a global RNG
state, and the recorded outputs are whatever the cells print. A failing cell stops the run,
records an ``error`` output, writes what has been executed so far and exits 1.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import re
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

CELL_MARKER = re.compile(r"^#\s*%%(\s*\[markdown\])?\s*$")

NB_METADATA: dict[str, Any] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
}


class BuildError(Exception):
    """The source file is not in the percent format this tool understands."""


# --- parsing ------------------------------------------------------------------------
def _strip_markdown_prefix(line: str) -> str:
    """Drop one leading ``# `` or ``#`` from a markdown line, leaving anything else alone."""
    if line.startswith("# "):
        return line[2:]
    if line.startswith("#"):
        return line[1:]
    return line


def parse_source(text: str) -> list[dict[str, Any]]:
    """Percent-format source -> ``[{"cell_type": ..., "lines": [...]}]``.

    Everything before the first marker must be blank or a comment; anything else is a
    mistake that would otherwise be dropped silently (the file would build, minus its first
    paragraph). Trailing blank lines of a cell are trimmed so they do not show up as
    whitespace at the end of the rendered cell.
    """
    cells: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        marker = CELL_MARKER.match(line)
        if marker:
            if current is not None:
                cells.append(current)
            current = {"cell_type": "markdown" if marker.group(1) else "code", "lines": []}
            continue
        if current is None:
            if line.strip() and not line.lstrip().startswith("#"):
                raise BuildError(
                    f"line {number}: content before the first '# %%' marker: {line.strip()!r}"
                )
            continue
        current["lines"].append(
            _strip_markdown_prefix(line) if current["cell_type"] == "markdown" else line
        )

    if current is not None:
        cells.append(current)
    for cell in cells:
        while cell["lines"] and not cell["lines"][-1].strip():
            cell["lines"].pop()
    return cells


# --- building -----------------------------------------------------------------------
def build_notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Cells -> an nbformat 4.5 notebook. Cell ids are positional, so they never move."""
    out: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        lines = cell["lines"]
        source = [line + "\n" for line in lines[:-1]]
        if lines:
            source.append(lines[-1])
        cell_id = f"cell-{index:03d}"
        if cell["cell_type"] == "markdown":
            out.append({"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source})
        else:
            out.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": cell_id,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )
    return {"cells": out, "metadata": NB_METADATA, "nbformat": 4, "nbformat_minor": 5}


def dump_notebook(notebook: dict[str, Any]) -> str:
    """The exact bytes written to ``.ipynb`` — one function, so builds compare equal."""
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


# --- execution ----------------------------------------------------------------------
def run_cell(source: str, namespace: dict[str, Any]) -> Any:
    """Execute one cell in ``namespace``; return the value of a trailing expression.

    A trailing expression is evaluated separately rather than executed as a statement, so
    the notebook gets the same ``execute_result`` a Jupyter kernel would record for the last
    line of a cell. ``None`` means there was nothing to record.
    """
    tree = ast.parse(source, filename="<cell>", mode="exec")
    body = list(tree.body)
    if body and isinstance(body[-1], ast.Expr):
        tail = body.pop()
        exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"), namespace)
        expression = ast.Expression(body=tail.value)
        ast.fix_missing_locations(expression)
        return eval(compile(expression, "<cell>", "eval"), namespace)
    exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"), namespace)
    return None


def _truncate(text: str, limit: int) -> str:
    """Cap one stream at ``limit`` characters, saying how much was dropped.

    A step that prints per column can produce megabytes on the real data; the notebook has
    to stay small enough to submit.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[output truncated: {len(text) - limit} characters omitted]"


def _stream_output(text: str, name: str, limit: int) -> dict[str, Any]:
    return {"name": name, "output_type": "stream", "text": _truncate(text, limit)}


def _execute_result(value: Any, execution_count: int) -> dict[str, Any]:
    """A repr, plus HTML when the object offers it (a DataFrame, for one)."""
    data: dict[str, Any] = {"text/plain": repr(value)}
    render = getattr(value, "_repr_html_", None)
    if callable(render):
        try:
            html = render()
        except Exception:  # noqa: BLE001 - a broken repr must not fail the cell
            html = None
        if html is not None:
            data["text/html"] = html
    return {
        "data": data,
        "execution_count": execution_count,
        "metadata": {},
        "output_type": "execute_result",
    }


def _figure_outputs() -> list[dict[str, Any]]:
    """Every open matplotlib figure as a PNG, then close them.

    Only if the notebook has already imported pyplot — this tool never imports matplotlib
    itself, so a cell that draws nothing costs nothing.
    """
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is None:
        return []
    outputs: list[dict[str, Any]] = []
    for number in pyplot.get_fignums():
        figure = pyplot.figure(number)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        width, height = figure.get_size_inches()
        outputs.append(
            {
                "data": {
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "text/plain": f"<Figure size {width:g}x{height:g} inches>",
                },
                "metadata": {},
                "output_type": "display_data",
            }
        )
    pyplot.close("all")
    return outputs


def execute_cells(
    notebook: dict[str, Any], out_path: Path, max_stream_chars: int
) -> tuple[tuple[int, str] | None, float]:
    """Run the notebook's code cells in order.

    Returns ``((failing cell index, exception name) or None, seconds)``. The caller writes
    the notebook either way: a partial run is evidence, not a failed build.
    """
    # Agg before any cell imports pyplot, and the same working directory Jupyter would use;
    # the notebook resolves its own paths from the repository root, not from the cwd.
    os.environ.setdefault("MPLBACKEND", "Agg")
    out_path = Path(out_path).resolve()
    os.chdir(out_path.parent)

    namespace: dict[str, Any] = {"__name__": "__main__"}
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    started = time.perf_counter()
    failure: tuple[int, str] | None = None

    for index, cell in enumerate(code_cells, start=1):
        cell_started = time.perf_counter()
        cell["execution_count"] = index
        stdout, stderr = io.StringIO(), io.StringIO()
        value: Any = None
        error: BaseException | None = None
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                value = run_cell("".join(cell["source"]), namespace)
        except BaseException as exc:  # noqa: BLE001 - every failure is recorded, not raised
            error = exc

        outputs: list[dict[str, Any]] = []
        if stdout.getvalue():
            outputs.append(_stream_output(stdout.getvalue(), "stdout", max_stream_chars))
        if stderr.getvalue():
            outputs.append(_stream_output(stderr.getvalue(), "stderr", max_stream_chars))
        if error is None and value is not None:
            outputs.append(_execute_result(value, index))
        outputs.extend(_figure_outputs())
        cell["outputs"] = outputs

        print(
            f"cell {index}/{len(code_cells)} done in {time.perf_counter() - cell_started:.1f} s",
            flush=True,
        )
        if error is not None:
            cell["outputs"].append(
                {
                    "ename": type(error).__name__,
                    "evalue": str(error),
                    "output_type": "error",
                    "traceback": traceback.format_exception(type(error), error, error.__traceback__),
                }
            )
            failure = (index, type(error).__name__)
            break

    return failure, time.perf_counter() - started


# --- entry point --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, help="percent-format .py source")
    parser.add_argument("output", type=Path, help="where to write the .ipynb")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run every code cell in this process and record its outputs",
    )
    parser.add_argument(
        "--max-stream-chars",
        type=int,
        default=20000,
        help="truncate each cell's stdout/stderr to this many characters (default: 20000)",
    )
    args = parser.parse_args(argv)

    cells = parse_source(args.source.read_text(encoding="utf-8"))
    notebook = build_notebook(cells)
    # Resolved before anything runs: execution changes the working directory to the notebook's
    # own, so a relative output path would not mean the same thing afterwards.
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.execute:
        args.output.write_text(dump_notebook(notebook), encoding="utf-8")
        print(f"built {args.output} · {len(notebook['cells'])} cells")
        return 0

    failure, seconds = execute_cells(notebook, args.output, args.max_stream_chars)
    args.output.write_text(dump_notebook(notebook), encoding="utf-8")
    n_code = sum(cell["cell_type"] == "code" for cell in notebook["cells"])
    if failure is None:
        print(f"NOTEBOOK DONE {args.output} cells={n_code} seconds={seconds:.1f}")
        return 0
    print(f"NOTEBOOK FAILED at cell {failure[0]}: {failure[1]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
