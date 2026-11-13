"""Build kaggle/notebook/qwen3-5.ipynb: cells 0-2 (install, load Qwen, generate_report) + our launch cell.

Usage: python kaggle/build_notebook.py [SOURCE]
  SOURCE = a kernel JSON pulled from the Kaggle API, or an .ipynb whose first 3 cells are used.
           Default: the committed kaggle/notebook/qwen3-5.ipynb (so only the launch cell changes).
Then:  kaggle kernels push -p kaggle/notebook
"""
import json
import pathlib
import re
import sys

# a hard-coded password assignment must never be published (the notebook is public)
PASSWORD_LITERAL = re.compile(r"""API_PASSWORD\s*=\s*["']""")

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "notebook"

source = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "qwen3-5.ipynb"
loaded = json.loads(source.read_text(encoding="utf-8"))
notebook = json.loads(loaded["blob"]["source"]) if "blob" in loaded else loaded
cells = notebook["cells"][:3]
for cell in cells:
    cell["outputs"] = []  # never publish old outputs
    cell["execution_count"] = None
    if PASSWORD_LITERAL.search("".join(cell["source"])):
        raise SystemExit("refusing to build: a password literal is in cells 0-2")

launch = (HERE / "qwen_service_launch_cell.py").read_text(encoding="utf-8")
cells.append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": launch.splitlines(keepends=True),
})
notebook["cells"] = cells

OUT.mkdir(exist_ok=True)
(OUT / "qwen3-5.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "rudraksh12/qwen3-5",
    "title": "qwen3.5",
    "code_file": "qwen3-5.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": False,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}, indent=2), encoding="utf-8")
print("built", OUT / "qwen3-5.ipynb", "with", len(cells), "cells")
