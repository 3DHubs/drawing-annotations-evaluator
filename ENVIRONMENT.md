# Environment Setup

This project evaluates detection annotations using the COCO tooling. The environment includes
`pycocotools` for evaluation metrics, along with supporting packages for numeric processing,
CLI helpers, and columnar data handling.

## Key Dependencies

- `pycocotools` – evaluation logic and COCO metric helpers
- `polars` – preferred columnar library for annotation data manipulation
- `numpy` – numerical processing required by pycocotools
- `pillow` – image loading and drawing for visualizations
- `rich` – enhanced console output and formatting
- `tqdm` – progress reporting
- `cython` – build dependency for pycocotools compilation
- `tabulate` – for markdown table generation

All dependencies are declared in `pyproject.toml` and managed through `uv`.

## Setup Instructions

### 1. Install uv

Install the `uv` package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ensure the `uv` binary is on your `PATH`.

### 2. Create virtual environment

This project requires Python 3.11 (`requires-python = ">=3.11,<3.12"`):

```bash
uv venv --python 3.11
source .venv/bin/activate
```

Use `--clear` flag if you need to recreate an existing `.venv`.

### 3. Install dependencies

Lock and install dependencies:

```bash
uv lock
uv sync
```

- `uv lock` resolves `pyproject.toml` into `uv.lock`
- `uv sync` installs the locked dependencies into `.venv`

Re-run both commands whenever you update `pyproject.toml`.

### 4. Verify installation

Test the environment:

```bash
python -c "from pycocotools.coco import COCO; import polars as pl; print('Environment ready')"
```

If this runs without errors, you're ready to use the evaluation tools.
