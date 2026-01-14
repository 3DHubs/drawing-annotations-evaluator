# Drawing Annotations Evaluator

Evaluation toolkit for object detection annotations using COCO metrics and tooling. This repository provides utilities to evaluate detection quality, generate test data, and visualize results.

## What This Repo Does

- **Evaluates detection annotations** using standardized COCO metrics (AP, AR at multiple IoU thresholds)
- **Generates realistic test predictions** from ground truth data with configurable noise
- **Visualizes results** by overlaying predictions on images with color-coded boxes
- **Computes per-category metrics** to identify which object classes perform well or poorly
- **Produces markdown reports** summarizing evaluation results

## Quick Start

```bash
# 1. Setup environment (see ENVIRONMENT.md for details)
uv venv --python 3.11
source .venv/bin/activate
uv sync

# 2. Download test images and ground truth (see `scripts/README.md` for details)
uv run python scripts/download_coco_subset.py

# 3. Generate predictions with default noise (see `scripts/README.md` for details)
uv run python scripts/generate_predictions.py

# 4. Run evaluation (outputs COCO metrics and saves markdown report)
uv run python src/eval.py \
  --gt-path data/ground_truth/test/gt_annotations.json \
  --pred-path data/predictions/test/predictions.json \
  --report-path reports/test/eval_results

# 5. Visualize results (saves annotated images)
uv run python src/visualizer.py \
  --gt-path data/ground_truth/test/gt_annotations.json \
  --pred-path data/predictions/test/predictions.json \
  --images-dir data/ground_truth/test/images \
  --output-dir data/visualizations/test/predictions
```


## Data Attribution

Test images are from the [COCO Dataset](https://cocodataset.org/):
- **Citation**: Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV 2014.
- **License**: [COCO License](https://cocodataset.org/#termsofuse)


## TODOs

- How to map labels from different services into a common format?
- Add support for attribute evaluation

