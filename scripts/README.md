# Annotation Evaluation Scripts

This directory contains scripts for downloading real COCO data and generating realistic predictions for testing the evaluation pipeline.

## Scripts

### 1. `download_coco_subset.py`

Downloads a subset of COCO val2017 images and their ground truth annotations.

**Usage:**
```bash
uv run python scripts/download_coco_subset.py
```

**What it does:**
- Downloads COCO val2017 annotations (~250MB)
- Selects 8 interesting images with diverse objects (cars, people, buses, trucks, traffic lights)
- Downloads only those specific images
- Creates ground truth annotations in COCO format
- Generates documentation with image links

**Output:**
- `data/ground_truth/test/gt_annotations.json` - Ground truth annotations
- `data/ground_truth/test/images/` - Downloaded images
- `data/ground_truth/test/DATASET_INFO.md` - Documentation with image details and links

### 2. `generate_predictions.py`

Generates realistic predictions from ground truth annotations by adding controlled noise.

**Basic Usage:**
```bash
uv run python scripts/generate_predictions.py
```

**Advanced Usage with Custom Parameters:**
```bash
# Generate predictions with different noise levels
uv run python scripts/generate_predictions.py \
  --position-noise 10.0 \
  --size-noise 0.15 \
  --detection-rate 0.85 \
  --misclassification-rate 0.1 \
  --false-positive-rate 0.2

# Use different input/output paths
uv run python scripts/generate_predictions.py \
  --gt-path data/ground_truth/test/gt_annotations.json \
  --output data/predictions/test/predictions.json
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gt-path` | `data/ground_truth/test/gt_annotations.json` | Path to ground truth COCO JSON |
| `--output` | `data/predictions/test/predictions.json` | Path to save predictions JSON |
| `--position-noise` | `5.0` | Standard deviation for position noise (pixels) |
| `--size-noise` | `0.1` | Standard deviation for size noise (fraction, 0.1 = 10%) |
| `--detection-rate` | `0.9` | Probability of detecting each object (0.9 = 90% recall) |
| `--misclassification-rate` | `0.05` | Probability of wrong category (5% misclassified) |
| `--false-positive-rate` | `0.1` | Expected false positives per image |
| `--seed` | `42` | Random seed for reproducibility |

**What it does:**
- Adds position noise to bounding boxes (simulates localization errors)
- Adds size noise to bounding boxes (simulates scale estimation errors)
- Randomly misses some detections (simulates false negatives)
- Randomly misclassifies some objects (simulates category confusion)
- Adds false positive detections (simulates spurious detections)
- Assigns realistic confidence scores

**Example Output:**
```
Generated 97 predictions
  Detected: 95 / 107 ground truth objects
  Missed (FN): 12
  Misclassified: 4
  False Positives: 2
```

## Typical Workflow

1. **Download ground truth data:**
   ```bash
   uv run python scripts/download_coco_subset.py
   ```

2. **Check the downloaded images:**
   - Open `data/ground_truth/test/DATASET_INFO.md`
   - Click on the links to view images on COCO website

3. **Generate predictions:**
   ```bash
   uv run python scripts/generate_predictions.py
   ```

4. **Run evaluation:**
   ```bash
   uv run python src/eval.py
   ```

## Experiment with Different Scenarios

### High-Quality Detector (Good localization, high recall)
```bash
uv run python scripts/generate_predictions.py \
  --position-noise 2.0 \
  --size-noise 0.05 \
  --detection-rate 0.95 \
  --misclassification-rate 0.02 \
  --false-positive-rate 0.05
```
Expected AP@0.5: ~70-80%

### Low-Quality Detector (Poor localization, low recall)
```bash
uv run python scripts/generate_predictions.py \
  --position-noise 20.0 \
  --size-noise 0.3 \
  --detection-rate 0.7 \
  --misclassification-rate 0.15 \
  --false-positive-rate 0.5
```
Expected AP@0.5: ~20-30%

### Category Confusion Test
```bash
uv run python scripts/generate_predictions.py \
  --position-noise 3.0 \
  --size-noise 0.08 \
  --detection-rate 0.95 \
  --misclassification-rate 0.3 \
  --false-positive-rate 0.05
```
Tests how well your evaluation handles misclassifications

## Understanding the Results

After running `eval.py`, you'll see metrics like:

- **AP@0.5:0.95**: Average Precision across IoU thresholds 0.5 to 0.95 (most important metric)
- **AP@0.5**: Average Precision at IoU threshold 0.5 (easier, focuses on detection)
- **AP@0.75**: Average Precision at IoU threshold 0.75 (harder, focuses on localization)
- **AR**: Average Recall (how many ground truth objects were detected)

With default parameters (moderate noise), expect:
- AP@0.5:0.95: ~15-25%
- AP@0.5: ~45-55%
- AR: ~20-30%

## Data Attribution

The ground truth images are from the [COCO Dataset](https://cocodataset.org/):
- **Citation**: Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV 2014.
- **License**: [COCO License](https://cocodataset.org/#termsofuse)

