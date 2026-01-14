"""
Generate realistic predictions from ground truth COCO annotations.

This script creates predictions with controlled noise to simulate a real object detector:
- Bounding boxes are slightly offset in position
- Bounding boxes are slightly larger or smaller
- Confidence scores vary realistically
- Some ground truth objects are missed (false negatives)
- Some extra detections are added (false positives)
- Occasional category misclassifications
"""

import json
import random
import argparse
from pathlib import Path
from typing import Any
import numpy as np


def add_bbox_noise(
    bbox: list[float],
    position_noise_std: float = 5.0,
    size_noise_std: float = 0.1,
) -> list[float]:
    """
    Add noise to a bounding box [x, y, width, height].

    Args:
        bbox: Original bbox [x, y, width, height]
        position_noise_std: Standard deviation for position offset in pixels
        size_noise_std: Standard deviation for size change as fraction of original size

    Returns:
        Noisy bbox [x, y, width, height]
    """
    x, y, w, h = bbox

    # Add position noise
    x_noise = random.gauss(0, position_noise_std)
    y_noise = random.gauss(0, position_noise_std)

    # Add size noise (as a multiplier)
    w_multiplier = 1.0 + random.gauss(0, size_noise_std)
    h_multiplier = 1.0 + random.gauss(0, size_noise_std)

    # Ensure positive dimensions
    new_w = max(1.0, w * w_multiplier)
    new_h = max(1.0, h * h_multiplier)

    # Ensure non-negative position
    new_x = max(0.0, x + x_noise)
    new_y = max(0.0, y + y_noise)

    return [new_x, new_y, new_w, new_h]


def generate_confidence_score(
    iou_target: float = 0.85,
    noise: float = 0.15,
) -> float:
    """
    Generate a realistic confidence score.

    Higher IoU detections should generally have higher confidence,
    but with some noise.

    Args:
        iou_target: Target IoU (affects base confidence)
        noise: Noise level for confidence

    Returns:
        Confidence score between 0 and 1
    """
    # Base confidence correlates with IoU
    base_conf = 0.3 + (iou_target * 0.6)

    # Add noise
    conf = base_conf + random.gauss(0, noise)

    # Clip to valid range
    return max(0.05, min(0.99, conf))


def should_detect(detection_rate: float = 0.9) -> bool:
    """
    Decide whether to detect an object (simulate false negatives).

    Args:
        detection_rate: Probability of detecting an object

    Returns:
        True if object should be detected
    """
    return random.random() < detection_rate


def should_misclassify(misclassification_rate: float = 0.05) -> bool:
    """
    Decide whether to misclassify an object.

    Args:
        misclassification_rate: Probability of misclassifying

    Returns:
        True if object should be misclassified
    """
    return random.random() < misclassification_rate


def get_similar_category(
    category_id: int,
    all_categories: list[dict],
) -> int:
    """
    Get a similar category for misclassification.

    For simplicity, just pick a random different category.
    In practice, you might want semantic similarity (e.g., car <-> truck).
    """
    other_cats = [cat["id"] for cat in all_categories if cat["id"] != category_id]
    if other_cats:
        return random.choice(other_cats)
    return category_id


def generate_false_positive(
    image_info: dict,
    categories: list[dict],
) -> dict:
    """
    Generate a false positive detection.

    Args:
        image_info: Image metadata
        categories: List of category definitions

    Returns:
        False positive detection
    """
    width = image_info["width"]
    height = image_info["height"]

    # Random small box
    box_w = random.uniform(20, 100)
    box_h = random.uniform(20, 100)
    box_x = random.uniform(0, width - box_w)
    box_y = random.uniform(0, height - box_h)

    return {
        "image_id": image_info["id"],
        "category_id": random.choice(categories)["id"],
        "bbox": [box_x, box_y, box_w, box_h],
        "score": random.uniform(0.1, 0.4),  # Low confidence for FP
    }


def generate_predictions(
    gt_path: Path,
    output_path: Path,
    position_noise: float = 5.0,
    size_noise: float = 0.1,
    detection_rate: float = 0.9,
    misclassification_rate: float = 0.05,
    false_positive_rate: float = 0.1,
    seed: int | None = 42,
) -> None:
    """
    Generate predictions from ground truth with controlled noise.

    Args:
        gt_path: Path to ground truth COCO JSON
        output_path: Path to save predictions JSON
        position_noise: Standard deviation for position noise (pixels)
        size_noise: Standard deviation for size noise (fraction)
        detection_rate: Probability of detecting each object
        misclassification_rate: Probability of misclassifying category
        false_positive_rate: Expected number of FP per image
        seed: Random seed for reproducibility
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Load ground truth
    print(f"Loading ground truth from: {gt_path}")
    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    images = gt_data["images"]
    annotations = gt_data["annotations"]
    categories = gt_data["categories"]

    print(f"Ground truth: {len(images)} images, {len(annotations)} annotations")

    # Generate predictions
    predictions = []

    # Process each annotation
    missed_count = 0
    misclassified_count = 0

    for ann in annotations:
        # Decide whether to detect this object
        if not should_detect(detection_rate):
            missed_count += 1
            continue

        # Add bbox noise
        noisy_bbox = add_bbox_noise(
            ann["bbox"],
            position_noise_std=position_noise,
            size_noise_std=size_noise,
        )

        # Decide on category
        cat_id = ann["category_id"]
        if should_misclassify(misclassification_rate):
            cat_id = get_similar_category(cat_id, categories)
            misclassified_count += 1

        # Generate confidence score
        # Higher noise -> lower typical IoU -> lower confidence
        avg_iou = max(0.6, 0.95 - (position_noise / 10) - size_noise)
        confidence = generate_confidence_score(iou_target=avg_iou)

        pred = {
            "image_id": ann["image_id"],
            "category_id": cat_id,
            "bbox": noisy_bbox,
            "score": confidence,
        }

        # Include attributes if present in ground truth
        if "attributes" in ann:
            pred["attributes"] = ann["attributes"].copy()

        predictions.append(pred)

    # Add false positives
    fp_count = 0
    for img in images:
        num_fp = np.random.poisson(false_positive_rate)
        for _ in range(num_fp):
            fp = generate_false_positive(img, categories)
            predictions.append(fp)
            fp_count += 1

    # Save predictions
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # Print statistics
    print(f"\nGenerated {len(predictions)} predictions")
    print(
        f"  Detected: {len(predictions) - fp_count} / {len(annotations)} ground truth objects"
    )
    print(f"  Missed (FN): {missed_count}")
    print(f"  Misclassified: {misclassified_count}")
    print(f"  False Positives: {fp_count}")
    print(f"\nPredictions saved to: {output_path}")

    # Show example predictions
    print("\nExample predictions (first 3):")
    for i, pred in enumerate(predictions[:3], 1):
        bbox = [f"{x:.1f}" for x in pred["bbox"]]
        print(
            f"  {i}. Image {pred['image_id']}, Cat {pred['category_id']}, "
            f"BBox [{', '.join(bbox)}], Score {pred['score']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions from ground truth with controlled noise"
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=Path("data/ground_truth/test/gt_annotations.json"),
        help="Path to ground truth COCO JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/predictions/test/predictions.json"),
        help="Path to save predictions JSON",
    )
    parser.add_argument(
        "--position-noise",
        type=float,
        default=5.0,
        help="Standard deviation for position noise in pixels (default: 5.0)",
    )
    parser.add_argument(
        "--size-noise",
        type=float,
        default=0.1,
        help="Standard deviation for size noise as fraction (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--detection-rate",
        type=float,
        default=0.9,
        help="Probability of detecting each object (default: 0.9)",
    )
    parser.add_argument(
        "--misclassification-rate",
        type=float,
        default=0.05,
        help="Probability of misclassifying category (default: 0.05)",
    )
    parser.add_argument(
        "--false-positive-rate",
        type=float,
        default=0.1,
        help="Expected number of false positives per image (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    generate_predictions(
        gt_path=args.gt_path,
        output_path=args.output,
        position_noise=args.position_noise,
        size_noise=args.size_noise,
        detection_rate=args.detection_rate,
        misclassification_rate=args.misclassification_rate,
        false_positive_rate=args.false_positive_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
