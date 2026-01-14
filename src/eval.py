from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import polars as pl
from pathlib import Path
from tabulate import tabulate


# Order matches how `coco_eval.stats` is filled inside
# `pycocotools/cocoeval.py`'s `summarize()` helper.
METRIC_NAMES = [
    "ap_50_95",
    "ap_50",
    "ap_75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_maxdets1",
    "ar_maxdets10",
    "ar_maxdets100",
    "ar_small",
    "ar_medium",
    "ar_large",
]


class CocoEvalResult:
    """Wrapper around pycocotools evaluation results.

    Computes and stores global metrics and per-category metrics for object detection
    evaluation using COCO.

    Attributes:
        gt_coco: COCO object with ground truth annotations
        dt_coco: COCO object with detection predictions
        iou_type: Type of IoU to compute ("bbox", "segm", or "keypoints")
    """

    def __init__(self, gt_coco: COCO, dt_coco: COCO, iou_type: str = "bbox"):
        self.gt_coco = gt_coco
        self.dt_coco = dt_coco
        self.iou_type = iou_type

        self.coco_eval = COCOeval(gt_coco, dt_coco, iou_type)
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

        self._stats = self._build_stats(self.coco_eval)
        self._per_label_stats = self._build_per_label_stats(self.coco_eval)

    def _build_stats(self, coco_eval: COCOeval) -> dict[str, float]:
        """Extract stats from COCOeval object into named dict.

        Args:
            coco_eval: COCOeval object with computed stats

        Returns:
            Dict mapping metric names to values (e.g., {"ap_50_95": 0.45, ...})
        """
        stats = {}
        values = coco_eval.stats
        if values is None:
            return stats
        for name, value in zip(METRIC_NAMES, values):
            stats[name] = float(value)
        return stats

    def _build_per_label_stats(
        self, coco_eval: COCOeval
    ) -> dict[int, dict[str, float]]:
        """Compute metrics for each category separately.

        Returns:
            Dict mapping category IDs to their stats dicts
            (e.g., {1: {"ap_50_95": 0.45, ...}, 2: {...}, ...})
        """
        if coco_eval.eval.get("precision") is None:
            return {}

        label_stats: dict[int, dict[str, float]] = {}

        for cat_id in self.coco_eval.params.catIds:
            # Simply changing the catIds in the params of coco_eval is not enough, we need to create a new COCOeval object per label
            cat_eval = COCOeval(self.gt_coco, self.dt_coco, self.iou_type)
            cat_eval.params.catIds = [cat_id]
            cat_eval.evaluate()
            cat_eval.accumulate()
            cat_eval.summarize()
            label_stats[cat_id] = self._build_stats(cat_eval)

        return label_stats

    @property
    def stats(self) -> dict[str, float]:
        """Global evaluation metrics across all categories."""
        return self._stats

    @property
    def per_label_stats(self) -> dict[int, dict[str, float]]:
        """Per-category evaluation metrics, keyed by category ID."""
        return self._per_label_stats

    def metric(self, name: str) -> float | None:
        return self._stats.get(name)

    def to_dict(self) -> dict[str, float]:
        """Convert to flat dict with prefixed columns for DataFrame use.

        Returns:
            Dict with keys like "global_ap50_95", "car_ap50", "person_ar_maxdets100"
            Metric names are cleaned (ap_50 -> ap50, ap_50_95 -> ap50_95)
        """
        # Extract label map from COCO dataset
        label_map = {
            cat["id"]: cat["name"] for cat in self.gt_coco.dataset["categories"]
        }

        row = {}

        # Add global stats with "global_" prefix
        for metric_name, value in self.stats.items():
            # Convert ap_50_95 -> ap50_95, ap_50 -> ap50, ap_75 -> ap75
            clean_name = metric_name.replace("_5", "5").replace("_7", "7")
            row[f"global_{clean_name}"] = value

        # Add per-label stats with "labelname_" prefix
        for label_id, label_stats in self.per_label_stats.items():
            label_name = label_map.get(label_id, f"label{label_id}")
            for metric_name, value in label_stats.items():
                clean_name = metric_name.replace("_5", "5").replace("_7", "7")
                row[f"{label_name}_{clean_name}"] = value

        return row


def eval_results_to_df(
    results: list[CocoEvalResult],
    metadata: list[dict] | None = None,
) -> pl.DataFrame:
    """Convert multiple CocoEvalResults to Polars DataFrame.

    Args:
        results: List of CocoEvalResult objects
        metadata: Optional list of dicts with metadata for each result
                  (e.g., [{"experiment_id": "exp1", "model": "good"}, ...])

    Returns:
        DataFrame where each row is one result, columns are prefixed
        (global_ap50_95, labelname_ap50, etc.)
    """
    rows = []
    for i, result in enumerate(results):
        row = {}
        if metadata and i < len(metadata):
            row.update(metadata[i])
        row.update(result.to_dict())
        rows.append(row)

    return pl.DataFrame(rows)


def save_results_to_markdown(
    df: pl.DataFrame,
    directory: str,
    name: str,
    metadata_cols: list[str] | None = None,
    prefix_cols: list[str] | None = None,
    max_cols: int | None = None,
) -> None:
    """Save evaluation results DataFrame to markdown table.

    Args:
        df: DataFrame with evaluation results
        directory: Directory to save markdown file
        name: Name of markdown file (without .md extension)
        metadata_cols: List of metadata column names to include (e.g., ["pred_name"])
                       If None, no metadata columns included
        prefix_cols: List of metric prefixes to include (e.g., ["global", "car"])
                     If None, all columns that are not metadata columns are included
        max_cols: Maximum number of metric columns to include after prefix filtering
                  If None, includes all matching columns
    """
    # Select metadata columns
    meta_cols = []
    if metadata_cols is not None:
        meta_cols = [col for col in metadata_cols if col in df.columns]

    # Select metric columns by prefix
    metric_cols = []
    if prefix_cols is not None:
        for prefix in prefix_cols:
            metric_cols.extend(
                [col for col in df.columns if col.startswith(f"{prefix}_")]
            )
    else:
        metric_cols = [x for x in df.columns if x not in meta_cols]

    # Limit to max_cols if provided
    if max_cols is not None and len(metric_cols) > max_cols:
        metric_cols = metric_cols[:max_cols]

    # Select columns and create table
    selected_cols = meta_cols + metric_cols
    table_df = df.select(selected_cols)

    # Write markdown file
    output_path = Path(directory) / f"{name}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to markdown using tabulate
    markdown_content = tabulate(
        table_df.to_dicts(),
        headers="keys",
        tablefmt="github",
        floatfmt=".3f",
    )
    output_path.write_text(markdown_content)


def load_predictions(pred_path: str) -> list[dict]:
    """Load predictions from JSON file, handling both array and object formats.

    Args:
        pred_path: Path to predictions JSON file

    Returns:
        List of prediction dictionaries
    """
    import json

    with open(pred_path, "r") as f:
        data = json.load(f)

    # If data is a dict with "annotations" key, extract it
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    # If data is already a list, return as-is
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(
            f"Predictions file must be either a list or a dict with 'annotations' key. "
            f"Got {type(data).__name__}"
        )


def main():
    """Run evaluation on ground truth and predictions from command line."""
    import argparse
    from rich import print as rprint

    parser = argparse.ArgumentParser(
        description="Evaluate object detection predictions using COCO metrics"
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        required=True,
        help="Path to ground truth COCO JSON file",
    )
    parser.add_argument(
        "--pred-path",
        type=str,
        required=True,
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        required=True,
        help="Path to save markdown report (e.g., reports/test/eval_results)",
    )
    parser.add_argument(
        "--report-prefixes",
        type=str,
        nargs="+",
        default=["global"],
        help="Metric prefixes to include in report (e.g., global car person)",
    )

    args = parser.parse_args()

    # Load ground truth and predictions
    rprint(f"[cyan]Loading ground truth from:[/cyan] {args.gt_path}")
    gt_coco = COCO(args.gt_path)

    rprint(f"[cyan]Loading predictions from:[/cyan] {args.pred_path}")
    predictions = load_predictions(args.pred_path)
    pred_coco = gt_coco.loadRes(predictions)

    # Evaluate
    rprint("[cyan]Running evaluation...[/cyan]")
    result = CocoEvalResult(gt_coco, pred_coco, iou_type="bbox")

    # Print results
    rprint("\n[bold green]Evaluation Results:[/bold green]")
    rprint("\n[bold]Global Metrics:[/bold]")
    for metric_name, value in result.stats.items():
        rprint(f"  {metric_name:20s}: {value:.4f}")

    # Save report
    pred_name = Path(args.pred_path).stem
    df = eval_results_to_df([result], metadata=[{"pred_name": pred_name}])

    # Parse report path
    report_path = Path(args.report_path)
    directory = report_path.parent
    name = report_path.stem

    save_results_to_markdown(
        df,
        str(directory),
        name,
        metadata_cols=["pred_name"],
        prefix_cols=args.report_prefixes,
    )
    rprint(f"\n[green]Report saved to:[/green] {directory}/{name}.md")


if __name__ == "__main__":
    main()
