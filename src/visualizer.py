from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools.coco import COCO


class BBoxVisualizer:
    """Overlay ground truth and prediction bounding boxes on images."""

    def __init__(
        self,
        gt_coco: COCO,
        pred_coco: COCO,
        images_dir: Path | str,
        output_dir: Path | str,
        gt_color: tuple[int, int, int] = (0, 255, 0),  # Green for all GT
        pred_colors: (
            dict[int, tuple[int, int, int]] | None
        ) = None,  # {category_id: (R, G, B)}
        line_width: int = 3,
        font_size: int = 16,
    ):
        """
        Args:
            gt_coco: COCO object with ground truth annotations
            pred_coco: COCO object with predictions (use gt_coco.loadRes())
            images_dir: Directory containing images
            output_dir: Directory to save visualizations
            gt_color: RGB color for all ground truth boxes
            pred_colors: Dict mapping category_id to RGB color for predictions
                        If None, generates distinct colors automatically
            line_width: Width of bounding box lines
            font_size: Font size for labels
        """
        self.gt_coco = gt_coco
        self.pred_coco = pred_coco
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gt_color = gt_color
        self.line_width = line_width
        self.font_size = font_size

        # Generate or use provided prediction colors per category
        if pred_colors is None:
            self.pred_colors = self._generate_category_colors()
        else:
            self.pred_colors = pred_colors

    def _generate_category_colors(self) -> dict[int, tuple[int, int, int]]:
        """Generate distinct colors for each category."""
        import colorsys

        category_ids = self.gt_coco.getCatIds()
        colors = {}

        for i, cat_id in enumerate(category_ids):
            # Use HSV color space for distinct hues
            hue = i / len(category_ids)
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[cat_id] = tuple(int(c * 255) for c in rgb)

        return colors

    def _draw_bbox(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: list[float],
        color: tuple[int, int, int],
        label: str | None = None,
    ):
        """Draw a single bounding box with optional label."""
        # bbox is [x, y, width, height] in COCO format
        x, y, w, h = bbox
        # Convert to corner coordinates
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)

        # Draw label if provided
        if label:
            # Get text bounding box
            bbox_text = draw.textbbox((0, 0), label, font=None)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            # Position label above the box, with some padding
            padding = 4
            label_x = x1
            label_y = y1 - text_height - padding * 2

            # If label would go off top of image, place it inside the box instead
            if label_y < 0:
                label_y = y1 + padding

            # Draw background rectangle for text
            bg_box = [
                label_x,
                label_y,
                label_x + text_width + padding * 2,
                label_y + text_height + padding * 2,
            ]
            draw.rectangle(bg_box, fill=color)

            # Draw text in white
            draw.text(
                (label_x + padding, label_y + padding),
                label,
                fill=(255, 255, 255),
                font=None,
            )

    def _get_image_annotations(self, image_id: int) -> tuple[list, list]:
        """Get ground truth and prediction annotations for an image."""
        gt_anns = self.gt_coco.loadAnns(self.gt_coco.getAnnIds(imgIds=[image_id]))
        pred_anns = self.pred_coco.loadAnns(self.pred_coco.getAnnIds(imgIds=[image_id]))
        return gt_anns, pred_anns

    def visualize_image(self, image_id: int) -> Path:
        """Visualize ground truth and predictions for a single image."""
        # Load image
        img_info = self.gt_coco.loadImgs([image_id])[0]
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Get annotations
        gt_anns, pred_anns = self._get_image_annotations(image_id)

        # Draw ground truth boxes (underneath, all same color)
        for ann in gt_anns:
            cat_name = self.gt_coco.loadCats([ann["category_id"]])[0]["name"]
            label = f"GT: {cat_name}"
            self._draw_bbox(draw, ann["bbox"], self.gt_color, label)

        # Draw prediction boxes (on top, color by category)
        for ann in pred_anns:
            cat_id = ann["category_id"]
            cat_name = self.pred_coco.loadCats([cat_id])[0]["name"]
            score = ann.get("score", 1.0)
            label = f"Pred: {cat_name} ({score:.2f})"

            # Get color for this category
            color = self.pred_colors.get(
                cat_id, (255, 0, 0)
            )  # Default to red if not found
            self._draw_bbox(draw, ann["bbox"], color, label)

        # Save result
        output_path = self.output_dir / f"{image_id:012d}_visualization.jpg"
        image.save(output_path)
        return output_path

    def visualize_all(self, max_images: int | None = None):
        """Visualize all images (or subset if max_images specified)."""
        image_ids = self.gt_coco.getImgIds()
        if max_images:
            image_ids = image_ids[:max_images]

        for image_id in image_ids:
            output_path = self.visualize_image(image_id)
            print(f"Saved visualization: {output_path}")


def visualize_predictions(
    gt_json_path: str,
    pred_json_path: str,
    images_dir: str,
    output_dir: str,
    max_images: int | None = None,
):
    """Visualize predictions on images with ground truth overlay.

    Args:
        gt_json_path: Path to ground truth COCO JSON file
        pred_json_path: Path to predictions JSON file
        images_dir: Directory containing images
        output_dir: Directory to save visualizations
        max_images: Maximum number of images to visualize (None for all)
    """
    from pycocotools.coco import COCO

    # Load COCO objects
    gt_coco = COCO(gt_json_path)
    pred_coco = gt_coco.loadRes(pred_json_path)

    # Create visualizer
    visualizer = BBoxVisualizer(
        gt_coco=gt_coco,
        pred_coco=pred_coco,
        images_dir=images_dir,
        output_dir=output_dir,
    )

    # Visualize all images
    visualizer.visualize_all(max_images=max_images)


def main():
    """Run visualization from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize object detection predictions on images"
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
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to visualize (default: all)",
    )

    args = parser.parse_args()

    print(f"Loading ground truth from: {args.gt_path}")
    print(f"Loading predictions from: {args.pred_path}")
    print(f"Images directory: {args.images_dir}")

    visualize_predictions(
        gt_json_path=args.gt_path,
        pred_json_path=args.pred_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
