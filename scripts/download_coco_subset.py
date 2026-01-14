"""
Download a subset of COCO val2017 images and annotations for testing.

This script:
1. Downloads COCO val2017 annotations
2. Selects 5-10 interesting images with diverse objects
3. Downloads only those specific images
4. Creates ground truth annotations file
5. Documents selected images with links
"""

import json
import urllib.request
from pathlib import Path
import zipfile
import tempfile
import shutil

# Paths
OUTPUT_DIR = Path("data/ground_truth/test")
IMAGES_DIR = OUTPUT_DIR / "images"
DOC_FILE = OUTPUT_DIR / "DATASET_INFO.md"
GT_FILE = OUTPUT_DIR / "gt_annotations.json"

# COCO URLs
ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"


def download_annotations(temp_dir: Path) -> dict:
    """Download and extract COCO annotations."""
    print("Downloading COCO val2017 annotations...")

    zip_path = temp_dir / "annotations.zip"
    urllib.request.urlretrieve(ANNOTATIONS_URL, zip_path)

    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Load instances_val2017.json
    with open(temp_dir / "annotations" / "instances_val2017.json", "r") as f:
        return json.load(f)


def select_interesting_images(coco_data: dict, num_images: int = 50) -> list[dict]:
    """
    Select interesting images with diverse objects.

    Criteria:
    - Mix of categories (person, car, truck, bus, traffic light)
    - Multiple objects per image
    - Not too crowded (< 15 objects per image)
    """
    target_categories = {
        1: "person",
        3: "car",
        6: "bus",
        8: "truck",
        10: "traffic light",
    }

    # Build image -> annotations mapping
    image_to_anns = {}
    for ann in coco_data["annotations"]:
        if ann["category_id"] in target_categories:
            img_id = ann["image_id"]
            if img_id not in image_to_anns:
                image_to_anns[img_id] = []
            image_to_anns[img_id].append(ann)

    # Score images by diversity and annotation count
    scored_images = []
    for img_id, anns in image_to_anns.items():
        if 3 <= len(anns) <= 15:  # Good range
            unique_cats = len(set(ann["category_id"] for ann in anns))
            score = unique_cats * 10 + len(anns)  # Prefer diverse categories
            scored_images.append((score, img_id))

    # Select top N images
    scored_images.sort(reverse=True)
    selected_ids = [img_id for _, img_id in scored_images[:num_images]]

    # Get image info
    id_to_image = {img["id"]: img for img in coco_data["images"]}
    return [id_to_image[img_id] for img_id in selected_ids]


def download_images(selected_images: list[dict], output_dir: Path):
    """Download selected images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(selected_images)} images...")
    for i, img_info in enumerate(selected_images, 1):
        filename = img_info["file_name"]
        url = f"{IMAGE_BASE_URL}/{filename}"
        output_path = output_dir / filename

        print(f"  [{i}/{len(selected_images)}] {filename}")
        urllib.request.urlretrieve(url, output_path)


def create_ground_truth(
    coco_data: dict, selected_images: list[dict], output_file: Path
):
    """Create ground truth annotations file for selected images."""
    selected_ids = {img["id"] for img in selected_images}

    # Filter annotations
    filtered_anns = [
        ann for ann in coco_data["annotations"] if ann["image_id"] in selected_ids
    ]

    # Get category IDs used in filtered annotations
    used_cat_ids = {ann["category_id"] for ann in filtered_anns}

    # Filter categories
    filtered_cats = [
        cat for cat in coco_data["categories"] if cat["id"] in used_cat_ids
    ]

    # Create new COCO format
    gt_data = {
        "images": selected_images,
        "annotations": filtered_anns,
        "categories": filtered_cats,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(gt_data, f, indent=2)

    print(f"\nGround truth saved to: {output_file}")
    print(f"  Images: {len(selected_images)}")
    print(f"  Annotations: {len(filtered_anns)}")
    print(f"  Categories: {len(filtered_cats)}")


def create_documentation(selected_images: list[dict], coco_data: dict, doc_file: Path):
    """Create markdown documentation with image links."""

    # Get annotations per image
    img_id_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    doc_content = """# COCO Dataset Subset - Ground Truth Images

This dataset consists of images from the COCO val2017 dataset.

## Dataset Information

- **Source**: [COCO Dataset](https://cocodataset.org/)
- **Split**: val2017
- **License**: [COCO License](https://cocodataset.org/#termsofuse)
- **Citation**: Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV 2014.

## Selected Images

"""

    for i, img_info in enumerate(selected_images, 1):
        img_id = img_info["id"]
        filename = img_info["file_name"]

        # Get annotations for this image
        anns = img_id_to_anns.get(img_id, [])

        # Count objects by category
        cat_counts = {}
        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"], "unknown")
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

        # Format object counts
        objects_str = ", ".join(
            f"{count} {cat}" for cat, count in sorted(cat_counts.items())
        )

        doc_content += f"""### {i}. {filename}

- **COCO Image ID**: {img_id}
- **Dimensions**: {img_info['width']}x{img_info['height']}
- **Objects**: {objects_str}
- **View on COCO**: [https://cocodataset.org/#explore?id={img_id}](https://cocodataset.org/#explore?id={img_id})
- **Direct Image URL**: [http://images.cocodataset.org/val2017/{filename}](http://images.cocodataset.org/val2017/{filename})

"""

    doc_content += """
## Usage

The ground truth annotations are in `gt_annotations.json` in COCO format.
The images are stored in the `images/` subdirectory.

## Attribution

Images from the COCO dataset are used under the COCO license terms.
Please see https://cocodataset.org/#termsofuse for full details.
"""

    doc_file.parent.mkdir(parents=True, exist_ok=True)
    with open(doc_file, "w") as f:
        f.write(doc_content)

    print(f"\nDocumentation saved to: {doc_file}")


def main():
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download annotations
        coco_data = download_annotations(temp_path)

        # Select interesting images
        print("\nSelecting interesting images...")
        selected_images = select_interesting_images(coco_data, num_images=50)

        # Download images
        download_images(selected_images, IMAGES_DIR)

        # Create ground truth file
        create_ground_truth(coco_data, selected_images, GT_FILE)

        # Create documentation
        create_documentation(selected_images, coco_data, DOC_FILE)

    print("\n✓ Done! Check the documentation file for image details and links.")


if __name__ == "__main__":
    main()
