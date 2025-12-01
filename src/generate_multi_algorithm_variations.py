"""
Generate variations using multiple algorithms for comparison.
Extends the Rust edge-snapping variations with Python-based algorithms.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any
from advanced_refinement import (
    GrabCutRefiner,
    ActiveContourRefiner,
    WatershedRefiner,
    SuperpixelRefiner,
    MorphologicalRefiner,
    ConvexHullRefiner,
    ThresholdRefiner,
)


def load_image(image_path: Path) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def generate_algorithm_variations(
    image: np.ndarray, polygon: List[float], algorithms: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate variations using different algorithms.

    Args:
        image: RGB image array
        polygon: Flattened polygon [x1, y1, x2, y2, ...]
        algorithms: List of algorithm names to use (None = all)

    Returns:
        List of variation dicts with 'name' and 'new_points' keys
    """
    variations = []

    if algorithms is None:
        algorithms = [
            "grabcut_fast",
            "grabcut_thorough",
            "active_contour_tight",
            "active_contour_loose",
            "watershed",
            "superpixel_fine",
            "superpixel_coarse",
            "morphological_close",
            "morphological_open",
            "convex_hull",
            "threshold_otsu",
            "threshold_adaptive",
        ]

    # GrabCut variations
    if "grabcut_fast" in algorithms:
        try:
            refiner = GrabCutRefiner(iterations=3, margin=10)
            refined = refiner.refine(image, polygon)
            variations.append({"name": "GrabCut (Fast)", "new_points": refined})
        except Exception as e:
            print(f"GrabCut fast failed: {e}")

    if "grabcut_thorough" in algorithms:
        try:
            refiner = GrabCutRefiner(iterations=10, margin=15)
            refined = refiner.refine(image, polygon)
            variations.append({"name": "GrabCut (Thorough)", "new_points": refined})
        except Exception as e:
            print(f"GrabCut thorough failed: {e}")

    # Active Contours variations
    if "active_contour_tight" in algorithms:
        try:
            refiner = ActiveContourRefiner(alpha=0.01, beta=10.0, iterations=100)
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Active Contour (Tight)", "new_points": refined})
        except Exception as e:
            print(f"Active contour tight failed: {e}")

    if "active_contour_loose" in algorithms:
        try:
            refiner = ActiveContourRefiner(alpha=0.03, beta=5.0, iterations=200)
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Active Contour (Loose)", "new_points": refined})
        except Exception as e:
            print(f"Active contour loose failed: {e}")

    # Watershed
    if "watershed" in algorithms:
        try:
            refiner = WatershedRefiner(edge_threshold=30)
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Watershed", "new_points": refined})
        except Exception as e:
            print(f"Watershed failed: {e}")

    # Superpixel variations
    if "superpixel_fine" in algorithms:
        try:
            refiner = SuperpixelRefiner(
                n_segments=200, compactness=10.0, overlap_threshold=0.3
            )
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Superpixel (Fine)", "new_points": refined})
        except Exception as e:
            print(f"Superpixel fine failed: {e}")

    if "superpixel_coarse" in algorithms:
        try:
            refiner = SuperpixelRefiner(
                n_segments=50, compactness=20.0, overlap_threshold=0.5
            )
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Superpixel (Coarse)", "new_points": refined})
        except Exception as e:
            print(f"Superpixel coarse failed: {e}")

    # Morphological variations
    if "morphological_close" in algorithms:
        try:
            refiner = MorphologicalRefiner(
                operation="close", kernel_size=5, iterations=2
            )
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Morphology (Close)", "new_points": refined})
        except Exception as e:
            print(f"Morphological close failed: {e}")

    if "morphological_open" in algorithms:
        try:
            refiner = MorphologicalRefiner(
                operation="open", kernel_size=5, iterations=2
            )
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Morphology (Open)", "new_points": refined})
        except Exception as e:
            print(f"Morphological open failed: {e}")

    # Convex Hull
    if "convex_hull" in algorithms:
        try:
            refiner = ConvexHullRefiner()
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Convex Hull", "new_points": refined})
        except Exception as e:
            print(f"Convex hull failed: {e}")

    # Threshold variations
    if "threshold_otsu" in algorithms:
        try:
            refiner = ThresholdRefiner(method="otsu", margin=30)
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Threshold (Otsu)", "new_points": refined})
        except Exception as e:
            print(f"Threshold otsu failed: {e}")

    if "threshold_adaptive" in algorithms:
        try:
            refiner = ThresholdRefiner(method="adaptive_gaussian", margin=30)
            refined = refiner.refine(image, polygon)
            variations.append({"name": "Threshold (Adaptive)", "new_points": refined})
        except Exception as e:
            print(f"Threshold adaptive failed: {e}")

    return variations


def augment_coco_with_variations(
    coco_path: Path, images_dir: Path, output_path: Path, algorithms: List[str] = None
):
    """
    Load COCO annotations and add algorithm variations to each polygon.

    Args:
        coco_path: Path to COCO JSON file
        images_dir: Path to images directory
        output_path: Where to save augmented COCO file
        algorithms: List of algorithms to use
    """
    # Load COCO
    with open(coco_path) as f:
        coco = json.load(f)

    # Create image ID to filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    total = len(coco["annotations"])

    for idx, ann in enumerate(coco["annotations"]):
        print(f"Processing annotation {idx + 1}/{total} (ID: {ann['id']})")

        if "segmentation" not in ann or not isinstance(ann["segmentation"], list):
            continue

        # Load image
        image_path = images_dir / id_to_file[ann["image_id"]]
        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            continue

        # Process each polygon
        new_segmentation = []
        for polygon in ann["segmentation"]:
            if len(polygon) < 6:  # Need at least 3 points
                new_segmentation.append(polygon)
                continue

            # Generate variations
            variations = generate_algorithm_variations(image, polygon, algorithms)

            # For now, just keep the original
            # In a full system, you'd present these as options
            new_segmentation.append(polygon)

            # Store variations in custom field for inspection
            if "algorithm_variations" not in ann:
                ann["algorithm_variations"] = []
            ann["algorithm_variations"].append(variations)

        ann["segmentation"] = new_segmentation

    # Save augmented COCO
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate multi-algorithm variations")
    parser.add_argument(
        "--coco", type=Path, required=True, help="COCO annotations JSON"
    )
    parser.add_argument("--images", type=Path, required=True, help="Images directory")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--algorithms", nargs="+", help="Specific algorithms to use")

    args = parser.parse_args()

    augment_coco_with_variations(args.coco, args.images, args.output, args.algorithms)
