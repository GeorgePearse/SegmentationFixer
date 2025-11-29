import click
import os
import glob
from pathlib import Path
import cv2
import numpy as np
import torch

# Import our modules
# Assuming running from root, so we need to fix path or install as package
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.segmentation_fixer import SegmentationFixer
from src.utils import load_image, load_mask, save_comparison

# Try importing segment_anything, handle if missing
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print(
        "Warning: segment_anything not installed. Mocks will be used if provided, otherwise this will fail."
    )
    SamPredictor = None
    sam_model_registry = None


@click.command()
@click.option(
    "--image_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing images",
)
@click.option(
    "--mask_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing masks",
)
@click.option(
    "--output_dir", required=True, type=click.Path(), help="Directory to save outputs"
)
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Path to SAM checkpoint",
)
@click.option(
    "--model_type", default="vit_h", help="SAM model type (vit_h, vit_l, vit_b)"
)
@click.option(
    "--prompt_type",
    default="box",
    type=click.Choice(["box", "center", "points"]),
    help="Prompting strategy",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run SAM on",
)
def main(image_dir, mask_dir, output_dir, checkpoint, model_type, prompt_type, device):
    """
    Run SAM over existing masks and save comparisons.
    """
    if SamPredictor is None:
        print("Error: segment-anything library is missing. Please install it.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "accepted"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rejected"), exist_ok=True)

    # Load Model
    print(f"Loading SAM model ({model_type}) from {checkpoint}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    fixer = SegmentationFixer(predictor)

    # Get images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    print(f"Found {len(image_paths)} images.")

    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        stem = Path(img_path).stem

        # Find corresponding mask
        # Assuming mask has same name or same stem
        mask_path = os.path.join(mask_dir, base_name)
        if not os.path.exists(mask_path):
            # Try png if image is jpg
            mask_path = os.path.join(mask_dir, stem + ".png")
            if not os.path.exists(mask_path):
                print(f"Mask not found for {base_name}, skipping.")
                continue

        print(f"Processing {base_name}...")

        try:
            image = load_image(img_path)
            mask = load_mask(mask_path)

            fixer.set_image(image)
            new_mask, score, _ = fixer.fix_segmentation(mask, prompt_type=prompt_type)

            if new_mask is None:
                print(f"Skipping {base_name} - empty prompt or error.")
                continue

            # Calculate IoU for logging
            iou = fixer.calculate_iou(mask, new_mask)
            print(f"  SAM Score: {score:.3f}, IoU with original: {iou:.3f}")

            # Save comparison
            out_file = os.path.join(output_dir, f"{stem}_comparison.png")
            save_comparison(image, mask, new_mask, score, out_file)

            # Here we could add logic to automatically accept/reject or move files
            # For now, we just save the comparison and the new mask

            # Save new mask
            new_mask_uint8 = (new_mask * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(output_dir, f"{stem}_new_mask.png"), new_mask_uint8
            )

        except Exception as e:
            print(f"Error processing {base_name}: {e}")


if __name__ == "__main__":
    main()
