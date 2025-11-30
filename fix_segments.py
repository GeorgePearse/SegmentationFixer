import click
import os
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.segmentation_fixer import SegmentationFixer
from src.utils import load_image, load_mask, save_comparison, save_metadata

# Import Wrappers
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    SamPredictor = None

try:
    from src.fastsam_wrapper import FastSAMWrapper
except ImportError:
    FastSAMWrapper = None


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
    help="Path to Checkpoint (SAM or FastSAM)",
)
@click.option(
    "--model_type",
    default="vit_h",
    help='Model type (vit_h, vit_l, vit_b) or "fastsam"',
)
@click.option(
    "--prompt_type",
    default="box",
    type=click.Choice(["box", "center", "points"]),
    help="Prompting strategy",
)
@click.option(
    "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device"
)
def main(image_dir, mask_dir, output_dir, checkpoint, model_type, prompt_type, device):
    """
    Run Segmentation Fixer over existing masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    predictor = None

    if model_type.lower() == "fastsam":
        if FastSAMWrapper is None:
            print(
                "FastSAM wrapper failed to import. Ensure requirements are installed."
            )
            return
        print(f"Loading FastSAM from {checkpoint}...")
        predictor = FastSAMWrapper(checkpoint, device=device)

    else:
        # Standard SAM
        if SamPredictor is None:
            print("segment-anything not installed.")
            return
        print(f"Loading SAM ({model_type}) from {checkpoint}...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)

    fixer = SegmentationFixer(predictor)

    # ... (rest of the loop is same as before) ...
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    print(f"Found {len(image_paths)} images.")

    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        stem = Path(img_path).stem

        # Find corresponding mask
        mask_path = os.path.join(mask_dir, base_name)
        if not os.path.exists(mask_path):
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

            iou = fixer.calculate_iou(mask, new_mask)
            print(f"  Score: {score:.3f}, IoU: {iou:.3f}")

            # Save comparison
            out_file = os.path.join(output_dir, f"{stem}_comparison.png")
            save_comparison(image, mask, new_mask, score, out_file)

            # Save metadata
            meta_file = os.path.join(output_dir, f"{stem}_metadata.json")
            save_metadata(score, iou, meta_file)

            # Save new mask
            new_mask_uint8 = (new_mask * 255).astype(np.uint8)

            cv2.imwrite(
                os.path.join(output_dir, f"{stem}_new_mask.png"), new_mask_uint8
            )

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
