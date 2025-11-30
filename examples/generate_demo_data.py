import cv2
import numpy as np
import os
import sys

# Add src to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import save_comparison


def create_synthetic_data(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create Image (512x512)
    # Dark background
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    # Draw a filled circle (The Object)
    center = (256, 256)
    radius = 100
    color = (200, 100, 50)  # BGR
    cv2.circle(image, center, radius, color, -1)

    # Save Image
    cv2.imwrite(os.path.join(output_dir, "demo_image.jpg"), image)

    # 2. Create "Bad" Mask (Existing Segmentation)
    # Let's say it's a shifted, smaller box (poor bounding box approximation or bad click)
    bad_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(bad_mask, (200, 200), (300, 300), 255, -1)

    # Save Bad Mask
    cv2.imwrite(
        os.path.join(output_dir, "demo_image.png"), bad_mask
    )  # Saving as png matching the name pattern

    # 3. Create "Good" Mask (Simulated SAM Correction)
    good_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(good_mask, center, radius, 255, -1)

    # Convert to boolean for visualization tool
    bad_mask_bool = bad_mask > 128
    good_mask_bool = good_mask > 128

    # 4. Generate Comparison
    # We need to swap channels for matplotlib (BGR -> RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    save_comparison(
        image_rgb,
        bad_mask_bool,
        good_mask_bool,
        score=0.985,
        filepath=os.path.join(output_dir, "demo_comparison.png"),
    )

    print(f"Generated demo data in {output_dir}")


if __name__ == "__main__":
    create_synthetic_data("examples/demo_data")
