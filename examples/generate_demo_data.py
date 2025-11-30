import cv2
import numpy as np
import os
import sys
import random

# Add src to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import save_comparison, save_metadata


def create_synthetic_data(output_dir, num_images=5):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        # 1. Create Image (512x512)
        image = np.zeros((512, 512, 3), dtype=np.uint8)

        # Random background color (dark)
        bg_color = np.random.randint(0, 50, 3).tolist()
        image[:] = bg_color

        # Draw a random shape (The Object)
        shape_type = random.choice(["circle", "rect"])
        color = np.random.randint(50, 255, 3).tolist()

        center_x = random.randint(100, 400)
        center_y = random.randint(100, 400)

        # Ground Truth Mask
        gt_mask = np.zeros((512, 512), dtype=np.uint8)

        if shape_type == "circle":
            radius = random.randint(30, 100)
            cv2.circle(image, (center_x, center_y), radius, color, -1)
            cv2.circle(gt_mask, (center_x, center_y), radius, 255, -1)
        else:
            w = random.randint(50, 150)
            h = random.randint(50, 150)
            pt1 = (center_x - w // 2, center_y - h // 2)
            pt2 = (center_x + w // 2, center_y + h // 2)
            cv2.rectangle(image, pt1, pt2, color, -1)
            cv2.rectangle(gt_mask, pt1, pt2, 255, -1)

        stem = f"demo_{i + 1:03d}"

        # Save Image
        cv2.imwrite(os.path.join(output_dir, f"{stem}_image.jpg"), image)

        # 2. Create "Bad" Mask (Existing Segmentation)
        # Shift and scale randomly to simulate bad initial annotation
        bad_mask = np.zeros((512, 512), dtype=np.uint8)

        shift_x = random.randint(-40, 40)
        shift_y = random.randint(-40, 40)

        # We can simulate this by extracting contours and drawing a shifted bounding box or similar
        # For simplicity, let's just shift the GT mask
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        bad_mask = cv2.warpAffine(gt_mask, M, (512, 512))

        # Save Bad Mask
        cv2.imwrite(os.path.join(output_dir, f"{stem}_mask.png"), bad_mask)

        # 3. "Good" Mask (Simulated SAM Correction) -> GT Mask
        good_mask = gt_mask

        # Convert to boolean for visualization tool
        bad_mask_bool = bad_mask > 128
        good_mask_bool = good_mask > 128

        # Calculate IoU
        intersection = np.logical_and(bad_mask_bool, good_mask_bool).sum()
        union = np.logical_or(bad_mask_bool, good_mask_bool).sum()
        iou = intersection / union if union > 0 else 0.0

        # Mock SAM Score
        score = random.uniform(0.85, 0.99)

        # 4. Generate Comparison
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        save_comparison(
            image_rgb,
            bad_mask_bool,
            good_mask_bool,
            score=score,
            filepath=os.path.join(output_dir, f"{stem}_comparison.png"),
        )

        # Save metadata
        save_metadata(score, iou, os.path.join(output_dir, f"{stem}_metadata.json"))

        # Also save the "new mask" file so the backend move logic works
        cv2.imwrite(os.path.join(output_dir, f"{stem}_new_mask.png"), good_mask)

    print(f"Generated {num_images} demo images in {output_dir}")


if __name__ == "__main__":
    create_synthetic_data("examples/demo_data", num_images=10)
