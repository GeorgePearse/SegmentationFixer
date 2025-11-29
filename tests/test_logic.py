import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.promoters import get_bounding_box, get_center_point, get_random_points
from src.segmentation_fixer import SegmentationFixer


# Mock SAM Predictor
class MockPredictor:
    def set_image(self, image):
        self.image = image

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
    ):
        # Return dummy masks, scores, logits
        # mask shape: (3, H, W)
        h, w, _ = self.image.shape
        masks = np.zeros((3, h, w), dtype=bool)
        # Make one mask cover the whole image for testing
        masks[0, :, :] = True
        scores = np.array([0.9, 0.5, 0.1])
        logits = np.random.randn(3, 256, 256)
        return masks, scores, logits


class TestSegmentationFixer(unittest.TestCase):
    def setUp(self):
        self.predictor = MockPredictor()
        self.fixer = SegmentationFixer(self.predictor)
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.fixer.set_image(self.image)

        # Create a dummy square mask
        self.mask = np.zeros((100, 100), dtype=bool)
        self.mask[25:75, 25:75] = True

    def test_get_bounding_box(self):
        bbox = get_bounding_box(self.mask)
        # Expect [25, 25, 74, 74]
        np.testing.assert_array_equal(bbox, [25, 25, 74, 74])

    def test_get_center_point(self):
        center, label = get_center_point(self.mask)
        # Center of (25, 74) is approx 49
        self.assertTrue(45 <= center[0][0] <= 55)
        self.assertTrue(45 <= center[0][1] <= 55)
        self.assertEqual(label[0], 1)

    def test_get_random_points(self):
        points, labels = get_random_points(self.mask, num_points=5)
        self.assertEqual(len(points), 5)
        self.assertEqual(len(labels), 5)
        # Check points are inside mask
        for x, y in points:
            self.assertTrue(self.mask[y, x])

    def test_fix_segmentation_box(self):
        mask, score, _ = self.fixer.fix_segmentation(self.mask, prompt_type="box")
        self.assertIsNotNone(mask)
        self.assertAlmostEqual(score, 0.9)
        self.assertEqual(mask.shape, (100, 100))

    def test_fix_segmentation_center(self):
        mask, score, _ = self.fixer.fix_segmentation(self.mask, prompt_type="center")
        self.assertIsNotNone(mask)


if __name__ == "__main__":
    unittest.main()
