"""
Advanced mask refinement algorithms beyond simple edge snapping.
Provides multiple algorithmic approaches to find true object boundaries.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy import ndimage
from skimage import segmentation, morphology


def polygon_to_mask(polygon: List[float], width: int, height: int) -> np.ndarray:
    """Convert polygon coordinates to binary mask."""
    points = np.array(polygon).reshape(-1, 2).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.001) -> List[float]:
    """Convert binary mask to polygon using contour extraction."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)

    # Simplify
    epsilon = epsilon_factor * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # Flatten to [x1, y1, x2, y2, ...]
    return approx.reshape(-1).astype(float).tolist()


class GrabCutRefiner:
    """
    Use GrabCut algorithm for mask refinement.
    Uses the polygon as initialization and lets GrabCut refine the boundary.
    """

    def __init__(self, iterations: int = 5, margin: int = 10):
        """
        Args:
            iterations: Number of GrabCut iterations
            margin: Pixels of margin around mask for probable background/foreground
        """
        self.iterations = iterations
        self.margin = margin

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon using GrabCut.

        Args:
            image: RGB image (H, W, 3)
            polygon: Flattened polygon [x1, y1, x2, y2, ...]

        Returns:
            Refined polygon as flattened list
        """
        h, w = image.shape[:2]
        mask = polygon_to_mask(polygon, w, h)

        # Initialize GrabCut mask
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        gc_mask[mask > 0] = cv2.GC_PR_FGD  # Probable foreground

        # Set definite foreground (eroded mask)
        kernel = np.ones((self.margin, self.margin), np.uint8)
        definite_fg = cv2.erode(mask, kernel, iterations=1)
        gc_mask[definite_fg > 0] = cv2.GC_FGD

        # Set definite background (outside dilated mask)
        definite_bg = cv2.dilate(mask, kernel, iterations=2)
        gc_mask[definite_bg == 0] = cv2.GC_BGD

        # Run GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                image,
                gc_mask,
                None,
                bgd_model,
                fgd_model,
                self.iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error as e:
            print(f"GrabCut failed: {e}")
            return polygon

        # Extract refined mask
        refined_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Convert back to polygon
        return mask_to_polygon(refined_mask)


class ActiveContourRefiner:
    """
    Use active contours (snakes) for boundary refinement.
    Evolves the polygon toward image edges using energy minimization.
    """

    def __init__(
        self,
        alpha: float = 0.015,
        beta: float = 10.0,
        gamma: float = 0.001,
        iterations: int = 100,
    ):
        """
        Args:
            alpha: Snake length shape parameter (continuity)
            beta: Snake smoothness shape parameter (curvature)
            gamma: Explicit time stepping parameter
            iterations: Number of evolution iterations
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iterations = iterations

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon using active contours.

        Args:
            image: RGB or grayscale image
            polygon: Flattened polygon [x1, y1, x2, y2, ...]

        Returns:
            Refined polygon as flattened list
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Get initial contour
        points = np.array(polygon).reshape(-1, 2)

        # Create edge map (negative of gradient magnitude for energy minimization)
        edges = cv2.Canny(gray, 50, 150)

        # Distance transform of edges (attract to edges)
        edge_dist = ndimage.distance_transform_edt(255 - edges)

        # Run active contour
        try:
            from skimage.segmentation import active_contour

            refined = active_contour(
                edge_dist,
                points,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                max_iterations=self.iterations,
            )

            # Flatten
            return refined.reshape(-1).astype(float).tolist()
        except Exception as e:
            print(f"Active contour failed: {e}")
            return polygon


class WatershedRefiner:
    """
    Use watershed segmentation for mask refinement.
    Uses the polygon to seed watershed and finds natural boundaries.
    """

    def __init__(self, edge_threshold: int = 30):
        """
        Args:
            edge_threshold: Gradient threshold for watershed markers
        """
        self.edge_threshold = edge_threshold

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon using watershed.

        Args:
            image: RGB image
            polygon: Flattened polygon [x1, y1, x2, y2, ...]

        Returns:
            Refined polygon as flattened list
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Get initial mask
        mask = polygon_to_mask(polygon, w, h)

        # Create markers
        markers = np.zeros_like(gray, dtype=np.int32)

        # Sure foreground (eroded mask)
        kernel = np.ones((5, 5), np.uint8)
        sure_fg = cv2.erode(mask, kernel, iterations=2)
        markers[sure_fg > 0] = 1

        # Sure background (outside dilated mask)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        markers[sure_bg == 0] = 2

        # Compute gradient
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # Run watershed
        try:
            markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), markers)

            # Extract foreground
            refined_mask = np.where(markers == 1, 255, 0).astype(np.uint8)

            # Convert to polygon
            return mask_to_polygon(refined_mask)
        except Exception as e:
            print(f"Watershed failed: {e}")
            return polygon


class SuperpixelRefiner:
    """
    Use superpixel segmentation for refinement.
    Selects superpixels based on overlap with original mask.
    """

    def __init__(
        self,
        n_segments: int = 100,
        compactness: float = 10.0,
        overlap_threshold: float = 0.5,
    ):
        """
        Args:
            n_segments: Approximate number of superpixels
            compactness: Balance between color similarity and spatial proximity
            overlap_threshold: Minimum overlap to include superpixel
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.overlap_threshold = overlap_threshold

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon using superpixels.

        Args:
            image: RGB image
            polygon: Flattened polygon [x1, y1, x2, y2, ...]

        Returns:
            Refined polygon as flattened list
        """
        h, w = image.shape[:2]

        # Get initial mask
        mask = polygon_to_mask(polygon, w, h)

        # Compute superpixels
        try:
            from skimage.segmentation import slic

            segments = slic(
                image,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=1,
            )

            # Find which superpixels overlap with mask
            refined_mask = np.zeros_like(mask)

            for segment_id in np.unique(segments):
                segment_mask = segments == segment_id
                overlap = np.logical_and(segment_mask, mask > 0).sum()
                total = segment_mask.sum()

                if total > 0 and overlap / total > self.overlap_threshold:
                    refined_mask[segment_mask] = 255

            # Convert to polygon
            return mask_to_polygon(refined_mask)
        except Exception as e:
            print(f"Superpixel refinement failed: {e}")
            return polygon


class MorphologicalRefiner:
    """
    Use morphological operations for refinement.
    Applies closing/opening to smooth boundaries and fill holes.
    """

    def __init__(
        self, operation: str = "close", kernel_size: int = 5, iterations: int = 1
    ):
        """
        Args:
            operation: 'close', 'open', or 'gradient'
            kernel_size: Size of morphological kernel
            iterations: Number of iterations
        """
        self.operation = operation
        self.kernel_size = kernel_size
        self.iterations = iterations

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon using morphological operations.

        Args:
            image: RGB image (not used, but kept for interface consistency)
            polygon: Flattened polygon [x1, y1, x2, y2, ...]

        Returns:
            Refined polygon as flattened list
        """
        h, w = image.shape[:2]
        mask = polygon_to_mask(polygon, w, h)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

        if self.operation == "close":
            refined = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=self.iterations
            )
        elif self.operation == "open":
            refined = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel, iterations=self.iterations
            )
        elif self.operation == "gradient":
            refined = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        else:
            refined = mask

        return mask_to_polygon(refined)


class ConvexHullRefiner:
    """
    Compute convex hull of the polygon.
    Useful for objects that should be convex.
    """

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon by computing convex hull.

        Args:
            image: RGB image (not used)
            polygon: Flattened polygon [x1, y1, x2, y2, ...]

        Returns:
            Convex hull as flattened list
        """
        points = np.array(polygon).reshape(-1, 2).astype(np.int32)
        hull = cv2.convexHull(points)
        return hull.reshape(-1).astype(float).tolist()


class ThresholdRefiner:
    """
    Use adaptive thresholding within the polygon region.
    Good for objects with distinct intensity from background.
    """

    def __init__(self, method: str = "otsu", margin: int = 20):
        """
        Args:
            method: 'otsu', 'adaptive_mean', 'adaptive_gaussian'
            margin: Pixels to expand region for context
        """
        self.method = method
        self.margin = margin

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon using thresholding.

        Args:
            image: RGB image
            polygon: Flattened polygon [x1, y1, x2, y2, ...]

        Returns:
            Refined polygon as flattened list
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Get bounding box with margin
        points = np.array(polygon).reshape(-1, 2)
        x_min = max(0, int(points[:, 0].min()) - self.margin)
        x_max = min(w, int(points[:, 0].max()) + self.margin)
        y_min = max(0, int(points[:, 1].min()) - self.margin)
        y_max = min(h, int(points[:, 1].max()) + self.margin)

        # Extract ROI
        roi = gray[y_min:y_max, x_min:x_max]

        # Apply thresholding
        if self.method == "otsu":
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.method == "adaptive_mean":
            thresh = cv2.adaptiveThreshold(
                roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif self.method == "adaptive_gaussian":
            thresh = cv2.adaptiveThreshold(
                roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            thresh = roi

        # Create full-size mask
        refined_mask = np.zeros((h, w), dtype=np.uint8)
        refined_mask[y_min:y_max, x_min:x_max] = thresh

        # Intersect with original mask region
        original_mask = polygon_to_mask(polygon, w, h)
        dilated = cv2.dilate(
            original_mask, np.ones((self.margin, self.margin), np.uint8)
        )
        refined_mask = cv2.bitwise_and(refined_mask, dilated)

        return mask_to_polygon(refined_mask)


# Registry of all available refiners
REFINERS = {
    "grabcut": GrabCutRefiner,
    "active_contour": ActiveContourRefiner,
    "watershed": WatershedRefiner,
    "superpixel": SuperpixelRefiner,
    "morphological": MorphologicalRefiner,
    "convex_hull": ConvexHullRefiner,
    "threshold": ThresholdRefiner,
}


def get_refiner(method: str, **kwargs):
    """
    Get a refiner instance by name.

    Args:
        method: Refiner name from REFINERS
        **kwargs: Parameters to pass to refiner constructor

    Returns:
        Refiner instance
    """
    if method not in REFINERS:
        raise ValueError(
            f"Unknown refiner: {method}. Available: {list(REFINERS.keys())}"
        )

    return REFINERS[method](**kwargs)
