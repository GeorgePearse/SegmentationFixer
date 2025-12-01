"""
Advanced mask refinement algorithms beyond simple edge snapping.
Provides multiple algorithmic approaches to find true object boundaries.
"""

import numpy as np
import cv2
from typing import List
from scipy import ndimage


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
        mode: str = "default",
    ):
        """
        Args:
            n_segments: Approximate number of superpixels
            compactness: Balance between color similarity and spatial proximity
            overlap_threshold: Minimum overlap to include superpixel
            mode: 'default' (both), 'reduce' (only shrink), 'expand' (only grow)
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.overlap_threshold = overlap_threshold
        self.mode = mode

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

            # Apply mode constraints
            if self.mode == "reduce":
                refined_mask = cv2.bitwise_and(refined_mask, mask)
            elif self.mode == "expand":
                refined_mask = cv2.bitwise_or(refined_mask, mask)

            # Convert to polygon
            return mask_to_polygon(refined_mask)
        except Exception as e:
            print(f"Superpixel refinement failed: {e}")
            return polygon


class SmartSuperpixelRefiner:
    """
    Smart superpixel refinement combining:
    1. Edge-awareness (bias toward staying on Canny edges)
    2. Smooth point bias (morphological smoothing + adaptive polygon simplification)
    3. Bilateral pre-filtering (reduce texture sensitivity)

    Produces clean, smooth polygons that follow real object boundaries.
    """

    def __init__(
        self,
        n_segments: int = 100,
        compactness: float = 20.0,
        overlap_threshold: float = 0.5,
        edge_stickiness: float = 0.3,
        smooth_kernel: int = 7,
        polygon_epsilon: float = 0.003,
        canny_low: int = 50,
        canny_high: int = 150,
        bilateral_sigma: int = 75,
        mode: str = "default",
    ):
        """
        Args:
            n_segments: Approximate number of superpixels
            compactness: Balance between color/spatial (higher = more regular)
            overlap_threshold: Base threshold for superpixel inclusion
            edge_stickiness: Bias toward keeping boundary on edges (0-1)
            smooth_kernel: Morphological kernel size for smoothing
            polygon_epsilon: Douglas-Peucker epsilon factor (higher = smoother)
            canny_low: Canny edge detection low threshold
            canny_high: Canny edge detection high threshold
            bilateral_sigma: Bilateral filter sigma for pre-smoothing
            mode: 'default', 'reduce', or 'expand'
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.overlap_threshold = overlap_threshold
        self.edge_stickiness = edge_stickiness
        self.smooth_kernel = smooth_kernel
        self.polygon_epsilon = polygon_epsilon
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.bilateral_sigma = bilateral_sigma
        self.mode = mode

    def _smooth_mask_to_polygon(self, mask: np.ndarray) -> List[float]:
        """Convert mask to polygon with morphological smoothing."""
        # Apply morphological smoothing: close then open
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.smooth_kernel, self.smooth_kernel)
        )
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        # Extract contours
        contours, _ = cv2.findContours(
            smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)

        # Apply Douglas-Peucker with adaptive epsilon for smooth output
        epsilon = self.polygon_epsilon * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        return approx.reshape(-1).astype(float).tolist()

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon using smart superpixels.

        Pipeline:
        1. Detect Canny edges for edge-awareness
        2. Pre-smooth image with bilateral filter
        3. Generate superpixels
        4. Select superpixels with edge-aware thresholding
        5. Smooth the mask morphologically
        6. Convert to smooth polygon
        """
        h, w = image.shape[:2]
        mask = polygon_to_mask(polygon, w, h)

        try:
            from skimage.segmentation import slic

            # 1. Detect Canny edges
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)

            # 2. Find where mask boundary aligns with edges
            mask_boundary = cv2.Canny(mask, 100, 200)
            edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            edge_aligned_boundary = cv2.bitwise_and(mask_boundary, edges_dilated)
            edge_zone = cv2.dilate(
                edge_aligned_boundary, np.ones((5, 5), np.uint8), iterations=2
            )

            # 3. Pre-smooth image with bilateral filter
            smoothed_img = cv2.bilateralFilter(
                image,
                d=9,
                sigmaColor=self.bilateral_sigma,
                sigmaSpace=self.bilateral_sigma,
            )

            # 4. Generate superpixels
            segments = slic(
                smoothed_img,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=1,
            )

            # 5. Build refined mask with edge-awareness
            refined_mask = np.zeros_like(mask)

            for segment_id in np.unique(segments):
                segment_mask = segments == segment_id
                total = segment_mask.sum()
                if total == 0:
                    continue

                overlap = np.logical_and(segment_mask, mask > 0).sum()
                overlap_ratio = overlap / total

                # Edge-aware threshold adjustment
                edge_zone_overlap = np.logical_and(segment_mask, edge_zone > 0).sum()
                touches_edge_zone = edge_zone_overlap > 0

                if touches_edge_zone:
                    if overlap_ratio > 0.5:
                        effective_threshold = self.overlap_threshold * (
                            1 - self.edge_stickiness
                        )
                    else:
                        effective_threshold = self.overlap_threshold * (
                            1 + self.edge_stickiness
                        )
                else:
                    effective_threshold = self.overlap_threshold

                if overlap_ratio > effective_threshold:
                    refined_mask[segment_mask] = 255

            # Apply mode constraints
            if self.mode == "reduce":
                refined_mask = cv2.bitwise_and(refined_mask, mask)
            elif self.mode == "expand":
                refined_mask = cv2.bitwise_or(refined_mask, mask)

            # 6. Convert to smooth polygon
            return self._smooth_mask_to_polygon(refined_mask)

        except Exception as e:
            print(f"Smart superpixel refinement failed: {e}")
            return polygon


class ConservativeSmartSuperpixelRefiner:
    """
    Smart superpixel refinement with penalty for large area changes.

    Extends SmartSuperpixelRefiner with:
    - Area change penalty: superpixels that would cause large area changes
      are less likely to be included/excluded
    - Preserves overall mask area while improving boundary quality

    This is more conservative - it refines boundaries without drastically
    changing the segmentation size.
    """

    def __init__(
        self,
        n_segments: int = 100,
        compactness: float = 20.0,
        overlap_threshold: float = 0.5,
        edge_stickiness: float = 0.3,
        area_penalty: float = 0.5,
        max_area_change: float = 0.15,
        smooth_kernel: int = 7,
        polygon_epsilon: float = 0.003,
        canny_low: int = 50,
        canny_high: int = 150,
        bilateral_sigma: int = 75,
        mode: str = "default",
    ):
        """
        Args:
            n_segments: Approximate number of superpixels
            compactness: Balance between color/spatial (higher = more regular)
            overlap_threshold: Base threshold for superpixel inclusion
            edge_stickiness: Bias toward keeping boundary on edges (0-1)
            area_penalty: How much to penalize area changes (0-1)
            max_area_change: Maximum allowed fractional area change (0-1)
            smooth_kernel: Morphological kernel size for smoothing
            polygon_epsilon: Douglas-Peucker epsilon factor (higher = smoother)
            canny_low: Canny edge detection low threshold
            canny_high: Canny edge detection high threshold
            bilateral_sigma: Bilateral filter sigma for pre-smoothing
            mode: 'default', 'reduce', or 'expand'
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.overlap_threshold = overlap_threshold
        self.edge_stickiness = edge_stickiness
        self.area_penalty = area_penalty
        self.max_area_change = max_area_change
        self.smooth_kernel = smooth_kernel
        self.polygon_epsilon = polygon_epsilon
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.bilateral_sigma = bilateral_sigma
        self.mode = mode

    def _smooth_mask_to_polygon(self, mask: np.ndarray) -> List[float]:
        """Convert mask to polygon with morphological smoothing."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.smooth_kernel, self.smooth_kernel)
        )
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)
        epsilon = self.polygon_epsilon * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        return approx.reshape(-1).astype(float).tolist()

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon with area-conservative smart superpixels.

        Pipeline:
        1. Detect Canny edges for edge-awareness
        2. Pre-smooth image with bilateral filter
        3. Generate superpixels
        4. Score each superpixel considering:
           - Overlap with original mask
           - Edge alignment
           - Area change impact (penalty for large changes)
        5. Select superpixels that improve boundary without changing area too much
        6. Smooth and convert to polygon
        """
        h, w = image.shape[:2]
        mask = polygon_to_mask(polygon, w, h)
        original_area = (mask > 0).sum()

        try:
            from skimage.segmentation import slic

            # 1. Detect Canny edges
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)

            # 2. Find where mask boundary aligns with edges
            mask_boundary = cv2.Canny(mask, 100, 200)
            edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            edge_aligned_boundary = cv2.bitwise_and(mask_boundary, edges_dilated)
            edge_zone = cv2.dilate(
                edge_aligned_boundary, np.ones((5, 5), np.uint8), iterations=2
            )

            # 3. Pre-smooth image with bilateral filter
            smoothed_img = cv2.bilateralFilter(
                image,
                d=9,
                sigmaColor=self.bilateral_sigma,
                sigmaSpace=self.bilateral_sigma,
            )

            # 4. Generate superpixels
            segments = slic(
                smoothed_img,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=1,
            )

            # 5. Build refined mask with edge-awareness AND area penalty
            refined_mask = np.zeros_like(mask)
            running_area_change = 0

            # Sort superpixels by their "borderline-ness" (closer to threshold = process first)
            superpixel_scores = []
            for segment_id in np.unique(segments):
                segment_mask = segments == segment_id
                total = segment_mask.sum()
                if total == 0:
                    continue

                overlap = np.logical_and(segment_mask, mask > 0).sum()
                overlap_ratio = overlap / total

                # How borderline is this superpixel? (closer to 0.5 = more borderline)
                borderline_score = abs(overlap_ratio - 0.5)
                superpixel_scores.append((segment_id, overlap_ratio, total, borderline_score))

            # Process non-borderline superpixels first (clear decisions)
            superpixel_scores.sort(key=lambda x: -x[3])

            for segment_id, overlap_ratio, total, _ in superpixel_scores:
                segment_mask = segments == segment_id

                # Edge-aware threshold adjustment
                edge_zone_overlap = np.logical_and(segment_mask, edge_zone > 0).sum()
                touches_edge_zone = edge_zone_overlap > 0

                if touches_edge_zone:
                    if overlap_ratio > 0.5:
                        effective_threshold = self.overlap_threshold * (
                            1 - self.edge_stickiness
                        )
                    else:
                        effective_threshold = self.overlap_threshold * (
                            1 + self.edge_stickiness
                        )
                else:
                    effective_threshold = self.overlap_threshold

                # Calculate area change if we include/exclude this superpixel
                currently_in_mask = np.logical_and(segment_mask, mask > 0).sum()
                would_add = total - currently_in_mask  # pixels we'd add
                would_remove = currently_in_mask  # pixels we'd remove

                if overlap_ratio > effective_threshold:
                    # Would include this superpixel
                    area_change = would_add / original_area if original_area > 0 else 0
                else:
                    # Would exclude this superpixel
                    area_change = -would_remove / original_area if original_area > 0 else 0

                # Apply area penalty: penalize decisions that change area significantly
                area_change_magnitude = abs(area_change)

                # Check if this would exceed max area change
                if abs(running_area_change + area_change) > self.max_area_change:
                    # Fall back to original mask state for this superpixel
                    if currently_in_mask > total / 2:
                        refined_mask[segment_mask] = 255
                    continue

                # Adjust threshold based on area penalty
                # Large area changes require stronger evidence (higher overlap or lower overlap)
                area_adjusted_threshold = effective_threshold
                if area_change_magnitude > 0.01:  # Only apply penalty for non-trivial changes
                    penalty = self.area_penalty * area_change_magnitude * 10
                    if overlap_ratio > 0.5:
                        # To include, need even higher overlap
                        area_adjusted_threshold = min(0.9, effective_threshold + penalty)
                    else:
                        # To exclude, need even lower overlap
                        area_adjusted_threshold = max(0.1, effective_threshold - penalty)

                if overlap_ratio > area_adjusted_threshold:
                    refined_mask[segment_mask] = 255
                    running_area_change += area_change

            # Apply mode constraints
            if self.mode == "reduce":
                refined_mask = cv2.bitwise_and(refined_mask, mask)
            elif self.mode == "expand":
                refined_mask = cv2.bitwise_or(refined_mask, mask)

            # 6. Convert to smooth polygon
            return self._smooth_mask_to_polygon(refined_mask)

        except Exception as e:
            print(f"Conservative smart superpixel refinement failed: {e}")
            return polygon


class EdgeAwareSuperpixelRefiner:
    """
    Superpixel refinement with bias toward staying on Canny edges.

    Key behavior: If the current mask boundary follows a Canny edge,
    the refinement will preserve that edge-following behavior rather
    than pulling the boundary away.
    """

    def __init__(
        self,
        n_segments: int = 100,
        compactness: float = 20.0,
        overlap_threshold: float = 0.5,
        edge_stickiness: float = 0.3,
        canny_low: int = 50,
        canny_high: int = 150,
        smooth_sigma: int = 75,
        mode: str = "default",
    ):
        """
        Args:
            n_segments: Approximate number of superpixels
            compactness: Balance between color/spatial (higher = more regular shapes)
            overlap_threshold: Base threshold for superpixel inclusion
            edge_stickiness: How much to bias toward keeping boundary on edges (0-1)
            canny_low: Canny edge detection low threshold
            canny_high: Canny edge detection high threshold
            smooth_sigma: Bilateral filter sigma for pre-smoothing
            mode: 'default', 'reduce', or 'expand'
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.overlap_threshold = overlap_threshold
        self.edge_stickiness = edge_stickiness
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.smooth_sigma = smooth_sigma
        self.mode = mode

    def refine(self, image: np.ndarray, polygon: List[float]) -> List[float]:
        """
        Refine polygon using edge-aware superpixels.

        The algorithm:
        1. Detect Canny edges in the image
        2. Find where current mask boundary aligns with edges (edge zones)
        3. Pre-smooth image to reduce texture sensitivity
        4. Generate superpixels
        5. For superpixels near edge zones, bias toward preserving the edge
        6. For superpixels away from edges, use standard overlap threshold
        """
        h, w = image.shape[:2]
        mask = polygon_to_mask(polygon, w, h)

        try:
            from skimage.segmentation import slic

            # 1. Detect Canny edges
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)

            # 2. Find current mask boundary
            mask_boundary = cv2.Canny(mask, 100, 200)

            # 3. Find where mask boundary aligns with image edges
            # Dilate edges slightly to allow for small misalignments
            edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            edge_aligned_boundary = cv2.bitwise_and(mask_boundary, edges_dilated)

            # Create "edge zone" - areas near edge-aligned boundaries that should be preserved
            edge_zone = cv2.dilate(edge_aligned_boundary, np.ones((5, 5), np.uint8), iterations=2)

            # 4. Pre-smooth image to reduce SLIC sensitivity to texture
            smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=self.smooth_sigma, sigmaSpace=self.smooth_sigma)

            # 5. Generate superpixels on smoothed image
            segments = slic(
                smoothed,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=1,
            )

            # 6. Build refined mask with edge-awareness
            refined_mask = np.zeros_like(mask)

            for segment_id in np.unique(segments):
                segment_mask = segments == segment_id
                total = segment_mask.sum()
                if total == 0:
                    continue

                # Basic overlap ratio
                overlap = np.logical_and(segment_mask, mask > 0).sum()
                overlap_ratio = overlap / total

                # Check if this superpixel touches the edge zone
                edge_zone_overlap = np.logical_and(segment_mask, edge_zone > 0).sum()
                touches_edge_zone = edge_zone_overlap > 0

                if touches_edge_zone:
                    # In edge zone: bias toward keeping the boundary on the edge
                    # If superpixel is mostly inside mask, keep it inside
                    # If mostly outside, keep it outside
                    # This prevents the boundary from "jumping" off the edge
                    if overlap_ratio > 0.5:
                        # Mostly inside - include it (lower threshold)
                        effective_threshold = self.overlap_threshold * (1 - self.edge_stickiness)
                    else:
                        # Mostly outside - exclude it (higher threshold)
                        effective_threshold = self.overlap_threshold * (1 + self.edge_stickiness)
                else:
                    # Away from edges: use standard threshold
                    effective_threshold = self.overlap_threshold

                if overlap_ratio > effective_threshold:
                    refined_mask[segment_mask] = 255

            # Apply mode constraints
            if self.mode == "reduce":
                refined_mask = cv2.bitwise_and(refined_mask, mask)
            elif self.mode == "expand":
                refined_mask = cv2.bitwise_or(refined_mask, mask)

            return mask_to_polygon(refined_mask)

        except Exception as e:
            print(f"Edge-aware superpixel refinement failed: {e}")
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
    "smart_superpixel": SmartSuperpixelRefiner,
    "conservative_smart_superpixel": ConservativeSmartSuperpixelRefiner,
    "edge_aware_superpixel": EdgeAwareSuperpixelRefiner,
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
