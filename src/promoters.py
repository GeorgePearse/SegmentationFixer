import numpy as np
import cv2


def get_bounding_box(mask):
    """
    Computes the bounding box of a binary mask.
    Returns: [x_min, y_min, x_max, y_max]
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None  # Empty mask

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    return np.array([x_min, y_min, x_max, y_max])


def get_center_point(mask):
    """
    Computes the center point of the mask.
    Strategies could include center of mass or center of bounding box.
    Here we use center of bounding box for simplicity and robustness.
    Returns: [[x, y]], [label]
    """
    bbox = get_bounding_box(mask)
    if bbox is None:
        return None, None

    x_min, y_min, x_max, y_max = bbox
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)

    # Check if center is actually inside the mask, if not, find the nearest point inside
    if mask[center_y, center_x] == 0:
        y_indices, x_indices = np.where(mask > 0)
        distances = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
        min_idx = np.argmin(distances)
        center_x, center_y = x_indices[min_idx], y_indices[min_idx]

    return np.array([[center_x, center_y]]), np.array([1])


def get_random_points(mask, num_points=1, erosion_iters=0):
    """
    Get random points inside the mask.
    Optionally erode the mask first to ensure points are not on the edge.
    Returns: points, labels (all 1 for foreground)
    """
    work_mask = mask.copy()
    if erosion_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        work_mask = cv2.erode(
            work_mask.astype(np.uint8), kernel, iterations=erosion_iters
        )

    y_indices, x_indices = np.where(work_mask > 0)
    if len(y_indices) == 0:
        return None, None

    if len(y_indices) < num_points:
        chosen_indices = np.arange(len(y_indices))
    else:
        chosen_indices = np.random.choice(len(y_indices), num_points, replace=False)

    points = np.stack([x_indices[chosen_indices], y_indices[chosen_indices]], axis=1)
    labels = np.ones(len(points))

    return points, labels


def get_prompts_from_mask(mask, prompt_type="box"):
    """
    Wrapper to get prompts based on type.
    """
    if prompt_type == "box":
        return get_bounding_box(mask)
    elif prompt_type == "center":
        return get_center_point(mask)
    elif prompt_type == "points":
        return get_random_points(mask, num_points=3)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
