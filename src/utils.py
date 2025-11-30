import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def save_comparison(image, old_mask, new_mask, score, filepath):
    """
    Save a side-by-side comparison of the old and new mask.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Old Mask
    axes[0].imshow(image)
    show_mask(old_mask, axes[0])
    axes[0].set_title("Original Mask")
    axes[0].axis("off")

    # New Mask
    axes[1].imshow(image)
    show_mask(new_mask, axes[1])
    axes[1].set_title(f"SAM Prediction (Score: {score:.3f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_metadata(score, iou, filepath):
    """
    Save metadata (like score and iou) to a JSON file.
    """
    data = {"score": float(score), "iou": float(iou)}
    with open(filepath, "w") as f:
        json.dump(data, f)


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask > 128  # Return boolean
