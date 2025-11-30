# Future Improvements

## Visual Diff for Segmentation Adjustments

One planned improvement is to enhance how segmentation changes are visualized. Currently, the tool displays the original mask and the new SAM-predicted mask side-by-side. While effective for large changes, this makes it difficult to spot subtle boundary adjustments.

A proposed solution is to implement a "Git Diff" style visualization for masks:
- Display the original image as the background.
- Highlight **added pixels** (areas present in the new mask but not the old) in a distinct color (e.g., Green).
- Highlight **removed pixels** (areas present in the old mask but not the new) in a contrasting color (e.g., Red).
- Unchanged mask regions would remain a neutral color or transparent.

This "Diff View" would allow users to instantly assess the exact adjustment made by the model on the same image, significantly speeding up the review process for fine-tuning annotations.
