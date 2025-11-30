import torch
import numpy as np
import sys
import types

# Monkey-patch ultralytics for FastSAM compatibility
try:
    import ultralytics

    # Check if 'yolo' submodule is missing (newer ultralytics)
    if not hasattr(ultralytics, "yolo"):
        # Create a dummy 'yolo' module
        yolo_module = types.ModuleType("ultralytics.yolo")
        # Map submodules
        yolo_module.cfg = ultralytics.cfg
        yolo_module.utils = ultralytics.utils
        yolo_module.data = ultralytics.data
        if hasattr(ultralytics, "engine"):
            yolo_module.engine = ultralytics.engine

        # Inject into sys.modules
        sys.modules["ultralytics.yolo"] = yolo_module
        sys.modules["ultralytics.yolo.cfg"] = ultralytics.cfg
        sys.modules["ultralytics.yolo.utils"] = ultralytics.utils
        sys.modules["ultralytics.yolo.data"] = ultralytics.data
        if hasattr(ultralytics, "engine"):
            sys.modules["ultralytics.yolo.engine"] = ultralytics.engine

        # Also need specific functions if FastSAM imports them directly from yolo...
        # But mostly it does `from ultralytics.yolo.utils import ops` which works if sys.modules is set.

        # Update ultralytics package to have yolo attribute
        ultralytics.yolo = yolo_module

except ImportError:
    pass

from fastsam import FastSAM, FastSAMPrompt


class FastSAMWrapper:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = FastSAM(model_path)
        self.image = None
        self.cached_results = None

    def set_image(self, image):
        self.image = image
        self.cached_results = None

    def _ensure_inference(self):
        if self.cached_results is None:
            # Run everything inference
            self.cached_results = self.model(
                self.image,
                device=self.device,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
            )

    def predict(
        self, point_coords=None, point_labels=None, box=None, multimask_output=True
    ):
        if self.image is None:
            raise RuntimeError("Image not set")

        self._ensure_inference()

        prompt_process = FastSAMPrompt(
            self.image, self.cached_results, device=self.device
        )

        ann = None

        if box is not None:
            bboxes = [box.tolist()]
            ann = prompt_process.box_prompt(bboxes=bboxes)

        elif point_coords is not None and point_labels is not None:
            ann = prompt_process.point_prompt(
                points=point_coords.tolist(), pointlabel=point_labels.tolist()
            )

        if ann is None:
            return None, 0.0, None

        if isinstance(ann, list):
            ann = np.array(ann)

        if ann.ndim == 2:
            ann = ann[np.newaxis, ...]

        scores = np.ones(ann.shape[0]) * 0.9
        logits = np.zeros(ann.shape)

        return ann, scores, logits

    def to(self, device):
        self.device = device
