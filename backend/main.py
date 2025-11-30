from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
from typing import List

app = FastAPI()

# Config
# We assume the script is run from the root or we resolve relative to this file
# Let's verify paths.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "examples" / "demo_data"
ACCEPTED_DIR = DATA_DIR / "accepted"
REJECTED_DIR = DATA_DIR / "rejected"

# Ensure directories exist
ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)

# CORS config to allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images statically so frontend can display them
# Mounting at /images
app.mount("/images", StaticFiles(directory=str(DATA_DIR)), name="images")


import json

# ...


@app.get("/api/pending")
def get_pending_images():
    """Returns list of pending comparison images with metadata."""
    # Look for *_comparison.png
    files = sorted(list(DATA_DIR.glob("*_comparison.png")))

    results = []
    for f in files:
        stem = f.name.replace("_comparison.png", "")
        meta_file = DATA_DIR / f"{stem}_metadata.json"

        score = 0.0
        iou = 0.0

        if meta_file.exists():
            try:
                with open(meta_file, "r") as mf:
                    data = json.load(mf)
                    score = data.get("score", 0.0)
                    iou = data.get("iou", 0.0)
            except Exception:
                pass

        results.append({"filename": f.name, "stem": stem, "score": score, "iou": iou})

    # Sort by IoU ascending (most different first)
    results.sort(key=lambda x: x["iou"])

    return results


@app.post("/api/accept/{filename}")
def accept_image(filename: str):
    return move_files(filename, ACCEPTED_DIR)


@app.post("/api/reject/{filename}")
def reject_image(filename: str):
    return move_files(filename, REJECTED_DIR)


def move_files(comparison_filename: str, destination_dir: Path):
    """Moves related files to destination."""
    source_path = DATA_DIR / comparison_filename

    if not source_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine base name (remove _comparison.png)
    base_name = comparison_filename.replace("_comparison.png", "")

    # Files to move
    # 1. The comparison image
    # 2. The new mask (*_new_mask.png)
    # 3. The metadata json (*_metadata.json)

    files_to_move = [
        source_path,
        DATA_DIR / f"{base_name}_new_mask.png",
        DATA_DIR / f"{base_name}_metadata.json",
    ]

    moved_files = []
    for f in files_to_move:
        if f.exists():
            dest = destination_dir / f.name
            shutil.move(str(f), str(dest))
            moved_files.append(f.name)

    return {"status": "success", "moved": moved_files}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
