import requests
import zipfile
import io
import json
import os
import random
from pathlib import Path


def download_coco_subset(output_dir, num_images=5):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download Annotations
    print("Downloading annotations (~241MB)...")
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    print("Extracting instances_val2017.json...")
    with z.open("annotations/instances_val2017.json") as f:
        data = json.load(f)

    # 2. Select Random Images
    print(f"Selecting {num_images} random images...")
    all_images = data["images"]
    selected_images = random.sample(all_images, num_images)
    selected_ids = set(img["id"] for img in selected_images)

    # 3. Filter Annotations
    print("Filtering annotations...")
    selected_annotations = [
        ann for ann in data["annotations"] if ann["image_id"] in selected_ids
    ]

    # 4. Construct Mini Dataset
    mini_dataset = {
        "info": data["info"],
        "licenses": data["licenses"],
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": data["categories"],
    }

    # 5. Save JSON
    json_path = output_dir / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(mini_dataset, f, indent=2)
    print(f"Saved annotations to {json_path}")

    # 6. Download Images
    print("Downloading images...")
    for img in selected_images:
        img_url = img["coco_url"]
        file_name = img["file_name"]
        save_path = images_dir / file_name

        print(f"Downloading {file_name}...")
        img_data = requests.get(img_url).content
        with open(save_path, "wb") as f:
            f.write(img_data)

    print("Done!")


if __name__ == "__main__":
    download_coco_subset("data/coco", num_images=5)
