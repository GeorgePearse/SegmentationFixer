# Segmentation Fixer

A tool for running the Segment Anything Model (SAM) over existing segmentation annotations or predictions to refine them.

## Features

- Supports multiple prompting strategies:
    - Bounding Box from existing mask.
    - Central Point of existing mask.
    - Points inside and outside existing mask.
- Allows accepting or rejecting SAM corrections.

## Installation

```bash
pip install -r requirements.txt
```

## Demo

To generate synthetic demo data and see a comparison between an "imperfect" mask and a simulated "corrected" mask:

```bash
python examples/generate_demo_data.py
```

This will create `examples/demo_data/demo_comparison.png`, showing the original segmentation on the left and the corrected version on the right.

## Usage

(Coming soon)
