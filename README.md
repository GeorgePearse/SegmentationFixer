# Segmentation Fixer

A tool for running the Segment Anything Model (SAM) over existing segmentation annotations or predictions to refine them.

## Features

- Supports multiple prompting strategies:
    - Bounding Box from existing mask.
    - Central Point of existing mask.
    - Points inside and outside existing mask.
- React-based UI for accepting or rejecting SAM corrections.
- FastAPI backend.

## Installation

### Backend

```bash
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

### Development Tools (Pre-commit)

We use `ruff` for linting/formatting and `zuban` for type checking.

1. Install pre-commit: `pip install pre-commit`
2. Install hooks: `pre-commit install`
3. Make sure you have `zuban` installed (`pip install zuban`).

## Demo

To generate synthetic demo data:

```bash
python examples/generate_demo_data.py
```

## Running the App

1. **Start the Backend:**

   ```bash
   python backend/main.py
   ```
   (Runs on http://localhost:8000)

2. **Start the Frontend:**

   ```bash
   cd frontend
   npm run dev
   ```
   (Runs on http://localhost:3000)

   Open your browser to http://localhost:3000 to review the segmentations.

## Building Frontend

To lint the frontend using `oxc` (oxlint):

```bash
cd frontend
npm run lint
```
