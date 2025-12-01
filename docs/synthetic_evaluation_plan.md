# Synthetic Dataset Evaluation Plan

## Overview
Create synthetic degraded masks from clean ground truth to systematically evaluate and optimize mask correction algorithms.

## Degradation Types

### 1. Vertex Noise
Simulate human annotation errors by adding random noise to polygon vertices.

**Parameters:**
- `noise_std`: Standard deviation of Gaussian noise (pixels)
  - Light: 1-2px
  - Medium: 3-5px
  - Heavy: 6-10px
- `noise_type`: 'gaussian', 'uniform'
- `affected_vertices`: Percentage of vertices to perturb (0-100%)

**Implementation:**
```python
def add_vertex_noise(polygon, noise_std=3.0, affected_pct=100.0):
    """Add random noise to polygon vertices"""
    for i, (x, y) in enumerate(polygon):
        if random.random() * 100 < affected_pct:
            polygon[i] = (
                x + np.random.normal(0, noise_std),
                y + np.random.normal(0, noise_std)
            )
    return polygon
```

### 2. Morphological Operations
Simulate systematic over/under-segmentation.

**Parameters:**
- `operation`: 'erode', 'dilate'
- `kernel_size`: Size of morphological kernel (pixels)
  - Small: 3-5px
  - Medium: 7-11px
  - Large: 13-21px
- `iterations`: Number of times to apply operation (1-5)

**Use Cases:**
- Erosion: Simulates conservative annotations
- Dilation: Simulates loose/optimistic annotations

### 3. Vertex Decimation
Simulate simplified/low-resolution masks.

**Parameters:**
- `reduction_factor`: Percentage of vertices to remove (0-80%)
- `method`: 'uniform', 'rdp' (Ramer-Douglas-Peucker), 'random'
- `epsilon`: Tolerance for RDP algorithm

**Implementation:**
```python
def decimate_vertices(polygon, reduction_factor=0.5, method='uniform'):
    """Remove vertices while preserving approximate shape"""
    if method == 'uniform':
        keep_every_n = int(1 / (1 - reduction_factor))
        return polygon[::keep_every_n]
    elif method == 'rdp':
        # Use Ramer-Douglas-Peucker algorithm
        return rdp_simplify(polygon, epsilon)
```

### 4. Smoothing (Over-smoothing)
Simulate automated polygon smoothing that loses detail.

**Parameters:**
- `smoothing_iterations`: Number of Laplacian smoothing passes (1-10)
- `smoothing_factor`: Weight of smoothing (0.0-1.0)

### 5. Quantization Noise
Simulate mask rasterization/discretization errors.

**Parameters:**
- `grid_size`: Snap vertices to grid (e.g., 0.5px, 1px, 2px)

### 6. Boundary Shift
Systematic offset of entire boundary.

**Parameters:**
- `shift_pixels`: Amount to shift boundary (can be negative)
- `direction`: 'inward', 'outward', 'random_radial'

### 7. Local Perturbations
Simulate specific annotation errors in local regions.

**Parameters:**
- `perturbation_type`: 'bulge', 'indent', 'wiggle'
- `num_perturbations`: Number of local errors (1-5)
- `severity`: Magnitude of perturbation (pixels)

## Degradation Combinations

Create realistic degradation by combining multiple types:

### Profile A: "Sloppy Human Annotation"
```python
degradations = [
    ('vertex_noise', {'noise_std': 4.0, 'affected_pct': 100}),
    ('smoothing', {'iterations': 2, 'factor': 0.3}),
]
```

### Profile B: "Automated Tool Output"
```python
degradations = [
    ('morphological', {'operation': 'erode', 'kernel': 3, 'iterations': 1}),
    ('vertex_decimation', {'reduction': 0.3, 'method': 'rdp'}),
]
```

### Profile C: "Low-Resolution Masks"
```python
degradations = [
    ('quantization', {'grid_size': 2.0}),
    ('vertex_decimation', {'reduction': 0.5, 'method': 'uniform'}),
    ('smoothing', {'iterations': 3}),
]
```

### Profile D: "Boundary Uncertainty"
```python
degradations = [
    ('vertex_noise', {'noise_std': 2.0}),
    ('boundary_shift', {'shift': -1.5, 'direction': 'random_radial'}),
    ('local_perturbations', {'type': 'wiggle', 'num': 3, 'severity': 5}),
]
```

## Dataset Structure

```
synthetic_evaluation/
├── original/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── annotations.json  # Ground truth
├── degraded/
│   ├── profile_a_light/
│   │   └── annotations.json
│   ├── profile_a_medium/
│   │   └── annotations.json
│   ├── profile_a_heavy/
│   │   └── annotations.json
│   ├── profile_b_light/
│   │   └── annotations.json
│   └── ...
└── corrected/
    ├── profile_a_light/
    │   ├── tight_3px/
    │   │   └── annotations.json
    │   ├── medium_8px/
    │   │   └── annotations.json
    │   └── ...
    └── ...
```

## Evaluation Metrics

### 1. Geometric Metrics

**Intersection over Union (IoU)**
```python
def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0
```

**Boundary F1-Score**
```python
def boundary_f1(pred_polygon, gt_polygon, threshold=2.0):
    """F1 score based on boundary point distances"""
    precision = compute_boundary_precision(pred_polygon, gt_polygon, threshold)
    recall = compute_boundary_recall(pred_polygon, gt_polygon, threshold)
    return 2 * (precision * recall) / (precision + recall)
```

**Hausdorff Distance**
```python
def hausdorff_distance(poly1, poly2):
    """Maximum distance from a point in one set to nearest point in other"""
    from scipy.spatial.distance import directed_hausdorff
    return max(
        directed_hausdorff(poly1, poly2)[0],
        directed_hausdorff(poly2, poly1)[0]
    )
```

### 2. Vertex-Level Metrics

**Mean Vertex Distance**
```python
def mean_vertex_distance(pred_polygon, gt_polygon):
    """Average distance between corresponding vertices"""
    # Assumes polygons have same number of vertices or uses DTW alignment
    distances = [distance(p, g) for p, g in zip(pred_polygon, gt_polygon)]
    return np.mean(distances)
```

**Vertex Position Error (VPE)**
- L2 distance for each vertex
- Report: mean, median, p90, p95, max

### 3. Shape Similarity

**Shape Context Distance**
- Compares local shape descriptors around boundary points

**Curvature Preservation**
```python
def curvature_similarity(poly1, poly2):
    """Compare curvature distributions"""
    curv1 = compute_curvature(poly1)
    curv2 = compute_curvature(poly2)
    return 1 - wasserstein_distance(curv1, curv2)
```

### 4. Improvement Metrics

**Recovery Rate**
```python
def recovery_rate(degraded_iou, corrected_iou, gt_iou=1.0):
    """Percentage of lost quality recovered"""
    lost = gt_iou - degraded_iou
    recovered = corrected_iou - degraded_iou
    return (recovered / lost * 100) if lost > 0 else 100
```

**Relative Improvement**
```python
def relative_improvement(degraded_metric, corrected_metric):
    """Percentage improvement over degraded"""
    return ((corrected_metric - degraded_metric) / degraded_metric) * 100
```

## Evaluation Pipeline

### Step 1: Generate Degraded Dataset
```python
def generate_synthetic_dataset(
    original_annotations_path,
    output_dir,
    degradation_profiles
):
    """
    Generate multiple degraded versions of clean annotations
    
    Args:
        original_annotations_path: Path to ground truth COCO JSON
        output_dir: Where to save degraded versions
        degradation_profiles: List of (name, degradation_config) tuples
    """
    gt_dataset = load_coco_annotations(original_annotations_path)
    
    for profile_name, degradations in degradation_profiles:
        degraded_dataset = copy.deepcopy(gt_dataset)
        
        for ann in degraded_dataset['annotations']:
            for deg_type, deg_params in degradations:
                ann['segmentation'] = apply_degradation(
                    ann['segmentation'], 
                    deg_type, 
                    deg_params
                )
        
        save_path = output_dir / profile_name / 'annotations.json'
        save_coco_annotations(degraded_dataset, save_path)
```

### Step 2: Apply Correction Algorithms
```python
def evaluate_correction_methods(
    degraded_annotations_path,
    images_dir,
    correction_configs,
    output_dir
):
    """
    Apply different correction configurations to degraded masks
    
    Args:
        degraded_annotations_path: Path to degraded COCO JSON
        images_dir: Path to images
        correction_configs: List of (name, config) tuples
        output_dir: Where to save corrected versions
    """
    for config_name, config in correction_configs:
        corrected = apply_edge_snapping(
            degraded_annotations_path,
            images_dir,
            snap_distance=config['snap_dist'],
            smooth_iterations=config['smooth_iters'],
            edge_config=config['edge_params']
        )
        
        save_path = output_dir / config_name / 'annotations.json'
        save_coco_annotations(corrected, save_path)
```

### Step 3: Compute Metrics
```python
def evaluate_all_metrics(
    ground_truth_path,
    degraded_path,
    corrected_paths,
    images_dir
):
    """
    Compute all metrics comparing degraded and corrected to GT
    
    Returns:
        DataFrame with metrics for each annotation and configuration
    """
    results = []
    
    gt_dataset = load_coco_annotations(ground_truth_path)
    deg_dataset = load_coco_annotations(degraded_path)
    
    for config_name, corrected_path in corrected_paths.items():
        corr_dataset = load_coco_annotations(corrected_path)
        
        for gt_ann, deg_ann, corr_ann in zip_annotations(
            gt_dataset, deg_dataset, corr_dataset
        ):
            metrics = {
                'annotation_id': gt_ann['id'],
                'config': config_name,
                
                # IoU metrics
                'iou_degraded': compute_iou(deg_ann, gt_ann),
                'iou_corrected': compute_iou(corr_ann, gt_ann),
                
                # Boundary metrics
                'boundary_f1_degraded': boundary_f1(deg_ann, gt_ann),
                'boundary_f1_corrected': boundary_f1(corr_ann, gt_ann),
                
                # Distance metrics
                'hausdorff_degraded': hausdorff_distance(deg_ann, gt_ann),
                'hausdorff_corrected': hausdorff_distance(corr_ann, gt_ann),
                
                # Improvement
                'recovery_rate': recovery_rate(
                    compute_iou(deg_ann, gt_ann),
                    compute_iou(corr_ann, gt_ann)
                ),
            }
            
            results.append(metrics)
    
    return pd.DataFrame(results)
```

### Step 4: Analyze Results
```python
def analyze_results(results_df):
    """
    Generate comprehensive analysis of correction performance
    """
    # Group by configuration
    by_config = results_df.groupby('config').agg({
        'iou_corrected': ['mean', 'median', 'std'],
        'boundary_f1_corrected': ['mean', 'median'],
        'recovery_rate': ['mean', 'median'],
        'hausdorff_corrected': ['mean', 'median'],
    })
    
    # Find best configuration per degradation profile
    best_configs = results_df.loc[
        results_df.groupby('degradation_profile')['iou_corrected'].idxmax()
    ]
    
    # Statistical significance tests
    from scipy.stats import wilcoxon
    for config_a, config_b in itertools.combinations(configs, 2):
        stat, p_value = wilcoxon(
            results_df[results_df.config == config_a]['iou_corrected'],
            results_df[results_df.config == config_b]['iou_corrected']
        )
        print(f"{config_a} vs {config_b}: p={p_value:.4f}")
    
    return by_config, best_configs
```

## Correction Configurations to Test

```python
correction_configs = [
    # Baseline
    ('no_correction', {'snap_dist': 0, 'smooth_iters': 0}),
    
    # Tight snapping
    ('tight_sharp', {'snap_dist': 3, 'smooth_iters': 0}),
    ('tight_smooth_light', {'snap_dist': 3, 'smooth_iters': 1}),
    ('tight_smooth_medium', {'snap_dist': 3, 'smooth_iters': 2}),
    
    # Medium snapping
    ('medium_sharp', {'snap_dist': 8, 'smooth_iters': 0}),
    ('medium_smooth_light', {'snap_dist': 8, 'smooth_iters': 2}),
    ('medium_smooth_heavy', {'snap_dist': 8, 'smooth_iters': 4}),
    
    # Loose snapping
    ('loose_sharp', {'snap_dist': 15, 'smooth_iters': 0}),
    ('loose_smooth', {'snap_dist': 15, 'smooth_iters': 3}),
    
    # Very loose
    ('very_loose', {'snap_dist': 25, 'smooth_iters': 3}),
    
    # Edge detection variations
    ('medium_canny_sensitive', {
        'snap_dist': 8, 
        'smooth_iters': 2,
        'edge_method': 'canny',
        'canny_low': 30,
        'canny_high': 80,
    }),
    ('medium_sobel', {
        'snap_dist': 8,
        'smooth_iters': 2, 
        'edge_method': 'sobel',
        'gradient_threshold': 20,
    }),
]
```

## Expected Outcomes

### Hypothesis Testing

**H1: Optimal snap distance varies by degradation type**
- Vertex noise → smaller snap distance
- Morphological erosion → larger snap distance
- Boundary shift → medium snap distance

**H2: Smoothing helps with noisy degradations**
- High vertex noise → benefits from smoothing
- Systematic shifts → minimal smoothing better

**H3: Edge detection method matters**
- High-contrast images → Canny performs well
- Textured images → Sobel or combined approach better

### Performance Targets

For each degradation profile:
- **Recovery Rate:** >70% of lost IoU recovered
- **Boundary F1:** >0.85 on corrected masks
- **Hausdorff Distance:** <3px from ground truth

### Visualization

Generate comparison visualizations:
```python
def visualize_comparison(image, gt_mask, degraded_mask, corrected_masks):
    """
    Create side-by-side comparison showing:
    - Original image + GT boundary
    - Degraded mask overlay
    - Each corrected version with metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show ground truth
    axes[0, 0].imshow(image)
    plot_polygon_boundary(axes[0, 0], gt_mask, color='green', label='GT')
    axes[0, 0].set_title('Ground Truth')
    
    # Show degraded
    axes[0, 1].imshow(image)
    plot_polygon_boundary(axes[0, 1], degraded_mask, color='red', label='Degraded')
    iou_deg = compute_iou(degraded_mask, gt_mask)
    axes[0, 1].set_title(f'Degraded (IoU: {iou_deg:.3f})')
    
    # Show each correction
    for idx, (name, corrected) in enumerate(corrected_masks.items()):
        ax = axes.flatten()[idx + 2]
        ax.imshow(image)
        plot_polygon_boundary(ax, corrected, color='blue', label='Corrected')
        iou = compute_iou(corrected, gt_mask)
        recovery = recovery_rate(iou_deg, iou)
        ax.set_title(f'{name}\nIoU: {iou:.3f} | Recovery: {recovery:.1f}%')
```

## Implementation Checklist

- [ ] Implement degradation functions
  - [ ] Vertex noise
  - [ ] Morphological operations
  - [ ] Vertex decimation
  - [ ] Smoothing
  - [ ] Quantization
  - [ ] Boundary shift
  - [ ] Local perturbations
- [ ] Create degradation profiles
- [ ] Generate synthetic dataset
- [ ] Implement evaluation metrics
  - [ ] IoU
  - [ ] Boundary F1
  - [ ] Hausdorff distance
  - [ ] Mean vertex distance
  - [ ] Recovery rate
- [ ] Build evaluation pipeline
- [ ] Run experiments
- [ ] Statistical analysis
- [ ] Generate visualizations
- [ ] Write results report

## Files to Create

```
src/
├── synthetic/
│   ├── __init__.py
│   ├── degradations.py      # All degradation functions
│   ├── profiles.py          # Degradation profile definitions
│   ├── generator.py         # Dataset generation
│   └── visualize.py         # Visualization utilities
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # All metric computations
│   ├── pipeline.py          # Evaluation pipeline
│   └── analysis.py          # Statistical analysis
└── scripts/
    ├── generate_synthetic_dataset.py
    ├── run_evaluation.py
    └── analyze_results.py
```

## Usage Example

```bash
# 1. Generate degraded dataset
python scripts/generate_synthetic_dataset.py \
  --input data/coco/annotations.json \
  --images data/coco/images \
  --output data/synthetic_evaluation/degraded \
  --profiles all

# 2. Run correction on all profiles
python scripts/run_evaluation.py \
  --degraded data/synthetic_evaluation/degraded \
  --images data/coco/images \
  --output data/synthetic_evaluation/corrected \
  --configs configs/correction_sweep.yaml

# 3. Analyze results
python scripts/analyze_results.py \
  --ground-truth data/coco/annotations.json \
  --degraded data/synthetic_evaluation/degraded \
  --corrected data/synthetic_evaluation/corrected \
  --output results/evaluation_report.html
```
