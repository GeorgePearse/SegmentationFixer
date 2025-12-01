# Mask Refinement Algorithms Comparison

## Overview
Multiple algorithmic approaches to find true object boundaries from imperfect masks. Each has different strengths and use cases.

---

## 1. Edge Snapping (Current - Rust Implementation)

**How it works:**
- Detect edges using Canny/Sobel edge detection
- For each polygon vertex, search within radius for nearest edge pixel
- Optionally smooth the result

**Parameters:**
- `snap_distance`: Search radius in pixels (5-50px)
- `smooth_iterations`: Laplacian smoothing passes (0-5)
- `edge_method`: 'canny', 'sobel', 'both'

**Strengths:**
- Very fast (implemented in Rust)
- Works well for clean, high-contrast edges
- Predictable behavior
- Low memory usage

**Weaknesses:**
- Sensitive to edge detection parameters
- Can snap to wrong edges in textured regions
- Doesn't consider region properties (color, texture)
- May produce jagged results without smoothing

**Best for:**
- Objects with clear, high-contrast boundaries
- When speed is critical
- Post-processing SAM or other model outputs

**Current Variations:**
- Conservative (5px): Minimal adjustment, preserves detail
- Moderate (12px): Balanced approach
- Aggressive (20px): Finds distant edges
- Max Search (35px): Maximum search radius
- Ultra (50px): For very poor initial masks

---

## 2. GrabCut

**How it works:**
- Graph cut algorithm that models foreground/background as Gaussian Mixture Models (GMM)
- Iteratively refines boundary based on color similarity
- Uses polygon to initialize foreground/background regions

**Parameters:**
- `iterations`: Number of GrabCut iterations (3-10)
- `margin`: Border width for probable fg/bg regions (10-20px)

**Strengths:**
- Considers color/texture coherence
- Good for objects with distinct color from background
- Robust to initial mask quality
- Finds natural boundaries

**Weaknesses:**
- Slower than edge snapping
- Can fail on similar fg/bg colors
- Requires good initialization
- May merge/split incorrectly with complex backgrounds

**Best for:**
- Objects with uniform color/texture
- Clear foreground/background separation
- Portrait segmentation, product photos
- When initial mask is roughly correct

---

## 3. Active Contours (Snakes)

**How it works:**
- Evolves curve to minimize energy function
- Energy combines: internal (smoothness) + external (edges)
- Converges to object boundary

**Parameters:**
- `alpha`: Continuity weight (0.01-0.05)
- `beta`: Curvature weight (5-20)
- `gamma`: Step size (0.001-0.01)
- `iterations`: Evolution steps (100-500)

**Strengths:**
- Smooth, continuous boundaries
- Good for regular shapes
- Handles gradual boundaries well
- Theoretically well-founded

**Weaknesses:**
- Can get stuck in local minima
- Sensitive to initialization
- Slow convergence
- May leak through weak boundaries

**Best for:**
- Smooth organic shapes (cells, fruits)
- Medical imaging
- When smoothness is important
- Objects with soft boundaries

---

## 4. Watershed Segmentation

**How it works:**
- Treats gradient magnitude as topographic surface
- Floods from markers (seeds) to find boundaries
- Natural region boundaries

**Parameters:**
- `edge_threshold`: Gradient threshold for markers (20-50)

**Strengths:**
- Finds natural region boundaries
- Good for touching/overlapping objects
- Respects object topology
- Fast

**Weaknesses:**
- Over-segmentation in textured regions
- Sensitive to noise
- Requires good marker placement
- May create many small regions

**Best for:**
- Separating touching objects
- Cell segmentation
- Objects with clear gradient boundaries
- When topology matters

---

## 5. Superpixel Refinement

**How it works:**
- Segments image into perceptually uniform regions (superpixels)
- Selects superpixels based on overlap with initial mask
- Boundary follows superpixel edges

**Parameters:**
- `n_segments`: Number of superpixels (50-500)
- `compactness`: Shape regularity (10-40)
- `overlap_threshold`: Minimum overlap to include (0.3-0.7)

**Strengths:**
- Respects image structure
- Natural-looking boundaries
- Computationally efficient
- Good for textured objects

**Weaknesses:**
- Boundary restricted to superpixel edges
- Can be blocky with coarse superpixels
- Sensitive to superpixel parameters
- May include/exclude wrong regions

**Best for:**
- Natural images with texture
- When object boundaries align with color/texture changes
- Semantic segmentation refinement
- Quick prototyping

---

## 6. Morphological Operations

**How it works:**
- Applies mathematical morphology (erosion, dilation, opening, closing)
- Smooths boundaries, fills holes, removes noise

**Parameters:**
- `operation`: 'close', 'open', 'gradient'
- `kernel_size`: Structuring element size (3-21px)
- `iterations`: Number of applications (1-5)

**Strengths:**
- Very fast
- Predictable
- Good for cleaning up masks
- No image needed (works on mask alone)

**Weaknesses:**
- Doesn't use image information
- Can remove thin structures
- May merge nearby objects
- Fixed-size operations

**Best for:**
- Cleaning up noisy masks
- Filling small holes
- Smoothing boundaries
- Post-processing any method

**Operations:**
- **Closing**: Fill gaps, connect nearby regions
- **Opening**: Remove small objects, smooth boundaries
- **Gradient**: Boundary extraction

---

## 7. Convex Hull

**How it works:**
- Computes smallest convex polygon containing all points
- Forces boundary to be convex

**Strengths:**
- Extremely fast
- Simple, robust
- Guaranteed convex result

**Weaknesses:**
- Loses all concave features
- Over-simplifies most objects
- No refinement, just simplification

**Best for:**
- Objects that should be convex (balls, heads)
- Bounding region estimation
- Removing concave artifacts
- Quick approximation

---

## 8. Threshold-Based

**How it works:**
- Applies intensity thresholding within polygon region
- Finds objects based on brightness/color

**Parameters:**
- `method`: 'otsu', 'adaptive_mean', 'adaptive_gaussian'
- `margin`: Context around mask (20-50px)

**Strengths:**
- Fast
- Good for high-contrast objects
- Automatic threshold selection (Otsu)
- Works well with lighting variations (adaptive)

**Weaknesses:**
- Assumes intensity differences
- Fails with similar fg/bg intensities
- Sensitive to lighting
- May split/merge incorrectly

**Best for:**
- High-contrast objects (text, logos)
- Objects on uniform backgrounds
- Microscopy, document images
- Binary objects

---

## Comparison Matrix

| Algorithm | Speed | Edge Quality | Color Aware | Texture Robust | Use Case |
|-----------|-------|--------------|-------------|----------------|----------|
| Edge Snap | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ❌ | ❌ | General, fast |
| GrabCut | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ | ✅ | Color distinct |
| Active Contours | ⚡⚡ | ⭐⭐⭐⭐⭐ | ❌ | ❌ | Smooth shapes |
| Watershed | ⚡⚡⚡⚡ | ⭐⭐⭐ | ❌ | ❌ | Touching objects |
| Superpixel | ⚡⚡⚡⚡ | ⭐⭐⭐ | ✅ | ✅ | Natural images |
| Morphological | ⚡⚡⚡⚡⚡ | ⭐⭐ | ❌ | ❌ | Cleanup |
| Convex Hull | ⚡⚡⚡⚡⚡ | ⭐ | ❌ | ❌ | Convex objects |
| Threshold | ⚡⚡⚡⚡ | ⭐⭐ | ❌ | ❌ | High contrast |

---

## Recommended Combinations

### For Natural Photos
```python
pipeline = [
    ('superpixel', {'n_segments': 200}),
    ('edge_snap', {'snap_dist': 12, 'smooth': 2}),
]
```

### For Medical Images
```python
pipeline = [
    ('threshold', {'method': 'otsu'}),
    ('morphological', {'operation': 'close'}),
    ('active_contour', {'iterations': 200}),
]
```

### For Product Photos
```python
pipeline = [
    ('grabcut', {'iterations': 5}),
    ('edge_snap', {'snap_dist': 8, 'smooth': 3}),
]
```

### For Noisy/Poor Masks
```python
pipeline = [
    ('morphological', {'operation': 'close', 'kernel': 7}),
    ('edge_snap', {'snap_dist': 35, 'smooth': 4}),
    ('morphological', {'operation': 'open', 'kernel': 3}),
]
```

---

## Integration Strategy

### Phase 1: Pure Edge Snapping (Current)
✅ Rust implementation  
✅ Multiple distance variations  
✅ Grid UI for selection  

### Phase 2: Hybrid System (Recommended)
1. Keep edge snapping variations (fast, Rust)
2. Add 3-4 Python algorithms as additional options
3. Show all in grid (e.g., 9 edge snap + 4 advanced = 13 total)

**Suggested additions:**
- GrabCut (thorough)
- Superpixel (fine)
- Morphological Close
- Active Contours (tight)

### Phase 3: Intelligent Selection
- Auto-select best algorithm per image based on:
  - Edge strength histogram
  - Color variance in fg/bg
  - Initial mask quality
  - Object category
- Still show grid for manual override

### Phase 4: Cascading Refinement
- Apply multiple algorithms in sequence
- Each refines output of previous
- Example: Morphological → GrabCut → Edge Snap → Smooth

---

## Implementation Roadmap

### Immediate (Increase edge snap tolerance)
- [x] Increase snap distances to 5-50px
- [x] Add more variation options
- [x] Update variation names

### Short-term (Add Python algorithms)
- [ ] Integrate advanced_refinement.py with backend
- [ ] Create Python server endpoint for algorithm variations
- [ ] Update frontend to handle mixed Rust/Python variations
- [ ] Add algorithm type badges to variation cards

### Medium-term (Optimization)
- [ ] Profile performance of each algorithm
- [ ] Implement caching for expensive operations
- [ ] Parallelize algorithm execution
- [ ] Add progress indicators for slow algorithms

### Long-term (Intelligence)
- [ ] Build algorithm recommendation model
- [ ] A/B test different combinations
- [ ] Collect user preferences for learning
- [ ] Auto-select based on image analysis

---

## Performance Expectations

### Edge Snapping (Rust)
- Time: ~1-5ms per polygon
- Memory: ~10MB per image
- Scalability: Excellent (can process 1000s/sec)

### GrabCut
- Time: ~100-500ms per polygon  
- Memory: ~50MB per image
- Scalability: Good (10-100/sec)

### Active Contours
- Time: ~200-1000ms per polygon
- Memory: ~20MB per image
- Scalability: Moderate (5-20/sec)

### Superpixel
- Time: ~50-200ms per polygon
- Memory: ~100MB per image
- Scalability: Good (20-50/sec)

### Morphological
- Time: ~5-20ms per polygon
- Memory: ~5MB per image
- Scalability: Excellent (100s/sec)

**Total for all methods:** ~0.5-2 seconds per polygon

**Strategy:** Run edge snapping first (fast), show results immediately. Run Python algorithms in background, update grid as they complete.
