# Keyframe Detection and Matching Pipeline

This document describes how keyframes are detected and how matching works in MASt3R-SLAM.

---

## Table of Contents
1. [Keyframe Detection](#keyframe-detection)
2. [MASt3R Outputs](#mast3r-outputs)
3. [Matching Pipeline](#matching-pipeline)
4. [Notation: X11, X21, D11, D21](#notation-x11-x21-d11-d21)
5. [Backend Processing](#backend-processing)

---

## Keyframe Detection

### Initialization (First Frame)
- The first frame is automatically made a keyframe
- Performs monocular MASt3R inference to get initial pointmap
- Adds to keyframe list and queues for global optimization

### Detection Criteria
Keyframes are detected in `tracker.track()` based on **two overlap metrics**:

```python
match_frac_k = n_valid / valid_kf.numel()
unique_frac_f = torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]
```

| Metric | Description |
|--------|-------------|
| `match_frac_k` | Fraction of valid matches between current frame and last keyframe |
| `unique_frac_f` | Fraction of unique keyframe pixels that are matched (coverage measure) |

**Decision Rule**: If **either metric falls below 0.333** (33.3%), a new keyframe is created.

> **Note**: The threshold `0.333` is configured in `config/base.yaml`:
> ```yaml
> tracking:
>   match_frac_thresh: 0.333
> ```
> And accessed via `self.cfg["match_frac_thresh"]`

This ensures keyframes are added when:
- The scene changes significantly
- Camera moves substantially
- Overlap with previous keyframe becomes insufficient

---

### Understanding `unique_frac_f` (Unique Pixels)

#### What is `idx_f2k`?

`idx_f2k` is the **matching index array** where:
- For each pixel in the **current frame (f)**, it stores which pixel in the **keyframe (k)** it matches to
- Shape: `(H*W,)` — one match index per pixel in frame f
- Value: linear pixel index in keyframe k (0 to H*W-1)

#### The Problem It Detects

**Multiple pixels in the current frame can match to the SAME pixel in the keyframe.**

Example with a 4-pixel image:
```
Current frame pixels:  [0, 1, 2, 3]
idx_f2k (matches to):  [5, 5, 5, 7]  ← pixels 0,1,2 all match to keyframe pixel 5
```

This happens when:
- Camera moves closer → current frame "zooms in" on part of keyframe
- Large viewpoint change → many-to-one correspondences
- Scene content changed significantly

#### The Computation

```python
torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0]
```

1. `valid_match_k[:, 0]` — boolean mask of valid matches
2. `idx_f2k[valid_match_k[:, 0]]` — only keep valid match indices
3. `torch.unique(...)` — remove duplicate indices (keyframe pixels matched multiple times)
4. `.shape[0]` — count how many **unique** keyframe pixels are covered

#### Concrete Example

```
Total pixels: 1000
idx_f2k for valid matches: [5, 5, 5, 7, 10, 10, 20, 21, 22, ...]
After torch.unique:        [5, 7, 10, 20, 21, 22, ...]  → 500 unique values

unique_frac_f = 500 / 1000 = 0.5 (50%)
```

#### Why This Matters

| Metric | What It Measures |
|--------|------------------|
| `match_frac_k` | How many matches are valid (quality) |
| `unique_frac_f` | How much of the keyframe is still visible (coverage) |

If `unique_frac_f` is low:
- Current frame sees only a small part of keyframe
- Or significant viewpoint change causing many-to-one matches
- **Time for a new keyframe!**

### Valid Match Filtering
Matches must pass multiple confidence thresholds:
- **C_conf**: Canonical confidence > 0.0 (both frames)
- **Q_conf**: Descriptor confidence > 1.5
- **min_match_frac**: At least 5% valid matches required for tracking

See [Understanding Confidence Values](#understanding-confidence-values) for detailed explanation.

---

## MASt3R Outputs

MASt3R does **NOT** directly output pixel correspondences. The neural network (encoder + decoder) outputs **per-pixel predictions**:

| Output | Symbol | Description |
|--------|--------|-------------|
| 3D Points | X | Dense pointmap/depth |
| Point Confidence | C | 3D point position quality scores |
| Dense Descriptors | D | Feature vectors per pixel |
| Descriptor Confidence | Q | Feature descriptor quality scores |

### Key Insight
**Matches are computed FROM the pointmaps, not BY the model directly.**

This is fundamentally different from methods like LoFTR or SuperGlue that directly predict correspondences.

---

## Understanding Confidence Values

MASt3R predicts **two separate confidence values** for each pixel:

### 1. Point Confidence (C) - 3D Geometry Quality

**What MASt3R Outputs:**
```python
res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)  # Raw C from network
```

The confidence is computed via exponential transform:
```python
def reg_dense_conf(x, mode):
    mode, vmin, vmax = mode  # e.g., ('exp', 1, inf)
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
```

**What It Measures:** How confident the model is about the **3D point position/depth**.

### 2. Descriptor Confidence (Q) - Feature Quality

**What MASt3R Outputs:**
```python
res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)  # Q from network
```

**What It Measures:** How confident the model is about the **descriptor/feature vector** for matching.

### 3. Canonical Confidence (Cf, Ck) - Accumulated Over Time

The term **"canonical"** refers to confidence **accumulated over multiple observations** of the same point.

**Computation in Frame Updates:**
```python
# Default mode: weighted_pointmap
def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
    self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
    self.C = self.C + C   # Accumulate confidence
    self.N += 1

def get_average_conf(self):
    return self.C / self.N    # Average over N updates
```

When the same point is observed from multiple frames:
- Raw confidence **C** accumulates: `self.C = self.C + C`
- **Canonical confidence** = `self.C / self.N` (averaged over observations)

### Usage in Tracking

From `tracker.py`:
```python
valid_Cf = Cf > self.cfg["C_conf"]  # Canonical confidence check (0.0)
valid_Ck = Ck > self.cfg["C_conf"]  
valid_Q = Qk > self.cfg["Q_conf"]   # Descriptor confidence check (1.5)

valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
```

### Summary Table

| Variable | Source | Computation | Purpose |
|----------|--------|-------------|---------|
| **Q** (Qff, Qkf) | Direct from MASt3R | `res['desc_conf']` | Filter poor descriptor matches |
| **C** (raw) | Direct from MASt3R | `res['conf']` | Initial 3D point quality |
| **Cf, Ck** (canonical) | Accumulated in Frame | `self.C / self.N` | Filter points with poor geometry over time |

### Why Two Separate Confidences?

- **C_conf threshold (0.0)**: Filters points with poor 3D geometry estimation
- **Q_conf threshold (1.5)**: Filters matches with unreliable feature descriptors
- Using both ensures matches are good in **both geometry and appearance**

---

## Matching Pipeline

### Overview
```
Input Images
    ↓
ViT Encoder → Features (cached per frame)
    ↓
Decoder → Dense 3D points + descriptors + confidences
    ↓
Iterative Projection → Continuous optimization for correspondences
    ↓
Distance Filtering → Remove occluded/bad matches (<10cm)
    ↓
Descriptor Refinement → Local search with features (radius=3)
    ↓
Output: idx_i2j, valid_matches
```

### Step 1: Feature Encoding
```python
frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)
```
ViT encoder extracts features for each frame (cached for reuse).

### Step 2: Asymmetric Decoding
```python
res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
```

Produces:
- **Xii**: 3D points of frame i when observing itself
- **Xji**: 3D points of frame i when observed from frame j's viewpoint
- **Dii, Dji**: Corresponding dense descriptors
- **Qii, Qji**: Descriptor confidences

### Step 3: Iterative Projection Matching

The actual pixel correspondences are found via **Levenberg-Marquardt optimization**:

**Objective**: Minimize angular distance between normalized 3D points and interpolated rays

```
err = ray(u, v) - pts_3d_norm
cost = ||err||²
```

**Algorithm** (implemented in CUDA):
1. Initialize correspondence guesses (identity mapping or previous matches)
2. For each iteration (max 10):
   - Bilinearly interpolate ray field at current pixel position
   - Compute Jacobian using ray gradients: `J = [∂ray/∂u, ∂ray/∂v]`
   - Solve 2×2 linear system: `(J^T J + λI) δp = -J^T err`
   - Update pixel position: `(u,v) ← (u,v) + δp`
3. Check convergence

### Step 4: Distance Filtering
```python
dists2 = torch.linalg.norm(X11[..., p1[..., :], :] - X21, dim=-1)
valid_dists2 = (dists2 < 0.1)  # 10cm threshold
```
Removes matches where 3D points are too far apart (occlusions, outliers).

### Step 5: Descriptor Refinement
```python
mast3r_slam_backends.refine_matches(D11, D21, p1, radius=3, dilation_max=5)
```
- Searches in local neighborhoods using dense descriptors
- Computes dot product similarity: `score = D21 · D11`
- Refines matches with feature similarity

---

## Notation: X11, X21, D11, D21

The subscript notation indicates **which frame's pixels** and **from which viewpoint**:

### Notation: X[target_frame][viewing_frame]
- **First subscript**: Which frame's pixels/scene we're looking at
- **Second subscript**: From which frame's viewpoint those pixels are observed

### In Symmetric Matching
```python
Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
```

| Symbol | Description |
|--------|-------------|
| **Xii** | Frame i's pixels, observed from frame i's viewpoint (self-observation) |
| **Xji** | Frame i's pixels, observed from frame j's viewpoint (cross-view) |
| **Xjj** | Frame j's pixels, observed from frame j's viewpoint (self-observation) |
| **Xij** | Frame j's pixels, observed from frame i's viewpoint (cross-view) |

### After Concatenation
```python
X11 = torch.cat((Xii, Xjj), dim=0)  # Self-observations (reference ray field)
X21 = torch.cat((Xji, Xij), dim=0)  # Cross-observations (points to project)
```

### Why This Matters
- **X11**: Provides the canonical ray directions for each pixel (projection target)
- **X21**: Contains points that should correspond to pixels in X11 (to be projected)
- Finding where X21 projects onto X11 gives pixel correspondences

---

## Backend Processing

The `run_backend(states, keyframes)` function handles **global optimization** tasks:

### 1. Relocalization Mode
If the system lost tracking (`Mode.RELOC`):
- Attempts to relocalize against the retrieval database
- If successful, switches back to `Mode.TRACKING`

### 2. Global Optimization for Queued Keyframes

**a) Graph Construction:**
- Links to previous consecutive keyframe (immediate temporal neighbor)
- Queries retrieval database for similar keyframes (loop closure, k=3)

**b) Factor Graph:**
```python
factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])
```
- Adds pose constraints between keyframes
- Requires minimum 10% match fraction

**c) Global Bundle Adjustment:**
```python
if config["use_calib"]:
    factor_graph.solve_GN_calib()
else:
    factor_graph.solve_GN_rays()
```
- Runs Gauss-Newton optimization over all keyframe poses

### Why Called Every Frame?
Even though it's "backend", it runs in the main thread after each frame to:
- Process one queued keyframe per iteration (incremental optimization)
- Keep latency low by doing small chunks of work
- Allow real-time operation without blocking tracking

---

## Key Configuration Parameters

From `config/base.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `match_frac_thresh` | 0.333 | Keyframe threshold (lower = more keyframes) |
| `min_match_frac` | 0.05 | Minimum tracking match fraction |
| `C_conf` | 0.0 | Canonical confidence threshold |
| `Q_conf` | 1.5 | Quality confidence threshold |
| `max_iter` | 10 | Matching Gauss-Newton iterations |
| `dist_thresh` | 0.1 | 3D distance threshold (10cm) |
| `radius` | 3 | Descriptor search radius |
| `retrieval.k` | 3 | Number of similar keyframes to retrieve |

---

## Summary

MASt3R-SLAM uses a hybrid approach:
1. **Neural network** predicts dense 3D structure (not matches directly)
2. **Geometric optimization** finds correspondences via iterative projection
3. **Learned descriptors** refine matches locally
4. **Backend optimization** maintains global consistency via factor graph

This design leverages the best of both worlds: deep learning for robust 3D prediction and classical optimization for accurate correspondence estimation.
