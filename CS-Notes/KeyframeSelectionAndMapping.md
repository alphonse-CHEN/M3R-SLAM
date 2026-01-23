# Keyframe Selection and Mapping in MASt3R-SLAM

This document explains the keyframe selection strategy and map storage architecture in MASt3R-SLAM. 

## Table of Contents
- [Keyframe Selection Strategy](#keyframe-selection-strategy)
- [Map Storage Architecture](#map-storage-architecture)
- [Data Structures](#data-structures)
- [Pointmap Update Strategies](#pointmap-update-strategies)
- [Backend Optimization](#backend-optimization)

---

## Keyframe Selection Strategy

### Tracking:  Single Keyframe Comparison

During real-time tracking, each new frame is compared against **only the immediate past keyframe** (the most recent one). This is implemented in `mast3r_slam/tracker. py`:

```python
def track(self, frame: Frame):
    keyframe = self.keyframes. last_keyframe()  # Only the last keyframe
    
    idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
        self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
    )
```

### Keyframe Selection Decision

A new keyframe is created based on two metrics: 
1. **Match Fraction (`match_frac_k`)**: Ratio of valid matches to total pixels
2. **Unique Fraction (`unique_frac_f`)**: Ratio of unique matched pixels

```python
# Keyframe selection criteria
n_valid = valid_kf. sum()
match_frac_k = n_valid / valid_kf.numel()
unique_frac_f = torch.unique(idx_f2k[valid_match_k[: , 0]]).shape[0] / valid_kf.numel()

# New keyframe if either metric drops below threshold
new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]
```

**Interpretation**: When the overlap between the current frame and the last keyframe becomes insufficient (too few matches or too many unique/new regions), a new keyframe is created.

### Backend:  Multiple Keyframes for Optimization

While tracking uses only 1 keyframe, the **backend optimization** considers more keyframes for graph construction:

```python
# In main.py - backend graph construction
n_consec = 1  # Number of consecutive previous keyframes
for j in range(min(n_consec, idx)):
    kf_idx. append(idx - 1 - j)

# Plus loop closure candidates from retrieval database
retrieval_inds = retrieval_database.update(
    frame,
    add_after_query=True,
    k=config["retrieval"]["k"],           # Number of retrieved candidates
    min_thresh=config["retrieval"]["min_thresh"],
)
kf_idx += retrieval_inds
```

| Component | Keyframes Used | Purpose |
|-----------|----------------|---------|
| **Tracking** | 1 (immediate past) | Real-time pose estimation |
| **Keyframe Selection** | 1 (immediate past) | Decide if new keyframe needed |
| **Backend Optimization** | 1 consecutive + k retrieved | Global consistency, loop closure |

---

## Map Storage Architecture

### Overview

MASt3R-SLAM uses **dense per-keyframe pointmaps** rather than a unified global point cloud. Each keyframe stores its own complete dense 3D reconstruction. 

### Why Per-Keyframe Storage?

1. **Memory Efficient**: No need for global map fusion
2. **Easy Updates**:  Pointmaps can be updated during optimization without global recomputation
3. **Multi-Process Access**: Shared memory tensors allow tracker and backend to operate concurrently
4. **Scale Handling**:  Sim3 poses allow monocular scale recovery per-keyframe

### Coordinate Frames

- **`X_canon`**: 3D points stored in **camera (canonical) frame**
- **World coordinates**:  Obtained via `T_WC. act(X_canon)` transformation
- **Pose**:  Stored as `Sim3` (translation + rotation + scale)

---

## Data Structures

### Frame Class (`mast3r_slam/frame.py`)

Each frame/keyframe contains: 

```python
@dataclasses.dataclass
class Frame:
    frame_id: int                           # Dataset frame index
    img:  torch.Tensor                       # RGB image [3, H, W]
    img_shape: torch.Tensor                 # Processed image dimensions
    img_true_shape: torch. Tensor            # Original image dimensions
    uimg: torch.Tensor                      # Unprocessed image [H, W, 3] (CPU)
    T_WC: lietorch.Sim3                     # Camera pose (world-from-camera, Sim3)
    X_canon: Optional[torch.Tensor]         # Dense pointmap [H*W, 3] in camera frame
    C:  Optional[torch. Tensor]               # Per-point confidence [H*W, 1]
    feat: Optional[torch. Tensor]            # MASt3R feature descriptors
    pos: Optional[torch. Tensor]             # Patch positions for features
    N:  int                                  # Number of pointmap updates
    N_updates: int                          # Total update count
    K: Optional[torch.Tensor]               # Camera intrinsics (if calibrated)
```

### SharedKeyframes Class

Pre-allocated GPU tensors in shared memory for multi-process access: 

```python
class SharedKeyframes:
    def __init__(self, manager, h, w, buffer=512, dtype=torch.float32, device="cuda"):
        # Pre-allocate for up to 512 keyframes
        self. X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, .. .).share_memory_()
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()
        self.feat = torch.zeros(buffer, 1, num_patches, 1024, .. .).share_memory_()
        self.pos = torch.zeros(buffer, 1, num_patches, 2, .. .).share_memory_()
        # ... additional fields
```

### Memory Layout Summary

| Field | Shape | Description |
|-------|-------|-------------|
| `X` | `[buffer, H*W, 3]` | 3D pointmaps (camera frame) |
| `C` | `[buffer, H*W, 1]` | Confidence values |
| `T_WC` | `[buffer, 1, 8]` | Sim3 poses (7 DoF + 1 scale) |
| `img` | `[buffer, 3, H, W]` | RGB images |
| `feat` | `[buffer, 1, N_patches, 1024]` | MASt3R features |
| `pos` | `[buffer, 1, N_patches, 2]` | Patch positions |

---

## Pointmap Update Strategies

When a frame is revisited or re-matched, pointmaps can be updated using various strategies (configured via `config["tracking"]["filtering_mode"]`):

### Available Modes

| Mode | Description |
|------|-------------|
| `first` | Keep the first pointmap, ignore updates |
| `recent` | Always use the most recent pointmap |
| `best_score` | Keep pointmap with highest confidence score |
| `indep_conf` | Per-point:  keep higher confidence value |
| `weighted_pointmap` | Weighted average in Cartesian space |
| `weighted_spherical` | Weighted average in spherical coordinates (default) |

### Spherical Averaging (Default)

```python
def update_pointmap(self, X:  torch.Tensor, C: torch. Tensor):
    # Convert to spherical coordinates (r, φ, θ)
    spherical1 = cartesian_to_spherical(self.X_canon)
    spherical2 = cartesian_to_spherical(X)
    
    # Confidence-weighted average
    spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)
    
    # Convert back to Cartesian
    self.X_canon = spherical_to_cartesian(spherical)
    self.C = self.C + C
    self.N += 1
```

**Why spherical? ** Averaging in spherical coordinates better preserves ray directions for monocular depth estimation. 

---

## Backend Optimization

### Factor Graph Construction

The backend builds a factor graph connecting keyframes: 

```python
class FactorGraph: 
    def __init__(self, model, frames:  SharedKeyframes, K=None, device="cuda"):
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)  # Source keyframe indices
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)  # Target keyframe indices
        self.idx_ii2jj = ...   # Pixel correspondences i→j
        self. idx_jj2ii = ...  # Pixel correspondences j→i
        self.valid_match_j = ...  # Valid match masks
        self.Q_ii2jj = ...  # Match quality scores
```

### Global Optimization

The optimization jointly refines: 
- **Keyframe poses** (`T_WC` as Sim3)
- **Pointmap depths** (along rays)

Using CUDA kernels (`mast3r_slam/backend/src/gn_kernels.cu`) for efficient Gauss-Newton optimization. 

---

## Reconstruction Export

When saving the final reconstruction, all keyframe pointmaps are aggregated: 

```python
for i in range(len(keyframes)):
    keyframe = keyframes[i]
    # Transform to world coordinates
    pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
    color = (keyframe.uimg. cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
    # Filter by confidence
    valid = keyframe.get_average_conf() > c_conf_threshold
    pointclouds.append(pW[valid])
    colors.append(color[valid])

# Save as PLY
pointclouds = np.concatenate(pointclouds, axis=0)
save_ply(savedir / filename, pointclouds, colors)
```

---

## Configuration Parameters

Key parameters in `config.yaml`:

```yaml
tracking:
  match_frac_thresh: 0.7    # Threshold for new keyframe creation
  filtering_mode: "weighted_spherical"  # Pointmap update strategy
  filtering_score:  "median"  # Score function for best_score mode

retrieval:
  k: 5                      # Number of retrieved keyframes for loop closure
  min_thresh: 0.3           # Minimum similarity threshold

local_opt:
  window_size: 10           # Optimization window size
  min_match_frac: 0.1       # Minimum match fraction for factor creation
```

---

## References

- [MASt3R Paper](https://arxiv.org/abs/2412.12392)
- [Project Page](https://edexheim.github.io/mast3r-slam/)
- Source files: 
  - `mast3r_slam/tracker.py` - Tracking and keyframe selection
  - `mast3r_slam/frame.py` - Frame and SharedKeyframes data structures
  - `mast3r_slam/global_opt.py` - Factor graph and optimization
  - `main.py` - Main loop and backend coordination