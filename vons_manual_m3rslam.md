# Von's MASt3R-SLAM Manual

Personal reference for running MASt3R-SLAM on Windows with Rerun visualization.

---

## Environment

| Component | Version / Detail |
|-----------|-----------------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 4060 Laptop (8 GB VRAM, Ada Lovelace, compute 8.9) |
| Python | 3.11.14 (micromamba env `sfm3r`) |
| PyTorch | 2.10.0+cu126 |
| CUDA | 12.6 (nvcc V12.6.20) |
| lietorch | 0.3 (custom-built with Windows NVCC flags) |
| rerun-sdk | 0.29.2 |
| Model | `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` (~1.2 GB) |

### Activate environment

```powershell
micromamba activate sfm3r
```

---

## Running the Pipeline

### Basic run (headless, no visualization)

```powershell
python main.py --dataset data/normal-apt-tour.MOV --no-viz
```

### Run with Rerun live visualization + .rrd recording

```powershell
python main.py --dataset data/normal-apt-tour.MOV --rerun
```

### Run with calibration

```powershell
python main.py --dataset data/my-video.MOV --rerun --calib config/my_calib.yaml
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--dataset PATH` | Path to video file or dataset directory |
| `--config PATH` | YAML config (default: `config/base.yaml`) |
| `--rerun` | Enable Rerun visualization (spawns viewer + saves .rrd) |
| `--no-viz` | Disable all visualization (fastest, headless) |
| `--calib PATH` | Camera calibration YAML file |
| `--save-as NAME` | Custom output directory name (rarely needed) |

---

## Output Organization

Every run creates a **timestamped directory** under `logs/<dataset_name>/`:

```
logs/
└── normal-apt-tour/
    ├── 2026-02-18_131009/         ← run 1
    │   ├── normal-apt-tour.txt    ← camera trajectory (TUM format)
    │   ├── normal-apt-tour.ply    ← 3D reconstruction point cloud
    │   ├── normal-apt-tour.rrd    ← Rerun recording (if --rerun)
    │   └── keyframes/             ← keyframe images
    ├── 2026-02-18_133305/         ← run 2
    │   └── ...
    └── ...
```

- **Nothing is ever overwritten** — each run gets a unique `YYYY-MM-DD_HHMMSS` folder.
- Trajectory is saved in TUM format: `timestamp tx ty tz qx qy qz qw`
- Reconstruction `.ply` can be opened in MeshLab, CloudCompare, or Blender.
- The `.rrd` file can be replayed later in Rerun.

### Replay a past run in Rerun

```powershell
rerun logs/normal-apt-tour/2026-02-18_133305/normal-apt-tour.rrd
```

---

## GPU Auto-Adapt

The pipeline automatically detects GPU VRAM and adjusts the `subsample` rate to prevent OOM:

| VRAM | Subsample | Effective Framerate |
|------|-----------|-------------------|
| < 10 GB (e.g. RTX 4060 Laptop 8 GB) | 5 | Every 5th frame |
| 10–13 GB (e.g. RTX 4070 12 GB) | 3 | Every 3rd frame |
| 14–17 GB (e.g. RTX 4080 16 GB) | 2 | Every 2nd frame |
| 18+ GB (e.g. RTX 4090 24 GB) | as config | Full rate |

This overrides `config.dataset.subsample` when the configured value would cause OOM.
On the 8 GB RTX 4060 Laptop, the pipeline runs at ~1.2–1.5 FPS with subsample=5.

---

## What We Changed (and Why)

### 1. Lietorch CPU Workaround

**Problem:** All lietorch custom CUDA kernels (inv, exp, log, mul, act) crash with access violations on Windows + CUDA 12.x + Ada Lovelace GPUs.

**Solution:** Monkey-patch `LieGroup.apply_op` to route all Lie group operations through CPU. Since these operate on tiny pose data (8 floats per element), performance impact is zero.

**Files:**
- `mast3r_slam/lietorch_cpu.py` — The monkey-patch + self-test
- `mast3r_slam/__init__.py` — Auto-applies the patch on Windows at import time

### 2. Lietorch Build Fixes (Windows)

**Files:**
- `thirdparty/lietorch/setup.py` — Added `--expt-relaxed-constexpr`, `--expt-extended-lambda` NVCC flags, plus gencode flags for architectures 70–90 (and 120 for CUDA 12.8+)
- `thirdparty/lietorch/lietorch/include/common.h` — Removed `EIGEN_RUNTIME_NO_MALLOC` (causes access violations when Eigen needs temporaries in device code)
- `thirdparty/lietorch/.gitignore` — Added `*.so`, `*.pyd`, `*.dll` to avoid tracking binaries

### 3. Single-Thread Mode Support

**Problem:** The original code always called `mp.Manager()` and `mp.set_start_method("spawn")`, which fails or is unnecessary in single-thread mode.

**Solution:** Added `FakeManager` class that provides the same interface using plain Python objects (threading locks, lists, simple values) instead of multiprocessing shared memory.

**Files:**
- `mast3r_slam/multiprocess_utils.py` — Added `FakeManager` and `FakeValue` classes
- `main.py` — Uses `FakeManager` when `single_thread=True`

### 4. Rerun Visualization

**Problem:** The original project used `in3d` (OpenGL-based), which is abandoned and doesn't build on Windows.

**Solution:** Created a full Rerun-based visualizer that provides:
- 3D point cloud visualization (per-keyframe, confidence-filtered)
- Camera frustum display (current frame in green, keyframes in red)
- Keyframe connectivity graph (green edges)
- Current frame + latest keyframe image panels
- Blueprint layout: 3D Scene (3/4 width) + image panels (1/4 width)

**Key implementation detail:** Rerun's `rr.save()` **replaces** the viewer connection with a file sink. To get both live viewer AND .rrd file recording, we use `set_sinks()` with both `GrpcSink()` and `FileSink()` simultaneously.

**Files:**
- `mast3r_slam/rerun_viz.py` — `RerunVisualizer` class + `WindowMsg` dataclass
- `main.py` — Import chain (Rerun → in3d → dummy), `--rerun` flag, inline `viz.update()` in main loop

### 5. GPU Auto-Adapt

**Problem:** At subsample=1, the 8 GB GPU hits OOM around frame 74.

**Solution:** Auto-detect VRAM at startup and adjust subsample accordingly.

**File:** `main.py` (after `load_config`, before dataset loading)

### 6. Output Directory Consolidation

**Problem:** Trajectory, reconstruction, keyframes, and .rrd files were saved to different locations, and runs could overwrite each other.

**Solution:** Every run creates `logs/<dataset_name>/<YYYY-MM-DD_HHMMSS>/` and puts all outputs there.

**Files:**
- `main.py` — Creates `output_dir`, passes it to all save functions
- `mast3r_slam/evaluate.py` — Updated `prepare_savedir` to accept `run_id` parameter

### 7. Minor Fixes

- `mast3r_slam/frame.py` — Added `.copy()` to `torch.from_numpy(img["unnormalized_img"].copy())` to fix non-writable tensor from video decoder
- `thirdparty/mast3r/pyproject.toml` — Changed `include = ["mast3r", "dust3r"]` to `include = ["mast3r*", "dust3r*"]` to include subpackages
- `main.py` — Fixed `datetime.datetime.now()` string format (colons are illegal in Windows filenames)

---

## Files We Created

| File | Purpose |
|------|---------|
| `mast3r_slam/lietorch_cpu.py` | Lietorch CPU monkey-patch for Windows GPU kernel crashes |
| `mast3r_slam/rerun_viz.py` | Rerun-based SLAM visualizer (replaces in3d) |

## Files We Modified

| File | Change |
|------|--------|
| `main.py` | Rerun integration, GPU auto-adapt, single-thread support, output consolidation, datetime fix |
| `mast3r_slam/__init__.py` | Auto-apply lietorch CPU patch on Windows |
| `mast3r_slam/multiprocess_utils.py` | Added `FakeManager` / `FakeValue` for single-thread mode |
| `mast3r_slam/frame.py` | `.copy()` fix for non-writable numpy array |
| `mast3r_slam/evaluate.py` | `prepare_savedir` accepts optional `run_id` |
| `thirdparty/lietorch/setup.py` | Windows NVCC flags + gencode architectures |
| `thirdparty/lietorch/lietorch/include/common.h` | Removed `EIGEN_RUNTIME_NO_MALLOC` |
| `thirdparty/lietorch/.gitignore` | Exclude `*.so`, `*.pyd`, `*.dll` |
| `thirdparty/mast3r/pyproject.toml` | Wildcard package includes |

---

## Rerun Visualization Guide

### What you see in the viewer

| Panel | Content |
|-------|---------|
| **3D Scene** (left, 3/4 width) | Point clouds (colored by keyframe image), camera frustums, connectivity edges |
| **Current Frame** (top-right) | Live camera feed |
| **Keyframe** (bottom-right) | Latest selected keyframe image |

### Entity hierarchy in Rerun

```
/                          ← ViewCoordinates (RIGHT_HAND_Y_DOWN)
/world/
  current_camera/          ← Current camera pose (Transform3D)
    pinhole/               ← Camera frustum (green, Pinhole)
  keyframes/
    kf_0/                  ← Keyframe 0 pose
      pinhole/             ← Keyframe 0 frustum (red)
    kf_1/ ...
  pointclouds/
    kf_0/                  ← Keyframe 0 point cloud (Points3D)
    kf_1/ ...
  current_points/          ← Current frame depth-colored point cloud
  edges/                   ← Keyframe connectivity graph (LineStrips3D)
/images/
  current/                 ← Current RGB frame (Image)
  keyframe/                ← Latest keyframe RGB (Image)
```

### Rerun timeline

- Timeline name: `frame`
- Each frame index corresponds to a processed video frame (after subsampling)
- Use the timeline scrubber in Rerun to replay the SLAM session

### Tips

- **Zoom to fit:** Press `F` in the 3D view to auto-frame the scene
- **Scrub timeline:** Drag the timeline bar at the bottom to review earlier frames
- **Filter entities:** Click entities in the left panel to show/hide them
- **Confidence threshold:** Currently hardcoded at 1.5 in the visualizer. Points below this confidence are filtered out.

---

## Config Reference (`config/base.yaml`)

The default config runs in **single-thread mode** with **no calibration**:

```yaml
use_calib: False        # True if you have camera intrinsics
single_thread: True     # True = no multiprocessing (required on Windows)
dataset:
  subsample: 1          # Auto-adapted based on GPU VRAM
  img_downsample: 1     # Additional image downscaling
```

Key tuning parameters:
- `tracking.match_frac_thresh: 0.333` — If match fraction drops below this, a new keyframe is created
- `tracking.Q_conf: 1.5` — Confidence threshold for tracking
- `retrieval.k: 3` — Number of nearest keyframes to retrieve for loop closure
- `local_opt.use_cuda: True` — GPU-accelerated optimization (uses mast3r_slam_backends)

---

## Troubleshooting

### OOM on 8 GB GPU
The auto-adapt should handle this. If you still get OOM, increase subsample manually:
```powershell
python main.py --dataset data/video.MOV --config config/base.yaml --rerun
# Edit config/base.yaml: subsample: 8
```

### Rerun viewer shows empty scene
1. Make sure no stale `rerun.exe` processes are running (`taskkill /f /im rerun.exe`)
2. The visualizer automatically kills stale viewers before spawning a new one
3. If using `rr.save()` instead of `set_sinks()`, data goes to file only — this is the #1 gotcha

### Lietorch CUDA kernel crashes
The CPU workaround in `mast3r_slam/lietorch_cpu.py` is auto-applied on Windows. If you see Lie group errors, verify the patch loaded:
```python
python -c "import mast3r_slam; import lietorch; T = lietorch.Sim3.Identity(1, device='cuda:0'); print(T.inv().data)"
```

### Pipeline hangs or runs very slowly
- Check GPU utilization: `nvidia-smi -l 1`
- Normal FPS is ~1.2–1.5 on RTX 4060 Laptop with subsample=5
- The retrieval model loads on first keyframe — expect a brief pause

---

## Quick Reference

```powershell
# Headless run (fastest)
python main.py --dataset data/video.MOV --no-viz

# Live visualization
python main.py --dataset data/video.MOV --rerun

# Replay a recorded session
rerun logs/video/2026-02-18_133305/video.rrd

# View reconstruction
# Open the .ply file in MeshLab, CloudCompare, or Blender
```
