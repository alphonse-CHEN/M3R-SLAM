# Local Installation Guide

## Environment Setup

### 1. Create Micromamba Environment

```bash
micromamba create -n sfm3r python=3.11 -y
```

### 2. Activate Environment

```bash
micromamba activate sfm3r
```

### 3. Configure pip to use Aliyun Mirror

This helps avoid SSL connection issues when downloading dependencies:

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
```

### 4. Install PyTorch Dependencies

Pre-install common dependencies from Aliyun mirror:

```bash
pip3 install filelock sympy typing-extensions numpy pillow jinja2 networkx fsspec mpmath
```

### 5. Install PyTorch, xformers, and torchvision

Install from PyTorch official index with CUDA 12.6 support:

```bash
pip3 install torch xformers torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Package Versions

- Python: 3.11
- PyTorch: 2.10.0+cu126
- torchvision: 0.25.0+cu126
- xformers: (compatible with CUDA 12.6)
- CUDA: 12.6

## Install Project Dependencies

### 6. Install curope (CUDA 2D RoPE kernel)

Separated from the dust3r/croco tree into `thirdparty/curope` for independent
compilation, debugging and testing.

Features:
- Auto-detects CUDA version for supported architectures
- RTX 3090 (sm_86), RTX 4090 (sm_89): Always supported
- RTX 5090 (sm_120): Only enabled for CUDA 12.8+
- Windows MSVC workaround via `USE_CUDA` define

```bash
pip install --no-build-isolation thirdparty/curope

# Test independently:
python thirdparty/curope/test_curope.py
```

### 8. Install MASt3R from thirdparty

```bash
pip install --no-build-isolation thirdparty/mast3r
```

### 9. Install LieTorch from thirdparty

LieTorch uses shared Eigen from `thirdparty/eigen`, so install without build isolation:

```bash
pip install --no-build-isolation thirdparty/lietorch
```

### 10. Install MASt3R-SLAM (this project)

```bash
pip install --no-build-isolation -e .
```

## Additional Packages

(To be added as installation progresses)

## Notes

- The repository has integrated third-party code directly (not as git submodules)
- `--no-build-isolation` is required for lietorch and the main project due to shared dependencies
- MASt3R doesn't need `-e` flag as it's not being actively developed
