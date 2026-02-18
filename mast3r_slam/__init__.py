"""
MASt3R-SLAM package initialization.

On Windows, lietorch CUDA kernels crash due to GPU compatibility issues.
We patch lietorch to run Lie group operations on CPU (they're tiny ops on
pose data, so there's zero performance impact).
"""

import platform
import torch

if platform.system() == "Windows" and torch.cuda.is_available():
    import mast3r_slam.lietorch_cpu  # noqa: F401 — patches lietorch at import time
