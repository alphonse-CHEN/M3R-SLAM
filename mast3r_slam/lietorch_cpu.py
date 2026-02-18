"""
Lietorch CPU workaround for Windows GPU kernel crashes.

On Windows + CUDA 12.x + Ada Lovelace GPUs, lietorch's custom CUDA kernels
(inv, exp, log, mul, act, etc.) crash with access violations. Since Lie group
operations work on tiny pose data (8 floats), running them on CPU is perfectly
fine and avoids the broken GPU kernels entirely.

Usage:
    import mast3r_slam.lietorch_cpu  # patches lietorch at import time

This module monkey-patches lietorch.LieGroup.apply_op so that all Lie group
operations transparently move data to CPU, compute, and move results back.
"""

import lietorch
import torch
from lietorch.groups import LieGroup

# Store the original apply_op
_original_apply_op = LieGroup.apply_op.__func__


@classmethod
def _cpu_apply_op(cls, op, x, y=None):
    """Route lietorch operations through CPU to avoid GPU kernel crashes."""
    device = x.device
    on_gpu = device.type == "cuda"

    if on_gpu:
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu() if y is not None else None
        result = _original_apply_op(cls, op, x_cpu, y_cpu)
        return result.to(device)
    else:
        return _original_apply_op(cls, op, x, y)


# Apply the patch
LieGroup.apply_op = _cpu_apply_op

# Verify it works
def _self_test():
    """Quick sanity check that the patch works."""
    T = lietorch.Sim3.Identity(1, device="cuda:0")
    Ti = T.inv()
    T2 = T * Ti
    p = torch.tensor([[1.0, 2.0, 3.0]], device="cuda:0")
    q = T.act(p)
    assert Ti.data.device.type == "cuda", "Result should be back on GPU"
    assert torch.allclose(q, p, atol=1e-6), "Identity should not change points"


if torch.cuda.is_available():
    try:
        _self_test()
    except Exception as e:
        import warnings
        warnings.warn(f"lietorch CPU workaround self-test failed: {e}")
