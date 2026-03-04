from pathlib import Path
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import platform
import re
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
has_cuda = torch.cuda.is_available()


def get_cuda_version():
    """Detect CUDA version from nvcc to determine supported architectures."""
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return None


def get_nvcc_gencode_flags():
    """Build gencode flags based on detected CUDA version."""
    cuda_ver = get_cuda_version()
    archs = ["70", "75", "80", "86", "89", "90"]
    if cuda_ver and (cuda_ver[0] > 12 or (cuda_ver[0] == 12 and cuda_ver[1] >= 8)):
        archs.append("120")
    flags = []
    for a in archs:
        flags.append(f"-gencode=arch=compute_{a},code=sm_{a}")
    return flags


include_dirs = [
    os.path.join(ROOT, "mast3r_slam/backend/include"),
    os.path.join(ROOT, "thirdparty/eigen"),
]

sources = [
    "mast3r_slam/backend/src/gn.cpp",
]

# Platform-specific compiler flags
if platform.system() == "Windows":
    extra_compile_args = {
        "cores": ["j8"],
        "cxx": ["/O2", "/DUSE_CUDA"],
    }
else:
    extra_compile_args = {
        "cores": ["j8"],
        "cxx": ["-O3"],
    }

if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension

    sources.append("mast3r_slam/backend/src/gn_kernels.cu")
    sources.append("mast3r_slam/backend/src/matching_kernels.cu")

    nvcc_flags = ["-O3"] + get_nvcc_gencode_flags()
    if platform.system() == "Windows":
        nvcc_flags.append("-DUSE_CUDA")
        nvcc_flags.append("-allow-unsupported-compiler")
    extra_compile_args["nvcc"] = nvcc_flags

    ext_modules = [
        CUDAExtension(
            "mast3r_slam_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    print("CUDA not found, cannot compile backend!")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)