# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
import re

# Detect CUDA version to determine supported architectures
def get_cuda_version():
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        version_match = re.search(r'release (\d+)\.(\d+)', nvcc_output)
        if version_match:
            major, minor = map(int, version_match.groups())
            return (major, minor)
    except:
        pass
    return (12, 6)  # Default fallback

cuda_version = get_cuda_version()
print(f"Detected CUDA version: {cuda_version[0]}.{cuda_version[1]}")

# Compile for all RTX GPU architectures (3090, 4090, 5090)
# Support is conditional based on CUDA version
all_cuda_archs = [
    '-gencode', 'arch=compute_70,code=sm_70',  # V100
    '-gencode', 'arch=compute_75,code=sm_75',  # RTX 20 series
    '-gencode', 'arch=compute_80,code=sm_80',  # A100
    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 3090
    '-gencode', 'arch=compute_89,code=sm_89',  # RTX 4090
    '-gencode', 'arch=compute_90,code=sm_90',  # H100
]

# Add RTX 5090 support (sm_120) only for CUDA 12.8+
if cuda_version >= (12, 8):
    all_cuda_archs.extend(['-gencode', 'arch=compute_120,code=sm_120'])  # RTX 5090
    print("Added RTX 5090 (sm_120) support")
else:
    print(f"Skipping RTX 5090 (sm_120) support - requires CUDA 12.8+, found {cuda_version[0]}.{cuda_version[1]}")

import platform
# On Windows, MSVC + CUDA 12.6 has a std namespace ambiguity bug in
# torch/csrc/dynamo/compiled_autograd.h. Defining USE_CUDA triggers
# PyTorch's own Windows workaround guard in that header.
extra_cxx = ['-O3']
extra_nvcc = ['-O3','--ptxas-options=-v',"--use_fast_math"]
if platform.system() == 'Windows':
    extra_cxx.append('/DUSE_CUDA')
    extra_nvcc.append('-DUSE_CUDA')
    # CUDA 12.6 host_config.h may reject newer MSVC; allow override
    extra_nvcc.append('-allow-unsupported-compiler')

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=extra_nvcc+all_cuda_archs,
                    cxx=extra_cxx)
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
