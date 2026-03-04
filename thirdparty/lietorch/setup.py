from setuptools import setup
import os
import platform
import re
import subprocess
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))


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


# Platform-specific compiler flags
if platform.system() == "Windows":
    extra_cxx = ["/O2", "/DUSE_CUDA"]
    extra_nvcc = [
        "-O2", "-DUSE_CUDA",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-allow-unsupported-compiler",  # CUDA 12.6 host_config.h may reject newer MSVC
    ] + get_nvcc_gencode_flags()
else:
    extra_cxx = ["-O2"]
    extra_nvcc = ["-O2"]


setup(
    name="lietorch",
    version="0.3",
    description="Lie Groups for PyTorch",
    author="Zachary Teed",
    packages=["lietorch"],
    ext_modules=[
        CUDAExtension("lietorch_backends", 
            include_dirs=[
                os.path.join(ROOT, "lietorch/include"), 
                os.path.join(ROOT, "../eigen")],
            sources=[
                "lietorch/src/lietorch.cpp", 
                "lietorch/src/lietorch_gpu.cu",
                "lietorch/src/lietorch_cpu.cpp"],
            extra_compile_args={
                "cxx": extra_cxx, 
                "nvcc": extra_nvcc,
            }),

        CUDAExtension("lietorch_extras", 
            sources=[
                "lietorch/extras/altcorr_kernel.cu",
                "lietorch/extras/corr_index_kernel.cu",
                "lietorch/extras/se3_builder.cu",
                "lietorch/extras/se3_inplace_builder.cu",
                "lietorch/extras/se3_solver.cu",
                "lietorch/extras/extras.cpp",
            ],
            extra_compile_args={
                "cxx": extra_cxx, 
                "nvcc": extra_nvcc,
            }),
    ],
    cmdclass={ "build_ext": BuildExtension }
)
