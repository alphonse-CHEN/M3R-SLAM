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

---

## Appendix A: Windows CUDA Compilation Issues & Workarounds

### A.1 `compute_120` Unsupported on CUDA 12.6

**Symptom:**

```
nvcc fatal   : Unsupported gpu architecture 'compute_120'
```

**Root Cause:**

The RTX 5090 uses SM 12.0 (`compute_120` / `sm_120`), but this architecture
is only supported starting from **CUDA 12.8**. Hardcoding `-gencode=arch=compute_120,code=sm_120`
in `setup.py` causes `nvcc` to fail on CUDA 12.6 and earlier.

**Fix:**

Auto-detect the installed CUDA version by parsing `nvcc --version` output, and
only add `compute_120` when CUDA >= 12.8:

```python
import re, subprocess

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return None

def get_nvcc_gencode_flags():
    cuda_ver = get_cuda_version()
    archs = ["70", "75", "80", "86", "89", "90"]
    if cuda_ver and (cuda_ver[0] > 12 or (cuda_ver[0] == 12 and cuda_ver[1] >= 8)):
        archs.append("120")
    return [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]
```

**Affected files:** `setup.py` (main project), `thirdparty/curope/setup.py`,
`thirdparty/lietorch/setup.py`.

---

### A.2 `std` Namespace Ambiguity (`error C2872`) on Windows MSVC

**Symptom:**

```
torch/csrc/dynamo/compiled_autograd.h(1135): error C2872: "std": 不明确的符号
  可能是"std"  (from Eigen/src/Core/arch/Default/BFloat16.h)
  或     "std"  (from compiled_autograd.h)
```

The error fires during NVCC compilation of any `.cu` file that transitively
includes both PyTorch headers and Eigen headers.

**Root Cause:**

In PyTorch 2.10, the header `torch/csrc/dynamo/compiled_autograd.h` contains:

```cpp
namespace torch::dynamo::autograd {
  using namespace torch::autograd;   // <-- pulls in torch::autograd::*
  ...
}
```

`torch::autograd` itself does `using namespace std;` in some paths. When Eigen
headers are also included, they define items in `::std` (e.g. `BFloat16`
numeric traits). MSVC then cannot resolve bare `std::` references inside the
`IValuePacker` template at line 1135 of `compiled_autograd.h` because it sees
two candidates for the `std` symbol.

This does **not** occur on Linux/GCC because GCC handles nested `using namespace`
differently in template instantiation.

**Attempted fixes that did NOT work:**

1. **Reordering `#include` directives** — moving Eigen before/after PyTorch
   headers. The conflict is structural, not order-dependent.

2. **`#define TORCH_DISABLE_DYNAMO_COMPILE_AUTOGRAD`** — this macro does not
   exist and has no effect.

3. **Patching the header with `sed`** — replacing `std::` with `::std::` in
   `compiled_autograd.h`. This is fragile, breaks on PyTorch upgrades, and
   requires maintaining a `.bak` file.

4. **`#define TORCH_STABLE_ONLY`** — does not guard the problematic code path.

**Fix that works — define `USE_CUDA`:**

PyTorch's `compiled_autograd.h` has a platform guard:

```cpp
#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))
  // Skips the problematic if-constexpr chain that triggers C2872
#endif
```

When `USE_CUDA` is defined, PyTorch takes an alternative code path on Windows
that avoids the ambiguous `std` resolution. The fix is to add the define to
**both** CXX and NVCC compiler flags in `setup.py`:

```python
import platform

if platform.system() == "Windows":
    extra_cxx  = ["/O2", "/DUSE_CUDA"]
    extra_nvcc = ["-O2", "-DUSE_CUDA"] + get_nvcc_gencode_flags()
else:
    extra_cxx  = ["-O3"]
    extra_nvcc = ["-O3"]
```

**Affected files:** `thirdparty/curope/setup.py`, `thirdparty/lietorch/setup.py`,
`setup.py` (main project) — every `setup.py` that builds a `CUDAExtension`
on Windows with PyTorch 2.10.

---

### A.3 Summary of `setup.py` Modifications

Each `setup.py` that compiles CUDA extensions needed two changes on Windows:

| Change | Why |
|--------|-----|
| Auto-detect CUDA version for gencode flags | Avoid `compute_120` on CUDA < 12.8 |
| Add `/DUSE_CUDA` (CXX) and `-DUSE_CUDA` (NVCC) | Bypass `std` ambiguity in PyTorch's `compiled_autograd.h` |

These changes are **no-ops on Linux** (the `platform.system()` guard ensures
only the original flags are used on non-Windows platforms).

---

### A.4 Linker Error: `mutable_data_ptr<long>` Unresolved (`LNK2001` / `LNK1120`)

**Symptom:**

```
gn_kernels.obj : error LNK2001: unresolved external symbol
  "public: long * __cdecl at::TensorBase::mutable_data_ptr<long>(void)const"
```

**Root Cause:**

Windows uses the **LLP64** data model:

| Type | Linux (LP64) | Windows (LLP64) |
|------|-------------|-----------------|
| `long` | 64-bit | **32-bit** |
| `long long` | 64-bit | 64-bit |
| `int64_t` | `long` | `long long` |

PyTorch's `torch.long` maps to `int64_t`. On Linux, `int64_t` == `long`, so
`packed_accessor32<long,...>` works. On Windows, `int64_t` == `long long`, so
`packed_accessor32<long,...>` instantiates a **different** template specialization
that PyTorch never exports — causing `LNK2001`.

**Fix:**

Replace all `long` template arguments with `int64_t` in CUDA kernel code:

```cpp
// Before (works on Linux, fails on Windows):
ii.packed_accessor32<long,1,torch::RestrictPtrTraits>()
auto ii_acc = ii_cpu.accessor<long,1>();
typedef std::vector<std::vector<long>> graph_t;

// After (works on both):
ii.packed_accessor32<int64_t,1,torch::RestrictPtrTraits>()
auto ii_acc = ii_cpu.accessor<int64_t,1>();
typedef std::vector<std::vector<int64_t>> graph_t;
```

**Affected files:** `mast3r_slam/backend/src/gn_kernels.cu`,
`mast3r_slam/backend/src/matching_kernels.cu`.
