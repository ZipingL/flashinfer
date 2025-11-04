# FlashInfer Custom Build - Blackwell SM 12.0 Support

## Overview

This is a custom build of FlashInfer with comprehensive NVIDIA Blackwell (SM 12.0) architecture support and CUDA 12.8 compatibility patches. It enables FlashInfer to run on next-generation NVIDIA GPUs that require `compute_120` and `sm_120a` compilation targets.

## Why This Custom Build?

### Problem Statement

The upstream FlashInfer repository (as of v0.5.1) has the following limitations:

1. **No SM 12.0 Support**: Missing compilation flags for Blackwell GPUs (GB200, B200, etc.)
2. **CUDA 12.x Incompatibility**: References unsupported `compute_103a` architecture that doesn't exist until CUDA 13.x
3. **Static Architecture Flags**: No runtime control over which SM 12.0 variants to compile

### What This Build Fixes

This custom build adds:

✅ **Dynamic SM 12.0 Architecture Detection**  
✅ **CUDA 12.8 Compatibility** (strips incompatible `compute_103a` references)  
✅ **Environment-Based Control** for SM 12.0 variants  
✅ **Backward Compatibility** with all upstream features  

## Key Differences from Upstream

### 1. Helper Functions for Dynamic Architecture Support

**File**: `flashinfer/jit/core.py` (lines 110-137)

```python
def _env_flag_is_true(name: str) -> bool:
    """Check if environment variable is set to true-like value"""
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _collect_arch_suffixes(major: int) -> list[str]:
    """Collect architecture suffixes for a given major version from compilation context"""
    suffixes = [
        f"{arch_major}{minor}"
        for arch_major, minor in current_compilation_context.TARGET_CUDA_ARCHS
        if arch_major == major
    ]
    # Special handling for SM 12.0 variants
    if major == 12:
        if _env_flag_is_true("FLASHINFER_ENABLE_SM120"):
            suffixes.append("120")
        if _env_flag_is_true("FLASHINFER_ENABLE_SM120A"):
            suffixes.append("120a")
    return sorted(set(suffixes))

def _make_nvcc_flags(suffixes: list[str], default_suffixes: list[str]) -> list[str]:
    """Generate NVCC flags from architecture suffixes"""
    effective_suffixes = suffixes if suffixes else default_suffixes
    gencodes = [
        f"-gencode=arch=compute_{suffix},code=sm_{suffix}" 
        for suffix in effective_suffixes
    ]
    return gencodes + common_nvcc_flags
```

**Why**: Allows runtime control of which Blackwell GPU variants to compile for, without hardcoding.

### 2. SM 12.0 Architecture Flags

**File**: `flashinfer/jit/core.py` (line 151)

```python
# Upstream (missing):
# No sm120a_nvcc_flags defined

# Custom Build:
sm120a_nvcc_flags = _make_nvcc_flags(_collect_arch_suffixes(12), ["120a"])
```

**Result**: Dynamically generates flags like:
```bash
-gencode=arch=compute_120a,code=sm_120a
-DFLASHINFER_ENABLE_FP8_E8M0
-DFLASHINFER_ENABLE_FP4_E2M1
```

### 3. CUDA 12.x Compatibility Patch

**File**: `flashinfer/jit/core.py` (lines 145-149)

```python
# Upstream (breaks on CUDA 12.8):
sm103a_nvcc_flags = ["-gencode=arch=compute_103a,code=sm_103a"] + common_nvcc_flags

# Custom Build:
# --- PATCH: disable unsupported architecture on CUDA <13 ---
sm103a_nvcc_flags: list[str] = []  # compute_103a not recognized until CUDA 13.x+
# --- END PATCH ---
```

**Why**: `compute_103a` doesn't exist in CUDA 12.x, causing compilation failures. This patch disables it for compatibility.

### 4. Ninja Build File Patching

**File**: `flashinfer/jit/core.py` (lines 292-301)

```python
def write_ninja(self) -> None:
    # ...existing code...
    content = generate_ninja_build_for_op(...)
    
    # --- PATCH: Strip unsupported compute_103a for CUDA <13 ---
    if "compute_103a" in content or "sm_103a" in content:
        filtered = []
        for line in content.splitlines():
            if "103a" not in line:
                filtered.append(line)
        content = "\n".join(filtered)
        print(f"[PATCH] Removed compute_103a/sm_103a from ninja for {self.name}")
    # --- END PATCH ---
    
    write_if_different(ninja_path, content)
```

**Why**: Even if we disable `sm103a_nvcc_flags`, the ninja build generator might still inject the flags. This ensures they're stripped from build files.

### 5. JIT Build Spec Patching

**File**: `flashinfer/jit/core.py` (lines 437-447)

```python
def build_jit_specs(specs: List[JitSpec], verbose: bool = False, skip_prebuilt: bool = True) -> None:
    # ...existing code...
    
    # --- PATCH: Remove unsupported compute_103a entries from ninja content ---
    cleaned_lines = []
    for l in lines:
        if "103a" in l or "compute_103a" in l or "sm_103a" in l:
            continue
        cleaned_lines.append(l)
    lines = cleaned_lines
    print("[PATCH] Stripped all compute_103a and sm_103a entries from JIT ninja build spec")
    # --- END PATCH ---
```

**Why**: Second layer of protection to ensure `compute_103a` doesn't leak into batch JIT builds.

## Environment Variables

This custom build introduces new environment variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `FLASHINFER_ENABLE_SM120` | Enable plain SM 12.0 support | `export FLASHINFER_ENABLE_SM120=1` |
| `FLASHINFER_ENABLE_SM120A` | Enable SM 12.0a variant (recommended for Blackwell) | `export FLASHINFER_ENABLE_SM120A=1` |
| `FLASHINFER_CUDA_ARCH_LIST` | Specify target architectures | `export FLASHINFER_CUDA_ARCH_LIST="12.0"` |
| `FLASHINFER_NVCC` | Path to CUDA 12.8+ nvcc compiler | `export FLASHINFER_NVCC=/usr/local/cuda-12.8/bin/nvcc` |

## Installation

### Prerequisites

- CUDA 12.8 or later (required for SM 12.0 support)
- Python 3.10+
- NVIDIA Blackwell GPU (GB200, B200, or compatible)

### Build from Source

```bash
# Clone this repository
git clone <your-fork-url>
cd flashinfer

# Set environment variables
export FLASHINFER_ENABLE_SM120A=1
export FLASHINFER_CUDA_ARCH_LIST="12.0"
export FLASHINFER_NVCC=/usr/local/cuda-12.8/bin/nvcc

# Install dependencies
pip install apache-tvm-ffi>=0.1,<0.2

# Build and install (editable mode for development)
pip install -e . --no-build-isolation

# OR build wheel for distribution
python -m build --wheel
pip install dist/flashinfer_python-0.5.1-*.whl
```

### Verify Installation

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1  # If using older flashinfer-cubin

python -c "
from flashinfer.jit import core
print('Version:', __import__('flashinfer').__version__)
print('SM 12.0a flags:', core.sm120a_nvcc_flags)
print('SM 10.3a flags:', core.sm103a_nvcc_flags)
"
```

Expected output:
```
Version: 0.5.1
SM 12.0a flags: ['-gencode=arch=compute_120a,code=sm_120a', '-DFLASHINFER_ENABLE_FP8_E8M0', '-DFLASHINFER_ENABLE_FP4_E2M1']
SM 10.3a flags: []
```

## Usage with vLLM

This custom build is compatible with vLLM (even though vLLM specifies `flashinfer-python==0.4.0`):

```bash
# Set environment variables
export FLASHINFER_ENABLE_SM120A=1
export FLASHINFER_CUDA_ARCH_LIST="12.0"
export FLASHINFER_NVCC=/usr/local/cuda-12.8/bin/nvcc
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Install vLLM (it will use your custom flashinfer build)
pip install vllm

# Run vLLM
python -m vllm.entrypoints.openai.api_server --model <model_name>
```

### Why vLLM Compatibility Works

- ✅ FlashInfer 0.5.1 is **backward compatible** with 0.4.0 APIs
- ✅ All function signatures remain unchanged
- ✅ Only adds new features (SM 12.0 support), doesn't remove anything
- ✅ Performance improvements over 0.4.0

## Architecture Support Matrix

| Architecture | Compute Capability | Status | Notes |
|--------------|-------------------|--------|-------|
| Ampere (A100) | SM 8.0 | ✅ Upstream | No changes |
| Ada (RTX 4090) | SM 8.9 | ✅ Upstream | Added in v0.5.1 |
| Hopper (H100) | SM 9.0 | ✅ Upstream | No changes |
| Blackwell (GB200) | SM 10.0a | ✅ Upstream | No changes |
| **Blackwell (B200)** | **SM 12.0a** | **✅ Custom** | **New in this build** |
| Rubin (R100?) | SM 10.3a | ⚠️ Disabled | Requires CUDA 13.x+ |

## Technical Deep Dive

### Why SM 10.3a is Disabled

CUDA 12.8's `nvcc` doesn't recognize `compute_103a`:

```bash
$ nvcc --generate-code=arch=compute_103a,code=sm_103a test.cu
nvcc fatal   : Value 'compute_103a' is not defined for option 'gpu-architecture'
```

This is a **forward-looking architecture** that will be supported in CUDA 13.x. Our patch ensures builds don't fail on current CUDA versions.

### JIT Compilation Flow with SM 12.0

```
1. User imports flashinfer
   └─> Checks for precompiled cubin (flashinfer-cubin)
       └─> SM 12.0 cubin not found (doesn't exist yet)

2. Falls back to JIT compilation
   └─> Reads environment: FLASHINFER_ENABLE_SM120A=1
   └─> _collect_arch_suffixes(12) returns ["120a"]
   └─> _make_nvcc_flags generates: -gencode=arch=compute_120a,code=sm_120a

3. Compiles kernel with CUDA 12.8 nvcc
   └─> Generates SM 12.0 optimized code
   └─> Caches result in ~/.cache/flashinfer/jit/

4. Subsequent runs use cached compilation
   └─> Fast startup after first JIT compile
```

### Patch Locations Summary

| Patch | File | Lines | Purpose |
|-------|------|-------|---------|
| Helper Functions | `core.py` | 110-137 | Dynamic arch detection |
| SM 12.0 Flags | `core.py` | 151 | Blackwell support |
| SM 10.3a Disable | `core.py` | 145-149 | CUDA 12.x compat |
| Ninja Write Patch | `core.py` | 292-301 | Strip 103a from build files |
| Ninja Spec Patch | `core.py` | 437-447 | Strip 103a from batch builds |

## Known Limitations

1. **No Precompiled Cubins**: SM 12.0 kernels are JIT-compiled on first run (slight startup delay)
2. **CUDA 12.8+ Required**: Earlier CUDA versions don't support `compute_120a`
3. **Development Status**: These patches are not yet in upstream FlashInfer

## Contributing

If you encounter issues or want to improve these patches:

1. Test thoroughly on Blackwell hardware
2. Document any new edge cases
3. Consider submitting patches upstream to FlashInfer

## Version Information

- **Base Version**: FlashInfer 0.5.1
- **Custom Patches**: Blackwell SM 12.0 + CUDA 12.x compatibility
- **CUDA Requirement**: 12.8 or later
- **Python Requirement**: 3.10+

## License

This custom build maintains the same license as upstream FlashInfer. See `LICENSE` for details.

## Acknowledgments

- Original FlashInfer team for the excellent library
- NVIDIA for Blackwell architecture documentation
- Community testing and feedback

---

**⚠️ Disclaimer**: This is a custom build with unofficial patches. Use in production at your own risk. Consider testing thoroughly before deployment.