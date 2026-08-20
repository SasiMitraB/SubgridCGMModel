#!/bin/bash
set -e

# Prefer CUDA 12 for Ada89 architecture and Kokkos compatibility
for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
    if [ -d "$cuda_dir/bin" ]; then
        export CUDA_ROOT="$cuda_dir"
        export CUDA_HOME="$cuda_dir"
        export PATH="$cuda_dir/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_dir/lib64:${LD_LIBRARY_PATH:-}"
        break
    fi
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$REPO_DIR/builds/hr_build_gpu"
ATHENAK_DIR="$REPO_DIR/athenak"

# Ensure nvcc_wrapper is executable and defaults to Ada89
chmod +x "$ATHENAK_DIR/kokkos/bin/nvcc_wrapper"
export NVCC_WRAPPER_DEFAULT_ARCH="sm_89"

# Create hr_build_gpu directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# If previous cmake test failed, remove cache to recheck compiler
if [ -f "CMakeCache.txt" ] && grep -q "CMAKE_CXX_COMPILER-NOTFOUND" CMakeCache.txt; then
    rm -rf CMakeCache.txt CMakeFiles
fi

echo "=== Configuring AthenaK for GPU (Ada89) in $BUILD_DIR ==="
cmake -S "$ATHENAK_DIR" -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=kh_radiative_cooling \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_ADA89=ON \
  -DCMAKE_CXX_COMPILER="$ATHENAK_DIR/kokkos/bin/nvcc_wrapper"

echo "=== Building AthenaK ==="
cmake --build . -j"$(nproc)"

echo "=== Build complete! Executable is at $BUILD_DIR/src/athena ==="
