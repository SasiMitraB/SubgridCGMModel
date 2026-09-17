#!/bin/bash
set -e

BUILDS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATHENAK_DIR="$BUILDS_DIR/../athenak"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}========================================${NC}\n"
}

# Parse command line arguments
REBUILD_HR=true
REBUILD_HR_MPI=true
REBUILD_HR_GPU=true
REBUILD_SUBGRID=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --hr-only)
            REBUILD_HR_MPI=false
            REBUILD_HR_GPU=false
            REBUILD_SUBGRID=false
            shift
            ;;
        --mpi-only)
            REBUILD_HR=false
            REBUILD_HR_GPU=false
            REBUILD_SUBGRID=false
            shift
            ;;
        --gpu-only)
            REBUILD_HR=false
            REBUILD_HR_MPI=false
            REBUILD_SUBGRID=false
            shift
            ;;
        --subgrid-only)
            REBUILD_HR=false
            REBUILD_HR_MPI=false
            REBUILD_HR_GPU=false
            shift
            ;;
        --no-hr)
            REBUILD_HR=false
            shift
            ;;
        --no-mpi)
            REBUILD_HR_MPI=false
            shift
            ;;
        --no-gpu)
            REBUILD_HR_GPU=false
            shift
            ;;
        --no-subgrid)
            REBUILD_SUBGRID=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --hr-only              Rebuild only hr_build"
            echo "  --mpi-only             Rebuild only hr_build_mpi"
            echo "  --gpu-only             Rebuild only hr_build_gpu"
            echo "  --subgrid-only         Rebuild only subgrid_model"
            echo "  --no-hr                Skip hr_build"
            echo "  --no-mpi               Skip hr_build_mpi"
            echo "  --no-gpu               Skip hr_build_gpu"
            echo "  --no-subgrid           Skip subgrid_model"
            echo "  --help                 Show this help message"
            echo ""
            echo "Default: Rebuild all four builds"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper function to rebuild a CMake project
rebuild_cmake_project() {
    local name=$1
    local build_dir=$2
    shift 2
    local cmake_args=("$@")

    print_header "Rebuilding $name"

    if [ ! -d "$build_dir" ]; then
        print_error "Build directory not found: $build_dir"
        return 1
    fi

    cd "$build_dir"

    # Clear CMake cache
    print_status "Clearing CMake cache..."
    rm -rf CMakeCache.txt CMakeFiles

    # Configure
    print_status "Configuring with CMake..."
    if ! cmake -S "$ATHENAK_DIR" -B . "${cmake_args[@]}"; then
        print_error "CMake configuration failed for $name"
        return 1
    fi

    # Build
    print_status "Building (using $(nproc) cores)..."
    if ! cmake --build . -j"$(nproc)"; then
        print_error "Build failed for $name"
        return 1
    fi

    print_status "$name built successfully!"
    return 0
}

# Build hr_build (standard)
if [ "$REBUILD_HR" = true ]; then
    rebuild_cmake_project "hr_build" \
        "$BUILDS_DIR/hr_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPROBLEM=kh_radiative_cooling || exit 1
fi

# Build hr_build_mpi
if [ "$REBUILD_HR_MPI" = true ]; then
    rebuild_cmake_project "hr_build_mpi" \
        "$BUILDS_DIR/hr_build_mpi" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPROBLEM=kh_radiative_cooling \
        -DAthena_ENABLE_MPI=ON \
        -DCMAKE_CXX_COMPILER=/usr/bin/mpic++ \
        -DCMAKE_C_COMPILER=/usr/bin/mpicc || exit 1
fi

# Build hr_build_gpu
if [ "$REBUILD_HR_GPU" = true ]; then
    print_header "Rebuilding hr_build_gpu"

    BUILD_DIR="$BUILDS_DIR/hr_build_gpu"

    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found: $BUILD_DIR"
        exit 1
    fi

    # Setup CUDA environment
    for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
        if [ -d "$cuda_dir/bin" ]; then
            export CUDA_ROOT="$cuda_dir"
            export CUDA_HOME="$cuda_dir"
            export PATH="$cuda_dir/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_dir/lib64:${LD_LIBRARY_PATH:-}"
            print_status "Using CUDA from $cuda_dir"
            break
        fi
    done

    cd "$BUILD_DIR"

    # Clear CMake cache
    print_status "Clearing CMake cache..."
    rm -rf CMakeCache.txt CMakeFiles

    # Setup nvcc_wrapper
    chmod +x "$ATHENAK_DIR/kokkos/bin/nvcc_wrapper"
    export NVCC_WRAPPER_DEFAULT_ARCH="sm_89"

    # Configure
    print_status "Configuring for GPU (Ada89)..."
    if ! cmake -S "$ATHENAK_DIR" -B . \
        -DCMAKE_BUILD_TYPE=Release \
        -DPROBLEM=kh_radiative_cooling \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ARCH_ADA89=ON \
        -DCMAKE_CXX_COMPILER="$ATHENAK_DIR/kokkos/bin/nvcc_wrapper"; then
        print_error "CMake configuration failed for hr_build_gpu"
        exit 1
    fi

    # Build
    print_status "Building (using $(nproc) cores)..."
    if ! cmake --build . -j"$(nproc)"; then
        print_error "Build failed for hr_build_gpu"
        exit 1
    fi

    print_status "hr_build_gpu built successfully!"
fi

# Build subgrid_model
if [ "$REBUILD_SUBGRID" = true ]; then
    print_header "Rebuilding subgrid_model"

    BUILD_DIR="$BUILDS_DIR/subgrid_model"

    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found: $BUILD_DIR"
        exit 1
    fi

    # Use system Python (miniconda)
    PYTHON_BIN="$(which python3)"
    PYTHON_ROOT="$(python3 -c 'import sys; print(sys.prefix)')"

    print_status "Using Python: $PYTHON_BIN"
    print_status "Python prefix: $PYTHON_ROOT"

    cd "$BUILD_DIR"

    # Clear CMake cache
    print_status "Clearing CMake cache..."
    rm -rf CMakeCache.txt CMakeFiles

    # Configure
    print_status "Configuring with system Python..."
    if ! cmake -S "$ATHENAK_DIR" -B . \
        -DCMAKE_BUILD_TYPE=Release \
        -DPROBLEM=subgrid \
        -DPython_EXECUTABLE="$PYTHON_BIN" \
        -DPython_ROOT_DIR="$PYTHON_ROOT"; then
        print_error "CMake configuration failed for subgrid_model"
        exit 1
    fi

    # Build
    print_status "Building (using $(nproc) cores)..."
    if ! cmake --build . -j"$(nproc)"; then
        print_error "Build failed for subgrid_model"
        exit 1
    fi

    print_status "subgrid_model built successfully!"
fi

# Summary
print_header "Build Summary"
echo -e "${GREEN}All selected builds completed successfully!${NC}"
echo ""
echo "Built executables:"
if [ "$REBUILD_HR" = true ]; then
    echo "  ✓ hr_build: $BUILDS_DIR/hr_build/src/athena"
fi
if [ "$REBUILD_HR_MPI" = true ]; then
    echo "  ✓ hr_build_mpi: $BUILDS_DIR/hr_build_mpi/src/athena"
fi
if [ "$REBUILD_HR_GPU" = true ]; then
    echo "  ✓ hr_build_gpu: $BUILDS_DIR/hr_build_gpu/src/athena"
fi
if [ "$REBUILD_SUBGRID" = true ]; then
    echo "  ✓ subgrid_model: $BUILDS_DIR/subgrid_model/src/athena"
fi
