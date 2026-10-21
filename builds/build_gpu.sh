#!/usr/bin/env bash
# =============================================================================
# build_gpu.sh
# Location: builds/build_gpu.sh
#
# Builds the CUDA (Kokkos) AthenaK binary for the kh_radiative_cooling problem.
#
#   double precision (default) : builds/hr_build_gpu/src/athena      (used by run_3d.sh)
#   --single                   : builds/hr_build_gpu_sp/src/athena   (halves GPU memory)
#
# By default the build is incremental: only changed sources are recompiled.
# CMake is reconfigured from scratch (cache wiped) with --clean, or
# automatically when the cached problem, GPU architecture or precision differ
# from the ones requested.
#
# Usage:
#   builds/build_gpu.sh [--single] [--clean] [--problem NAME] [--arch ARCH]
#                       [--dir DIR] [-j N]
#   --problem  pgen name (default kh_radiative_cooling)
#   --arch     Kokkos GPU arch (default ADA89 = RTX 4000 Ada; e.g. AMPERE80 for
#              A100, AMPERE86, HOPPER90, VOLTA70)
#   --dir      build directory (default hr_build_gpu, or hr_build_gpu_sp with --single)
#   -j N       parallel compile jobs (default nproc)
# =============================================================================

set -euo pipefail

BUILDS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATHENAK_DIR="$(cd "${BUILDS_DIR}/../athenak" && pwd)"

SINGLE=OFF
CLEAN=0
PROBLEM=kh_radiative_cooling
ARCH=ADA89
BUILD_DIR=""
JOBS="$(nproc)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --single)  SINGLE=ON; shift ;;
        --clean)   CLEAN=1; shift ;;
        --problem) PROBLEM=$2; shift 2 ;;
        --arch)    ARCH=$2; shift 2 ;;
        --dir)     BUILD_DIR=$2; shift 2 ;;
        -j)        JOBS=$2; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1 (use --help)" >&2; exit 1 ;;
    esac
done

if [[ -z "${BUILD_DIR}" && ${SINGLE} == ON ]]; then
    BUILD_DIR="${BUILDS_DIR}/hr_build_gpu_sp"
elif [[ -z "${BUILD_DIR}" ]]; then
    BUILD_DIR="${BUILDS_DIR}/hr_build_gpu"
elif [[ "${BUILD_DIR}" != /* ]]; then
    BUILD_DIR="${BUILDS_DIR}/${BUILD_DIR}"
fi

# Kokkos arch -> nvcc sm_XX for nvcc_wrapper
case ${ARCH} in
    ADA89)    SM=sm_89 ;;
    HOPPER90) SM=sm_90 ;;
    AMPERE80) SM=sm_80 ;;
    AMPERE86) SM=sm_86 ;;
    AMPERE87) SM=sm_87 ;;
    TURING75) SM=sm_75 ;;
    VOLTA70)  SM=sm_70 ;;
    *) echo "ERROR: unsupported --arch ${ARCH}" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# CUDA toolkit
# ---------------------------------------------------------------------------
CUDA_FOUND=""
for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -d "${cuda_dir}/bin" ]]; then
        export CUDA_ROOT="${cuda_dir}" CUDA_HOME="${cuda_dir}"
        export PATH="${cuda_dir}/bin:${PATH}"
        export LD_LIBRARY_PATH="${cuda_dir}/lib64:${LD_LIBRARY_PATH:-}"
        CUDA_FOUND="${cuda_dir}"
        break
    fi
done
if [[ -z "${CUDA_FOUND}" ]]; then
    echo "ERROR: no CUDA toolkit found under /usr/local/cuda*" >&2
    exit 1
fi

NVCC_WRAPPER="${ATHENAK_DIR}/kokkos/bin/nvcc_wrapper"
chmod +x "${NVCC_WRAPPER}"
export NVCC_WRAPPER_DEFAULT_ARCH="${SM}"

# ---------------------------------------------------------------------------
# Configure (only when needed) and build
# ---------------------------------------------------------------------------
mkdir -p "${BUILD_DIR}"
CACHE="${BUILD_DIR}/CMakeCache.txt"

cached() { sed -n "s/^$1:[A-Z]*=//p" "${CACHE}" 2>/dev/null; }

if [[ -f "${CACHE}" && ${CLEAN} == 0 ]]; then
    if [[ "$(cached PROBLEM)" != "${PROBLEM}" \
       || "$(cached Kokkos_ARCH_${ARCH})" != "ON" \
       || "$(cached Athena_SINGLE_PRECISION)" != "${SINGLE}" ]]; then
        echo "Cached configuration differs from the requested one; reconfiguring."
        CLEAN=1
    fi
fi

echo "========================================================================"
echo " AthenaK GPU build"
echo "========================================================================"
echo " Problem        : ${PROBLEM}"
echo " Precision      : $([[ ${SINGLE} == ON ]] && echo single || echo double)"
echo " GPU arch       : ${ARCH} (${SM})"
echo " CUDA           : ${CUDA_FOUND}"
echo " Build dir      : ${BUILD_DIR}"
echo " Mode           : $([[ ${CLEAN} == 1 || ! -f ${CACHE} ]] && echo 'clean configure + build' || echo incremental)"
echo "========================================================================"

if [[ ${CLEAN} == 1 ]]; then
    rm -rf "${CACHE}" "${BUILD_DIR}/CMakeFiles"
fi

if [[ ! -f "${CACHE}" ]]; then
    cmake -S "${ATHENAK_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPROBLEM="${PROBLEM}" \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ARCH_${ARCH}=ON \
        -DAthena_SINGLE_PRECISION="${SINGLE}" \
        -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}"
fi

cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo ""
echo "Built: ${BUILD_DIR}/src/athena"
