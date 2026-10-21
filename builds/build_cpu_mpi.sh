#!/usr/bin/env bash
#SBATCH --job-name=build_athenak_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=build_athenak_mpi_%j.log
##SBATCH --partition=<partition>      # uncomment and set for your cluster
##SBATCH --account=<account>
# =============================================================================
# build_cpu_mpi.sh
# Location: builds/build_cpu_mpi.sh
#
# Builds the CPU + MPI AthenaK binary for the kh_radiative_cooling problem.
# Runs either directly (bash builds/build_cpu_mpi.sh ...) or as a Slurm batch
# job (sbatch builds/build_cpu_mpi.sh ...; the #SBATCH lines above apply only
# then, and command-line flags of sbatch override them).
#
#   double precision (default) : builds/hr_build_mpi/src/athena      (used by run_3d.sh --backend mpi)
#   --single                   : builds/hr_build_mpi_sp/src/athena
#
# By default the build is incremental: only changed sources are recompiled.
# CMake is reconfigured from scratch (cache wiped) with --clean, or
# automatically when the cached problem, compiler, precision or CPU tuning
# differ from the ones requested.
#
# The MPI compiler wrapper decides which MPI library the binary links, and jobs
# must be launched with the mpirun/srun of that same MPI. Locally that is the
# system OpenMPI (/usr/bin/mpic++, launch with /usr/bin/mpirun), not
# miniconda's MPI, which comes first on PATH. Under Slurm the default is the
# mpicxx on PATH, so load the compiler + MPI modules (--modules) first.
#
# Building inside a Slurm job compiles on a compute node, so --native tunes
# for the CPUs the simulation will run on (a login node may differ).
#
# Usage:
#   [sbatch|bash] builds/build_cpu_mpi.sh [--single] [--clean] [--native]
#       [--modules "gcc openmpi"] [--problem NAME] [--compiler MPICXX]
#       [--dir DIR] [-j N]
#   --native    tune for the build machine's CPU (-march=native)
#   --modules   environment modules to load first (space-separated)
#   --problem   pgen name (default kh_radiative_cooling)
#   --compiler  MPI C++ wrapper (default: /usr/bin/mpic++ locally, mpicxx on
#               PATH under Slurm)
#   --dir       build directory (default hr_build_mpi, or hr_build_mpi_sp with --single)
#   -j N        parallel compile jobs (default: Slurm's cpus-per-task, else nproc)
#
# Example (cluster):
#   sbatch --partition=cpu builds/build_cpu_mpi.sh --native --modules "gcc/12 openmpi/4.1"
# =============================================================================

set -euo pipefail

# Under sbatch, BASH_SOURCE is Slurm's spool copy of this script; recover the
# original path from the job record, else assume submission from the repo root.
SCRIPT="${BASH_SOURCE[0]}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    ORIG="$( { scontrol show job "${SLURM_JOB_ID}" 2>/dev/null || true; } \
            | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')"
    if [[ -n "${ORIG}" && -f "${ORIG}" ]]; then
        SCRIPT="${ORIG}"
    else
        SCRIPT="${SLURM_SUBMIT_DIR}/builds/build_cpu_mpi.sh"
    fi
fi
BUILDS_DIR="$(cd "$(dirname "${SCRIPT}")" && pwd)"
ATHENAK_DIR="$(cd "${BUILDS_DIR}/../athenak" && pwd)"

SINGLE=OFF
CLEAN=0
NATIVE=OFF
MODULES=""
PROBLEM=kh_radiative_cooling
COMPILER=""
BUILD_DIR=""
JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --single)   SINGLE=ON; shift ;;
        --clean)    CLEAN=1; shift ;;
        --native)   NATIVE=ON; shift ;;
        --modules)  MODULES=$2; shift 2 ;;
        --problem)  PROBLEM=$2; shift 2 ;;
        --compiler) COMPILER=$2; shift 2 ;;
        --dir)      BUILD_DIR=$2; shift 2 ;;
        -j)         JOBS=$2; shift 2 ;;
        -h|--help)  sed -n '/^# =====/,/^# =====/p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1 (use --help)" >&2; exit 1 ;;
    esac
done

if [[ -z "${BUILD_DIR}" && ${SINGLE} == ON ]]; then
    BUILD_DIR="${BUILDS_DIR}/hr_build_mpi_sp"
elif [[ -z "${BUILD_DIR}" ]]; then
    BUILD_DIR="${BUILDS_DIR}/hr_build_mpi"
elif [[ "${BUILD_DIR}" != /* ]]; then
    BUILD_DIR="${BUILDS_DIR}/${BUILD_DIR}"
fi

# ---------------------------------------------------------------------------
# Environment modules (batch shells often lack the `module` function)
# ---------------------------------------------------------------------------
if [[ -n "${MODULES}" ]]; then
    if ! type module >/dev/null 2>&1; then
        for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
            if [[ -f "${init}" ]]; then
                set +u; source "${init}"; set -u
                break
            fi
        done
    fi
    set +u
    # shellcheck disable=SC2086
    module load ${MODULES}
    set -u
fi

# ---------------------------------------------------------------------------
# MPI compiler wrapper (full path, so the cache check is exact)
# ---------------------------------------------------------------------------
if [[ -z "${COMPILER}" ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" && -x /usr/bin/mpic++ ]]; then
        COMPILER=/usr/bin/mpic++
    else
        COMPILER=mpicxx
    fi
fi
if ! CXX_PATH="$(command -v "${COMPILER}")"; then
    echo "ERROR: MPI compiler wrapper not found: ${COMPILER}" >&2
    exit 1
fi
MPI_BIN="$(dirname "${CXX_PATH}")"
CC_PATH=""
for cc in mpicc; do
    [[ -x "${MPI_BIN}/${cc}" ]] && CC_PATH="${MPI_BIN}/${cc}"
done
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MPIRUN_PATH="srun (with the same modules loaded)"
else
    MPIRUN_PATH="${MPI_BIN}/mpirun"
    [[ -x "${MPIRUN_PATH}" ]] || MPIRUN_PATH="(mpirun/srun of the same MPI)"
fi

# ---------------------------------------------------------------------------
# Configure (only when needed) and build
# ---------------------------------------------------------------------------
mkdir -p "${BUILD_DIR}"
CACHE="${BUILD_DIR}/CMakeCache.txt"

cached() {  # cached value of a CMake variable; unset booleans count as OFF
    local v; v="$(sed -n "s/^$1:[A-Z]*=//p" "${CACHE}" 2>/dev/null)"
    echo "${v:-OFF}"
}

if [[ -f "${CACHE}" && ${CLEAN} == 0 ]]; then
    if [[ "$(cached PROBLEM)" != "${PROBLEM}" \
       || "$(cached CMAKE_CXX_COMPILER)" != "${CXX_PATH}" \
       || "$(cached Athena_ENABLE_MPI)" != "ON" \
       || "$(cached Athena_SINGLE_PRECISION)" != "${SINGLE}" \
       || "$(cached Kokkos_ARCH_NATIVE)" != "${NATIVE}" ]]; then
        echo "Cached configuration differs from the requested one; reconfiguring."
        CLEAN=1
    fi
fi

echo "========================================================================"
echo " AthenaK CPU + MPI build"
echo "========================================================================"
echo " Problem        : ${PROBLEM}"
echo " Precision      : $([[ ${SINGLE} == ON ]] && echo single || echo double)"
echo " CPU tuning     : $([[ ${NATIVE} == ON ]] && echo '-march=native' || echo generic)"
echo " MPI compiler   : ${CXX_PATH} ($("${CXX_PATH}" --version 2>/dev/null | head -1))"
echo " Launch with    : ${MPIRUN_PATH}"
echo " Build dir      : ${BUILD_DIR}"
echo " Compile jobs   : ${JOBS}"
echo " Modules        : ${MODULES:-(none)}"
echo " Slurm job      : ${SLURM_JOB_ID:-(not under Slurm)}$([[ -n ${SLURM_JOB_ID:-} ]] && echo " on $(hostname)" || true)"
echo " Mode           : $([[ ${CLEAN} == 1 || ! -f ${CACHE} ]] && echo 'clean configure + build' || echo incremental)"
echo "========================================================================"

if [[ ${CLEAN} == 1 ]]; then
    rm -rf "${CACHE}" "${BUILD_DIR}/CMakeFiles"
fi

if [[ ! -f "${CACHE}" ]]; then
    cmake -S "${ATHENAK_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPROBLEM="${PROBLEM}" \
        -DAthena_ENABLE_MPI=ON \
        -DAthena_SINGLE_PRECISION="${SINGLE}" \
        -DKokkos_ARCH_NATIVE="${NATIVE}" \
        -DCMAKE_CXX_COMPILER="${CXX_PATH}" \
        ${CC_PATH:+-DCMAKE_C_COMPILER="${CC_PATH}"}
fi

cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo ""
echo "Built: ${BUILD_DIR}/src/athena"
echo "Run with the matching launcher, e.g. MPIRUN=${MPIRUN_PATH} shell_scripts/run_3d.sh N --backend mpi"
