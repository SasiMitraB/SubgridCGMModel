#!/usr/bin/env bash
# =============================================================================
# run_3d.sh
# Location: shell_scripts/run_3d.sh
#
# Runs the 3D Kelvin-Helmholtz + ISM cooling setup (kh_radiative_cooling,
# iprob=1 analytic shear layer) on a fixed box:
#   x1, x3 (short) : [-5, 5] pc      x2 (long, shear-normal) : [-20, 20] pc
# The resolution is set by N = cells along the short dimension; the long
# dimension gets 4N so cells are cubic (dx = 10/N pc). The base deck is
# <PROJECT_ROOT>/athinputs/kh_fid3D_64_cool.athinput; only <mesh>/<meshblock> nx* (and
# optionally tlim) are patched.
#
# Run name: kh3d_<N>x<4N>x<N>, e.g.
#   N    run name           dx [pc]   matches /data/sasi run
#   4    kh3d_4x16x4        2.5       fid3D_16_cool
#   8    kh3d_8x32x8        1.25      fid3D_32_cool
#   16   kh3d_16x64x16      0.625     fid3D_64_cool
#   32   kh3d_32x128x32     0.3125    fid3D_128_cool
#
# MeshBlocks: 2 x 4 x 2 blocks of (N/2) x N x (N/2) when N is even and N >= 8,
# otherwise 1 x 4 x 1 blocks of N x N x N. Override with MB="a b c".
#
# Usage:
#   shell_scripts/run_3d.sh [N] [--backend cpu|mpi|gpu] [--tlim T] [--dry-run]
#   --dry-run prints the configuration, time and storage estimates, then exits.
#   N defaults to 32, backend to gpu (mpi uses one rank per MeshBlock).
#   MPI_NP overrides the rank count; CUDA_VISIBLE_DEVICES picks the GPU.
#   MPIRUN overrides the launcher. It defaults to /usr/bin/mpirun because
#   hr_build_mpi links the system OpenMPI; miniconda's mpirun would launch
#   MPI_NP independent serial copies instead of one parallel job.
#
# Output directory : ${OUT_ROOT:-/data/sasi/simulation_outputs}/kh3d_<N>x<4N>x<N>/
# Log file         : <output_dir>/kh3d_<N>x<4N>x<N>.log
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
N=128
BACKEND=gpu
TLIM=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)    BACKEND=$2; shift 2 ;;
        --tlim)       TLIM=$2; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    sed -n '2,36p' "$0"; exit 0 ;;
        *[!0-9]*|'')  echo "ERROR: unknown argument: $1 (use --help)" >&2; exit 1 ;;
        *)            N=$1; shift ;;
    esac
done

if (( N < 4 )); then
    echo "ERROR: N must be >= 4 (got ${N})" >&2
    exit 1
fi

NX=(${N} $((4 * N)) ${N})
if [[ -n "${MB:-}" ]]; then
    read -r -a MB <<< "${MB}"
elif (( N % 2 == 0 && N >= 8 )); then
    MB=($((N / 2)) ${N} $((N / 2)))
else
    MB=(${N} ${N} ${N})
fi
for d in 0 1 2; do
    if (( NX[d] % MB[d] != 0 )); then
        echo "ERROR: MeshBlock ${MB[*]} does not divide mesh ${NX[*]}" >&2
        exit 1
    fi
done
N_MB=$(( (NX[0]/MB[0]) * (NX[1]/MB[1]) * (NX[2]/MB[2]) ))

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_ATHINPUT="${PROJECT_ROOT}/athinputs/kh_fid3D_64_cool.athinput"

case ${BACKEND} in
    cpu) BUILD_SRC="${PROJECT_ROOT}/builds/hr_build/src" ;;
    mpi) BUILD_SRC="${PROJECT_ROOT}/builds/hr_build_mpi/src" ;;
    gpu) BUILD_SRC="${PROJECT_ROOT}/builds/hr_build_gpu/src" ;;
    *) echo "ERROR: --backend must be cpu, mpi or gpu" >&2; exit 1 ;;
esac
ATHENA="${BUILD_SRC}/athena"

TAG="kh3d_${NX[0]}x${NX[1]}x${NX[2]}"
OUT_ROOT="${OUT_ROOT:-/data/sasi/simulation_outputs}"
OUT_DIR="${OUT_ROOT}/${TAG}"
ATHINPUT="${OUT_DIR}/${TAG}.athinput"
if (( DRY_RUN )); then
    # Never touch OUT_DIR on a dry run (it may hold a finished or running sim)
    ATHINPUT="$(mktemp --suffix=.athinput)"
    trap 'rm -f "${ATHINPUT}"' EXIT
fi
LOG_FILE="${OUT_DIR}/${TAG}.log"

MPI_NP="${MPI_NP:-${N_MB}}"
MPIRUN="${MPIRUN:-/usr/bin/mpirun}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -x "${ATHENA}" ]]; then
    echo "ERROR: Athena binary not found or not executable: ${ATHENA}" >&2
    echo "       Build it with builds/rebuild_all.sh" >&2
    exit 1
fi

if [[ ! -f "${BASE_ATHINPUT}" ]]; then
    echo "ERROR: Base athinput not found: ${BASE_ATHINPUT}" >&2
    exit 1
fi

if (( MPI_NP > N_MB )) && [[ ${BACKEND} == mpi ]]; then
    echo "ERROR: MPI_NP=${MPI_NP} exceeds the number of MeshBlocks (${N_MB})" >&2
    exit 1
fi

(( DRY_RUN )) || mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Generate athinput for this resolution
# ---------------------------------------------------------------------------
python3 - "${BASE_ATHINPUT}" "${ATHINPUT}" "${NX[@]}" "${MB[@]}" "${TLIM}" <<'PY'
import sys, re, pathlib

src, out = sys.argv[1], pathlib.Path(sys.argv[2])
nx, mb, tlim = sys.argv[3:6], sys.argv[6:9], sys.argv[9]
patch = {'mesh': dict(zip(('nx1', 'nx2', 'nx3'), nx)),
         'meshblock': dict(zip(('nx1', 'nx2', 'nx3'), mb))}
if tlim:
    patch['time'] = {'tlim': tlim}

block, lines = None, []
for line in pathlib.Path(src).read_text().splitlines():
    m = re.match(r'\s*<(\w+)>', line)
    if m:
        block = m.group(1)
    else:
        for key, val in patch.get(block, {}).items():
            line = re.sub(rf'^(\s*{key}\s*=\s*)\S+', rf'\g<1>{val}', line)
    lines.append(line)

out.write_text('\n'.join(lines) + '\n')
print(f"Generated input file: {out}")
PY

# ---------------------------------------------------------------------------
# Display setup summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo " Athena 3D KH Simulation Configuration"
echo "========================================================================"
echo " Tag            : ${TAG}"
echo " Resolution     : ${NX[0]} x ${NX[1]} x ${NX[2]}"
echo " Meshblock Size : ${MB[0]} x ${MB[1]} x ${MB[2]} (${N_MB} MeshBlocks)"
echo " Backend        : ${BACKEND}$([[ ${BACKEND} == mpi ]] && echo " (${MPI_NP} ranks)")"
echo " tlim           : ${TLIM:-from base deck} Myr"
echo " Output Dir     : ${OUT_DIR}"
echo " Log File       : ${LOG_FILE}"
echo " Executable     : ${ATHENA}"

# ---------------------------------------------------------------------------
# Estimated wall time and storage. Empirical model calibrated on this machine
# (RTX 4000 Ada, 32 cores) from the kh3d_16/32/64 runs:
#   cycles  ~ 51 * N * tlim   (dt scales with dx = 10/N pc)
#   gpu     : zone-cycles/s ~ 5e7 * cells / (cells + 3.1e4)  (saturates)
#   mpi     : zone-cycles/s ~ 4.6e5 per rank
#   cpu     : zone-cycles/s ~ 6e5
# plus ~3 s startup/output overhead. Expect +-25%.
# Storage is exact for this deck:
#   bin : 24 B/cell (6 float32 prims) + 6104 B header, every <output2> dt
#   rst : 48 B/cell incl. ghost zones (6 float64) + 5003 B, every <output3> dt
# ---------------------------------------------------------------------------
mapfile -t EST < <(python3 - "${ATHINPUT}" "${N}" "${BACKEND}" "${MPI_NP}" "${OUT_DIR}" "${MB[@]}" <<'PY'
import os, sys, re, shutil, datetime
deck, n, backend, np_, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
mb = [int(x) for x in sys.argv[6:9]]
txt = open(deck).read()

def par(block, key):
    m = re.search(rf'^<{block}>(.*?)(?=^<|\Z)', txt, re.S | re.M)
    return float(re.search(rf'^\s*{key}\s*=\s*(\S+)', m.group(1), re.M).group(1))

tlim, ng = par('time', 'tlim'), int(par('mesh', 'nghost'))
cells = 4 * n**3
n_mb = (n // mb[0]) * (4 * n // mb[1]) * (n // mb[2])

# time
cycles = 51 * n * tlim
zcps = {'gpu': 5e7 * cells / (cells + 3.1e4),
        'mpi': 4.6e5 * np_,
        'cpu': 6e5}[backend]
secs = cells * cycles / zcps + 3
done = datetime.datetime.now() + datetime.timedelta(seconds=secs)
h, rem = divmod(int(secs), 3600)
m, s = divmod(rem, 60)
dur = f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")

# storage
n_bin = int(tlim / par('output2', 'dt') + 1e-9) + 1
n_rst = int(tlim / par('output3', 'dt') + 1e-9) + 1
n_hst = int(tlim / par('output1', 'dt') + 1e-9) + 1
bin_b = 24 * cells + 6104
rst_b = 48 * n_mb * (mb[0] + 2*ng) * (mb[1] + 2*ng) * (mb[2] + 2*ng) + 5003
total = n_bin * bin_b + n_rst * rst_b + n_hst * 175

def fmt(b):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if b < 1024 or u == 'TB':
            return f"{b:.1f}{u}" if u != 'B' else f"{b:.0f}B"
        b /= 1024

d = out_dir
while not os.path.exists(d):
    d = os.path.dirname(d)
free = shutil.disk_usage(d).free

eta = f"~{dur} (~{cycles:,.0f} cycles, done around {done:%H:%M on %b %d})"
sto = (f"~{fmt(total)} ({n_bin} bin x {fmt(bin_b)} + {n_rst} rst x {fmt(rst_b)}; "
       f"{fmt(free)} free)")
print(eta, sto, int(free > 1.1 * total), sep='\n')
PY
)
ETA=${EST[0]}; STORAGE=${EST[1]}; FREE_OK=${EST[2]}
echo " Estimated time : ${ETA}"
echo " Est. storage   : ${STORAGE}"
echo "========================================================================"
echo ""

if (( DRY_RUN )); then
    exit 0
fi

if [[ "${FREE_OK}" != 1 ]]; then
    echo "ERROR: not enough free space in ${OUT_ROOT} for this run" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch Athena simulation
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Athena (${BACKEND}) ..."

cd "${BUILD_SRC}"

case ${BACKEND} in
    mpi)
        "${MPIRUN}" -np "${MPI_NP}" "${ATHENA}" -i "${ATHINPUT}" -d "${OUT_DIR}" \
            2>&1 | tee "${LOG_FILE}"
        ;;
    gpu)
        for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
            if [ -d "$cuda_dir/bin" ]; then
                export CUDA_HOME="$cuda_dir"
                export LD_LIBRARY_PATH="$cuda_dir/lib64:${LD_LIBRARY_PATH:-}"
                break
            fi
        done
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        "${ATHENA}" -i "${ATHINPUT}" -d "${OUT_DIR}" 2>&1 | tee "${LOG_FILE}"
        ;;
    cpu)
        "${ATHENA}" -i "${ATHINPUT}" -d "${OUT_DIR}" 2>&1 | tee "${LOG_FILE}"
        ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Simulation completed!"
echo "Outputs stored in: ${OUT_DIR}"
