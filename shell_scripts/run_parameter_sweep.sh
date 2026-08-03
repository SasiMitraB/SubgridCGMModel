#!/usr/bin/env bash
# ==============================================================================
# run_parameter_sweep.sh
#
# Runs AthenaK simulations for a sweep over parameter pairs (shear velocity & cold gas fraction).
# For each parameter pair, runs:
#   1. HR_build using MPI for resolution 512x256 (nx1=256, nx2=512)
#   2. lr_build for resolution 32x16 (nx1=16, nx2=32)
#   3. subgrid_model build for resolution 32x16 (nx1=16, nx2=32)
#
# Model Retraining Policy:
#   - Does NOT retrain the neural network model.
#   - Uses pre-trained model weights from default path: outputs/model_saves/pdf_model_saves
#
# Folder Structure:
#   simulation_outputs/sweeps/vshear_<V>_coldfrac_<F>/
#       ├── hr_build_512x256/
#       ├── lr_build_32x16/
#       ├── subgrid_model_32x16/
#       └── logs/
# ==============================================================================

set -euo pipefail

# ==============================================================================
# EASY TO TINKER PARAMETERS (TINKER AT TOP OF SCRIPT)
# ==============================================================================

# List of total shear velocities v_shear (in code units) to sweep over.
# In zero-momentum frame with rho_hot=0.001 & rho_cold=0.1:
#   vx_hot = v_shear / 1.1
#   vx_cold = -0.1 * vx_hot
# Example: v_shear = 31.09183 -> vx_hot = 28.2653, vx_cold = -2.82653
SHEAR_VELOCITIES=(31.09183)

# List of cold gas volume fractions (cold_frac) to sweep over.
COLD_FRACS=(0.5)

# Simulation time limit (tlim in Myr) - can be overridden via environment variable
TLIM="${TLIM:-10.0}"

# Number of MPI processes for HR_build (512x256) - can be overridden via environment variable
NUM_MPI_PROCS="${NUM_MPI_PROCS:-16}"

# Base directory for saving simulation output folders - can be overridden via environment variable
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/simulation_outputs/sweeps}"

# Pre-trained model directory (default model path; retraining disabled)
DEFAULT_MODEL_DIR="${PROJECT_ROOT}/outputs/model_saves/pdf_model_saves"

# Virtual environment python path
VENV_ACTIVATE="${PROJECT_ROOT}/venv/bin/activate"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

separator() {
    echo "========================================================================"
}

check_builds() {
    log "Checking binary executables..."
    local missing=0

    if [[ ! -x "${PROJECT_ROOT}/builds/hr_build_mpi/src/athena" ]]; then
        log "ERROR: HR MPI binary not found at ${PROJECT_ROOT}/builds/hr_build_mpi/src/athena"
        missing=1
    fi
    if [[ ! -x "${PROJECT_ROOT}/builds/hr_build/src/athena" ]]; then
        log "ERROR: LR binary not found at ${PROJECT_ROOT}/builds/hr_build/src/athena"
        missing=1
    fi
    if [[ ! -x "${PROJECT_ROOT}/builds/subgrid_model/src/athena" ]]; then
        log "ERROR: Subgrid binary not found at ${PROJECT_ROOT}/builds/subgrid_model/src/athena"
        missing=1
    fi

    if [[ ${missing} -eq 1 ]]; then
        log "Please compile the builds before running this script."
        exit 1
    fi
    log "All binary executables verified successfully."
}

generate_athinput() {
    local target_file="$1"
    local job_name="$2"
    local nx1="$3"
    local nx2="$4"
    local mb_nx1="$5"
    local mb_nx2="$6"
    local ism_cooling="$7"
    local user_srcs="$8"
    local nscalars="$9"
    local vx_hot="${10}"
    local vx_cold="${11}"
    local cold_frac="${12}"
    local ndiag="${13}"

    cat <<EOF > "${target_file}"
# Athena++ (Kokkos version) input file for hydro KH test

<comment>
problem   = Kelvin-Helmholtz instability
reference = Lecoanet et al., Fielding et al.

<job>
basename = ${job_name}

<mesh>
nghost    = 3
nx1       = ${nx1}        # Number of zones in X1-direction
x1min     = -5.0        # minimum value of X1
x1max     =  5.0        # maximum value of X1
ix1_bc    = periodic    # inner-X1 boundary flag
ox1_bc    = periodic    # inner-X1 boundary flag

nx2       = ${nx2}        # Number of zones in X2-direction
x2min     = -10.0       # minimum value of X2
x2max     =  10.0       # maximum value of X2
ix2_bc    = reflect     # inner-X2 boundary flag
ox2_bc    = user        # inner-X2 boundary flag

nx3       = 1           # Number of zones in X3-direction
x3min     = -0.5        # minimum value of X3
x3max     = 0.5         # maximum value of X3
ix3_bc    = periodic    # inner-X3 boundary flag
ox3_bc    = periodic    # inner-X3 boundary flag

<meshblock>
nx1       = ${mb_nx1}        # Number of cells in each MeshBlock, X1-dir
nx2       = ${mb_nx2}      # Number of cells in each MeshBlock, X2-dir
nx3       = 1          # Number of cells in each MeshBlock, X3-dir

<time>
evolution  = dynamic    # dynamic/kinematic/static
#integrator = rk3       # time integration algorithm
integrator = rk2
cfl_number = 0.4        # The Courant, Friedrichs, & Lewy (CFL) Number
nlim       = -1         # cycle limit
tlim       = ${TLIM}        # time limit
ndiag      = ${ndiag}        # cycles between diagostic output

<hydro>
eos         = ideal     # EOS type
#reconstruct = ppmx     # spatial reconstruction method
reconstruct = plm
rsolver     = hllc      # Riemann-solver to be used
nscalars    = ${nscalars}         # number of passive scalars in hydro
gamma       = 1.666667  # gamma = C_p/C_v
#fofc        = true     # Enable first order flux correction
ism_cooling = ${ism_cooling}
hrate 	    = 0

<problem>
iprob = 1               # flag to select setup
amp   = 0.01            # amplitude of sinusoidal perturbation (vshear/100)
sigma = 0.25            # width of gaussian profile

#vx_hot = 30.784         # hot phase velocity (~ +30.1 km/s)
vx_hot = ${vx_hot}        #new velocity consistent with Dimotakis
#vx_hot = 0.0
#vx_hot = 56.5306
#vx_cold = -0.3078 	    # cold phase velocity (~ -0.301 km/s)
vx_cold = ${vx_cold}
#vx_cold = -5.6530

a_char = 0.0625		    # width of tanh profile
#a_char = 0.125
rho_cold = 0.1
rho_hot  = 0.001
press = 13.925145161290322 # Since mu was updated, even pressure should be updated from 8.63359
#cold_frac = 0.16666667	# fraction of region occupied by cold gas
cold_frac = ${cold_frac}
#cold_frac = 0.3300000
EOF

    if [[ "${user_srcs}" == "true" ]]; then
        echo "user_srcs = true" >> "${target_file}"
    fi

    cat <<EOF >> "${target_file}"

<units>
length_cgs  = 3.08568e+18
mass_cgs    = 4.91417e+31
time_cgs    = 3.15576e+13
mu = 0.62
# This is because we're in plasma and things need to be ionized.

# Units :-
# length = 1 pc = 3.08568e+18 cm
# time   = 1 My  = 3.15576e+13 s
# mass   = 4.91417e+31 g cm^-3 		(this is so that density has units of m_p per cm^3)
# velocity = 9.77793e+4  cm s^-1
# density  = 1.67262e-24 g cm^-3
# energy   = 4.69834e+41 ergs
# power    = 1.48881e+28 erg/s
# energy density/pressure = 1.59916e-14 dyne/cm^2

<output1>
file_type  = hst       # History data dump
dt         = 0.01    # time increment between outputs

<output2>
file_type  = bin       # bin data dump
variable   = hydro_w   # variables to be output
dt         = 0.01      # time increment between outputs

<output3>
file_type  = bin       # bin data dump
variable   = hydro_u   # variables to be output
dt         = 0.01      # time increment between outputs

<output4>
file_type  = rst       # rstin data dump
dt         = 1.0      # time increment between outputs
EOF
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

check_builds

# Environment setup
if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_ACTIVATE}"
    log "Activated python venv: ${VENV_ACTIVATE}"
else
    log "WARNING: Virtual environment not found at ${VENV_ACTIVATE}"
fi

export MODEL_SAVES_DIR="${DEFAULT_MODEL_DIR}"
log "Default pre-trained model directory: ${MODEL_SAVES_DIR}"
log "Retraining: OFF (using existing model weights)"

separator
log "STARTING PARAMETER SWEEP"
log "Shear Velocities : ${SHEAR_VELOCITIES[*]}"
log "Cold Gas Fracs   : ${COLD_FRACS[*]}"
log "Time Limit (tlim): ${TLIM} Myr"
log "MPI Processes    : ${NUM_MPI_PROCS}"
separator

for v_shear in "${SHEAR_VELOCITIES[@]}"; do
    for cold_frac in "${COLD_FRACS[@]}"; do
        # Calculate vx_hot and vx_cold in zero-momentum frame
        # v_shear = vx_hot - vx_cold; vx_cold = -0.1 * vx_hot => v_shear = 1.1 * vx_hot
        vx_hot=$(python3 -c "print(f'{$v_shear / 1.1:.6f}')")
        vx_cold=$(python3 -c "print(f'{-0.1 * ($v_shear / 1.1):.6f}')")

        pair_name="vshear_${v_shear}_coldfrac_${cold_frac}"
        pair_dir="${OUTPUT_BASE_DIR}/${pair_name}"
        log_dir="${pair_dir}/logs"

        hr_dir="${pair_dir}/hr_build_512x256"
        lr_dir="${pair_dir}/lr_build_32x16"
        sg_dir="${pair_dir}/subgrid_model_32x16"

        mkdir -p "${hr_dir}" "${lr_dir}" "${sg_dir}" "${log_dir}"

        separator
        log "Processing parameter pair: v_shear=${v_shear} (vx_hot=${vx_hot}, vx_cold=${vx_cold}), cold_frac=${cold_frac}"
        log "Output folder: ${pair_dir}"
        separator

        # ----------------------------------------------------------------------
        # 1. HR Simulation (512x256, MPI)
        # ----------------------------------------------------------------------
        hr_athinput="${pair_dir}/hr_512x256.athinput"
        generate_athinput "${hr_athinput}" "KH_HR_512x256" \
            128 256 32 64 "true" "false" 1 "${vx_hot}" "${vx_cold}" "${cold_frac}" 100

        log "Running 1/3: HR_build (512x256, MPI with ${NUM_MPI_PROCS} processes)..."
        mpirun -np "${NUM_MPI_PROCS}" \
            "${PROJECT_ROOT}/builds/hr_build_mpi/src/athena" \
            -i "${hr_athinput}" \
            -d "${hr_dir}" \
            > "${log_dir}/hr_build_512x256.log" 2>&1
        log "HR_build simulation finished. Log: ${log_dir}/hr_build_512x256.log"

        # ----------------------------------------------------------------------
        # 2. LR Simulation (32x16, ISM cooling)
        # ----------------------------------------------------------------------
        lr_athinput="${pair_dir}/lr_32x16.athinput"
        generate_athinput "${lr_athinput}" "KH_LR_32x16" \
            8 16 8 16 "true" "false" 2 "${vx_hot}" "${vx_cold}" "${cold_frac}" 100

        log "Running 2/3: lr_build (32x16, ISM cooling)..."
        "${PROJECT_ROOT}/builds/hr_build/src/athena" \
            -i "${lr_athinput}" \
            -d "${lr_dir}" \
            > "${log_dir}/lr_build_32x16.log" 2>&1
        log "lr_build simulation finished. Log: ${log_dir}/lr_build_32x16.log"

        # ----------------------------------------------------------------------
        # 3. Subgrid Model Simulation (32x16, PyTorch model)
        # ----------------------------------------------------------------------
        sg_athinput="${pair_dir}/subgrid_32x16.athinput"
        generate_athinput "${sg_athinput}" "KH_SG_32x16" \
            8 16 8 16 "false" "true" 2 "${vx_hot}" "${vx_cold}" "${cold_frac}" 5

        log "Running 3/3: subgrid_model build (32x16, pre-trained CNN model)..."
        (
            export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/data:${PROJECT_ROOT}/models/conv_nn:${PROJECT_ROOT}/builds/subgrid_model/src:${PYTHONPATH:-}"
            export MODEL_SAVES_DIR="${DEFAULT_MODEL_DIR}"
            cd "${PROJECT_ROOT}/builds/subgrid_model/src"
            ./athena \
                -i "${sg_athinput}" \
                -d "${sg_dir}" \
                > "${log_dir}/subgrid_model_32x16.log" 2>&1
        )
        log "subgrid_model simulation finished. Log: ${log_dir}/subgrid_model_32x16.log"

        # ----------------------------------------------------------------------
        # 4. Diagnostic Visualizations (density animation & profiles using ergane)
        # ----------------------------------------------------------------------
        log "Generating diagnostic visualizations (density animation & profiles)..."
        python3 "${PROJECT_ROOT}/shell_scripts/visualize_sweeps.py" "${pair_dir}" \
            > "${log_dir}/visualize_sweeps.log" 2>&1
        log "Visualizations saved: density_animation.mp4 & profiles_perpendicular.png"

        log "Successfully completed parameter pair: ${pair_name}"
    done
done

separator
log "ALL PARAMETER SWEEP SIMULATIONS COMPLETED SUCCESSFULLY"
log "Outputs saved under: ${OUTPUT_BASE_DIR}"
separator
