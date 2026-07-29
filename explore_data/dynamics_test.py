import os
import sys
import gc
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# Append project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from ergane import SimulationData

RESOLUTION_TEST_ROOT = Path('/home/sasi/Projects/SubgridCGMModel/simulation_outputs/resolution_test')
OUTPUT_DIR = Path('/home/sasi/Projects/SubgridCGMModel/outputs/explore_data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECONDS_PER_MYR = 3.15576e13
CM_PER_PC = 3.08568e18
CM_PER_KM = 1.0e5
MIN_PROFILE_TIME_MYR = 5.0


def iter_resolution_runs(root: Path):
    def sort_key(run_dir: Path):
        match = re.fullmatch(r'(\d+)x(\d+)', run_dir.name)
        if match is None:
            return (float('inf'), run_dir.name)
        nx = int(match.group(1))
        ny = int(match.group(2))
        return (nx * ny, nx, ny, run_dir.name)

    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=sort_key):
        athinput_files = sorted(run_dir.glob('*.athinput'))
        if not athinput_files:
            continue
        data_dir = run_dir / 'bin'
        if not data_dir.is_dir():
            continue
        yield run_dir.name, athinput_files[0], data_dir


def mixing_measure(frame) -> float:
    """Compute \int c(1-c) dV for a 2-D frame using the cell areas."""
    c = frame.scalar_00
    if c.ndim != 2:
        raise ValueError(
            f"Expected a 2-D scalar field for mixing measure, got shape {c.shape!r}."
        )

    dx = np.diff(frame.x)
    dy = np.diff(frame.y)
    cell_area = dy[:, None] * dx[None, :]
    return float(np.sum(c * (1.0 - c) * cell_area))


def time_axis_label(simulation: SimulationData) -> str:
    """Return a label that reflects the physical time unit from athinput."""
    if simulation.units.system == 'code':
        return 'Time [code units]'
    if np.isclose(simulation.units.time, SECONDS_PER_MYR):
        return 'Time [Myr]'
    if simulation.units.system in {'CGS', 'SI'}:
        return 'Time [s]'
    return f'Time [{simulation.units.system}]'


def time_axis_values(simulation: SimulationData, times: np.ndarray) -> np.ndarray:
    """Convert the plotted time axis to the most natural physical unit."""
    if simulation.units.system == 'code':
        return times
    if np.isclose(simulation.units.time, SECONDS_PER_MYR):
        return times / SECONDS_PER_MYR
    return times


def length_axis_label(simulation: SimulationData) -> str:
    """Return a label that reflects the physical length unit from athinput."""
    if simulation.units.system == 'code':
        return 'y [code units]'
    if np.isclose(simulation.units.length, CM_PER_PC):
        return 'y [pc]'
    if simulation.units.system in {'CGS', 'SI'}:
        return 'y [cm]'
    return f'y [{simulation.units.system}]'


def length_axis_values(simulation: SimulationData, lengths: np.ndarray) -> np.ndarray:
    """Convert the plotted length axis to the most natural physical unit."""
    if simulation.units.system == 'code':
        return lengths
    if np.isclose(simulation.units.length, CM_PER_PC):
        return lengths / CM_PER_PC
    return lengths


def x_average_profile(frame, values: np.ndarray) -> np.ndarray:
    """Compute the x-averaged profile of a 2-D field as a function of y."""
    if values.ndim != 2:
        raise ValueError(f'Expected a 2-D field, got shape {values.shape!r}.')

    dx = np.abs(np.diff(frame.x))
    if values.shape[1] != dx.size:
        raise ValueError(
            f'Field shape {values.shape!r} is incompatible with x grid of size {dx.size!r}.'
        )

    weighted_sum = np.sum(values * dx[None, :], axis=1)
    return weighted_sum / np.sum(dx)


def collect_profile_statistics(
    simulation: SimulationData,
    field_getter,
    minimum_time_myr: float = MIN_PROFILE_TIME_MYR,
):
    """Return the post-threshold x-averaged profile mean and standard deviation."""
    y_values = None
    profiles = []

    for frame_num in tqdm(
        simulation.frame_numbers,
        desc='Computing profile statistics',
    ):
        frame = simulation.get_frame(frame_num)
        time_myr = time_axis_values(simulation, np.asarray([frame.time]))[0]
        if time_myr <= minimum_time_myr:
            del frame
            gc.collect()
            continue

        field = field_getter(frame)
        if field is None:
            raise RuntimeError('Requested field is not available for this simulation.')

        profile = x_average_profile(frame, field)
        current_y = length_axis_values(simulation, frame.yc)
        if y_values is None:
            y_values = current_y
        elif not np.allclose(y_values, current_y):
            raise RuntimeError('Y grids differ across frames; cannot combine profiles.')

        profiles.append(profile)
        del frame
        gc.collect()

    if not profiles:
        raise RuntimeError(
            f'No frames in the post-{minimum_time_myr:g} Myr window for this simulation.'
        )

    stacked = np.vstack(profiles)
    return y_values, np.mean(stacked, axis=0), np.std(stacked, axis=0)


def temperature_field(frame):
    return frame.temperature


def velocity_x_field(frame):
    if frame.velx is None:
        return None
    return frame.velx / CM_PER_KM

def passive_scalar_field(frame):
    if frame.scalar_00 is None:
        return None 
    return frame.scalar_00


def plot_profile_comparison(
    root: Path,
    save_name: str,
    title: str,
    field_getter,
    ylabel: str,
) -> Path:
    """Plot time-averaged x-profiles for every resolution test run."""
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap('tab10')
    x_label = None

    for idx, (run_name, athinp_path, data_dir) in enumerate(iter_resolution_runs(root)):
        simulation = SimulationData(
            athinp=str(athinp_path),
            datafolder=str(data_dir),
        )

        if field_getter is temperature_field and 'temperature' not in simulation.fields_available:
            raise RuntimeError(f'Temperature is not available in {run_name}.')
        if field_getter is velocity_x_field and 'velx' not in simulation.fields_available:
            raise RuntimeError(f'Velocity field v_x is not available in {run_name}.')

        y_values, profile_mean, profile_std = collect_profile_statistics(
            simulation,
            field_getter,
        )

        color = cmap(idx % cmap.N)
        ax.plot(y_values, profile_mean, lw=2, color=color, label=run_name)
        ax.fill_between(
            y_values,
            profile_mean - profile_std,
            profile_mean + profile_std,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

        if x_label is None:
            x_label = length_axis_label(simulation)

    ax.set_xlabel(x_label or 'y')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Run', fontsize=9)
    fig.tight_layout()

    save_path = OUTPUT_DIR / save_name
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_run(run_name: str, athinp_path: Path, data_dir: Path) -> Path:
    simulation = SimulationData(
        athinp=str(athinp_path),
        datafolder=str(data_dir),
    )

    if 'scalar_00' not in simulation.fields_available:
        raise RuntimeError(
            f"Passive scalar 'scalar_00' is not available in {run_name}."
        )

    times = []
    mixing = []

    for frame_num in tqdm(
        simulation.frame_numbers,
        desc=f'Computing mixing measure ({run_name})',
    ):
        frame = simulation.get_frame(frame_num)
        times.append(frame.time)
        mixing.append(mixing_measure(frame))
        del frame
        gc.collect()

    times = time_axis_values(simulation, np.asarray(times))
    mixing = np.asarray(mixing)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, mixing, color='tab:blue', lw=2)
    ax.set_xlabel(time_axis_label(simulation))
    ax.set_ylabel(r'$\int c(1-c)\,dV$')
    ax.set_title(f'Mixing Measure vs Time ({run_name})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = OUTPUT_DIR / f'mixing_measure_{run_name}.png'
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return save_path


fig, ax = plt.subplots(figsize=(9, 6))
saved_paths = []
x_label = None

for run_name, athinp_path, data_dir in iter_resolution_runs(RESOLUTION_TEST_ROOT):
    simulation = SimulationData(
        athinp=str(athinp_path),
        datafolder=str(data_dir),
    )

    if 'scalar_00' not in simulation.fields_available:
        raise RuntimeError(
            f"Passive scalar 'scalar_00' is not available in {run_name}."
        )

    times = []
    mixing = []

    for frame_num in tqdm(
        simulation.frame_numbers,
        desc=f'Computing mixing measure ({run_name})',
    ):
        frame = simulation.get_frame(frame_num)
        times.append(frame.time)
        mixing.append(mixing_measure(frame))
        del frame
        gc.collect()

    times = time_axis_values(simulation, np.asarray(times))
    mixing = np.asarray(mixing)
    ax.plot(times, mixing, lw=2, label=run_name)
    if x_label is None:
        x_label = time_axis_label(simulation)

ax.set_xlabel(x_label or 'Time')
ax.set_ylabel(r'$\int c(1-c)\,dV$')
ax.set_title('Mixing Measure vs Time for Resolution Test Runs')
ax.grid(True, alpha=0.3)
ax.legend(title='Run', fontsize=9)
fig.tight_layout()

save_path = OUTPUT_DIR / 'mixing_measure_all_resolutions.png'
fig.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved plot to: {save_path}')

temperature_plot = plot_profile_comparison(
    RESOLUTION_TEST_ROOT,
    'temperature_profile_all_resolutions.png',
    'Mean Temperature Profile vs y for Resolution Test Runs',
    temperature_field,
    r'$\langle T \rangle_x$ [K]',
)
print(f'Saved plot to: {temperature_plot}')

velocity_plot = plot_profile_comparison(
    RESOLUTION_TEST_ROOT,
    'velocity_x_profile_all_resolutions.png',
    'Mean x-Velocity Profile vs y for Resolution Test Runs',
    velocity_x_field,
    r'$\langle v_x \rangle_x$ [km s$^{-1}$]',
)
print(f'Saved plot to: {velocity_plot}')

passive_scalar_plot = plot_profile_comparison(
    RESOLUTION_TEST_ROOT,
    'passive_scalar_profile.png',
    'Passive Scalar profile mean x vs y',
    passive_scalar_field,
    r'$\langle c \rangle_x$'
)
print(f'Saved Plot to: {passive_scalar_plot}')