#!/usr/bin/env python3
"""
Regenerate coarse-grain cache files with the updated coarse-graining approach.

This script finds all cached coarse-grained data and regenerates it using the
new mass-weighted velocity averaging, mass-weighted scalars, and EOS-derived
temperature calculations.

Usage:
    python regenerate_cache.py                    # Auto-detect all caches
    python regenerate_cache.py /path/to/sim/bin   # Regenerate specific simulation
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

try:
    from data_preprocess import simulation_data
    from coarse_grain_utils import cache_scheme_matches, cnn_input_fields, write_cache_scheme
except ImportError as e:
    print(f"Error importing data preprocessing utilities: {e}")
    sys.exit(1)


def find_cache_directories():
    """Find all coarse_* directories in simulation_outputs."""
    cache_dirs = []
    sim_outputs = PROJECT_ROOT / "simulation_outputs"

    if not sim_outputs.exists():
        print(f"No simulation_outputs directory found at {sim_outputs}")
        return cache_dirs

    for cache_dir in sim_outputs.rglob("coarse_*"):
        if cache_dir.is_dir():
            cache_dirs.append(cache_dir)

    return sorted(cache_dirs)


def get_simulation_dir(cache_dir):
    """Infer simulation data directory from cache directory path."""
    # Cache structure: simulation_outputs/*/cache/coarse_*
    # Find the bin directory: simulation_outputs/*/bin

    # Go up to the simulation directory (parent of cache dir)
    sim_dir = cache_dir.parent.parent
    bin_dir = sim_dir / "bin"

    if bin_dir.exists():
        return bin_dir

    # Fallback: look for bin directory in siblings
    for sibling in sim_dir.iterdir():
        if sibling.name == "bin" and sibling.is_dir():
            return sibling

    return None


def parse_cache_config(cache_dirname):
    """Parse cache directory name to extract resolution, downsample, bins.

    Example: coarse_512x256_ds32_bins40
    Returns: (resolution, downsample, bins) or None if parse fails
    """
    try:
        parts = cache_dirname.split("_")
        if len(parts) < 4:
            return None

        res_str = parts[1]  # "512x256"
        resolution = tuple(map(int, res_str.split("x")))

        ds_str = parts[2]   # "ds32"
        downsample = int(ds_str[2:])

        bins_str = parts[3] # "bins40"
        bins = int(bins_str[4:])

        return resolution, downsample, bins
    except (IndexError, ValueError):
        return None


def regenerate_cache(bin_dir, cache_dir, force=False):
    """Regenerate cache files for a simulation.

    Args:
        bin_dir: Path to simulation binary files
        cache_dir: Path to cache directory
        force: If True, regenerate even if cache exists

    Returns:
        True if successful, False otherwise
    """
    cache_config = parse_cache_config(cache_dir.name)
    if not cache_config:
        print(f"  ✗ Could not parse cache directory name: {cache_dir.name}")
        return False

    resolution, downsample, bins = cache_config
    inputs_file = cache_dir / "cg_inputs.npy"
    pdfs_file = cache_dir / "cg_pdfs.npy"

    # Check if cache is up-to-date
    if inputs_file.exists() and pdfs_file.exists() and not force and cache_scheme_matches(cache_dir):
        print(f"  ✓ Cache already up to date, skipping (use --force to regenerate)")
        return True

    print(f"  Generating cache with resolution={resolution}, downsample={downsample}, bins={bins}")

    try:
        # Load simulation data
        print(f"    Loading simulation data from {bin_dir}...")
        sim_data = simulation_data()
        sim_data.resolution = resolution
        sim_data.down_sample = downsample
        sim_data.input_data(str(bin_dir))

        print(f"    Loaded {sim_data.rho.shape[0]} snapshots")

        # Generate coarse-grained inputs
        print(f"    Computing coarse-grained inputs...")
        num_snapshots = sim_data.rho.shape[0]
        cg_nx = sim_data.rho.shape[1] // downsample
        cg_ny = sim_data.rho.shape[2] // downsample

        # Channel layout and averaging shared with training and pdf_plot:
        # (rho, T, ux, uy, s0), see coarse_grain_utils.cnn_input_fields
        cg_inputs = np.zeros((num_snapshots, 5, cg_nx, cg_ny), dtype=np.float32)
        for i in range(num_snapshots):
            cg_inputs[i] = cnn_input_fields(
                sim_data.rho[i], sim_data.ux[i], sim_data.uy[i],
                sim_data.pressure[i], sim_data.ps[i], downsample,
                P_unit=sim_data.P_unit, mu=sim_data.mu, k_b=sim_data.kb,
            )

        print(f"    Computing PDFs...")
        cg_pdfs = np.asarray(sim_data.calc_pixel_pdf(bins=bins), dtype=np.float32)

        # Save cache
        print(f"    Saving cache to {cache_dir}...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(inputs_file, cg_inputs)
        np.save(pdfs_file, cg_pdfs)
        write_cache_scheme(cache_dir)

        size_mb = (cg_inputs.nbytes + cg_pdfs.nbytes) / (1024**2)
        print(f"  ✓ Cache regenerated ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ Error regenerating cache: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate coarse-grain cache files with updated coarse-graining approach"
    )
    parser.add_argument(
        "sim_dirs",
        nargs="*",
        type=Path,
        help="Simulation directories to regenerate (default: auto-detect all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all found cache directories and exit"
    )

    args = parser.parse_args()

    # Find all caches
    caches = find_cache_directories()

    if args.list:
        print(f"Found {len(caches)} cache directories:")
        for cache_dir in caches:
            config = parse_cache_config(cache_dir.name)
            sim_dir = get_simulation_dir(cache_dir)
            print(f"  {cache_dir}")
            if config:
                print(f"    Config: res={config[0]}, ds={config[1]}, bins={config[2]}")
            if sim_dir:
                print(f"    Data: {sim_dir}")
        return

    # Filter by specified directories if provided
    if args.sim_dirs:
        filtered_caches = []
        for cache_dir in caches:
            sim_dir = get_simulation_dir(cache_dir)
            if sim_dir:
                for specified_dir in args.sim_dirs:
                    if specified_dir.resolve() in [sim_dir.parent, sim_dir.parent.parent, sim_dir]:
                        filtered_caches.append(cache_dir)
                        break
        caches = filtered_caches

    if not caches:
        if args.sim_dirs:
            print("No caches found for specified directories.")
        else:
            print("No caches found. Run with --list to see available caches.")
        return

    print(f"\nRegenerating {len(caches)} cache directories with new coarse-graining approach...\n")

    success_count = 0
    for cache_dir in caches:
        print(f"Cache: {cache_dir.name}")
        sim_dir = get_simulation_dir(cache_dir)

        if not sim_dir or not sim_dir.exists():
            print(f"  ✗ Could not find simulation data directory")
            continue

        if regenerate_cache(sim_dir, cache_dir, force=args.force):
            success_count += 1
        print()

    print(f"\nDone! Successfully regenerated {success_count}/{len(caches)} caches.")
    print("\n⚠️  WARNING: Models trained on old cache data will need to be retrained!")
    print("   The input distribution has changed (mass-weighted velocity, EOS-derived temperature)")


if __name__ == "__main__":
    main()
