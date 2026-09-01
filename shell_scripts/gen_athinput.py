#!/usr/bin/env python3
import json
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Generate .athinput from config.json")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--step", required=True, choices=["hr", "lr", "lr_build", "sg"], help="Step name to generate for")
    parser.add_argument("--output", required=True, help="Output path for the .athinput file")
    parser.add_argument("--nx1", type=int, default=None, help="Override nx1")
    parser.add_argument("--nx2", type=int, default=None, help="Override nx2")
    parser.add_argument("--nx3", type=int, default=None, help="Override nx3")
    parser.add_argument("--mb_nx1", type=int, default=None, help="Override mb_nx1")
    parser.add_argument("--mb_nx2", type=int, default=None, help="Override mb_nx2")
    parser.add_argument("--mb_nx3", type=int, default=None, help="Override mb_nx3")
    parser.add_argument("--x1min", type=float, default=None, help="Override x1min")
    parser.add_argument("--x1max", type=float, default=None, help="Override x1max")
    parser.add_argument("--x2min", type=float, default=None, help="Override x2min")
    parser.add_argument("--x2max", type=float, default=None, help="Override x2max")
    parser.add_argument("--sigma", type=float, default=None, help="Override sigma")
    parser.add_argument("--a_char", type=float, default=None, help="Override a_char")
    parser.add_argument("--cold_frac", type=float, default=None, help="Override cold_frac")
    parser.add_argument("--downsample", type=int, default=None, help="Override downsample factor")
    parser.add_argument("--tlim", type=float, default=None, help="Override tlim")
    parser.add_argument("--iprob", type=int, default=None, help="Override iprob")
    parser.add_argument("--init_file", type=str, default=None, help="Override init_file path")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Base params
    params = config.get("global", {}).copy()
    
    # HR params as a baseline if we are computing LR
    hr_params = config.get("hr", {})
    
    # Step-specific params
    step_params = config.get(args.step, {})
    
    # Apply LR downsampling for any step that runs on the coarse grid
    if args.step in ("lr", "lr_build", "sg"):
        lr_params = config.get("lr", {})
        downsample = args.downsample if args.downsample is not None else lr_params.get("downsample_factor", 1)
        params["nx1"] = hr_params.get("nx1", 512) // downsample
        params["nx2"] = hr_params.get("nx2", 1024) // downsample
        # Scale meshblock down, then clamp to mesh size so mb never exceeds nx.
        # Prefer any explicit mb_nx overrides in the lr config section.
        derived_mb_nx1 = min(max(1, hr_params.get("mb_nx1", 64) // downsample), params["nx1"])
        derived_mb_nx2 = min(max(1, hr_params.get("mb_nx2", 1024) // downsample), params["nx2"])
        params["mb_nx1"] = lr_params.get("mb_nx1", derived_mb_nx1)
        params["mb_nx2"] = lr_params.get("mb_nx2", derived_mb_nx2)

    # Override with step specific
    for k, v in step_params.items():
        if k != "downsample_factor":
            params[k] = v

    # Override with explicit CLI arguments if provided
    cli_overrides = {
        "nx1": args.nx1, "nx2": args.nx2, "nx3": args.nx3,
        "mb_nx1": args.mb_nx1, "mb_nx2": args.mb_nx2, "mb_nx3": args.mb_nx3,
        "x1min": args.x1min, "x1max": args.x1max,
        "x2min": args.x2min, "x2max": args.x2max,
        "sigma": args.sigma, "a_char": args.a_char,
        "cold_frac": args.cold_frac, "tlim": args.tlim,
        "iprob": args.iprob, "init_file": args.init_file,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            params[k] = v

    # Group into sections
    blocks = {
        "job": {"basename": "KH"},
        "mesh": {
            "nghost": params.get("nghost"),
            "nx1": params.get("nx1"), "x1min": params.get("x1min"), "x1max": params.get("x1max"), "ix1_bc": params.get("ix1_bc"), "ox1_bc": params.get("ox1_bc"),
            "nx2": params.get("nx2"), "x2min": params.get("x2min"), "x2max": params.get("x2max"), "ix2_bc": params.get("ix2_bc"), "ox2_bc": params.get("ox2_bc"),
            "nx3": params.get("nx3"), "x3min": params.get("x3min"), "x3max": params.get("x3max"), "ix3_bc": params.get("ix3_bc"), "ox3_bc": params.get("ox3_bc"),
        },
        "meshblock": {
            "nx1": params.get("mb_nx1"),
            "nx2": params.get("mb_nx2"),
            "nx3": params.get("mb_nx3"),
        },
        "time": {
            "evolution": params.get("evolution"),
            "integrator": params.get("integrator"),
            "cfl_number": params.get("cfl_number"),
            "nlim": params.get("nlim"),
            "tlim": params.get("tlim"),
            "ndiag": params.get("ndiag"),
        },
        "hydro": {
            "eos": params.get("eos"),
            "reconstruct": params.get("reconstruct"),
            "rsolver": params.get("rsolver"),
            "nscalars": params.get("nscalars"),
            "gamma": params.get("gamma"),
            "ism_cooling": "true" if params.get("ism_cooling") else "false",
            "hrate": params.get("hrate"),
        },
        "problem": {
            "iprob": params.get("iprob"),
            "init_file": params.get("init_file"),
            "amp": params.get("amp"),
            "sigma": params.get("sigma"),
            "vx_hot": params.get("vx_hot"),
            "vx_cold": params.get("vx_cold"),
            "a_char": params.get("a_char"),
            "rho_cold": params.get("rho_cold"),
            "rho_hot": params.get("rho_hot"),
            "press": params.get("press"),
            "cold_frac": params.get("cold_frac"),
        },
        "units": {
            "length_cgs": params.get("length_cgs"),
            "mass_cgs": params.get("mass_cgs"),
            "time_cgs": params.get("time_cgs"),
            "mu": params.get("mu"),
        },
        "output1": {"file_type": "hst", "dt": params.get("hst_dt")},
        "output2": {"file_type": "bin", "variable": "hydro_w", "dt": params.get("bin_w_dt")},
        "output3": {"file_type": "bin", "variable": "hydro_u", "dt": params.get("bin_u_dt")},
        "output4": {"file_type": "rst", "dt": params.get("rst_dt")},
    }

    if "user_srcs" in params:
        blocks["problem"]["user_srcs"] = "true" if params["user_srcs"] else "false"

    with open(args.output, "w") as f:
        f.write("# Athena++ (Kokkos version) input file generated from config.json\n\n")
        f.write("<comment>\nproblem   = Kelvin-Helmholtz instability\n\n")
        
        for block_name, block_params in blocks.items():
            f.write(f"<{block_name}>\n")
            for k, v in block_params.items():
                if v is not None:
                    if block_name == "problem" and k == "a_char":
                        f.write(f"a_char = {v}\t\t    # width of tanh profile\n#a_char = 0.125\n")
                    elif block_name == "problem" and k == "press":
                        f.write(f"# press = 8.63359\t\t# pressure = 1.380649e-13 dyne cm^-3; 8.63359\n")
                        f.write(f"press = {v} # Since mu was updated, even pressure should be updated from 8.63359\n")
                    elif block_name == "problem" and k == "cold_frac":
                        f.write(f"#cold_frac = 0.16666667\t# fraction of region occupied by cold gas\n")
                        f.write(f"cold_frac = {v:.8f}\n")
                        f.write(f"#cold_frac = 0.3300000\n")
                    elif block_name == "units" and k in ["length_cgs", "mass_cgs", "time_cgs"]:
                        f.write(f"{k:<11} = {v:.5e}\n")
                    elif block_name == "units" and k == "mu":
                        f.write(f"mu = {v}\n")
                        f.write("# This is because we're in plasma and things need to be ionized.\n\n")
                        f.write("# Units :-\n")
                        f.write("# length = 1 pc = 3.08568e+18 cm\n")
                        f.write("# time   = 1 My  = 3.15576e+13 s\n")
                        f.write("# mass   = 4.91417e+31 g cm^-3 \t\t(this is so that density has units of m_p per cm^3)\n")
                        f.write("# velocity = 9.77793e+4  cm s^-1\n")
                        f.write("# density  = 1.67262e-24 g cm^-3\n")
                        f.write("# energy   = 4.69834e+41 ergs\n")
                        f.write("# power    = 1.48881e+28 erg/s\n")
                        f.write("# energy density/pressure = 1.59916e-14 dyne/cm^2\n")
                    else:
                        f.write(f"{k:<11} = {v}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
