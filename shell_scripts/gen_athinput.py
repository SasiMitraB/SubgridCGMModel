#!/usr/bin/env python3
import json
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Generate .athinput from config.json")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--step", required=True, choices=["hr", "lr", "sg"], help="Step name to generate for")
    parser.add_argument("--output", required=True, help="Output path for the .athinput file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Base params
    params = config.get("global", {}).copy()
    
    # HR params as a baseline if we are computing LR
    hr_params = config.get("hr", {})
    
    # Step-specific params
    step_params = config.get(args.step, {})
    
    # Apply LR downsampling if needed before overriding with step params
    if args.step == "lr":
        downsample = step_params.get("downsample_factor", 1)
        params["nx1"] = hr_params.get("nx1", 256) // downsample
        params["nx2"] = hr_params.get("nx2", 512) // downsample
        # By default, scale meshblock down too, unless explicitly overriden
        params["mb_nx1"] = max(1, hr_params.get("mb_nx1", 32) // downsample)
        params["mb_nx2"] = max(1, hr_params.get("mb_nx2", 512) // downsample)

    # Override with step specific
    for k, v in step_params.items():
        if k != "downsample_factor":
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
                    # Format float without python's e notation if it's not a scientific notation by default
                    f.write(f"{k:<11} = {v}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
