# `/data/sasi/fid3D_*` simulation parameters

Source: the `PAR_DUMP` block in the header of each run's `bin/KH.hydro_w.*.bin`
(AthenaK writes the full resolved input deck into every binary output). The
headers were read directly (first ~20 KB of each file); no field data was loaded.

All eight runs are the same 3D Kelvin–Helmholtz shear-layer setup with ISM
cooling (`iprob = 2`). They differ only in resolution, and one run also changes
the density/pressure normalisation and box width.

## Per-run summary

Cell size Δ is in pc (code length unit = 1 pc).

| Run | Mesh (nx1 × nx2 × nx3) | Box x × y × z [pc] | Δx = Δz | Δy | MeshBlock | ρ_cold / ρ_hot | P (code) | tlim [Myr] | bin dt / hst dt | # bin |
|---|---|---|---|---|---|---|---|---|---|---|
| `fid3D_16_cool`  | 4 × 16 × 4     | 10 × 40 × 10 | 2.5    | 2.5    | 4×8×4    | 0.1 / 0.001 | 14.03 | 10 | 0.08 / 0.04 | 126 |
| `fid3D_32_cool`  | 8 × 32 × 8     | 10 × 40 × 10 | 1.25   | 1.25   | 4×8×4    | 0.1 / 0.001 | 14.03 | 10 | 0.08 / 0.04 | 126 |
| `fid3D_32_cool2` | 8 × 32 × 8     | 10 × 40 × 10 | 1.25   | 1.25   | 4×8×4    | 0.1 / 0.001 | 14.03 | 10 | 0.08 / 0.04 | 126 |
| `fid3D_64_cool`  | 16 × 64 × 16   | 10 × 40 × 10 | 0.625  | 0.625  | 8×16×8   | 0.1 / 0.001 | 14.03 | 10 | 0.08 / 0.04 | 126 |
| `fid3D_128_cool` | 32 × 128 × 32  | 10 × 40 × 10 | 0.3125 | 0.3125 | 16×32×16 | 0.1 / 0.001 | 14.03 | 10 | 0.08 / 0.04 | 126 |
| `fid3D_130_cool` | 35 × 130 × 35  | 10 × 40 × 10 | 0.2857 | 0.3077 | 35×26×35 | 0.1 / 0.001 | 14.03 | 10 | 0.08 / 0.04 | 126 |
| `fid3D_260_cool` | 70 × 260 × 70  | 10 × 40 × 10 | 0.1429 | 0.1538 | 35×52×35 | 0.1 / 0.001 | 14.03 | 10 | 0.08 / 0.04 | 126 |
| `fid3D_5xmoredens_520halfbox_cool` | 70 × 520 × 70 | **5** × 40 × **5** | 0.0714 | 0.0769 | 35×52×35 | **0.5 / 0.005** | **70.13** | **2** | **0.008 / 0.004** | 251 |

Notes:
- **`fid3D_32_cool2` duplicates `fid3D_32_cool`.** Its parameters match exactly, the history file is byte-identical, and the bin files spot-checked at snapshots 0, 60 and 125 have identical MD5 sums. It also has none of the post-processing outputs (movies, `snaps*/`).
- **`_130`, `_260` and `_5xmoredens_520halfbox` have non-cubic cells**: Δx = Δz differs from Δy by about 7–8%. The power-of-two runs (16 to 128) have cubic cells.
- **`_5xmoredens_520halfbox`** multiplies ρ and P by 5, so temperatures are unchanged and n is 5× higher. The cooling time therefore drops about 5×, which matches the 10× shorter output cadence and the 5× shorter `tlim`. Its box is also half as wide in x and z (5 pc instead of 10 pc).

## Common parameters (identical in all runs)

**Physics / problem**
| Parameter | Value | Physical |
|---|---|---|
| `problem/iprob` | 2 | KH shear layer with cooling |
| `hydro/ism_cooling` | true | `hrate = 0` (no heating) |
| `hydro/gamma` | 5/3 | |
| `units/mu` | 0.62 | |
| `problem/vx_hot` | 28.1818 | +27.56 km/s |
| `problem/vx_cold` | −2.8182 | −2.76 km/s |
| Δv (shear) | 31.0 | 30.31 km/s |
| `problem/cold_frac` | 2/3 | cold gas fills the lower 2/3 of the y-extent |
| `problem/a_char` | 0.125 | tanh interface width [pc] |
| `problem/amp` | 0.05 | perturbation amplitude (the deck comment says `vshear/100`, which would be 0.31; the actual value is 0.05) |
| `problem/sigma` | 0.5 | Gaussian perturbation width [pc] |
| `hydro/nscalars` | 1 | passive scalar `s_00` |

**Fiducial thermodynamic state** (all runs except `5xmoredens`; computed from the unit system below with m_p and μ = 0.62)
| | ρ (code) | n [cm⁻³] | T [K] |
|---|---|---|---|
| Cold | 0.1   | 0.161  | 1.0 × 10⁴ |
| Hot  | 0.001 | 1.6 × 10⁻³ | 1.0 × 10⁶ |
| Pressure | 14.026 | P/k_B ≈ 1625 K cm⁻³ (2.24 × 10⁻¹³ dyn cm⁻²) | |

For `5xmoredens` the temperatures are the same, n is 5× higher (0.81 cold, 8.1 × 10⁻³ hot), and P/k_B ≈ 8120 K cm⁻³.

**Units**
| Key | Value | Meaning |
|---|---|---|
| `length_cgs` | 3.08568e18 | 1 pc |
| `time_cgs`   | 3.15576e13 | 1 Myr |
| `mass_cgs`   | 4.91417e31 | gives ρ_unit = 1.6726e-24 g cm⁻³ (1 m_p cm⁻³) |
| derived v_unit | | 0.978 km/s (1 pc/Myr) |
| derived P_unit | | 1.599e-14 dyn cm⁻² |

**Numerics**
| Parameter | Value |
|---|---|
| `time/integrator` | rk2 |
| `time/cfl_number` | 0.4 |
| `hydro/reconstruct` | plm |
| `hydro/rsolver` | hllc |
| `hydro/fofc` | false |
| `mesh/nghost` | 2 |
| `mesh_refinement` | none |
| floors (`d/p/t/sfloor`) | 1.17549e-38 |
| y-extent | −20 to +20 pc |
| BCs | x1, x3 periodic; x2 inner `reflect`, outer `user` |

**Outputs**
| Block | Type | dt | Contents |
|---|---|---|---|
| `output1` | hst | 0.04 (0.004 for 5xmoredens) | time, dt, mass, momenta, tot-E, KEs, Cold/Hot/Intermediate mass, Edot_total |
| `output2` | bin | 0.08 (0.008) | `hydro_w`: dens, velx, vely, velz, eint, s_00 (float32) |
| `output3` | rst | 1.0 | restarts (11 per 10 Myr run; 3 for 5xmoredens) |

## Disk usage
| Run | Size |
|---|---|
| `fid3D_16_cool` | 487 MB |
| `fid3D_32_cool` | 521 MB |
| `fid3D_32_cool2` | 14 MB |
| `fid3D_64_cool` | 644 MB |
| `fid3D_128_cool` | 1.4 GB |
| `fid3D_130_cool` | 1.6 GB |
| `fid3D_260_cool` | 5.7 GB |
| `fid3D_5xmoredens_520halfbox_cool` | 17 GB |

Each run except `_32_cool2` also contains post-processing products: `*.mp4` movies,
`make_videos.sh`, and a `snaps*/` folder holding a `header.txt` copy of the
deck and 1D-array `.npz` files.
