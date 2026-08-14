import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Suppress matplotlib missing font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ------------------------------------------------------------
# Your lambda_cool() function goes here
# ------------------------------------------------------------
# New Version that truncates to 10^4.5 to 10^5.5
def lambda_cool(temp):
    """
    Cooling function ISMCoolFn translated from AthenaK C++.
    Works on scalars or numpy arrays (any shape).
    Returns Λ(T) in erg cm^3 / s.
    """
    logt = np.log10(temp)
    lhd = np.array(
        [
            -22.5977,
            -21.9689,
            -21.5972,
            -21.4615,
            -21.4789,
            -21.5497,
            -21.6211,
            -21.6595,
            -21.6426,
            -21.5688,
            -21.4771,
            -21.3755,
            -21.2693,
            -21.1644,
            -21.0658,
            -20.9778,
            -20.8986,
            -20.8281,
            -20.7700,
            -20.7223,
            -20.6888,
            -20.6739,
            -20.6815,
            -20.7051,
            -20.7229,
            -20.7208,
            -20.7058,
            -20.6896,
            -20.6797,
            -20.6749,
            -20.6709,
            -20.6748,
            -20.7089,
            -20.8031,
            -20.9647,
            -21.1482,
            -21.2932,
            -21.3767,
            -21.4129,
            -21.4291,
            -21.4538,
            -21.5055,
            -21.5740,
            -21.6300,
            -21.6615,
            -21.6766,
            -21.6886,
            -21.7073,
            -21.7304,
            -21.7491,
            -21.7607,
            -21.7701,
            -21.7877,
            -21.8243,
            -21.8875,
            -21.9738,
            -22.0671,
            -22.1537,
            -22.2265,
            -22.2821,
            -22.3213,
            -22.3462,
            -22.3587,
            -22.3622,
            -22.3590,
            -22.3512,
            -22.3420,
            -22.3342,
            -22.3312,
            -22.3346,
            -22.3445,
            -22.3595,
            -22.3780,
            -22.4007,
            -22.4289,
            -22.4625,
            -22.4995,
            -22.5353,
            -22.5659,
            -22.5895,
            -22.6059,
            -22.6161,
            -22.6208,
            -22.6213,
            -22.6184,
            -22.6126,
            -22.6045,
            -22.5945,
            -22.5831,
            -22.5707,
            -22.5573,
            -22.5434,
            -22.5287,
            -22.5140,
            -22.4992,
            -22.4844,
            -22.4695,
            -22.4543,
            -22.4392,
            -22.4237,
            -22.4087,
            -22.3928,
        ]
    )

    lam = np.zeros_like(temp, dtype=float)

    # KI02 regime (logt <= 4.2)
    mask_ki = logt <= 4.2
    if np.any(mask_ki):
        t_ki = temp[mask_ki]
        lam[mask_ki] = 2.0e-19 * np.exp(
            -1.184e5 / (t_ki + 1.0e3)
        ) + 2.8e-28 * np.sqrt(t_ki) * np.exp(-92.0 / t_ki)

    # CGOLS fit (logT > 8.15)
    mask_hi = logt > 8.15
    if np.any(mask_hi):
        lam[mask_hi] = 10.0 ** (0.45 * logt[mask_hi] - 26.065)

    # SPEX interpolation (4.2 < logT <= 8.15)
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        ipps = (25.0 * logt[mask_mid] - 103).astype(int)
        ipps = np.clip(ipps, 0, 100)
        x0 = 4.12 + 0.04 * ipps
        dx = logt[mask_mid] - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0**logcool

    return lam

# Temperature range: 10^3 K -> 10^7 K
T = np.logspace(3, 7, 1000)

# Evaluate cooling function
Lambda = lambda_cool(T)

# ------------------------------------------------------------
# XKCD-style plot
# ------------------------------------------------------------
with plt.xkcd(scale=1, length=100, randomness=2):
    plt.rcParams['font.family'] = ['xkcd Script', 'Humor Sans', 'Comic Neue']

    fig, ax = plt.subplots(figsize=(9, 6))

    # 1. Force the plot area background to be solid white
    ax.set_facecolor('white')
    ax.patch.set_alpha(1.0) 

    # 2. Move both major and minor ticks inside the plot area
    ax.tick_params(axis='both', which='both', direction='in')

    ax.loglog(
        T,
        Lambda,
        color="xkcd:blue",
        linewidth=2.5,
        label=r"$\Lambda(T)$"
    )

    ax.axvspan(1.05e4, 0.95e6, alpha=0.25, label='Active Cooling Window')

    ax.set_xlabel(r"Temperature, $T$ [K]", fontsize=14)
    ax.set_ylabel(
        r"Cooling function, $\Lambda$ [erg cm$^3$ s$^{-1}$]",
        fontsize=14
    )

    ax.set_title(
        r"Radiative Cooling Function",
        fontsize=17
    )

    # Temperature limits
    #ax.set_xlim(10**4.5, 10**5.5)

    # Let matplotlib choose the y range
    ax.set_ylim(
        Lambda[Lambda > 0].min() * 0.8,
        Lambda.max() * 1.3
    )

    # Grid
    # ax.grid(
    #     True,
    #     which="both",
    #     alpha=0.25,
    #     linestyle="-"
    # )

    ax.legend(
        frameon=False,
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig("xkcd_plot.png", transparent=True)