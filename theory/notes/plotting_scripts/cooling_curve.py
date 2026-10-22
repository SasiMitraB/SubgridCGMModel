"""
The ISM cooling function Lambda(T) of the radiative KH runs, against the stock
AthenaK curve it was modified from.

The runs use srcterms/ismcooling.hpp: Koyama & Inutsuka (2002) below 10^4.2 K,
the SPEX table of Schure et al. (2009) up to 10^8.15 K, the CGOLS power law above
-- and, the one change, Lambda = 0 outside 1.05e4 < T <= 0.95e6 K, so the gas
neither cools below the cold phase nor above the hot one. Without the cut-offs
the curve would keep cooling the cold phase (T ~ 1e4 K) and the hot phase
(T ~ 1e6 K) away from their initial state.

Figure (Overleaf/images/chapter2/): cooling_curve.pdf

    python theory/notes/plotting_scripts/cooling_curve.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "theory" / "sharma+2025"))
from recreating_plots import cool_lambda  # noqa: E402

import notes_style as ns  # noqa: E402

T_WINDOW = (1.05e4, 0.95e6)  # Lambda = 0 outside this window
T_PHASES = ((1e4, "cold"), (1e6, "hot"))


def stock_lambda(T):
    """The curve without the cut-offs: cool_lambda with its zeroing undone."""
    T = np.asarray(T, dtype=float)
    lam = cool_lambda(T)
    out = (T <= T_WINDOW[0]) | (T > T_WINDOW[1])
    lo = T <= 10**4.2
    ki = lo & out
    lam[ki] = (2.0e-19 * np.exp(-1.184e5 / (T[ki] + 1.0e3))
               + 2.8e-28 * np.sqrt(T[ki]) * np.exp(-92.0 / T[ki]))
    hi = out & ~lo
    lam[hi] = _spex(T[hi])
    return lam


def _spex(T):
    """SPEX table + CGOLS tail, cut-offs ignored (T > 10^4.2 K)."""
    from vis_athenak.utils.ism_cooling import ISMCoolFn
    return ISMCoolFn(T)


def main():
    logT = np.linspace(3.5, 8.6, 2000)
    T = 10**logT
    mod, stock = cool_lambda(T), stock_lambda(T)

    fig, ax = plt.subplots(figsize=ns.figsize(7, ns.WIDE), layout="constrained")
    ax.plot(logT, stock, color=ns.HAIRLINE, lw=1.6, ls=(0, (4, 2)), label="stock AthenaK")
    pos = mod > 0
    ax.plot(logT[pos], mod[pos], color=ns.ACCENT, label="as used in the runs")
    for edge in T_WINDOW:
        ax.axvline(np.log10(edge), color=ns.HAIRLINE, ls=(0, (1, 2)), lw=0.8)
    for T0, name in T_PHASES:
        ax.annotate(name, (np.log10(T0), 3e-24), ha="center", va="bottom",
                    fontsize=ns.SMALL, color=ns.MUTED)
    ax.text(0.99, 0.04, r"\itshape $\Lambda=0$ outside $1.05\times10^{4}<T\le0.95\times10^{6}$\,K",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=ns.MUTED)
    ax.set(xlabel=r"$\log_{10}\,T$ [K]", ylabel=r"$\Lambda(T)$ [erg\,cm$^{3}$\,s$^{-1}$]",
           yscale="log", xlim=(3.5, 8.6), ylim=(1e-24, 1e-20))
    ax.legend(loc="upper right")
    ns.save(fig, ns.image_path(2, "cooling_curve.pdf"))


if __name__ == "__main__":
    main()
