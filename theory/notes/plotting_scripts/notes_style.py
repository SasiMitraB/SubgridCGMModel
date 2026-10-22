r"""
Matplotlib style of the notes (Overleaf/bookstyle.sty): Pagella text and math
typeset by XeLaTeX with the notes' own macros.tex, the book's colours, and a
transparent background so figures sit on the cream page.

Importing the module applies the style. From a script in this folder:

    import notes_style as ns

    fig, ax = plt.subplots(figsize=ns.figsize(7.4), layout="constrained")
    ax.plot(x, y, label=r"$\Tcg$")           # any macro of macros.tex works
    ns.save(fig, ns.image_path(2, "my_plot.pdf"))

Always save through ns.save: it renders with the pgf backend whatever backend
pyplot was switched to, which LaTeX labels and macros need.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

OVERLEAF = Path(__file__).resolve().parents[1] / "Overleaf"

# ─── Colours ─────────────────────────────────────────────────────────────────
# The book's palette (bookstyle.sty). PAPER is the page colour; figures are
# saved transparent, so it is only needed for previews.
PAPER = "#ECE8DC"
INK = "#1B1B1B"        # labels, titles
MUTED = "#4A4845"      # axes, ticks, annotations
HAIRLINE = "#BDB7AA"   # guide lines, decorative only
ACCENT = "#7A2828"     # brick: one highlighted line per figure
LINKBLUE = "#294570"   # slate

# Categorical series, in this fixed order: the book's hues lifted to be
# legible as thin lines on PAPER. Checked for colour-blind separation between
# neighbours (dataviz validate_palette.js: all checks passed on the earlier paper #F7F3EA; not re-run on #ECE8DC).
# For more than four series, use small multiples rather than new colours.
SERIES = ["#3D62A3", "#B04A3A", "#0F9474", "#AD7B17"]

# Sequential (magnitude): magma, reversed so counts darken as they grow, with its
# washed-out pale-yellow end cut off so the smallest non-empty bin still stands
# off PAPER. OVERLAY is a line colour that reads on both ends of it (median
# lines drawn over a histogram).
SEQ = LinearSegmentedColormap.from_list(
    "notes_seq", plt.get_cmap("magma_r")(np.linspace(0.15, 1.0, 256)))
OVERLAY = "#12A0C4"
# Diverging (signed): slate below zero, brick above, neutral warm grey at zero.
DIVERGING = LinearSegmentedColormap.from_list(
    "notes_diverging", [LINKBLUE, "#8FA0BC", "#E4E0D8", "#C49690", ACCENT])

# ─── Sizes ───────────────────────────────────────────────────────────────────
CM = 1 / 2.54          # inches per cm
TEXTWIDTH = 11.8       # cm, geometry of bookstyle.sty
MARGINWIDTH = 4.6      # cm, for margin figures
WIDE = 16.5            # cm, \widegraphics: text width + margin column, less 0.6 cm
SMALL = 9              # pt, in-plot annotations (the smallest text to use)


def figsize(height, width=TEXTWIDTH):
    """(width, height) in inches from cm; full text width by default."""
    return width * CM, height * CM


# ─── rcParams ────────────────────────────────────────────────────────────────
_FONTS = "Extension=.otf, UprightFont=*-regular, ItalicFont=*-italic, " \
         "BoldFont=*-bold, BoldItalicFont=*-bolditalic"

plt.rcParams.update({
    # XeLaTeX with the fonts of bookstyle.sty and the notes' macros
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage{unicode-math}",
        rf"\setmainfont{{texgyrepagella}}[{_FONTS}]",
        rf"\setsansfont{{texgyreheros}}[{_FONTS}, Scale=0.92]",
        r"\setmathfont{texgyrepagella-math.otf}",
        rf"\setmathsf{{texgyreheros}}[{_FONTS}]",
        rf"\input{{{(OVERLEAF / 'macros.tex').as_posix()}}}",
    ]),
    "font.family": "serif",
    # Text sizes: body is 11 pt and captions 10 pt; figures are saved at the
    # width they are printed, so these are the sizes on the page
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    # Colours
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.facecolor": "none",
    "axes.prop_cycle": cycler(color=SERIES),
    "image.cmap": "notes_seq",
    "grid.color": HAIRLINE,
    "grid.linewidth": 0.4,
    # Thin, quiet axes
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    # Output
    "savefig.transparent": True,
    "savefig.dpi": 300,     # for rasterized artists
})
plt.colormaps.register(SEQ, force=True)
plt.colormaps.register(DIVERGING, force=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def colorbar(fig, mappable, ax, label, **kwargs):
    """Colorbar in the notes' style: no outline, muted ticks, ink label."""
    kwargs.setdefault("pad", 0.015)
    cb = fig.colorbar(mappable, ax=ax, aspect=30, **kwargs)
    cb.set_label(label, color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED, width=0.6)
    return cb


# ─── Data ────────────────────────────────────────────────────────────────────
# Frames before T_MIN are the initial transient of the KH run: the layer is not
# yet turbulent, and the subgrid velocity is close to one- or two-dimensional
# (S near-degenerate). The models drop them too (rvv_tensor_model.py --t-min).
T_MIN = 2.5  # Myr (= code time units)


def drop_transient(d):
    """The per-cell arrays of d (and its list of frames) restricted to t >= T_MIN;
    d["time"] holds the time of every cell, d["frame"] its frame number."""
    keep = d["time"] >= T_MIN
    n = keep.size
    out = {k: (v[keep] if np.ndim(v) and len(v) == n else v) for k, v in d.items()}
    out["frames"] = np.unique(out["frame"])
    return out


def image_path(chapter, name):
    """Overleaf/images/chapter<chapter>/<name>, creating the folder."""
    path = OVERLEAF / "images" / f"chapter{chapter}" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save(fig, path):
    """Save with the pgf backend (XeLaTeX), whatever pyplot's backend is."""
    fig.savefig(path, backend="pgf")
    print(f"saved {path}")
