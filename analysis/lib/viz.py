"""viz.py — shared matplotlib defaults so every figure looks consistent."""

import matplotlib.pyplot as plt

PALETTE = ["#2f6f9f", "#c0603f", "#5a8f5a", "#8a6fb0", "#c9a13b", "#7a7a7a"]


def apply_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE)


def savefig(fig, name, outdir="../outputs"):
    """Save a figure as PNG + PDF into the outputs folder."""
    from pathlib import Path
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
