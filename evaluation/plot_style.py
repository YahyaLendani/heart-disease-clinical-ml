"""One visual style for every figure the pipeline writes.

Colours come from a categorical palette validated for colour-vision
deficiency (OKLab ΔE ≥ 8 between adjacent series in light and dark mode; the
first three slots also pass for every pair). Charts with up to three model
series therefore stay distinguishable for colour-blind readers; legends and
direct labels carry identity too, so colour is never the only channel.

Usage::

    from evaluation.plot_style import apply_style, series_color, finish_axes
    apply_style()                     # once, before creating the figure
    ax.plot(x, y, color=series_color(0))
    finish_axes(ax)                   # recessive grid, no top/right spines
"""

from __future__ import annotations

from pathlib import Path

PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "good": "#0ca30c",
        "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "good": "#0ca30c",
        "critical": "#d03b3b",
    },
}

FONT_FAMILY = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]
_current_mode = "light"


def apply_style(mode: str = "light") -> dict:
    """Set matplotlib defaults for ``mode`` ('light' or 'dark'); return the palette."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler

    global _current_mode
    _current_mode = mode
    p = PALETTES[mode]
    plt.rcParams.update({
        "figure.facecolor": p["surface"],
        "axes.facecolor": p["surface"],
        "savefig.facecolor": p["surface"],
        "axes.edgecolor": p["axis"],
        "axes.labelcolor": p["ink_secondary"],
        "axes.titlecolor": p["ink"],
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=p["series"]),
        "grid.color": p["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": p["muted"],
        "ytick.color": p["muted"],
        "xtick.labelcolor": p["ink_secondary"],
        "ytick.labelcolor": p["ink_secondary"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": p["ink"],
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.labelcolor": p["ink_secondary"],
        "figure.titlesize": 14,
        "figure.titleweight": "semibold",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
    return p


def palette(mode: str | None = None) -> dict:
    return PALETTES[mode or _current_mode]


def series_color(index: int, mode: str | None = None) -> str:
    """Categorical colour for series ``index`` in fixed order (never cycled past 8)."""
    colors = PALETTES[mode or _current_mode]["series"]
    if index >= len(colors):
        raise ValueError("More than 8 series: fold extra series into 'Other' or use small multiples.")
    return colors[index]


def reference_color(mode: str | None = None) -> str:
    """Neutral ink for reference lines (chance, perfect calibration, treat-all)."""
    return PALETTES[mode or _current_mode]["muted"]


def finish_axes(ax) -> None:
    """Recessive grid on the value axis only and no top/right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)


def subtitle(ax, text: str, legend_row: bool = False) -> None:
    """Secondary line under the axes title (cohort, sample size, caveat).

    Call after ``set_title``. With ``legend_row`` a band is reserved between the
    subtitle and the plot for ``legend_above(ax)``.
    """
    p = palette()
    gap = 22 if legend_row else 0
    # Lift the (left-aligned) title to make room for the lines beneath it.
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=24 + gap)
    ax.annotate(
        text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6 + gap), textcoords="offset points",
        ha="left", va="bottom", fontsize=9.5, color=p["ink_secondary"],
    )


def legend_above(ax, ncol: int = 2) -> None:
    """One-row legend in the band reserved by ``subtitle(..., legend_row=True)``."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.15,
              handletextpad=0.3, columnspacing=1.6)


def save(fig, path: str | Path, also_dark: bool = False, draw=None) -> Path:
    """Save ``fig``; with ``also_dark`` re-draw via ``draw(mode)`` into ``<stem>_dark.png``."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    if also_dark and draw is not None:
        apply_style("dark")
        dark_fig = draw("dark")
        dark_fig.savefig(path.with_name(f"{path.stem}_dark{path.suffix}"))
        plt.close(dark_fig)
        apply_style("light")
    return path
