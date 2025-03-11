import matplotlib.pyplot as plt
import numpy as np
from ..cmap import BivariateCmap


def scatterplot(
    x: np.ndarray,
    y: np.ndarray,
    hue_x: np.ndarray,
    hue_y: np.ndarray,
    cmap: BivariateCmap | np.ndarray,
    s: int = 30,
    erase_ticks: bool = True,
    erase_ticklabels: bool = True,
    erase_spines: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    cmap = BivariateCmap(cmap)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    ax.scatter(x=x, y=y, c=cmap.hues(hue_x, hue_y), s=s)
    fig.tight_layout()
    if erase_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if erase_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if erase_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    return ax


def legend(cmap: BivariateCmap | np.ndarray, ax: plt.Axes) -> plt.Axes:
    cmap = BivariateCmap(cmap)
    fig = ax.figure
    cmap.plot_cmap_heatmap(ax)
    fig.tight_layout()
    return ax
