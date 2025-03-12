import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal
from ..cmap import BivariateCmap


def scatterplot(
    cmap: BivariateCmap | np.ndarray,
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    hue_x: np.ndarray | pd.Series,
    hue_y: np.ndarray | pd.Series,
    hue_x_binning_method: Literal["equal", "quantile", "fisher"] | None = None,
    hue_y_binning_method: Literal["equal", "quantile", "fisher"] | None = None,
    s: int = 30,
    erase_ticks: bool = True,
    erase_ticklabels: bool = True,
    erase_spines: bool = True,
    ax: plt.Axes | None = None,
    mpl_scatter_kwargs: dict = {},
) -> plt.Axes:
    """Plot a scatterplot using the bivariate colormap.
    
    Parameters
    ----------
    cmap : BivariateCmap | np.ndarray
        The bivariate colormap.

    x : np.ndarray | pd.Series
        The `x` values.

    y : np.ndarray | pd.Series
        The `y` values.

    hue_x : np.ndarray | pd.Series
        The `x` hue values.

    hue_y : np.ndarray | pd.Series
        The `y` hue values.

    hue_x_binning_method : {"equal", "quantile", "fisher"}, default="quantile"
        The method to compute the `x` hue bins.
        If set here, it overrides the value set during initialization.

    hue_y_binning_method : {"equal", "quantile", "fisher"}, default="quantile"
        The method to compute the `y` hue bins.
        If set here, it overrides the value set during initialization.

    s : int, default=30
        The marker size.

    erase_ticks : bool, default=True
        Whether to erase the ticks.

    erase_ticklabels : bool, default=True
        Whether to erase the tick labels.

    erase_spines : bool, default=True
        Whether to erase the spines.

    ax : plt.Axes | None
        The matplotlib axes to plot the heatmap. \
        If `None`, a new figure and axes are created.

    mpl_scatter_kwargs : dict, default={}
        Additional keyword arguments to pass to `ax.scatter()`.

    Returns
    -------
    plt.Axes
        The matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2))
    return cmap.plot_scatterplot(
        x=x,
        y=y,
        hue_x=hue_x,
        hue_y=hue_y,
        hue_x_binning_method=hue_x_binning_method,
        hue_y_binning_method=hue_y_binning_method,
        s=s,
        erase_ticks=erase_ticks,
        erase_ticklabels=erase_ticklabels,
        erase_spines=erase_spines,
        ax=ax,
        mpl_scatter_kwargs=mpl_scatter_kwargs,
    )


def legend(
    cmap: BivariateCmap | np.ndarray,
    hue_x: np.ndarray | pd.Series | None = None,
    hue_y: np.ndarray | pd.Series | None = None,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2))
    if isinstance(cmap, np.ndarray):
        cmap = BivariateCmap(cmap)
    if not isinstance(cmap, BivariateCmap):
        raise ValueError("cmap must be a BivariateCmap or a numpy array")
    return cmap.plot_legend(
        hue_x=hue_x,
        hue_y=hue_y,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        ax=ax,
    )
