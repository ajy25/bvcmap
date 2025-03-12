"""
This file is a modified version of the xycmap implementation by Remco Bastiaan Jansen.
The license is included below.
The repository for xycmap is: https://github.com/rbjansen/xycmap/


----------------------------------------------------------------------------------------
MIT License

Copyright (c) 2021 Remco Bastiaan Jansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings
import numpy as np
import pandas as pd
from pandas.api import types
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage.interpolation import zoom


def custom_xycmap(corner_colors=("grey", "green", "blue", "red"), n=(5, 5)):
    """Creates a colormap according to specified colors and n categories.

    Args:
        corner_colors: List or Tuple of four matplotlib colors.
        n: Tuple containing the number of columns and rows (x, y).

    Returns:
        Custom two-dimensional colormap in np.ndarray.

    Raises:
        ValueError: If less than two columns or rows are passed.
    """
    xn, yn = n
    if xn < 2 or yn < 2:
        raise ValueError("Expected n >= 2 categories.")

    color_array = np.array(
        [
            [
                list(colors.to_rgba(corner_colors[0])),
                list(colors.to_rgba(corner_colors[1])),
            ],
            [
                list(colors.to_rgba(corner_colors[2])),
                list(colors.to_rgba(corner_colors[3])),
            ],
        ]
    )
    zoom_factor_x = xn / 2  # Divide by the original two categories.
    zoom_factor_y = yn / 2
    zcolors = zoom(color_array, (zoom_factor_y, zoom_factor_x, 1), order=1)

    return zcolors


def mean_xycmap(xcmap=plt.cm.Greens, ycmap=plt.cm.Reds, n=(5, 5)):
    """Creates a colormap according to two colormaps and n categories.

    Args:
        xcmap: Matplotlib colormap along the x-axis. Defaults to Greens.
        ycmap: Matplotlib colormap along the y-axis. Defaults to Reds.
        n: Tuple containing the number of columns and rows (x, y).

    Returns:
        Custom two-dimensional colormap in np.ndarray.

    Raises:
        ValueError: If less than two columns or rows are passed.
    """
    xn, yn = n
    if xn < 2 or yn < 2:
        raise ValueError("Expected n >= 2 categories.")

    sy, sx = np.mgrid[0:yn, 0:xn]

    # Rescale the mock series into the colormap range (0, 255).
    xvals = np.array(255 * (sx - sx.min()) / (sx.max() - sx.min()), dtype=int)
    yvals = np.array(255 * (sy - sy.min()) / (sy.max() - sy.min()), dtype=int)

    xcolors = xcmap(xvals)
    ycolors = ycmap(yvals)

    # Take the mean of the two colormaps.
    zcolors = np.sum([xcolors, ycolors], axis=0) / 2

    return zcolors


def bivariate_color(
    sx,
    sy,
    cmap,
    xlims=None,
    ylims=None,
    xbins=None,
    ybins=None,
    x_method="equal",
    y_method="equal",
):
    """Creates a color series for a combination of two series.

    Args:
        sx: Initial pd.Series to plot.
        sy: Secondary pd.Series to plot.
        cmap: A two-dimensional colormap in np.ndarray.
        xlims: Optional tuple specifying limits to the x-axis.
        ylims: Optional tuple specifying limits to the y-axis.
        xbins: Optional iterable containing bins for the x-axis.
        ybins: Optional iterable containing bins for the y-axis.
        x_method: Binning method for sx: "equal", "quantile", or "fisher".
        y_method: Binning method for sy: "equal", "quantile", or "fisher".

    Returns:
        pd.Series of assigned colors per cmap provided.
    """
    x_numeric = types.is_numeric_dtype(sx)
    y_numeric = types.is_numeric_dtype(sy)
    x_categorical = types.is_categorical_dtype(sx)
    y_categorical = types.is_categorical_dtype(sy)

    msg = (
        "The provided {s} is not numeric or categorical. If {s} contains "
        "categories, transform the series to (ordered) pd.Categorical first."
    )
    if not x_numeric and not x_categorical:
        raise TypeError(msg.format(s="sx"))
    if not y_numeric and not y_categorical:
        raise TypeError(msg.format(s="sy"))

    # If categorical, the number of categories must match the cmap dimensions.
    if x_categorical:
        if len(sx.categories) != cmap.shape[1]:
            raise ValueError(
                f"Length of x-axis colormap ({cmap.shape[1]}) does not match "
                f"the number of categories in sx ({len(sx.categories)}). "
                "Adjust the n of your cmap."
            )
    if y_categorical:
        if len(sy.categories) != cmap.shape[0]:
            raise ValueError(
                f"Length of y-axis colormap ({cmap.shape[0]}) does not match "
                f"the number of categories in sy ({len(sy.categories)}). "
                "Adjust the n of your cmap."
            )

    # If numeric, compute bins based on the chosen method.
    if x_numeric:
        data_x = sx if xlims is None else sx[(sx >= xlims[0]) & (sx <= xlims[1])]
        if xbins is None:
            if x_method == "equal":
                xmin, xmax = data_x.min(), data_x.max()
                _, xbins = pd.cut(pd.Series([xmin, xmax]), cmap.shape[1], retbins=True)
            elif x_method == "quantile":
                _, xbins = pd.qcut(
                    data_x, q=cmap.shape[1], retbins=True, duplicates="drop"
                )
            elif x_method == "fisher":
                try:
                    import jenkspy
                except ImportError:
                    raise ImportError("jenkspy library is required for fisher method")
                xbins = np.array(
                    jenkspy.jenks_breaks(
                        data_x.dropna().values, n_classes=cmap.shape[1]
                    )
                )
            else:
                raise ValueError(f"Unknown x_method: {x_method}")
    else:
        if xlims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sx: the xticks of the "
                "cmap are indivisible. Instead, limit your data to the "
                "categories and adjust the n of cmap accordingly."
            )
        if xbins is not None:
            raise RuntimeError(
                "Cannot apply bins to a categorical sx: the xticks of the "
                "cmap are indivisible."
            )

    if y_numeric:
        data_y = sy if ylims is None else sy[(sy >= ylims[0]) & (sy <= ylims[1])]
        if ybins is None:
            if y_method == "equal":
                ymin, ymax = data_y.min(), data_y.max()
                _, ybins = pd.cut(pd.Series([ymin, ymax]), cmap.shape[0], retbins=True)
            elif y_method == "quantile":
                _, ybins = pd.qcut(
                    data_y, q=cmap.shape[0], retbins=True, duplicates="drop"
                )
            elif y_method == "fisher":
                try:
                    import jenkspy
                except ImportError:
                    raise ImportError("jenkspy library is required for fisher method")
                ybins = np.array(
                    jenkspy.jenks_breaks(
                        data_y.dropna().values, n_classes=cmap.shape[0]
                    )
                )
            else:
                raise ValueError(f"Unknown y_method: {y_method}")
    else:
        if ylims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sy: the yticks of the "
                "cmap are indivisible. Instead, limit your data to the "
                "categories and adjust the n of cmap accordingly."
            )
        if ybins is not None:
            raise RuntimeError(
                "Cannot apply bins to a categorical sy: the yticks of the "
                "cmap are indivisible."
            )

    def _bin_value(x, bins):
        # If x is less than or equal to the minimum bin value, return 0.
        if x <= bins[0]:
            return 0
        # If x is greater than or equal to the maximum bin value, return the last valid index.
        elif x >= bins[-1]:
            return (
                len(bins) - 2
            )  # Because if there are N+1 bin edges, there are N bins.
        # Otherwise, find the appropriate bin index.
        for i in range(len(bins) - 1):
            if bins[i] < x <= bins[i + 1]:
                return i
        # Fallback (should not happen if bins are well-defined)
        return np.nan

    def _return_color(x, y, cmap):
        if np.isnan(x) or np.isnan(y):
            return (0.0, 0.0, 0.0, 0.0)  # Transparent if either is NaN.
        xidx = _bin_value(x, xbins) if x_numeric else x
        yidx = _bin_value(y, ybins) if y_numeric else y
        return tuple(cmap[yidx, xidx])

    sx = pd.Series(sx.codes) if x_categorical else sx
    sy = pd.Series(sy.codes) if y_categorical else sy

    df = pd.DataFrame({"x": sx, "y": sy})
    colors_assigned = df.apply(lambda g: _return_color(g["x"], g["y"], cmap), axis=1)

    return colors_assigned


def bivariate_legend(
    ax,
    sx,
    sy,
    cmap,
    alpha=1,
    xlims=None,
    ylims=None,
    xlabels=None,
    ylabels=None,
    xbins=None,
    ybins=None,
    x_method="equal",
    y_method="equal",
):
    """Plots bivariate cmap onto an ax to use as a legend.

    Args:
        ax: Matplotlib ax to plot into.
        sx: Initial pd.Series to plot.
        sy: Secondary pd.Series to plot.
        cmap: A two-dimensional colormap in np.ndarray.
        alpha: Optional alpha (0-1) to pass to imshow.
        xlims: Optional tuple specifying limits to the x-axis, if numeric.
        ylims: Optional tuple specifying limits to the y-axis, if numeric.
        xlabels: Optional list of ordered labels for the bins along x.
        ylabels: Optional list of ordered labels for the bins along y.
        xbins: Optional iterable for x-axis bins.
        ybins: Optional iterable for y-axis bins.
        x_method: Binning method for sx: "equal", "quantile", or "fisher".
        y_method: Binning method for sy: "equal", "quantile", or "fisher".

    Returns:
        An ax containing the plotted cmap and relevant tick labels.
    """
    x_numeric = types.is_numeric_dtype(sx)
    y_numeric = types.is_numeric_dtype(sy)
    x_categorical = types.is_categorical_dtype(sx)
    y_categorical = types.is_categorical_dtype(sy)

    msg = (
        "The provided {s} is not numeric or categorical. If {s} contains "
        "categories, transform the series to (ordered) pd.Categorical first."
    )
    if not x_numeric and not x_categorical:
        raise TypeError(msg.format(s="sx"))
    if not y_numeric and not y_categorical:
        raise TypeError(msg.format(s="sy"))

    # Compute bins for tick labels if numeric.
    if x_numeric:
        data_x = sx if xlims is None else sx[(sx >= xlims[0]) & (sx <= xlims[1])]
        if xbins is None:
            if x_method == "equal":
                xmin, xmax = data_x.min(), data_x.max()
                _, xbins = pd.cut(pd.Series([xmin, xmax]), cmap.shape[1], retbins=True)
            elif x_method == "quantile":
                _, xbins = pd.qcut(
                    data_x, q=cmap.shape[1], retbins=True, duplicates="drop"
                )
            elif x_method == "fisher":
                try:
                    import jenkspy
                except ImportError:
                    raise ImportError("jenkspy library is required for fisher method")
                xbins = np.array(
                    jenkspy.jenks_breaks(
                        data_x.dropna().values, n_classes=cmap.shape[1]
                    )
                )
            else:
                raise ValueError(f"Unknown x_method: {x_method}")
        if xlabels is None:
            xlabels = [f"{np.round(i, 2)}" for i in xbins]
    else:
        if xlabels is None:
            xlabels = sx.categories
        if xlims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sx: the xticks of the cmap are indivisible. "
                "Instead, limit your data to the categories and adjust the n of cmap accordingly."
            )

    if y_numeric:
        data_y = sy if ylims is None else sy[(sy >= ylims[0]) & (sy <= ylims[1])]
        if ybins is None:
            if y_method == "equal":
                ymin, ymax = data_y.min(), data_y.max()
                _, ybins = pd.cut(pd.Series([ymin, ymax]), cmap.shape[0], retbins=True)
            elif y_method == "quantile":
                _, ybins = pd.qcut(
                    data_y, q=cmap.shape[0], retbins=True, duplicates="drop"
                )
            elif y_method == "fisher":
                try:
                    import jenkspy
                except ImportError:
                    raise ImportError("jenkspy library is required for fisher method")
                ybins = np.array(
                    jenkspy.jenks_breaks(
                        data_y.dropna().values, n_classes=cmap.shape[0]
                    )
                )
            else:
                raise ValueError(f"Unknown y_method: {y_method}")
        if ylabels is None:
            ylabels = [f"{np.round(i, 2)}" for i in ybins]
    else:
        if ylabels is None:
            ylabels = sy.categories
        if ylims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sy: the yticks of the cmap are indivisible. "
                "Instead, limit your data to the categories and adjust the n of cmap accordingly."
            )

    # Plot the legend.
    ax.imshow(cmap, alpha=alpha, origin="lower")

    # Set tick positions.
    xticks = (
        np.arange(0, cmap.shape[1])
        if x_categorical
        else np.arange(-0.5, cmap.shape[1], 1)
    )
    yticks = (
        np.arange(0, cmap.shape[0])
        if y_categorical
        else np.arange(-0.5, cmap.shape[0], 1)
    )
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Adjust label lengths.
    if len(xlabels) > len(xticks):
        warnings.warn(
            f"More xlabels ({len(xlabels)}) than xticks ({len(xticks)}). Labels were truncated."
        )
        xlabels = xlabels[: len(xticks)]
    if len(xlabels) < len(xticks):
        raise RuntimeError(
            f"Fewer xlabels ({len(xlabels)}) than xticks ({len(xticks)})."
        )
    if len(ylabels) > len(yticks):
        warnings.warn(
            f"More ylabels ({len(ylabels)}) than yticks ({len(yticks)}). Labels were truncated."
        )
        ylabels = ylabels[: len(yticks)]
    if len(ylabels) < len(yticks):
        raise RuntimeError(
            f"Fewer ylabels ({len(ylabels)}) than yticks ({len(yticks)})."
        )

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    return ax
