import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Literal
import pandas as pd
import copy
from .xycmap_modified import custom_xycmap, bivariate_color, bivariate_legend
from .colortype import MPLColor


def make_ndarray_cmap_from_corner_colors(
    bottom_left_corner: MPLColor,
    top_left_corner: MPLColor,
    top_right_corner: MPLColor,
    bottom_right_corner: MPLColor,
    shape: tuple[int, int],
) -> np.ndarray:
    """Creates a bivariate colormap as defined by the corners.

    Parameters
    ----------
    bottom_left_corner : MPLColor
        The color of the bottom left corner.
        Note: The `MPLColor` type is any valid matplotlib color.

    top_left_corner : MPLColor
        The color of the top left corner.

    top_right_corner : MPLColor
        The color of the top right corner.

    bottom_right_corner : MPLColor
        The color of the bottom right corner.

    shape: tuple[int, int]
        The shape of the colormap.

    Returns
    -------
    np.ndarray
        A bivariate colormap. The shape of the colormap is `shape`.
    """
    bottom_left_corner = MPLColor(bottom_left_corner)
    top_left_corner = MPLColor(top_left_corner)
    top_right_corner = MPLColor(top_right_corner)
    bottom_right_corner = MPLColor(bottom_right_corner)

    return custom_xycmap(
        corner_colors=(
            bottom_left_corner.as_hex(),
            bottom_right_corner.as_hex(),
            top_left_corner.as_hex(),
            top_right_corner.as_hex(),
        ),
        n=shape,
    )


class BivariateCmap:
    """A class to represent a bivariate colormap."""

    def __init__(
        self,
        cmap: np.ndarray | Any,
        hue_x_var_name: str = "x",
        hue_y_var_name: str = "y",
        hue_x_binning_method: Literal["equal", "quantile", "fisher"] = "equal",
        hue_y_binning_method: Literal["equal", "quantile", "fisher"] = "equal",
    ):
        """Initialize a bivariate colormap.

        Parameters
        ----------
        cmap : np.ndarray | BvariateCmap
            A NumPy array representing the colormap.
            The shape of the array is (bottom_to_top_dim, left_to_right_dim, 4).

        hue_x_var_name : str, default="x"
            The name of the `x` hue variable.

        hue_y_var_name : str, default="y"
            The name of the `y` hue variable.

        hue_x_binning_method : {"equal", "quantile", "fisher"}, default="equal"
            The method to compute the `x` hue bins.
            - "equal": Divide the `x` values into equal bins.
            - "quantile": Divide the `x` values into quantiles.
            - "fisher": Use Fisher-Jenks natural breaks classification.

        hue_y_binning_method : {"equal", "quantile", "fisher"}, default="equal"
            The method to compute the `y` hue bins.
            - "equal": Divide the `x` values into equal bins.
            - "quantile": Divide the `x` values into quantiles.
            - "fisher": Use Fisher-Jenks natural breaks classification.
        """
        if not isinstance(cmap, BivariateCmap):
            if not isinstance(cmap, np.ndarray):
                try:
                    cmap = np.asarray(cmap)
                except Exception as e:
                    raise TypeError("The colormap must be a NumPy array.") from e
            if cmap.ndim != 3:
                raise ValueError("The colormap must be a 3D NumPy array.")
            if cmap.shape[2] != 4:
                raise ValueError("The colormap must have 4 channels (RGBA).")
            self._ndarray_cmap = cmap
            self._hue_x_var_name = hue_x_var_name
            self._hue_y_var_name = hue_y_var_name
            self._hue_x_binning_method = hue_x_binning_method
            self._hue_y_binning_method = hue_y_binning_method
        else:
            self._ndarray_cmap = cmap.to_numpy()
            self._hue_x_var_name = cmap._hue_x_var_name
            self._hue_y_var_name = cmap._hue_y_var_name
            self._hue_x_binning_method = cmap._hue_x_binning_method
            self._hue_y_binning_method = cmap._hue_y_binning_method

    def __str__(self):
        return f"BivariateCmap(cmap={self._ndarray_cmap})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, BivariateCmap):
            return np.array_equal(self._ndarray_cmap, other._ndarray_cmap)
        return False

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the shape of the colormap NumPy array. \
        The shape is (bottom_to_top_dim, left_to_right_dim, 4).
        """
        return self._ndarray_cmap.shape

    @staticmethod
    def from_corner_colors(
        bottom_left_corner: MPLColor,
        top_left_corner: MPLColor,
        top_right_corner: MPLColor,
        bottom_right_corner: MPLColor,
        left_to_right_dim: int,
        bottom_to_top_dim: int,
        hue_x_var_name: str = "x",
        hue_y_var_name: str = "y",
        hue_x_binning_method: Literal["equal", "quantile", "fisher"] = "equal",
        hue_y_binning_method: Literal["equal", "quantile", "fisher"] = "equal",
    ) -> "BivariateCmap":
        """Creates a bivariate colormap as defined by the corners. \
        The scale for `x` increases from left to right, and \
        the scale for `y` increases from bottom to top. 
        
        Parameters
        ----------
        bottom_left_corner : MPLColor
            The color of the bottom left corner. \
            Note: The `MPLColor` type is any valid matplotlib color.

        top_left_corner : MPLColor
            The color of the top left corner.

        top_right_corner : MPLColor
            The color of the top right corner.

        bottom_right_corner : MPLColor
            The color of the bottom right corner.

        left_to_right_dim : int
            The number of boxes/bins along the `x` axis.

        bottom_to_top_dim : int
            The number of boxes/bins along the `y` axis.

        hue_x_var_name : str, default="x"
            The name of the `x` hue variable.

        hue_y_var_name : str, default="y"
            The name of the `y` hue variable.

        hue_x_binning_method : {"equal", "quantile", "fisher"}, default="equal"
            The method to compute the `x` hue bins.

        hue_y_binning_method : {"equal", "quantile", "fisher"}, default="equal"
            The method to compute the `y` hue bins.

        Returns
        -------
        BivariateCmap
            A bivariate colormap. The shape of the colormap is `shape`.
        """
        return BivariateCmap(
            make_ndarray_cmap_from_corner_colors(
                bottom_left_corner=bottom_left_corner,
                top_left_corner=top_left_corner,
                top_right_corner=top_right_corner,
                bottom_right_corner=bottom_right_corner,
                shape=(left_to_right_dim, bottom_to_top_dim),
            ),
            hue_x_var_name=hue_x_var_name,
            hue_y_var_name=hue_y_var_name,
            hue_x_binning_method=hue_x_binning_method,
            hue_y_binning_method=hue_y_binning_method,
        )

    def set_specific_box_color(
        self,
        bottom_to_top_index: int,
        left_to_right_index: int,
        color: MPLColor,
        inplace: bool = True,
    ) -> "BivariateCmap":
        """Set the color of a specific box in the colormap.

        Parameters
        ----------
        bottom_to_top_index : int
            The index along the `y` axis.

        left_to_right_index : int
            The index along the `x` axis.

        color : MPLColor
            The color to set. \
            Note: The `MPLColor` type is any valid matplotlib color.

        inplace : bool, default=True
            Whether to modify the colormap in-place.

        Returns
        -------
        BivariateCmap
            self
        """
        if inplace:
            self._ndarray_cmap[bottom_to_top_index, left_to_right_index] = MPLColor(
                color
            ).as_rgba()
            return self
        else:
            cmap_copy = self.copy()
            cmap_copy._ndarray_cmap[bottom_to_top_index, left_to_right_index] = (
                MPLColor(color).as_rgba()
            )
            return cmap_copy

    def plot_cmap_heatmap(
        self,
        label_fontsize: int = 12,
        ax: plt.Axes | None = None,
        mpl_imshow_kwargs: dict = {},
    ) -> plt.Axes:
        """Plot the colormap as a heatmap. For bin/box tick label cmap \
        visualization, please refer to `bvcmap.legend()`.

        Parameters
        ----------
        label_fontsize : int, default=12
            The font size of the axis labels.

        ax : plt.Axes | None
            The matplotlib axes to plot the heatmap. \
            If `None`, a new figure and axes are created.

        mpl_imshow_kwargs : dict, default={}
            Additional keyword arguments to pass to `ax.imshow()`.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))
        else:
            fig = ax.figure
        ax.imshow(self._ndarray_cmap, origin="lower", **mpl_imshow_kwargs)
        ax.set_xlabel(
            self._hue_x_var_name + r" $\longrightarrow$", fontsize=label_fontsize
        )
        ax.set_ylabel(
            self._hue_y_var_name + r" $\longrightarrow$", fontsize=label_fontsize
        )
        # remove ticks and ticklabels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # remove boundary
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        fig.tight_layout()
        return ax

    def plot_legend(
        self,
        hue_x: np.ndarray | pd.Series | None = None,
        hue_y: np.ndarray | pd.Series | None = None,
        label_fontsize: int = 12,
        tick_fontsize: int = 10,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot the legend for the colormap.

        Parameters
        ----------
        hue_x : np.ndarray | pd.Series | None
            The `x` hue values. \
            If None, no tick labels are shown.

        hue_y : np.ndarray | pd.Series
            The `y` hue values. \
            If None, no tick labels are shown.

        label_fontsize : int, default=12
            The font size of the axis labels.

        tick_fontsize : int, default=10
            The font size of the tick labels.
            Ignored if `hue_x` is None, `hue_y` is None.

        ax : plt.Axes | None
            The matplotlib axes to plot the legend. \
            If `None`, a new figure and axes are created.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))
        else:
            fig = ax.figure
        if hue_x is None and hue_y is None:
            self.plot_cmap_heatmap(ax=ax, label_fontsize=label_fontsize)
        elif hue_x is not None and hue_y is not None:
            bivariate_legend(
                x_method=self._hue_x_binning_method,
                y_method=self._hue_y_binning_method,
                ax=ax,
                sx=hue_x,
                sy=hue_y,
                cmap=self.to_numpy(),
            )
            ax.set_xlabel(self._hue_x_var_name, fontsize=label_fontsize)
            ax.set_ylabel(self._hue_y_var_name, fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
        fig.tight_layout()
        return ax

    def plot_scatterplot(
        self,
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
            - "equal": Divide the `x` values into equal bins.
            - "quantile": Divide the `x` values into quantiles.
            - "fisher": Use Fisher-Jenks natural breaks classification.

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
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure
        ax.scatter(
            x=x,
            y=y,
            c=self.compute_hues(
                hue_x=hue_x,
                hue_y=hue_y,
                hue_x_binning_method=hue_x_binning_method,
                hue_y_binning_method=hue_y_binning_method,
            ),
            s=s,
            **mpl_scatter_kwargs,
        )
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

    def _repr_pretty_(self, p, cycle):
        """Pretty print representation that displays the colormap as a heatmap."""
        if cycle:
            p.text(self.__repr__())
        else:
            fig, ax = plt.subplots(figsize=(2, 2))
            self.plot_cmap_heatmap(ax=ax)
            plt.show(fig)
            plt.close(fig)

    def to_numpy(self) -> np.ndarray:
        """Return the colormap as a NumPy array.

        Returns
        -------
        np.ndarray
            A NumPy array representing the colormap. Shape: (bottom_to_top_dim, left_to_right_dim, 4).
        """
        return self._ndarray_cmap

    def compute_hues(
        self,
        hue_x: np.ndarray | pd.Series,
        hue_y: np.ndarray | pd.Series,
        hue_x_var_name: str | None = None,
        hue_y_var_name: str | None = None,
        hue_x_binning_method: Literal["equal", "quantile", "fisher"] | None = None,
        hue_y_binning_method: Literal["equal", "quantile", "fisher"] | None = None,
    ) -> pd.Series:
        """Compute the hues for the given `hue_x` and `hue_y` values.

        Parameters
        ----------
        hue_x : np.ndarray | pd.Series
            The `x` values. \
            If a pandas Series is provided, \
            the name of the Series is used as the variable name, \
            overriding the value of `hue_x_var_name` set during initialization.

        hue_y : np.ndarray | pd.Series
            The `y` values. \
            If a pandas Series is provided, \
            the name of the Series is used as the variable name, \
            overriding the value of `hue_y_var_name` set during initialization.

        hue_x_var_name : str, default=None
            The name of the `x` hue variable. \
            If set here, it overrides the value set during initialization.

        hue_y_var_name : str, default=None
            The name of the `y` hue variable. \
            If set here, it overrides the value set during initialization.

        hue_x_binning_method : {"equal", "quantile", "fisher"}, default=None
            The method to compute the `x` hue bins. \
            If set here, it overrides the value set during initialization.
            - "equal": Divide the `x` values into equal bins.
            - "quantile": Divide the `x` values into quantiles.
            - "fisher": Use Fisher-Jenks natural breaks classification.

        hue_y_binning_method : {"equal", "quantile", "fisher"}, default=None
            The method to compute the `y` hue bins. \
            If set here, it overrides the value set during initialization.

        Returns
        -------
        pd.Series
            The hues for the given `x` and `y` values.
        """
        if isinstance(hue_x, pd.Series):
            hue_x_var_name = hue_x.name
        elif isinstance(hue_x, np.ndarray):
            hue_x = pd.Series(hue_x, name=hue_x_var_name)
        if isinstance(hue_y, pd.Series):
            hue_y_var_name = hue_y.name
        elif isinstance(hue_y, np.ndarray):
            hue_y = pd.Series(hue_y, name=hue_y_var_name)

        if hue_x_var_name is not None:
            self._hue_x_var_name = hue_x_var_name
        if hue_y_var_name is not None:
            self._hue_y_var_name = hue_y_var_name
        if hue_x_binning_method is not None:
            self._hue_x_binning_method = hue_x_binning_method
        if hue_y_binning_method is not None:
            self._hue_y_binning_method = hue_y_binning_method

        return bivariate_color(
            sx=hue_x,
            sy=hue_y,
            x_method=self._hue_x_binning_method,
            y_method=self._hue_y_binning_method,
            cmap=self.to_numpy(),
        )

    def copy(self) -> "BivariateCmap":
        """Return a copy of the colormap.

        Returns
        -------
        BivariateCmap
            A copy of the colormap.
        """
        return copy.deepcopy(self)
