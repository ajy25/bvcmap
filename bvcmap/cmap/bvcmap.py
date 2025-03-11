import xycmap
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
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

    return xycmap.custom_xycmap(
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

    def __init__(self, cmap: np.ndarray | Any):
        """Initialize a bivariate colormap.

        Parameters
        ----------
        cmap : np.ndarray | BvariateCmap
            A NumPy array representing the colormap.
            The shape of the array is (bottom_to_top_dim, left_to_right_dim, 4).
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
        else:
            self._ndarray_cmap = cmap.to_numpy()

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
    ) -> "BivariateCmap":
        """Creates a bivariate colormap as defined by the corners.
        The scale for `x` increases from left to right, and \
        the scale for `y` increases from bottom to top. 
        
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

        left_to_right_dim : int
            The number of boxes/bins along the `x` axis.

        bottom_to_top_dim : int
            The number of boxes/bins along the `y` axis.

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
            )
        )

    def plot_cmap_heatmap(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot the colormap as a heatmap.

        Parameters
        ----------
        ax : plt.Axes | None
            The matplotlib axes to plot the heatmap.
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
        ax.imshow(self._ndarray_cmap, origin="lower")
        ax.set_xlabel(r"$x \rightarrow$")
        ax.set_ylabel(r"$y \rightarrow$")
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

    def _repr_pretty_(self, p, cycle):
        """Pretty print representation that displays the colormap as a heatmap."""
        if cycle:
            p.text(self.__repr__())
        else:
            fig, ax = plt.subplots(figsize=(2, 2))
            self.plot_cmap_heatmap(ax)
            plt.show(fig)
            plt.close(fig)

    def to_numpy(self) -> np.ndarray:
        """Return the colormap as a NumPy array.

        Returns
        -------
        np.ndarray
            A NumPy array representing the colormap.
            Shape: (n, m, 4)
        """
        return self._ndarray_cmap

    def hues(self, hue_x: np.ndarray, hue_y: np.ndarray) -> np.ndarray:
        """Compute the hues for the given `hue_x` and `hue_y` values.

        Parameters
        ----------
        hue_x : np.ndarray
            The `x` values.

        hue_y : np.ndarray
            The `y` values.

        Returns
        -------
        np.ndarray
            The hues for the given `x` and `y` values.
        """
        return xycmap.bivariate_color(
            sx=hue_x, sy=hue_y, cmap=self.to_numpy()
        ).to_numpy()
