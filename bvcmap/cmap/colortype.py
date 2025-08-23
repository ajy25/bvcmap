import matplotlib.colors as mcolors


class MPLColor:
    """A class to represent and validate Matplotlib colors."""

    def __init__(self, color):
        if isinstance(color, MPLColor):
            self.color = color.color
        else:
            self.color = self._validate_color(color)

    def _validate_color(self, color):
        """Validate and convert color to RGBA format."""
        try:
            return mcolors.to_rgba(color)
        except ValueError:
            raise ValueError(f"Invalid Matplotlib color: {color}")

    def as_rgba(self):
        """Return color as an RGBA tuple."""
        return self.color

    def as_hex(self):
        """Return color as a hex string."""
        return mcolors.to_hex(self.color)

    def __repr__(self):
        return f"MPLColor(rgba={self.color})"

    def __eq__(self, other):
        if isinstance(other, MPLColor):
            return self.color == other.color
        return False
