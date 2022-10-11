from typing import Iterator, Optional
import seaborn as sns

DEFAULT_COLORS = {
    "u": "lightsteelblue",
    "g": "forestgreen",
    "r": "tab:red",
    "i": "purple",
    "z": "fuchsia",
    "B": "royalblue",
    "V": "olivedrab",
    "R": "darkred",
    "J": "sandybrown",
    "H": "saddlebrown",
    "K": "coral",
}

GENERIC_COLORS = sns.color_palette()

ColorType = str | tuple[float, float, float]
ColorIterable = (
    dict[str, str] | dict[str, ColorType] | list[ColorType] | sns.palettes._ColorPalette 
)


class Color:
    def __init__(self, colors: ColorIterable = DEFAULT_COLORS) -> None:
        self.colors = colors
        self._color_iter = self._set_color_iter()

    def _set_color_iter(self) -> Iterator[ColorType]:
        return iter(
            self.colors.values() if isinstance(self.colors, dict) else self.colors
        )

    def _refresh_color_iter(self) -> None:
        self._color_iter = self._set_color_iter()

    def get_next_color(self) -> ColorType:
        try:
            return next(self._color_iter)
        except StopIteration:
            self._refresh_color_iter()
            return next(self._color_iter)

    def get_color(self, band: Optional[str] = None) -> ColorType:
        if band is None:
            return self.get_next_color()
        if isinstance(self.colors, dict):
            return self.colors.get(band, self.get_next_color())
        return self.get_next_color()
