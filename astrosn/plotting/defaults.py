from functools import partial
from typing import Any, Optional, cast

import matplotlib as mpl
import seaborn as sns
from dataclasses import dataclass, field

from .colors import GENERIC_COLORS, Color
from .settings import PlotSettings

legend_defaults = {
    "frameon": True,
    "columnspacing": 0.3,
    "handletextpad": 0.1,
    "handlelength": 1.5,
}

plot_defaults = {
    "label": "",
    "alpha": 1.0,
    "marker": "o",
    "markersize": 5,
    "linestyle": "None",
    "color": None,
}

figure_defaults = {
    "figsize": (8, 6),
    "dpi": 200,
}

snake_defaults = {"alpha": 0.7, "zorder": 0}


def _default_factory(d: dict[str, Any]) -> dict[str, Any]:
    return d


@dataclass
class DefaultSettings(PlotSettings):
    seaborn_context: dict[str, float] = field(
        default_factory=partial(sns.plotting_context, "paper")
    )  # seaborn context. See `seaborn.plotting_context`
    seaborn_style: str = "ticks"
    font_scale: float = 2  # scales all other font sizes. By Seaborn.
    legend_kwargs: dict[str, Any] = field(
        default_factory=partial(_default_factory, legend_defaults)
    )  # kwargs for the legend
    figure_kwargs: dict[str, Any] = field(
        default_factory=partial(_default_factory, figure_defaults)
    )  # kwargs for plt.figure
    plot_kwargs: dict[str, Any] = field(
        default_factory=partial(_default_factory, plot_defaults)
    )  # kwargs for the plot function at call time.
    snake_kwargs: dict[str, Any] = field(
        default_factory=partial(_default_factory, snake_defaults)
    )  # kwargs for the error snake plot (See `plt.fill_between`)
    color: Color = field(default_factory=Color)  # `.colors.Color` instantiated with a ColorIterable.
    # Color Iterable Either map bands to colors or uses a color palette that cycles through colors.
    labelweight: str = "semibold"  # semibold is another good option.
    rc: dict[str, Any] = field(default_factory=dict)
    fontsize: Optional[str | float] = None  # overwrite fontsize for the plot
    scaled: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.set_defaults()

    def set_defaults(self):
        if not self.scaled:
            self._scale_fontsize()
            self.scaled = True
        sns.set_theme(
            context=self.seaborn_context, style=self.seaborn_style, font_scale=self.font_scale, rc=self.get_rc()
        )
        # mpl.rcParams.update(self.get_rc())

    def _scale_fontsize(self) -> None:
        """Scale fontsize by font_scale."""
        # Now independently scale the fonts
        font_keys = [
            "axes.labelsize",
            "axes.titlesize",
            "legend.fontsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "font.size",
        ]
        font_dict = {k: self.seaborn_context[k] * self.font_scale for k in font_keys}
        self.seaborn_context.update(font_dict)

    def get_rc(self):
        # axisfontsize = {"axes.fontsize": self.fontsize} if self.fontsize else {}
        axisfontsize = {}
        return {
            "axes.labelweight": self.labelweight,
            **self.rc,
            **axisfontsize,
        }

    @property
    def axis_kwargs(self) -> dict[str, Any]:
        return {
            "fontsize": self.seaborn_context["font.size"],
            "fontweight": self.labelweight,
        }


max_seaborn_context: dict[str, float] = {
    "axes.linewidth": 2,
    "grid.linewidth": 2,
    "lines.linewidth": 1,
    "lines.markersize": 8,
    "patch.linewidth": 2,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.width": 2,
    "ytick.minor.width": 2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 12,
}


max_rc: dict[str, Any] = {
    "xtick.direction": "in",
    "ytick.direction": "in",
    "errorbar.capsize": 0.0,
    # Font:
    "font.weight": "bold",
    # ALL Legend:
    "legend.loc": "best",  # or 'upper right' or (0.99, 0.8), etc.
    "legend.borderaxespad": 0.5,
    "legend.borderpad": 0.4,
    "legend.columnspacing": 0.3,
    "legend.edgecolor": "0.8",
    "legend.facecolor": "inherit",
    "legend.fancybox": True,
    "legend.framealpha": 0.8,
    "legend.frameon": True,
    "legend.handleheight": 0.7,
    "legend.handlelength": 1 / 5,
    "legend.handletextpad": 0.1,
    "legend.labelcolor": "None",
    "legend.labelspacing": 0.5,
    "legend.markerscale": 1.0,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    "legend.shadow": False,
    # ALL ticks:
    "xtick.alignment": "center",
    "xtick.bottom": True,
    "xtick.color": "black",
    "xtick.labelbottom": True,
    "xtick.labelcolor": "inherit",
    "xtick.labeltop": False,
    "xtick.major.bottom": True,
    "xtick.major.pad": 3.5,
    "xtick.major.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.pad": 3.4,
    "xtick.minor.top": True,
    "xtick.minor.visible": False,
    "xtick.top": False,
    "yaxis.labellocation": "center",
    "ytick.alignment": "center_baseline",
    "ytick.color": "black",
    "ytick.labelcolor": "inherit",
    "ytick.labelleft": True,
    "ytick.labelright": False,
    "ytick.left": True,
    "ytick.major.left": True,
    "ytick.major.pad": 3.5,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.pad": 3.4,
    "ytick.minor.right": True,
    "ytick.minor.visible": False,
    "ytick.right": False,
}


@dataclass
class MaxSettings(DefaultSettings):
    seaborn_context: dict[str, float] = field(default_factory=partial(_default_factory, max_seaborn_context))
    labelweight: str = "bold"  # semibold is another good option.
    rc: dict[str, Any] = field(default_factory=partial(_default_factory, max_rc))
    font_scale: float = 2


@dataclass
class GenericSettings(DefaultSettings):
    color: Color = Color(colors=GENERIC_COLORS)
    labelweight: str = "semibold"
