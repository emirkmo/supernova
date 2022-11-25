from typing import Any, Optional, Protocol

from dataclasses import dataclass

from .colors import Color


@dataclass
class PlotSettings(Protocol):
    """Container for different types of plot settings.
    If a setting is defined multiple times, those defined
    further down will override the ones from above.

    Parameters

    """

    seaborn_context: dict[str, float]  # seaborn context. See `seaborn.plotting_context`
    seaborn_style: str
    font_scale: float  # scales all other font sizes. By Seaborn.
    legend_kwargs: dict[str, Any]  # kwargs for the legend
    figure_kwargs: dict[str, Any]  # kwargs for plt.figure
    plot_kwargs: dict[str, Any]  # kwargs for the plot function at call time.
    snake_kwargs: dict[str, Any]  # kwargs for the snake plot
    color: Color  # A color palette for the plot such as `Color(sns.color_palette("colorblind"))``
    fontsize: Optional[str | float] = None  # fontsize for the plot

    def set_defaults(self) -> None:
        ...

    def __post_init__(self) -> None:
        ...

    @property
    def axis_kwargs(self) -> dict[str, Any]:
        ...
