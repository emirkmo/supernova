from typing import Any, Iterable, Optional

from dataclasses import asdict, dataclass, field

from astrosn.filters import Filter

from .colors import DEFAULT_COLORS, ColorIterable
from .matplotlib_classes import Axes, Line2D, Patch
from .plot_lc import format_axis, label_axis
from .defaults import DefaultSettings

DEFAULTS = DefaultSettings()


@dataclass
class PlotParameters:
    """Parameters used for the astrosn plots."""

    absmag: bool = False
    site: str = "all"  # or Site | None
    band: Filter = Filter("Unknown")
    shift: float = 0
    error_snake: bool = False
    as_flux: bool = False
    make_labels: bool = True
    label_sites: bool = False
    split_by_site: bool = True
    colors: ColorIterable = field(default_factory=lambda: DEFAULT_COLORS)
    shifts: Optional[list[float]] = None
    mark_unsub: bool = False

    def get(self, key: str) -> Any:
        return getattr(self, key)

    def items(self) -> Iterable[tuple[str, Any]]:
        d = asdict(self)
        d["band"] = self.band
        return d.items()


@dataclass
class AxisParameters:
    """Commonly used Parameters used for matplotlib axes plotting."""

    xlabel: str = "Phase (days)"
    ylabel: str = "Magnitude"
    legend: bool = True
    legend_kwargs: Optional[dict[str, Any]] = None
    invert: bool = False
    log: bool = False
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    x_tick_interval: Optional[float] = None
    y_tick_interval: Optional[float] = None
    x_minors: Optional[int] = 5
    y_minors: Optional[int] = 5
    custom_labels: Optional[list[str]] = None
    custom_handles: Optional[list[Line2D | Patch | Any]] = None

    def get(self, key: str) -> Any:
        return getattr(self, key)

    def items(self) -> Iterable[tuple[str, Any]]:
        return asdict(self).items()

    def format_axis(self, ax: Axes):
        kwargs = self.legend_kwargs or DEFAULTS.legend_kwargs
        ax, leg = label_axis(
            ax,
            self.xlabel,
            self.ylabel,
            self.legend,
            custom_handles=self.custom_handles,
            custom_labels=self.custom_labels,
            **kwargs,
        )
        ax = format_axis(
            ax,
            self.invert,
            self.log,
            self.xlim,
            self.ylim,
            self.x_tick_interval,
            self.y_tick_interval,
            self.x_minors,
            self.y_minors,
        )
        return ax
