from dataclasses import dataclass, field, asdict, replace
from enum import Enum
import warnings
from typing import Iterable, Optional, Any, Callable, Sequence, TypeGuard, cast
import corner
from lmfit.minimizer import MinimizerResult
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.legend import Legend
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import LineCollection
import inspect

from supernova.filters import Filter
from supernova.sites import Site
from supernova.supernova import SN
from supernova.photometry import (
    HasFlux,
    HasLuminosity,
    HasMag,
    ImplementsFlux,
    ImplementsLuminosity,
    ImplementsMag,
    MagPhotometry,
    Photometry,
    Series,
)
from .colors import ColorIterable, DEFAULT_COLORS, ColorType, Color


# Plot styling reasonable defaults.
sns.set_style("ticks")
sns.set_context("paper", font_scale=2)
mpl.rcParams["axes.labelweight"] = "semibold"
plot_defaults = {
    "label": "",
    "alpha": 1.0,
    "marker": "o",
    "markersize": 5,
    "linestyle": "None",
    "color": None,
}
snake_defaults = {"alpha": 0.7, "zorder": 0}
figure_defaults = {
    "figsize": (8, 6),
    "dpi": 200,
}
legend_defaults = {
    "frameon": True,
    "columnspacing": 0.3,
    "handletextpad": 0.1,
    "handlelength": 1.5,
}
default_color = Color(DEFAULT_COLORS)
PlotType = Callable[..., Axes]


def plot_snake(
    x: Series[float], y: Series[float], z: Series[float], ax: Axes, **plot_kwargs: Any
) -> Axes:
    kwargs = {**snake_defaults, **plot_kwargs}
    ax.fill_between(x, y - z, y + z, **kwargs)  # type: ignore
    return ax


def plot_mag(
    phot: ImplementsMag,
    ax: Axes,
    shift: float = 0,
    error_snake: bool = False,
    **plot_kwargs: Any,
) -> Axes:
    if error_snake:
        ps = phot.mag + shift
        return plot_snake(phot.phase, phot.mag + shift, phot.mag_err, ax, **plot_kwargs)
    kwargs = {**plot_defaults, **plot_kwargs}
    ax.errorbar(phot.phase, phot.mag + shift, phot.mag_err, **kwargs)
    return ax


def plot_flux(
    phot: ImplementsFlux,
    ax: Axes,
    shift: float = 0,
    error_snake: bool = False,
    **plot_kwargs: Any,
) -> Axes:
    if error_snake:
        return plot_snake(
            phot.phase, phot.flux + shift, phot.flux_err, ax, **plot_kwargs
        )
    kwargs = {**plot_defaults, **plot_kwargs}
    ax.errorbar(phot.phase, phot.flux + shift, phot.flux_err, **kwargs)
    return ax


def plot_lum(phot: ImplementsLuminosity, ax: Axes, **plot_kwargs: Any) -> Axes:
    kwargs = {**plot_defaults, **plot_kwargs}
    ax.errorbar(phot.phase, phot.lum, phot.lum_err, **kwargs)
    return ax


def plot_lim(
    phot: Photometry,
    ax: Axes,
    shift: float = 0,
    as_flux: bool = False,
    **plot_kwargs: Any,
) -> Axes:
    lim_defaults = {"label": "", "marker": "v", "markerfacecolor": "None"}
    kwargs = {**plot_defaults, **lim_defaults, **plot_kwargs}
    if as_flux:
        if isinstance(phot, HasFlux):
            ax.plot(phot.phase, phot.flux + shift, **kwargs)
    else:
        if not isinstance(phot, HasMag):
            raise TypeError(
                "Cannot plot limits as magnitudes if photometry does not have `mag`."
            )
        ax.plot(phot.phase, phot.mag + shift, **kwargs)
    return ax


def plot_lims(
    sn: SN,
    band: Filter,
    ax: Axes,
    absmag: bool = False,
    as_flux: bool = False,
    **plot_kwargs: Any,
) -> Axes:
    """
    Plot limits for a given band.
    """
    phot = sn.band(band.name, return_absmag=absmag, lims=True)
    ax = plot_lim(phot, ax, band.plot_shift, as_flux, **plot_kwargs)
    return ax


def plot_band(
    sn: SN,
    band: Filter,
    ax: Axes,
    site: str = "all",
    plot_type: PlotType = plot_mag,
    absmag: bool = False,
    mark_unsub: bool = False,
    **kwargs: Any,
) -> Axes:
    """
    Plot a single band.
    """
    _site = sn.sites[site] if site != "all" else site
    marker = kwargs.get("marker", get_marker(_site))

    band_phot = sn.band(band.name, site=site, return_absmag=absmag)
    ax = plot_type(band_phot, ax, **kwargs)
    if not mark_unsub or sn.sub_only:
        return ax

    # Put an X tick mark on the plot for unsubtracted photometry.
    sn.sub_only = True
    sub_phot = sn.band(band.name, site=site, return_absmag=absmag)
    sn.sub_only = False
    unsub_phot = sn.band(band.name, site=site, return_absmag=absmag)
    # Get ONLY unsubtracted photometry:
    conds = [not any(np.isclose(jd, sub_phot.jd, atol=1e-7)) for jd in unsub_phot.jd]
    unsub_phot = unsub_phot.masked(conds)

    unsub_marker = "x"
    newkwargs = {
        **kwargs,
        **{"marker": unsub_marker, "color": "black", "linestyle": "None", "label": ""},
    }
    newkwargs["markersize"] = newkwargs.get(
        "markersize", plot_defaults["markersize"] + 2
    )
    newkwargs["zorder"] = 5
    ax = plot_type(unsub_phot, ax, **newkwargs)
    return ax


def _is_single_figure(axs: list[list[Axes]] | Axes) -> TypeGuard[Axes]:
    return not isinstance(axs, list)


def _is_multi_figure(axs: list[list[Axes]] | Axes) -> TypeGuard[list[list[Axes]]]:
    return isinstance(axs, list)


def make_figure(**fig_kwargs: Any) -> tuple[Figure, list[list[Axes]] | Axes]:
    fig_kwargs = figure_defaults | fig_kwargs
    fig, axs = plt.subplots(**fig_kwargs)
    if _is_single_figure(axs):
        return fig, axs
    return fig, axs


def make_single_figure(**fig_kwargs: Any) -> tuple[Figure, Axes]:
    fig, axs = make_figure(**fig_kwargs)
    if _is_single_figure(axs):
        return fig, axs
    elif _is_multi_figure(axs):
        ax = cast(Axes, axs[0][0])
        return fig, ax

    raise TypeError("Unknown axes type.")


def make_plot_shifts(
    sn: SN,
    shifts: Optional[Sequence[float | int]] = None,
    reversed_for_abs: bool = False,
) -> SN:
    """
    Modify state of SN object with the plot shift
    for each filter set. If Shifts are not given
    then they will be +/- 1 for each filter.
    By default, first shift is negative, middle is zero, and last is positive.
    If reversed_for_abs is True, then the first shift is positive, middle is zero, and last is negative
    for use in plotting in absolute magnitude space. In that case, input shifts order is also reversed.
    """
    n_bands = len(sn.bands)
    if shifts is not None and len(shifts) < n_bands:
        raise ValueError(
            "shifts must be the same or greater "
            "length as the number of bands or `None`."
        )

    if shifts is None:
        use_shifts = np.linspace(0, n_bands, n_bands + 1) - (n_bands // 2)
    else:
        use_shifts = shifts

    if reversed_for_abs:
        use_shifts = use_shifts[::-1]

    for i, f in enumerate(sn.bands.values()):
        f.plot_shift = use_shifts[i]

    return sn


def make_plot_colors(sn: SN, colors: Optional[ColorIterable] = None) -> SN:
    """
    Modify state of SN object with the plot color
    for each filter set. If colors are not given
    then they will be a default set of colors.
    colors should be a dict or list of matplotlib
    compatible colors (ex.:3 floats or a color string).
    """
    if colors is None:
        use_colors = sns.color_palette("Paired", n_colors=len(sn.bands))
    else:
        use_colors = colors
    # if isinstance(use_colors, sns.colors._ColorPalette.LienarSeg
    if isinstance(use_colors, LinearSegmentedColormap | ListedColormap):
        raise TypeError(
            "Color map LinearSegmentedColormap | ListedColormap is not a valid color iterable."
        )

    if len(use_colors) < len(sn.bands):
        raise ValueError(
            "colors must be the same or greater "
            "length as the number of bands or `None`."
        )
    if isinstance(use_colors, list):
        use_colors = {
            band.name: color for band, color in zip(sn.bands.values(), use_colors)
        }
    for band in sn.bands.values():
        band.plot_color = use_colors[band.name]
    return sn


def _get_markers_to_use() -> list[str]:
    """
    Get a list of markers to use for plotting.
    """
    markers = ["o", "d", "s"]
    # Pad smartly with markers using more legible markers first.
    filled_markers = list(Line2D.filled_markers)
    # remove markers that are already in use
    for m in markers:
        filled_markers.remove(m)

    # pad filled_markers with unfilled markers
    markers += filled_markers
    markers += ["1", "2", "3", "4", "+", "X"]
    return markers


def get_site_markers(site_markers: dict[int, Optional[str]]) -> dict[int, str]:
    """
    Given a dictionary of site markers: mapping of int to str or None
    return dictionary of int to str using more legible markers first.
    """
    markers = _get_markers_to_use()
    new_site_markers = {}
    if len(site_markers) > len(markers):
        raise ValueError("Too many sites to plot with unique markers.")
    used_markers = set(site_markers.values())
    if None not in used_markers:
        # unnecessary, just for pylance:
        new_site_markers = {k: v for k, v in site_markers.items() if v is not None}
        return new_site_markers
    used_markers.remove(None)
    for m in used_markers:
        if m in markers:
            markers.remove(m)

    for site, marker in site_markers.items():
        if marker is None:
            new_site_markers[site] = markers.pop(0)
        else:
            new_site_markers[site] = marker

    return new_site_markers


def get_label(filt: Filter, site_name: Optional[str] = "") -> str:
    """
    Get a label for a filter, optionally with a site name.
    If sitename is not desired, pass an empty string or None
    """
    site_name = "" if site_name is None else site_name
    if filt.plot_shift == 0:
        label = f"${filt.name}$ {site_name}"
        return label.strip()
    shift_label = f"{filt.plot_shift:.1f}"
    if not shift_label.startswith("-") and shift_label != "":
        shift_label = f"+{shift_label}"
    if shift_label.endswith(".0"):
        shift_label = shift_label[:-2]
    label = f"${filt.name}${shift_label} {site_name}"
    return label.strip()


def get_marker(site: Site | str) -> str:
    """
    Get a marker for a site.
    """
    default_marker = _get_markers_to_use()[0]
    if site == "all":
        return default_marker
    if isinstance(site, str):
        raise TypeError("site must be a Site object or 'all'")
    marker = site.marker
    if marker is None:
        return default_marker
    return marker


def get_color(band: Filter, colors: Color = default_color) -> ColorType:
    color = band.plot_color
    if color is not None:
        return color
    return colors.get_color(band.name)


def get_plot_kwargs(
    site: str | Site,
    band: Filter,
    site_label: str = "",
    update_label: bool = True,
    **plot_kwargs: Any,
) -> dict:
    if plot_kwargs is None:
        plot_kwargs = {}
    # if plot_kwargs.get('marker', None) == None:
    plot_kwargs["marker"] = get_marker(site)
    # if plot_kwargs.get('color', None) is None:
    plot_kwargs["color"] = get_color(band)
    if update_label:
        plot_kwargs["label"] = get_label(band, site_label)
    return {**plot_defaults, **plot_kwargs}


def plot_all_bands(
    sn: SN,
    ax: Axes,
    site: str = "all",
    plot_type: PlotType = plot_mag,
    absmag: bool = False,
    make_labels: bool = True,
    mark_unsub: bool = False,
    **plot_kwargs: Any,
) -> Axes:
    # _ax: Axes, _sn: SN, _site: str, _site_label: str, _marker: str, _plot_kwargs: dict,
    #           update_label: bool = True
    """
    Plot all bands of a site, or of all sites if not given, on a single axis.
    """
    _site = "all"
    if site != "all":
        _site = sn.sites[site]

    for band in sn.bands.values():
        plot_kwargs = get_plot_kwargs(
            _site, band, update_label=make_labels, **plot_kwargs
        )
        if not make_labels:
            plot_kwargs["label"] = None
        try:
            band_phot = sn.band(
                band.name,
                site=site,
                return_absmag=absmag,
                flux=plot_type == plot_flux,
                lims=plot_type == plot_lims,
            )
        except KeyError as e:
            warnings.warn(
                f"Could not get photometry for {band.name} in {sn.name},\n got exception: {e}"
            )
            band_phot = None

        if band_phot is not None and len(band_phot) > 0:
            ax = plot_band(
                sn=sn,
                band=band,
                ax=ax,
                site=site,
                plot_type=plot_type,
                absmag=absmag,
                shift=band.plot_shift,
                mark_unsub=mark_unsub,
                **plot_kwargs,
            )
    return ax


def get_unsub_mark(
    markersize: int = 7, label: str = "Unsub"
) -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D(
            [0], [0], color="black", marker="x", linestyle="None", markersize=markersize
        )
    ]
    labels = [label]
    return handles, labels


def get_photcol(phot: Photometry, plot_type: PlotType) -> Series[float]:
    if plot_type == plot_mag:
        if isinstance(phot, HasMag):
            return phot.mag
    if plot_type == plot_flux:
        if isinstance(phot, HasFlux):
            return phot.flux
    if plot_type == plot_lum:
        if isinstance(phot, HasLuminosity):
            return phot.lum
    if plot_type == plot_lims:
        if isinstance(phot, HasFlux):
            return phot.flux
        elif isinstance(phot, HasMag):
            return phot.mag

    raise ValueError(f"Invalid plot type: {plot_type}")



def plot_split_by_site(
    sn: SN,
    ax: Axes,
    plot_type: PlotType = plot_mag,
    absmag: bool = False,
    label_sites: bool = True,
    mark_unsub: bool = False,
    **plot_kwargs: Any,
) -> Axes:
    """
    Plot all bands of a supernova, split by site.
    """
    sn.sites.update_markers(get_site_markers(sn.sites.markers))
    #first = True
    legend_labels = []
    legend_handles = []
    for site in sn.sites:
        make_labels = False
        ax = plot_all_bands(
            sn,
            ax,
            site=site.name,
            plot_type=plot_type,
            absmag=absmag,
            make_labels=make_labels,
            mark_unsub=mark_unsub,
            **plot_kwargs,
        )
        #first = False
        if label_sites: # add site as black markers
            legend_labels.append(site.name)
            usemarker = site.marker if site.marker is not None else _get_markers_to_use()[0]
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker=usemarker,
                    linestyle="None",
                    markersize=plot_kwargs.get("markersize", 6),
                )
            )

    for band in sn.bands:
        phot = sn.band(band, flux=plot_type == plot_flux,
                lims=plot_type == plot_lims)
        phot_col = get_photcol(phot, plot_type=plot_type)
        if len(phot_col.dropna()) > 0:


    
            temp_kwargs = get_plot_kwargs('all', sn.bands[band], **plot_kwargs)
            legend_labels.append(temp_kwargs['label'])

            line = Line2D(
                    [0],
                    [0],
                    color=temp_kwargs.get('color', sn.bands[band].plot_color),
                    marker=_get_markers_to_use()[0],
                    linestyle=temp_kwargs.get('linestyle', 'None'),
                    markersize=temp_kwargs.get("markersize", 6),
                )
            # barline = LineCollection(np.empty((2,2,2)))
            # err = ErrorbarContainer((line, [line], [barline]), has_xerr=True, has_yerr=True)
            legend_handles.append(line)

    leg, ax = _safe_draw_legend(ax, legend_handles, legend_labels)
    return ax


def update_legend_handles_labels(
    ax: Axes, legend_handles: list[Any], legend_labels: list[str]
) -> tuple[Axes, list[Any], list[str]]:
    handles, labels = _safe_get_legend_handles_labels(ax)
    for handle, label in zip(handles, labels):
        if label in legend_labels:
            continue
        legend_handles.append(handle)
        legend_labels.append(label)
    return ax, legend_handles, legend_labels


def _get_legend_handles_labels(legend: Legend) -> tuple[list[Any], list[str]]:
    handles = legend.legendHandles
    labels = legend.get_texts()
    labels = [label.get_text() for label in labels]
    return handles, labels


def _safe_get_legend_handles_labels(ax: Axes) -> tuple[list[Any], list[str]]:
    # First, get the handles and labels from the axis, if exists
    if isinstance(ax.legend_, Legend):
        handles, labels = _get_legend_handles_labels(ax.legend_)
        if len(handles) > 0:
            return handles, labels

    # If we didn't find handles, try to get them from the figure/axes
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def _safe_draw_legend(
    ax: Axes, handles: list[Any], labels: list[str], **legend_kwargs: Any
) -> tuple[Legend, Axes]:
    legend_kwargs = legend_defaults | legend_kwargs
    ax, legend_handles, legend_labels = update_legend_handles_labels(
        ax, handles, labels
    )
    leg = ax.legend(legend_handles, legend_labels, **legend_kwargs)
    ax.legend_ = leg
    return leg, ax


def plot_abs_mag(
    sn: SN,
    ax: Optional[Axes] = None,
    split_by_site: bool = False,
    label_sites: bool = False,
    mark_unsub: bool = False,
    **plot_kwargs: Any,
) -> Axes:

    if ax is None:
        fig, ax = make_single_figure()

    if split_by_site:
        ax = plot_split_by_site(
            sn,
            ax,
            plot_type=plot_mag,
            absmag=True,
            label_sites=label_sites,
            mark_unsub=mark_unsub,
            **plot_kwargs,
        )
        return ax

    ax = plot_all_bands(
        sn,
        ax,
        site="all",
        plot_type=plot_mag,
        absmag=True,
        mark_unsub=mark_unsub,
        **plot_kwargs,
    )
    return ax


def label_axis(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    legend: bool = True,
    custom_labels: Optional[list[str]] = None,
    custom_handles: Optional[list[Line2D | Patch | Any]] = None,
    **legend_kwargs: Any,
) -> tuple[Axes, Optional[Legend]]:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)  # fontweigt = 'semibold'
    if legend:  # draw legend
        legend_kwargs = legend_defaults | legend_kwargs

        # Safely get handles and labels from the axis
        # Append custom handles and labels if given
        # Then (re)draw the legend
        if custom_labels is not None and custom_handles is not None:
            leg, ax = _safe_draw_legend(
                ax, custom_handles, custom_labels, **legend_kwargs
            )
            return ax, leg

        handles, labels = _safe_get_legend_handles_labels(ax)
        leg, ax = _safe_draw_legend(ax, handles, labels, **legend_kwargs)
        return ax, leg
    return ax, None


def format_axis(
    ax: Axes,
    invert: bool = False,
    log: bool = False,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    x_tick_interval: Optional[float] = None,
    y_tick_interval: Optional[float] = None,
    x_minors: Optional[int] = 5,
    y_minors: Optional[int] = 5,
) -> Axes:
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if x_tick_interval is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
    if y_tick_interval is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    if x_minors is not None:
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minors))
    if x_minors is not None:
        ax.yaxis.set_minor_locator(AutoMinorLocator(y_minors))
    if invert:
        ax.invert_yaxis()
    if log:
        ax.set_yscale("log")
    return ax


def plot_emcee(res: MinimizerResult):
    emcee_plot = corner.corner(
        res.flatchain,
        labels=res.var_names,
        truths=list(res.params.valuesdict().values()),
    )
    return emcee_plot

# class GenericPlot:
#     def __init__(self, f: PlotType):
#         self.f = f

#     def __call__(self, *args: Any, **kwargs: Any) -> Axes:
#         return self.f(*args, **kwargs)

class SNPlotTypes(Enum):
    ABS_MAG = plot_abs_mag
    LIM = plot_lims
    BAND = plot_band
    BANDS = plot_all_bands
    SITES = plot_split_by_site

    @classmethod
    def values(cls):
        return cls.__dict__.values()

    def __call__(self, *args: Any, **kwargs: Any) -> Axes:
        return self.value(*args, **kwargs)

class PhotPlotTypes(Enum):
    MAG = plot_mag
    LIM = plot_lim
    FLUX = plot_flux
    LUM = plot_lum

    @classmethod
    def values(cls):
        return cls.__dict__.values()


@dataclass
class PlotParameters:
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
        kwargs = self.legend_kwargs or legend_defaults
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


class Plotter:
    def __init__(
        self,
        sn: SN,
        plot_type: PlotType = SNPlotTypes.ABS_MAG,
        params: PlotParameters = PlotParameters(),
        axis_params: AxisParameters = AxisParameters(),
        **plot_kwargs: Any,
    ):
        self.sn = sn
        self.sn.sites.markers = cast(
            dict[int, str | None], get_site_markers(sn.sites.markers)
        )
        self.plot_type = plot_type
        self.signature = list(inspect.signature(self.plot_type).parameters.keys())
        self.plot_params = params
        self.plot_kwargs = plot_kwargs
        self.axis_params = axis_params

    def update_plot_type(self, plot_type: PlotType, **plot_kwargs: Any):
        self.plot_type = plot_type
        self.signature = list(inspect.signature(plot_type).parameters.keys())
        self.plot_kwargs = plot_kwargs

    def get_sn_or_phot(self) -> SN | Photometry:
        if self.plot_type in SNPlotTypes.values():
            return self.sn
        elif self.plot_type in PhotPlotTypes.values():
            lims = self.plot_type == PhotPlotTypes.LIM
            return self.sn.band(
                self.plot_params.band.name,
                site=self.plot_params.site,
                return_absmag=self.plot_params.absmag,
                flux=self.plot_params.as_flux,
                lims=lims,
            )
        raise ValueError(f"Plot type {self.plot_type} not recognised")

    def get_plot_params(self) -> dict[str, Any]:
        kwargs = {k: v for k, v in self.plot_params.items() if k in self.signature}
        return kwargs | self.plot_kwargs

    def save_current_legend(self, ax: Axes):
        if ax.get_legend() is not None:
            h, l = ax.get_legend_handles_labels()
            if self.axis_params.custom_labels is None:
                self.axis_params.custom_labels = l
            if self.axis_params.custom_handles is None:
                self.axis_params.custom_handles = h
            ax, h, l = update_legend_handles_labels(
                ax, self.axis_params.custom_handles, self.axis_params.custom_labels
            )
            self.axis_params.custom_handles = h
            self.axis_params.custom_labels = l

    def plot(
        self, ax: Optional[Axes] = None, **fig_kwargs: Any
    ) -> tuple[Figure | None, Axes]:
        kwargs = self.get_plot_params()
        arg = self.get_sn_or_phot()
        fig = None
        if ax is None:
            fig, ax = make_single_figure(**fig_kwargs)
        ax = self.plot_type(arg, ax=ax, **kwargs)
        self.save_current_legend(ax)
        return fig, ax

    def label_axis(
        self,
        ax: Axes,
        xlabel: str,
        ylabel: str,
        legend: bool = True,
        **legend_kwargs: Any,
    ) -> tuple[Axes, Optional[Legend]]:
        self.axis_params = replace(
            self.axis_params,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            legend_kwargs=legend_kwargs,
        )
        return label_axis(ax, xlabel, ylabel, legend, **legend_kwargs)

    def format_axis(
        self,
        ax: Axes,
        invert: bool = False,
        log: bool = False,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        x_tick_interval: Optional[float] = None,
        y_tick_interval: Optional[float] = None,
        x_minors: Optional[int] = 5,
        y_minors: Optional[int] = 5,
    ) -> Axes:
        self.axis_params = replace(
            self.axis_params,
            invert=invert,
            log=log,
            xlim=xlim,
            ylim=ylim,
            x_tick_interval=x_tick_interval,
            y_tick_interval=y_tick_interval,
            x_minors=x_minors,
            y_minors=y_minors,
        )
        return format_axis(
            ax,
            invert,
            log,
            xlim,
            ylim,
            x_tick_interval,
            y_tick_interval,
            x_minors,
            y_minors,
        )

    def plot_emcee(self, res: MinimizerResult):
        return plot_emcee(res)

    def setup_plot(self):
        sn = self.sn
        # Can also set automatic plot shifts & colors, but here using custom ones.
        sn = make_plot_shifts(
            sn=sn,
            reversed_for_abs=self.plot_params.absmag,
            shifts=self.plot_params.shifts,
        )
        sn = make_plot_colors(sn=sn, colors=self.plot_params.colors)
        self.sn = sn

    def legend_additions(self, ax: Axes) -> None:
        if self.plot_params.mark_unsub:
            handles, labels = get_unsub_mark()
            ax, handles, labels = update_legend_handles_labels(ax, handles, labels)
            if self.axis_params.custom_labels is None:
                self.axis_params.custom_handles = handles
                self.axis_params.custom_labels = labels
            else:
                for handle, label in zip(handles, labels):
                    if label not in self.axis_params.custom_labels:
                        self.axis_params.custom_labels.append(label)
                        if self.axis_params.custom_handles is None:
                            raise ValueError(
                                f"Custom handles should not be None when custom labels are {self.axis_params.custom_labels}"
                            )
                        self.axis_params.custom_handles.append(handle)

    def __call__(
        self, ax: Optional[Axes] = None, **fig_kwargs: Any
    ) -> tuple[Optional[Figure], Axes]:
        self.setup_plot()
        fig, ax = self.plot(ax, **fig_kwargs)

        ax = self.axis_params.format_axis(ax)
        return fig, ax
