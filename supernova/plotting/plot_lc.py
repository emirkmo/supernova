import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib as mpl
from supernova.plotting.colors import ColorIterable
from supernova.supernova import SN, Photometry, MagPhot, FluxPhot
from supernova.filters import Filter
from typing import Optional, Any
from numpy.typing import ArrayLike
import warnings
import seaborn as sns
from .colors import DEFAULT_COLORS

# Plot styling reasonable defaults.
sns.set_style("ticks")
sns.set_context("paper", font_scale=2)
mpl.rcParams['axes.labelweight'] = 'semibold'
plot_defaults = {"label": "",
                 "alpha": 1.0,
                 "marker": 'o',
                 "markersize": 5,
                 "linestyle": 'None',
                 'color': None}

figure_defaults = {"figsize": (8, 6),
                   "dpi": 200, }
legend_defaults = {"frameon": True,
                   "columnspacing": 0.3,
                   "handletextpad": 0.1,
                   "handlelength": 1.5}


def plot_mag(band: MagPhot, ax: plt.Axes, shift: float = 0, **plot_kwargs: Any) -> plt.Axes:
    kwargs = {**plot_defaults, **plot_kwargs}
    ax.errorbar(band.phase, band.mag+shift, band.mag_err, **kwargs)
    return ax


def plot_flux(band: FluxPhot, ax: plt.Axes, shift: float = 0, **plot_kwargs: Any) -> plt.Axes:
    kwargs = {**plot_defaults, **plot_kwargs}
    ax.errorbar(band.phase, band.flux+shift, band.flux_err, **kwargs)
    return ax


def plot_lim(band: Photometry, ax: plt.Axes, shift: float = 0, as_flux: bool = False, **plot_kwargs: Any) -> plt.Axes:
    lim_defaults = {"label": "", "marker": 'v', "markerfacecolor": 'None'}
    kwargs = {**plot_defaults, **lim_defaults, **plot_kwargs}
    if as_flux:
        ax.plot(band.phase, band.flux+shift, **kwargs)
    else:
        ax.plot(band.phase, band.mag+shift, **kwargs)
    return ax


def plot_lims(sn: SN, filt: Filter, ax: plt.Axes, absmag: bool = False, as_flux: bool = False, **plot_kwargs: Any) -> plt.Axes:
    """
    Plot limits for a given band.
    """
    band = sn.band(filt.name, return_absmag=absmag, lims=True)
    ax = plot_lim(band, ax, filt.plot_shift, as_flux, **plot_kwargs)
    return ax


def make_figure(fig_kwargs: Optional[dict] = None) -> tuple[plt.Figure, tuple[plt.Axes] | plt.Axes]:
    if fig_kwargs is None:
        fig_kwargs = figure_defaults
    fig, axs = plt.subplots(**fig_kwargs)
    return fig, axs


def make_plot_shifts(sn: SN, shifts: Optional[ArrayLike] = None,
                     reversed_for_abs: bool = False) -> SN:
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
        raise ValueError("shifts must be the same or greater "
                         "length as the number of bands or `None`.")
    if shifts is None:
        shifts = np.linspace(0, n_bands, n_bands + 1) - (n_bands // 2)

    if reversed_for_abs:
        shifts = shifts[::-1]

    for i, f in enumerate(sn.bands.values()):
        f.plot_shift = shifts[i]

    return sn


def make_plot_colors(sn: SN,
                     colors: Optional[ColorIterable] = None) -> SN:
    """
    Modify state of SN object with the plot color
    for each filter set. If colors are not given
    then they will be a default set of colors.
    colors should be a dict or list of matplotlib
    compatible colors (ex.:3 floats or a color string).
    """
    if colors is None:
        colors = sns.color_palette("Paired", n_colors=len(sn.bands))
    if len(colors) < len(sn.bands):
        raise ValueError("colors must be the same or greater "
                         "length as the number of bands or `None`.")
    if isinstance(colors, list):
        colors = {band.name: color for band, color in zip(sn.bands.values(), colors)}
    for band in sn.bands.values():
        band.plot_color = colors[band.name]
    return sn


def _get_markers_to_use() -> list[str]:
    """
    Get a list of markers to use for plotting.
    """
    markers = ['o', 'd', 's']
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
    Given a Site collection (Sites), return Sitesdict of site: marker
    using more legible markers first.
    """
    markers = _get_markers_to_use()
    if len(site_markers) > len(markers):
        raise ValueError("Too many sites to plot with unique markers.")
    used_markers = set(site_markers.values())
    if None not in used_markers:
        return site_markers
    used_markers.remove(None)
    for m in used_markers:
        if m in markers:
            markers.remove(m)

    for site, marker in site_markers.items():
        if marker is None:
            site_markers[site] = markers.pop(0)

    return site_markers


def get_label(filt: Filter, site_name: Optional[str] = '') -> str:
    """
    Get a label for a filter, optionally with a site name.
    If sitename is not desired, pass an empty string or None
    """
    site_name = '' if site_name is None else site_name
    if filt.plot_shift == 0:
        label = f"${filt.name}$ {site_name}"
        return label.strip()
    shift_label = f"{filt.plot_shift:.1f}"
    if not shift_label.startswith("-") and shift_label != '':
        shift_label = f"+{shift_label}"
    if shift_label.endswith(".0"):
        shift_label = shift_label[:-2]
    label = f"${filt.name}${shift_label} {site_name}"
    return label.strip()


def get_plot_kwargs(marker: str, filt: Filter, site_label: str = '',
                    update_label: bool = True, plot_kwargs: Optional[dict] = None) -> dict:
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs["marker"] = marker
    plot_kwargs['color'] = filt.plot_color
    if update_label:
        plot_kwargs['label'] = get_label(filt, site_label)
    return {**plot_defaults, **plot_kwargs}


def plot_abs_mag(sn: SN, ax: Optional[plt.Axes] = None,
                 split_by_site: bool = False,
                 label_sites: bool = False,
                 **plot_kwargs: Any) -> tuple[Optional[plt.Figure], plt.Axes]:

    def _plot(_ax: plt.Axes, _sn: SN, _site: str, _site_label: str, _marker: str, _plot_kwargs: dict,
              update_label: bool = True) -> plt.Axes:
        for filtname, filt in _sn.bands.items():
            _plot_kwargs = get_plot_kwargs(_marker, filt, _site_label, update_label, _plot_kwargs)

            # Skip if no data
            try:
                band_phot = _sn.band(filt.name, site=_site, return_absmag=True)
            except KeyError as e:  # Sometimes sninfo does not have the right extinction.
                warnings.warn(f"Could not find absolute magnitude for {filt.name} in {sn.name},\n got exception: {e}")
                band_phot = None

            if band_phot is not None:
                _ax = plot_mag(band_phot, _ax, shift=filt.plot_shift, **_plot_kwargs)

        return _ax

    fig = None
    if ax is None:
        fig, ax = make_figure()

    if not split_by_site:
        return fig, _plot(ax, sn, 'all', '', plot_kwargs.get("marker", 'o'), plot_kwargs)

    # plot splitting by site
    sn.sites.markers = get_site_markers(sn.sites.markers)
    sn.sites.update_markers()
    first = True
    for site in sn.sites.sites.values():
        site_label = site.name if label_sites else ''
        if not first and not label_sites:
            plot_kwargs['label'] = ''
        ax = _plot(ax, sn, site.name, site_label, site.marker, plot_kwargs, update_label=first)
        first = False

    return fig, ax


def label_axis(ax: plt.Axes, xlabel: str, ylabel: str, legend: bool = True,
               legend_kwargs: Optional[dict[str, Any]] = None) -> plt.Axes:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)  # fontweigt = 'semibold'
    if legend:
        if legend_kwargs is None:
            legend_kwargs = {}
        ax.legend(**legend_kwargs)
    return ax


def format_axis(ax: plt.Axes,
                invert: bool = False,
                log: bool = False,
                xlim: Optional[tuple[float, float]] = None,
                ylim: Optional[tuple[float, float]] = None,
                x_tick_interval: Optional[float] = None,
                y_tick_interval: Optional[float] = None,
                x_minors: Optional[float] = 5,
                y_minors: Optional[float] = 5) -> plt.Axes:
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
        ax.set_yscale('log')
    return ax
