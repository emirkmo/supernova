from typing import Any, cast

from matplotlib.axes import Axes

# from matplotlib.axes._secondary_axes import SecondaryAxes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from astrosn import SN, Photometry

from . import plot_lc as plot
from .defaults import DefaultSettings
from .parameters import AxisParameters, PlotParameters
from .plotter import Plotter
from .settings import PlotSettings

AbsMagPlotParams: PlotParameters = PlotParameters(absmag=True, split_by_site=True, label_sites=False, mark_unsub=False)
AbsMagAxisParams: AxisParameters = AxisParameters(
    xlabel="Restframe days",
    ylabel="Absolute Magnitude",
    x_tick_interval=20,
    x_minors=4,
    invert=True,
    y_tick_interval=2,
    y_minors=2,
    legend_kwargs={"loc": "upper right", "ncol": 3},
)


def _get_x_tick_interval(phot: Photometry) -> float:
    """
    Get the x_tick_interval based on the range of the photometry.
    """
    p_range: float = phot.phase.max() - phot.phase.min()
    return max(p_range // 12, 10)


def plot_flows_absmag(
    sn: SN,
    axpars: AxisParameters = AbsMagAxisParams,
    plotpars: PlotParameters = AbsMagPlotParams,
    secondary_y: bool = True,
    shift: bool = True,
    defaults: PlotSettings = DefaultSettings(),
    **fig_kwargs: Any,
) -> tuple[Figure, Axes, Axes | None]:
    """
    Plot the absolute magnitude of a FLOWS supernova.
    """
    sn = sn.restframe()  # Make sure we're in the restframe

    # Check if we can split by site.
    if plotpars.split_by_site:
        try:
            sn.sites.update_markers(plot.get_site_markers(sn.sites.markers))
        except ValueError:
            plotpars.split_by_site = False
            print(
                "Too many sites to plot with unique markers and some site markers were `None`. Plotting all together."
                "If you want to split by site, set the marker of each site to a matplotlib marker string manually."
                "You can use `supernova.plotting.plot._get_markers_to_use` to get a list of unique markers."
                "use `SN.sites.update_markers` to set the markers of each site at once."
            )

    # Automatically set the x_tick_interval based on the range of the photometry.
    if axpars.x_tick_interval in (10, 10.0, None):
        _phot = sn.phot
        if sn.sub_only:
            _phot = _phot.masked(_phot.sub.tolist())
        axpars.x_tick_interval = _get_x_tick_interval(_phot)

    # Turn off autoshifts if desired.
    if not shift:
        plotpars.shifts = [0.0 for b in sn.bands]

    # plot
    plotter = Plotter(sn, plot.plot_abs_mag, plotpars, axpars)
    fig, ax = plotter(ax=None, **fig_kwargs)

    # Make tight layout
    fig = cast(Figure, fig)
    fig.tight_layout()

    # Add app mag as secondary
    if secondary_y:
        secax: Axes = cast(
            Axes, ax.secondary_yaxis("right", functions=(lambda l: l + sn.distance, lambda x: x - sn.distance))
        )
        secax.set_ylabel("App. Mag", **defaults.axis_kwargs)
        minor = 1 / axpars.y_minors if axpars.y_minors else 0.2
        secax.yaxis.set_minor_locator(MultipleLocator(minor))
        return fig, ax, secax
    return fig, ax, None
