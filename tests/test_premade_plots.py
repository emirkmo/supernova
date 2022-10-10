from typing import cast
from supernova.ingest import ingest_sn, Ingestors
from supernova import SN, SNInfo
from supernova.photometry import ImplementsMag
import supernova.plotting as plot
from supernova.plotting.plot_lc import _get_legend_handles_labels
from pytest import fixture
from matplotlib.figure import Figure
from matplotlib.axes import Axes

@fixture(scope="module")
def sn() -> SN:
    sn = ingest_sn("inputs/2021aess/", "SN2021aess", Ingestors.flows)
    sn.sub_only = False
    sn.sninfo.sub_only = False
    return sn.restframe()


def test_flows_plot(sn: SN) -> None:
    plotpars = plot.AbsMagPlotParams
    plotpars.label_sites = False
    plotpars.split_by_site = False
    fig, ax = plot.plot_flows_absmag(sn, shift=False)
    handles, labels = _get_legend_handles_labels(ax.legend_)
    
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(labels) == 9
    assert len(handles) == len(labels)

    # bands in legend
    for b in sn.bands:
        magphot = cast(ImplementsMag, sn.band(b))
        if len(magphot.mag.dropna()) > 0:
            assert f"${b}$" in labels

    plotpars.label_sites = True
    plotpars.split_by_site = True
    fig, ax = plot.plot_flows_absmag(sn, shift=False)
    handles, labels = _get_legend_handles_labels(ax.legend_)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(labels) == 19
    assert len(handles) == len(labels)

    # bands in legend
    for b in sn.bands:
        magphot = cast(ImplementsMag, sn.band(b))
        if len(magphot.mag.dropna()) > 0:
            assert f"${b}$" in labels
 

def test_functional_flows_phot(sn: SN) -> None:
    fig, ax = plot.make_single_figure()
    plot.make_plot_shifts(sn, reversed_for_abs=True)
    plot.make_plot_colors(sn, colors=plot.DEFAULT_COLORS)

    plot.plot_abs_mag(sn, ax=ax, split_by_site=False, label_sites=False)
    plot.label_axis(
        ax,
        "Restframe days since explosion",
        "Absolute Magnitude [mag]",
        loc="upper right",
        ncol=4,
    )
    plot.format_axis(
        ax,
        invert=True,
        x_tick_interval=10,
        y_tick_interval=1,
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
