from typing import cast

import matplotlib.pyplot as plt
import pytest
import warnings
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pytest import fixture

import astrosn.plotting as plot
from astrosn import SN, SNInfo
from astrosn.ingest import Ingestors, ingest_sn
from astrosn.photometry import ImplementsMag
from astrosn.plotting.plot_lc import _get_legend_handles_labels


@fixture(scope="module")
def sn() -> SN:
    sn = ingest_sn("tests/inputs/2021aess/", "SN2021aess", Ingestors.flows)
    sn.sub_only = False
    sn.sninfo.sub_only = False
    return sn.restframe()


def test_unit_flows_plot(sn: SN) -> None:
    plotpars = plot.AbsMagPlotParams
    plotpars.label_sites = False
    plotpars.split_by_site = False
    fig, ax, secax = plot.plot_flows_absmag(sn, shift=False)
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
    fig, ax, secax = plot.plot_flows_absmag(sn, shift=False)
    handles, labels = _get_legend_handles_labels(ax.legend_)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(labels) == 16
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


@pytest.mark.mpl_image_compare
def test_plot_flows_absmag_split(sn: SN) -> Figure:
    plotpars = plot.AbsMagPlotParams
    plotpars.label_sites = True
    plotpars.split_by_site = True
    fig, ax, secax = plot.plot_flows_absmag(sn, shift=False)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_flows_absmag_unlabeled(sn: SN) -> Figure:
    plotpars = plot.AbsMagPlotParams
    plotpars.label_sites = False
    plotpars.split_by_site = False
    fig, ax, secax = plot.plot_flows_absmag(sn, plotpars=plotpars, shift=True)
    return fig
