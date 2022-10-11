from astrosn.ingest import ingest_sn, Ingestors
from astrosn import SN, Site
import astrosn.plotting as plot
from astrosn.plotting.plot_lc import _get_legend_handles_labels
import matplotlib.pyplot as plt

def test_flows_ingest() -> SN:
    sn = ingest_sn("inputs/2021aess/", "SN2021aess", "flows")
    assert sn.name == "SN2021aess"
    assert sn.redshift > 0.0
    ztf_site = sn.sites["ZTF"]
    assert isinstance(ztf_site, Site)
    assert ztf_site.name == "ZTF"
    assert ztf_site.id == 0
    assert len(sn.phot) > 0

    sn = ingest_sn("inputs/2021aess/", "SN2021aess", Ingestors.flows)
    assert sn.name == "SN2021aess"
    assert sn.redshift > 0.0
    ztf_site = sn.sites["ZTF"]
    assert isinstance(ztf_site, Site)
    assert ztf_site.name == "ZTF"
    assert ztf_site.id == 0
    assert len(sn.phot) > 0

    return sn


if __name__ == "__main__":
    sn = test_flows_ingest()
    sn.sub_only = False
    sn.sninfo.sub_only = False
    sn = sn.restframe()

    plotpars = plot.AbsMagPlotParams
    plotpars.label_sites = False
    plotpars.split_by_site = False
    fig, ax = plot.plot_flows_absmag(sn, shift=False)
    handles, labels = _get_legend_handles_labels(ax.legend_)
    plt.show(block=True)
    print(labels, len(labels))

    plotpars = plot.AbsMagPlotParams
    plotpars.label_sites = True
    plotpars.split_by_site = True
    fig, ax = plot.plot_flows_absmag(sn, shift=False)
    handles, labels = _get_legend_handles_labels(ax.legend_)
    print(labels, len(labels))

    print(len(handles), len(labels))
