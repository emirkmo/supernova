from typing import cast
from matplotlib import pyplot as plt
from astrosn import SN, ImplementsLuminosity
import astrosn.plotting as plot
from astrosn.bolo import bolo_filt, blackbody_filt, ArnettModelFit
import numpy as np


def load_data():
    # load bolometric interpolation data in rest frame
    lao_lum = SN.from_csv("./lum/SNClass_SN2020lao/").restframe()
    # Rescale for plotting
    lao_lum.phot.lum /= 1e42
    lao_lum.phot.lum_err /= 1e42

    mf = ArnettModelFit.from_json("./lum/SNClass_SN2020lao/arnett_peak.json")
    return lao_lum, mf


def plot_fit(sn: SN, mf: ArnettModelFit):
    full = np.linspace(1, 90, 400)
    fullm = mf.get_model(full)

    peak = np.linspace(4.9, 30, 100)
    peakm = mf.get_model(peak)

    label = "M$_{ej}$" + f": {mf.with_error_1('mej')}" + "M$_{Ni}$" + f"{mf.with_error('mni')}"
    fig, ax = plot.make_single_figure()

    # plot bolometric luminosity
    # ax = plot.plot_band(
    #     sn,
    #     bolo_filt,
    #     ax=ax,
    #     plot_type=plot.plot_lum,
    #     label="quasiBolometric",
    #     color="black",
    # )

    # plot arnett fits
    ax.plot(full, fullm / 1e42, color="red", zorder=3, label=label)
    ax.plot(peak, peakm / 1e42, color="orange", zorder=3, label="fit region")

    # label
    bb_phot = cast(ImplementsLuminosity, sn.band(bolo_filt.name))
    bb_phot.lum_err *= 2.
    plot.plot_lum(bb_phot, ax=ax, color="black", label="quasiBolometric")
    ax, legend = plot.label_axis(
        ax,
        xlabel="Restframe days since explosion",
        ylabel="Bolometric Luminosity [$10^{42}$ erg/s]",
    )
    ax = plot.format_axis(
        ax,
        xlim=(-2, int(bb_phot.phase.max()) + 3),
        ylim=(0, bb_phot.lum.max() + 0.8),
        x_tick_interval=10,
    )

    fig.savefig("arnett_fit_peak.pdf", dpi=200, format="pdf")
    fig.show()
    plt.show(block=True)


def main():
    lao_lum, mf = load_data()
    lao_lum
    plot_fit(lao_lum, mf)


if __name__ == "__main__":
    main()
