from typing import Optional
from matplotlib import pyplot as plt
from supernova import SN
import supernova.plotting as plot
from supernova.bolo import (fit_arnett, ArnettParams, 
    bolo_weighted_xyz, BoloType, ArnettModelFit, bolo_filt, get_arnett_params)
from lmfit import report_fit
from lmfit.minimizer import MinimizerResult
import numpy as np

# # load bolometric interpolation data in rest frame
# lao_lum = SN.from_csv("./lum/SNClass_SN2020lao/").restframe()
# x, y, w = bolo_weighted_xyz(lao_lum, BoloType.quasi)
# apars = ArnettParams(1.0, vph=30., vary_e_exp=False)
# res, emcee_res = fit_arnett(x, y, w, apars, regularization=0.0001, fit_emcee=False)
# # plot.plot_emcee(emcee_res)


sn = SN.from_csv("lum/SNClass_SN2020lao/").restframe()
def fit_arnett_full(sn: SN) -> tuple[ArnettModelFit, Optional[MinimizerResult]]:
    x, y, z, arnett_pars = get_arnett_params(sn, phase_min=1, phase_max=70, vph=22.)
    res, emcee_res = fit_arnett(xdata=x, ydata=y, weights=None, arnett_params=arnett_pars,
                            regularization=0, method='leastsq', fit_emcee=False)
    print(report_fit(res))
    mf = ArnettModelFit(params=res.params)   # type: ignore              
    return mf, emcee_res


def fit_arnett_peak(sn: SN) -> tuple[ArnettModelFit, Optional[MinimizerResult]]:
    x, y, z, arnett_pars = get_arnett_params(sn, phase_min=4.9, phase_max=30, vph=22.)
    res, emcee_res = fit_arnett(xdata=x, ydata=y, weights=None, arnett_params=arnett_pars,
                            regularization=0, method='leastsq', fit_emcee=False)
    print(report_fit(res))
    mf = ArnettModelFit(params=res.params)   # type: ignore              
    return mf, emcee_res


def plot_fit(sn: SN, mf: ArnettModelFit, adjust: bool = False):
    full = np.linspace(1, 90)
    fullm = mf.get_model(full)

    if adjust:
        sn.phot.lum /= 1e42
        sn.phot.lum_err /= 1e42

    # Plot
    fig, ax = plot.make_single_figure()  # make figure
    ax = plot.plot_band(sn, bolo_filt, ax=ax,plot_type=plot.plot_lum,
                        label='qBol', color='black')
    ax.plot(full, fullm/1e42, color='red')
    bb_phot = sn.band(bolo_filt.name)
    ax, legend = plot.label_axis(ax, 'Restframe days since explosion', 'Bolometric Luminosity [$10^{42}$ erg/s]')
    ax = plot.format_axis(ax, xlim=(-2, int(bb_phot.phase.max()) + 3),
                        ylim=(0, bb_phot.lum.max() + 0.8), x_tick_interval=10)
    fig.show()
    plt.show(block=True)


mf, emcee_res = fit_arnett_full(sn)
mf.to_json("lum/SNClass_SN2020lao/arnett_full.json")

mfp, emcee_res = fit_arnett_peak(sn)
mfp.to_json("lum/SNClass_SN2020lao/arnett_peak.json")

plot_fit(sn, mf, adjust=True)
plot_fit(sn, mfp)

