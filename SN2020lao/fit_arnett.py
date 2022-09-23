import emcee
from matplotlib import pyplot as plt
import snowballstemmer
from supernova import SN
import supernova.plotting as plot
from supernova.bolo import (fit_arnett, ArnettParams, 
    bolo_weighted_xyz, BoloType, ArnettModelFit, bolo_filt)
from lmfit import report_fit
import numpy as np

# # load bolometric interpolation data in rest frame
# lao_lum = SN.from_csv("./lum/SNClass_SN2020lao/").restframe()
# x, y, w = bolo_weighted_xyz(lao_lum, BoloType.quasi)
# apars = ArnettParams(1.0, vph=30., vary_e_exp=False)
# res, emcee_res = fit_arnett(x, y, w, apars, regularization=0.0001, fit_emcee=False)
# # plot.plot_emcee(emcee_res)


# def fit_simple(sn: SN):
sn = SN.from_csv("lum/SNClass_SN2020lao/").restframe()
x, y, w = bolo_weighted_xyz(sn, BoloType.quasi)
apars = ArnettParams(1.0, vph=22., vary_e_exp=False)
res, emcee_res = fit_arnett(xdata=x, ydata=y, weights=None, arnett_params=apars,
                            regularization=0, method='leastsq', fit_emcee=False)
#    return res, emcee_res
print(report_fit(res))

mf = ArnettModelFit(params=res.params)
mf.to_json("lum/SNClass_SN2020lao/arnett.json")

full = np.linspace(1, 90)
fullm = mf.get_model(full)

sn.phot.lum /= 1e42
sn.phot.lum_err /= 1e42

# Plot
fig, ax = plot.make_figure()  # make figure
ax = plot.plot_band(sn, bolo_filt, ax=ax,plot_type=plot.plot_lum,
                    label='qBol', color='black')
ax.plot(full, fullm/1e42, color='red')
bb_phot = sn.band(bolo_filt.name)
ax = plot.label_axis(ax, 'Restframe days since explosion', 'Bolometric Luminosity [$10^{42}$ erg/s]')
ax = plot.format_axis(ax, xlim=(-2, int(bb_phot.phase.max()) + 3),
                      ylim=(0, bb_phot.lum.max() + 0.8), x_tick_interval=10)
fig.show()
plt.show(block=True)
