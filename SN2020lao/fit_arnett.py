from supernova import SN
import supernova.plotting as plot
from supernova.bolo import fit_arnett, ArnettParams, bolo_weighted_xyz, BoloType
from lmfit import report_fit

# # load bolometric interpolation data in rest frame
# lao_lum = SN.from_csv("./lum/SNClass_SN2020lao/").restframe()
# x, y, w = bolo_weighted_xyz(lao_lum, BoloType.quasi)
# apars = ArnettParams(1.0, vph=30., vary_e_exp=False)
# res, emcee_res = fit_arnett(x, y, w, apars, regularization=0.0001, fit_emcee=False)
# plot.plot_emcee(emcee_res)


#def fit_simple(sn: SN):
sn = SN.from_csv("./lum/SNClass_SN2020lao/").restframe()
x, y, w = bolo_weighted_xyz(sn, BoloType.quasi)
apars = ArnettParams(1.0, vph=22., vary_e_exp=False)
res, emcee_res = fit_arnett(xdata=x, ydata=y, weights=None, arnett_params=apars,
                            regularization=0, method='leastsq', fit_emcee=False)
#    return res, emcee_res



#res, eres = fit_simple(lao_lum)