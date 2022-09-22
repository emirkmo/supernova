from supernova import SN
import supernova.plotting as plot
from supernova.bolo import fit_arnett, ArnettParams, bolo_weighted_xyz, BoloType

# load bolometric interpolation data in rest frame
lao_lum = SN.from_csv("./lum/SNClass_SN2020lao/").restframe()
x, y, w = bolo_weighted_xyz(lao_lum, BoloType.quasi)
apars = ArnettParams(1.0, vph=30., vary_e_exp=False)
res, emcee_res = fit_arnett(x, y, w, apars, regularization=0.0001)
plot.plot_emcee(emcee_res)
