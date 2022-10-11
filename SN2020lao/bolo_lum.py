from astrosn import SN
from astrosn.bolo import fit_sn

# Load interpolation
lao_interp = SN.from_csv("./interp/SNClass_SN2020lao/")
lao_interp = lao_interp.restframe()

# Fit the bolometric luminosity
lao_lum = fit_sn(lao_interp, quasi=True, blackbody=True, n_samples=1000)
lao_lum.to_csv("./lum/")  # save
