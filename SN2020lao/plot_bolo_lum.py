from matplotlib import pyplot as plt
from supernova import SN
import supernova.plotting as plot
from supernova.bolo import bolo_filt, blackbody_filt, ArnettModelFit
import numpy as np

# load bolometric interpolation data in rest frame
lao_lum = SN.from_csv("./lum/SNClass_SN2020lao/").restframe()
# Rescale for plotting
lao_lum.phot.lum /= 1e42
lao_lum.phot.lum_err /= 1e42

mf = ArnettModelFit.from_json("./lum/SNClass_SN2020lao/arnett.json")
full = np.linspace(1, 90, 200)
label = "M$_{ej}$" + f": {mf.with_error('mej')} Mni: {mf.with_error('mni')}" 

# Plot
fig, ax = plot.make_figure()  # make figure
ax = plot.plot_band(lao_lum, bolo_filt, ax=ax,plot_type=plot.plot_lum,
                    label='qBol', color='black')
ax.plot(full, mf.get_model(full)/1e42, color='red', label=label)
# ax = plot.plot_band(lao_lum, blackbody_filt, ax=ax, plot_type=plot.plot_lum,
#                     label='BB', color='purple')
# Label
bb_phot = lao_lum.band(bolo_filt.name)
ax = plot.label_axis(ax, 'Restframe days since explosion', 'Bolometric Luminosity [$10^{42}$ erg/s]')
ax = plot.format_axis(ax, xlim=(-2, int(bb_phot.phase.max()) + 3),
                      ylim=(0, bb_phot.lum.max() + 0.8), x_tick_interval=10)


fig.show()
fig.savefig('arnett_fit.pdf', format='pdf')
plt.show(block=True)
