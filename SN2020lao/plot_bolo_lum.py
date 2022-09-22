from supernova import SN
import supernova.plotting as plot

# load bolometric interpolation data in rest frame
lao_lum = SN.from_csv("./lum/SNClass_SN2020lao/").restframe()
# Rescale for plotting
lao_lum.phot.lum /= 1e42
lao_lum.phot.lum_err /= 1e42

# Plot
fig, ax = plot.make_figure()  # make figure
ax = plot.plot_band(lao_lum, 'Bolo', ax=ax, plot_type=plot.plot_lum, label='qBol', color='black')
ax = plot.plot_band(lao_lum, 'BB', ax=ax, plot_type=plot.plot_lum, label='BB', color='purple')
# Label
bb_phot = lao_lum.band('BB')
ax = plot.label_axis(ax, 'Restframe days since explosion', 'Bolometric Luminosity [$10^{42}$ erg/s]')
ax = plot.format_axis(ax, xlim=(-2, int(bb_phot.phase.max()) + 3),
                      ylim=(0, bb_phot.lum.max() + 0.8), x_tick_interval=10)
fig.show()

