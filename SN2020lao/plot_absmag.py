from matplotlib import pyplot as plt
from astrosn import SN
import astrosn.plotting as plot
import pandas as pd
import numpy as np

from astrosn.plotting.plot_lc import Plotter

# Load SN data.
lao = SN.from_csv("./SNClass_SN2020lao/")
laor = lao.restframe()


shifts = [-1, 0, 0.5, 1.5, 1.5]

plotpars = plot.PlotParameters(absmag=True, shifts=shifts, split_by_site=True, label_sites=True)
axpars = plot.AxisParameters(xlabel='Restframe days since explosion', ylabel='Absolute Magnitude',
                            xlim=(-5, 90), x_tick_interval=10, invert=True,
                            y_tick_interval=1, legend_kwargs={'loc': 'upper right', 'ncol': 3})
plotter = Plotter(laor, plot.plot_abs_mag, plotpars, axpars)
fig, ax = plotter(figsize=(10, 8))
if fig is not None:
    fig.tight_layout()

plotter.plot_params.band = laor.bands['r']
ymin, ymax = ax.get_ylim()
ax.set_ylim(ax.get_ylim())

plotter.update_plot_type(plot.plot_lims, color='red', label='$r$ limit')
_, ax = plotter.plot(ax)

ax.axvline(0, color='black', linestyle='--', alpha=0.5)


# load and plot spectra
df = pd.read_json('inputs/spectra_df.json')
df['restframe'] = (df['jd']-laor.phase_zero)/(1+laor.redshift)
ax.vlines(df['restframe'], ymin=ax.get_ylim()[0] * np.ones_like(df.restframe) - 0.35,
          ymax=ax.get_ylim()[0] * np.ones_like(df.restframe),
          colors='black', lw=2)

# Interpolation snake plot
lao_interp = SN.from_csv("./interp/SNClass_SN2020lao/")

lao_interp = lao_interp.restframe()
lao_interp = plot.make_plot_shifts(sn=lao_interp, reversed_for_abs=True, shifts=shifts)
lao_interp = plot.make_plot_colors(sn=lao_interp, colors=plot.DEFAULT_COLORS)
for name, band in lao_interp.bands.items():
    band_phot = lao_interp.band(band.name, return_absmag=True)
    ax = plot.plot_mag(band_phot, ax=ax, shift=band.plot_shift,
                       error_snake=True, alpha=0.3, color=band.plot_color)


secax = ax.secondary_yaxis('right', functions=(lambda l: l+laor.distance, lambda x: x-laor.distance))
secax.set_ylabel('App. Mag', fontsize=16, fontweight='semibold')
secax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

#fig.tight_layout()  # just to make it prettier.
# Display and save
#fig.show()
fig.savefig("absmag_new.pdf", dpi=200, format='pdf')
fig.show()
plt.show(block=True)


# # Can also set automatic plot shifts & colors, but here using custom ones.
# shifts = [-1, 0, 0.5, 1.5, 1.5]
# laor = plot.make_plot_shifts(sn=laor, reversed_for_abs=True, shifts=shifts)
# laor = plot.make_plot_colors(sn=laor, colors=plot.DEFAULT_COLORS)

# # Plot
# fig, ax = plot.make_figure()  # make figure
# # Plot absolute magnitude lcs for each band, with separate markers for each site.
# _, ax = plot.plot_abs_mag(laor, ax=ax, split_by_site=True, label_sites=True, alpha=0.7)
# # Plot limits for 'r' band.
# ax = plot.plot_lims(laor, band=laor.bands.get('r'), ax=ax, absmag=True, color='r', label='$r$ limit')
# # Label the axes and turn on legend; add some legend parameters for location.
# ax = plot.label_axis(ax, 'Restframe days since explosion', 'Absolute Magnitude [mag]',
#                      legend_kwargs={'loc': (0.59, 0.74), 'ncol': 2})
# # Format the figure, (invert y-axis, set limits, set tick intervals, etc.)
# ax = plot.format_axis(ax, invert=True, xlim=(-10, 98), ylim=ax.get_ylim(),
#                       x_tick_interval=10, y_tick_interval=1)
# # Draw a vertical line at phase zero
# ax.axvline(0, color='black', linestyle='--', alpha=0.5)


# # load and plot spectra
# df = pd.read_json('inputs/spectra_df.json')
# df['restframe'] = (df['jd']-laor.phase_zero)/(1+laor.redshift)
# ax.vlines(df['restframe'], ymin=ax.get_ylim()[0] * np.ones_like(df.restframe) - 0.35,
#           ymax=ax.get_ylim()[0] * np.ones_like(df.restframe),
#           colors='black', lw=2)

# # Interpolation snake plot
# lao_interp = SN.from_csv("./interp/SNClass_SN2020lao/")
# lao_interp = lao_interp.restframe()
# lao_interp = plot.make_plot_shifts(sn=lao_interp, reversed_for_abs=True, shifts=shifts)
# lao_interp = plot.make_plot_colors(sn=lao_interp, colors=plot.DEFAULT_COLORS)
# for name, band in lao_interp.bands.items():
#     band_phot = lao_interp.band(band.name, return_absmag=True)
#     ax = plot.plot_mag(band_phot, ax=ax, shift=band.plot_shift,
#                        error_snake=True, alpha=0.3, color=band.plot_color)

# #fig.tight_layout()  # just to make it prettier.
# # Display and save
# #fig.show()
# fig.savefig("absmag2.pdf", dpi=200, format='pdf')
# fig.show()
# plt.show(block=True)
