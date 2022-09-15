from supernova import SN
import supernova.plotting as plot
import pandas as pd
import numpy as np

# Load SN data.
lao = SN.from_csv("../SNClass_SN2020lao/")
laor = lao.restframe()

# Can also set automatic plot shifts & colors, but here using custom ones.
shifts = [-1, 0, 0.5, 1.5, 1.5]
laor = plot.make_plot_shifts(sn=laor, reversed_for_abs=True, shifts=shifts)
laor = plot.make_plot_colors(sn=laor, colors=plot.DEFAULT_COLORS)

# Plot
fig, ax = plot.make_figure()  # make figure
# Plot absolute magnitude lcs for each band, with separate markers for each site.
_, ax = plot.plot_abs_mag(laor, ax=ax, split_by_site=True, label_sites=False, alpha=0.7)
# Plot limits for 'r' band.
ax = plot.plot_lims(laor, filt=laor.bands.get('r'), ax=ax, absmag=True, color='r', label='$r$ limit')
# Label the axes and turn on legend; add some legend parameters for location.
ax = plot.label_axis(ax, 'Restframe days since explosion', 'Absolute Magnitude [mag]',
                     legend_kwargs={'loc': (0.59, 0.74), 'ncol': 2})
# Format the figure, (invert y-axis, set limits, set tick intervals, etc.)
ax = plot.format_axis(ax, invert=True, xlim=(-10, 98), ylim=ax.get_ylim(),
                      x_tick_interval=10, y_tick_interval=1)
# Draw a vertical line at phase zero
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
fig.tight_layout()  # just to make it prettier.

# load and plot spectra
df = pd.read_json('spectra_df.json')
df['restframe'] = (df['jd']-laor.phase_zero)/(1+laor.redshift)
ax.vlines(df['restframe'], ymin=ax.get_ylim()[0] * np.ones_like(df.restframe) - 0.35,
          ymax=ax.get_ylim()[0] * np.ones_like(df.restframe),
          colors='black', lw=2)

# Display and save
fig.show()
fig.savefig("absmag.pdf", dpi=200, format='pdf')
