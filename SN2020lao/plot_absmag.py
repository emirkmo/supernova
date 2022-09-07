import supernova.plotting as plot
from supernova.supernova import PhotFactory
from supernova.plotting.colors import DEFAULT_COLORS
from supernova.sn_data import lao
import pandas as pd
import numpy as np

# 20lao photometry
lao.phot.band = lao.phot.band.apply(lambda l: l[0] if l.endswith('p') else l)
lao.bands = lao.make_bands(bands=list(lao.phot.band.unique()))
laor = lao.restframe()

# Remove R and u band since u band is not visible and R band is wrongly observed.
laor.bands.pop(-2)
laor.bands.pop(0)

# upper limit
rlim = pd.read_csv("../SNClass_20lao/SN2020lao_r_limits.csv")
rlim['band'] = 'r'
rlim['mag'] = rlim['lim']
rlim['mag_err'] = rlim['plot_err']
rlim['site'] = 0
rlim = PhotFactory.from_df(rlim)
rlim.calc_phases(laor.phase_zero)
rlim.phase = rlim.restframe_phases(laor.redshift)
rlim.mag = rlim.absmag(laor.distance, laor.sninfo.rext)  # absolute mag

# Automatic plot shifts & colors
shifts = [-1, 0, 0.5, 1.5, 1.5]
laor = plot.make_plot_shifts(sn=laor, reversed_for_abs=True, shifts=shifts)
laor = plot.make_plot_colors(sn=laor, colors=DEFAULT_COLORS)

# Plot
fig, ax = plot.make_figure()
_, ax = plot.plot_abs_mag(laor, ax=ax, split_by_site=True, label_sites=False, alpha=0.7)
plot.plot_lim(rlim, ax=ax, shift=laor.bands[-2].plot_shift, color='r', label='$r$ limit')
ax = plot.label_axis(ax, 'Restframe days since explosion', 'Absolute Magnitude [mag]',
                     legend_kwargs={'loc': (0.59, 0.74), 'ncol': 2, "frameon": True,
                                    "columnspacing": 0.3, "handletextpad": 0.1, "handlelength": 1.5})
ax = plot.format_axis(ax, invert=True, xlim=(-10, 98), ylim=(-19.68, -13.4),
                      x_tick_interval=10, y_tick_interval=1)
ax.axvline(0, color='black', linestyle='--', alpha=0.5)

# Spectra
df = pd.read_json('spectra_df.json')
df['restframe'] = (df['jd']-laor.phase_zero)/(1+laor.redshift)
ax.plot(df['restframe'], np.ones_like(df.restframe)*-13.6,
        marker='|', color='black', markersize=20, linestyle='None')
ax.plot(df['restframe']*1.001, np.ones_like(df.restframe)*-13.6,
        marker='|', color='black', markersize=20, linestyle='None')

fig.tight_layout()
fig.show()
fig.savefig("absmag.pdf", dpi=200, format='pdf')

# save lao
# lao.phot.calc_phases(lao.phases.tess_patrick)
lao.to_csv()
