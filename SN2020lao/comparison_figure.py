from pathlib import Path

import pandas as pd
from astropy.cosmology import WMAP5  # type: ignore
import astropy.units as u

from supernova import SNInfo, SN
from supernova.ingest import DataIngestor, collate_csv, read_astropy_table_as_df_with_times_as_jd
import supernova.plotting as plot


def load_06aj() -> SN:
    # 06aj
    sninfo_06aj = {'z': 0.033023,
                  'filters' : ['B', 'V', 'R'],
                   'coord': '03:21:39.710 +16:52:02.60',
                   'ref'  : 'Ferrero et al. (2006)' ,
                   'link' : 'https://www.aanda.org/articles/aa/full/2006/39/aa5530-06/aa5530-06.right.html',
                   'localAv' : 0.39,
                   'hostAv' : 0.13,
                   'comments': 'In all bands, SN 2006aj is about 30% less luminous than SN 1998bw. \
                   From the light curve data, independent of any fit, we find the following peak absolute \
                   magnitudes of the SN: $M_B=-18.29\pm0.05$, $M_{V}=-18.76\pm0.05$, $M_R=-18.89\pm0.05$, \
                   $M_I=-18.95\pm0.10$. ',
                   'peakR': '-18.89',
                   'ra': '03:21:39.710',
                   'dec': '+16:52:02.60',
                   'phase_zero': 0,
                   'name': 'SN2006aj',
                   'redshift': 0.033023,}
    filters = sninfo_06aj['filters']
    sninfo_06aj['dm'] = WMAP5.distmod(sninfo_06aj['z']).value
    sninfo_06aj = SNInfo.from_dict(sninfo_06aj)

    path = Path('~/Dropbox/SN2020lao/emir/20lao/Tables/06aj').expanduser()
    dfs = []
    for filt in filters:
        df = read_astropy_table_as_df_with_times_as_jd(path / f'{filt}_06aj.ecsv')
        df['jd'] = df['phase']
        df['band'] = filt
        df['site'] = 1
        df.rename({'err': 'mag_err'}, axis=1, inplace=True)
        df['sub'] = True
        dfs.append(df)
    sn06aj_phot = pd.concat(dfs, ignore_index=True)
    #SN.from_phot(sn06aj_phot, sninfo_06aj.name, sninfo_06aj.redshift, )
    sn06aj = SN(phot=sn06aj_phot, phases=pd.Series(sninfo_06aj), sninfo=sninfo_06aj)
    sn06aj.set_phases()

    return sn06aj


def load_98bw() -> SN:
    # 98bw
    sninfo_98bw = {'redshift':0.008499,
                  'filters' : ['B', 'V', 'R'],
                   'dist' : 37.3*u.Mpc, #From redshift with H0 = 74.2
                   'link' : 'https://arxiv.org/pdf/1106.1695.pdf',
                   'ref'  : 'Clocchiatti et al. (2011)',
                   'localAv' : 0.20,
                   'coord': '19:35:03.310 -52:50:44.81',
                   'ra': '19:35:03.310',
                   'dec': '-52:50:44.81',
                    'phase_zero': 2450930.0,
                    'name': 'SN1998bw',
                  }

    sninfo_98bw = SNInfo.from_dict(sninfo_98bw)
    sninfo_98bw.dm = WMAP5.distmod(sninfo_98bw.redshift).value
    path98 = Path('~/Dropbox/SN2020lao/emir/20lao/Tables/98bw/').expanduser()

    sn98bw = DataIngestor(path98, collators={'csv': collate_csv}, sninfo=sninfo_98bw, sitemap={1: 'lit'}).load_sn()
    sn98bw.set_phases()
    return sn98bw


sn98bw = load_98bw()
sn06aj = load_06aj()
sn20lao = SN.from_csv('SNClass_SN2020lao/')

fig, ax = plot.make_figure()
sn98bw = plot.make_plot_colors(sn98bw)

plot.plot_mag(sn98bw.band('R', return_absmag=True), ax, color='darkred', label='SN1998bw')
plot.plot_mag(sn06aj.band('R', return_absmag=True), ax, color='green', label='SN2006aj')
plot.plot_mag(sn20lao.band('r', return_absmag=True), ax, color='red', label='SN2020lao')
#plot.format_axis(ax, invert=True, xlim=(-5, 100), ylim=(-19.5, -16.7))
fig.show()
