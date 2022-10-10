from pathlib import Path
from astropy.coordinates.representation import PhysicsSphericalRepresentation
import matplotlib.pyplot as plt
from astropy.modeling import custom_model, fitting, models

import pandas as pd
import numpy as np
from astropy.cosmology import WMAP5  # type: ignore
import astropy.units as u

from supernova import SNInfo, SN, Photometry, sn_factory, ImplementsMag, HasMag
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
    sn06aj = sn_factory(phot=sn06aj_phot, sninfo=sninfo_06aj)
    #sn06aj = SN(phot=sn06aj_phot, phases=pd.Series(sninfo_06aj), sninfo=sninfo_06aj)
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


def ingest_lcs_get_restframe_SNe():


    sn98bw = load_98bw()
    sn06aj = load_06aj()
    sn20lao = SN.from_csv('SNClass_SN2020lao/')

    sn06aj.to_csv()
    sn98bw.to_csv()


    # Restframe R band comparison
    sn98bw = sn98bw.restframe()
    sn06aj = sn06aj.restframe()
    sn20lao = sn20lao.restframe()

    return sn98bw, sn06aj, sn20lao


def plot_rband(band: ImplementsMag, ax: plot.Axes, color: str, label: str) -> plot.Axes:
    ax = plot.plot_mag(phot=band, ax=ax, color=color, label=label)
    return ax


def plot_rband_comparison():
    sn98bw, sn06aj, sn20lao = ingest_lcs_get_restframe_SNe()
    sn98bw = plot.make_plot_colors(sn98bw)

    fig, ax = plot.make_single_figure()
    plot.plot_mag(sn98bw.band('R', return_absmag=True), ax, color='darkred', label='SN1998bw')  # type: ignore
    plot.plot_mag(sn06aj.band('R', return_absmag=True), ax, color='green', label='SN2006aj')  # type: ignore
    plot.plot_mag(sn20lao.band('r', return_absmag=True), ax, color='red', label='SN2020lao')  # type: ignore
    ax.axvline(0, color='red', linestyle='--', lw=0.8)
    plot.format_axis(ax, invert=True, xlim=(-5, 100), ylim=(-19.5, -15.3))
    plot.label_axis(ax, xlabel='Days since explosion', ylabel='Absolute magnitude')
    
    fig.tight_layout()
    fig.savefig('rband_comp.pdf', dpi=200, format='pdf')
    fig.show()

def get_max(mag: Photometry) -> tuple[float, float]:
    if not isinstance(mag, HasMag):
        raise TypeError(f'Expected HasMag, got {type(mag)}')
    max_mag = np.nanmin(mag.mag)
    max_phase = mag.phase[mag.mag==max_mag].values[0]
    if not isinstance(max_phase, float):
        raise ValueError(f'Max Phase: {max_phase}, is not a float and not found.')
    if not isinstance(max_mag, float):
        raise ValueError(f'Max Mag: {max_mag}, is not a float and not found.')
    return max_phase, max_mag

def add_max(sn: SN, band: str = 'R') -> SN:
    sn.phases[f"{band}max"], sn.phases[f"{band}max_mag"] = get_max(sn.band(band, return_absmag=True))
    return sn

def read_ptf() -> pd.DataFrame:
    savedf = pd.read_json(Path('~/Dropbox/SN2020lao/emir/20lao/Photometry/icbl_comp.json').expanduser())
    savedf.columns = ['X','Y','Z','marker','color','modX','modY']
    savedf['markersize']=6
    savedf['color'] = 'grey'
    savedf['marker'] = None
    savedf['alpha'] = 0.6
    savedf.loc['SN2020lao','markersize']=12
    savedf.loc['SN2020lao', 'color']='red'
    savedf.loc['SN2020lao', 'marker']='d'
    del savedf.at['SN2020lao','Y'][14]
    del savedf.at['SN2020lao','X'][14]
    del savedf.at['SN2020lao','Z'][14]
    return savedf


@custom_model
def contardo(t,f0=-18.0,gamma=0.06,t0=-30.0,g0=109,sig0=10.0,theta=10.0,Tau=-18.0):    
    return (f0 + gamma*(t-t0)+ g0*np.exp(-((t-t0)**2)/(2*sig0**2))) /(1-np.exp((Tau-t)/theta))

def fit_contardo(rband: Photometry):
    # Fit the contardo model to the R/r band light curve
    f0=29
    m=0.001
    t0=-5
    g0=-9
    sigma0=30
    theta=3
    tau=-7
    par0=[f0, m, t0, g0, sigma0, theta, tau]

    bounds_dict={'f0':(-40,40),
             'gamma':(0.00001,0.3),
             't0':(-40.0,20.0),
             'g0':(-50.0,30.0),
             'sig0':(0.01,200.0),
             'Tau':(-100,2.0),
             'theta':(0.1,100),
            }

    fitter=fitting.LevMarLSQFitter()
    c_init = contardo(*par0, bounds=bounds_dict)
    X=rband.phase.values
    Y=rband.mag.values
    Z=(1.0/rband.mag_err.values)**2
    c_fit = fitter(c_init, X, Y, weights=Z, maxiter=10000)
    modrange=np.linspace(X[0],X[-1],num=200)
    maxfit=np.min(c_fit(modrange))
    tmax=np.nanmean(np.atleast_1d(modrange[c_fit(modrange)==maxfit]))
    return tmax, maxfit, c_fit

def plot_rband_abs():
    sn98bw, sn06aj, sn20lao = ingest_lcs_get_restframe_SNe()
    sn98bw = plot.make_plot_colors(sn98bw)

    sn98bw = add_max(sn98bw, 'R')
    sn06aj = add_max(sn06aj, 'R')
    sn20lao = add_max(sn20lao, 'r')

    tmax, maxfit, c_fit = fit_contardo(sn20lao.band('r', return_absmag=True))

    savedf = read_ptf()
    fig, ax = plot.make_single_figure()
    for sn in savedf.index:
        if sn == 'SN2020lao':
            continue
        ax.plot(savedf.loc[sn,'X'],savedf.loc[sn,'Y'],
                    marker=savedf.loc[sn,'marker'],color=savedf.loc[sn,'color'],

                    
                    markersize=savedf.loc[sn,'markersize'],alpha=savedf.loc[sn,'alpha'])
    ax.axvline(-sn20lao.phases.rmax, linestyle='--', color='black')

    # sn98bw.set_phases(sn98bw.phases.Rmax)
    # sn06aj.set_phases(sn06aj.phases.Rmax)
    # sn20lao.set_phases(sn20lao.phases.rmax)

    r98bw = sn98bw.band('R', return_absmag=True)
    r06aj = sn06aj.band('R', return_absmag=True)
    r20lao = sn20lao.band('r', return_absmag=True)

    r98bw.phase = (r98bw.phase - sn98bw.phases.Rmax)/(sn98bw.redshift + 1)
    r06aj.phase = (r06aj.phase - sn06aj.phases.Rmax)/(sn06aj.redshift + 1)
    r20lao.phase = (r20lao.phase - sn20lao.phases.rmax)/(sn20lao.redshift + 1)

    r98bw.mag = r98bw.mag - sn98bw.phases.Rmax_mag #- sn98bw.bands['R'].ext
    r06aj.mag = r06aj.mag - sn06aj.phases.Rmax_mag #- sn06aj.bands['R'].ext
    r20lao.mag = r20lao.mag - sn20lao.phases.rmax_mag #- sn20lao.bands['r'].ext

    plot.plot_mag(r98bw, ax, color='darkred', label='SN1998bw')
    plot.plot_mag(r06aj, ax, color='green', label='SN2006aj')
    plot.plot_mag(r20lao, ax, color='red', label='SN2020lao')

    plot.format_axis(ax, invert=True, xlim=(-40, 70), ylim=(-0.35,3.0))
    plot.label_axis(ax, xlabel='Restframe days since R/r max', ylabel='$\Delta Mag$')
    
    fig.savefig('rband_abs_comp.pdf', dpi=200, format='pdf')
    fig.show()


if __name__ == '__main__':
    plot_rband_comparison()
    plot_rband_abs()
    plt.show(block=True)
