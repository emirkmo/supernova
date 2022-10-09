from functools import partial
from typing import Any, Optional, Iterable
import numpy as np
from numpy.typing import NDArray, ArrayLike
from astropy import constants  # type: ignore
import astropy.units as u
from astropy.modeling import fitting, physical_models
import multiprocessing
from astropy.cosmology import WMAP5  # type: ignore
from astropy.table import QTable
import pandas as pd
import tqdm
import warnings

from supernova import Photometry, Filter, SN, LumPhot, BBLumPhot, PhotFactory
from supernova.filters import FilterSorter
from .bololc import bolo_filt, blackbody_filt

# Constants
a = 4 * constants.sigma_sb / constants.c  # type: ignore
a_cg = a.to(u.erg / u.cm**3 / u.K**4)  # type: ignore
flam = u.erg / u.cm**2 / u.s / u.angstrom  # type: ignore
slam = flam / u.sr
DEFAULT_KWARGS = {"temperature": 10000 * u.K}
flux_units = u.erg / u.s / u.cm**2  # type: ignore
MyArrayLike = ArrayLike | u.Quantity




def bb_err(bb_fit: physical_models.BlackBody) -> u.Quantity:
    return (
        np.sqrt(
            ((bb_fit.scale.std) / bb_fit.scale) ** 2
            + (4 * bb_fit.temperature.std / bb_fit.temperature.value) ** 2  # type: ignore
        )
        * bb_fit.bolometric_flux
    )


@u.quantity_input(y=flam)
def init_bb_fit(
    y: flam, init_kwargs: Optional[dict[str, Any]] = None, normalize_flux: bool = False
) -> tuple[u.Quantity, physical_models.BlackBody, Optional[u.Quantity], float]:
    y = y / np.pi / u.sr
    norm = None
    if normalize_flux:
        norm = np.max(y)
        y = y / norm

    if init_kwargs is None:
        init_kwargs = DEFAULT_KWARGS

    scale_guess = np.mean(y)

    if normalize_flux:
        scale_guess = scale_guess << slam

    rescale = 1.0 / scale_guess.value
    init_kwargs = init_kwargs | {"scale": 1.0 * slam}
    return y, physical_models.BlackBody(**init_kwargs), norm, rescale


@u.quantity_input(y=slam, x=u.angstrom)
def fit_bb(
    bb_init: physical_models.BlackBody,
    y: slam,
    x: u.Quantity,
    rescaling: float,
    more_precise: bool = False,
    fitter: fitting._FitterMeta = fitting.LevMarLSQFitter,
) -> physical_models.BlackBody:

    fitter = fitter(calc_uncertainties=True)

    if more_precise:
        bb_fit = fitter(
            bb_init, x, y * rescaling, epsilon=1e-18, acc=1e-15, maxiter=1000
        )
    else:
        bb_fit = fitter(bb_init, x, y * rescaling)

    bb_fit.scale /= rescaling
    # bb_fit.scale.std /= rescaling

    return bb_fit


def get_bolo_flux(bb_fit: physical_models.BlackBody) -> u.Quantity:
    return bb_fit.bolometric_flux * np.pi


def calculate_radius(
    bb_scale: MyArrayLike, distance: u.Quantity = u.Quantity(10.0, unit=u.parsec)
) -> u.Quantity:
    """
    `distance` should be the luminosity distance if the blackbody was fitted to apparent flux, else it
    is 10 parsec, i.e. the absolute magnitude distance.
    """
    radius = bb_scale**0.5 * distance  # type: ignore
    return radius.to(u.cm)


def calculate_luminosity(bb_flux: u.Quantity, distance: u.Quantity) -> u.Quantity:
    lum: u.Quantity = (bb_flux * 4 * np.pi * distance**2).cgs  # type: ignore
    return lum.cgs


def get_parameters(
    bb_fit: physical_models.BlackBody,
    distance: u.Quantity = u.Quantity(10.0, unit=u.parsec),
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    return (
        u.Quantity(bb_fit.temperature.quantity),
        calculate_radius(u.Quantity(bb_fit.scale.quantity), distance),
        get_bolo_flux(bb_fit),
    )


def fast_get_parameters(
    bb_fit: physical_models.BlackBody,
) -> tuple[float, float, float]:
    return bb_fit.temperature.value, bb_fit.scale.value, get_bolo_flux(bb_fit).value  # type: ignore


def bootstrap_mean_and_error(
    x: ArrayLike, axis: Optional[int] = None
) -> tuple[MyArrayLike, MyArrayLike]:
    return np.mean(x, axis=axis), np.std(x, axis=axis)


def safe_vary_flux(flux_varied: NDArray, flux_orig: NDArray) -> NDArray:
    return np.where(flux_varied > np.min(flux_varied) / 15, flux_varied, flux_orig)


def bootstrap_parameters(
    flux: u.Quantity,
    unc: u.Quantity,
    waves: u.Quantity,
    n_samples: int = 500,
    sigma_unc: float = 2,
    distance: u.Quantity = u.Quantity(10.0, unit=u.parsec),
    normalize_flux: bool = False,
) -> tuple[float]:
    """Return temperature, radius, and bolometric flux of a blackbody fit to a set of flux measurements.
    as well as their errors.
    """
    # npr = (np.random.random_sample(n_samples) - 0.5) * 2 * sigma_unc
    npr = (np.random.random_sample((n_samples, len(flux))) - 0.5) * 2 * sigma_unc

    flux_sr, bb_init, flux_normalization, rescaling = init_bb_fit(
        flux, normalize_flux=normalize_flux
    )

    unc_sr = unc / np.pi / u.sr
    _pars = []
    for n in npr:
        _flux_bs = flux_sr + n * unc_sr
        _flux_bs = safe_vary_flux(_flux_bs, flux_sr)
        bb_fit = fit_bb(bb_init, _flux_bs, waves, rescaling)
        if normalize_flux:
            bb_fit.scale *= flux_normalization
        _pars.append(fast_get_parameters(bb_fit))

    pars = np.array(_pars)
    pars[:, 1] = calculate_radius(pars[:, 1], distance=distance).value
    par_mean, par_se = bootstrap_mean_and_error(pars, axis=0)
    return tuple(np.hstack((par_mean, par_se)))


def multi_threaded_fit_with_bootstrap(
    fluxes: list[NDArray],
    flux_uncs: list[NDArray],
    waves: u.Quantity,
    n_samples: int = 500,
    sigma_unc: float = 2,
    distance: u.Quantity = u.Quantity(10.0, unit=u.parsec),
    normalize_flux: bool = False,
    threads: int = multiprocessing.cpu_count(),
) -> NDArray:

    if len(fluxes) == len(waves):
        flux_iter = [
            (f << flam, u << flam)
            for f, u in zip(*as_zipped_iterable_array(fluxes, flux_uncs))
        ]
        n_rows = fluxes[0].shape[0]
    else:
        flux_iter = [(f << flam, u << flam) for f, u in zip(fluxes, flux_uncs)]
        n_rows = len(fluxes)

        if len(waves) != len(fluxes[0]):
            raise ValueError(
                "Number of wavelengths and fluxes must be equal, and in same order."
            )

    pars = np.empty((n_rows, 8))

    partial_func = partial(
        bootstrap_parameters,
        waves=waves,
        distance=distance,
        normalize_flux=normalize_flux,
        sigma_unc=sigma_unc,
        n_samples=n_samples,
    )

    with multiprocessing.Pool(threads) as pool:
        with tqdm.tqdm(total=n_rows) as pbar:
            for i, par in enumerate(pool.starmap(partial_func, flux_iter)):
                pars[i] = (*par, 0, 0)
                pbar.update()

    pars[:, 6] = calculate_luminosity(pars[:, 2] * flux_units, distance=distance).value
    pars[:, 7] = calculate_luminosity(pars[:, 5] * flux_units, distance=distance).value
    return pars


def make_qtable(pars: NDArray) -> QTable:
    return QTable(
        data=pars,
        names=["T", "R", "F", "T_err", "R_err", "F_err", "L", "L_err"],
        units=(u.K, u.cm, flux_units, u.K, u.cm, flux_units, u.erg / u.s, u.erg / u.s),
    )


def as_zipped_iterable_array(
    fluxes: list[NDArray], flux_uncs: list[NDArray]
) -> tuple[NDArray, NDArray]:
    return make_zipped_stacked(fluxes), make_zipped_stacked(flux_uncs)


def make_zipped_stacked(list_of_arrays: list[NDArray]) -> NDArray:
    return np.vstack([array for array in list_of_arrays]).T


def load_pandas_flux_table(filename: str) -> tuple[list[NDArray], list[NDArray]]:
    fluxes = pd.read_json(filename)
    fluxes.index = fluxes.index.to_julian_date()  # type: ignore
    fluxes_zipped = list(
        fluxes[[name for name in fluxes.columns if not name.endswith("err")]].values
    )
    errors_zipped = list(
        fluxes[[name for name in fluxes.columns if name.endswith("err")]].values
    )
    return fluxes_zipped, errors_zipped


def load_phot(phot: Photometry, bands: dict[str, Filter],
              sorter: FilterSorter = FilterSorter()) -> tuple[u.Quantity, list[NDArray], list[NDArray], NDArray[float]]:
    bands = sorter.wave_sort(bands.values())

    fluxes = [b.name+'_flux' for b in bands]
    flux_uncs = [b.name+'_flux_err' for b in bands]
    waves = np.array([b.wave_eff.value for b in bands]) << u.AA

    df = phot.as_dataframe()
    df = df.pivot(index='jd', columns='band', values=['flux', 'flux_err'])
    df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
    df = df[fluxes + flux_uncs]

    fluxes_zipped = list(
        df[[name for name in df.columns if not name.endswith("err")]].values
    )
    errors_zipped = list(
        df[[name for name in df.columns if name.endswith("err")]].values
    )
    jds = df.index.values
    return waves, fluxes_zipped, errors_zipped, jds


def fit_sn(sn: SN, quasi: bool = False, blackbody: bool = False, n_samples: int = 1000,
           n_sigma: float = 2) -> SN:
    """
    Fit the SN with a blackbody and/or quasi-blackbody model.
    Returns a copy of the SN with the fitted values
    added to the photometry table, with sites and bands as "Bolo" and "BB".

    n_samples: number of bootstrap samples to use
    n_sigma: number of sigma to use for the bootstrap (e.g. vary errors by +/- 2 sigma)
    """
    waves, fluxes, flux_uncs, jds = load_phot(sn.phot, sn.bands)
    distance = sn.sninfo.cosmology.luminosity_distance(sn.sninfo.redshift).to(u.pc)
    new_sn = sn.copy()
    if not quasi and not blackbody:
        raise ValueError("Must fit at least one model. Set quasi=True or blackbody=True.")

    fac = PhotFactory(sn_phot=sn.phot)
    if quasi:
        # Setup
        fac.extend_phot(LumPhot)
        site_id = new_sn.sites.generate_id()
        new_sn.add_site(bolo_filt.name, id=site_id, marker='o')

        # Fit
        qbt = bootstrap_quasi_bolo(
            waves, np.array(fluxes), np.array(flux_uncs), n_samples=n_samples, sigma_unc=n_sigma,
            distance=distance)
        # Save

        qbt = qbt.to_pandas().rename({'F': 'flux', 'F_err': 'flux_err', 'L': 'lum', 'L_err': 'lum_err'}, axis=1)
        qbt['jd'] = jds
        qbt['band'] = bolo_filt.name
        qbt['site'] = site_id
        fac.concat_phot(fac.from_df(qbt))

    if blackbody:
        # Setup
        fac.extend_phot(BBLumPhot)
        site_id = new_sn.sites.generate_id()
        new_sn.add_site(blackbody_filt.name, id=site_id, marker='s')

        # Fit
        print("Fitting blackbody to flux measurements")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_params = multi_threaded_fit_with_bootstrap(
                waves=waves,
                fluxes=fluxes,
                flux_uncs=flux_uncs,
                n_samples=n_samples,
                sigma_unc=n_sigma,
                distance=distance,  # type: ignore
                normalize_flux=False,
                threads=max(multiprocessing.cpu_count()-3, 1),
            )

        # Save
        print("Done: Saving results!")
        qt = make_qtable(fitted_params).to_pandas()
        qt = qt.rename({'F': 'flux', 'F_err': 'flux_err', 'L': 'lum', 'L_err': 'lum_err',
                        'R': 'radius', 'R_err': 'radius_err', 'T': 'temp', 'T_err': 'temp_err'},
                       axis=1)
        qt['jd'] = jds
        qt['band'] = blackbody_filt.name
        qt['site'] = site_id
        fac.concat_phot(fac.from_df(qt))

    new_sn.phot = fac.sn_phot
    new_sn.set_phases()
    new_sn.make_bands(new_sn.phot.band.unique().tolist(), ebv=sn.sninfo.ebv)

    return new_sn


def bootstrap_quasi_bolo(
    waves: NDArray,
    fluxes: NDArray,
    flux_errs: NDArray,
    n_samples: int = 500,
    sigma_unc: float = 2,
    distance: u.Quantity = u.Quantity(10.0, unit=u.parsec),
) -> QTable:
    fluxes = fluxes << flam
    flux_errs = flux_errs << flam
    min_flux = fluxes.min() / 10
    bs_samples = (np.random.random((n_samples, *fluxes.shape)) - 0.5) * 2 * sigma_unc * flux_errs + fluxes
        
    bs_samples[bs_samples < 0] = min_flux
    bs_samples = np.trapz(bs_samples, waves) 
    bs_lums = calculate_luminosity(bs_samples, distance=distance)

    return QTable(
        data={
            "F": bs_samples.mean(axis=0),
            "F_err": bs_samples.std(axis=0),
            "L": bs_lums.mean(axis=0),
            "L_err": bs_lums.std(axis=0),
        }
    )


def main():
    pass
    # waves = np.array([4299.2, 4900.139, 6241.273, 7563.767, 5393.9]) * u.AA
    # fluxes, flux_uncs = load_pandas_flux_table(
    #     "/home/emir/Dropbox/SN2020lao/emir/20lao/Tables/absflux_per_filter.json"
    # )
    # redshift = 0.03112
    # cosmology = WMAP5
    # distance = cosmology.luminosity_distance(redshift)
    # absolute = True
    # if absolute:
    #     distance = u.Quantity(10.0, unit=u.parsec)
    #
    # qbt = bootstrap_quasi_bolo(
    #     waves, np.array(fluxes), np.array(flux_uncs), n_samples=1000, sigma_unc=2,
    # distance=distance)
    # qbt.write(
    #     "/home/emir/Dropbox/SN2020lao/emir/20lao/Tables/quasi_bolometric_flux_lum.ecsv",
    #     format="ascii.ecsv",
    #     overwrite=True,
    # )
    #
    # print("Fitting blackbody to flux measurements")
    #
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     fitted_params = multi_threaded_fit_with_bootstrap(
    #         waves=waves,
    #         fluxes=fluxes,
    #         flux_uncs=flux_uncs,
    #         n_samples=1000,
    #         sigma_unc=3,
    #         distance=distance,  # type: ignore
    #         normalize_flux = False,
    #         threads= 12
    #     )
    # print("Done: Saving results!")
    #
    # qt = make_qtable(fitted_params)
    # qt.write("/home/emir/Dropbox/SN2020lao/emir/20lao/Tables/bolo_from_bb_1000_3.ecsv", overwrite=True, format="ascii.ecsv")
    #
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     fitted_params = multi_threaded_fit_with_bootstrap(
    #         waves=waves,
    #         fluxes=fluxes,
    #         flux_uncs=flux_uncs,
    #         n_samples=500,
    #         sigma_unc=2,
    #         distance=distance,  # type: ignore
    #         normalize_flux = False,
    #         threads= 12
    #     )
    # print("Done: Saving results!")
    #
    # qt = make_qtable(fitted_params)
    # qt.write("/home/emir/Dropbox/SN2020lao/emir/20lao/Tables/bolo_from_bb_500_2.ecsv", overwrite=True, format="ascii.ecsv")
    #
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     fitted_params = multi_threaded_fit_with_bootstrap(
    #         waves=waves,
    #         fluxes=fluxes,
    #         flux_uncs=flux_uncs,
    #         n_samples=1000,
    #         sigma_unc=2,
    #         distance=distance,  # type: ignore
    #         normalize_flux = False,
    #         threads= 12
    #     )
    # print("Done: Saving results!")
    #
    # qt = make_qtable(fitted_params)
    # qt.write("/home/emir/Dropbox/SN2020lao/emir/20lao/Tables/bolo_from_bb_1000_2.ecsv", overwrite=True, format="ascii.ecsv")


if __name__ == "__main__":
    main()
