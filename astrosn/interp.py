from typing import Any, Mapping, Optional, Protocol, cast

import astropy.units as u
import importlib
import numpy as np
import pandas as pd
import piscola
from numpy._typing import _ArrayLike as ArrayLike
from numpy.typing import NDArray
from pathlib import Path
from tempfile import TemporaryDirectory

from .filters import FLAM, Filter, MagSys, SVOFilter, zero_point_flux
from .photometry import FluxPhotometry, HasFlux, HasMag, MagPhotometry, Phot, PhotFactory, Photometry, SeriesSite
from .sites import Sites
from .supernova import SN, SNSerializer
from .utils import Number, QuantityArrayType, quantity_array


class GPInterp(Protocol):
    def __init__(self, sn: SN) -> None:
        ...

    def __call__(self) -> pd.DataFrame:
        ...


class PiscolaInterp:
    piscola_names = [
        "time",
        "flux",
        "flux_err",
        "mag",
        "mag_err",
        "zp",
        "band",
        "mag_sys",
    ]

    def __init__(self, sn: SN) -> None:
        self.sn = sn
        self.sns = SNSerializer(self.sn)

    def prepare_piscola_filters(self) -> None:
        """Prepare Piscola filters for interpolation."""
        self.sns.save_piscola_filters(sites=False)
        importlib.reload(piscola)

    @staticmethod
    def _fix_zp(lcdf: pd.DataFrame, correct_zp: float) -> pd.DataFrame:
        """Fix the zero point of the light curves."""
        lcdf["flux"] = piscola.utils.change_zp(lcdf.flux, lcdf.zp, correct_zp)
        lcdf["flux_err"] = piscola.utils.change_zp(lcdf.flux_err, lcdf.zp, correct_zp)
        lcdf["zp"] = correct_zp
        return lcdf

    def make_df(self, lc: Any, init_lcs: Any) -> pd.DataFrame:
        """Make a DataFrame from a Piscola light curve."""
        dfs = []
        for band in lc.bands:
            df = pd.DataFrame({colname: lc[band][colname] for colname in self.piscola_names})
            zp_corr = init_lcs[band].zp
            df = self._fix_zp(df, zp_corr)
            dfs.append(df)
        return pd.concat(dfs)

    def gp_interp(self) -> pd.DataFrame:
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            piscofile = self.sns.make_piscola_file(tmpdir)
            # lcs_df = pd.read_csv(piscofile, delim_whitespace=True, skiprows=2)

            sn = piscola.call_sn(str(piscofile.absolute()))
            sn._fit_lcs()
            lcs = self.make_df(sn._init_lc_fits, sn.init_lcs)
        return lcs

    @staticmethod
    def _fix_site_band_names(df: pd.DataFrame) -> pd.DataFrame:
        """Fix band names."""
        df["site"] = df["band"].str.split("_").str[0]
        df["band"] = df["band"].str.split("_").str[1]
        return df

    def __call__(self) -> pd.DataFrame:
        self.prepare_piscola_filters()
        df = self.gp_interp()
        df = self._fix_site_band_names(df)
        return df.rename(columns={"time": "jd"})


class LCInterp:
    def __init__(self, reference_times: ArrayLike, phot: Photometry, band: Filter) -> None:
        self._time = phot.jd
        self.reference_times = reference_times
        self.band = band
        self.svo_filter = band.svo
        self._phot = phot

        if not isinstance(phot, HasMag):
            raise TypeError("Photometry must have `mag` and `mag_err`.")
        # assert isinstance(phot, MagPhotometry)
        self._mag = self.interp(phot.mag)
        self._mag_err = self.interp(phot.mag_err)

        if isinstance(phot, HasFlux):
            self._flux = self.interp(phot.flux)
            self._flux_err = self.interp(phot.flux_err)
        else:
            if self.svo_filter is None:
                raise ValueError("Photometry must have `flux` and `flux_err` if `svo_filter` is not set.")
            self._flux = mag_to_flux(self.mag, self.svo_filter, magsys=self.band.magsys)
            self._flux_err = mag_to_flux_err(self._mag_err, self.flux)

    def interp(self, yp: QuantityArrayType) -> np.ndarray[Number, Any]:
        return np.interp(self.reference_times, self._time, yp)

    @property
    def mag(self):
        return self._mag

    @property
    def mag_err(self):
        return self._mag_err

    @property
    def flux(self):
        return self._flux

    @property
    def flux_err(self):
        return self._flux_err

    def to_phot(self) -> Photometry:
        df = pd.DataFrame(
            {
                "mag": self.mag,
                "mag_err": self.mag_err,
                "flux": self.flux,
                "flux_err": self.flux_err,
                "jd": self.reference_times,
            }
        )
        df["site"] = self._phot.site.unique()[0]
        df["band"] = self.band.name
        df["sub"] = True
        return PhotFactory.from_df(df)


class SNInterp:
    def __init__(
        self,
        sn: SN,
        filters_dict: Mapping[str, Filter],
        reference_times: NDArray[np.float64],
        interp_type: type[GPInterp] = PiscolaInterp,
    ) -> None:
        self.filters_dict = filters_dict
        self.reference_times = reference_times
        self.sn = sn
        self.interp: GPInterp = interp_type(sn)
        self._sitemap = self.sn.sites.sites

    def update_sitemap(self, phot: Photometry | pd.DataFrame) -> Photometry | pd.DataFrame:
        new_sites = phot.site.unique()
        sites = Sites(sites=self._sitemap)
        for site in new_sites:
            if site not in sites:
                new_site = sites.add_site(name=site)
                self._sitemap[new_site.id] = new_site
        phot.site = cast(SeriesSite, phot.site.apply(lambda x: sites[x].id))
        sites = Sites(sites=self._sitemap.copy())
        for site in self._sitemap.values():
            if site.name not in new_sites:
                sites.remove_site(site)
        self._sitemap = sites.sites
        return phot

    def _fix_df_for_phot(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sub"] = True
        p0 = self.sn.phase_zero if self.sn.phase_zero is not None else df["jd"].iloc[0]
        df["phase"] = df["jd"] - p0
        return df

    def gp_interp(self) -> Photometry:
        df = self.interp()
        df = self._fix_df_for_phot(df)
        df = self.update_sitemap(df)
        if isinstance(df, pd.DataFrame):
            return PhotFactory.from_df(df)
        return df

    def lc_interp(self, phot: Optional[Photometry] = None) -> Photometry:
        if phot is None:
            phot = self.sn.phot
        fac = PhotFactory(Phot())
        for band_name, band in self.filters_dict.items():
            band_phot = phot.masked(phot.band == band_name)
            lc = LCInterp(self.reference_times, band_phot, band)
            fac.concat_phot(lc.to_phot())
        return fac.sn_phot

    def create_interp_sn(self) -> SN:
        interp_phot = self.gp_interp()
        interp_phot_ref = self.lc_interp(interp_phot)
        sn = self.sn
        return SN.from_phot(
            interp_phot_ref,
            sn.name,
            sn.redshift,
            sn.phase_zero,
            sn.sub_only,
            sninfo=sn.sninfo,
            lims=sn.limits,
            sites=Sites(sites=self._sitemap),
        )


def mag_to_flux(mag: QuantityArrayType, filt: SVOFilter, magsys=MagSys.AB) -> u.Quantity:
    qmag = quantity_array(mag)
    if magsys == "AB":
        qmag = qmag << u.ABmag
        return qmag.to(FLAM, equivalencies=u.spectral_density(filt.wave_eff))
    elif magsys != "Vega":
        raise ValueError("`magsys` has to be one of `AB` or `Vega`")

    return 10 ** (qmag.value / -2.5) * zero_point_flux(filt, magsys)


def mag_to_flux_err(mag_err: QuantityArrayType, flux: QuantityArrayType) -> u.Quantity:
    return 2.303 * quantity_array(mag_err) * quantity_array(flux)
