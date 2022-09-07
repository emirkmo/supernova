import astropy.units as u
import numpy as np
from svo_filters import svo
from .supernova import SN, Number
from .filters import get_flows_filter
from numpy.typing import ArrayLike
filter_dict = dict(zip(['B', 'g', 'V', 'r', 'i'],
                       ['Bctio', 'ps1_g', 'Vctio', 'ps1_r', 'ps1_i']))

ArrayLike = ArrayLike | Number


class SNInterp:

    def __init__(self, sn: SN, filters_dict: dict[str, str], reference_times):
        self.filters_dict = filters_dict
        self.filters = list(filters_dict.keys())
        self.reference_times = reference_times
        self.sn = sn

    def get_lcinterp(self):
        interps = {}
        # modrange = np.linspace(pep.tess_patrick, simple.time.max(), 200)
        for filt, filtname in self.filters_dict.items():
            s = self.sn.band(filtname)
            lc = LCInterp(self.reference_times,
                          s.jd.values,
                          s.mag.values - self.sn.distance - self.sn.sninfo[filt + 'ext'],
                          s.mag_err.values,
                          filt,
                          s.flux.values,
                          s.flux_err.values
                          )
            interps[filt] = lc


class LCInterp:

    def __init__(self, reference_times, time, mag, mag_err, band,
                 flux=None, flux_err=None):
        self._time = time
        self.reference_times = reference_times
        self.band = band
        self.svo_filter = get_flows_filter(band)

        self._mag = self.interp(mag)
        self._mag_err = self.interp(mag_err)

        self._flux = np.array([mag_to_flux(m, self.svo_filter, magsys=self.svo_filter.magsys) for m in
                               self.mag]) if flux is None else flux
        self._flux_err = mag_to_flux_err(self._mag_err, self.flux) if flux_err is None else flux_err

    def interp(self, yp):
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


flam = u.erg / u.cm ** 2 / u.s / u.angstrom


def wave_eff(svo_filt: svo.Filter, magsys='AB'):
    """magsys has to be one of AB or Vega
    """
    if magsys == 'Vega':
        return svo_filt.wave_eff.to(u.AA)
    elif magsys != "AB":
        raise ValueError("`magsys` has to be one of `AB` or `Vega`")

    refjy = 3631.0 * u.Jy
    jj_spec_flux = refjy.to(svo_filt.flux_units, equivalencies=u.spectral_density(svo_filt.wave_pivot))
    top = np.trapz((svo_filt.wave ** 2 * svo_filt.throughput * jj_spec_flux), x=svo_filt.wave)
    bot = np.trapz((svo_filt.wave * svo_filt.throughput * jj_spec_flux), x=svo_filt.wave)
    _wave_eff = top / bot
    return _wave_eff[0].to(u.AA)


def zero_point_flux(svo_filt: svo.Filter, magsys='AB'):
    if magsys == 'Vega':
        return svo_filt.zp
    elif magsys != "AB":
        raise ValueError("`magsys` has to be one of `AB` or `Vega`")
    refjy = 3631.0 * u.Jy
    return refjy.to(flam, equivalencies=u.spectral_density(svo_filt.wave_pivot))

# class QuantityMeta(metaclass=u.Quantity):
#     def __rshift__(self, other):
#         return self * other


def mag_to_flux(mag: ArrayLike, filt: svo.Filter, magsys='AB'):
    if magsys == 'AB':
        mag = mag << u.ABmag
        return mag.to(flam, equivalencies=u.spectral_density(wave_eff(filt, magsys)))
    elif magsys != "Vega":
        raise ValueError("`magsys` has to be one of `AB` or `Vega`")

    return 10 ** (mag / -2.5) * zero_point_flux(filt, magsys)


def mag_to_flux_err(mag_err, flux: u.Quantity):
    return 2.303 * mag_err * flux
