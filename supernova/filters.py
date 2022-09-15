from dataclasses import dataclass
from typing import Optional, Any
from svo_filters import svo
import astropy.units as u
import numpy as np
from dust_extinction.parameter_averages import F99
from .utils import StrEnum
flam = u.erg / u.cm ** 2 / u.s / u.angstrom
Number = int | float | u.Quantity


class SVOFilter(svo.Filter):
    MagSys: str

    @property
    def magsys(self) -> str:
        return self.MagSys

    @magsys.setter
    def magsys(self, value: str):
        self.MagSys = value


flam = u.erg / u.cm ** 2 / u.s / u.angstrom
FLOWS_FILTERS = {
    "u": SVOFilter("SDSS.u"),
    "g": SVOFilter("PS1.g"),
    "r": SVOFilter("PS1.r"),
    "i": SVOFilter("PS1.i"),
    "z": SVOFilter("PS1.z"),
    "B": SVOFilter('MISC/APASS.B'),
    "V": SVOFilter('MISC/APASS.V'),
    "R": SVOFilter('NOT/ALFOSC.Bes_R'),
    "J": SVOFilter('2MASS.J'),
    "H": SVOFilter('2MASS.H'),
    "K": SVOFilter('2MASS.Ks')
}

_MAG_SYS = {
    "u": 'AB',
    "g": 'AB',
    "r": 'AB',
    "i": 'AB',
    "z": 'AB',
    "B": 'Vega',
    "V": 'Vega',
    "R": 'Vega',
    "J": 'Vega',
    "H": 'Vega',
    "K": 'Vega'
}


class FilterSorter:
    """
    Use in list.sort(key=FilterSorter(order: list[str]))
    Will use DEFAULT_ORDER if order is None. You can also
    import and append to DEFAULT_ORDER.
    """
    default_order = ["u", "B", "V", "g", "r", "R", "i", "I", "z", "Y", "J", "H", "K"]

    def __init__(self, order: Optional[list] = None):
        self.order = order if order is not None else self.default_order

    def __call__(self, band: str) -> int:
        if band not in self.order:
            return len(self.order)
        return self.order.index(band)


for name, filt in FLOWS_FILTERS.items():
    filt.magsys = _MAG_SYS[name]
    FLOWS_FILTERS[name] = filt
    if name == 'B':
        filt.zp = 6.49135e-9 * flam
    if name == 'V':
        filt.zp = 3.73384e-9 * flam


def get_flows_filter(band: str) -> SVOFilter:
    if band not in FLOWS_FILTERS.keys():
        raise ValueError(f"Band: `{band}` not found in flows filter list: {set(FLOWS_FILTERS.keys())}")
    return FLOWS_FILTERS.get(band)


class MagSys(StrEnum):
    AB = 'AB'
    Vega = 'Vega'


@dataclass
class Filter:
    """
    A class to represent a filter.
    Stores the filter name, the filter object from the SVO Filter Profile Service,
    the shift and color for plotting,
    as well as the effective wavelength and zero point flux (defaulting to zero),
    if relevant.
    """
    name: str
    wave_eff: Number = 0 * u.AA
    magsys: MagSys = MagSys.AB
    zp: Number = 0.0  # zero point flux in erg/s/cm^2/A
    plot_shift: float = 0.0
    plot_color: Optional[str | tuple[float, float, float]] = None
    svo: Optional[Any] = None  # should be SVOFilter but that's broken.
    svo_name: Optional[str] = None
    ext: Number = 0.0  # extinction in magnitudes in band.

    def __post_init__(self):
        self.wave_eff = self.wave_eff << u.AA
        if isinstance(self.magsys, str):
            self.magsys = MagSys(self.magsys)
        if isinstance(self.zp, (int, float)):
            self.zp = self.zp << u.erg / u.cm ** 2 / u.s / u.AA
        self.svo = self.get_svo_filter()
        if self.svo is not None:
            self.wave_eff = wave_eff(self.svo)
            self.zp = zero_point_flux(self.svo, self.magsys)
            self.svo_name = self.svo.name

    def get_svo_filter(self) -> Optional[SVOFilter]:
        if isinstance(self.svo, SVOFilter):
            return self.svo
        if self.svo_name is not None:
            try:
                return SVOFilter(self.svo_name)  # will error if you don't know what you are doing.
            except IndexError:
                pass
        try:
            return get_flows_filter(self.name)
        except ValueError:
            pass
        try:
            return SVOFilter(self.name)
        except IndexError:
            return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "wave_eff": self.wave_eff.value,
            "magsys": self.magsys,
            "zp": self.zp.value,
            "plot_shift": self.plot_shift,
            "plot_color": self.plot_color,
            "svo_name": self.svo_name,
            "ext": self.ext
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Filter":
        return cls(**d)

    def __str__(self):
        return self.name

    def set_extinction(self, ebv: float, rv: float = 3.1) -> "Filter":
        ext = F99(Rv=rv)
        self.ext = ext(self.wave_eff) * rv * ebv
        return self
        #self.ext = -2.5 * np.log10(ext.extinguish(self.wave_eff, Ebv=ebv))


def wave_eff(svo_filt: SVOFilter, magsys='AB'):
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


def zero_point_flux(svo_filt: SVOFilter, magsys='AB'):
    if magsys == 'Vega':
        return svo_filt.zp
    elif magsys != "AB":
        raise ValueError("`magsys` has to be one of `AB` or `Vega`")
    refjy = 3631.0 * u.Jy
    return refjy.to(flam, equivalencies=u.spectral_density(svo_filt.wave_pivot))

