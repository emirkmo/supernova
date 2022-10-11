from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from typing import Collection, Hashable, Mapping, MutableMapping, Optional, Any, Iterable, TypeVar, cast
from pathlib import Path
from unittest.util import strclass
import pandas as pd
from svo_filters import svo
import astropy.units as u
import numpy as np
from dust_extinction.parameter_averages import F99
from importlib.util import find_spec
from numpy.typing import NDArray
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler

from .utils import StrEnum, PathType, Number
FLAM = u.erg / u.cm ** 2 / u.s / u.angstrom

@dataclass
class FilterProfile:
    wave: u.Quantity
    throughput: NDArray


class MagSys(StrEnum):
    AB = 'AB'
    Vega = 'Vega'


def wave_pivot(throughput: NDArray, wave: u.Quantity) -> u.Quantity:
    return np.sqrt(np.trapz(throughput) / np.trapz(throughput / wave ** 2))


def get_resampled_vega(waves: u.Quantity) -> u.Quantity:
    vega_data = np.genfromtxt(Path(__file__).parent.absolute()/'vega.dat', unpack=True)[: 2]
    spec = Spectrum1D(spectral_axis=vega_data[0] * u.AA, flux=vega_data[1] * FLAM)
    new_spec = FluxConservingResampler()(spec, waves)
    return new_spec.flux


def wave_eff(waves: u.Quantity, tputs: NDArray,
             magsys: MagSys = MagSys.AB, flux_units: u.Quantity = FLAM) -> u.Quantity:
    """magsys has to be one of AB or Vega
    """
    if magsys == MagSys.Vega:
        jj_spec_flux = get_resampled_vega(waves)
    elif magsys == MagSys.AB:
        refjy = 3631.0 * u.Jy
        jj_spec_flux = np.ones_like(tputs) * refjy.to(flux_units, equivalencies=u.spectral_density(waves))
    else:
        raise ValueError("`magsys` has to be one of `AB` or `Vega`")
    top = np.trapz((waves * tputs * jj_spec_flux), x=waves)
    bot = np.trapz((tputs * jj_spec_flux), x=waves)
    _wave_eff = top / bot
    return _wave_eff.to(u.AA)


class SVOFilter(svo.Filter):
    MagSys: "MagSys"
    flux_units: u.Quantity = FLAM

    def __init__(self, band: str, profile: Optional[FilterProfile] = None,
                 mag_sys: Optional["MagSys"] = None, **kwargs: Any):
        if profile is None:
            super().__init__(band, wave_units=u.AA, **kwargs)
        else:
            self.wave = profile.wave.to(u.AA)
            self.throughput = profile.throughput
            self.MagSys = mag_sys if mag_sys is not None else MagSys.AB
        self._waves = cast(u.Quantity, self.wave.flatten()).to(u.AA)
        self._tputs = self.throughput.flatten()
        self._wave_pivot = wave_pivot(self.throughputs, self.waves)
        self._wave_eff = wave_eff(self.waves, self.throughputs, self.magsys)

    @property
    def magsys(self) -> "MagSys":
        return self.MagSys

    @magsys.setter
    def magsys(self, value: str):
        self.MagSys = MagSys(value)

    @property
    def waves(self) -> u.Quantity:
        return self._waves

    @waves.setter
    def waves(self, value: u.Quantity):
        self._waves = value.to(u.AA)

    @property
    def throughputs(self) -> NDArray:
        return self._tputs

    @property
    def wave_pivot(self) -> u.Quantity:
        return self._wave_pivot

    @wave_pivot.setter
    def wave_pivot(self, value: u.Quantity):
        self._wave_pivot = value.to(u.AA)

    @property
    def wave_eff(self) -> u.Quantity:
        return self._wave_eff

    @wave_eff.setter
    def wave_eff(self, value: u.Quantity):
        self._wave_eff = value.to(u.AA)

    def write_filter(self, path: PathType) -> None:
        tempdf = pd.DataFrame({'#lambda': self.waves.value,
                               'transmission': self.throughputs.flatten()})
        tempdf.to_csv(path, sep=' ', index=False)


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


for name, filt in FLOWS_FILTERS.items():
    filt.magsys = _MAG_SYS[name]
    FLOWS_FILTERS[name] = filt
    if name == 'B':
        filt.zp = 6.49135e-9 * FLAM
    if name == 'V':
        filt.zp = 3.73384e-9 * FLAM


def get_flows_filter(band: str) -> SVOFilter:
    if band not in FLOWS_FILTERS.keys():
        raise ValueError(f"Band: `{band}` not found in flows filter list: {set(FLOWS_FILTERS.keys())}")
    return FLOWS_FILTERS[band]


def zero_point_flux(svo_filt: SVOFilter, magsys='AB') -> u.Quantity:
    if magsys == 'Vega':
        return svo_filt.zp
    elif magsys != "AB":
        raise ValueError("`magsys` has to be one of `AB` or `Vega`")
    refjy = 3631.0 * u.Jy
    return refjy.to(FLAM, equivalencies=u.spectral_density(svo_filt.wave_pivot))


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
    wave_eff: u.Quantity = u.Quantity(0, u.AA)
    magsys: MagSys = MagSys.AB
    zp: u.Quantity = u.Quantity(0., FLAM)  # zero point flux in erg/s/cm^2/A
    plot_shift: float = 0
    plot_color: Optional[str | tuple[float, float, float]] = None
    svo: Optional[Any] = None  # should be SVOFilter but that's broken.
    svo_name: Optional[str] = None
    ext: float = 0.0  # extinction in magnitudes in band.

    def __post_init__(self):
        # Not needed if using factory or type hints.
        self.wave_eff = self.wave_eff << u.AA
        if isinstance(self.magsys, str):
            self.magsys = MagSys(self.magsys)
        if isinstance(self.zp, (int, float)):
            self.zp = self.zp << u.erg / u.cm ** 2 / u.s / u.AA
        # enf of defensive unit tests
        self.svo = self.get_svo_filter()
        if self.svo is not None:
            self.wave_eff = self.svo.wave_eff
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

    def set_extinction(self, ebv: float, rv: float = 3.1) -> "Filter":
        ext = F99(Rv=rv)
        self.ext = ext(self.wave_eff) * rv * ebv  # type: ignore
        return self
        

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "wave_eff": self.wave_eff.value,
            "magsys": self.magsys.value,
            "zp": self.zp.value,
            "plot_shift": self.plot_shift,
            "plot_color": self.plot_color,
            "svo_name": self.svo_name,
            "ext": self.ext
        }

    @classmethod
    def from_dict(cls, d: MutableMapping[str, Any]) -> "Filter":
        if 'name' not in d:
            raise ValueError("Filter dictionary must have a `name` key.")
        name: str = d.pop('name')
        return cls.filter_factory(name=name, **d)

    def __str__(self):
        return self.name

    @classmethod
    def filter_factory(cls, name: str, **kwargs: Any) -> "Filter":
        """
        A factory method to create a filter from a name.
        """
        wave_eff = kwargs.get("wave_eff", 0.) << u.AA
        zp = kwargs.get("zp", 0.) << u.erg / u.cm ** 2 / u.s / u.AA
        magsys = kwargs.get("magsys", "AB")
        magsys = MagSys(magsys)
        kwargs.update({"name": name, "wave_eff": wave_eff, "zp": zp, "magsys": magsys})

        return cls(**kwargs)

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

    @staticmethod
    def wave_sort(bands: Iterable[Filter]) -> list[Filter]:
        return sorted(bands, key=lambda x: x.wave_eff.value)

    def order_sort(self, bands: Iterable[Filter]) -> list[Filter]:
        return sorted(bands, key=lambda x: self(x.name))

BandsType = (
    Mapping[str, Filter | Mapping[str, Any]]
    | Collection[str]
    | Collection[Filter]
    | Collection[Mapping[str, Any]]
)

def _get_bands_dict(bands: BandsType) -> dict[str, Filter]:
    # Bands
    new_bands = {}
    if isinstance(bands, MutableMapping):
        for band_name, band in bands.items():
            if isinstance(band, MutableMapping):
                new_bands[band_name] = Filter.from_dict(band)
            if isinstance(band, Filter):
                new_bands[band_name] = band
            else:
                raise ValueError(f"Band must be a mapping or Filter, but is {type(band)}")

    if isinstance(bands, (list, tuple)):
        for band in bands:
            if isinstance(band, str):
                new_bands[band] = Filter(band)
            elif isinstance(band, Filter):
                new_bands[band.name] = band
            elif isinstance(band, MutableMapping):
                new_bands[band["name"]] = Filter.from_dict(band)

    return new_bands


def make_bands(
        bands: BandsType,
        band_order: Optional[list[str]] = None,
        ebv: float = 0.0,
        rv: float = 3.1,
    ) -> Mapping[str, Filter]:
        """
        Creates a sorted dictionary of Filter objects.
        Any unknown filters are added to the end. If band_order is given, the
        order of the bands is set to that, otherwise default_order from
        supernova.filters.FilterSorter is used, which is hard-coded based on wave_eff.
        """
        new_bands = _get_bands_dict(bands)

        wave_sort = band_order is None
        sorter = FilterSorter(band_order)
        if wave_sort:
            new_bands = sorter.wave_sort(new_bands.values())
        else:
            new_bands = sorter.order_sort(new_bands.values())

        return {
            b.name: b 
            if b.svo is None 
            else b.set_extinction(ebv, rv)
            for b in new_bands
        }

def pisco_filter_path() -> Path:
    pisc_spec: ModuleSpec | None = find_spec("piscola")
    if pisc_spec is None: 
        raise ImportError("piscola is not installed. Please install it with `pip install piscola`")
    pisc_folder = pisc_spec.origin
    if pisc_folder is None:
        raise ImportError("Cannot determine install location of Piscola. Got None.")
    return Path(pisc_folder).parent.absolute() / 'filters/'    

PISCO_FILTER_PATH = pisco_filter_path()
