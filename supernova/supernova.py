import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Dict, Union, Optional, TypeVar

import astropy.units as u
import numpy as np
import pandas as pd

from .filters import Filter, FilterSorter

Number = TypeVar('Number', int, float, u.Quantity)


@dataclass
class AbstractMagPhot(ABC):
    mag: pd.Series[Number]
    mag_err: pd.Series[Number]

    def __post_init__(self):
        if self.__class__ == AbstractMagPhot:
            raise TypeError("Cannot instantiate abstract class.")

    @abstractmethod
    def absmag(self, dm: Number, ext: Number) -> pd.Series[Number]:
        pass


@dataclass
class AbstractFluxPhot(ABC):
    flux: pd.Series[Number]
    flux_err: pd.Series[Number]

    def __post_init__(self):
        if self.__class__ == AbstractFluxPhot:
            raise TypeError("Cannot instantiate abstract class.")


@dataclass
class AbstractBasePhot(ABC):
    """Abstract Photometry dataclass"""
    jd: pd.Series[Number]
    band: pd.Series[str]
    phase: pd.Series[Number]
    sub: pd.Series[bool]
    site: pd.Series[int]

    def __post_init__(self):
        if self.__class__ == AbstractBasePhot:
            raise TypeError("Cannot instantiate abstract class.")

    @abstractmethod
    def calc_phases(self, phase_zero: Number) -> None:
        pass

    @abstractmethod
    def restframe_phases(self, redshift: Number) -> pd.Series[Number]:
        pass

    @abstractmethod
    def masked(self, cond: list) -> 'Photometry':
        pass

    @abstractmethod
    def as_dataframe(self) -> pd.DataFrame:
        pass


@dataclass
class BasePhot(AbstractBasePhot):
    jd: pd.Series[Number] = field(default_factory=pd.Series)
    band: pd.Series[str] = field(default_factory=pd.Series)
    phase: pd.Series[Number] = field(default_factory=pd.Series)
    sub: pd.Series[bool] = field(default_factory=pd.Series)
    site: pd.Series[int] = field(default_factory=pd.Series)

    def calc_phases(self, phase_zero):
        self.phase = self.jd - phase_zero

    def restframe_phases(self, redshift):
        if self.phase is None:
            raise AttributeError("self.phase must not be None, "
                                 "calculate it first using calc_phases with a phase zero")
        return self.phase / (1.0 + redshift)

    @classmethod
    def from_dict(cls, d: dict):
        if cls == BasePhot:
            raise TypeError("Cannot instantiate Base class. Use a subclass.")
        return cls(**{k: v for k, v in d.items() if k in [f.name for f in fields(cls)]})

    def masked(self, cond: list):
        d2 = {name: value[cond] for name, value in asdict(self).items()}
        return self.from_dict(d2)

    def as_dataframe(self) -> pd.DataFrame:
        yield pd.DataFrame(asdict(self))

    def __len__(self):
        return len(self.jd)


@dataclass
class MagPhot(BasePhot, AbstractMagPhot):
    mag: pd.Series[Number] = field(default_factory=pd.Series)
    mag_err: pd.Series[Number] = field(default_factory=pd.Series)

    def absmag(self, dm: Number, ext: Number) -> pd.Series[Number]:
        return self.mag-dm-ext

@dataclass    
class FluxPhot(BasePhot, AbstractFluxPhot):
    flux: pd.Series[Number] = field(default_factory=pd.Series)
    flux_err: pd.Series[Number] = field(default_factory=pd.Series)


@dataclass
class Phot(FluxPhot, MagPhot):
    mag: pd.Series[Number] = field(default_factory=pd.Series)
    mag_err: pd.Series[Number] = field(default_factory=pd.Series)
    flux: pd.Series[Number] = field(default_factory=pd.Series)
    flux_err: pd.Series[Number] = field(default_factory=pd.Series)


Photometry = FluxPhot | MagPhot | Phot


class PhotFactory:
    """
    Factory class to create Photometry objects.
    """

    @staticmethod
    def from_df(df: pd.DataFrame) -> Photometry:
        d = df.to_dict('series')
        mag = 'mag' in d
        flux = 'flux' in d
        if mag and flux:
            return Phot.from_dict(d)
        elif mag:
            return MagPhot.from_dict(d)
        elif flux:
            return FluxPhot.from_dict(d)
        raise ValueError("df must contain either 'mag' or 'flux' columns")


@dataclass
class SN:
    phot: pd.DataFrame | Photometry
    phases: pd.Series
    sninfo: pd.Series
    sites: Dict[int, str] = field(default_factory=dict)
    sites_r: Dict[str, int] = field(init=False)
    name: str = '20lao'
    sub_only: bool = True
    distance: float = None
    bands: list[Filter] = field(default_factory=list)
    limits: Optional[Photometry] = None
    # spectral_info:
    
    def __post_init__(self):
        if isinstance(self.phot, pd.DataFrame):
            self.phot = PhotFactory.from_df(self.phot)
        self.set_sites_r()
        self.rng = np.random.default_rng()
        self.distance = self.sninfo.dm
        if len(self.bands) == 0:
            self.bands = self.make_bands(bands=list(self.phot.band.unique()))

    def set_sites_r(self) -> None:
        self.sites_r = {value: key for key, value in self.sites.items()}
        
    def add_site(self, name: str, site_id: int = None) -> None:
        if site_id is None:
            site_id = self.rng.choice(set(range(100)) - set(self.sites.keys()))
        self.sites[site_id] = name
        self.set_sites_r()
    
    def band(self, filt: str, site: str = 'all', return_absmag: bool = False) -> Photometry:
        phot = self.phot.masked(self.phot.band == filt)
        if self.sub_only:
            phot = phot.masked(phot.sub.tolist())
        
        if site != 'all' and site not in self.sites_r.keys():
            raise ValueError('not a valid site name. Define first with `add_site`')
        
        if site != 'all':
            site_id = self.sites_r[site]
            phot = phot.masked(phot.site == site_id)
        if return_absmag:
            if not isinstance(phot, AbstractMagPhot):
                raise TypeError("Photometry must be MagPhot or Phot to return absolute magnitude.")
            phot.mag = phot.absmag(self.distance, self.sninfo[filt+'ext'])
        return phot
    
    def site(self, site: str) -> Photometry:
        site_id = self.sites_r[site]
        return self.phot.masked(self.phot.site == site_id)
    
    def absmag(self, filt: str, phot: AbstractMagPhot = None):
        if phot is None:
            phot = self.phot
        return phot.absmag(self.distance, self.sninfo[filt+'ext'])

    @staticmethod
    def make_bands(bands: Optional[list[str]] = None,
                   band_order: Optional[list[str]] = None) -> list[Filter]:
        """
        Creates a sorted list of Filter objects from the bands in the photometry.
        Any unknown filters are added to the end. If band_order is given, the
        order of the bands is set to that, otherwise default_order from
        supernova.filters.FilterSorter is used, which is hard-coded based on wave_eff.
        """
        bands.sort(key=FilterSorter(band_order))
        return [Filter(f) for f in bands]
    
    def to_csv(self, basepath: Union[str, Path] = Path('../')) -> None:
        SNSerializer(self).to_csv(basepath)
            
    @classmethod
    def from_csv(cls, dirpath: Union[str, Path]):
        return SNSerializer.from_csv(dirpath)


class SNSerializer:
    names = 'phot,phases,sninfo,sites,filters,limits'.split(',')

    def __init__(self, sn: SN):
        self.sn = sn

    def to_csv(self, basepath: Union[str, Path] = Path('../')) -> None:
        sn = self.sn
        names = SNSerializer.names

        basepath = Path(basepath).joinpath(Path(f"SNClass_{sn.name}/"))
        os.makedirs(basepath, exist_ok=True)

        sn.sninfo['name'] = sn.name
        sn.sninfo['sub_only'] = sn.sub_only
        for name, _field in zip(names, [sn.phot, sn.phases, sn.sninfo, sn.sites, sn.bands]):
            if isinstance(_field, dict):
                _field = pd.Series(_field)
            if isinstance(_field, Photometry):
                _field = pd.DataFrame.from_dict(asdict(_field))
            if isinstance(_field, list):
                _field = pd.Series([f.name for f in _field])
            save_name = f"{basepath.absolute()}/{sn.name}_{name}.csv"
            _field.to_csv(save_name, index=True)

    @staticmethod
    def from_csv(dirpath: Union[str, Path]) -> SN:
        import glob
        names = SNSerializer.names
        _fields = {}
        csvs = glob.glob(str(Path(dirpath) / '*.csv'))
        _sn_dict = {}
        for csv in csvs:
            df = pd.read_csv(csv, index_col=0).squeeze("columns")
            for name in names:
                if name in csv:
                    df.name = name
                    _fields[name] = df

        # Dirty trick, unneeded if we would serialize to json, but then it's less readable
        # by astronomers.
        _fields['sninfo'] = force_numeric_sninfo(_fields['sninfo'])
        _fields['sites'] = _fields['sites'].to_dict()

        return SN(
            phot=_fields['phot'],
            phases=_fields['phases'],
            sninfo=_fields['sninfo'],
            sites=_fields['sites'],
            name=_fields['sninfo'].loc['name'] if 'name' in _fields['sninfo'].index else 'unknown',
            sub_only=bool(_fields['sninfo'].sub_only) if 'sub_only' in _fields['sninfo'].index else False,
            bands=SN.make_bands(bands=_fields['filters'].tolist(),
                                band_order=_fields['filters'].index.tolist()) if 'filters' in _fields else [],
            limits=_fields['limits'] if 'limits' in _fields else None
            )


def force_numeric_sninfo(sninfo: pd.Series) -> pd.Series:
    _sninfo = pd.to_numeric(sninfo, errors='coerce')
    _sninfo = _sninfo.mask(_sninfo.isna(), sninfo)
    return _sninfo
