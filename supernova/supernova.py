import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, asdict, replace
from pathlib import Path
from typing import Dict, Union, Optional, Any, Mapping
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Cosmology, WMAP5, realizations, default_cosmology
# noinspection PyProtectedMember
from astropy.coordinates.name_resolve import NameResolveError
from .filters import Filter, FilterSorter
from .sites import Sites
Number = int | float | u.Quantity


@dataclass
class AbstractMagPhot(ABC):
    mag: pd.Series
    mag_err: pd.Series

    def __post_init__(self):
        if self.__class__ == AbstractMagPhot:
            raise TypeError("Cannot instantiate abstract class.")

    @abstractmethod
    def absmag(self, dm: Number, ext: Number) -> pd.Series:
        pass


@dataclass
class AbstractFluxPhot(ABC):
    flux: pd.Series
    flux_err: pd.Series

    def __post_init__(self):
        if self.__class__ == AbstractFluxPhot:
            raise TypeError("Cannot instantiate abstract class.")


@dataclass
class AbstractBasePhot(ABC):
    """Abstract Photometry dataclass"""
    jd: pd.Series
    band: pd.Series
    filter: pd.Series
    phase: pd.Series
    sub: pd.Series
    site: pd.Series

    # restframe: pd.Series

    def __post_init__(self):
        if self.__class__ == AbstractBasePhot:
            raise TypeError("Cannot instantiate abstract class.")

    @abstractmethod
    def calc_phases(self, phase_zero: Number) -> None:
        pass

    @abstractmethod
    def restframe_phases(self, redshift: Number) -> pd.Series:
        pass

    @abstractmethod
    def masked(self, cond: list) -> 'Photometry':
        pass

    @abstractmethod
    def as_dataframe(self) -> pd.DataFrame:
        pass


@dataclass
class BasePhot(AbstractBasePhot):
    jd: pd.Series = field(default=pd.Series(dtype=float))
    band: pd.Series = field(default=pd.Series(dtype=str))
    phase: pd.Series = field(default=pd.Series(dtype=float))
    sub: pd.Series = field(default=pd.Series(dtype=bool))
    site: pd.Series = field(default=pd.Series(dtype=int))
    # only for backwards compatibility do not access directly as it overrides builtin filter.
    filter: pd.Series = field(default=pd.Series(dtype=str))

    # restframe: pd.Series = field(default=pd.Series(dtype=float))

    def __post_init__(self):
        if 0 < len(self.filter) == len(self):
            warnings.warn("Phot: `filter` is deprecated, use `band` instead", DeprecationWarning)
            self.band = self.filter

        if len(self.band) != len(self):
            raise ValueError("Either band or filter must be given the same length as jd."
                             f"Got band: {len(self.band)}, "
                             f"filter: {len(self.filter)}, jd: {len(self)}")
        self.filter = self.band

    def calc_phases(self, phase_zero):
        self.phase = self.jd - phase_zero

    def restframe_phases(self, redshift):
        if self.phase is None:
            raise AttributeError("self.phase must not be None, "
                                 "calculate it first using calc_phases with a phase_zero")
        return self.phase / (1 + redshift)

    @classmethod
    def from_dict(cls, d: dict):
        if cls == BasePhot:
            raise TypeError("Cannot instantiate Base class. Use a subclass.")
        return cls(**{k: v for k, v in d.items() if k in [f.name for f in fields(cls)]})

    def masked(self, cond: list):
        d2 = {name: value[cond] for name, value in asdict(self).items()}
        return self.from_dict(d2)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(self))

    def __len__(self):
        return len(self.jd)


@dataclass
class MagPhot(BasePhot, AbstractMagPhot):
    mag: pd.Series = field(default=pd.Series(dtype=float))
    mag_err: pd.Series = field(default=pd.Series(dtype=float))

    def absmag(self, dm: Number, ext: Number) -> pd.Series:
        return self.mag - dm - ext


@dataclass
class FluxPhot(BasePhot, AbstractFluxPhot):
    flux: pd.Series = field(default=pd.Series(dtype=float))
    flux_err: pd.Series = field(default=pd.Series(dtype=float))


@dataclass
class Phot(FluxPhot, MagPhot):
    mag: pd.Series = field(default=pd.Series(dtype=float))
    mag_err: pd.Series = field(default=pd.Series(dtype=float))
    flux: pd.Series = field(default=pd.Series(dtype=float))
    flux_err: pd.Series = field(default=pd.Series(dtype=float))


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
class SNInfo:
    name: str
    redshift: float = 0
    phase_zero: Optional[float] = None
    ra: Optional[Number | str] = None
    dec: Number | str = None

    ebv: Number = -99
    dm: Number = -99
    cosmology: str | Cosmology = WMAP5

    sub_only: bool = True
    coords: SkyCoord = field(init=False, default=None)

    def __post_init__(self):
        if isinstance(self.cosmology, str):
            self.cosmology = cosmo_from_name(self.cosmology)
        self.set_coords()
        if self.dm == -99 and self.redshift != 0:
            self.dm = self.distance_modulus_from_cosmo()
        if self.ebv == -99:
            self.get_ebv()

    def set_coords(self) -> None:
        def from_str(ra_s: str, dec_s: str):
            return SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg),
                            frame='icrs', equinox='J2000')

        if isinstance(self.ra, str) and isinstance(self.dec, str):
            self.coords = from_str(self.ra, self.dec)

        elif isinstance(self.ra, Number) and isinstance(self.dec, Number):
            self.coords = SkyCoord(self.ra, self.dec, unit=(u.deg, u.deg),
                                   frame='icrs', equinox='J2000')
        elif self.name.startswith("SN"):
            try:
                self.coords = SkyCoord.from_name(self.name)
            except NameResolveError:
                pass

        if self.coords is None:
            query_str = ("Could not resolve coordinates for this SN."
                         "Please enter them manually. ra format:"
                         " hh:mm:ss or 00h00m00.0s dec "
                         "format: dd:mm:ss or +/-00d00m00.0s. RA first, RA:")
            ra = self.query_user(query_str)
            dec = self.query_user("Dec:")
            self.coords = SkyCoord(  # will raise appropriate error if format is wrong.
                ra, dec, unit=(u.hourangle, u.deg),
                frame='icrs', equinox='J2000')
        self.ra = self.coords.ra.deg
        self.dec = self.coords.dec.deg

    @staticmethod
    def query_user(query: str) -> str:
        return input(query)

    def distance_modulus_from_cosmo(self) -> float:
        if self.redshift == 0:
            raise ValueError("Redshift cannot be 0 if calculating distance modulus.")
        dm = self.cosmology.distmod(self.redshift)  # type: ignore
        return dm.value

    def get_ebv(self):
        try:
            ebv = get_extinction_irsa(self.coords)

        except Exception as e:
            print(e)
            ebv = self.query_user("Could not get E(B-V) from IRSA."
                                  " Please enter manually:")
        self.ebv = float(ebv)

    def to_series(self) -> pd.Series:
        d = asdict(self)
        d['cosmology'] = d['cosmology'].name
        d.pop('coords')
        return pd.Series(d)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, item):
        return hasattr(self, item)

    @classmethod
    def from_dict(cls, d: Mapping) -> "SNInfo":
        return cls(**{k: v for k, v in d.items() if k in [f.name for f in fields(cls)]})

    @classmethod
    def from_csv(cls, csv: Path) -> "SNInfo":
        df = pd.read_csv(csv, index_col=0).squeeze("columns")
        df = force_numeric_sninfo(df)
        return cls.from_dict(df.to_dict())


@dataclass
class SN:
    phot: pd.DataFrame | Photometry
    phases: pd.Series
    sninfo: pd.Series | SNInfo
    sites: Sites = field(default_factory=Sites)
    # sites_r: Dict[str, int] = field(init=False)
    name: str = '20lao'
    sub_only: bool = True
    distance: float = 0.  # distance modulus. 0 means not set.
    bands: dict[str, Filter] = field(default_factory=dict)
    limits: Optional[pd.DataFrame | Photometry] = None
    phase_zero: float = field(init=False, default=0)
    redshift: float = field(init=False, default=0)

    # spectral_info:

    def __post_init__(self):
        if isinstance(self.phot, pd.DataFrame):
            self.phot = PhotFactory().from_df(self.phot)
        if isinstance(self.limits, pd.DataFrame):
            self.limits = PhotFactory().from_df(self.limits)
        # self.set_sites_r()
        self.rng = np.random.default_rng()
        self.distance = self.sninfo.get('dm', 0.)
        if len(self.bands) == 0:
            self.bands = self.make_bands(bands=list(self.phot.band.unique()), ebv=self.sninfo.ebv)
        if "redshift" in self.sninfo:
            self.redshift = self.sninfo.redshift
        if "phase_zero" in self.phases:
            self.phase_zero = self.phases.phase_zero
            self.set_phases()

    def add_site(self, name: str, site_id: Optional[int] = None, **sitekwargs) -> None:
        if site_id is not None:
            sitekwargs['id'] = site_id
        self.sites.add_site(name, **sitekwargs)

    def band(self, filt: str, site: str = 'all', return_absmag: bool = False, lims: bool = False) -> Photometry:
        phot = self.phot.masked(self.phot.band == filt) if not lims else self.limits.masked(self.limits.band == filt)
        if self.sub_only:
            phot = phot.masked(phot.sub.tolist())

        if site != 'all' and site not in self.sites:
            raise ValueError('not a valid site name. Define first with `add_site`')

        if site != 'all':
            site_id = self.sites[site].id
            phot = phot.masked(phot.site == site_id)
        if return_absmag:
            if not isinstance(phot, AbstractMagPhot):
                raise TypeError("Photometry must be MagPhot or Phot to return absolute magnitude.")
            phot.mag = phot.absmag(self.distance, self.bands[filt].ext)
        return phot

    def site(self, site: str) -> Photometry:
        site_id = self.sites[site].id
        return self.phot.masked(self.phot.site == site_id)

    def absmag(self, filt: str, phot: AbstractMagPhot = None):
        if phot is None:
            phot = self.phot
        return phot.absmag(self.distance, self.bands[filt].ext)

    @staticmethod
    def make_bands(bands: Optional[list[str]] = None,
                   band_order: Optional[list[str]] = None,
                   ebv: float = 0.0) -> dict[str, Filter]:
        """
        Creates a sorted list of Filter objects from the bands in the photometry.
        Any unknown filters are added to the end. If band_order is given, the
        order of the bands is set to that, otherwise default_order from
        supernova.filters.FilterSorter is used, which is hard-coded based on wave_eff.
        """
        bands.sort(key=FilterSorter(band_order))
        return {b: filt if filt.svo is None else filt.set_extinction(ebv) for b in bands if (filt := Filter(b))}
        # for f in bands:
        #     filt = Filter(f)
        #     if filt.svo is not None:
        #         filt.set_extinction(0.2)
        #     yield filt

    def to_csv(self, basepath: Union[str, Path] = Path('../')) -> None:
        SNSerializer(self).to_csv(basepath)

    @classmethod
    def from_csv(cls, dirpath: Union[str, Path]):
        return SNSerializer.from_csv(dirpath)

    def set_phases(self, phase: Optional[float] = None) -> None:
        """
        Sets the phase of the SN to the given value.
        Photometry objects are updated with the new phase.
        """
        self.phase_zero = phase if phase is not None else self.phase_zero
        self.phot.calc_phases(self.phase_zero)
        self.phases.loc['phase_zero'] = self.phase_zero
        if self.limits is not None:
            self.limits.calc_phases(self.phase_zero)

    def restframe(self) -> 'SN':
        """
        Returns a copy of the SN object with the phases shifted to restframe.
        """
        sn = replace(self)
        sn.set_phases()
        sn.phot.phase = sn.phot.restframe_phases(self.redshift)
        if self.limits is not None:
            sn.limits.phase = sn.limits.restframe_phases(self.redshift)
        return sn

    @classmethod
    def from_phot(cls, phot: Photometry, name: str, redshift: float, sub_only: bool = False,
                  phase_zero: Optional[float] = None, lims: Optional[Photometry] = None,
                  sninfo: Optional[SNInfo] = None, sites: Optional[Sites] = None, **sninfo_kwargs: Any) -> "SN":
        if sninfo is not None:
            sninfo.redshift = redshift
            sninfo.name = name
        else:
            sninfo = SNInfo(redshift=redshift, name=name, sub_only=sub_only, **sninfo_kwargs)

        if phase_zero is None:
            phase_zero = phot.jd.min()
            warnings.warn(f"`phase_zero` not given. Setting to first photometry point: {phase_zero}."
                          " If this is not correct, set `phase_zero` manually by calling "
                          "`set_phases(<correct_phase_zero>)` on the SN object.")

        return cls(
            name=name,
            phot=phot,
            phases=pd.Series({'phase_zero': phase_zero}),
            sninfo=sninfo.to_series(),
            sites=sites if sites is not None else Sites.from_sitemap({site: str(site) for site in phot.site.unique()}),
            sub_only=sub_only,
            bands=SN.make_bands(bands=phot.band.unique().tolist(), ebv=sninfo.ebv),
            limits=lims
        )


class SNSerializer:
    names = 'phot,phases,sninfo,sites,bands,limits'.split(',')

    def __init__(self, sn: SN):
        self.sn = sn

    def to_csv(self, basepath: Union[str, Path] = Path('../')) -> None:
        sn = self.sn
        names = SNSerializer.names

        basepath = Path(basepath).joinpath(Path(f"SNClass_{sn.name}/"))
        os.makedirs(basepath, exist_ok=True)

        sn.sninfo['name'] = sn.name
        sn.sninfo['sub_only'] = sn.sub_only
        field_vals = {f.name: getattr(sn, f.name) for f in fields(sn) if f.name in names}
        for name, _field in field_vals.items():
            if name == 'bands':
                _field = pd.DataFrame([f.to_dict() for f in _field.values()])
            if isinstance(_field, dict):
                _field = pd.Series(_field)
            if isinstance(_field, Photometry):
                _field = pd.DataFrame.from_dict(asdict(_field))
            if isinstance(_field, Sites):
                _field = _field.to_df()
            if isinstance(_field, list):
                _field = pd.Series([f.name for f in _field])
            if isinstance(_field, SNInfo):
                _field = _field.to_series()
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
        _fields['sites'] = Sites.from_df(_fields['sites'])
        if isinstance(_fields['bands'], pd.Series):
            _fields['bands'] = SN.make_bands(
                bands=_fields['bands'].tolist(),
                band_order=_fields['bands'].index.tolist(),
                ebv=_fields['sninfo'].ebv) if 'bands' in _fields else {}

        elif isinstance(_fields['bands'], pd.DataFrame):
            _bands_dict = _fields['bands'].to_dict('index')
            _fields['bands'] = {v['name']: Filter.from_dict(v) for v in _bands_dict.values()}
        else:
            _fields['bands'] = {}
        sninfo = SNInfo.from_dict(_fields['sninfo'].to_dict())
        return SN(
            phot=_fields['phot'],
            phases=_fields.get('phases', pd.Series({'phase_zero': sninfo.phase_zero})),
            sninfo=sninfo,
            sites=_fields['sites'],
            name=sninfo.name,
            sub_only=sninfo.sub_only,
            bands=_fields['bands'],
            limits=_fields['limits'] if 'limits' in _fields else None
        )


def cosmo_from_name(name: str) -> Cosmology:
    if name in realizations.__all__:
        return getattr(realizations, name)
    warnings.warn(f"Could not find cosmology {name}. Using WMAP5 instead.")
    return WMAP5


def get_extinction_irsa(coords: SkyCoord) -> float:
    from astroquery.irsa_dust import IrsaDust
    table = IrsaDust.get_query_table(coords, section='ebv')
    return table["ext SandF ref"][0]

def force_numeric_sninfo(sninfo: pd.Series) -> pd.Series:
    _sninfo = pd.to_numeric(sninfo, errors='coerce')
    _sninfo = _sninfo.mask(_sninfo.isna(), sninfo)
    return _sninfo
