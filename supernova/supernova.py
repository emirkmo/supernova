from lib2to3.pgen2.token import OP
import os
import warnings
from dataclasses import dataclass, field, fields, asdict, replace
from pathlib import Path
from typing import Collection, Hashable, Union, Optional, Any, Mapping, cast
from collections.abc import MutableSequence
from typing_extensions import NotRequired, Self
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Cosmology, WMAP5, realizations  # type: ignore

# noinspection PyProtectedMember
from astropy.coordinates.name_resolve import NameResolveError
from .utils import Number, VerbosePrinter
from .filters import (BandsType, Filter, FilterSorter, PISCO_FILTER_PATH, SVOFilter, make_bands)
from .sites import Sites
from .photometry import (
    _HasFlux,
    HasMag,
    Phot,
    Photometry,
    PhotFactory,
    ImplementsFlux,
    ImplementsMagFlux,
    ImplementsMag,
)


@dataclass
class SNInfo:
    name: str
    redshift: float = 0
    phase_zero: Optional[float] = None
    ra: Optional[Number | str] = None
    dec: Optional[Number | str] = None

    ebv: float = -99
    dm: float = -99
    cosmology: str | Cosmology = WMAP5

    sub_only: bool = True
    coords: SkyCoord = field(init=False)

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
            return SkyCoord(
                ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs", equinox="J2000"
            )

        coords = None
        if isinstance(self.ra, str) and isinstance(self.dec, str):
            coords = from_str(self.ra, self.dec)

        elif isinstance(self.ra, Number) and isinstance(self.dec, Number):
            coords = SkyCoord(
                self.ra, self.dec, unit=(u.deg, u.deg), frame="icrs", equinox="J2000"
            )
        elif self.name.startswith("SN"):
            try:
                coords = SkyCoord.from_name(self.name)
            except NameResolveError:
                pass

        if coords is None:
            query_str = (
                "Could not resolve coordinates for this SN."
                "Please enter them manually. ra format:"
                " hh:mm:ss or 00h00m00.0s dec "
                "format: dd:mm:ss or +/-00d00m00.0s. RA first, RA:"
            )
            ra = self.query_user(query_str)
            dec = self.query_user("Dec:")
            coords = SkyCoord(  # will raise appropriate error if format is wrong.
                ra, dec, unit=(u.hourangle, u.deg), frame="icrs", equinox="J2000"
            )
        self.coords = coords
        self.ra = self.coords.ra.deg  # type: ignore
        self.dec = self.coords.dec.deg  # type: ignore

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
            ebv = self.query_user(
                "Could not get E(B-V) from IRSA." " Please enter manually:"
            )
        self.ebv = float(ebv)

    def to_series(self) -> pd.Series:
        d = asdict(self)
        d["cosmology"] = d["cosmology"].name
        d.pop("coords")
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

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


def sn_factory(
    phot: pd.DataFrame | Photometry,
    sninfo: pd.Series | SNInfo | dict,
    limits: Optional[pd.DataFrame | Photometry] = None,
    phases: Optional[pd.Series | dict] = None,
    bands: Optional[BandsType] = None,
    sites: Optional[Sites] = None,
) -> "SN":
    # Photometry & Limits
    if isinstance(phot, pd.DataFrame):
        phot = PhotFactory.from_df(phot)
    if isinstance(limits, pd.DataFrame):
        limits = PhotFactory.from_df(limits)

    # SNINFO
    if isinstance(sninfo, pd.Series):
        sninfo = SNInfo.from_dict(sninfo.to_dict())
    elif isinstance(sninfo, dict):
        sninfo = SNInfo.from_dict(sninfo)

    # Phases
    if phases is None:
        phases = {}
    if isinstance(phases, dict):
        phases = pd.Series(phases)
    if "phase_zero" not in phases:
        phases["phase_zero"] = sninfo.phase_zero

    # Bands
    usebands = phot.band.unique().tolist() if bands is None else bands
    new_bands = make_bands(usebands, ebv=sninfo.ebv)

    # Sites
    if sites is None:
        sites = Sites()
        if len(phot) > 0:
            sites = Sites.from_list(phot.site.unique().astype(str))

    return SN(phot=phot, sninfo=sninfo, limits=limits, phases=phases, bands=new_bands, sites=sites)



@dataclass
class SN:
    phot: Photometry
    phases: pd.Series
    sninfo: SNInfo
    sites: Sites = field(default_factory=Sites)
    name: str = "SN2020XXX"
    sub_only: bool = True
    distance: float = 0.0  # distance modulus. 0 means not set.
    bands: Mapping[str, Filter] = field(default_factory=dict)
    limits: Optional[Photometry] = None
    phase_zero: float = field(init=False, default=0)
    redshift: float = field(init=False, default=0)
    spectral_info: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.rng = np.random.default_rng()
        # Update metadata from sninfo
        self.meta_from_sninfo(self.sninfo)

        if len(self.sites) == 0:
            if len(self.phot) > 0:
                self.generate_sites()

    def meta_from_sninfo(self, sninfo: SNInfo):
        self.sninfo = sninfo
        self.distance = self.sninfo.dm if self.sninfo.dm != -99 else self.distance
        self.name = self.sninfo.name
        self.sub_only = self.sninfo.sub_only
        self.phase_zero = (
            self.sninfo.phase_zero
            if self.sninfo.phase_zero is not None
            else self.phase_zero
        )
        self.redshift = self.sninfo.redshift
        self.set_phases()

    def add_site(self, name: str, site_id: Optional[int] = None, **sitekwargs) -> None:
        if site_id is not None:
            sitekwargs["id"] = site_id
        self.sites.add_site(name=name, **sitekwargs)

    def generate_sites(self) -> None:
        """(Re)generate Sites from Photometry."""
        self.sites = Sites.from_phot(self.phot)

    def band(
        self,
        filt: str,
        site: str = "all",
        return_absmag: bool = False,
        lims: bool = False,
        flux: bool = False,
    ) -> Photometry:
        """Return photometry for a given band. If site is not specified,
        return all sites.
        """
        if lims and self.limits is None:
            raise ValueError("No limits set.")
        phot = (
            self.phot.masked(self.phot.band == filt)
            if not lims
            else self.limits.masked(self.limits.band == filt)  # type: ignore
        )
        if self.sub_only:
            phot = phot.masked(phot.sub.tolist())

        if site != "all" and site not in self.sites:
            raise ValueError("not a valid site name. Define first with `add_site`")

        if site != "all":
            site_id = self.sites[site].id
            phot = phot.masked(phot.site == site_id)
        if return_absmag:
            if not isinstance(phot, HasMag):
                raise TypeError(
                    "Photometry must be MagPhot or Phot to return absolute magnitude."
                )
            phot.mag = phot.absmag(self.distance, self.bands[filt].ext)
        if flux:
            if isinstance(phot, ImplementsMag):
                phot = Phot.from_magphot(phot, self.bands[filt])
            if isinstance(phot, ImplementsFlux):
                return phot

        return phot

    def site(self, site: str) -> Photometry:
        """Return photometry for a given site."""
        site_id = self.sites[site].id
        return self.phot.masked(self.phot.site == site_id)

    def absmag(self, filt: str, phot: Optional[HasMag] = None):
        if phot is None:
            phot = cast(ImplementsMag, self.phot)
        return phot.absmag(self.distance, self.bands[filt].ext)

    def make_bands(
        self,
        bands: Optional[BandsType] = None,
        band_order: Optional[list[str]] = None,
        ebv: Optional[float] = None,
        rv: float = 3.1
    ) -> Mapping[str, Filter]:
        """
        Creates a sorted dictionary of Filter.name: Filter 
        (from the bands in the photometry unless new bands are passed in).
        Any unknown filters are added to the end. If band_order is given, the
        order of the bands is set to that, otherwise default_order from
        supernova.filters.FilterSorter is used,
        which is hard-coded based on wave_eff.
        """
        usebands = self.phot.band.unique().tolist() if bands is None else bands
        use_ebv = self.sninfo.ebv if ebv is None else ebv
        new_bands = make_bands(usebands, band_order=band_order, ebv=max(use_ebv, 0), rv=rv)
        return new_bands

    def to_csv(self, basepath: Union[str, Path] = Path("../")) -> None:
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
        self.phases.loc["phase_zero"] = self.phase_zero
        if self.limits is not None:
            self.limits.calc_phases(self.phase_zero)

    def restframe(self) -> "SN":
        """
        Returns a copy of the SN object with the phases shifted to restframe.
        """
        sn = replace(self)
        sn.set_phases()
        sn.phot.phase = sn.phot.restframe_phases(self.redshift)
        if sn.limits is not None:
            sn.limits.phase = sn.limits.restframe_phases(self.redshift)
        return sn

    def calc_flux(self) -> None:
        """
        Calculates the fluxes of the photometry in each band
        and converts MagPhot to Phot with fluxes.
        """
        fac = PhotFactory(sn_phot=Phot())
        for band in self.bands:
            fluxphot = self.band(band, flux=True)
            if not isinstance(fluxphot, _HasFlux):
                raise TypeError("Photometry must have mag, mag_err to calculate flux.")
            fluxphot = fluxphot.masked(fluxphot.flux.notna().tolist())
            fac.concat_phot(fluxphot)
        self.phot = fac.sn_phot

    @classmethod
    def from_phot(
        cls,
        phot: Photometry | pd.DataFrame,
        name: str,
        redshift: float,
        phase_zero: Optional[float] = None,
        sub_only: bool = False,
        
        limits: Optional[Photometry] = None,
        sninfo: Optional[SNInfo] = None,
        sites: Optional[Sites] = None,
        **sninfo_kwargs: Any,
    ) -> "SN":

        if phase_zero is None:
            if sninfo is not None:
                if sninfo.phase_zero not in [None, -99]:
                    phase_zero = sninfo.phase_zero
                else:
                    phase_zero = phot.jd.min()
                    warnings.warn(
                        f"`phase_zero` not given. Setting to first photometry point: {phase_zero}."
                        " If this is not correct, set `phase_zero` manually by calling "
                        "`set_phases(<correct_phase_zero>)` on the SN object."
                    )

        if sninfo is not None:
            sninfo.redshift = redshift
            sninfo.name = name
            sninfo.phase_zero = phase_zero
        else:
            sninfo = SNInfo(
                redshift=redshift, name=name, sub_only=sub_only, phase_zero=phase_zero, **sninfo_kwargs
            )

        return sn_factory(phot=phot, sninfo=sninfo, limits=limits, sites=sites)


    def copy(self):
        return replace(self)


class SNSerializer:
    names = "phot,phases,sninfo,sites,bands,limits".split(",")

    def __init__(self, sn: SN):
        self.sn = sn

    def to_csv(self, basepath: Union[str, Path] = Path("../"), verbose: bool = True) -> None:
        printer: VerbosePrinter = VerbosePrinter(verbose=verbose)

        sn = self.sn
        names = SNSerializer.names

        basepath = Path(basepath).joinpath(Path(f"SNClass_{sn.name}/"))
        os.makedirs(basepath, exist_ok=True)

        sn.sninfo["name"] = sn.name
        sn.sninfo["sub_only"] = sn.sub_only
        field_vals = {
            f.name: getattr(sn, f.name) for f in fields(sn) if f.name in names
        }
 
        printer.print(f"saving {sn.name} with fields:")
        for name, _field in field_vals.items():
            printer.print(name, type(_field))
            if _field is None:
                continue
            if name == "bands":
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
        csvs = glob.glob(str(Path(dirpath) / "*.csv"))
        if len(csvs) < 2:
            raise FileNotFoundError(f"Only found: {csvs} in {dirpath}.")
        _sn_dict = {}
        for csv in csvs:
            df = pd.read_csv(csv, index_col=0).squeeze("columns")
            for name in names:
                if name in csv:
                    if name in ["bands", "sites"]:
                        df.replace(
                            {np.nan: None, "nan": None, "NAN": None, "NaN": None},
                            inplace=True,
                        )
                    _fields[name] = df

        # Dirty trick, unneeded if we would serialize to json, but then it's less readable
        # by astronomers.
        _fields["sninfo"] = force_numeric_sninfo(_fields["sninfo"])
        _fields["sites"] = Sites.from_df(_fields["sites"])
        if isinstance(_fields["bands"], pd.Series):
            _fields["bands"] = (
                make_bands(
                    bands=_fields["bands"].tolist(),
                    band_order=_fields["bands"].index.tolist(),
                    ebv=_fields["sninfo"].ebv,
                )
                if "bands" in _fields
                else {}
            )

        elif isinstance(_fields["bands"], pd.DataFrame):
            _bands_dict = _fields["bands"].to_dict("index")  # type: ignore
            _fields["bands"] = {
                v["name"]: Filter.from_dict(v) for v in _bands_dict.values()
            }
        else:
            _fields["bands"] = {}
        sninfo = SNInfo.from_dict(_fields["sninfo"].to_dict())
        if isinstance(sninfo.sub_only, str):
            sninfo.sub_only = sninfo.sub_only == "True"

        return sn_factory(
            phot=_fields["phot"],
            phases=_fields.get("phases", pd.Series({"phase_zero": sninfo.phase_zero})),
            sninfo=sninfo,
            sites=_fields["sites"],
            bands=_fields["bands"],
            limits=_fields["limits"] if "limits" in _fields else None,
        )

        # return SN(
        #     phot=_fields["phot"],
        #     phases=_fields.get("phases", pd.Series({"phase_zero": sninfo.phase_zero})),
        #     sninfo=sninfo,
        #     sites=_fields["sites"],
        #     name=sninfo.name,
        #     sub_only=sninfo.sub_only,
        #     bands=_fields["bands"],
        #     limits=_fields["limits"] if "limits" in _fields else None,
        # )

    @staticmethod
    def add_piscola_magsys(
        name: str, mag: float = 0, file: str = "ab_sys_zps.dat", verbose: bool = True
    ) -> None:
        """Add a new magnitude system to the Piscola mag system file."""
        printer = VerbosePrinter(verbose=verbose)
        file = str(PISCO_FILTER_PATH.parent / "mag_sys" / file)
        filt_names, _ = np.loadtxt(file, dtype=str).T
        if name in filt_names:
            warnings.warn(f"Filter {name} already in {file}")
            return
        with open(file, "a") as f:
            f.write("\n")
            f.write(f"{name} {mag}")

        printer.print(f"Added {name} to {file}")

    def save_piscola_filters(self, sites: bool = True):
        for band in self.sn.bands.values():
            if not isinstance(band.svo, SVOFilter):
                warnings.warn(
                    f"Filter {band.name} does not have an SVO Filter defined."
                    f" Skipping."
                )
                continue
            if sites:
                site_names = self.sn.sites.site_names
            else:
                site_names = ["all"]
            for site in site_names:
                path = PISCO_FILTER_PATH / site
                path.mkdir(exist_ok=True)
                path = path / f"{site}_{band.name}.dat"
                if not path.is_file():
                    band.svo.write_filter(path)
                    self.add_piscola_magsys(name=f"{site}_{band.name}")

    def make_piscola_file(self, base_directory: Path, sites: bool = False) -> Path:
        """
        Make a piscola file for the SN.
        """
        self.save_piscola_filters(sites=sites)
        sn = self.sn
        sn.calc_flux()
        if not isinstance(sn.phot, _HasFlux):
            raise TypeError("Photometry must have fluxes to make a piscola file.")
        line0 = "name z ra dec\n"
        line1 = f"{sn.name} {sn.sninfo.redshift} {sn.sninfo.ra} {sn.sninfo.dec}"
        line2 = "time flux flux_err zp band mag_sys"

        site_keys = "all"
        if sites:
            site_keys = sn.phot.site.apply(lambda l: sn.sites[l].name)
        zps = sn.phot.band.apply(lambda l: np.log10(sn.bands[l].zp.value) * 2.5)

        df = pd.DataFrame(columns=line2.split())
        df["time"] = sn.phot.jd
        df["flux"] = sn.phot.flux
        df["flux_err"] = sn.phot.flux_err
        df["zp"] = zps
        df["band"] = site_keys + "_" + sn.phot.band
        df["mag_sys"] = "AB"  # Dummy, not needed for fit

        path = base_directory / f"{sn.name}.dat"
        with open(path, "w") as piscofile:
            piscofile.write(line0)
            piscofile.write(line1)
            piscofile.write("\n")
            piscofile.write(line2)
            piscofile.write("\n")
            piscofile.write(df.to_string(header=False, index=False))
        return path


def cosmo_from_name(name: str) -> Cosmology:
    if name in realizations.__all__:
        return getattr(realizations, name)
    warnings.warn(f"Could not find cosmology {name}. Using WMAP5 instead.")
    return WMAP5


def get_extinction_irsa(coords: SkyCoord) -> float:
    from astroquery.irsa_dust import IrsaDust

    table = IrsaDust.get_query_table(coords, section="ebv")
    ebv = table["ext SandF ref"][0] # type: ignore
    return cast(float, ebv)


def force_numeric_sninfo(sninfo: pd.Series) -> pd.Series:
    _sninfo = pd.to_numeric(sninfo, errors="coerce")
    _sninfo = _sninfo.mask(_sninfo.isna(), sninfo)
    return _sninfo


#
# @dataclass
# class AbstractBasePhot(ABC):
#     """Abstract Photometry dataclass"""
#     jd: pd.Series
#     band: pd.Series
#     filter: pd.Series
#     phase: pd.Series
#     sub: pd.Series
#     site: pd.Series
#
#     # restframe: pd.Series
#
#     def __post_init__(self):
#         if self.__class__ == AbstractBasePhot:
#             raise TypeError("Cannot instantiate abstract class.")
#
#     @abstractmethod
#     def _fill_missing(self) -> None:
#         pass
#
#     @abstractmethod
#     def calc_phases(self, phase_zero: Number) -> None:
#         pass
#
#     @abstractmethod
#     def restframe_phases(self, redshift: Number) -> pd.Series:
#         pass
#
#     @abstractmethod
#     def masked(self, cond: list) -> 'AbstractPhotometry':
#         pass
#
#     @abstractmethod
#     def as_dataframe(self) -> pd.DataFrame:
#         pass
#
#     @classmethod
#     @abstractmethod
#     def from_dict(cls, d: dict) -> 'AbstractPhotometry':
#         pass
#
#     @abstractmethod
#     def __len__(self):
#         pass
#
#
# @dataclass
# class AbstractMagPhot(ABC):
#     mag: pd.Series
#     mag_err: pd.Series
#
#     def __post_init__(self):
#         if self.__class__ == AbstractMagPhot:
#             raise TypeError("Cannot instantiate abstract class.")
#
#     @abstractmethod
#     def absmag(self, dm: Number, ext: Number) -> pd.Series:
#         pass
#
#
# @dataclass
# class AbstractFluxPhot(ABC):
#     flux: pd.Series
#     flux_err: pd.Series
#
#     def __post_init__(self):
#         if self.__class__ == AbstractFluxPhot:
#             raise TypeError("Cannot instantiate abstract class.")
#
#
# @dataclass
# class AbstractPhot(ABC):
#     flux: pd.Series
#     flux_err: pd.Series
#     mag: pd.Series
#     mag_err: pd.Series
#
#
# @dataclass
# class AbstractLumPhot(ABC):
#     lum: pd.Series
#     lum_err: pd.Series
#
#
# @dataclass
# class AbstractBBLumPhot(ABC):
#     radius: pd.Series
#     radius_err: pd.Series
#     temp: pd.Series
#     temp_err: pd.Series
#
#
# @dataclass
# class AbstractPhotometry(AbstractMagPhot, AbstractFluxPhot,
#                          AbstractLumPhot, AbstractBBLumPhot, AbstractBasePhot, ABC):
#     pass
#
#
# @dataclass
# class BasePhot(AbstractBasePhot):
#     jd: pd.Series = field(default=pd.Series(dtype=float))
#     band: pd.Series = field(default=pd.Series(dtype=str))
#     phase: pd.Series = field(default=pd.Series(dtype=float))
#     sub: pd.Series = field(default=pd.Series(dtype=bool))
#     site: pd.Series = field(default=pd.Series(dtype=int))
#     # only for backwards compatibility do not access directly as it overrides builtin filter.
#     filter: pd.Series = field(default=pd.Series(dtype=str))
#
#     # restframe: pd.Series = field(default=pd.Series(dtype=float))
#
#     def __post_init__(self):
#         if 0 < len(self.filter) == len(self):
#             warnings.warn("Phot: `filter` is deprecated, use `band` instead", DeprecationWarning)
#             self.band = self.filter
#
#         if len(self.band) != len(self):
#             raise ValueError("Either band or filter must be given the same length as jd."
#                              f"Got band: {len(self.band)}, "
#                              f"filter: {len(self.filter)}, jd: {len(self)}")
#         self.filter = self.band
#         # @TODO: remove this try/except after tests.
#         try:
#             self._fill_missing()
#         except Exception as e:
#             warnings.warn(f"Could not fill missing values of {self.__class__}: {e}")
#
#     def _fill_missing(self) -> None:
#         if len(self) == 0:
#             return
#         for _field in fields(self):
#
#             if len(getattr(self, _field.name)) == 0:
#                 dtype, default = get_field_dtype_default(_field)
#                 setattr(self, _field.name,
#                         pd.Series([default] * len(self), dtype=dtype, name=_field.name))
#
#     def calc_phases(self, phase_zero):
#         self.phase = self.jd - phase_zero
#
#     def restframe_phases(self, redshift):
#         if self.phase is None:
#             raise AttributeError("self.phase must not be None, "
#                                  "calculate it first using calc_phases with a phase_zero")
#         return self.phase / (1 + redshift)
#
#     @classmethod
#     def from_dict(cls, d: dict):
#         if cls == BasePhot:
#             raise TypeError("Cannot instantiate Base class. Use a subclass.")
#         return cls(**{k: v for k, v in d.items() if k in [f.name for f in fields(cls)]})
#
#     def masked(self, cond: list):
#         d2 = {name: value[cond] for name, value in asdict(self).items()}
#         return self.from_dict(d2)
#
#     def as_dataframe(self) -> pd.DataFrame:
#         return pd.DataFrame(asdict(self))
#
#     def __len__(self):
#         return len(self.jd)
#
#
# @dataclass
# class MagPhot(BasePhot, AbstractMagPhot):
#     mag: pd.Series = field(default=pd.Series(dtype=float))
#     mag_err: pd.Series = field(default=pd.Series(dtype=float))
#
#     def absmag(self, dm: Number, ext: Number) -> pd.Series:
#         return self.mag - dm - ext
#
#
# @dataclass
# class FluxPhot(BasePhot, AbstractFluxPhot):
#     flux: pd.Series = field(default=pd.Series(dtype=float))
#     flux_err: pd.Series = field(default=pd.Series(dtype=float))
#
#     @staticmethod
#     def mag_to_flux(mag: ArrayLike, filt: Filter) -> ArrayLike:
#         return 10 ** (mag / -2.5) * filt.zp
#
#     @staticmethod
#     def mag_to_flux_err(mag_err: ArrayLike, flux: ArrayLike) -> ArrayLike:
#         return 2.303 * mag_err * flux
#
#
# @dataclass
# class Phot(FluxPhot, MagPhot, AbstractPhot):
#     mag: pd.Series = field(default=pd.Series(dtype=float))
#     mag_err: pd.Series = field(default=pd.Series(dtype=float))
#     flux: pd.Series = field(default=pd.Series(dtype=float))
#     flux_err: pd.Series = field(default=pd.Series(dtype=float))
#
#     @classmethod
#     def from_magphot(cls, magphot: AbstractMagPhot, band: Filter) -> 'Phot':
#         flux = cls.mag_to_flux(magphot.mag, band)
#         flux_err = cls.mag_to_flux_err(magphot.mag_err, flux)
#         return cls.from_dict(asdict(magphot) | {"flux": flux, "flux_err": flux_err})
#
#
# @dataclass
# class LumPhot(Phot, AbstractLumPhot):
#     lum: pd.Series = field(default=pd.Series(dtype=float))
#     lum_err: pd.Series = field(default=pd.Series(dtype=float))
#
#
# @dataclass
# class BBLumPhot(LumPhot, AbstractBBLumPhot):
#     radius: pd.Series = field(default=pd.Series(dtype=float))
#     radius_err: pd.Series = field(default=pd.Series(dtype=float))
#     temp: pd.Series = field(default=pd.Series(dtype=float))
#     temp_err: pd.Series = field(default=pd.Series(dtype=float))
#
#
# Photometry = FluxPhot | MagPhot | Phot | LumPhot | BBLumPhot
