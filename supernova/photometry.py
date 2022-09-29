from dataclasses import dataclass, asdict, field, fields
from typing import TypeVar, Protocol, Mapping, Iterable, Generic, Type, overload, Collection

import warnings
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from astropy.units import Quantity

from .filters import Filter
from .utils import Number, get_field_dtype_default
from .sites import SiteType

ArrayLike = TypeVar("ArrayLike", np.ndarray, Number, Series)
N = TypeVar("N", Quantity, float, int)


class HasPhot(Protocol):
    jd: Collection[N]
    band: Collection[str]
    sub: Collection[bool]
    site: Collection[SiteType]
    phase: Collection[N]


class HasMag(Protocol):
    mag: Collection[N]
    mag_err: Collection[N]

    def absmag(self, dm: N, ext: N) -> Collection[N]:
        ...


class _HasFlux(Protocol):
    flux: Collection[N]
    flux_err: Collection[N]
    

class HasFlux(_HasFlux, Protocol):
    
    @staticmethod
    def mag_to_flux(mag: Collection[N], filt: Filter) -> Collection[N]:
        ...

    @staticmethod
    def mag_to_flux_err(mag_err: Collection[N], flux: Collection[N]) -> Collection[N]:
        ...


class HasMagFlux(HasMag, HasFlux, Protocol):

    @classmethod
    def from_magphot(cls, magphot: HasMag, band: Filter) -> "HasMagFlux":
        ...


class HasLuminosity(Protocol):
    lum: Collection[N]
    lum_err: Collection[N]


class HasLumFlux(HasLuminosity, _HasFlux, Protocol):
    ...


class HasBlackBody(HasLuminosity, Protocol):
    radius: Collection[N]
    radius_err: Collection[N]
    temp: Collection[N]
    temp_err: Collection[N]


class HasBlackBodyFlux(HasBlackBody, _HasFlux, Protocol):
    ...


phot_mixins = HasMag, HasFlux, HasMagFlux, HasLuminosity, HasBlackBody, HasLumFlux, HasBlackBodyFlux
P = TypeVar("P", *phot_mixins)


# class ImplementsPhot(HasPhot, Generic[P]):
# 
#     def calc_phases(self, phase_zero: N) -> None:
#         ...
# 
#     def restframe_phases(self, redshift: N) -> Series[N]:
#         ...
# 
#     @classmethod
#     def from_dict(cls, data: Mapping[str, ArrayLike]) -> Type[P]:
#         ...
# 
#     def masked(self, cond: Iterable[bool]) -> Type[P]:
#         ...
#     
#     def as_dataframe(self) -> DataFrame:
#         ...
# 
#     def __len__(self) -> int:
#         ...
# 
#     def _fill_missing(self) -> None:
#         ...


@dataclass
class BasePhot(HasPhot, Generic[P]):
    jd: Series[Number] = field(default=Series(dtype=float))
    band: Series[str] = field(default=Series(dtype=str))
    phase: Series[Number] = field(default=Series(dtype=float))
    sub: Series[bool] = field(default=Series(dtype=bool))
    site: Series[SiteType] = field(default=Series(dtype=int))
    # only for backwards compatibility do not access directly as it overrides builtin filter.
    filter: Series[str] = field(default=Series(dtype=str))

    def __post_init__(self) -> None:
        if 0 < len(self.filter) == len(self):
            warnings.warn("Phot: `filter` is deprecated, use `band` instead", DeprecationWarning)
            self.band = self.filter

        if len(self.band) != len(self):
            raise ValueError("Either band or filter must be given the same length as jd."
                             f"Got band: {len(self.band)}, "
                             f"filter: {len(self.filter)}, jd: {len(self)}")
        self.filter = self.band
        # @TODO: remove this try/except after tests.
        try:
            self._fill_missing()
        except Exception as e:
            warnings.warn(f"Could not fill missing values of {self.__class__}: {e}")

    def _fill_missing(self) -> None:
        if len(self) == 0:
            return
        for _field in fields(self):

            if len(getattr(self, _field.name)) == 0:
                dtype, default = get_field_dtype_default(_field)
                setattr(self, _field.name,
                        Series([default] * len(self), dtype=dtype, name=_field.name))

    def calc_phases(self, phase_zero: Number) -> None:
        self.phase = self.jd - phase_zero

    def restframe_phases(self, redshift: N) -> Series[N]:
        if self.phase is None:
            raise AttributeError("self.phase must not be None, "
                                 "calculate it first using calc_phases with a phase_zero")
        return self.phase / (1 + redshift)

    def _get_self_class(self) -> Type[P]:
        return type(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, ArrayLike]) -> Type[P]:
        if cls == BasePhot:
            raise TypeError("Cannot instantiate Base class. Use a subclass.")
        self_type = cls._get_self_class(cls())
        return self_type(**{k: v for k, v in d.items() if k in [f.name for f in fields(self_type)]})

    def masked(self, cond: Iterable[bool]) -> Type[P]:
        d2 = {name: value[cond] for name, value in asdict(self).items()}
        return self.from_dict(d2)

    def as_dataframe(self) -> DataFrame:
        return DataFrame(asdict(self))

    def __len__(self) -> int:
        return len(self.jd)


@dataclass
class Photometry(BasePhot[P], P):

    @overload
    @classmethod
    def from_dict(cls, d: Mapping[str, ArrayLike]) -> 'Photometry':
        ...

    @classmethod
    def from_dict(cls, d: Mapping[str, ArrayLike]) -> 'Photometry':
        return super().from_dict(d)


@dataclass
class MagPhot(Photometry[HasMag]):
    mag: Series[Number] = field(default=Series(dtype=float))
    mag_err: Series[Number] = field(default=Series(dtype=float))

    def absmag(self, dm: N, ext: N) -> Series[Number]:
        return self.mag - dm - ext


@dataclass
class FluxPhot(Photometry[HasFlux]):
    flux: Series[Number] = field(default=Series(dtype=float))
    flux_err: Series[Number] = field(default=Series(dtype=float))

    @staticmethod
    def mag_to_flux(mag: Series[N], filt: Filter) -> Series[N]:
        return 10 ** (mag / -2.5) * filt.zp

    @staticmethod
    def mag_to_flux_err(mag_err: Series[N], flux: Series[N]) -> Series[N]:
        return 2.303 * mag_err * flux
    

@dataclass
class Phot(Photometry[HasMagFlux]):
    mag: Series[Number] = field(default=Series(dtype=float))
    mag_err: Series[Number] = field(default=Series(dtype=float))
    flux: Series[Number] = field(default=Series(dtype=float))
    flux_err: Series[Number] = field(default=Series(dtype=float))

    @classmethod
    def from_magphot(cls, magphot: MagPhot, band: Filter) -> HasMagFlux:
        flux = cls.mag_to_flux(magphot.mag, band)
        flux_err = cls.mag_to_flux_err(magphot.mag_err, flux)
        magdict = asdict(magphot)
        fluxdict = {"flux": flux, "flux_err": flux_err}
        return cls.from_dict(magdict | fluxdict)


@dataclass
class LumPhot(Photometry[HasLumFlux]):
    lum: Series[Number] = field(default=Series(dtype=float))
    lum_err: Series[Number] = field(default=Series(dtype=float))


@dataclass
class BBLumPhot(Photometry[HasBlackBodyFlux]):
    radius: Series[Number] = field(default=Series(dtype=float))
    radius_err: Series[Number] = field(default=Series(dtype=float))
    temp: Series[Number] = field(default=Series(dtype=float))
    temp_err: Series[Number] = field(default=Series(dtype=float))


# Photometry = MagPhot | FluxPhot | Phot | LumPhot | BBLumPhot
# PhotType = Type[Photometry]
# PhotTypes = TypeVar("PhotTypes", MagPhot, FluxPhot, Phot, LumPhot, BBLumPhot)


class PhotFactory:
    """
    Factory class to create Photometry objects.
    """
    def __init__(self, sn_phot: Photometry) -> None:
        self.sn_phot = sn_phot

    @staticmethod
    def from_df(df: DataFrame) -> Photometry:
        d = df.to_dict('series')
        mag = 'mag' in d
        flux = 'flux' in d
        lum = 'lum' in d
        bb = 'radius' in d and 'temp' in d
        if bb:
            return BBLumPhot.from_dict(d)
        if lum:
            return LumPhot.from_dict(d)
        if mag and flux:
            return Phot.from_dict(d)
        elif mag:
            return MagPhot.from_dict(d)
        elif flux:
            return FluxPhot.from_dict(d)
        raise ValueError("df must contain either 'mag' or 'flux' columns")

    @classmethod
    def add_phot(cls, sn_phot: Photometry, new_phot: Photometry) -> Photometry:
        """
        Add photometry to a Supernova.
        """
        df = pd.concat([sn_phot.as_dataframe(), new_phot.as_dataframe()], ignore_index=True)
        df = df[sn_phot.as_dataframe().columns.drop('filter')]
        return cls.from_df(df)

    def concat_phot(self, new_phot: Photometry) -> Photometry:
        """
        Add photometry to a Supernova.
        """
        self.sn_phot = self.add_phot(self.sn_phot, new_phot)
        return self.sn_phot

    def extend_phot(self, new_phot_type: type[Photometry]) -> Photometry:
        """
        Extend the photometry of a Supernova to a new type.
        """
        d = self.sn_phot.as_dataframe().to_dict('series')
        self.sn_phot = new_phot_type.from_dict(d)
        return self.sn_phot
