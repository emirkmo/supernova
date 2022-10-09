from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields
from typing import (
    Any,
    Generic,
    Sequence,
    TypeVar,
    Protocol,
    Mapping,
    Iterable,
    Hashable,
    overload,
    runtime_checkable,
)
import warnings

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
#from pandera.typing import Series, DataFrame
from astropy.units import Quantity
import astropy.units as u

from .filters import Filter
from .utils import Number, get_field_dtype_default, _is_quantity
T = TypeVar("T")
GenericType = TypeVar("GenericType", covariant=True)
NumType = TypeVar('NumType', float, int, Quantity)
class Series(Series, Generic[GenericType]):
    ...


ArrayLike = TypeVar("ArrayLike", np.ndarray[Any, Any], Sequence[Number], Series)
N = TypeVar("N", Quantity, float, int, Number)

SeriesSite = Series[str] | Series[int]
DFDict = Mapping[str, ArrayLike] | Mapping[Hashable, Any]


@dataclass
class HasPhot(Protocol):
    jd: Series[float]
    band: Series[str]
    sub: Series[bool]
    site: SeriesSite
    phase: Series[float]


@dataclass
@runtime_checkable
class HasMag(Protocol):
    mag: Series[float]
    mag_err: Series[float]

    def absmag(self, dm: N, ext: N) -> Series[float]:
        ...


@dataclass
@runtime_checkable
class HasFlux(Protocol):
    flux: Series[float]
    flux_err: Series[float]


@dataclass
@runtime_checkable
class _HasFlux(HasFlux, Protocol):
    @staticmethod
    def mag_to_flux(mag: Series[float], filt: Filter) -> Series[float]:
        ...

    @staticmethod
    def mag_to_flux_err(mag_err: Series[float], flux: Series[float]) -> Series[float]:
        ...


@dataclass
@runtime_checkable
class HasMagFlux(HasMag, _HasFlux, Protocol):
    @classmethod
    def from_magphot(cls, magphot: HasMag, band: Filter) -> "HasMagFlux":
        ...


@dataclass
@runtime_checkable
class HasLuminosity(Protocol):
    lum: Series[float]
    lum_err: Series[float]


@dataclass
@runtime_checkable
class HasLumFlux(HasLuminosity, HasFlux, Protocol):
    ...


@dataclass
@runtime_checkable
class HasBlackBody(HasLuminosity, Protocol):
    radius: Series[float]
    radius_err: Series[float]
    temp: Series[float]
    temp_err: Series[float]


@dataclass
@runtime_checkable
class HasBlackBodyFlux(HasBlackBody, HasFlux, Protocol):
    ...


@dataclass
class PhotMethods(Protocol):
    def calc_phases(self, phase_zero: N) -> None:
        ...

    def restframe_phases(self, redshift: N) -> Series[float]:
        ...

    def as_dataframe(self) -> DataFrame:
        ...

    def __len__(self) -> int:
        ...

    @classmethod
    def from_dict(cls, d: DFDict[Any]) -> "Photometry":
        ...

    def masked(self, cond: Iterable[bool]) -> "Photometry":
        ...

    def sorted(self) -> "Photometry":
        ...

@runtime_checkable
class ImplementsMag(HasPhot, HasMag, PhotMethods, Protocol):
    pass

@runtime_checkable
class ImplementsFlux(HasPhot, _HasFlux, PhotMethods, Protocol):
    pass

@runtime_checkable
class ImplementsMagFlux(HasPhot, HasMagFlux, PhotMethods, Protocol):
    pass

@runtime_checkable
class ImplementsLuminosity(HasPhot, HasLuminosity, PhotMethods, Protocol):
    pass

@runtime_checkable
class ImplementsBlackBody(HasPhot, HasBlackBody, PhotMethods, Protocol):
    pass

Photometry = (
    ImplementsMag
    | ImplementsFlux
    | ImplementsMagFlux
    | ImplementsLuminosity
    | ImplementsBlackBody
)

FluxPhotometry = ImplementsFlux | ImplementsMagFlux 
MagPhotometry = ImplementsMag | ImplementsMagFlux


P = TypeVar('P', bound=Photometry)


@dataclass
class BasePhotMixin(PhotMethods, Generic[P]):
    jd: Series[float] = Series(dtype=float)
    band: Series[str] = Series(dtype=str)
    phase: Series[float] = Series(dtype=float)
    sub: Series[bool] = Series(dtype=bool)
    site: Series[int] | Series[str] = Series(dtype=int)
    # only for backwards compatibility do not access directly as it overrides builtin filter.
    filter: Series[str] = Series(dtype=str)

    def __post_init__(self) -> None:
        if 0 < len(self.filter) == len(self):
            warnings.warn(
                "Phot: `filter` is deprecated, use `band` instead", DeprecationWarning
            )
            self.band = self.filter

        if len(self.band) != len(self):
            raise ValueError(
                "Either band or filter must be given the same length as jd."
                f"Got band: {len(self.band)}, "
                f"filter: {len(self.filter)}, jd: {len(self)}"
            )
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
                setattr(
                    self,
                    _field.name,
                    Series([default] * len(self), dtype=dtype, name=_field.name),
                )

    @classmethod
    def from_dict(cls, d: DFDict[Any]) -> P:
        if cls == BasePhotMixin:
            raise TypeError("Cannot instantiate Base class. Use a subclass.")
        return cls(**{k: v for k, v in d.items() if k in [f.name for f in fields(cls)]})  # type: ignore

    def masked(self, cond: Iterable[bool]) -> P:
        d2 = {name: value[cond] for name, value in asdict(self).items()}
        return self.from_dict(d2)

    def calc_phases(self, phase_zero: float | int) -> None:
        self.phase = self.jd - phase_zero

    def restframe_phases(self, redshift: float | int) -> Series[float]:
        if self.phase is None:
            raise AttributeError(
                "self.phase must not be None, "
                "calculate it first using calc_phases with a phase_zero"
            )
        return self.phase / (1 + redshift)

    def as_dataframe(self) -> DataFrame:
        return DataFrame(asdict(self))

    def __len__(self) -> int:
        return len(self.jd)

    def sorted(self) -> "Photometry":
        df = self.as_dataframe().sort_values(by="jd")
        df.reset_index(drop=True, inplace=True)
        return self.from_dict(df.to_dict("series"))



class _MagMethodsMixin:
    mag: Series[float]

    def absmag(self, dm: N, ext: N) -> Series[N]:
        return self.mag - dm - ext

@dataclass
class MagMixin:
    mag: Series[float] = Series(dtype=float)
    mag_err: Series[float] = Series(dtype=float)

@dataclass
class MagPhot(BasePhotMixin[ImplementsMag], MagMixin, _MagMethodsMixin, ImplementsMag):
    pass


class _FluxMethodsMixin:
    
    @staticmethod
    def mag_to_flux(mag: Series[float], filt: Filter) -> Series[float]:
        return 10 ** (mag / -2.5) * filt.zp

    @staticmethod
    def mag_to_flux_err(mag_err: Series[float], flux: Series[float]) -> Series[float]:
        return 2.303 * mag_err * flux

@dataclass
class FluxMixin:
    flux: Series[float] = Series(dtype=float)
    flux_err: Series[float] = Series(dtype=float)

@dataclass
class FluxPhot(BasePhotMixin[ImplementsFlux], FluxMixin, _FluxMethodsMixin, ImplementsFlux):
    pass


@dataclass
class Phot(BasePhotMixin[ImplementsMagFlux], MagMixin, FluxMixin, _MagMethodsMixin, _FluxMethodsMixin, ImplementsMagFlux, ImplementsFlux, ImplementsMag):


    @classmethod
    def from_magphot(cls, magphot: HasMag, band: Filter) -> ImplementsMagFlux:
        flux = cls.mag_to_flux(magphot.mag, band)
        flux_err = cls.mag_to_flux_err(magphot.mag_err, flux)
        magdict = asdict(magphot)
        fluxdict = {"flux": flux, "flux_err": flux_err}
        return cls.from_dict(magdict | fluxdict)

@dataclass
class LumMixin:
    lum: Series[float] = Series(dtype=float)
    lum_err: Series[float] = Series(dtype=float)

@dataclass
class LumPhot(BasePhotMixin[ImplementsLuminosity], LumMixin, FluxMixin, ImplementsLuminosity, HasLumFlux):
    pass

@dataclass
class BBLumPhot(BasePhotMixin[ImplementsBlackBody], LumMixin, FluxMixin, ImplementsBlackBody, HasBlackBodyFlux):
    radius: Series[float] = Series(dtype=float)
    radius_err: Series[float] = Series(dtype=float)
    temp: Series[float] = Series(dtype=float)
    temp_err: Series[float] = Series(dtype=float)


class PhotFactory:
    """
    Factory class to create Photometry objects.
    """

    def __init__(self, sn_phot: Photometry) -> None:
        self.sn_phot = sn_phot

    @staticmethod
    def from_df(df: DataFrame) -> Photometry:
        #d = cast(Mapping[str, Series], df.to_dict("series"))
        d = df.to_dict("series")
        mag = "mag" in d
        flux = "flux" in d
        lum = "lum" in d
        bb = "radius" in d and "temp" in d
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
        df = pd.concat(
            [sn_phot.as_dataframe(), new_phot.as_dataframe()], ignore_index=True
        )
        df = df[sn_phot.as_dataframe().columns.drop("filter")]
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
        d = self.sn_phot.as_dataframe().to_dict("series")
        self.sn_phot = new_phot_type.from_dict(d)
        return self.sn_phot
