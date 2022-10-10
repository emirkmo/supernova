import enum

from pathlib import Path
from types import ModuleType
from typing import Any, Sequence, TypeGuard, cast, Protocol, overload
from dataclasses import Field
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units.quantity import Quantity
from astropy.units.core import Unit, dimensionless_unscaled
from numpy.typing import DTypeLike, NDArray
from pandas import Series
import numpy as np
from urllib.errors import HTTPError
Number = int | float | Quantity
PathType = Path | str


class SupportsArray(Protocol):
    @overload
    def __array__(self, __dtype: DTypeLike = ...) -> NDArray:
        ...
    @overload
    def __array__(self, dtype: DTypeLike = ...) -> NDArray:
        ...

    @overload
    def __getitem__(self, __key: int) -> Number:
        ...

    @overload
    def __getitem__(self, __key: slice) -> 'SupportsArray':
        ...


QuantityArrayType = Quantity | SupportsArray | np.ndarray[Number, Any]


def _is_quantity_arr(val: Any) -> TypeGuard[Quantity]:
    return isinstance(val[0], Quantity)

def _is_quantity(val: Any) -> TypeGuard[Quantity]:
    return isinstance(val, Quantity)

def quantity_array(array: QuantityArrayType) -> Quantity:
    if isinstance(array, Quantity):
            return array
    if _is_quantity_arr(array):
        unit: Unit = array[0].unit
        return Quantity([val.value for val in array], unit)
    
    return np.array(array) * dimensionless_unscaled


class StrEnum(str, enum.Enum):
    """Enum with string values."""
    def __str__(self) -> str:
        return str(self.value)


def get_field_dtype_default(_field: Field) -> tuple[DTypeLike, float | int | bool | str]:
    dtype = float
    default = np.nan
    if isinstance(_field.default, Series):
        if _field.default.dtype == object:
            dtype = str
            default = ""
        if _field.default.dtype == bool:
            dtype = bool
            default = True
        if _field.default.dtype == int:
            dtype = int
            default = -99
    return dtype, default


class VerbosePrinter:
    """A class to print messages if verbose is True."""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.print(*args, **kwargs)

FloatArr = NDArray[np.float64]


def tendrils_api() -> ModuleType:
    """Use tendrils to access FLOWS API if
    optional dependency tendrils is installed.
    """
    try:
        from tendrils import api
    except ImportError:
        raise ImportError("tendrils is not installed. To connect to FLOWS database,"
                          "install tendrils with `pip install tendrils`.")

    return api    

def get_flows_sninfo(snname: str) -> tuple[SkyCoord, float]:
    """Get SN info from FLOWS database."""
    api = tendrils_api()
    if snname.startswith("SN"):
        snname = snname[2:]
    sninfo = api.get_target(snname)
    coords = SkyCoord(ra=sninfo["ra"], dec=sninfo["dec"], unit="deg", frame="icrs")
    redshift = sninfo["redshift"]
    return coords, redshift


def is_http_error(err: Exception) -> TypeGuard[HTTPError]:
    """Check if an exception is an HTTPError."""
    return hasattr(err, "response")
