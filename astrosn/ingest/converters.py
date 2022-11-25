from dataclasses import dataclass, field
from typing import Optional, Protocol, ClassVar
import numpy as np
import pandas as pd


# noinspection PyUnusedLocal
@dataclass(frozen=True)
class Converter(Protocol):
    glob_str: ClassVar[str]
    jd_col: str
    mag_col: str
    mag_err_col: str
    band_col: str
    sub_col: str
    site_col: str
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame) -> None:
        ...

    def jd_convert(self) -> None:
        ...

    def mag_convert(self) -> None:
        ...

    def mag_err_convert(self) -> None:
        ...

    def band_convert(self) -> None:
        ...

    def sub_convert(self) -> None:
        ...

    def site_convert(self) -> None:
        ...

    def convert(self) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class BaseConverter(Converter):
    glob_str: ClassVar[str] = "*"
    jd_col: str = "time"
    mag_col: str = "mag"
    mag_err_col: str = "mag_err"
    band_col: str = "filter"
    sub_col: str = "sub"
    site_col: str = "site"
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def jd_convert(self) -> None:
        self.df["jd"] = self.df[self.jd_col]

    def mag_convert(self) -> None:
        self.df["mag"] = self.df[self.mag_col]

    def mag_err_convert(self) -> None:
        self.df["mag_err"] = self.df[self.mag_err_col]

    def band_convert(self) -> None:
        self.df["band"] = self.df[self.band_col]

    def sub_convert(self) -> None:
        self.df["sub"] = self.df[self.sub_col]

    def site_convert(self) -> None:
        self.df["site"] = self.df[self.site_col]

    def convert(self) -> pd.DataFrame:
        self.jd_convert()
        self.mag_convert()
        self.mag_err_convert()
        self.band_convert()
        self.sub_convert()
        self.site_convert()
        return self.df


@dataclass(frozen=True)
class GenericCSVConverter(BaseConverter):
    glob_str: ClassVar[str] = "*.csv"

    def sub_convert(self) -> None:
        self.df["sub"] = True

    def site_convert(self) -> None:
        self.df["site"] = np.random.randint(10, 100)


@dataclass(frozen=True)
class GenericECSVConverter(BaseConverter):
    glob_str: ClassVar[str] = "*.ecsv"

    def sub_convert(self) -> None:
        self.df["sub"] = True

    def site_convert(self) -> None:
        self.df["site"] = np.random.randint(10, 100)


@dataclass(frozen=True)
class GenericJSONConverter(BaseConverter):
    glob_str: ClassVar[str] = "*.json"

    def sub_convert(self) -> None:
        self.df["sub"] = True

    def site_convert(self) -> None:
        self.df["site"] = np.random.randint(10, 100)
