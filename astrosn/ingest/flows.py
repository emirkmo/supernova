from typing import Optional
from .converters import BaseConverter, Converter
from .readers import read_astropy_table, times_as_jd, PathType
from .collators import Collator
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class FlowsConverter(BaseConverter, Converter):
    glob_str: str = 'flows_*lc.ecsv'
    jd_col: str = 'time'
    mag_col: str = 'mag_raw'
    mag_err_col: str = 'mag_raw_error'
    band_col: str = 'photfilter'
    sub_col: str = 'None'
    site_col: Optional[str] = 'site'

    def mag_convert(self) -> None:
        self.df['mag'] = self.df['mag_sub'].fillna(self.df['mag_raw'])

    def mag_err_convert(self) -> None:
        self.df['mag_err'] = self.df['mag_sub_error'].fillna(self.df['mag_raw_error'])

    def sub_convert(self) -> None:
        self.df['sub'] = self.df["mag_sub"].apply(pd.notna)

    def band_convert(self) -> None:
        self.df['band'] = self.df[self.band_col].str[:1]


def read_flows_lc(path: PathType) -> pd.DataFrame:
    at = read_astropy_table(path)
    at = times_as_jd(at)
    return at.to_pandas()


collate_flows = Collator(converter=FlowsConverter, reader=read_flows_lc)
