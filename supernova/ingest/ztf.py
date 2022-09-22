import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from astropy.time import Time

from supernova import SN
from supernova.ingest.utils import (
    add_phot, update_sn, verify_columns)
from supernova.supernova import Photometry, PhotFactory
from supernova.utils import StrEnum
from .collators import Collator
from .converters import BaseConverter, Converter
from .readers import PathType, read_astropy_table, read_pandas_csv


@dataclass(frozen=True)
class ZTFFlowsConverter(BaseConverter, Converter):
    glob_str: str = '-ztf.ecsv'
    jd_col: str = 'time'
    mag_col: str = 'mag'
    mag_err_col: str = 'mag_err'
    band_col: Optional[str] = 'photfilter'
    sub_col: Optional[str] = None
    site_col: Optional[str] = None

    def site_convert(self) -> None:
        self.df['site'] = 0  # refactor to "ZTF" and for phot to use Site class.

    def sub_convert(self) -> None:
        self.df['sub'] = True

    def band_convert(self) -> None:
        self.df['band'] = self.df[self.band_col].str[:1]


def read_ztf_flows(path: PathType) -> pd.DataFrame:
    """
    Read a ZTF flows file and return a DataFrame.
    """
    at = read_astropy_table(path, fmt='ascii.ecsv')
    at[ZTFFlowsConverter.jd_col] = Time(at[ZTFFlowsConverter.jd_col], format='mjd')
    at[ZTFFlowsConverter.jd_col] = at[ZTFFlowsConverter.jd_col] + 2400000.5  # Convert MJD to JD
    return at.to_pandas()


collate_ztf_flows = Collator(ZTFFlowsConverter, read_ztf_flows)


@dataclass(frozen=True)
class ZTFPhotConverter(BaseConverter, Converter):
    glob_str: str = 'SN*_detections.csv'
    jd_col: str = 'jd'
    mag_col: str = 'mag'
    mag_err_col: str = 'mag_err'
    band_col: Optional[str] = 'filter'
    sub_col: Optional[str] = None
    site_col: Optional[str] = None

    def site_convert(self) -> None:
        self.df['site'] = 0  # refactor to "ZTF" and for phot to use Site class.

    def sub_convert(self) -> None:
        self.df['sub'] = True

    def band_convert(self) -> None:
        self.df['band'] = self.df[self.band_col].str[-1]


@dataclass(frozen=True)
class ZTFPhotLimConverter(ZTFPhotConverter):
    glob_str: str = 'SN*_limits.csv'
    mag_col: str = 'lim'


def read_ztfphot2(path: PathType) -> pd.DataFrame:
    return read_pandas_csv(path, index_col=0)


collate_ztfphot = Collator(ZTFPhotConverter, read_ztfphot2)
collate_ztfphot_limits = Collator(ZTFPhotLimConverter, read_ztfphot2)


def add_ztf_photometry(sn: SN, path: str | Path) -> SN:
    """
    Add ZTF photometry to a Supernova.
    """
    limits = collate_ztfphot_limits(path)
    detections = collate_ztfphot(path)

    new_phot = add_phot(sn.phot, detections)
    sn = update_sn(sn, new_phot)
    new_limits = add_phot(sn.limits, limits)
    sn.limits = new_limits
    return sn

# Below is Deprecated and will be removed in a future release


class ZTFPhotSuffix(StrEnum):
    """
    Photometry from ZTF
    """
    detections = '*_detections.csv'
    limits = '*_limits.csv'


def read_ztfphot_old(path: PathType, ztf_site_id: int = 0) -> pd.DataFrame:
    path = Path(path).resolve(strict=True)  # strict=True raises FileNotFoundError if doesn't exist
    name, filt, phot_type = path.stem.split('_')[:3]
    df = pd.read_csv(path, index_col=0)

    # Modify the columns to match the standard
    if phot_type == 'limits':
        df['mag'] = df['lim']
    df['site'] = ztf_site_id
    df['band'] = filt
    df['sub'] = True
    if not verify_columns(df):  # Double check that the columns are there since file versions can change.
        raise ValueError(f'Not all required keys are present in {path}')

    return df


def collate_ztfphot_old(path: PathType, suffix: ZTFPhotSuffix = ZTFPhotSuffix.limits) -> Photometry:
    path = Path(path).resolve(strict=True)
    phot = pd.concat([read_ztfphot_old(p) for p in path.glob(suffix)])
    phot = PhotFactory.from_df(phot)
    return phot


def collate_ztf_detections(path: PathType) -> Photometry:
    """
    Give a path to a directory containing ZTF photometry, return a Photometry object with the detections.
    """
    warnings.warn('collate_ztf_detections is deprecated. Use collate_ztfphot ', DeprecationWarning)
    return collate_ztfphot_old(path, ZTFPhotSuffix.detections)


def collate_ztf_limits(path: PathType) -> Photometry:
    """
    Give a path to a directory containing ZTF photometry, return a Photometry object with the limits.
    """
    warnings.warn('collate_ztf_limits is deprecated. Use collate_ztfphot_limits', DeprecationWarning)
    return collate_ztfphot_old(path, ZTFPhotSuffix.limits)
