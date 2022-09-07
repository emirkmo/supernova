from pathlib import Path
import enum
import pandas as pd
from supernova.supernova import PhotFactory, Photometry, SN
from typing import Sequence, Optional
NEEDED_KEYS = 'site,band,mag,mag_err,jd,sub'.split(',')


class ZTFPhotSuffix(enum.StrEnum):
    """
    Photometry from ZTF
    """
    detections = '*_detections.csv'
    limits = '*_limits.csv'


def verify_columns(df: pd.DataFrame, keys: Optional[Sequence['str']] = None) -> bool:
    if keys is None:
        keys = NEEDED_KEYS
    return set(keys) - set(df.columns) == set()


def read_ztfphot(path: str | Path, ztf_site_id: int = 0) -> pd.DataFrame:
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


def collate_ztfphot(path: str | Path, suffix: ZTFPhotSuffix = ZTFPhotSuffix.limits) -> Photometry:
    path = Path(path).resolve(strict=True)
    phot = pd.concat([read_ztfphot(p) for p in path.glob(suffix)])
    phot = PhotFactory.from_df(phot)
    return phot


def get_ztf_detections(path: str | Path) -> Photometry:
    """
    Give a path to a directory containing ZTF photometry, return a Photometry object with the detections.
    """
    return collate_ztfphot(path, ZTFPhotSuffix.detections)


def get_ztf_limits(path: str | Path) -> Photometry:
    """
    Give a path to a directory containing ZTF photometry, return a Photometry object with the limits.
    """
    return collate_ztfphot(path, ZTFPhotSuffix.limits)


def update_sn(sn: SN, phot: Photometry) -> SN:
    # Add new sites.
    new_sites = set(phot.sites.unique()) - set(sn.sites.keys())
    if new_sites != set():
        for site in new_sites:
            sn.add_site(name=str(site), site_id=site)

    # Add new bands.
    sn.bands = sn.make_bands(list(sn.phot.band.unique()))
    sn.set_phases()  # also recalculates the old phases.
    return sn


def add_phot(sn_phot: Photometry, new_phot: Photometry) -> Photometry:
    """
    Add photometry to a Supernova.
    """
    df = pd.concat([sn_phot.as_dataframe(), new_phot.as_dataframe()], ignore_index=True)
    df = df[sn_phot.as_dataframe().columns]
    return PhotFactory.from_df(df)


def add_ztf_photometry(sn: SN, path: str | Path) -> SN:
    """
    Add ZTF photometry to a Supernova.
    """
    limits = get_ztf_limits(path)
    detections = get_ztf_detections(path)

    new_phot = add_phot(sn.phot, detections)
    sn = update_sn(sn, new_phot)
    new_limits = add_phot(sn.limits, limits)
    sn.limits = new_limits

    return sn


# rlim = pd.read_csv("../SNClass_20lao/SN2020lao_r_limits.csv")
# rlim['band'] = 'r'
# rlim['mag'] = rlim['lim']
# rlim['mag_err'] = rlim['plot_err']
# rlim['site'] = 0
# rlim = MagPhot.from_df(rlim)
# rlim.calc_phases(lao.phases.tess_patrick)
# rlim.phase = rlim.restframe_phases(lao.sninfo.redshift)
# rlim.mag = rlim.absmag(lao.distance, lao.sninfo.rext)  # absolute mag


rlim = read_ztfphot(Path(__file__)/"../../SNClass_20lao/SN2020lao_r_limits.csv")
