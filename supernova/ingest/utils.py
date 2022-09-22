from pathlib import Path
import pandas as pd
from supernova.supernova import PhotFactory, Photometry, SN, SNSerializer
from typing import Sequence, Optional, Type, Callable
from .readers import PathType, resolve_path, PhotReader
from .converters import Converter
NEEDED_KEYS = 'site,band,mag,mag_err,jd,sub'.split(',')
Collator = Callable[[PathType], Photometry]


def collate_phot(path: PathType, converter: Type[Converter], reader: PhotReader) -> Photometry:
    path = resolve_path(path)
    df = pd.concat([reader(p) for p in path.glob(converter.glob_str)], ignore_index=True)
    phot_df = converter(df=df).convert()
    return PhotFactory.from_df(phot_df)


add_phot = PhotFactory.add_phot


def create_sn(name: str, dir_path: PathType, collator: Collator,
              redshift: float = 0, phase_zero: float = 0,
              save_sn: bool = True) -> SN:
    """
    Create a Supernova object from a directory of photometry files,
    optionally saving it to the parent directory.
    """
    phot = collator(dir_path)
    sn = SN.from_phot(name=name, phot=phot, redshift=redshift, phase_zero=phase_zero)
    if save_sn:
        SNSerializer(sn).to_csv(Path(dir_path).parent)
    return sn


def update_sn(sn: SN, phot: Photometry) -> SN:
    # Add new sites.
    new_sites = set(phot.site.unique()) - set(sn.sites.site_ids)
    if new_sites != set():
        for site in new_sites:
            sn.add_site(name=str(site), site_id=site)

    # Add new bands.
    sn.bands = sn.make_bands(list(sn.phot.band.unique()), ebv=sn.sninfo.ebv)
    sn.set_phases()  # also recalculates the old phases.
    return sn


def verify_columns(df: pd.DataFrame, keys: Optional[Sequence['str']] = None) -> bool:
    if keys is None:
        keys = NEEDED_KEYS
    return set(keys) - set(df.columns) == set()
