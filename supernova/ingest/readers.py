from typing import Optional, Callable, Sequence
from astropy.table import Table
from astropy.time import Time
import pandas as pd
from pathlib import Path
import warnings

PathType = Path | str
PhotReader = Callable[[PathType], pd.DataFrame]


def resolve_path(path: PathType) -> Path:
    return Path(path).resolve(strict=True)


def read_astropy_table(path: PathType, fmt: str = 'ascii.ecsv') -> Table:
    path = resolve_path(path)
    return Table.read(path, format=fmt)


def read_pandas_csv(path: PathType, index_col: Optional[int | str] = None,
                    is_series: bool = False) -> pd.DataFrame:
    path = resolve_path(path)
    df = pd.read_csv(path, index_col=index_col)
    return df if not is_series else df.squeeze('columns')


def read_astropy_table_as_df_with_times_as_jd(path: PathType, fmt: str = 'ascii.ecsv') -> pd.DataFrame:
    at = read_astropy_table(path, fmt=fmt)
    at = times_as_jd(at)
    return at.to_pandas()


def get_astropy_times_as_jd(at: Table) -> Table:
    times_cols = get_times_cols(at)

    if len(times_cols) == 0:
        warnings.warn(f"No time columns found in astropy table with columns: {at.colnames}", UserWarning)
        return at

    for col in times_cols:
        fmt = detect_mjd_or_jd(at[col])
        if fmt is None:
            warnings.warn(f"Could not determine time format for column: {col}.", UserWarning)
        else:
            at[col] = Time(at[col], format=fmt).jd
    return at


def detect_mjd_or_jd(col: Sequence) -> Optional[str]:
    def mjd(sval: str) -> bool:
        return len(sval) == 5 and sval[0] in ['5', '6']

    def jd(sval: str) -> bool:
        return len(sval) == 7 and sval.startswith("245")

    mjds = []
    jds = []
    for val in col:
        sval = str(int(val))
        if mjd(sval):
            mjds.append(val)
        elif jd(sval):
            jds.append(val)
        if len(mjds) > 1 and len(jds) > 1:
            break
    if len(mjds) == len(col):
        return 'mjd'
    elif len(jds) == len(col):
        return 'jd'
    return None


def get_times_cols(at: Table) -> list[str]:
    time_names = ["time", "jd", "mjd"]
    return [col for col in at.colnames if col.strip().lower() in time_names]


def times_as_jd(at: Table) -> Table:
    for name, col in at.columns.items():
        if isinstance(col, Time):
            at[name] = col.jd
    return at
