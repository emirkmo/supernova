import pandas as pd
from .converters import Converter, GenericCSVConverter, GenericECSVConverter, GenericJSONConverter
from .readers import PathType, resolve_path, PhotReader, read_astropy_table, read_pandas_csv, read_astropy_table_as_df_with_times_as_jd
from typing import Protocol, Type, Callable
from ..supernova import Photometry, PhotFactory
CollatorType = Callable[[PathType], Photometry]


class AbstractCollator(Protocol):

    def __init__(self, converter: Type[Converter],
                 reader: PhotReader, ignore_processed: bool = True) -> None:
        self.path = 'Unset'  # last processed path.
        self.converter = converter
        self.reader = reader
        self.processed_files: list[PathType] = []  # all processed files.
        self.ignore = ignore_processed
        raise NotImplementedError("This is an abstract Protocol class.")

    def __call__(self, path: PathType) -> Photometry:
        ...

    def process(self, file: PathType) -> pd.DataFrame:
        ...

    def collate(self, path: PathType) -> Photometry:
        ...


class Collator:

    def __init__(self, converter: Type[Converter],
                 reader: PhotReader, ignore_processed: bool = True) -> None:
        self.path = "Unset"
        self.converter = converter
        self.reader = reader
        self.processed_files: list[PathType] = []
        self.ignore = ignore_processed

    def __call__(self, path: PathType) -> Photometry:
        self.path = resolve_path(path)
        return self.collate(self.path)

    def process(self, file: PathType) -> pd.DataFrame:
        if self.ignore and file in self.processed_files:
            return pd.DataFrame()  # empty dataframes in pd.concat are ignored.
        self.processed_files.append(file)
        return self.reader(file)

    def collate(self, path: PathType) -> Photometry:
        df = pd.concat([self.process(p) for p in path.glob(self.converter.glob_str)], ignore_index=True)
        phot_df = self.converter(df=df).convert()
        return PhotFactory.from_df(phot_df)


collate_ecsv = Collator(GenericECSVConverter, read_astropy_table_as_df_with_times_as_jd)
collate_csv = Collator(GenericCSVConverter, read_pandas_csv)
collate_json = Collator(GenericECSVConverter, pd.read_json)

