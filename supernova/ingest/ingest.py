from pathlib import Path
from .flows import collate_flows
from .readers import PathType
from .ztf import collate_ztf_flows, collate_ztfphot, collate_ztfphot_limits
from .utils import add_phot
from .collators import AbstractCollator, collate_ecsv, collate_csv, collate_json
from supernova.supernova import SN, SNInfo
from supernova.photometry import Photometry
from supernova.sites import Sites, flows_sites, SiteDict
from enum import Enum
from typing import Generator, Mapping, Optional, Any
from tempfile import TemporaryDirectory
from contextlib import contextmanager


PREDEFINED_COLLATORS = {
    "flows": collate_flows,
    "ztf_flows": collate_ztf_flows,
    "ztfphot": collate_ztfphot,
}
PREDEFINED_LIMITS_COLLATORS = {"ztfphot_limits": collate_ztfphot_limits}
GENERIC_COLLATORS = {"ecsv": collate_ecsv, "csv": collate_csv, "json": collate_json}


class DataIngestor:
    """
    Base class for data ingestors.
    """

    def __init__(
        self,
        path: PathType,
        collators: Optional[Mapping[str, AbstractCollator]] = None,
        limits_collators: Optional[Mapping[str, AbstractCollator]] = None,
        sninfo: Optional[SNInfo] = None,
        sitemap: Optional[SiteDict] = None,
        **sninfo_kwargs: Any,
    ) -> None:
        """
        A class to collect data from a directory of files and create an SN object.
        RA and Dec can be provided in degrees, or they can be looked up from the SN name,
        which should match IAU specifications.Redshift can be given, if known.
        """
        self.path = Path(path).resolve()
        self.tmp_path = self.path
        self.files = self.get_files()
        self.processed_files = []
        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found in {self.path}.")

        if sninfo is None:
            sninfo = SNInfo.from_csv(self.path)
        self.sninfo = sninfo
        self.sitemap = sitemap
        self.collators = collators or PREDEFINED_COLLATORS
        self.limits_collators = limits_collators or PREDEFINED_LIMITS_COLLATORS
        self.sninfo_kwargs = sninfo_kwargs

    def setup(self) -> None:

        """
        Setup the data ingestor
        """
        pass

    def get_files(self) -> list[Path]:
        return list(self.path.glob("*"))

    @contextmanager
    def tmpdir(self) -> Generator[TemporaryDirectory, None, None]:
        _tmpdir = TemporaryDirectory()
        self.tmp_path = Path(_tmpdir.name)
        try:
            yield _tmpdir
        finally:
            _tmpdir.cleanup()

    def move_files_to_temp_dir(self, tmpdir_path: Path) -> None:
        for file in self.files:
            temp_file = tmpdir_path / file.name
            temp_file.symlink_to(file)

    def process_file(self, file: Path, unlink: bool = True) -> None:
        if file in self.processed_files:
            return
        if file.is_symlink() and unlink:
            file.unlink()
        self.processed_files.append(file)

    def load_phot(self, lims: bool = False, unlink: bool = True) -> list[Photometry]:
        """
        Load the photometry from the data in the directory.
        Unlink the files after they are processed.
        """
        phots = []
        phot_type = "limits" if lims else "photometry"
        collators = self.limits_collators if lims else self.collators
        for source, collator in collators.items():
            try:
                phots.append(collator(self.tmp_path))
                print(f"Loaded {phot_type} from {source}.")

                for file in self.tmp_path.glob(collator.converter.glob_str):
                    self.process_file(file, unlink=unlink)

            except FileNotFoundError:
                print(f"Did not find {phot_type} from {source}.")
            if len(phots) == 2:
                phots.append(add_phot(phots.pop(0), phots.pop(0)))

        return phots

    def load_sn(self) -> SN:
        """
        Load the SN object from the data in the directory.
        But first symlink all the files to a temporary directory,
        so that the collators can delete the files they process.
        """
        with self.tmpdir() as tmpdir:
            self.move_files_to_temp_dir(Path(tmpdir.name))
            phots = self.load_phot()
            if self.limits_collators is not None:
                lims = self.load_phot(lims=True)
            else:
                lims = []
            if len(phots) == 0:
                raise FileNotFoundError(
                    f"No photometry found in {self.path}."
                    f"With defined data collators: {self.collators}"
                )

            phot = phots[0]
            sites = None
            if self.sitemap:
                sites = Sites.from_sitemap(
                    {s: self.sitemap.get(s, str(s)) for s in phot.site.unique()}
                )

            phot = phot.sorted()
            return SN.from_phot(
                phot=phot,
                name=self.sninfo.name,
                redshift=self.sninfo.redshift,
                sub_only=self.sninfo.sub_only,
                phase_zero=self.sninfo.phase_zero,
                limits=lims[0] if len(lims) > 0 else None,
                sninfo=self.sninfo,
                sites=sites,
                **self.sninfo_kwargs,
            )

    @staticmethod
    def query_user(query: str) -> str:
        return input(query)


class GenericIngest(DataIngestor):
    def __init__(
        self,
        path: PathType,
        sninfo: Optional[SNInfo] = None,
        sitemap: Optional[SiteDict] = None,
        **sninfo_kwargs: Any,
    ) -> None:
        collators = GENERIC_COLLATORS
        super().__init__(path, collators, None, sninfo, sitemap, **sninfo_kwargs)


class ECSVIngest(DataIngestor):
    def __init__(
        self,
        path: PathType,
        sninfo: Optional[SNInfo] = None,
        sitemap: Optional[SiteDict] = None,
        **sninfo_kwargs: Any,
    ) -> None:
        collators = {"ecsv": collate_ecsv}
        super().__init__(path, collators, None, sninfo, sitemap, **sninfo_kwargs)


class CSVIngest(DataIngestor):
    def __init__(
        self,
        path: PathType,
        sninfo: Optional[SNInfo] = None,
        sitemap: Optional[SiteDict] = None,
        **sninfo_kwargs: Any,
    ) -> None:
        collators = {"csv": collate_csv}
        super().__init__(path, collators, None, sninfo, sitemap, **sninfo_kwargs)


class JSONIngest(DataIngestor):
    def __init__(
        self,
        path: PathType,
        sninfo: Optional[SNInfo] = None,
        sitemap: Optional[SiteDict] = None,
        **sninfo_kwargs: Any,
    ) -> None:
        collators = {"json": collate_json}
        super().__init__(path, collators, None, sninfo, sitemap, **sninfo_kwargs)


class FlowsIngest(DataIngestor):
    def __init__(
        self,
        path: PathType,
        sninfo: Optional[SNInfo] = None,
        sitemap: Optional[SiteDict] = flows_sites,
        **sninfo_kwargs: Any,
    ) -> None:
        collators = PREDEFINED_COLLATORS
        limits_collators = PREDEFINED_LIMITS_COLLATORS

        super().__init__(
            path, collators, limits_collators, sninfo, sitemap, **sninfo_kwargs
        )


class ZTFIngest(DataIngestor):
    def __init__(
        self,
        path: PathType,
        sninfo: Optional[SNInfo] = None,
        sitemap: Optional[SiteDict] = {0 : "ZTF"},
        **sninfo_kwargs: Any,
    ) -> None:
        collators = {"ztf": collate_ztf_flows, 'ztfphot': collate_ztfphot}
        limits_collators = None
        super().__init__(
            path, collators, limits_collators, sninfo, sitemap, **sninfo_kwargs
        )


class Ingestors(Enum):
    flows = FlowsIngest
    generic = GenericIngest
    ecsv = ECSVIngest
    csv = CSVIngest
    json = JSONIngest
    ztf = ZTFIngest


def ingest_sn(
    path: PathType,
    snname: str = "SN1990Unknown",
    ingestor: str | Ingestors = Ingestors.flows,
    sninfo: Optional[SNInfo] = None,
    sitemap: Optional[SiteDict] = None,
    **sninfo_kwargs: Any,
) -> SN:
    """
    Ingest a SN from a directory of data.

    Parameters
    ----------
    path: str | Path
        The path to the directory of data.
    snname: str
        The name of the SN.
    ingestor: str | Ingestors = Ingestors.flows
        The ingestor to use. Can be a string or an Ingestors enum. Flows is the default.
    sninfo: Optional[SNInfo] = None
        An SNInfo object to use for the SN. If None, will create via query.
    sitemap: Optional[dict[int, str]] = None
        A dictionary mapping site numbers to site names. Uses FLOWS sites by default if ingestor is flows.
        Pass None to have them automatically generated from the files. (Will check for file ending in _sites,
        and if not found, will use the site numbers in the photometry file. If photometry file has sitenames
        instead of numbers, will use those.)
        Else pass a dictionary mapping site numbers to site names. Remember that photometry files must have
        the corresponding site numbers for each telescope in every row.
    sninfo_kwargs: Any
        Any additional kwargs to pass to the SNInfo constructor.
        Such as redshift, phase_zero, sub_only, etc. Redshift will be looked up from flows if possible and not passed.
        tendrils must be installed for this.
    """
    if isinstance(ingestor, str):
        ingestor = Ingestors[ingestor]

    redshift = sninfo_kwargs.pop("redshift", 0.0)
    if isinstance(redshift, str):
        redshift = float(redshift)  # try to convert to float
    if not isinstance(redshift, float):
        raise TypeError(
            f"redshift must be a float or a string that can be converted to a float. Got {redshift}."
        )
    #if redshift == 0.0:
        

    if sninfo is None:
        sninfo = SNInfo.from_name(snname, redshift=redshift, **sninfo_kwargs)

    if ingestor == Ingestors.flows:
        sitemap = flows_sites if sitemap is None else sitemap

    return ingestor.value(path, sninfo, sitemap, **sninfo_kwargs).load_sn()
