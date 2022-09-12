from pathlib import Path
from .flows import collate_flows
from .readers import PathType
from .ztf import collate_ztf_flows, collate_ztfphot, collate_ztfphot_limits
from .utils import add_phot
from .collators import (
    AbstractCollator, GenericCSVCollator, GenericECSVCollator, GenericJSONCollator)
from supernova.supernova import SN, SNInfo, Photometry
from typing import Optional, Any
from tempfile import TemporaryDirectory

PREDEFINED_COLLATORS = {'flows': collate_flows,
                        'ztf_flows': collate_ztf_flows,
                        'ztfphot': collate_ztfphot}
PREDEFINED_LIMITS_COLLATORS = {'ztfphot_limits': collate_ztfphot_limits}
GENERIC_COLLATORS =


class DataIngestor:

    def __init__(self, path: PathType, sninfo: Optional[SNInfo] = None,
                 collators: Optional[dict[str, AbstractCollator]] = None,
                 limits_collators: Optional[dict[str, AbstractCollator]] = None,
                 **sninfo_kwargs: Any) -> None:
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

        self.collators = collators or PREDEFINED_COLLATORS
        self.limits_collators = limits_collators or PREDEFINED_LIMITS_COLLATORS
        self.sninfo_kwargs = sninfo_kwargs

    def get_files(self) -> list[Path]:
        return list(self.path.glob('*'))

    def tmpdir(self) -> TemporaryDirectory:
        _tmpdir = TemporaryDirectory()
        self.tmp_path = Path(_tmpdir.name)
        yield _tmpdir
        _tmpdir.cleanup()

    def move_files_to_temp_dir(self, tmpdir_path: Path) -> None:
        for file in self.files:
            temp_file = tmpdir_path / file.name
            temp_file.symlink_to(file)

    def process_file(self, file: Path) -> None:
        if file in self.processed_files:
            return
        if file.is_symlink():
            file.unlink()
        self.processed_files.append(file)

    def load_phot(self, lims: bool = False) -> list[Photometry]:
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
                    self.process_file(file)

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
            lims = self.load_phot(lims=True)
            if len(phots) == 0:
                raise FileNotFoundError(f"No photometry found in {self.path}."
                                        f"With defined data collators: {self.collators}")

            phot = phots[0]

            return SN.from_phot(
                phot=phot,
                name=self.sninfo.name,
                redshift=self.sninfo.redshift,
                sub_only=self.sninfo.sub_only,
                phase_zero=self.sninfo.phase_zero,
                lims=lims[0] if len(lims) > 0 else None,
                **self.sninfo_kwargs)

    @staticmethod
    def query_user(query: str) -> str:
        return input(query)