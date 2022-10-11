from .supernova import SN
from pathlib import Path

lao = SN.from_csv(Path(__file__).parent.absolute()/"../SNClass_SN2020lao/")
