from .supernova import SN
from pathlib import Path


lao = SN.from_csv(Path(__file__).parent.absolute()/"./sndata/SNClass_20lao/")
