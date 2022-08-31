"""
Supernova data analysis library by @emirkmo
"""
from .version import version as __version__
from .version import version_info as __version_info__
from .supernova import SN, MagPhot
from .interp import *
from .filters import *
from .sn_data import lao
