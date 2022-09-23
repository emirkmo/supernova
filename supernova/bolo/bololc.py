from numpy.typing import NDArray
import numpy as np
from supernova import SN, Photometry, Filter
from supernova.utils import PathType, StrEnum
from typing import Any, Protocol, TypeVar

# Filters
bolo_filt = Filter('Bolo')
blackbody_filt = Filter("BB")

class BoloType(StrEnum):
    """Bolometric luminosity type."""
    quasi = 'Bolo'  # Quasi-bolometric
    blackbody = 'BB'  # Blackbody


def bolo_weighted_xyz(sn: SN,
                      band: BoloType = BoloType.quasi) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Get the bolometric luminosity weighted x, y, z coordinates."""
    sn.set_phases()
    snr = sn.restframe()
    phot = snr.band(band)
    weights = 1./phot.lum_err.values
    return phot.phase.values, phot.lum.values, weights


class ModelFit(Protocol):

    def __init__(self, params: Any, **kwargs):
        ...

    def get_model(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    def to_json(self, path: PathType) -> None:
        ...
