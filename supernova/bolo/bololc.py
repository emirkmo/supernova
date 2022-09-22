from numpy.typing import NDArray

from supernova import SN, Photometry
from supernova.utils import StrEnum


class BoloType(StrEnum):
    """Bolometric luminosity type."""
    quasi = 'Bolo'  # Quasi-bolometric
    blackbody = 'BB'  # Blackbody


def bolo_weighted_xyz(sn: SN,
                      band: BoloType = BoloType.quasi) -> tuple[NDArray[float], NDArray[float], NDArray[float]]:
    """Get the bolometric luminosity weighted x, y, z coordinates."""
    phot = sn.band(band)
    weights = 1./phot.lum_err.values
    return phot.jd.values, phot.lum.values, weights
