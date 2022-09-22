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
    sn.set_phases()
    snr = sn.restframe()
    phot = snr.band(band)
    weights = 1./phot.lum_err.values
    return phot.phase.values, phot.lum.values, weights
