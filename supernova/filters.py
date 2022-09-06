from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum
from svo_filters import svo
import astropy.units as u


flam = u.erg/u.cm**2/u.s/u.angstrom
FLOWS_FILTERS = {
"u": svo.Filter("SDSS.u"),
"g": svo.Filter("PS1.g"),
"r": svo.Filter("PS1.r"),
"i": svo.Filter("PS1.i"),
"z": svo.Filter("PS1.z"),
"B": svo.Filter('MISC/APASS.B'),
"V": svo.Filter('MISC/APASS.V'),
"R": svo.Filter('NOT/ALFOSC.Bes_R'),
"J": svo.Filter('2MASS.J'),
"H": svo.Filter('2MASS.H'),
"K": svo.Filter('2MASS.Ks')
}

_MAG_SYS = {
    "u":'AB',
    "g":'AB',
    "r":'AB',
    "i":'AB',
    "z":'AB',
    "B":'Vega',
    "V":'Vega',
    "R":'Vega',
    "J":'Vega',
    "H":'Vega',
    "K":'Vega'
}


class FilterSorter:
    """
    Use in list.sort(key=FilterSorter(order: list[str]))
    Will use DEFAULT_ORDER if order is None. You can also
    import and append to DEFAULT_ORDER.
    """
    default_order = ["u", "B", "V", "g", "r", "R", "i", "I", "z", "Y", "J", "H", "K"]

    def __init__(self, order: Optional[list] = None):
        self.order = order if order is not None else self.default_order

    def __call__(self, band: str) -> int:
        if band not in self.order:
            return len(self.order)
        return self.order.index(band)


for name, filt in FLOWS_FILTERS.items():
    filt.magsys =_MAG_SYS[name]
    FLOWS_FILTERS[name] = filt
    if name == 'B':
        filt.zp = 6.49135e-9 * flam
    if name == 'V':
        filt.zp = 3.73384e-9 * flam


def get_flows_filter(band: str) -> svo.Filter:
    if band not in FLOWS_FILTERS.keys():
        raise ValueError(f"Band: `{band}` not found in flows filter list: {set(FLOWS_FILTERS.keys())}")
    return FLOWS_FILTERS.get(band)


class MagSys(Enum):
    AB = 'AB'
    Vega = 'Vega'


@dataclass
class Filter:
    """
    A class to represent a filter.
    Stores the filter name, the filter object from the SVO Filter Profile Service,
    the shift and color for plotting,
    as well as the effective wavelength and zero point flux (defaulting to zero),
    if relevant.
    """
    name: str
    wave_eff: u.Quantity = 0 * u.AA
    magsys: MagSys = MagSys.AB
    zp: float = 0.0
    plot_shift: float = 0.0
    plot_color: Optional[str | tuple[float, float, float]] = None
    svo: Optional[Any] = None  # should be svo.Filter but that's broken.

    def __str__(self):
        return self.name


