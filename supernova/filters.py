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

