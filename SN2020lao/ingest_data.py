from supernova.ingest import FlowsIngestor
from supernova.supernova import SNInfo, SN
from supernova.interp import SNInterp
from pathlib import Path
import numpy as np


def ingest_20lao() -> SN:
    sninfo = SNInfo.from_csv(Path("./inputs/SN2020lao_sninfo.csv"))
    sn = FlowsIngestor("./inputs/", sninfo=sninfo).load_sn()
    sn.restframe()
    sn.bands.pop('R')
    sn.bands.pop('u')
    sn.phases['discovery'] = 2458994.9131366
    sn.phases['peak_r'] = 2459001.7783796
    sn.phases['peak_g'] = 2459001.8907986
    sn.to_csv('./')
    return sn


def reference_times(sn: SN, band: str, phase_end: float) -> tuple[float, float]:
    return sn.band(band).jd.values[0], sn.band(band).jd.values[0] + phase_end * (1. + sn.redshift)


def interpolate_with_piscola(sn: SN, jd_begin: float, jd_end: float) -> tuple[SN, SNInterp]:
    times = np.arange(jd_begin, jd_end)
    interp = SNInterp(sn, sn.bands, times)
    return interp.create_interp_sn(), interp


def main() -> tuple[SN, SN, SNInterp]:
    _ = ingest_20lao()
    sn = SN.from_csv("./SNClass_SN2020lao/")  # load SN from CSV to test serialization
    jd_begin, jd_end = reference_times(sn, 'g', 86)
    sn_interp, interp = interpolate_with_piscola(sn, jd_begin, jd_end)
    sn_interp.to_csv("./interp/")
    return sn, sn_interp, interp


if __name__ == '__main__':
    lao, lao_interp, lao_sninterp = main()
