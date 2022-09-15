from supernova.ingest import FlowsIngestor
from supernova.supernova import SNInfo
from pathlib import Path


def ingest_20lao():
    sninfo = SNInfo.from_csv(Path("./inputs/SN2020lao_sninfo.csv"))
    sn = FlowsIngestor("./inputs/", sninfo=sninfo).load_sn()
    sn.restframe()
    sn.bands.pop('R')
    sn.bands.pop('u')
    sn.phases['discovery'] = 2458994.9131366
    sn.phases['peak_r'] = 2459001.7783796
    sn.phases['peak_g'] = 2459001.8907986
    sn.to_csv()


if __name__ == '__main__':
    ingest_20lao()
