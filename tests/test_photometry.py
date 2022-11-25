from typing import get_args

import pandas as pd
from dataclasses import fields

from astrosn import Photometry
from astrosn.photometry import BBLumPhot, FluxPhot, LumPhot, MagPhot, Phot, PhotFactory

phot_classes = [MagPhot, FluxPhot, LumPhot, Phot, BBLumPhot]


def test_phot_load():
    df = pd.read_csv("tests/inputs/SN2020lao_phot.csv", index_col=0).squeeze("columns")
    phot = PhotFactory.from_df(df)
    assert isinstance(phot, Photometry)


def test_phot_partial_init():
    jds = pd.Series([1.0, 2.0, 3.0])
    bands = pd.Series(["g", "r", "i"])

    for phot_class in phot_classes:
        phot = phot_class(jd=jds, band=bands)
        for field in fields(phot):
            field_instance = getattr(phot, field.name)
            assert len(getattr(phot, field.name)) == len(phot) == 3
            assert field.default.dtype == field_instance.dtype  # type: ignore
    # for phot_class in get_args(Photometry):

    # phot = phot_class(jd=jds, band=bands)
    # for field in fields(phot):
    #     field_instance = getattr(phot, field.name)
    #     assert len(getattr(phot, field_instance)) == len(phot) == 3
    #     assert field.default.dtype == field_instance.dtype


if __name__ == "__main__":
    test_phot_load()
    test_phot_partial_init()
