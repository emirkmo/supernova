from supernova import Photometry
from typing import get_args
import pandas as pd
from dataclasses import fields


def test_phot_partial_init():
    jds = pd.Series([1., 2., 3.])
    bands = pd.Series(['g', 'r', 'i'])
    for phot_class in get_args(Photometry):
        phot = phot_class(jd=jds, band=bands)
        for field in fields(phot):
            field_instance = getattr(phot, field.name)
            assert len(getattr(phot, field_instance)) == len(phot) == 3
            assert field.default.dtype == field_instance.dtype


if __name__ == '__main__':
    test_phot_partial_init()
