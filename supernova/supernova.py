import pandas as pd
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Union
import astropy.units as u
from pathlib import Path
import os
import numpy as np

sites = {8: 'LT', 5: 'NOT', 1: 'LCOGT'}
site_markers = {1: 's', 5: 'd', 8: 'v'}
site_err_scales = {1: 2, 5: 2, 8: 5}


@dataclass
class MagPhot:
    jd: pd.Series = None
    mag: pd.Series = None
    mag_err: pd.Series = None
    filter: pd.Series = None
    band: pd.Series = None  # alternative to filter, avoid overwriting
    phase: pd.Series = None
    sub: pd.Series = None
    site: pd.Series = None
    
    def __post_init__(self):
        self.filter = self.band if self.band is not None else self.filter
        self.band = self.filter if self.filter is not None else self.band
        
    
    def calc_phases(self, phase_zero):
        self.phase = self.jd - phase_zero
        
    def restframe_phases(self, redshift):
        if self.phase is None:
            raise AttributeError("self.phase must not be None, calculate it first using calc_phases with a phase zero")
        return self.phase/(1.0+redshift)
    
    @classmethod
    def from_df(cls, df):
        d = df.to_dict('series')
        return cls.from_dict(d)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**{k:v for k, v in d.items() if k in [f.name for f in fields(cls)]})
    
    def masked(self, cond: list):
        d2 = {name: value[cond] for name, value in asdict(self).items()}
        return MagPhot.from_dict(d2)
    
    def absmag(self, dm, ext):
        return self.mag-dm-ext
        
        
    
    
@dataclass    
class FluxPhot(MagPhot):
    flux: pd.Series = None
    flux_err: pd.Series  = None
    

@dataclass
class SN:
    phot: pd.DataFrame | MagPhot
    phases: pd.Series
    sninfo: pd.Series
    sites: Dict[int,str] = field(default_factory=dict)
    name: str = '20lao'
    sub_only: bool = True
    distance: float = None
    #specinfo:
    
    def __post_init__(self):
        if isinstance(self.phot, pd.DataFrame):
            self.phot = MagPhot.from_df(self.phot)
        self.set_sites_r()
        self.rng = np.random.default_rng()
        self.distance = self.sninfo.dm
        
    def set_sites_r(self) -> None:
        self.sites_r = {value: key for key, value in self.sites.items()}
        
    def add_site(self, name: str, site_id: int = None) -> None:
        if site_id is None:
            site_id = self.rng.choice(set(range(100)) - set(self.sites.keys()))
        self.sites[site_id] = name
        self.set_sites_r()
    
    def band(self, filt: str, site: str = 'all', return_absmag: bool = False) -> MagPhot:
        phot = self.phot.masked(self.phot.band == filt)
        if self.sub_only:
            phot = phot.masked(phot.sub)
        
        if site!='all' and site not in self.sites_r.keys():
            raise ValueError('not a valid site name. Define first with `add_site`')
        
        if site!='all':
            site_id = self.sites_r[site]
            phot = phot.masked(phot.site==site_id)
        if return_absmag:
            phot.absmag(self.sninfo.dm, self.sninfo[filt+'ext'])
        return phot
    
    def site(self, site: str) -> MagPhot:
        site_id = self.sites_r[site]
        return self.phot.masked(self.phot.site==site_id)
    
    def absmag(self, phot: MagPhot = None):
        if phot is None:
            phot = self.phot
        return phot.absmag(self.sninfo.dm, self.sninfo[filt+'ext'])
        
    
    
    def to_csv(self, basepath: Union[str, Path] = Path('../')):
        basepath = Path(basepath).joinpath(Path(f"SNClass_{self.name}/"))
        
        os.makedirs(basepath, exist_ok=True)
        
        self.sninfo['name'] = self.name
        self.sninfo['sub_only'] = self.sub_only
        names = 'phot,phases,sninfo,sites'.split(',')
        for name, field in zip(names,[self.phot, self.phases, self.sninfo, self.sites]):
            if isinstance(field, dict):
                field = pd.Series(field)
            if isinstance(field, MagPhot):
                field = pd.DataFrame.from_dict(asdict(field))
            save_name = f"{basepath.absolute()}/{self.name}_{name}.csv"
            #save_name += "_series.csv" if isinstance(lao.sninfo,pd.Series) else ".csv"
            field.to_csv(save_name, index=True)
            
    @classmethod
    def from_csv(cls, dirpath: Union[str, Path]):
        import glob
        names = 'phot,phases,sninfo,sites'.split(',')
        fields = {}
        csvs = glob.glob( str(Path(dirpath) / '*.csv'))
        for csv in csvs:
            df = pd.read_csv(csv, index_col=0).squeeze("columns")
            for name in names:
                if name in csv:
                    fields[name] = df
                    
        return cls(phot=fields['phot'],
            phases=fields['phases'],
            sninfo=fields['sninfo'],
            sites=fields['sites'],
            name=fields['sninfo'].name if 'name' in fields['sninfo'].index else 'unknown',
            sub_only=fields['sninfo'].sub_only if 'name' in fields['sninfo'].index else False
           )
