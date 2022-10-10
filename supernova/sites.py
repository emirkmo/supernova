from dataclasses import dataclass, asdict, field, fields
from typing import Any, MutableMapping, TypeVar, Optional, Collection, cast
import warnings

import numpy as np
import pandas as pd
from typing_extensions import Unpack, TypedDict, NotRequired

from supernova.photometry import Photometry
from supernova.utils import tendrils_api

SiteType = TypeVar('SiteType', int, str)
SiteDict = MutableMapping[int, str]


@dataclass
class Site:
    id: int
    name: str
    marker: Optional[str] = None
    err_scale: float = 1.  # some photometry pipelines misrepresent errors.

    def __call__(self):
        return self.name

SiteKeys = TypedDict("SiteKeys", {'id': int, 'name': str, 'marker': NotRequired[str],
                     'err_scale': NotRequired[float]})
SiteKeysGenID = TypedDict("SiteKeys", {'id': NotRequired[int], 'name': str, 'marker': NotRequired[str],
                     'err_scale': NotRequired[float]})


@dataclass
class Sites:
    sites: dict[int, Site] = field(default_factory=dict)
    site_names: list[str] = field(default_factory=list, init=False)
    site_ids: list[int] = field(default_factory=list, init=False)
    sites_by_name: dict[str, Site] = field(default_factory=dict, init=False)
    markers: dict[int, Optional[str]] = field(default_factory=dict)
    site_err_scales: dict[int, float] = field(default_factory=dict)
    # for repeatable siteid generation:
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        self._refresh_sites()

    def __getitem__(self, site: SiteType) -> Site:
        if site in self.sites:  # int
            return self.sites[site]
        elif site in self.sites_by_name:  # str
            return self.sites_by_name[site]
        raise KeyError(f"site not found: {site}")

    def __contains__(self, site: SiteType) -> bool:
        return site in self.site_names or site in self.site_ids

    def __iter__(self):
        yield from self.sites.values()

    def _refresh_sites(self):
        self.site_names = [site.name for site in self.sites.values()]
        self.site_ids = [site.id for site in self.sites.values()]
        self.sites_by_name = {value.name: value for value in self.sites.values()}
        self.markers = {site.id: site.marker for site in self.sites.values()}

    def update_markers(self, site_markers: Optional[dict[int, str]] = None) -> None:
        if site_markers is not None:
            self.markers = site_markers  # type: ignore
        for site in self.sites.values():
            site.marker = self.markers[site.id]

    def get_marker(self, site: SiteType) -> str | None:
        return self[site].marker

    def get_err_scale(self, site: SiteType) -> float:
        return self[site].err_scale

    def asdict(self) -> dict:
        return asdict(self)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.asdict()['sites']).T

    def __len__(self) -> int:
        return len(self.site_ids)

    def generate_id(self):
        return self.rng.choice(list(set(range(100)) - set(self.site_ids)))

    def add_site(self, **site_kwargs: Unpack[SiteKeysGenID]) -> Site:
        if site_kwargs.get('id') is None:
            site_kwargs['id'] = self.generate_id()
        siteid = site_kwargs.get('id', self.generate_id())
        self.sites[siteid] = Site(**site_kwargs)
        self._refresh_sites()
        return self.sites[siteid]

    def remove_site(self, site: Site) -> Site:
        rmd_site = self.sites.pop(site.id)
        self._refresh_sites()
        return rmd_site

    @classmethod
    def from_sitemap(cls, sitemap: SiteDict) -> 'Sites':
        return cls(sites={k: Site(id=k, name=v) for k, v in sitemap.items()})

    @classmethod
    def from_dict(cls, d: dict[int, SiteKeys]) -> 'Sites':
        site_names = [f.name for f in fields(Site)]
        required_keys = [f for f in fields(Site) if not isinstance(f.default, f.type)]

        return cls(
            sites={k: Site(**{sk: sv for sk, sv in v.items() if sk in site_names}) # type: ignore
                   for k, v in d.items() if all([vk for vk in v if vk in required_keys])}
        )

    @classmethod
    def from_phot(cls, phot: pd.DataFrame | Photometry) -> 'Sites':
        return cls.from_list(dict(phot.site.unique().astype(str)))

    @classmethod
    def from_list(cls, sitenames: Collection[str]) -> 'Sites':
        return cls(sites={i: Site(i, s) for i, s in enumerate(sitenames)})

    @classmethod
    def from_df(cls, df: pd.DataFrame | pd.Series) -> 'Sites':
        if isinstance(df, pd.Series):
            return cls.from_sitemap(cast(SiteDict, df.to_dict()))
        if 'id' not in df:
            df['id'] = df.index
        if 'name' not in df:
            df.rename({df.columns[0]: 'name'}, axis=1, inplace=True)
        return cls.from_dict(cast(dict[int, SiteKeys], df.to_dict('index')))


def get_flows_sites() -> SiteDict:
    flows_sites = {8: 'LT', 5: 'NOT', 1: 'LCOGT', 0: 'ZTF'}
    try:
        api = tendrils_api()
        sites = api.get_sites()
        flows_sites = cast(SiteDict, {s['siteid']:s['sitename'] for s in sites})
        flows_sites[1] = 'LCOGT'
        flows_sites[0] = 'ZTF'
    except ImportError:
        warnings.warn("tendrils_api not found, using default flows_sites")
    
    return flows_sites
    

flows_sites = get_flows_sites()
site_markers: SiteDict = {1: 's', 5: 'd', 8: 'v', 0: 'o'}
site_err_scales: dict[int, int] = {1: 2, 5: 2, 8: 5, 0: 1}
