from dataclasses import dataclass, asdict, field
from typing import TypeVar, Mapping, Optional
from typing_extensions import Unpack
import numpy as np

SiteType = TypeVar('SiteType', int, str)
SiteDict = Mapping[int, str]


@dataclass
class Site:
    id: int
    name: str
    marker: Optional[str] = None
    err_scale: float = 1.  # some photometry pipelines misrepresent errors.

    def __call__(self):
        return self.name


@dataclass
class Sites:
    sites: dict[int, Site] = field(default_factory=dict)
    site_names: list[str] = field(default_factory=list, init=False)
    site_ids: list[int] = field(default_factory=list, init=False)
    sites_by_name: dict[str, Site] = field(default_factory=dict, init=False)
    markers: dict[int, str] = field(default_factory=dict)
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

    def _refresh_sites(self):
        self.site_names = [site.name for site in self.sites.values()]
        self.site_ids = [site.id for site in self.sites.values()]
        self.sites_by_name = {value.name: value for value in self.sites.values()}
        self.markers = {site.id: site.marker for site in self.sites.values()}

    def get_marker(self, site: SiteType) -> str:
        return self[site].marker

    def get_err_scale(self, site: SiteType) -> float:
        return self[site].err_scale

    def asdict(self) -> dict:
        return asdict(self)

    def __len__(self) -> int:
        return len(self.site_ids)

    def generate_id(self):
        return self.rng.choice(set(range(100)) - set(self.site_ids))

    def add_site(self, name: str = 'Unknown', **site_kwargs: Unpack[Site]) -> None:
        if 'id' not in site_kwargs:
            site_kwargs['id'] = self.generate_id()
        self.sites[site_kwargs['id']] = Site(name=name, **site_kwargs)
        self._refresh_sites()

    @classmethod
    def from_dict(cls, d: dict[int, str]) -> 'Sites':
        return cls(sites={k: Site(id=k, name=v) for k, v in d.items()})

    @classmethod
    def from_list(cls, sitenames: list) -> 'Sites':
        return cls(sites={i: s for i, s in enumerate(sitenames)})


flows_sites = {8: 'LT', 5: 'NOT', 1: 'LCOGT'}
site_markers = {1: 's', 5: 'd', 8: 'v'}
site_err_scales = {1: 2, 5: 2, 8: 5}
