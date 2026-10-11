"""NASA Exoplanet Archive inputs for population-wide Gordion examples."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tdpy.verbosity import print

from .paths import get_data_path


ARCHIVE_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
ARCHIVE_QUERY = (
    "select pl_name,hostname,pl_bmassj,pl_radj,pl_bmassprov,pl_orbper,st_mass "
    "from pscomppars where pl_bmassj is not null and pl_radj is not null"
)


@dataclass(frozen=True)
class ExoplanetRecord:
    """A catalog row with measured mass, radius, orbit, and host-star mass."""

    name: str
    host_name: str
    mass_jupiter: float
    radius_jupiter: float
    mass_provenance: str
    period_days: float | None
    host_mass_solar: float | None


def fetch_confirmed_exoplanets(*, refresh: bool = False,
                               cache_path: str | Path | None = None) -> list[ExoplanetRecord]:
    """Load all catalog planets with mass, radius, period, and host mass.

    The first call downloads one NASA Exoplanet Archive TAP CSV snapshot. Later
    calls use the retained CSV unless ``refresh=True``. A planet's mass
    provenance is preserved because some catalog masses are minimum masses.
    """

    path = Path(cache_path) if cache_path is not None else get_data_path() / "exoplanet_archive" / "planetary_systems.csv"
    if refresh or not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{ARCHIVE_TAP_URL}?{urlencode({'query': ARCHIVE_QUERY, 'format': 'csv'})}"
        print(f"Reading from {url}...")
        request = Request(url, headers={"User-Agent": "Gordion dark-matter phase-space example"})
        with urlopen(request, timeout=90) as response:
            content = response.read().decode("utf-8")
        print(f"Writing to {path}...")
        path.write_text(content)
    print(f"Reading from {path}...")
    records = []
    for row in csv.DictReader(io.StringIO(path.read_text())):
        try:
            mass = float(row["pl_bmassj"])
            radius = float(row["pl_radj"])
        except (TypeError, ValueError):
            continue
        if min(mass, radius) <= 0.0:
            continue
        try:
            period = float(row["pl_orbper"])
        except (TypeError, ValueError):
            period = None
        try:
            host_mass = float(row["st_mass"])
        except (TypeError, ValueError):
            host_mass = None
        if period is not None and period <= 0.0:
            period = None
        if host_mass is not None and host_mass <= 0.0:
            host_mass = None
        records.append(ExoplanetRecord(
            name=row.get("pl_name", ""),
            host_name=row.get("hostname", ""),
            mass_jupiter=mass,
            radius_jupiter=radius,
            mass_provenance=row.get("pl_bmassprov", "unknown"),
            period_days=period,
            host_mass_solar=host_mass,
        ))
    if not records:
        raise ValueError("No catalog rows have positive mass, radius, period, and host mass")
    return records