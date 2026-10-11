"""Tests for the retained NASA Exoplanet Archive CSV reader."""

from tdpy.verbosity import print

from gordion.catalog import fetch_confirmed_exoplanets


def test_catalog_keeps_mass_provenance_and_optional_orbital_data(tmp_path):
    path = tmp_path / "planetary_systems.csv"
    print(f"Writing to {path}...")
    path.write_text(
        "pl_name,hostname,pl_bmassj,pl_radj,pl_bmassprov,pl_orbper,st_mass\n"
        "Planet b,Host A,0.05,0.12,Mass,8.0,0.9\n"
        "Planet c,Host B,0.02,0.10,Msin(i),,,\n"
        "Planet d,Host C,,0.20,Mass,4.0,1.0\n",
        encoding="utf-8",
    )

    records = fetch_confirmed_exoplanets(cache_path=path)

    assert [record.name for record in records] == ["Planet b", "Planet c"]
    assert records[0].period_days == 8.0
    assert records[0].host_mass_solar == 0.9
    assert records[1].mass_provenance == "Msin(i)"
    assert records[1].period_days is None
    assert records[1].host_mass_solar is None