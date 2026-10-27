"""Tests for the retained NASA Exoplanet Archive CSV reader."""

import csv

import pytest
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


@pytest.mark.parametrize('field', ['pl_bmassj', 'pl_radj', 'pl_orbper', 'st_mass'])
@pytest.mark.parametrize('invalid', ['nan', 'inf', '-inf'])
def test_catalog_does_not_promote_nonfinite_measurements(field, invalid, tmp_path):
    path = tmp_path / 'nonfinite_catalog.csv'
    control = {
        'pl_name': 'Finite planet', 'hostname': 'Host',
        'pl_bmassj': '0.05', 'pl_radj': '0.12', 'pl_bmassprov': 'Msin(i)',
        'pl_orbper': '8.0', 'st_mass': '0.9',
    }
    candidate = dict(control, pl_name='Nonfinite planet')
    candidate[field] = invalid
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(control))
        writer.writeheader()
        writer.writerows([candidate, control])
    original = path.read_bytes()

    records = fetch_confirmed_exoplanets(cache_path=path)

    if field in {'pl_bmassj', 'pl_radj'}:
        assert [record.name for record in records] == ['Finite planet']
    else:
        assert [record.name for record in records] == ['Nonfinite planet', 'Finite planet']
        assert getattr(records[0], 'period_days' if field == 'pl_orbper' else 'host_mass_solar') is None
    assert records[-1].mass_jupiter == 0.05
    assert records[-1].radius_jupiter == 0.12
    assert records[-1].period_days == 8.0
    assert records[-1].host_mass_solar == 0.9
    assert records[-1].mass_provenance == 'Msin(i)'
    assert path.read_bytes() == original
    if field in {'pl_bmassj', 'pl_radj'}:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(control))
            writer.writeheader()
            writer.writerow(candidate)
        invalid_only = path.read_bytes()
        with pytest.raises(ValueError, match='finite positive mass and radius'):
            fetch_confirmed_exoplanets(cache_path=path)
        assert path.read_bytes() == invalid_only