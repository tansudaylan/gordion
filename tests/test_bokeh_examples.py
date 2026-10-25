import importlib.util
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("bokeh")
SCRIPT = Path(__file__).parents[1] / "examples" / "planetary_phase_space" / "run.py"
SPEC = importlib.util.spec_from_file_location("gordion_phase_space_example", SCRIPT)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_exoplanet_phase_space_bokeh_view_writes_html(tmp_path, monkeypatch, typeplotback, inherit):
    from bokeh.models import Axis, Plot
    import tdpy.plotting as plotting

    rows = [
        {
            "planet": "Planet b",
            "host": "Host A",
            "mass_earth": 4.0,
            "surface_escape_speed_km_s": 8.0,
            "mean_interior_dm_speed_km_s": 120.0,
            "mass_provenance": "measured",
        },
        {
            "planet": "Planet c",
            "host": "Host B",
            "mass_earth": 9.0,
            "surface_escape_speed_km_s": 12.0,
            "mean_interior_dm_speed_km_s": 150.0,
            "mass_provenance": "estimated",
        },
    ]

    save = plotting.save_bokeh_html
    captured = []

    def capture_and_save(figure, *args, **kwargs):
        result = save(figure, *args, **kwargs)
        background, foreground = plotting.plot_background_colors(typeplotback)
        for plot in figure.select({'type': Plot}):
            assert plot.background_fill_color == background
            assert plot.border_fill_color == background
        for axis in figure.select({'type': Axis}):
            assert axis.axis_line_color == foreground
            assert axis.major_label_text_color == foreground
        captured.append(figure)
        return result

    monkeypatch.setattr(plotting, 'save_bokeh_html', capture_and_save)
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plotting.plot_style(outer_theme):
        output_path = example._plot_bokeh_exoplanet_summary(
            rows, np.array([100.0, 120.0, 150.0]), tmp_path / "catalog.html", **options,
        )

    html = output_path.read_text(encoding="utf-8")
    assert output_path.is_file()
    assert "Planet b" in html
    assert "Mean DM speed" in html
    assert "Core mass fraction" in html
    assert "Inner density" in html
    assert len(captured) == 1


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_phase_space_cli_applies_requested_theme(monkeypatch, typeplotback):
    import sys
    from tdpy.plotting import plot_background_colors, plot_style

    calls = []

    def capture_grid(seed, tag):
        assert plot_background_colors() == (('black', 'white') if typeplotback == 'dark' else ('white', 'black'))
        calls.append((seed, tag))

    monkeypatch.setattr(example, '_plot_cmf_grid', capture_grid)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--mode', 'cmf', '--typeplotback', typeplotback])
    with plot_style('white'):
        original = dict(example.plt.rcParams)
        assert example.main() == 0
        assert dict(example.plt.rcParams) == original
        assert len(calls) == 1


def test_catalog_phase_space_covers_each_planet_and_cmf(tmp_path, monkeypatch):
    import csv
    from gordion.catalog import ExoplanetRecord

    records = [
        ExoplanetRecord('Earth analog', 'Host A', 1.0 / example.EARTH_MASSES_PER_JUPITER,
                        1.0 / example.EARTH_RADII_PER_JUPITER, 'measured', None, None),
        ExoplanetRecord('Super-Earth', 'Host B', 5.0 / example.EARTH_MASSES_PER_JUPITER,
                        1.5 / example.EARTH_RADII_PER_JUPITER, 'estimated', 10.0, 1.0),
    ]
    monkeypatch.setattr(example, '_visual_directory', lambda: tmp_path)
    monkeypatch.setattr(example, '_data_directory', lambda: tmp_path)
    assert example._plot_exoplanets(records, 12, 17, 'grid', core_mass_fractions=[0.1, 0.33, 0.7]) == 6
    with (tmp_path / 'exoplanet_phase_space_summary_grid.csv').open(newline='') as stream:
        summaries = list(csv.DictReader(stream))
    with (tmp_path / 'exoplanet_phase_space_samples_grid.csv').open(newline='') as stream:
        samples = list(csv.DictReader(stream))
    assert len(summaries) == 6 and len(samples) == 72
    expected = {(record.name, fraction) for record in records for fraction in (0.1, 0.33, 0.7)}
    assert {(row['planet'], float(row['assumed_cmf'])) for row in summaries} == expected
    assert {(row['planet'], float(row['assumed_cmf'])) for row in samples} == expected
    assert all(float(row['inner_density_kg_m3']) > float(row['outer_density_kg_m3']) for row in summaries)
    assert example._plot_exoplanets(records, 12, 17, 'default') == 2
    with (tmp_path / 'exoplanet_phase_space_samples_default.csv').open(newline='') as stream:
        original = list(csv.DictReader(stream))
    assert [row for row in samples if float(row['assumed_cmf']) == 0.33] == original


@pytest.mark.parametrize('fractions', [[], [-0.1], [1.1], [np.nan], [np.inf], [0.3, 0.3]])
def test_catalog_cmf_grid_rejects_invalid_values(fractions):
    with pytest.raises(ValueError, match='Core mass fractions'):
        example._plot_exoplanets([], 12, 17, 'invalid', core_mass_fractions=fractions)


def test_cmf_grid_themes_leave_seeded_numerical_results_unchanged(tmp_path, monkeypatch):
    import csv
    from PIL import Image
    from tdpy.plotting import plot_style

    monkeypatch.setattr(example, '_visual_directory', lambda: tmp_path)
    monkeypatch.setattr(example, '_data_directory', lambda: tmp_path)
    tables = []
    with plot_style('white'):
        for typeplotback in ('white', 'dark'):
            assert example._plot_cmf_grid(17, typeplotback, typeplotback=typeplotback) == 15
            path = tmp_path / f'cmf_profile_summary_{typeplotback}.csv'
            print(f'Reading from {path}...')
            with path.open(newline='') as stream:
                tables.append(list(csv.DictReader(stream)))
            image_path = tmp_path / f'exoplanet_cmf_density_profiles_{typeplotback}.png'
            with Image.open(image_path) as image:
                expected = (0, 0, 0) if typeplotback == 'dark' else (255, 255, 255)
                assert image.convert('RGB').getpixel((0, 0)) == expected
    assert tables[0] == tables[1]