import importlib.util
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("bokeh")
SCRIPT = Path(__file__).parents[1] / "examples" / "planetary_phase_space" / "run.py"
SPEC = importlib.util.spec_from_file_location("gordion_phase_space_example", SCRIPT)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def test_exoplanet_phase_space_bokeh_view_writes_html(tmp_path):
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

    output_path = example._plot_bokeh_exoplanet_summary(
        rows, np.array([100.0, 120.0, 150.0]), tmp_path / "catalog.html"
    )

    html = output_path.read_text(encoding="utf-8")
    assert output_path.is_file()
    assert "Planet b" in html
    assert "Mean DM speed" in html