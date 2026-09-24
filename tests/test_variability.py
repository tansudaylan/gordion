import numpy as np
import pytest
import matplotlib.image as mpimg

from gordion.variability import (
    compute_variability_products,
    plot_variability_summary,
)


def test_compute_variability_products_returns_robust_maps():
    flux_cube = np.array(
        [
            [[9.0, 20.0], [30.0, np.nan]],
            [[10.0, 20.0], [33.0, np.nan]],
            [[11.0, 20.0], [27.0, np.nan]],
        ]
    )

    products = compute_variability_products(flux_cube)

    np.testing.assert_allclose(
        products["temporal_median"][:2, :1],
        np.array([[10.0], [30.0]]),
    )
    np.testing.assert_allclose(
        products["relative_variability_percent"][:2, :1],
        np.array([[14.826], [14.826]]),
    )
    assert products["relative_variability_percent"][0, 1] == 0.0
    assert np.isnan(products["temporal_median"][1, 1])


def test_compute_variability_products_rejects_non_cube_input():
    with pytest.raises(ValueError, match="time, y, x"):
        compute_variability_products(np.ones((3, 3)))


def test_plot_variability_summary_writes_nonblank_png(tmp_path, capsys):
    time = np.arange(5, dtype=float)
    flux_cube = np.full((5, 3, 3), 100.0)
    flux_cube[:, 1, 2] = np.array([80.0, 100.0, 120.0, 100.0, 80.0])
    output_path = tmp_path / "variability.png"

    products = plot_variability_summary(
        time,
        flux_cube,
        output_path,
        title="Controlled variability diagnostic",
    )

    image = mpimg.imread(output_path)
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    assert np.nanargmax(products["relative_variability_percent"]) == 5
    assert f"Writing to {output_path}..." in capsys.readouterr().out