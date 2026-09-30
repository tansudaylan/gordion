"""Pixel-level variability summaries for astronomical image cubes."""

from tdpy.verbosity import print

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_variability_products(flux_cube: np.ndarray) -> dict[str, np.ndarray]:
    """Return temporal median and robust relative scatter maps.

    Parameters
    ----------
    flux_cube
        Flux measurements with shape ``(time, y, x)``.
    """

    flux_cube = np.asarray(flux_cube, dtype=float)
    if flux_cube.ndim != 3:
        raise ValueError("flux_cube must have shape (time, y, x).")
    if flux_cube.shape[0] < 2:
        raise ValueError("flux_cube must contain at least two time samples.")

    masked_flux = np.ma.masked_invalid(flux_cube)
    temporal_median = np.ma.median(masked_flux, axis=0).filled(np.nan)
    valid_baseline = np.isfinite(temporal_median) & (temporal_median != 0.0)

    relative_residual = np.full(flux_cube.shape, np.nan)
    np.divide(
        flux_cube - temporal_median,
        temporal_median,
        out=relative_residual,
        where=np.isfinite(flux_cube) & valid_baseline,
    )
    masked_residual = np.ma.masked_invalid(relative_residual)
    residual_median = np.ma.median(masked_residual, axis=0)
    relative_mad = np.ma.median(
        np.ma.abs(masked_residual - residual_median), axis=0
    ).filled(np.nan)

    return {
        "temporal_median": temporal_median,
        "relative_variability_percent": 100.0 * 1.4826 * relative_mad,
    }


def plot_variability_summary(
    time: np.ndarray,
    flux_cube: np.ndarray,
    output_path: Path,
    title: str,
) -> dict[str, np.ndarray]:
    """Plot the input image, robust variability, and strongest pixel series."""

    time = np.asarray(time, dtype=float)
    flux_cube = np.asarray(flux_cube, dtype=float)
    if time.ndim != 1 or time.size != flux_cube.shape[0]:
        raise ValueError("time must match the first axis of flux_cube.")
    if output_path.suffix not in {".png", ".pdf"}:
        raise ValueError("output_path must end in .png or .pdf.")

    products = compute_variability_products(flux_cube)
    variability = products["relative_variability_percent"]
    if not np.isfinite(variability).any():
        raise ValueError("flux_cube contains no pixel with a valid baseline.")
    temporal_median = products["temporal_median"]
    finite_median = temporal_median[np.isfinite(temporal_median)]
    background_level = np.median(finite_median)
    background_scatter = 1.4826 * np.median(
        np.abs(finite_median - background_level)
    )
    source_mask = temporal_median > background_level + 3.0 * background_scatter
    if not source_mask.any():
        source_mask = np.isfinite(temporal_median)
    source_variability = np.where(source_mask, variability, np.nan)
    pixel_y, pixel_x = np.unravel_index(
        np.nanargmax(source_variability), source_variability.shape
    )
    pixel_flux = flux_cube[:, pixel_y, pixel_x]
    pixel_baseline = temporal_median[pixel_y, pixel_x]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.2),
        facecolor="white",
        constrained_layout=True,
    )
    median_image = axes[0].imshow(
        temporal_median, origin="lower", cmap="gray_r"
    )
    figure.colorbar(median_image, ax=axes[0], label=r"Median flux [electron s$^{-1}$]")
    axes[0].set_title("Temporal median image")

    variability_image = axes[1].imshow(
        source_variability,
        origin="lower",
        cmap="magma",
    )
    axes[1].scatter(
        pixel_x,
        pixel_y,
        marker="x",
        s=70,
        linewidth=2.0,
        color="white",
    )
    figure.colorbar(
        variability_image,
        ax=axes[1],
        label="Robust relative variability [%]",
    )
    axes[1].set_title("Source-pixel variability map")

    axes[2].plot(
        time,
        pixel_flux / pixel_baseline,
        color="#1B6CA8",
        linewidth=0.8,
        label=f"Highest variability pixel ({pixel_x}, {pixel_y})",
    )
    axes[2].set_xlabel("Time [BTJD]")
    axes[2].set_ylabel("Normalized flux")
    axes[2].set_title("Selected pixel time series")
    legend = axes[2].legend(frameon=True, fancybox=True, framealpha=1.0)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    for axis in axes[:2]:
        axis.set_xlabel("Detector x pixel")
        axis.set_ylabel("Detector y pixel")
    for axis in axes:
        axis.grid(False)
    figure.suptitle(title, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return products