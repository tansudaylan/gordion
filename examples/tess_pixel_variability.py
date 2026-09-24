#!/usr/bin/env python3
"""Map pixel-level variability in public TESS observations of pi Mensae."""

import argparse
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.mast import Tesscut

import gordion


TARGET_NAME = "pi Mensae"
TARGET_COORDINATE = SkyCoord(84.291188, -80.469119, unit="deg")
SECTOR = 1


def run_example(output_path: Path) -> dict[str, np.ndarray]:
    """Retrieve a public TESS cutout and plot Gordion variability products."""

    print(f"Reading from MAST Tesscut Sector {SECTOR} for {TARGET_NAME}...")
    cutouts = Tesscut.get_cutouts(
        coordinates=TARGET_COORDINATE,
        size=15,
        sector=SECTOR,
    )
    if len(cutouts) != 1:
        raise RuntimeError(f"Expected one Sector {SECTOR} cutout, received {len(cutouts)}.")

    table = cutouts[0][1].data
    quality_mask = table["QUALITY"] == 0
    time = np.asarray(table["TIME"][quality_mask], dtype=float)
    flux_cube = np.asarray(table["FLUX"][quality_mask], dtype=float)
    cutouts[0].close()

    return gordion.plot_variability_summary(
        time,
        flux_cube,
        output_path,
        title=f"TESS Sector {SECTOR} pixel variability around {TARGET_NAME}",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Gordion variability products from public TESS data."
    )
    parser.add_argument(
        "--typefileplot",
        choices=("png", "pdf"),
        default="png",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_path = Path(__file__).with_name(
        f"tess_pixel_variability.{arguments.typefileplot}"
    )
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())