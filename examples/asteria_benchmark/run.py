"""Compare Gordion with the separately installed Asteria Python v1.0 release."""

from __future__ import annotations

import importlib
import math

import numpy as np
from tdpy.verbosity import print

from gordion.capture import CaptureModel


class _ReferenceNumpy:
    """Supply Asteria v1.0's removed np.math alias without changing NumPy."""

    math = math

    def __getattr__(self, name: str):
        return getattr(np, name)


def main() -> None:
    try:
        asteria = importlib.import_module("asteria")
    except ModuleNotFoundError as error:
        if error.name != "asteria":
            raise
        raise SystemExit(
            "Download Asteria Python v1.0 (DOI 10.5281/zenodo.8150110) "
            "and put the directory containing asteria.py on PYTHONPATH. "
            "Asteria is needed only for this comparison example."
        ) from error

    if not hasattr(np, "math"):
        asteria.np = _ReferenceNumpy()

    composition = ((0.75, 1.0), (0.25, 4.0))
    asteria.set_elements(composition)
    model = CaptureModel(composition)
    radius_m = 7.0e7  # [m]
    mass_kg = 1.898e27  # [kg]
    density_gev_cm3 = 0.4  # [GeV/cm^3]
    speed_m_s = 270000.0  # [m/s]
    nucleon_mass_gev = 0.94  # [GeV]
    dm_mass_gev = 10.0  # [GeV]

    print("Asteria Python v1.0 comparison, Jupiter with 75% H and 25% He by mass")
    print(
        "convention,sigma_cm2,gordion_regime,gordion_s-1,asteria_s-1,relative_difference"
    )
    for scaled, sigmas in (
        (True, (1e-38, 1e-34, 1e-30)),
        (False, (1e-38, 1e-32, 1e-28)),
    ):
        reference_function = (
            asteria.multicapture if scaled else asteria.multicapture_noA
        )
        for sigma_cm2 in sigmas:
            result = model.calculate(
                radius_m,
                mass_kg,
                density_gev_cm3,
                speed_m_s,
                dm_mass_gev,
                sigma_cm2,
                nucleon_mass_gev,
                scale_from_nucleon=scaled,
            )
            reference = float(
                reference_function(
                    radius_m,
                    mass_kg,
                    density_gev_cm3,
                    speed_m_s,
                    nucleon_mass_gev,
                    sigma_cm2,
                    dm_mass_gev,
                )
            )
            if not np.isfinite(reference) or reference <= 0.0:
                raise ValueError(
                    "Asteria returned a nonpositive or nonfinite reference rate"
                )
            difference = (result.rate_per_second - reference) / reference
            convention = "nucleon-scaled" if scaled else "nucleus-level"
            print(
                f"{convention},{sigma_cm2:.2e},{result.regime},"
                f"{result.rate_per_second:.6e},{reference:.6e},{difference:.6e}"
            )


if __name__ == "__main__":
    main()
