"""Self-contained Gordion dark-matter capture-rate calculations.

The reference implementation is Leane & Smirnov (2023), Zenodo DOI
10.5281/zenodo.8150110, licensed CC-BY 4.0. This implementation removes the
reference package's mutable global composition and preserves its units: radius
[m], mass [kg], density [GeV/cm^3], speed [m/s], masses [GeV], cross section
[cm^2], and capture rate [s^-1].
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, expm1, log, log1p, pi, sqrt
from typing import Iterable

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc


G = 6.67430e-11  # [m^3 kg^-1 s^-2]
GEV_TO_KG = 1.78266192e-27  # [kg/GeV]


@dataclass(frozen=True)
class NuclearSpecies:
    """One nuclear species with its mass fraction and mass number."""

    mass_fraction: float
    mass_number: float
    name: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass_fraction) or self.mass_fraction <= 0.0:
            raise ValueError("mass_fraction must be finite and positive")
        if not np.isfinite(self.mass_number) or self.mass_number <= 0.0:
            raise ValueError("mass_number must be finite and positive")


def _composition(elements: Iterable[NuclearSpecies | tuple[float, float]]) -> tuple[NuclearSpecies, ...]:
    result = tuple(
        item if isinstance(item, NuclearSpecies) else NuclearSpecies(*item)
        for item in elements
    )
    if not result:
        raise ValueError("at least one nuclear species is required")
    if sum(item.mass_fraction for item in result) > 1.0 + 1e-8:
        raise ValueError("element mass fractions cannot sum to more than one")
    return result


def escape_speed(radius_m: float, mass_kg: float) -> float:
    """Return surface escape speed for a spherical body in m/s."""

    if not np.isfinite(radius_m) or not np.isfinite(mass_kg) or min(radius_m, mass_kg) <= 0.0:
        raise ValueError("radius_m and mass_kg must be finite and positive")
    return sqrt(2.0 * G * mass_kg / radius_m)


def geometric_capture_rate(
    radius_m: float,
    mass_kg: float,
    density_gev_cm3: float,
    speed_m_s: float,
    dm_mass_gev: float,
) -> float:
    """Return the Asteria-convention effective-speed geometric rate in s^-1.

    ``speed_m_s`` is the RMS effective speed. The factor 3/2 in the focusing
    term follows the Maxwellian moment relation; the flux prefactor uses the
    RMS speed in place of the exact mean speed for rate-model parity.
    """

    _validate_inputs(radius_m, mass_kg, density_gev_cm3, speed_m_s, dm_mass_gev, 1.0)
    focusing = 1.0 + 1.5 * (escape_speed(radius_m, mass_kg) / speed_m_s) ** 2
    return pi * radius_m**2 * (density_gev_cm3 / dm_mass_gev) * 1e6 * speed_m_s * focusing


def _validate_inputs(
    radius_m: float,
    mass_kg: float,
    density_gev_cm3: float,
    speed_m_s: float,
    dm_mass_gev: float,
    sigma_cm2: float,
) -> None:
    values = (radius_m, mass_kg, density_gev_cm3, speed_m_s, dm_mass_gev, sigma_cm2)
    if not np.isfinite(values).all():
        raise ValueError("capture inputs must be finite")
    if min(radius_m, mass_kg, density_gev_cm3, speed_m_s, dm_mass_gev) <= 0.0:
        raise ValueError("radius, mass, density, speed, and DM mass must be positive")
    if sigma_cm2 < 0.0:
        raise ValueError("sigma_cm2 must be nonnegative")


def _reduced_mass(first_gev: float, second_gev: float) -> float:
    return first_gev * second_gev / (first_gev + second_gev)


def _nucleus_cross_section(
    species: NuclearSpecies,
    nucleon_mass_gev: float,
    dm_mass_gev: float,
    sigma_cm2: float,
    scale_from_nucleon: bool,
) -> float:
    if not scale_from_nucleon:
        return sigma_cm2
    reduced_nucleus = _reduced_mass(species.mass_number, dm_mass_gev)
    reduced_nucleon = _reduced_mass(nucleon_mass_gev, dm_mass_gev)
    return species.mass_number**2 * (reduced_nucleus / reduced_nucleon) ** 2 * sigma_cm2


def _target_count(mass_kg: float, mass_number: float) -> float:
    return mass_kg / GEV_TO_KG / mass_number


def _transition_cross_section(radius_m: float, mass_kg: float, mass_number: float) -> float:
    return pi * (radius_m * 100.0) ** 2 / _target_count(mass_kg, mass_number)


def _beta(dm_mass_gev: float, target_mass_gev: float) -> float:
    value = 4.0 * dm_mass_gev * target_mass_gev / (dm_mass_gev + target_mass_gev) ** 2
    return min(value, 1.0 - 1e-12)


def _scatter_probability(optical_depth: float, scatters: int | float) -> float:
    if optical_depth <= 0.0 or scatters <= 0.0:
        return 0.0
    return 2.0 * (scatters + 1.0) * float(gammainc(scatters + 2.0, optical_depth)) / optical_depth**2


def _capture_probability(speed_m_s: float, escape_m_s: float, dm_mass_gev: float,
                         target_mass_gev: float, scatters: float) -> float:
    """Evaluate the capture probability without inverse-power cancellation.

    With ``x = a * (alpha**(-scatters) - 1)`` and ``a = 1.5 * (vesc/v)**2``,
    the probability is ``gammainc(2, x) + exp(-x) * x * a / (1 + a)``.
    """
    focusing = 1.5 * (escape_m_s / speed_m_s) ** 2
    mass_ratio = min(dm_mass_gev, target_mass_gev) / max(dm_mass_gev, target_mass_gev)
    fractional_loss = 2.0 * mass_ratio / (1.0 + mass_ratio) ** 2
    log_gain = -scatters * log1p(-fractional_loss)
    if log_gain < 1.0:
        exponent_size = focusing * expm1(log_gain)
    else:
        log_exponent = log(focusing) + log_gain + log1p(-exp(-log_gain))
        if log_exponent > log(np.finfo(float).max):
            return 1.0
        exponent_size = exp(log_exponent)
    probability = (gammainc(2.0, exponent_size)
                   + exp(-exponent_size) * exponent_size * (focusing / (1.0 + focusing)))
    return float(np.clip(probability, 0.0, 1.0))


def _reflection_fraction(speed_m_s: float, escape_m_s: float, mass_ratio: float) -> float:
    speed_ratio = speed_m_s / escape_m_s
    log_term = log(sqrt(1.0 + speed_ratio**2))
    transition_scatter_count = 12.0 + 1.8 * log_term
    velocity_ratio = 1.0 / (1.0 + speed_ratio**2)
    root = velocity_ratio ** (1.0 / transition_scatter_count)
    mass_transition = (sqrt(max(2.0 * root - 1.0, 0.0)) - root) / (root - 1.0)
    mass_transition = max(mass_transition, 1e-8)
    mass_heavy = max(1.56 * (1.0 - 1.0 / (1.0 + 0.52 * log_term)), mass_transition + 1e-8)
    capture_heavy = 0.22 * (1.0 + 3.58 / (1.0 + 0.23 * log_term))
    f_light = sqrt(max(0.0, (4.0 / pi) *
                        log(max(1.0 - 2.0 * mass_ratio / (1.0 + mass_ratio) ** 2, 1e-300)) /
                        log(velocity_ratio)))
    if mass_ratio < mass_transition:
        fraction = f_light
    elif mass_ratio < mass_heavy:
        fraction = (mass_transition - f_light + mass_ratio * (f_light - 1.0)) / (mass_transition - 1.0)
    else:
        fraction = mass_ratio / (mass_ratio - mass_heavy + mass_heavy / capture_heavy)
    return float(np.clip(fraction, 0.0, 1.0))


def _capture_kinematic_asymptote(speed_m_s: float, escape_m_s: float,
                                 dm_mass_gev: float, target_mass_gev: float,
                                 scatters: float, heavy: bool) -> float:
    ratio = (escape_m_s / speed_m_s) ** 2
    mu = dm_mass_gev / target_mass_gev
    if heavy:
        numerator = (
            8.0 * mu * (3.0 * (mu - 1.0) * mu + 1.0)
            + scatters**3 * (64.0 - 3.0 * (9.0 * ratio**2 * (ratio - 6.0) + 76.0 * ratio))
            + 4.0 * (mu - 3.0) * scatters**2 * (9.0 * ratio**2 - 30.0 * ratio + 16.0)
            - 4.0 * scatters * (3.0 * (mu - 2.0) * mu + 5.0) * (3.0 * ratio - 4.0)
        )
        denominator = 8.0 * mu**4 * (3.0 * escape_m_s**2 * speed_m_s**2 + 2.0 * speed_m_s**4)
        numerator *= 3.0 * scatters * escape_m_s**4
    else:
        numerator = (
            24.0
            + mu**3 * scatters**3 * (64.0 - 3.0 * (9.0 * ratio**2 * (ratio - 6.0) + 76.0 * ratio))
            - 4.0 * mu**2 * (3.0 * mu - 1.0) * scatters**2 * (9.0 * ratio**2 - 30.0 * ratio + 16.0)
            - 4.0 * mu * (mu * (5.0 * mu - 6.0) + 3.0) * scatters * (3.0 * ratio - 4.0)
            + 8.0 * mu * (mu - 3.0)
        )
        denominator = 8.0 * (3.0 * escape_m_s**2 * speed_m_s**2 + 2.0 * speed_m_s**4)
        numerator *= 3.0 * mu * scatters * escape_m_s**4
    if denominator <= 0.0:
        return 0.0
    return min(1.0, abs(numerator / denominator))


@dataclass(frozen=True)
class CaptureResult:
    """Capture-rate calculation and regime metadata."""

    rate_per_second: float
    geometric_rate_per_second: float
    optical_depth: float
    transition_cross_section_cm2: float
    regime: str
    reflection_fraction: float


class CaptureModel:
    """Instance-scoped multi-regime capture model requiring no Asteria imports."""

    def __init__(self, composition: Iterable[NuclearSpecies | tuple[float, float]]) -> None:
        self.composition = _composition(composition)

    def optical_depth(self, radius_m: float, mass_kg: float, dm_mass_gev: float,
                      sigma_cm2: float, nucleon_mass_gev: float = 0.94,
                      scale_from_nucleon: bool = True) -> float:
        tau = 0.0
        for species in self.composition:
            sigma_target = _nucleus_cross_section(
                species, nucleon_mass_gev, dm_mass_gev, sigma_cm2, scale_from_nucleon
            )
            tau += species.mass_fraction * sigma_target / _transition_cross_section(
                radius_m, mass_kg, species.mass_number
            )
        tau *= 1.5
        if tau == 0.0:
            return 0.0
        if not scale_from_nucleon:
            target_mass = self._effective_target_mass(radius_m, mass_kg, dm_mass_gev,
                                                      sigma_cm2, nucleon_mass_gev, False)
            tau = min(tau, _target_count(mass_kg, target_mass) ** (1.0 / 3.0))
        return float(tau)

    def _effective_target_mass(self, radius_m: float, mass_kg: float, dm_mass_gev: float,
                               sigma_cm2: float, nucleon_mass_gev: float,
                               scale_from_nucleon: bool) -> float:
        """Return the optical-depth-weighted target mass used above one scatter.

        Thin-regime capture sums species rates separately. In the thick regime,
        the species optical depths are summed and their target masses are
        combined as ``sum(tau_i * A_i) / sum(tau_i)`` before evaluating the
        multiscatter capture probability. This bulk-mixture approximation does
        not follow a sequence of species-specific collisions.
        """
        weights = []
        for species in self.composition:
            sigma_target = _nucleus_cross_section(
                species, nucleon_mass_gev, dm_mass_gev, sigma_cm2, scale_from_nucleon
            )
            tau_i = 1.5 * species.mass_fraction * sigma_target / _transition_cross_section(
                radius_m, mass_kg, species.mass_number
            )
            weights.append(tau_i)
        total = sum(weights)
        return sum(weight * species.mass_number for weight, species in zip(weights, self.composition)) / total

    @staticmethod
    def _weak_species_rate(radius_m: float, mass_kg: float, density_gev_cm3: float,
                           speed_m_s: float, dm_mass_gev: float, nucleon_mass_gev: float,
                           sigma_cm2: float, species: NuclearSpecies,
                           scale_from_nucleon: bool) -> float:
        target_mass = species.mass_number
        sigma_target = _nucleus_cross_section(
            species, nucleon_mass_gev, dm_mass_gev, sigma_cm2, scale_from_nucleon
        )
        tau = 1.5 * species.mass_fraction * sigma_target / _transition_cross_section(
            radius_m, mass_kg, target_mass
        )
        if tau <= 0.0:
            return 0.0
        vesc = escape_speed(radius_m, mass_kg)
        mu = dm_mass_gev / target_mass
        geometric = geometric_capture_rate(radius_m, mass_kg, density_gev_cm3,
                                            speed_m_s, dm_mass_gev)
        mismatch = mu / (1.0 + mu**2) * (vesc / speed_m_s) ** 2
        if mismatch <= 1e-5:
            k_value = (9.0 * vesc**4 * mu) / (
                (2.0 + 3.0 * (vesc / speed_m_s) ** 2) * speed_m_s**4 * (1.0 + mu) ** 2
            )
            return geometric * sum(
                _scatter_probability(tau, number) * number * k_value
                for number in range(1, 10)
            )

        beta = _beta(dm_mass_gev, target_mass)
        beta_log = log(1.0 / (1.0 - beta))
        xibar = speed_m_s**2 / 3.0
        xiesc = vesc**2 / 2.0
        rate = 0.0
        for number in range(1, 10):
            exponent = number - 1
            ximin = xiesc * ((beta_log / beta) ** exponent - 1.0)
            ximax = xiesc * ((beta_log / beta) ** exponent / (1.0 - beta) - 1.0)
            p_value = exp(-ximin / xibar) - exp(-ximax / xibar)
            q_value = ((xibar + xiesc + ximin) * exp(-ximin / xibar)
                       - (xibar + xiesc + ximax) * exp(-ximax / xibar))
            kappa = (xiesc / beta**number) * beta_log**exponent * p_value - (1.0 / beta - 1.0) * q_value
            rate += _scatter_probability(tau, number) * max(
                0.0, pi * radius_m**2 * sqrt(8.0 / pi)
                * (density_gev_cm3 * 1e6 / dm_mass_gev) / sqrt(xibar) * kappa
            )
        return rate

    def _strong_rate(self, radius_m: float, mass_kg: float, density_gev_cm3: float,
                     speed_m_s: float, dm_mass_gev: float, nucleon_mass_gev: float,
                     sigma_cm2: float, scale_from_nucleon: bool,
                     tau: float) -> tuple[float, str, float]:
        geometric = geometric_capture_rate(radius_m, mass_kg, density_gev_cm3,
                                            speed_m_s, dm_mass_gev)
        vesc = escape_speed(radius_m, mass_kg)
        target_mass = self._effective_target_mass(radius_m, mass_kg, dm_mass_gev,
                                                  sigma_cm2, nucleon_mass_gev,
                                                  scale_from_nucleon)
        reflection = _reflection_fraction(speed_m_s, vesc, dm_mass_gev / target_mass)
        nmax = max(10, int(ceil(exp(1.0) * tau)))
        if tau <= 100.0:
            rate = geometric * sum(
                _scatter_probability(tau, number)
                * _capture_probability(speed_m_s, vesc, dm_mass_gev, target_mass, number)
                for number in range(1, nmax + 1)
            )
            return min(rate, reflection * geometric), "multiscatter", reflection

        nreq = _required_scatters(vesc, speed_m_s, dm_mass_gev, target_mass)
        mu = dm_mass_gev / target_mass

        def integrand(number: float) -> float:
            probability = _scatter_probability(tau, number)
            if tau < 1e6 and nmax > 10.0 * nreq and mu < 1e4:
                return probability
            if tau >= 1e6:
                return probability * _capture_kinematic_asymptote(
                    speed_m_s, vesc, dm_mass_gev, target_mass, number,
                    dm_mass_gev > target_mass,
                )
            return probability * _capture_probability(
                speed_m_s, vesc, dm_mass_gev, target_mass, number
            )

        rate = geometric * quad(integrand, 1.0, float(nmax), limit=500)[0]
        return min(rate, reflection * geometric), "asymptotic-multiscatter", reflection

    def calculate(self, radius_m: float, mass_kg: float, density_gev_cm3: float,
                  speed_m_s: float, dm_mass_gev: float, sigma_cm2: float,
                  nucleon_mass_gev: float = 0.94,
                  scale_from_nucleon: bool = True) -> CaptureResult:
        """Return the capture rate and kinematic-regime metadata."""

        _validate_inputs(radius_m, mass_kg, density_gev_cm3, speed_m_s,
                         dm_mass_gev, sigma_cm2)
        if not np.isfinite(nucleon_mass_gev) or nucleon_mass_gev <= 0.0:
            raise ValueError("nucleon_mass_gev must be finite and positive")
        tau = self.optical_depth(radius_m, mass_kg, dm_mass_gev, sigma_cm2,
                                 nucleon_mass_gev, scale_from_nucleon)
        geometric = geometric_capture_rate(radius_m, mass_kg, density_gev_cm3,
                                            speed_m_s, dm_mass_gev)
        if sigma_cm2 == 0.0 or tau == 0.0:
            return CaptureResult(0.0, geometric, tau, 0.0, "no-scattering", 1.0)
        opacity_per_cross_section = sum(
            item.mass_fraction
            * _nucleus_cross_section(item, nucleon_mass_gev, dm_mass_gev,
                                     1.0, scale_from_nucleon)
            / _transition_cross_section(radius_m, mass_kg, item.mass_number)
            for item in self.composition
        )
        transition = 1.0 / opacity_per_cross_section
        if tau < 1.5:
            rate = sum(
                self._weak_species_rate(radius_m, mass_kg, density_gev_cm3,
                                        speed_m_s, dm_mass_gev, nucleon_mass_gev,
                                        sigma_cm2, species, scale_from_nucleon)
                for species in self.composition
            )
            return CaptureResult(min(rate, geometric), geometric, tau, transition,
                                 "single-scatter", 1.0)
        rate, regime, reflection = self._strong_rate(
            radius_m, mass_kg, density_gev_cm3, speed_m_s, dm_mass_gev,
            nucleon_mass_gev, sigma_cm2, scale_from_nucleon, tau,
        )
        return CaptureResult(float(rate), geometric, tau, transition, regime, reflection)

    def multicapture(self, radius_m: float, mass_kg: float, density_gev_cm3: float,
                     speed_m_s: float, nucleon_mass_gev: float, sigma_cm2: float,
                     dm_mass_gev: float) -> float:
        """Use Asteria's positional API with nucleon-to-nucleus scaling."""

        return self.calculate(radius_m, mass_kg, density_gev_cm3, speed_m_s,
                              dm_mass_gev, sigma_cm2, nucleon_mass_gev,
                              scale_from_nucleon=True).rate_per_second

    def multicapture_noA(self, radius_m: float, mass_kg: float, density_gev_cm3: float,
                        speed_m_s: float, nucleon_mass_gev: float, sigma_cm2: float,
                        dm_mass_gev: float) -> float:
        """Use a supplied nucleus cross section without A-number scaling."""

        return self.calculate(radius_m, mass_kg, density_gev_cm3, speed_m_s,
                              dm_mass_gev, sigma_cm2, nucleon_mass_gev,
                              scale_from_nucleon=False).rate_per_second


AsteriaCapture = CaptureModel


def _required_scatters(escape_m_s: float, speed_m_s: float,
                       dm_mass_gev: float, target_mass_gev: float) -> float:
    mu = dm_mass_gev / target_mass_gev
    alpha = max(1.0 - 2.0 * mu / (1.0 + mu) ** 2, 1e-300)
    return log(escape_m_s**2 / (speed_m_s**2 + escape_m_s**2)) / log(alpha)
