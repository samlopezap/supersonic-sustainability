"""Contrail formation (Schmidt-Appleman criterion) and persistence.

Physics in three sentences
--------------------------
The hot, humid exhaust mixes with cold ambient air along a straight line in a
(temperature, water-vapour-pressure) diagram whose slope G depends on the fuel
and the engine. A contrail FORMS if that mixing line crosses liquid-water
saturation (droplets form, then freeze): the Schmidt-Appleman criterion (SAC).
It PERSISTS only if the ambient air is ice-supersaturated (RHi > 100 %);
otherwise the ice crystals sublimate within seconds to minutes.

    G = EI_H2O * cp * p / (epsilon * Q * (1 - eta))           [Pa/K]

Higher EI_H2O (SAF!) or higher pressure give a steeper line -> contrails
form at warmer temperatures. Low pressure at supersonic cruise altitudes
gives a flat line -> it must be MUCH colder for a contrail to form.
"""
from __future__ import annotations

import numpy as np

from .constants import CP_AIR, EPSILON
from .fuels import Fuel

# ---------------------------------------------------------------------------
# Saturation vapour pressures (Sonntag 1994, the formulas used in CoCiP)
# ---------------------------------------------------------------------------


def e_sat_liquid(T):
    """Saturation vapour pressure over (supercooled) liquid water [Pa], T in K."""
    T = np.asarray(T, dtype=float)
    return 100.0 * np.exp(-6096.9385 / T + 16.635794 - 0.02711193 * T
                          + 1.673952e-5 * T**2 + 2.433502 * np.log(T))


def e_sat_ice(T):
    """Saturation vapour pressure over ice [Pa], T in K."""
    T = np.asarray(T, dtype=float)
    return 100.0 * np.exp(-6024.5282 / T + 24.7219 + 0.010613868 * T
                          - 1.3198825e-5 * T**2 - 0.49382577 * np.log(T))


def e_sat_liquid_prime(T, dT: float = 1e-3):
    """d e_sat_liquid / dT [Pa/K] (central difference, accurate to ~1e-6)."""
    return (e_sat_liquid(T + dT) - e_sat_liquid(T - dT)) / (2 * dT)


# ---------------------------------------------------------------------------
# Humidity conversions
# ---------------------------------------------------------------------------


def vapour_pressure(q, p):
    """Water vapour partial pressure [Pa] from specific humidity q [kg/kg] and p [Pa]."""
    q = np.asarray(q, dtype=float)
    return q * p / (EPSILON + (1.0 - EPSILON) * q)


def q_from_vmr(vmr_ppmv):
    """Specific humidity [kg/kg] from a volume mixing ratio [ppmv]."""
    x = np.asarray(vmr_ppmv, dtype=float) * 1e-6
    return EPSILON * x / (1.0 - (1.0 - EPSILON) * x)


def rhi(T, p, q):
    """Relative humidity with respect to ice [-]."""
    return vapour_pressure(q, p) / e_sat_ice(T)


# ---------------------------------------------------------------------------
# Schmidt-Appleman criterion
# ---------------------------------------------------------------------------


def mixing_line_slope(p, fuel: Fuel, eta: float):
    """G [Pa/K] (Schumann 1996)."""
    return fuel.ei_h2o * CP_AIR * np.asarray(p, dtype=float) / (EPSILON * fuel.lhv * (1.0 - eta))


def t_lm(G):
    """Temperature where the mixing line is tangent to liquid saturation [K].

    Schumann (1996) approximation, valid for 0.24 < G < 15 Pa/K. This is the
    threshold for air saturated with respect to LIQUID water (RH_liquid = 100 %).
    """
    x = np.log(np.asarray(G, dtype=float) - 0.053)
    return -46.46 + 9.43 * x + 0.720 * x**2 + 273.15


def sac_threshold(p, fuel: Fuel, eta: float, rh_liquid=0.0, n_iter: int = 30):
    """Maximum ambient temperature [K] at which a contrail forms.

    Solves T = T_LM - (e_L(T_LM) - U e_L(T)) / G for the ambient relative
    humidity over liquid U (Newton iterations, as in CoCiP). For dry air
    (U = 0) this is explicit: T_crit = T_LM - e_L(T_LM) / G.
    The stratosphere is so dry (U ~ 1-3 %) that the dry value is what matters.
    """
    G = mixing_line_slope(p, fuel, eta)
    tlm = t_lm(G)
    e_lm = e_sat_liquid(tlm)
    u = np.asarray(rh_liquid, dtype=float)
    T = tlm - e_lm / G   # dry-air value, also a good first guess
    for _ in range(n_iter):
        f = T - tlm + (e_lm - u * e_sat_liquid(T)) / G
        fprime = 1.0 - u * e_sat_liquid_prime(T) / G
        T = T - f / fprime
    return T


def sac_satisfied(T, p, q, fuel: Fuel, eta: float):
    """True where a contrail forms (exact geometric form of the SAC).

    The mixing line through the ambient point (T, e) is e + G (T' - T). Its
    largest excess over liquid saturation is at T' = T_LM, so a contrail
    forms if  T < T_LM  and  e + G (T_LM - T) > e_L(T_LM).
    """
    G = mixing_line_slope(p, fuel, eta)
    tlm = t_lm(G)
    e = vapour_pressure(q, p)
    return (np.asarray(T) < tlm) & (e + G * (tlm - np.asarray(T)) > e_sat_liquid(tlm))


def persistent_contrail(T, p, q, fuel: Fuel, eta: float, rhi_threshold: float = 1.0):
    """SAC satisfied AND ambient air ice-supersaturated."""
    return sac_satisfied(T, p, q, fuel, eta) & (rhi(T, p, q) >= rhi_threshold)


def frost_point(e_pa, n_iter: int = 40):
    """Temperature [K] at which vapour pressure ``e_pa`` would saturate over ice.

    T - frost_point(e) is how much the air must cool to become ice-saturated:
    a direct measure of how far it is from supporting persistent contrails.
    (Newton iterations on ln e_sat_ice.)
    """
    e = np.asarray(e_pa, dtype=float)
    T = np.full_like(e, 200.0)
    for _ in range(n_iter):
        f = np.log(e_sat_ice(T)) - np.log(e)
        dfdT = (np.log(e_sat_ice(T + 1e-3)) - np.log(e_sat_ice(T - 1e-3))) / 2e-3
        T = T - f / dfdT
    return T


def humidity_correction_teoh(rhi_values, rhi_adj: float = 0.97, boost: float = 1.7,
                             cap: float = 1.7):
    """ERA5 humidity correction of Teoh et al. (2022, ACP), as implemented in
    pycontrails (ExponentialBoostHumidityScaling).

        RHi -> RHi / rhi_adj;  if > 1: RHi -> min(RHi ** boost, cap)

    It was derived from aircraft measurements in the UPPER TROPOSPHERE
    (IAGOS); CoCiP applies no humidity scaling in the stratosphere. We only
    use it as a sensitivity test.
    """
    r = np.asarray(rhi_values, dtype=float) / rhi_adj
    return np.where(r > 1.0, np.minimum(r**boost, cap), r)
