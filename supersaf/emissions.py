"""In-flight emission indices: Boeing FFM2, PoliTO S2F2 and the SAF nvPM model.

Notation (as in Fusaro et al. 2021):
    station 1 = free stream, station 2 = compressor inlet, station 3 = combustor inlet
    theta = T_amb / 288.15,  delta = p_amb / 101325   (ambient, static)
    wf    = fuel flow per engine [kg/s]

The idea of every fuel-flow method: an engine at altitude with combustor inlet
temperature T3 emits like the same engine at sea level with the same T3. So we
(a) find the sea-level fuel flow that gives that T3, (b) read the sea-level EI
from the certification curve, (c) correct the EI for the different combustor
pressure (and humidity) at altitude.

Subsonic flight uses the original Boeing FFM2. Supersonic flight uses S2F2,
which replaces the isentropic intake of FFM2 by a normal shock plus the
empirical corrections kp (pressure) and kt (temperature) of a real intake.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import theta_delta
from .constants import GAMMA
from .engines import Engine
from .fuels import Fuel, JETA1

# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


def humidity_factor(specific_humidity: float = 0.0) -> float:
    """Boeing humidity correction exponent H (DuBois & Paynter 2006).

    H = -19 (omega - 0.00634), omega = specific humidity [kg/kg].
    0.00634 is the ICAO reference humidity, so H = 0 at certification
    conditions. In the dry stratosphere omega ~ 0 and e^H ~ 1.13.
    """
    return -19.0 * (specific_humidity - 0.00634)


def kp(mach):
    """Fusaro Eq. 2: intake pressure-recovery gain vs a normal shock (1 for M <= 1)."""
    m = np.asarray(mach, dtype=float)
    return np.where(m <= 1.0, 1.0, 0.08 * m**2 + 0.06 * m + 0.86)


def kt(mach):
    """Fusaro Eq. 3: intake temperature correction vs a normal shock (1 for M <= 1)."""
    m = np.asarray(mach, dtype=float)
    return np.where(m <= 1.0, 1.0, 0.055 * m**2 - 0.395 * m + 1.34)


def normal_shock_ratios(mach):
    """Static pressure and temperature ratios across a normal shock (Fusaro Eqs. 4-5
    without kp, kt). Returns (p2/p1, T2/T1)."""
    m2 = np.asarray(mach, dtype=float) ** 2
    p_ratio = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (m2 - 1.0)
    t_ratio = p_ratio * (2.0 + (GAMMA - 1.0) * m2) / ((GAMMA + 1.0) * m2)
    return p_ratio, t_ratio


def saf_nvpm_change_pct(delta_H, thrust_pct, printed_form: bool = False):
    """Teoh et al. (2022) Eq. 1: % change of nvPM number EI when switching fuel.

    delta_H    : hydrogen content difference vs Jet A-1 [% points] (1.5 for SAF100)
    thrust_pct : engine thrust setting F/F00 [%]

    Brem et al. (2015) linear law for small delta_H, with an exponential
    damping above 0.5 % so the reduction does not exceed -100 %.

    IMPORTANT - typo in the printed equation. The paper prints the damping as
    exp(0.5 - dH). That form is not monotonic (SAF100 would cut soot LESS than
    SAF70) and does not reproduce the paper's own Table 2. The authors' code
    (pycontrails, ``nvpm_number_ei_pct_reduction_due_to_saf``) uses
    exp(0.5 * (0.5 - dH)) and clips the result to [-90 %, 0 %]; with it, a
    thrust setting of ~54 % reproduces Table 2 for every blend (SAF10 -8.5 %,
    SAF50 -37 %, SAF70 -45 %, SAF100 -51 %). We use the code's form by
    default; ``printed_form=True`` gives the equation exactly as printed.
    """
    a0, a1 = -114.21, 1.06
    dh = np.asarray(delta_H, dtype=float)
    f = np.clip(np.asarray(thrust_pct, dtype=float), 0.0, 100.0)
    base = (a0 + a1 * f) * dh
    if printed_form:
        return np.where(dh <= 0.5, base, base * np.exp(0.5 - dh))
    damped = np.where(dh <= 0.5, base, base * np.exp(0.5 * (0.5 - dh)))
    return np.clip(damped, -90.0, 0.0)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EmissionIndices:
    """Emission indices per kg of the fuel actually burned."""

    method: str
    wf_sl: np.ndarray            # sea-level-equivalent fuel flow per engine [kg/s]
    thrust_fraction: np.ndarray  # F/F00 [-]
    ei_nox: np.ndarray           # g/kg
    ei_co: np.ndarray            # g/kg
    ei_hc: np.ndarray            # g/kg
    ei_nvpm_n: np.ndarray        # #/kg
    nvpm_change_pct: np.ndarray  # SAF effect vs Jet A-1 at the same thrust [%]


# ---------------------------------------------------------------------------
# Boeing Fuel Flow Method 2 (subsonic)
# ---------------------------------------------------------------------------


def ffm2(engine: Engine, wf_alt, h_m, mach, fuel: Fuel = JETA1,
         specific_humidity: float = 0.0) -> EmissionIndices:
    """Boeing FFM2 (DuBois & Paynter 2006; Teoh Eq. 2 for the fuel flow).

    wf_SL  = wf_alt * theta^3.8 / delta * exp(0.2 M^2)
    EI_HC,CO  = EI_SL(wf_SL) * theta^3.3 / delta^1.02
    EI_NOx    = EI_SL(wf_SL) * (delta^1.02 / theta^3.3)^0.5 * e^H
    """
    theta, delta = theta_delta(h_m)
    wf_sl = np.asarray(wf_alt, dtype=float) * theta**3.8 / delta * np.exp(0.2 * np.asarray(mach) ** 2)
    corr = theta**3.3 / delta**1.02
    e_h = np.exp(humidity_factor(specific_humidity))
    return _finish(engine, fuel, "FFM2", wf_sl,
                   ei_nox=engine.ei_sea_level("nox", wf_sl) * (1.0 / corr) ** 0.5 * e_h,
                   ei_co=engine.ei_sea_level("co", wf_sl) * corr,
                   ei_hc=engine.ei_sea_level("hc", wf_sl) * corr)


# ---------------------------------------------------------------------------
# Sustainable Supersonic Fuel Flow method (S2F2, Fusaro et al. 2021)
# ---------------------------------------------------------------------------


def s2f2_factors(h_m, mach):
    """The three pieces S2F2 needs: (theta, delta), kp*p2/p1 and kt*T2/T1."""
    theta, delta = theta_delta(h_m)
    p_ratio, t_ratio = normal_shock_ratios(mach)
    return theta, delta, kp(mach) * p_ratio, kt(mach) * t_ratio


def s2f2(engine: Engine, wf_alt, h_m, mach, fuel: Fuel = JETA1,
         x: float = 1.0, y: float = 0.4, r_bleed: float = 1.0,
         specific_humidity: float = 0.0) -> EmissionIndices:
    """S2F2 written for the fuel actually burned (our 'explicit' form).

    ``wf_alt`` is the real fuel flow of ``fuel``. For SAF this is already
    k_blend times the Jet A-1 flow because less mass gives the same energy.

    Eq. 10:  wf_SL = r_bleed wf_alt theta^3.8/delta * p1/(kp p2) * (kt T2/T1)^2.95 * (1+0.2M^2)
    Eq. 7-8: EI_CO,HC  = EI_SL [theta^3.3/delta^1.02 * 1/(kp p2/p1) * (kt T2/T1)^3.3]^x
    Eq. 9:   EI_NOx    = EI_SL [delta^1.02/theta^3.3 * (kp p2/p1) * (kt T2/T1)^-3.3]^y e^H

    The paper's SAF form (Eqs. 14-17) multiplies wf_SL and every EI by
    k_blend and keeps the Jet A-1 fuel flow; ``s2f2_paper_form`` implements it
    literally and the tests check that both give the same grams emitted.
    """
    theta, delta, kp_pr, kt_tr = s2f2_factors(h_m, mach)
    m = np.asarray(mach, dtype=float)
    wf_sl = (r_bleed * np.asarray(wf_alt, dtype=float) * theta**3.8 / delta
             * (1.0 / kp_pr) * kt_tr**2.95 * (1.0 + 0.2 * m**2))
    bracket_co_hc = theta**3.3 / delta**1.02 * (1.0 / kp_pr) * kt_tr**3.3
    e_h = np.exp(humidity_factor(specific_humidity))
    return _finish(engine, fuel, "S2F2", wf_sl,
                   ei_nox=engine.ei_sea_level("nox", wf_sl) * (1.0 / bracket_co_hc) ** y * e_h,
                   ei_co=engine.ei_sea_level("co", wf_sl) * bracket_co_hc**x,
                   ei_hc=engine.ei_sea_level("hc", wf_sl) * bracket_co_hc**x)


def s2f2_paper_form(engine: Engine, wf_jet_equivalent, h_m, mach, fuel: Fuel,
                    x: float = 1.0, y: float = 0.4, r_bleed: float = 1.0,
                    specific_humidity: float = 0.0) -> dict:
    """Fusaro Eqs. 14-17 implemented literally (k_blend on fuel flow and EIs).

    Here the EIs are per kg of *Jet A-1-equivalent* fuel, so grams emitted
    are EI * wf_jet_equivalent.
    """
    k = fuel.k_blend
    theta, delta, kp_pr, kt_tr = s2f2_factors(h_m, mach)
    m = np.asarray(mach, dtype=float)
    wf_sl = (r_bleed * wf_jet_equivalent * k * theta**3.8 / delta
             * (1.0 / kp_pr) * kt_tr**2.95 * (1.0 + 0.2 * m**2))           # Eq. 14
    bracket = theta**3.3 / delta**1.02 * (1.0 / kp_pr) * kt_tr**3.3
    e_h = np.exp(humidity_factor(specific_humidity))
    return dict(
        wf_sl=wf_sl,
        ei_co=engine.ei_sea_level("co", wf_sl) * bracket**x * k,             # Eq. 15
        ei_hc=engine.ei_sea_level("hc", wf_sl) * bracket**x * k,             # Eq. 16
        ei_nox=engine.ei_sea_level("nox", wf_sl) * (1 / bracket) ** y * e_h * k,  # Eq. 17
    )


def cruise_indices(engine: Engine, wf_alt, h_m, mach, fuel: Fuel = JETA1,
                   **kwargs) -> EmissionIndices:
    """Pick the method like Fusaro et al.: FFM2 below Mach 1, S2F2 above."""
    if np.all(np.asarray(mach) >= 1.0):
        return s2f2(engine, wf_alt, h_m, mach, fuel, **kwargs)
    kwargs.pop("x", None), kwargs.pop("y", None), kwargs.pop("r_bleed", None)
    return ffm2(engine, wf_alt, h_m, mach, fuel, **kwargs)


def lto_indices(engine: Engine, mode_weights: dict, fuel: Fuel = JETA1) -> dict:
    """Fuel-weighted EIs over LTO modes (used for taxi/take-off/climb/descent).

    ``mode_weights`` maps an LTO mode name to its time-in-mode [min]. With SAF
    the engine burns k_blend times less fuel mass at the same thrust, so the
    EI is read at the lower sea-level fuel flow (same convention as S2F2).
    """
    modes = ["idle", "approach", "climb-out", "take-off"]
    # map mode name -> fuel flow (EDB engines have 4 modes; Olympus has 5, take 4)
    flows = _mode_fuel_flows(engine)
    fuel_mass = {m: flows[m] * fuel.k_blend * 60.0 * mode_weights.get(m, 0.0) for m in modes}
    total = sum(fuel_mass.values())
    out = {}
    for species in ("nox", "co", "hc"):
        out[f"ei_{species}"] = sum(
            fuel_mass[m] * float(engine.ei_sea_level(species, flows[m] * fuel.k_blend))
            for m in modes) / total
    thrust = {m: float(engine.thrust_fraction(flows[m])) for m in modes}
    out["ei_nvpm_n"] = sum(
        fuel_mass[m] * float(engine.nvpm_ein_jet(thrust[m]))
        * (1 + float(saf_nvpm_change_pct(fuel.delta_H, 100 * thrust[m])) / 100)
        for m in modes) / total
    return out


# ICAO LTO cycle times-in-mode [min]
ICAO_LTO_MINUTES = {"take-off": 0.7, "climb-out": 2.2, "approach": 4.0, "idle": 26.0}


def _mode_fuel_flows(engine: Engine) -> dict:
    """Fuel flow [kg/s] for the four standard ICAO modes."""
    names = ["idle", "approach", "climb-out", "take-off"]
    if len(engine.fuel_flow) == 4:
        return dict(zip(names, engine.fuel_flow))
    # Olympus table has an extra 'descent' point between idle and approach
    ff = engine.fuel_flow
    return {"idle": ff[0], "approach": ff[2], "climb-out": ff[3], "take-off": ff[4]}


def _finish(engine: Engine, fuel: Fuel, method: str, wf_sl, ei_nox, ei_co, ei_hc) -> EmissionIndices:
    """Add thrust setting and nvPM (Teoh Eq. 1) to the gaseous EIs."""
    # Thrust setting F/F00 ~ fuel-flow ratio for Jet A-1. With SAF the
    # sea-level fuel flow is k_blend lower for the SAME thrust, so we divide
    # by k_blend to recover the thrust setting (otherwise SAF would look like
    # a lower power setting).
    thrust = engine.thrust_fraction(np.asarray(wf_sl) / fuel.k_blend)
    change = saf_nvpm_change_pct(fuel.delta_H, 100.0 * thrust)
    ein = engine.nvpm_ein_jet(thrust) * (1.0 + change / 100.0)
    return EmissionIndices(method=method, wf_sl=np.asarray(wf_sl), thrust_fraction=thrust,
                           ei_nox=np.asarray(ei_nox), ei_co=np.asarray(ei_co),
                           ei_hc=np.asarray(ei_hc), ei_nvpm_n=np.asarray(ein),
                           nvpm_change_pct=np.asarray(change))
