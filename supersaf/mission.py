"""Mission fuel burn and emissions for one aircraft, one fuel and one route.

The mission is split into four segments:

    taxi + take-off -> climb/accelerate -> CRUISE-CLIMB -> descent + landing

Non-cruise segments use textbook weight fractions (W_after / W_before).
The cruise is integrated step by step with the Breguet equation

    dW/ds = - c g W / (V L/D)

so that at each step we know the weight, altitude, speed and fuel flow per
engine - exactly the inputs the emission methods need.

How SAF enters: SAF has a higher heating value, so the same thrust needs
k_blend = LHV_jet/LHV_SAF times less fuel MASS. We therefore multiply the
TSFC and the fuel burned in every segment by k_blend. The aircraft gets a
tiny bit lighter, and the emission indices are then evaluated per kg of
the fuel really burned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aircraft import Aircraft
from .atmosphere import isa
from .constants import EI_CO2, FT, G0
from .emissions import ICAO_LTO_MINUTES, cruise_indices, lto_indices
from .fuels import Fuel

PAX_MASS_KG = 100.0   # passenger + baggage


@dataclass
class MissionResult:
    aircraft: Aircraft
    fuel: Fuel
    distance_km: float
    passengers: float
    takeoff_weight_kg: float
    fuel_by_segment_kg: dict            # taxi_takeoff, climb, cruise, descent
    emissions_kg: dict                  # per segment group: {"cruise": {...}, "other": {...}}
    cruise: dict                        # step-by-step cruise arrays

    # ---- convenience totals ------------------------------------------------
    @property
    def fuel_kg(self) -> float:
        return sum(self.fuel_by_segment_kg.values())

    @property
    def pkm(self) -> float:
        return self.passengers * self.distance_km

    def total(self, species: str) -> float:
        return self.emissions_kg["cruise"][species] + self.emissions_kg["other"][species]

    def per_pkm(self, species: str, part: str = "total") -> float:
        if species == "fuel":
            value = self.fuel_kg
        elif part == "total":
            value = self.total(species)
        else:
            value = self.emissions_kg[part][species]
        return value / self.pkm

    def cruise_mean(self, key: str) -> float:
        """Fuel-weighted cruise average of a cruise array (e.g. 'ei_nox')."""
        w = self.cruise["fuel_kg"]
        return float(np.sum(self.cruise[key] * w) / np.sum(w))

    @property
    def pax_miles_per_us_gallon(self) -> float:
        """Classic fuel-economy figure (used to check Concorde: 15.8 at full load)."""
        litres = self.fuel_kg / 0.804
        return self.passengers * self.distance_km / 1.609344 / (litres / 3.785411784)


def fly_mission(ac: Aircraft, fuel: Fuel, distance_km: float, load_factor: float = 0.8,
                n_steps: int = 200, s2f2_kwargs: dict | None = None,
                tsfc_factor: float = 1.0, ld_factor: float = 1.0) -> MissionResult:
    """Fly ``ac`` on a route of ``distance_km`` with ``fuel``.

    ``tsfc_factor`` and ``ld_factor`` scale the aircraft's TSFC and L/D (used
    by the sensitivity analysis).
    """
    s2f2_kwargs = s2f2_kwargs or {}
    k = fuel.k_blend
    passengers = ac.seats * load_factor
    payload = passengers * PAX_MASS_KG
    w_land = (ac.oew_kg + payload) * (1.0 + ac.reserve_frac)

    # 1) weight ratios of the non-cruise segments for this fuel
    r_to = 1.0 - (1.0 - ac.f_taxi_takeoff) * k
    r_climb = 1.0 - (1.0 - ac.f_climb) * k
    r_desc = 1.0 - (1.0 - ac.f_descent_landing) * k

    # 2) cruise-climb grid: altitude rises linearly with distance
    cruise_km = distance_km - ac.climb_distance_km - ac.descent_distance_km
    if cruise_km <= 0:
        raise ValueError("route shorter than climb + descent distance")
    s = np.linspace(0.0, cruise_km * 1e3, n_steps + 1)           # node positions [m]
    h_nodes = np.linspace(ac.h_cruise_start_ft, ac.h_cruise_end_ft, n_steps + 1) * FT
    _, _, _, a_nodes = isa(h_nodes)
    v_nodes = ac.mach * a_nodes
    c = ac.tsfc_kg_Ns * tsfc_factor * k                          # actual-fuel TSFC
    ld = ac.lift_to_drag * ld_factor
    # d(ln W)/ds = -c g / (V L/D): integrate backwards from the end of cruise
    dlnw_ds = c * G0 / (v_nodes * ld)
    seg_int = 0.5 * (dlnw_ds[1:] + dlnw_ds[:-1]) * np.diff(s)    # trapezoid per step
    w_end = w_land / r_desc
    lnw = np.log(w_end) + np.concatenate([np.cumsum(seg_int[::-1])[::-1], [0.0]])
    w_nodes = np.exp(lnw)
    w_start = w_nodes[0]
    w0 = w_start / (r_to * r_climb)

    fuel_seg = dict(
        taxi_takeoff=w0 * (1.0 - r_to),
        climb=w0 * r_to * (1.0 - r_climb),
        cruise=w_start - w_end,
        descent=w_end * (1.0 - r_desc),
    )

    # 3) cruise emissions, step by step (values at step mid-points)
    w_mid = 0.5 * (w_nodes[1:] + w_nodes[:-1])
    h_mid = 0.5 * (h_nodes[1:] + h_nodes[:-1])
    v_mid = 0.5 * (v_nodes[1:] + v_nodes[:-1])
    thrust_total = w_mid * G0 / ld                                # N  (thrust = drag = W/(L/D))
    wf_engine = c * thrust_total / ac.n_engines                  # kg/s per engine
    fuel_step = -np.diff(w_nodes)                                 # kg per step
    ei = cruise_indices(ac.engine, wf_engine, h_mid, ac.mach, fuel, **s2f2_kwargs)

    cruise_em = dict(
        nox=np.sum(ei.ei_nox * fuel_step) / 1e3,
        co=np.sum(ei.ei_co * fuel_step) / 1e3,
        hc=np.sum(ei.ei_hc * fuel_step) / 1e3,
        nvpm_n=np.sum(ei.ei_nvpm_n * fuel_step),                  # number, not kg
        h2o=fuel.ei_h2o * fuel_seg["cruise"],
        co2=EI_CO2 * fuel_seg["cruise"],
        so2=fuel.ei_so2_gkg * fuel_seg["cruise"] / 1e3,
    )

    # 4) the other segments: LTO-cycle EIs for taxi/take-off and descent/landing,
    #    climb-out EIs for climb (simplification, these are not the focus)
    lto = lto_indices(ac.engine, ICAO_LTO_MINUTES, fuel)
    climb = lto_indices(ac.engine, {"climb-out": 1.0}, fuel)
    other_fuel = fuel_seg["taxi_takeoff"] + fuel_seg["descent"]
    other_em = {}
    for sp in ("nox", "co", "hc"):
        other_em[sp] = (lto[f"ei_{sp}"] * other_fuel + climb[f"ei_{sp}"] * fuel_seg["climb"]) / 1e3
    other_em["nvpm_n"] = lto["ei_nvpm_n"] * other_fuel + climb["ei_nvpm_n"] * fuel_seg["climb"]
    non_cruise = other_fuel + fuel_seg["climb"]
    other_em["h2o"] = fuel.ei_h2o * non_cruise
    other_em["co2"] = EI_CO2 * non_cruise
    other_em["so2"] = fuel.ei_so2_gkg * non_cruise / 1e3

    cruise = dict(
        s_km=0.5 * (s[1:] + s[:-1]) / 1e3, h_m=h_mid, weight_kg=w_mid, v_ms=v_mid,
        wf_engine=wf_engine, fuel_kg=fuel_step, wf_sl=ei.wf_sl,
        thrust_fraction=ei.thrust_fraction, ei_nox=ei.ei_nox, ei_co=ei.ei_co,
        ei_hc=ei.ei_hc, ei_nvpm_n=ei.ei_nvpm_n, nvpm_change_pct=ei.nvpm_change_pct,
        method=ei.method,
    )
    return MissionResult(aircraft=ac, fuel=fuel, distance_km=distance_km,
                         passengers=passengers, takeoff_weight_kg=w0,
                         fuel_by_segment_kg=fuel_seg,
                         emissions_kg={"cruise": cruise_em, "other": other_em},
                         cruise=cruise)


def overall_efficiency(ac: Aircraft, fuel: Fuel) -> float:
    """Overall propulsion efficiency eta = thrust power / fuel power = V / (TSFC * LHV).

    Needed by the Schmidt-Appleman criterion. It is fuel-independent: SAF
    lowers the TSFC by exactly the factor its LHV is higher.
    """
    h_mid = 0.5 * (ac.h_cruise_start_ft + ac.h_cruise_end_ft) * FT
    _, _, _, a = isa(h_mid)
    tsfc = ac.tsfc_kg_Ns * fuel.k_blend
    return ac.mach * a / (tsfc * fuel.lhv)
