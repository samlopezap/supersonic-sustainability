"""The three aircraft compared in this study.

Concorde is the only supersonic airliner with real data, so its parameters
come from the literature. The "Overture-like" aircraft is a generic Mach 1.7
design with numbers chosen to be consistent with Boom's public claims
(60-80 seats, Mach 1.7, up to 60,000 ft, 4 engines of 35,000 lbf) - its
aerodynamics and engine are ASSUMPTIONS, varied in the sensitivity study.
The A350-900 is the modern subsonic reference.

Mission-segment weight fractions are textbook values (Roskam, Airplane
Design Part I, Table 2.1): e.g. a supersonic airliner keeps 92 % of its
weight through climb and acceleration, a subsonic jet 98 %.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .constants import LB_PER_LBF_H
from .engines import Engine, load_engine


@dataclass(frozen=True)
class Aircraft:
    name: str
    seats: int
    oew_kg: float               # operating empty weight
    mtow_kg: float              # maximum take-off weight (feasibility check only)
    n_engines: int
    engine: Engine
    mach: float                 # cruise Mach number
    h_cruise_start_ft: float    # cruise-climb start altitude
    h_cruise_end_ft: float      # cruise-climb end altitude
    lift_to_drag: float         # cruise L/D
    tsfc_kg_Ns: float           # cruise thrust-specific fuel consumption, Jet A-1 [kg/(N s)]
    f_taxi_takeoff: float       # W_after / W_before for start, taxi and take-off
    f_climb: float              # ... for climb and acceleration to cruise Mach
    f_descent_landing: float    # ... for descent and landing
    reserve_frac: float         # reserve fuel carried, as fraction of (OEW + payload)
    climb_distance_km: float    # ground distance covered while climbing
    descent_distance_km: float  # ground distance covered while descending
    notes: str = field(default="", compare=False)

    @property
    def supersonic(self) -> bool:
        return self.mach >= 1.0


def concorde() -> Aircraft:
    return Aircraft(
        name="Concorde",
        seats=100,                      # BA/Air France cabin
        oew_kg=78_700, mtow_kg=185_070,
        n_engines=4, engine=load_engine("Olympus 593 Mk610"),
        mach=2.02,
        h_cruise_start_ft=52_000, h_cruise_end_ft=59_000,  # typical NY run ~56,000 ft, max 60,000
        lift_to_drag=7.1,               # low end of the 7.1-7.5 literature range (calibration)
        tsfc_kg_Ns=1.195 * LB_PER_LBF_H,  # Olympus 593 Mk610 supersonic cruise SFC
        f_taxi_takeoff=0.98, f_climb=0.92, f_descent_landing=0.977,  # Roskam, SST column
        reserve_frac=0.10,
        climb_distance_km=450, descent_distance_km=350,
        notes="L/D chosen so that full-load fuel economy (16.6 pax-miles/US gal) and "
              "cruise fuel flow (1.42 kg/s/engine) are within 5 % of the historical "
              "15.8 pax-miles/gal and the 1.5 kg/s of Fusaro et al. Table 3.",
    )


def overture_like() -> Aircraft:
    symphony_proxy = load_engine("CFM56-7B27E").scaled_to(
        35_000 * 4.4482216152605 / 1000, name="Symphony proxy (CFM56-7B27E scaled)")
    return Aircraft(
        name="Overture-like",
        seats=65,                        # Boom: 60-80 seats
        oew_kg=36_000, mtow_kg=77_000,   # ASSUMED (not published)
        n_engines=4, engine=symphony_proxy,
        mach=1.7,
        h_cruise_start_ft=50_000, h_cruise_end_ft=58_000,  # Boom: "up to 60,000 ft"
        lift_to_drag=8.5,               # ASSUMED, better than Concorde (7.3)
        tsfc_kg_Ns=1.0 * LB_PER_LBF_H,  # ASSUMED, medium-bypass turbofan at Mach 1.7
        f_taxi_takeoff=0.98, f_climb=0.94, f_descent_landing=0.977,
        reserve_frac=0.10,
        climb_distance_km=400, descent_distance_km=300,
        notes="Generic Mach 1.7 design; engine emissions from a scaled CFM56-7B27E "
              "(rich-burn combustor) because Symphony has no public LTO data.",
    )


def a350() -> Aircraft:
    return Aircraft(
        name="A350-900",
        seats=315,
        oew_kg=142_400, mtow_kg=283_000,
        n_engines=2, engine=load_engine("Trent XWB-84"),
        mach=0.85,
        h_cruise_start_ft=35_000, h_cruise_end_ft=39_000,  # step climb FL350 -> FL390
        lift_to_drag=18.5,
        tsfc_kg_Ns=14.5e-6,             # 0.51 lb/(lbf h)
        f_taxi_takeoff=0.99, f_climb=0.98, f_descent_landing=0.982,  # Roskam, transport column
        reserve_frac=0.05,
        climb_distance_km=200, descent_distance_km=200,
    )


def all_aircraft():
    return [concorde(), overture_like(), a350()]
