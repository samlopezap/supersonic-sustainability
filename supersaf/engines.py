"""Engine emission data and the sea-level "EI versus fuel flow" correlation.

Both fuel-flow methods (Boeing FFM2 and its supersonic extension S2F2) work
in two steps:

1. convert the in-flight fuel flow into the fuel flow the engine would have
   at sea level for the same combustor inlet temperature (T3);
2. read the sea-level emission index (EI) at that fuel flow from the
   certification (LTO) data, then correct it back to altitude.

This module does step 2's "read the EI" part. As in Fusaro et al. (2021,
Fig. 5) the correlation is a piecewise straight line in log(EI)-log(fuel flow)
space through the LTO points, held constant outside the measured range.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# EIs equal to zero cannot be put on a log scale. Fusaro et al. plot the
# Olympus take-off HC (0 g/kg) at 0.1 g/kg (their Fig. 5); we do the same.
EI_FLOOR_GKG = 0.1

# Teoh et al. (2022) assume this nvPM number EI when no engine data exist.
DEFAULT_NVPM_EIN = 1e15


@dataclass(frozen=True)
class Engine:
    """Certification (LTO) data for one engine, sorted by fuel flow."""

    name: str
    source: str
    rated_thrust_kN: float
    thrust_setting: np.ndarray   # F/F00 at each LTO point [-]
    fuel_flow: np.ndarray        # kg/s at each LTO point (sea level, ISA)
    ei_nox: np.ndarray           # g/kg
    ei_co: np.ndarray            # g/kg
    ei_hc: np.ndarray            # g/kg
    nvpm_ein: np.ndarray | None  # #/kg (None if no nvPM data)

    @property
    def fuel_flow_takeoff(self) -> float:
        """Maximum (take-off) fuel flow, used to normalise the thrust setting."""
        return float(self.fuel_flow.max())

    def scaled_to(self, rated_thrust_kN: float, name: str | None = None) -> "Engine":
        """Rubber-engine scaling: same EIs at the same thrust setting, fuel flows
        proportional to thrust. Used for the Overture-like aircraft, whose
        Symphony engine has no public emissions data."""
        k = rated_thrust_kN / self.rated_thrust_kN
        return replace(self, name=name or f"{self.name} (scaled x{k:.2f})",
                       rated_thrust_kN=rated_thrust_kN, fuel_flow=self.fuel_flow * k)

    def ei_sea_level(self, species: str, wf_sl_kg_s):
        """Sea-level EI [g/kg] at sea-level-equivalent fuel flow (log-log interpolation)."""
        ei = {"nox": self.ei_nox, "co": self.ei_co, "hc": self.ei_hc}[species]
        ei = np.maximum(ei, EI_FLOOR_GKG)
        x = np.log(np.asarray(wf_sl_kg_s, dtype=float))
        # np.interp holds the end values constant outside the data range
        return np.exp(np.interp(x, np.log(self.fuel_flow), np.log(ei)))

    def thrust_fraction(self, wf_sl_kg_s):
        """Thrust setting F/F00 estimated as fuel-flow ratio (Teoh et al. 2022),
        clipped to [0, 1]."""
        return np.clip(np.asarray(wf_sl_kg_s, dtype=float) / self.fuel_flow_takeoff, 0.0, 1.0)

    def nvpm_ein_jet(self, thrust_fraction):
        """Jet A-1 nvPM number EI, linear interpolation in thrust setting."""
        if self.nvpm_ein is None:
            return np.full_like(np.asarray(thrust_fraction, dtype=float), DEFAULT_NVPM_EIN)
        return np.interp(thrust_fraction, self.thrust_setting, self.nvpm_ein)


def load_engine(name: str) -> Engine:
    """Load one engine from data/engines_lto.csv."""
    df = pd.read_csv(DATA_DIR / "engines_lto.csv")
    sub = df[df["engine"] == name].sort_values("fuel_flow_kg_s")
    if sub.empty:
        raise KeyError(f"engine {name!r} not in engines_lto.csv")
    nvpm = sub["nvpm_ein_per_kg"].to_numpy()
    return Engine(
        name=name,
        source=str(sub["source"].iloc[0]),
        rated_thrust_kN=float(sub["rated_thrust_kN"].iloc[0]),
        thrust_setting=sub["thrust_setting"].to_numpy(),
        fuel_flow=sub["fuel_flow_kg_s"].to_numpy(),
        ei_nox=sub["ei_nox_gkg"].to_numpy(),
        ei_co=sub["ei_co_gkg"].to_numpy(),
        ei_hc=sub["ei_hc_gkg"].to_numpy(),
        nvpm_ein=None if np.all(np.isnan(nvpm)) else nvpm,
    )
