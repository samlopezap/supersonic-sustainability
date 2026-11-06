"""Atmosphere models.

Two things live here:

1. The International Standard Atmosphere (ISA). It gives the temperature,
   pressure and speed of sound that the engine-emission method (S2F2/FFM2)
   and the mission model need at any altitude up to 32 km.
2. The six AFGL reference atmospheres (Anderson et al., 1986). Unlike the ISA
   they also contain a realistic water-vapour profile, which the contrail
   module needs to decide whether contrails can *persist*.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import G0, GAMMA, P_SL, R_AIR, T_SL

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ISA layers: (base altitude [m], base temperature [K], lapse rate [K/m]).
# Up to 32 km: troposphere, tropopause/lower-stratosphere (isothermal) and
# the first stratospheric layer, where temperature rises again.
_ISA_LAYERS = [
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.0010),
    (32000.0, 228.65, 0.0028),
]


def _layer_base_pressures():
    """Pressure at the base of each ISA layer, integrated once at import."""
    pressures = [P_SL]
    for (h0, t0, lapse), (h1, _, _) in zip(_ISA_LAYERS[:-1], _ISA_LAYERS[1:]):
        p0 = pressures[-1]
        if lapse == 0.0:
            p1 = p0 * np.exp(-G0 * (h1 - h0) / (R_AIR * t0))
        else:
            t1 = t0 + lapse * (h1 - h0)
            p1 = p0 * (t1 / t0) ** (-G0 / (lapse * R_AIR))
        pressures.append(p1)
    return pressures


_P_BASE = _layer_base_pressures()


def isa(h_m):
    """ISA temperature [K], pressure [Pa], density [kg/m3], speed of sound [m/s].

    Parameters
    ----------
    h_m : float or array
        Geopotential altitude in metres (0 to 47 km).
    """
    h = np.asarray(h_m, dtype=float)
    temperature = np.empty_like(h)
    pressure = np.empty_like(h)
    for i, (h0, t0, lapse) in enumerate(_ISA_LAYERS):
        upper = _ISA_LAYERS[i + 1][0] if i + 1 < len(_ISA_LAYERS) else np.inf
        mask = (h >= h0) & (h < upper) if i > 0 else (h < upper)
        if not np.any(mask):
            continue
        dh = h[mask] - h0
        if lapse == 0.0:
            temperature[mask] = t0
            pressure[mask] = _P_BASE[i] * np.exp(-G0 * dh / (R_AIR * t0))
        else:
            temperature[mask] = t0 + lapse * dh
            pressure[mask] = _P_BASE[i] * (temperature[mask] / t0) ** (-G0 / (lapse * R_AIR))
    density = pressure / (R_AIR * temperature)
    sound_speed = np.sqrt(GAMMA * R_AIR * temperature)
    if np.ndim(h_m) == 0:
        return float(temperature), float(pressure), float(density), float(sound_speed)
    return temperature, pressure, density, sound_speed


def isa_altitude_from_pressure(p_pa):
    """Invert the ISA: altitude [m] that has pressure ``p_pa`` [Pa]."""
    grid = np.linspace(0.0, 40000.0, 40001)
    _, p_grid, _, _ = isa(grid)
    # pressure decreases with altitude -> reverse both arrays for np.interp
    return np.interp(np.log(p_pa), np.log(p_grid[::-1]), grid[::-1])


def theta_delta(h_m):
    """Non-dimensional ambient temperature and pressure (theta, delta)."""
    t, p, _, _ = isa(h_m)
    return t / T_SL, p / P_SL


# ---------------------------------------------------------------------------
# AFGL reference atmospheres (with water vapour)
# ---------------------------------------------------------------------------
AFGL_NAMES = [
    "Tropical",
    "Midlatitude summer",
    "Midlatitude winter",
    "Subarctic summer",
    "Subarctic winter",
    "US standard",
]


def load_afgl() -> pd.DataFrame:
    """Return the AFGL profiles as a tidy table.

    Columns: atmosphere, z_km, p_hPa, T_K, h2o_ppmv.
    Source: Anderson, G. P. et al. (1986), AFGL atmospheric constituent
    profiles (0-120 km), AFGL-TR-86-0110 (values as distributed with pyrtlib).
    """
    return pd.read_csv(DATA_DIR / "afgl_reference_atmospheres.csv")


def afgl_at_pressure(df: pd.DataFrame, name: str, p_hpa):
    """Interpolate one AFGL atmosphere to pressure level(s) [hPa].

    Interpolation is linear in log(pressure), which is close to linear in
    altitude. Water vapour is interpolated in log space as well because it
    varies over orders of magnitude.
    Returns (T [K], h2o [ppmv], z [km]).
    """
    sub = df[df["atmosphere"] == name].sort_values("p_hPa")
    lp = np.log(sub["p_hPa"].to_numpy())
    x = np.log(np.asarray(p_hpa, dtype=float))
    temperature = np.interp(x, lp, sub["T_K"].to_numpy())
    h2o = np.exp(np.interp(x, lp, np.log(sub["h2o_ppmv"].to_numpy())))
    z = np.interp(x, lp, sub["z_km"].to_numpy())
    return temperature, h2o, z
