"""OPTIONAL: contrail statistics from ERA5 reanalysis (needs xarray + netCDF4).

The reference atmospheres used in Step 3 are long-term MEANS: they tell you
whether contrails form on an average day, not how often they form. ERA5 gives
the day-to-day variability. Download a sample with scripts/download_era5.py
(needs a free Copernicus account), then run Step 3 again: it detects the file.

Two ERA5 caveats for stratospheric levels:
* ERA5 has a known MOIST bias in the lowermost stratosphere, so persistence
  frequencies are, if anything, overestimated there;
* the humidity correction of Teoh et al. was fitted in the upper
  troposphere; we offer it only as a sensitivity switch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contrails import humidity_correction_teoh, rhi, sac_satisfied
from .fuels import Fuel


def load_era5(path: str | Path):
    """Open an ERA5 pressure-level file and standardise names.

    Accepts both the new CDS layout (valid_time, pressure_level, t, q) and the
    old one (time, level, t, q). Returns a Dataset with variables T [K],
    q [kg/kg] and coordinates time, level [hPa], latitude, longitude.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    rename = {}
    for old, new in [("valid_time", "time"), ("pressure_level", "level"),
                     ("t", "T"), ("temperature", "T"), ("specific_humidity", "q")]:
        if old in ds.variables or old in ds.dims:
            rename[old] = new
    ds = ds.rename(rename)
    keep = [v for v in ("T", "q") if v in ds]
    if len(keep) < 2:
        raise ValueError(f"{path}: needs temperature and specific humidity, found {list(ds.data_vars)}")
    return ds[keep]


def contrail_statistics(ds, fuels: list[Fuel], eta: float,
                        humidity_correction: bool = False) -> pd.DataFrame:
    """Area-weighted frequency of contrail formation and persistence per level.

    Returns one row per (pressure level, fuel) with the fraction of
    (time, grid point) samples where: the SAC is met; the air is
    ice-supersaturated; both (= persistent contrail).
    """
    rows = []
    weights_lat = np.cos(np.deg2rad(ds["latitude"]))
    for level in ds["level"].values:
        sub = ds.sel(level=level)
        p = float(level) * 100.0
        T = sub["T"].values
        q = sub["q"].values
        r = rhi(T, p, q)
        if humidity_correction:
            r = humidity_correction_teoh(r)
        # weights broadcast to (time, lat, lon) or (lat, lon)
        w = np.broadcast_to(weights_lat.values[:, None], T.shape[-2:])
        w = np.broadcast_to(w, T.shape)
        issr = r >= 1.0
        for fuel in fuels:
            forms = sac_satisfied(T, p, q, fuel, eta)
            persistent = forms & issr
            rows.append(dict(
                level_hPa=float(level), fuel=fuel.name,
                sac_fraction=float(np.sum(w * forms) / np.sum(w)),
                issr_fraction=float(np.sum(w * issr) / np.sum(w)),
                persistent_fraction=float(np.sum(w * persistent) / np.sum(w)),
                mean_T_K=float(np.sum(w * T) / np.sum(w)),
                humidity_correction=humidity_correction,
            ))
    return pd.DataFrame(rows)
