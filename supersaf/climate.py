"""Turn emissions into one comparable climate metric: energy forcing (EF).

Energy forcing [J] = radiative forcing integrated over time and over the
Earth's surface. Teoh et al. (2022) use it to put contrails (minutes-to-hours
effect) and CO2 (centuries) on one scale (their Eq. 3):

    EF_CO2 = AGWP_CO2,100 * m_CO2 * S_earth           (1.49e9 J per kg CO2)

For the other species we only know, from global-model studies, how much
SUSTAINED forcing a SUSTAINED emission rate produces:

    S = RF / emission rate                [mW m-2 per (Tg/yr)]

For a linear species with lifetime tau, S = RE * tau, and the 100-year
integrated forcing of a 1 kg pulse is S * (1 - exp(-100/tau)) ~ S (tau of
months to a few years). So

    EF per kg = S * 1e-12 [W m-2 yr kg-1] * seconds_per_year * S_earth
              = S * 1.61e10  J per kg

Sanity check: Lee et al. (2021) water vapour, S = 0.00524, gives 8.4e7 J/kg,
i.e. 0.057 x CO2 -> the GWP100 of 0.06 published by Lee et al. It matches.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import AGWP_CO2_100, EI_CO2, FT, S_EARTH, SECONDS_PER_YEAR
from .fuels import Fuel, net_co2_factor
from .mission import MissionResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EF_PER_KG_CO2 = AGWP_CO2_100 * S_EARTH                          # 1.4895e9 J/kg
EF_PER_SENSITIVITY_UNIT = 1e-12 * SECONDS_PER_YEAR * S_EARTH     # J/kg per (mW m-2 per Tg/yr)
EI_H2O_IN_STUDIES = 1.237     # kg H2O / kg fuel assumed by the fleet studies
EI_SO2_IN_STUDIES = 1.2e-3    # kg SO2 / kg fuel assumed for the fleet studies (standard sulphur)

COMPONENTS = ["CO2 (life cycle)", "Water vapour", "NOx", "Sulphate aerosol", "Contrails"]


def load_sensitivities() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "literature_sensitivities.csv").set_index("key")


def _tri_ppf(u, low, mode, high):
    """Inverse CDF of a triangular distribution (vectorised, low <= mode <= high)."""
    u, low, mode, high = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (u, low, mode, high)))
    span = np.where(high > low, high - low, 1.0)
    fc = (mode - low) / span
    left = low + np.sqrt(u * span * (mode - low))
    right = high - np.sqrt((1 - u) * span * (high - mode))
    return np.where(high > low, np.where(u < fc, left, right), mode)


@dataclass
class ClimateParameters:
    """One set of climate sensitivities (central values by default).

    ``q_*`` are quantiles in [0, 1] (0.5 ~ central) used by the Monte Carlo;
    the deterministic run uses the 'central' column directly.
    """

    lca_reduction: float = 0.70       # SAF life-cycle CO2 cut (Teoh: 0.10-0.94)
    q_h2o_sst: float | None = None
    o3_sst: float | None = None
    sulfate_sst_scale: float = 1.0
    h2o_sub: float | None = None
    nox_sub: float | None = None
    sulfate_sub: float | None = None
    contrail_sub_scale: float = 1.0


class ClimateModel:
    def __init__(self, params: ClimateParameters | None = None):
        self.p = params or ClimateParameters()
        self.lit = load_sensitivities()

    # --- sensitivities ----------------------------------------------------
    def _lit(self, key, column="central"):
        return float(self.lit.loc[key, column])

    def _loglin(self, altitude_km, lo_key, hi_key, col):
        """Interpolate a sensitivity between the two van 't Hoff et al. (2025)
        fleets (band mid-points 14.9 km and 18.0 km) LINEARLY IN LOG SPACE,
        i.e. exponential growth with altitude, extended over the two bands
        (13.1-19.5 km) and held constant outside.

        Why log space: the forcing grows because the perturbation lifetime of
        water vapour (how long it stays before being flushed out of the
        stratosphere) grows roughly exponentially with height. A straight
        line would wrongly drop to zero at ~13.6 km.
        """
        z = np.clip(np.asarray(altitude_km, dtype=float), 13.1, 19.5)
        z0, z1 = self._lit(lo_key, "altitude_km"), self._lit(hi_key, "altitude_km")
        v0, v1 = self._lit(lo_key, col), self._lit(hi_key, col)
        sign = np.sign(v0)
        w = (z - z0) / (z1 - z0)
        return sign * np.exp(np.log(abs(v0)) + w * (np.log(abs(v1)) - np.log(abs(v0))))

    def h2o_sst_per_kg_h2o(self, altitude_km, q=None):
        """EF [J] per kg H2O emitted at ``altitude_km`` (stratospheric cruise).

        ``q`` in [0, 1] picks a point in the model spread (0 = lowest model,
        1 = highest); None = four-model mean.
        """
        vals = {col: self._loglin(altitude_km, "h2o_sst_low_alt", "h2o_sst_high_alt", col)
                for col in ("low", "central", "high")}
        q = self.p.q_h2o_sst if q is None else q
        s = vals["central"] if q is None else _tri_ppf(q, vals["low"], vals["central"], vals["high"])
        return s / EI_H2O_IN_STUDIES * EF_PER_SENSITIVITY_UNIT

    def sulfate_sst_per_kg_so2(self, altitude_km):
        s = self._loglin(altitude_km, "sulfate_sst_low_alt", "sulfate_sst_high_alt", "central")
        return s * self.p.sulfate_sst_scale / EI_SO2_IN_STUDIES * EF_PER_SENSITIVITY_UNIT

    def o3_sst_per_kg_no2(self):
        s = self.p.o3_sst if self.p.o3_sst is not None else self._lit("o3_sst")
        return s * EF_PER_SENSITIVITY_UNIT

    def h2o_sub_per_kg(self):
        s = self.p.h2o_sub if self.p.h2o_sub is not None else self._lit("h2o_sub")
        return s * EF_PER_SENSITIVITY_UNIT

    def nox_sub_per_kg_no2(self):
        s = self.p.nox_sub if self.p.nox_sub is not None else self._lit("nox_sub")
        return s * EF_PER_SENSITIVITY_UNIT

    def sulfate_sub_per_kg_so2(self):
        s = self.p.sulfate_sub if self.p.sulfate_sub is not None else self._lit("sulfate_sub")
        return s * EF_PER_SENSITIVITY_UNIT

    def contrail_sub_per_km(self, fuel: Fuel):
        """Teoh et al. (2022) Table 2: contrail EF per flight distance vs blend."""
        blends = np.array([0.0, 0.01, 0.10, 0.30, 0.50, 0.70, 1.00])
        ef_per_m = np.array([0.53, 0.53, 0.50, 0.42, 0.37, 0.33, 0.29]) * 1e8
        return float(np.interp(fuel.blend, blends, ef_per_m)) * 1e3 * self.p.contrail_sub_scale

    def contrail_sst_upper_bound_per_km(self, fuel: Fuel, nvpm_change_pct: float):
        """UPPER bound for supersonic contrails (not part of the central total).

        Scale the subsonic EF per km by the ratio of persistence probabilities:
        at most 2 % of lowermost-stratosphere air is ice-supersaturated
        (Gierens et al. 1999) vs 16.2 % of subsonic North-Atlantic flight
        distance forming persistent contrails (Teoh et al. 2022). The SAF
        effect is scaled with the nvPM change the way CoCiP responded in
        Teoh et al. (-51.5 % nvPM -> -44.8 % EF, i.e. x0.87).
        """
        ratio = self._lit("issr_strat_frac", "high") / self._lit("contrail_sub_persist_frac")
        base = self._lit("contrail_sub") * self.p.contrail_sub_scale
        return ratio * base * (1.0 + 0.87 * nvpm_change_pct / 100.0)

    # --- the metric ---------------------------------------------------------
    def energy_forcing(self, m: MissionResult) -> dict:
        """EF [J] per passenger-km, by component."""
        ac, fuel = m.aircraft, m.fuel
        cruise, other = m.emissions_kg["cruise"], m.emissions_kg["other"]
        z_km = 0.5 * (ac.h_cruise_start_ft + ac.h_cruise_end_ft) * FT / 1e3
        co2 = EI_CO2 * m.fuel_kg * net_co2_factor(fuel, self.p.lca_reduction) * EF_PER_KG_CO2
        if ac.supersonic:
            h2o = cruise["h2o"] * self.h2o_sst_per_kg_h2o(z_km) + other["h2o"] * self.h2o_sub_per_kg()
            nox = cruise["nox"] * self.o3_sst_per_kg_no2() + other["nox"] * self.nox_sub_per_kg_no2()
            sulf = (cruise["so2"] * self.sulfate_sst_per_kg_so2(z_km)
                    + other["so2"] * self.sulfate_sub_per_kg_so2())
            contrail = 0.0
            cruise_km = m.cruise["s_km"][-1] + 0.5 * (m.cruise["s_km"][1] - m.cruise["s_km"][0])
            bound = cruise_km * self.contrail_sst_upper_bound_per_km(
                fuel, m.cruise_mean("nvpm_change_pct"))
        else:
            total_h2o = cruise["h2o"] + other["h2o"]
            h2o = total_h2o * self.h2o_sub_per_kg()
            nox = (cruise["nox"] + other["nox"]) * self.nox_sub_per_kg_no2()
            sulf = (cruise["so2"] + other["so2"]) * self.sulfate_sub_per_kg_so2()
            cruise_km = m.distance_km - ac.climb_distance_km - ac.descent_distance_km
            contrail = cruise_km * self.contrail_sub_per_km(fuel)
            bound = contrail
        values = [co2, h2o, nox, sulf, contrail]
        out = {k: float(v) / m.pkm for k, v in zip(COMPONENTS, values)}
        out["Total"] = sum(out[k] for k in COMPONENTS)
        out["Contrail upper bound"] = float(bound) / m.pkm
        return out


def sample_parameters(n: int, seed: int = 42) -> list[ClimateParameters]:
    """Monte Carlo draws: triangular(low, central, high) for every sensitivity."""
    rng = np.random.default_rng(seed)
    lit = load_sensitivities()

    def tri(key):
        return rng.triangular(lit.loc[key, "low"], lit.loc[key, "central"], lit.loc[key, "high"], n)

    lca = rng.triangular(0.10, 0.70, 0.94, n)
    q_h2o = rng.uniform(0, 1, n)
    o3 = tri("o3_sst")
    sulf_scale = rng.triangular(0.5, 1.0, 1.5, n)
    h2o_sub, nox_sub, sulf_sub = tri("h2o_sub"), tri("nox_sub"), tri("sulfate_sub")
    contrail_scale = rng.triangular(0.5, 1.0, 1.5, n)
    return [ClimateParameters(lca_reduction=lca[i], q_h2o_sst=q_h2o[i], o3_sst=o3[i],
                              sulfate_sst_scale=sulf_scale[i], h2o_sub=h2o_sub[i],
                              nox_sub=nox_sub[i], sulfate_sub=sulf_sub[i],
                              contrail_sub_scale=contrail_scale[i]) for i in range(n)]
