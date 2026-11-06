"""Fuel properties for Jet A-1, SAF and their blends.

Every downstream calculation needs only a handful of fuel numbers:

* hydrogen mass content  -> drives the nvPM (soot) reduction (Teoh Eq. 1)
* lower heating value     -> fuel mass needed for the same energy, and the
                             k_blend factor of S2F2 (Fusaro Eqs. 12-13)
* water emission index    -> contrail formation (Schmidt-Appleman) and
                             stratospheric water vapour
* sulphur                 -> sulphate aerosol (SAF is practically sulphur-free)

Teoh et al. (2022) show (their Table 1, Fig. S7) that these properties are
linear in the blending ratio, so a blend is a linear interpolation between
Jet A-1 and 100% SAF.
"""
from __future__ import annotations

from dataclasses import dataclass

# End members, Teoh et al. (2022) Table 1 (Baseline and SAF100 rows)
JET_A1 = dict(H_pct=13.80, lhv_MJkg=43.10, ei_h2o=1.237)
SAF_100 = dict(H_pct=15.30, lhv_MJkg=44.17, ei_h2o=1.371)

# Sulphur: Jet A-1 with ~600 ppm fuel sulphur gives EI(SO2) = 2 x 0.6 g/kg.
# The 2018 global inventory used by Lee et al. (2021) implies ~1.1 g/kg.
EI_SO2_JET = 1.2   # g SO2 per kg fuel
EI_SO2_SAF = 0.0   # HEFA / FT synthetic kerosene contain essentially no sulphur


@dataclass(frozen=True)
class Fuel:
    """A fuel (or blend) defined by its SAF blending ratio."""

    name: str
    blend: float        # SAF fraction, 0 = Jet A-1, 1 = 100% SAF
    H_pct: float        # hydrogen mass content [%]
    lhv_MJkg: float     # lower heating value [MJ/kg]
    ei_h2o: float       # kg H2O per kg fuel
    ei_so2_gkg: float   # g SO2 per kg fuel

    @property
    def lhv(self) -> float:
        """Lower heating value in J/kg."""
        return self.lhv_MJkg * 1e6

    @property
    def delta_H(self) -> float:
        """Hydrogen-content difference vs the reference Jet A-1 [percentage points]."""
        return self.H_pct - JET_A1["H_pct"]

    @property
    def k_blend(self) -> float:
        """Fusaro Eq. 13: LHV_kerosene / LHV_blend (< 1 for SAF)."""
        return JET_A1["lhv_MJkg"] / self.lhv_MJkg


def make_fuel(blend: float, name: str | None = None) -> Fuel:
    """Linear interpolation of the fuel properties (Fusaro Eq. 12, Teoh Table 1)."""
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between 0 and 1")

    def lerp(key):
        return JET_A1[key] + blend * (SAF_100[key] - JET_A1[key])

    if name is None:
        name = "Jet A-1" if blend == 0 else f"SAF{round(100 * blend)}"
    return Fuel(
        name=name,
        blend=blend,
        H_pct=lerp("H_pct"),
        lhv_MJkg=lerp("lhv_MJkg"),
        ei_h2o=lerp("ei_h2o"),
        ei_so2_gkg=EI_SO2_JET + blend * (EI_SO2_SAF - EI_SO2_JET),
    )


JETA1 = make_fuel(0.0)
SAF50 = make_fuel(0.5)
SAF100 = make_fuel(1.0)
DEFAULT_FUELS = (JETA1, SAF50, SAF100)


def net_co2_factor(fuel: Fuel, lca_reduction: float) -> float:
    """Fraction of the fossil CO2 that remains on a life-cycle basis.

    Following Teoh et al. (2022): the SAF share of the fuel cuts life-cycle
    CO2 by ``lca_reduction`` (between 0.10 and 0.94 depending on feedstock
    and pathway). Combustion CO2 is still 3.16 kg/kg, but for SAF it was
    recently taken out of the atmosphere by the feedstock.
    """
    return 1.0 - fuel.blend * lca_reduction
