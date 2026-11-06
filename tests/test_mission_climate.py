import numpy as np
import pytest

from supersaf.aircraft import a350, concorde, overture_like
from supersaf.climate import (EF_PER_KG_CO2, ClimateModel, ClimateParameters, EI_H2O_IN_STUDIES,
                              load_sensitivities, sample_parameters)
from supersaf.constants import FT, G0
from supersaf.fuels import JETA1, SAF100
from supersaf.mission import fly_mission, overall_efficiency


def test_concorde_matches_historical_data():
    full = fly_mission(concorde(), JETA1, 5555, load_factor=1.0)
    assert full.pax_miles_per_us_gallon == pytest.approx(15.8, rel=0.10)
    m = fly_mission(concorde(), JETA1, 5840)
    i = int(np.argmin(np.abs(m.cruise["h_m"] - 54_000 * FT)))
    assert m.cruise["wf_engine"][i] == pytest.approx(1.5, rel=0.10)   # Fusaro Table 3
    assert m.takeoff_weight_kg < concorde().mtow_kg


def test_mass_balance_and_breguet():
    ac = overture_like()
    m = fly_mission(ac, JETA1, 5840)
    w_land = (ac.oew_kg + m.passengers * 100.0) * (1 + ac.reserve_frac)
    assert m.takeoff_weight_kg - m.fuel_kg == pytest.approx(w_land)
    # cruise lies in the isothermal ISA layer -> constant speed -> exact Breguet
    cruise_km = 5840 - ac.climb_distance_km - ac.descent_distance_km
    v = m.cruise["v_ms"][0]
    ratio = np.exp(cruise_km * 1e3 * ac.tsfc_kg_Ns * G0 / (v * ac.lift_to_drag))
    w_start = m.cruise["weight_kg"][0] + 0.5 * m.cruise["fuel_kg"][0]
    w_end = m.cruise["weight_kg"][-1] - 0.5 * m.cruise["fuel_kg"][-1]
    assert w_start / w_end == pytest.approx(ratio, rel=1e-3)


def test_saf_saves_at_least_the_heating_value_ratio():
    # Same energy -> k_blend less fuel mass, PLUS a small "snowball": lighter fuel
    # makes the aircraft lighter, so it needs even less. The snowball is largest
    # for Concorde, where fuel is ~40 % of the take-off weight.
    savings = {}
    for ac in (concorde(), overture_like(), a350()):
        jet = fly_mission(ac, JETA1, 5840).fuel_kg
        saf = fly_mission(ac, SAF100, 5840).fuel_kg
        assert SAF100.k_blend - 0.01 < saf / jet <= SAF100.k_blend
        savings[ac.name] = 1 - saf / jet
    assert savings["Concorde"] > savings["A350-900"]


def test_overall_efficiency_is_fuel_independent():
    for ac in (concorde(), a350()):
        assert overall_efficiency(ac, JETA1) == pytest.approx(overall_efficiency(ac, SAF100))
        assert 0.35 < overall_efficiency(ac, JETA1) < 0.45


def test_energy_forcing_constants():
    # Teoh et al. Table 2: 28.2e9 kg CO2 -> 42.0e18 J
    assert 28.2e9 * EF_PER_KG_CO2 == pytest.approx(42.0e18, rel=0.005)
    # Lee et al. (2021) water vapour GWP100 ~ 0.06
    assert ClimateModel().h2o_sub_per_kg() / EF_PER_KG_CO2 == pytest.approx(0.06, abs=0.01)


def test_h2o_interpolation_hits_literature_points():
    lit = load_sensitivities()
    model = ClimateModel()
    for key in ("h2o_sst_low_alt", "h2o_sst_high_alt"):
        z = lit.loc[key, "altitude_km"]
        s = model.h2o_sst_per_kg_h2o(z) * EI_H2O_IN_STUDIES / 1.60975e10
        assert float(s) == pytest.approx(lit.loc[key, "central"], rel=1e-3)


def test_saf_benefit_is_smaller_for_supersonic_aircraft():
    model = ClimateModel()
    change = {}
    for ac in (concorde(), overture_like(), a350()):
        jet = model.energy_forcing(fly_mission(ac, JETA1, 5840))["Total"]
        saf = model.energy_forcing(fly_mission(ac, SAF100, 5840))["Total"]
        change[ac.name] = saf / jet - 1
    assert change["A350-900"] < -0.4
    assert change["Concorde"] > change["A350-900"] + 0.2
    assert change["Overture-like"] > change["A350-900"] + 0.2


def test_monte_carlo_draws_are_reproducible_and_bounded():
    a, b = sample_parameters(50, seed=1), sample_parameters(50, seed=1)
    assert [p.lca_reduction for p in a] == [p.lca_reduction for p in b]
    assert all(0.10 <= p.lca_reduction <= 0.94 for p in a)
    assert all(0.0 <= p.q_h2o_sst <= 1.0 for p in a)
    assert isinstance(a[0], ClimateParameters)
