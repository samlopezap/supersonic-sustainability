"""Checks against Fusaro et al. (2021) and Teoh et al. (2022)."""
import numpy as np
import pytest

from supersaf.constants import FT
from supersaf.emissions import ffm2, kp, kt, s2f2, s2f2_paper_form, saf_nvpm_change_pct
from supersaf.engines import load_engine
from supersaf.fuels import JETA1, SAF100, make_fuel

OLYMPUS = load_engine("Olympus 593 Mk610")
H = 54_000 * FT


def test_intake_corrections_anchor_points():
    # Fusaro: kp = 1, 1.3, 1.76 and kt = 1, 0.77, 0.65 at Mach 1, 2, 3
    assert np.allclose(kp([1.0, 2.0, 3.0]), [1.0, 1.30, 1.76], atol=1e-9)
    assert np.allclose(kt([1.0, 2.0, 3.0]), [1.0, 0.77, 0.65], atol=1e-9)
    assert float(kp(0.8)) == 1.0 and float(kt(0.8)) == 1.0


def test_s2f2_reproduces_concorde_validation():
    r = s2f2(OLYMPUS, 1.5, H, 2.02, JETA1)
    assert float(r.ei_co) == pytest.approx(1.9, abs=0.05)     # paper Table 5
    assert float(r.ei_hc) == pytest.approx(0.17, abs=0.01)    # paper Table 5
    # NOx: within 10 % of the in-flight measurement (20.97 g/kg)
    assert float(r.ei_nox) == pytest.approx(20.97, rel=0.10)


def test_original_ffm2_is_off_the_engine_map_at_mach2():
    r = ffm2(OLYMPUS, 1.5, H, 2.02, JETA1)
    assert float(r.wf_sl) > 3 * OLYMPUS.fuel_flow_takeoff


def test_explicit_saf_form_equals_paper_equations_14_17():
    for fuel in (make_fuel(0.3), SAF100):
        wf_jet = 1.5
        ours = s2f2(OLYMPUS, wf_jet * fuel.k_blend, H, 2.02, fuel)
        paper = s2f2_paper_form(OLYMPUS, wf_jet, H, 2.02, fuel)
        assert float(ours.wf_sl) == pytest.approx(float(paper["wf_sl"]))
        for sp in ("nox", "co", "hc"):
            grams_ours = float(getattr(ours, f"ei_{sp}")) * wf_jet * fuel.k_blend
            grams_paper = float(paper[f"ei_{sp}"]) * wf_jet
            assert grams_ours == pytest.approx(grams_paper)


@pytest.mark.parametrize("blend,table2", [(0.10, -8.5), (0.30, -25.5), (0.50, -37.2),
                                          (0.70, -44.7), (1.00, -51.1)])
def test_nvpm_equation_reproduces_teoh_table2(blend, table2):
    # Fleet-mean changes of Teoh Table 2 are reproduced at an effective 54 % thrust
    f = make_fuel(blend)
    assert float(saf_nvpm_change_pct(f.delta_H, 54.0)) == pytest.approx(table2, abs=1.0)


def test_nvpm_printed_form_is_non_monotonic():
    # The equation as printed gives SAF100 a SMALLER cut than SAF70: the typo
    printed70 = float(saf_nvpm_change_pct(1.05, 54.0, printed_form=True))
    printed100 = float(saf_nvpm_change_pct(1.50, 54.0, printed_form=True))
    assert printed100 > printed70
    code70, code100 = (float(saf_nvpm_change_pct(d, 54.0)) for d in (1.05, 1.50))
    assert code100 < code70


def test_nvpm_reduction_shrinks_with_thrust():
    low, high = (float(saf_nvpm_change_pct(1.5, f)) for f in (40.0, 100.0))
    assert low < -60 and -10 < high < 0
