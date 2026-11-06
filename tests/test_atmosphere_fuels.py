import numpy as np
import pytest

from supersaf.atmosphere import AFGL_NAMES, afgl_at_pressure, isa, isa_altitude_from_pressure, load_afgl
from supersaf.fuels import JETA1, SAF100, make_fuel, net_co2_factor


def test_isa_reference_points():
    t0, p0, _, a0 = isa(0.0)
    assert t0 == pytest.approx(288.15)
    assert p0 == pytest.approx(101325.0)
    assert a0 == pytest.approx(340.29, abs=0.05)
    t11, p11, _, _ = isa(11000.0)
    assert t11 == pytest.approx(216.65)
    assert p11 == pytest.approx(22632.06, abs=1.0)
    _, p20, _, _ = isa(20000.0)
    assert p20 == pytest.approx(5474.89, abs=2.0)


def test_isa_inverse_round_trip():
    h = np.array([5000.0, 11000.0, 16459.0, 18288.0])
    _, p, _, _ = isa(h)
    assert np.allclose(isa_altitude_from_pressure(p), h, atol=1.0)


def test_afgl_profiles_loaded():
    df = load_afgl()
    assert set(df["atmosphere"]) == set(AFGL_NAMES)
    T, h2o, z = afgl_at_pressure(df, "US standard", 100.0)
    assert 15.5 < float(z) < 16.5          # 100 hPa ~ 16 km
    assert float(T) == pytest.approx(216.7, abs=0.5)
    assert 3 < float(h2o) < 6              # a few ppmv in the stratosphere


@pytest.mark.parametrize("blend,H,lhv,ei", [(0.10, 13.95, 43.21, 1.250), (0.50, 14.55, 43.64, 1.304),
                                            (0.70, 14.85, 43.85, 1.331)])
def test_fuel_blends_match_teoh_table1(blend, H, lhv, ei):
    f = make_fuel(blend)
    assert f.H_pct == pytest.approx(H, abs=0.006)
    assert f.lhv_MJkg == pytest.approx(lhv, abs=0.006)
    assert f.ei_h2o == pytest.approx(ei, abs=0.0006)


def test_k_blend_and_lca():
    assert SAF100.k_blend == pytest.approx(43.10 / 44.17)
    assert JETA1.k_blend == 1.0
    assert net_co2_factor(SAF100, 0.94) == pytest.approx(0.06)
    assert net_co2_factor(JETA1, 0.94) == 1.0
