import numpy as np
import pytest

from supersaf.contrails import (e_sat_ice, e_sat_liquid, frost_point, humidity_correction_teoh,
                                mixing_line_slope, q_from_vmr, rhi, sac_satisfied, sac_threshold,
                                t_lm, vapour_pressure)
from supersaf.fuels import JETA1, SAF100

P = np.array([23842.0, 11597.0, 7172.0])   # 35,000 / 50,000 / 60,000 ft


def test_saturation_pressures():
    assert float(e_sat_liquid(273.15)) == pytest.approx(611.2, rel=2e-3)
    assert float(e_sat_ice(273.15)) == pytest.approx(611.2, rel=2e-3)
    assert float(e_sat_ice(220.0)) < float(e_sat_liquid(220.0))


def test_dry_threshold_identity():
    G = mixing_line_slope(P, JETA1, 0.4)
    tlm = t_lm(G)
    assert np.allclose(sac_threshold(P, JETA1, 0.4), tlm - e_sat_liquid(tlm) / G)


@pytest.mark.parametrize("rh", [0.0, 0.5, 0.9])
def test_sac_switches_at_the_threshold(rh):
    t_crit = sac_threshold(P, JETA1, 0.4, rh_liquid=rh)
    for dT, expected in ((-0.05, True), (0.05, False)):
        T = t_crit + dT
        q = rh * e_sat_liquid(T) * 0.62198 / P          # specific humidity for that RH (dilute)
        assert np.all(sac_satisfied(T, P, q, JETA1, 0.4) == expected)


def test_saf_forms_contrails_at_warmer_temperatures():
    assert np.all(sac_threshold(P, SAF100, 0.4) > sac_threshold(P, JETA1, 0.4))


def test_supersonic_levels_are_too_warm_in_isa():
    # ISA lower stratosphere is 216.65 K; the dry threshold at 50,000+ ft is colder
    assert np.all(sac_threshold(P[1:], JETA1, 0.4) < 216.65)
    assert sac_threshold(P[0], JETA1, 0.4) > 219.0      # at 35,000 ft ISA (219 K) it forms


def test_frost_point_and_rhi():
    e = vapour_pressure(q_from_vmr(4.5), 9500.0)
    tf = frost_point(e)
    assert float(rhi(tf, 9500.0, q_from_vmr(4.5))) == pytest.approx(1.0, rel=1e-6)
    assert 185 < float(tf) < 195       # ~190 K for stratospheric humidity


def test_humidity_correction():
    out = humidity_correction_teoh(np.array([0.5, 0.97, 1.2, 2.0]))
    assert out[0] == pytest.approx(0.5 / 0.97)
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx((1.2 / 0.97) ** 1.7)
    assert out[3] == pytest.approx(1.7)


def test_cross_check_with_pycontrails():
    sac = pytest.importorskip("pycontrails.models.sac")
    G = sac.slope_mixing_line(np.zeros_like(P), P, 0.4, JETA1.ei_h2o, JETA1.lhv)
    for rh in (0.0, 0.5):
        theirs = sac.T_critical_sac(sac.T_sat_liquid(G), np.full_like(P, rh), G)
        # differences come only from the liquid saturation formula (< 0.3 K)
        assert np.allclose(sac_threshold(P, JETA1, 0.4, rh_liquid=rh), theirs, atol=0.3)
