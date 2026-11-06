"""The ERA5 pipeline, tested on a small synthetic file with the CDS layout."""
import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from supersaf.contrails import e_sat_ice  # noqa: E402
from supersaf.era5 import contrail_statistics, load_era5  # noqa: E402
from supersaf.fuels import JETA1, SAF100  # noqa: E402


def make_file(tmp_path):
    times = np.array(["2019-01-01T00", "2019-01-01T12"], dtype="datetime64[ns]")
    levels = np.array([100.0, 200.0])
    lat = np.array([60.0, 50.0, 40.0])
    lon = np.array([-40.0, -30.0])
    shape = (2, 2, 3, 2)
    T = np.full(shape, 216.0)
    T[:, 1] = 215.0               # 200 hPa: cold enough for contrails
    T[0, 1, 0, 0] = 212.0
    q = np.full(shape, 4e-6)      # stratosphere-like dryness
    # one ice-supersaturated grid point at 200 hPa (RHi = 1.2)
    q[0, 1, 0, 0] = 1.2 * e_sat_ice(212.0) * 0.62198 / 20000.0
    ds = xr.Dataset({"t": (("valid_time", "pressure_level", "latitude", "longitude"), T),
                     "q": (("valid_time", "pressure_level", "latitude", "longitude"), q)},
                    coords={"valid_time": times, "pressure_level": levels,
                            "latitude": lat, "longitude": lon})
    path = tmp_path / "era5_test.nc"
    ds.to_netcdf(path)
    return path


def test_load_and_statistics(tmp_path):
    ds = load_era5(make_file(tmp_path))
    assert set(ds.data_vars) == {"T", "q"}
    stats = contrail_statistics(ds, [JETA1, SAF100], eta=0.4)
    at100 = stats[stats.level_hPa == 100.0]
    at200 = stats[stats.level_hPa == 200.0]
    assert np.all(at100.sac_fraction == 0.0)        # 216 K at 100 hPa: too warm
    assert np.all(at200.sac_fraction == 1.0)        # 215 K at 200 hPa: forms
    assert np.all((at200.persistent_fraction > 0) & (at200.persistent_fraction < 0.2))
    assert np.all((stats[["sac_fraction", "issr_fraction", "persistent_fraction"]] >= 0).values)
