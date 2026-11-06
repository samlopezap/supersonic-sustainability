"""Re-create the CSV files in data/ from their original sources.

You do NOT need to run this: the CSVs are committed. It documents where every
number comes from, and needs two extra packages (pip install pyrtlib pycontrails).

* data/afgl_reference_atmospheres.csv  <- pyrtlib (AFGL profiles, Anderson et al. 1986)
* data/engines_lto.csv                 <- Fusaro et al. (2021) Table 4 (Olympus 593)
                                          + ICAO Engine Emissions Databank v32
                                            (as bundled in pycontrails)
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def afgl():
    from pyrtlib.climatology import AtmosphericProfiles as atmp

    names = {
        atmp.TROPICAL: "Tropical",
        atmp.MIDLATITUDE_SUMMER: "Midlatitude summer",
        atmp.MIDLATITUDE_WINTER: "Midlatitude winter",
        atmp.SUBARCTIC_SUMMER: "Subarctic summer",
        atmp.SUBARCTIC_WINTER: "Subarctic winter",
        atmp.US_STANDARD: "US standard",
    }
    rows = []
    for key, name in names.items():
        z, p, _, t, md = atmp.gl_atm(key)
        for zi, pi, ti, qi in zip(z, p, t, md[:, atmp.H2O]):
            if zi <= 50:
                rows.append(dict(atmosphere=name, z_km=zi, p_hPa=pi, T_K=ti, h2o_ppmv=qi))
    pd.DataFrame(rows).to_csv(DATA / "afgl_reference_atmospheres.csv", index=False)


def engines():
    rows = []
    # Olympus 593 Mk610 (Concorde), Fusaro et al. 2021 Table 4 (fuel flow in kg/h).
    # The afterburner row is not used (it is reheat fuel, not a dry operating point).
    olympus = [
        ("idle", 1140, 2.5, 118, 36),
        ("descent", 2360, 4.0, 82, 22),
        ("approach", 4550, 6.5, 55, 8.5),
        ("climb-out", 9100, 12.5, 20, 1.5),
        ("take-off", 12700, 22.3, 1.1, 0.0),
    ]
    for mode, wf_kgh, nox, co, hc in olympus:
        rows.append(dict(engine="Olympus 593 Mk610", source="Fusaro et al. 2021, Table 4",
                         rated_thrust_kN=139.4, mode=mode, thrust_setting=wf_kgh / 12700,
                         fuel_flow_kg_s=wf_kgh / 3600, ei_nox_gkg=nox, ei_co_gkg=co,
                         ei_hc_gkg=hc, nvpm_ein_per_kg=float("nan")))

    import pycontrails
    static = Path(pycontrails.__file__).parent / "models" / "emissions" / "static"
    gas = pd.read_csv(static / "edb-gaseous-v32-engines.csv")
    nvpm = pd.read_csv(static / "edb-nvpm-v32-engines.csv")
    nvpm.columns = [c.strip() for c in nvpm.columns]
    modes = [("idle", "Idle", 0.07), ("approach", "App", 0.30),
             ("climb-out", "C/O", 0.85), ("take-off", "T/O", 1.00)]
    for uid, label in [("01P18RR124", "Trent XWB-84"), ("01P11CM121", "CFM56-7B27E")]:
        g = gas[gas["UID No"] == uid].iloc[0]
        n = nvpm[nvpm["UID No"] == uid].iloc[0]
        for mode, col, f in modes:
            rows.append(dict(
                engine=label, source=f"ICAO EDB v32, UID {uid}",
                rated_thrust_kN=float(g["Rated Thrust (kN)"]), mode=mode, thrust_setting=f,
                fuel_flow_kg_s=float(g[f"Fuel Flow {col} (kg/sec)"]),
                ei_nox_gkg=float(g[f"NOx EI {col} (g/kg)"]),
                ei_co_gkg=float(g[f"CO EI {col} (g/kg)"]),
                ei_hc_gkg=float(g[f"HC EI {col} (g/kg)"]),
                # system-loss-corrected number EI, as used by Teoh et al. (2022)
                nvpm_ein_per_kg=float(n[f"nvPM EInum_SL {col} (#/kg)"]),
            ))
    pd.DataFrame(rows).to_csv(DATA / "engines_lto.csv", index=False)


if __name__ == "__main__":
    afgl()
    engines()
    print("written:", sorted(p.name for p in DATA.glob("*.csv")))
