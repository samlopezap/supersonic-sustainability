"""STEP 1 - Re-implement S2F2 and check it against the Concorde data of Fusaro et al. (2021).

Run:  python scripts/step1_validate_s2f2.py

Outputs
  results/tables/table1_s2f2_validation.(csv|md)
  results/figures/fig1_s2f2_method.png
"""
import _common  # noqa: F401  (adds the repo to the import path)
import numpy as np
import pandas as pd

from supersaf.constants import FT
from supersaf.emissions import ffm2, kp, kt, s2f2
from supersaf.engines import load_engine
from supersaf.fuels import JETA1, SAF100
from supersaf.plotting import INK, INK_2, MUTED, SERIES, apply_style, save, write_table

# Validation point: Fusaro et al. Table 3 (from Fahey et al. 1995, Science)
H_FT, MACH, WF = 54_000, 2.02, 1.5            # altitude, Mach, fuel flow per engine [kg/s]
MEASURED_EINOX = 20.97                        # g/kg, in-flight measurement
PAPER = {"EINOx": 22.3, "EICO": 1.9, "EIHC": 0.17}   # Fusaro et al. Table 5


def main():
    apply_style()
    olympus = load_engine("Olympus 593 Mk610")
    h = H_FT * FT

    ours = s2f2(olympus, WF, h, MACH, JETA1)                 # S2F2, x = 1, y = 0.4
    boeing = ffm2(olympus, WF, h, MACH, JETA1)               # original FFM2 for comparison
    saf = s2f2(olympus, WF * SAF100.k_blend, h, MACH, SAF100)

    rows = [
        ("Sea-level-equivalent fuel flow [kg/h]", np.nan, np.nan, float(ours.wf_sl) * 3600,
         float(boeing.wf_sl) * 3600, float(saf.wf_sl) * 3600),
        ("EINOx at cruise [g/kg]", MEASURED_EINOX, PAPER["EINOx"], float(ours.ei_nox),
         float(boeing.ei_nox), float(saf.ei_nox)),
        ("EICO at cruise [g/kg]", np.nan, PAPER["EICO"], float(ours.ei_co),
         float(boeing.ei_co), float(saf.ei_co)),
        ("EIHC at cruise [g/kg]", np.nan, PAPER["EIHC"], float(ours.ei_hc),
         float(boeing.ei_hc), float(saf.ei_hc)),
    ]
    table = pd.DataFrame(rows, columns=["Quantity", "Measured (Fahey 1995)", "Fusaro et al. S2F2",
                                        "This code S2F2", "This code FFM2 (subsonic method)",
                                        "This code S2F2, SAF100"])
    write_table(table, "table1_s2f2_validation")

    print("STEP 1 - S2F2 validation, Concorde at 54,000 ft, Mach 2.02, 1.5 kg/s per engine")
    print(table.round(3).to_string(index=False))
    print(f"\n  kp(M=2.02) = {float(kp(MACH)):.3f}   kt(M=2.02) = {float(kt(MACH)):.3f}")
    print(f"  NOx error vs measurement: {100 * (float(ours.ei_nox) / MEASURED_EINOX - 1):+.1f} % "
          f"(paper: {100 * (PAPER['EINOx'] / MEASURED_EINOX - 1):+.1f} %)")
    print(f"  FFM2 sea-level fuel flow is {float(boeing.wf_sl) / olympus.fuel_flow_takeoff:.1f}x "
          "the take-off fuel flow -> outside the engine map, as the paper says.")

    # ---------------- figure ----------------
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    m = np.linspace(1.0, 3.0, 101)
    ax1.plot(m, kp(m), color=SERIES[0], label="kp, pressure (Eq. 2)")
    ax1.plot(m, kt(m), color=SERIES[1], label="kt, temperature (Eq. 3)")
    anchors_m = [1, 2, 3]
    ax1.scatter(anchors_m, [1.0, 1.3, 1.76], s=36, color=SERIES[0], zorder=3,
                edgecolor="white", linewidth=1.5)
    ax1.scatter(anchors_m, [1.0, 0.77, 0.65], s=36, color=SERIES[1], zorder=3,
                edgecolor="white", linewidth=1.5)
    ax1.axvline(MACH, color=AXIS_LIGHT, linewidth=1)
    ax1.text(MACH + 0.04, 1.62, "Concorde\ncruise", color=INK_2, fontsize=8)
    ax1.set_xlabel("Free-stream Mach number")
    ax1.set_ylabel("Correction factor [-]")
    ax1.set_title("a  Intake corrections of S2F2")
    ax1.legend(loc="upper left")

    ff = np.logspace(np.log10(0.25), np.log10(4.5), 200)
    for species, colour, label in [("nox", SERIES[0], "NOx"), ("co", SERIES[1], "CO"),
                                   ("hc", SERIES[2], "HC")]:
        ax2.plot(ff, olympus.ei_sea_level(species, ff), color=colour, label=label)
        ei_pts = {"nox": olympus.ei_nox, "co": olympus.ei_co, "hc": olympus.ei_hc}[species]
        ax2.scatter(olympus.fuel_flow, np.maximum(ei_pts, 0.1), s=30, color=colour, zorder=3,
                    edgecolor="white", linewidth=1.5)
    ax2.axvline(float(ours.wf_sl), color=INK, linewidth=1)
    ax2.text(float(ours.wf_sl) * 0.96, 55, "S2F2 maps the\ncruise point here", ha="right",
             va="center", color=INK_2, fontsize=8)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax2.xaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 3, 4]))
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("Sea-level-equivalent fuel flow per engine [kg/s]")
    ax2.set_ylabel("Sea-level emission index [g/kg]")
    ax2.set_title("b  Olympus 593 EI correlation (LTO data)")
    ax2.legend(loc="lower left")
    ax2.text(0.63, 0.04, f"FFM2 would map cruise to {float(boeing.wf_sl):.0f} kg/s,\n"
             "3.4x take-off: off the engine map", transform=ax2.transAxes, ha="center",
             va="bottom", color=MUTED, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig1_s2f2_method.png")


AXIS_LIGHT = "#c3c2b7"

if __name__ == "__main__":
    main()
