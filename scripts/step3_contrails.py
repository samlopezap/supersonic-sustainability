"""STEP 3 - Can contrails form, and persist, at supersonic cruise altitudes?

Run:  python scripts/step3_contrails.py

Outputs
  results/tables/table3_sac_thresholds.(csv|md)        threshold temperature vs altitude & fuel
  results/tables/table4_reference_atmospheres.(csv|md) formation/persistence in 6 AFGL atmospheres
  results/figures/fig4_contrail_formation_persistence.png
  (+ table3c / fig5 if an ERA5 file is present in data/era5/)
"""
from _common import ROOT
import numpy as np
import pandas as pd

from supersaf.aircraft import all_aircraft
from supersaf.atmosphere import AFGL_NAMES, afgl_at_pressure, isa, load_afgl
from supersaf.constants import FT
from supersaf.contrails import (frost_point, mixing_line_slope, q_from_vmr, rhi, sac_satisfied,
                                sac_threshold, t_lm, vapour_pressure)
from supersaf.fuels import DEFAULT_FUELS, JETA1, SAF100
from supersaf.mission import overall_efficiency
from supersaf.plotting import INK, INK_2, MUTED, SERIES, apply_style, save, write_table

ETA_GENERIC = 0.40     # all three aircraft have eta = 0.40-0.41 (see Step 2 table)
FLIGHT_LEVELS_FT = [35_000, 39_000, 45_000, 50_000, 54_000, 58_000, 60_000]


def threshold_table():
    rows = []
    for h_ft in FLIGHT_LEVELS_FT:
        t_isa, p, _, _ = isa(h_ft * FT)
        row = {"Altitude [ft]": h_ft, "Altitude [km]": h_ft * FT / 1e3, "Pressure [hPa]": p / 100,
               "ISA temperature [C]": t_isa - 273.15}
        for fuel in DEFAULT_FUELS:
            row[f"Threshold dry air, {fuel.name} [C]"] = float(sac_threshold(p, fuel, ETA_GENERIC)) - 273.15
        row["Threshold liquid-saturated air (T_LM), Jet A-1 [C]"] = (
            float(t_lm(mixing_line_slope(p, JETA1, ETA_GENERIC))) - 273.15)
        row["ISA minus dry threshold, Jet A-1 [K]"] = (
            t_isa - float(sac_threshold(p, JETA1, ETA_GENERIC)))
        row["Contrail forms in ISA air?"] = "yes" if row["ISA minus dry threshold, Jet A-1 [K]"] < 0 else "no"
        rows.append(row)
    return pd.DataFrame(rows)


def reference_atmosphere_table(afgl):
    """Evaluate each aircraft's cruise band (flown at constant PRESSURE altitude)."""
    rows = []
    for ac in all_aircraft():
        eta = overall_efficiency(ac, JETA1)
        h = np.linspace(ac.h_cruise_start_ft, ac.h_cruise_end_ft, 41) * FT
        _, p, _, _ = isa(h)                              # pressure levels flown
        for name in AFGL_NAMES:
            T, h2o, z = afgl_at_pressure(afgl, name, p / 100)
            q = q_from_vmr(h2o)
            r = rhi(T, p, q)
            forms_jet = sac_satisfied(T, p, q, JETA1, eta)
            forms_saf = sac_satisfied(T, p, q, SAF100, eta)
            margin = T - sac_threshold(p, JETA1, eta)
            rows.append({
                "Aircraft": ac.name, "Atmosphere": name,
                "Ambient T [C] (min..max)": f"{T.min() - 273.15:.1f}..{T.max() - 273.15:.1f}",
                "T minus threshold, Jet A-1 [K] (min..max)": f"{margin.min():+.1f}..{margin.max():+.1f}",
                "Band with SAC met, Jet A-1 [%]": 100 * forms_jet.mean(),
                "Band with SAC met, SAF100 [%]": 100 * forms_saf.mean(),
                "Water vapour [ppmv] (min..max)": f"{h2o.min():.1f}..{h2o.max():.1f}",
                "Max RHi [%]": 100 * r.max(),
                "Cooling needed to reach ice saturation [K]": float(
                    np.min(T - frost_point(vapour_pressure(q, p)))),
                "Band with persistent contrails [%]": 100 * (forms_jet & (r >= 1)).mean(),
            })
    return pd.DataFrame(rows)


def plot_profiles(afgl):
    import matplotlib.pyplot as plt

    h_ft = np.linspace(25_000, 66_000, 300)
    h_km = h_ft * FT / 1e3
    _, p, _, _ = isa(h_ft * FT)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    bands = [(ac.name, ac.h_cruise_start_ft * FT / 1e3, ac.h_cruise_end_ft * FT / 1e3)
             for ac in all_aircraft()]
    for ax in (ax1, ax2):
        for name, lo, hi in bands:
            ax.axhspan(lo, hi, color="#f0efec", zorder=0, linewidth=0)
    for i, name in enumerate(AFGL_NAMES):
        T, h2o, _ = afgl_at_pressure(afgl, name, p / 100)
        ax1.plot(T - 273.15, h_km, color=SERIES[i], linewidth=1.6, label=name)
        ax2.plot(100 * rhi(T, p, q_from_vmr(h2o)), h_km, color=SERIES[i], linewidth=1.6, label=name)
    for fuel, colour, width in [(JETA1, INK, 2.2), (SAF100, MUTED, 1.4)]:
        ax1.plot(sac_threshold(p, fuel, ETA_GENERIC) - 273.15, h_km, color=colour, linewidth=width,
                 label=f"Contrail threshold, {fuel.name}")
    ax1.text(-88.5, 9.6, "left of (colder than) the\nblack line = contrails can form", color=INK_2,
             fontsize=8)
    labelled = {"Concorde / Overture-like": (min(b[1] for b in bands[:2]), max(b[2] for b in bands[:2])),
                "A350-900": (bands[2][1], bands[2][2])}
    for name, (lo, hi) in labelled.items():
        ax2.text(305, hi - 0.35, name, va="center", ha="right", color=INK_2, fontsize=8)
    ax1.set_xlim(-90, -30)
    ax1.set_ylim(7.0, 19.5)
    ax1.set_xlabel("Temperature [C]")
    ax1.set_ylabel("Pressure altitude [km]")
    ax1.set_title("a  Formation: ambient T vs Schmidt-Appleman threshold")
    handles, labels = ax1.get_legend_handles_labels()
    ax2.axvline(100, color=INK, linewidth=1.2)
    ax2.text(108, 18.9, "ice saturation", color=INK_2, fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlim(1, 320)
    ax2.set_xlabel("Relative humidity over ice, RHi [%]")
    ax2.set_title("b  Persistence: needs RHi above 100 %")
    fig.suptitle("Six AFGL reference atmospheres (climatological means); shaded = cruise bands",
                 x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    return save(fig, "fig4_contrail_formation_persistence.png")


def era5_part():
    files = sorted((ROOT / "data" / "era5").glob("*.nc"))
    if not files:
        print("\n  (No ERA5 file in data/era5/ - run scripts/download_era5.py to add "
              "day-to-day statistics. Skipping.)")
        return None
    from supersaf.era5 import contrail_statistics, load_era5

    frames = []
    for f in files:
        ds = load_era5(f)
        for corr in (False, True):
            frames.append(contrail_statistics(ds, list(DEFAULT_FUELS), ETA_GENERIC, corr))
    stats = pd.concat(frames).groupby(["level_hPa", "fuel", "humidity_correction"],
                                      as_index=False).mean()
    write_table(stats, "table3c_era5_contrail_frequencies")
    print("\n  ERA5 contrail statistics:")
    print(stats.round(4).to_string(index=False))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = stats[~stats.humidity_correction]
    for j, fuel in enumerate(["Jet A-1", "SAF100"]):
        s = sub[sub.fuel == fuel].sort_values("level_hPa")
        ax.plot(100 * s.persistent_fraction, s.level_hPa, marker="o", color=SERIES[2 * j],
                label=f"persistent, {fuel}")
        ax.plot(100 * s.sac_fraction, s.level_hPa, marker="o", color=SERIES[2 * j + 1],
                label=f"forming, {fuel}")
    ax.invert_yaxis()
    ax.set_xlabel("Share of samples [%]")
    ax.set_ylabel("Pressure level [hPa]")
    ax.set_title("ERA5: how often contrails form and persist")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig5_era5_contrail_frequency.png")
    return stats


def main():
    apply_style()
    afgl = load_afgl()
    t3 = threshold_table()
    write_table(t3, "table3_sac_thresholds")
    t4 = reference_atmosphere_table(afgl)
    write_table(t4, "table4_reference_atmospheres")
    plot_profiles(afgl)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("STEP 3 - Schmidt-Appleman thresholds (eta = 0.40)")
    print(t3.round(2).to_string(index=False))
    print("\nAFGL reference atmospheres, each aircraft's cruise band:")
    print(t4.round(1).to_string(index=False))
    era5_part()


if __name__ == "__main__":
    main()
