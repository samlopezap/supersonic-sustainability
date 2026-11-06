"""STEP 2 - Fly the three aircraft on Paris-New York with Jet A-1, SAF50 and SAF100.

Run:  python scripts/step2_mission_emissions.py

Outputs
  results/tables/table2_mission_emissions.(csv|md)
  results/tables/table2b_concorde_checks.(csv|md)
  results/figures/fig2_emissions_per_pkm.png
  results/figures/fig3_saf_soot_vs_thrust.png
"""
from _common import LOAD_FACTOR, ROUTE_KM, ROUTE_NAME
import numpy as np
import pandas as pd

from supersaf.aircraft import all_aircraft, concorde
from supersaf.constants import EI_CO2, FT
from supersaf.emissions import s2f2, saf_nvpm_change_pct
from supersaf.fuels import DEFAULT_FUELS, JETA1, SAF50, SAF100, net_co2_factor
from supersaf.mission import fly_mission, overall_efficiency
from supersaf.plotting import (FUEL_COLOURS, INK_2, MUTED, apply_style, save,
                               write_table)

LCA_CENTRAL = 0.70   # SAF life-cycle CO2 reduction used for the "net CO2" column


def run_missions():
    results = {}
    for ac in all_aircraft():
        for fuel in DEFAULT_FUELS:
            results[(ac.name, fuel.name)] = fly_mission(ac, fuel, ROUTE_KM, LOAD_FACTOR)
    return results


def mission_table(results):
    rows = []
    for (ac_name, fuel_name), m in results.items():
        base = results[(ac_name, "Jet A-1")]
        rows.append({
            "Aircraft": ac_name,
            "Fuel": fuel_name,
            "Take-off weight [t]": m.takeoff_weight_kg / 1e3,
            "Trip fuel [t]": m.fuel_kg / 1e3,
            "Fuel [g/pkm]": 1e3 * m.per_pkm("fuel"),
            "Net life-cycle CO2 [g/pkm]": 1e3 * EI_CO2 * m.per_pkm("fuel")
                                          * net_co2_factor(m.fuel, LCA_CENTRAL),
            "H2O [g/pkm]": 1e3 * m.per_pkm("h2o"),
            "H2O at cruise [g/pkm]": 1e3 * m.per_pkm("h2o", "cruise"),
            "NOx [g/pkm]": 1e3 * m.per_pkm("nox"),
            "nvPM [1e12 #/pkm]": m.per_pkm("nvpm_n") / 1e12,
            "Cruise EINOx [g/kg]": m.cruise_mean("ei_nox"),
            "Cruise thrust setting [%]": 100 * m.cruise_mean("thrust_fraction"),
            "Cruise nvPM EIn [1e15/kg]": m.cruise_mean("ei_nvpm_n") / 1e15,
            "Cruise nvPM change vs Jet A-1 [%]": m.cruise_mean("nvpm_change_pct"),
            "H2O at cruise vs Jet A-1 [%]": 100 * (m.per_pkm("h2o", "cruise")
                                                  / base.per_pkm("h2o", "cruise") - 1),
            "Overall efficiency eta [-]": overall_efficiency(m.aircraft, m.fuel),
            "Method at cruise": m.cruise["method"],
        })
    return pd.DataFrame(rows)


def concorde_checks():
    """Two independent checks of the Concorde mission model."""
    full = fly_mission(concorde(), JETA1, 5555, load_factor=1.0)     # London-New York, 100 pax
    m = fly_mission(concorde(), JETA1, ROUTE_KM, LOAD_FACTOR)
    h = m.cruise["h_m"]
    i = int(np.argmin(np.abs(h - 54_000 * FT)))
    return pd.DataFrame([
        {"Check": "Fuel economy at full load, London-New York [pax-miles per US gal]",
         "Model": full.pax_miles_per_us_gallon, "Reference": 15.8,
         "Reference source": "historical figure (e.g. Wikipedia 'Concorde', Acoustics Today 2015)"},
        {"Check": "Cruise fuel flow per engine at 54,000 ft [kg/s]",
         "Model": float(m.cruise["wf_engine"][i]), "Reference": 1.5,
         "Reference source": "Fusaro et al. 2021, Table 3"},
        {"Check": "Cruise EINOx at 54,000 ft [g/kg]",
         "Model": float(m.cruise["ei_nox"][i]), "Reference": 20.97,
         "Reference source": "Fahey et al. 1995 (via Fusaro Table 3)"},
    ])


def plot_emissions(table):
    import matplotlib.pyplot as plt

    metrics = [("Fuel [g/pkm]", "a  Fuel burned [g per passenger-km]"),
               ("H2O at cruise [g/pkm]", "b  Water vapour at cruise altitude [g/pkm]"),
               ("NOx [g/pkm]", "c  NOx [g/pkm]"),
               ("nvPM [1e12 #/pkm]", "d  Soot particles [10^12 per pkm]")]
    aircraft = ["Concorde", "Overture-like", "A350-900"]
    fuels = ["Jet A-1", "SAF50", "SAF100"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    width = 0.24
    x = np.arange(len(aircraft))
    for ax, (col, title) in zip(axes.flat, metrics):
        for j, fuel in enumerate(fuels):
            vals = [table[(table.Aircraft == a) & (table.Fuel == fuel)][col].iloc[0] for a in aircraft]
            ax.bar(x + (j - 1) * (width + 0.02), vals, width=width, color=FUEL_COLOURS[fuel],
                   label=fuel, edgecolor="none")
        jet = [table[(table.Aircraft == a) & (table.Fuel == "Jet A-1")][col].iloc[0] for a in aircraft]
        for xi, v in zip(x, jet):
            ax.text(xi - width - 0.02, v * 1.02, f"{v:.3g}", ha="center", va="bottom",
                    color=INK_2, fontsize=7.5)
        ax.set_xticks(x, aircraft)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    axes[0, 0].legend(loc="upper right", ncol=3)
    fig.suptitle(f"{ROUTE_NAME}, {int(100 * LOAD_FACTOR)} % seats filled  "
                 "(labels: Jet A-1 values)", x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout()
    return save(fig, "fig2_emissions_per_pkm.png")


def plot_soot(results):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    f = np.linspace(10, 100, 181)
    ax.plot(f, saf_nvpm_change_pct(SAF50.delta_H, f), color=FUEL_COLOURS["SAF50"], label="SAF50")
    ax.plot(f, saf_nvpm_change_pct(SAF100.delta_H, f), color=FUEL_COLOURS["SAF100"], label="SAF100")
    ax.plot(f, saf_nvpm_change_pct(SAF100.delta_H, f, printed_form=True), color=MUTED,
            linewidth=1.2, label="SAF100, Eq. 1 as printed (typo)")
    # aircraft cruise points (SAF100): neutral ink markers, identified by direct labels
    offsets = {"A350-900": (10, -22, "left"), "Overture-like": (10, -22, "left"),
               "Concorde": (-40, 14, "right")}
    for ac_name, (dx, dy, ha) in offsets.items():
        m = results[(ac_name, "SAF100")]
        ft = 100 * m.cruise_mean("thrust_fraction")
        dv = m.cruise_mean("nvpm_change_pct")
        ax.scatter([ft], [dv], s=64, color=INK_2, edgecolor="white", linewidth=2, zorder=4)
        ax.annotate(f"{ac_name} cruise (SAF100): {dv:.0f} %", (ft, dv), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", color=INK_2, fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=2, shrinkB=5)
                    if ac_name == "Concorde" else None)
    ax.set_xlabel("Engine thrust setting F/F00 [%]")
    ax.set_ylabel("Change in soot number EI vs Jet A-1 [%]")
    ax.set_title("SAF cuts soot much less at the high thrust of supersonic cruise")
    ax.set_xlim(10, 102)
    ax.set_ylim(-95, 2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, "fig3_saf_soot_vs_thrust.png")


def main():
    apply_style()
    results = run_missions()
    table = mission_table(results)
    write_table(table, "table2_mission_emissions")
    checks = concorde_checks()
    write_table(checks, "table2b_concorde_checks")
    plot_emissions(table)
    plot_soot(results)

    pd.set_option("display.width", 200)
    print(f"STEP 2 - missions on {ROUTE_NAME} ({ROUTE_KM:.0f} km), load factor {LOAD_FACTOR}")
    cols = ["Aircraft", "Fuel", "Fuel [g/pkm]", "H2O at cruise [g/pkm]", "NOx [g/pkm]",
            "Cruise EINOx [g/kg]", "Cruise thrust setting [%]", "Cruise nvPM change vs Jet A-1 [%]"]
    print(table[cols].round(2).to_string(index=False))
    print("\nConcorde model checks:")
    print(checks[["Check", "Model", "Reference"]].round(2).to_string(index=False))
    return results, table


if __name__ == "__main__":
    main()
