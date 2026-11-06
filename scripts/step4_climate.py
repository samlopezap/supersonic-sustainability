"""STEP 4 - Combine everything into one climate metric and test its uncertainty.

Metric: 100-year energy forcing per passenger-km, expressed as grams of
CO2-equivalent (GWP100 basis): EF / (AGWP_CO2,100 * S_earth).

Run:  python scripts/step4_climate.py

Outputs
  results/tables/table5_climate_components.(csv|md)
  results/tables/table6_saf_effect.(csv|md)
  results/tables/table7_altitude_sweep.(csv|md)
  results/figures/fig6_climate_components.png
  results/figures/fig7_saf_benefit_sensitivity.png
  results/figures/fig8_h2o_sensitivity_literature.png
  results/key_metrics.md
"""
from dataclasses import replace

from _common import LOAD_FACTOR, ROOT, ROUTE_KM, ROUTE_NAME
import numpy as np
import pandas as pd

from supersaf.aircraft import a350, all_aircraft, concorde, overture_like
from supersaf.climate import (COMPONENTS, EF_PER_KG_CO2, EF_PER_SENSITIVITY_UNIT, EI_H2O_IN_STUDIES,
                              ClimateModel,
                              ClimateParameters, load_sensitivities, sample_parameters)
from supersaf.constants import FT
from supersaf.fuels import DEFAULT_FUELS, JETA1, SAF100
from supersaf.mission import fly_mission
from supersaf.plotting import (AIRCRAFT_COLOURS, FUEL_COLOURS, INK, INK_2, MUTED, SERIES,
                               apply_style, save, write_table)

N_MONTE_CARLO = 4000
AIRCRAFT = ["Concorde", "Overture-like", "A350-900"]
FUELS = ["Jet A-1", "SAF50", "SAF100"]
COMPONENT_COLOURS = dict(zip(COMPONENTS, SERIES[:5]))


def g_co2e(ef_joule):
    """Energy forcing [J] -> grams CO2-equivalent (same 100-yr forcing as that CO2)."""
    return 1e3 * np.asarray(ef_joule) / EF_PER_KG_CO2


def run_missions():
    return {(ac.name, f.name): fly_mission(ac, f, ROUTE_KM, LOAD_FACTOR)
            for ac in all_aircraft() for f in DEFAULT_FUELS}


def central(missions):
    model = ClimateModel()
    return {k: {c: g_co2e(v) for c, v in model.energy_forcing(m).items()} for k, m in missions.items()}


def monte_carlo(missions, n=N_MONTE_CARLO):
    """Total metric for every mission and every parameter draw -> dict of arrays."""
    draws = sample_parameters(n)
    out = {k: np.empty(n) for k in missions}
    for i, params in enumerate(draws):
        model = ClimateModel(params)
        for k, m in missions.items():
            out[k][i] = g_co2e(model.energy_forcing(m)["Total"])
    return out


def pct(a, q):
    return float(np.percentile(a, q))


def component_table(cen, mc):
    rows = []
    for (ac, fuel), comp in cen.items():
        row = {"Aircraft": ac, "Fuel": fuel}
        row.update({f"{c} [g CO2e/pkm]": comp[c] for c in COMPONENTS})
        row["Total, central [g CO2e/pkm]"] = comp["Total"]
        row["Total, MC 5 % [g CO2e/pkm]"] = pct(mc[(ac, fuel)], 5)
        row["Total, MC median [g CO2e/pkm]"] = pct(mc[(ac, fuel)], 50)
        row["Total, MC 95 % [g CO2e/pkm]"] = pct(mc[(ac, fuel)], 95)
        row["Contrail upper bound (SST only) [g CO2e/pkm]"] = (
            comp["Contrail upper bound"] if ac != "A350-900" else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def saf_effect_table(cen, mc):
    rows = []
    sub_mc = {f: mc[("A350-900", f)] / mc[("A350-900", "Jet A-1")] - 1 for f in ("SAF50", "SAF100")}
    for ac in AIRCRAFT:
        base = cen[(ac, "Jet A-1")]
        for fuel in ("SAF50", "SAF100"):
            c = cen[(ac, fuel)]
            rel = mc[(ac, fuel)] / mc[(ac, "Jet A-1")] - 1
            row = {"Aircraft": ac, "Fuel": fuel,
                   "Change in total, central [%]": 100 * (c["Total"] / base["Total"] - 1),
                   "Change in total, MC 5 % [%]": 100 * pct(rel, 5),
                   "Change in total, MC 95 % [%]": 100 * pct(rel, 95),
                   "Change in CO2 alone [%]": 100 * (c["CO2 (life cycle)"] / base["CO2 (life cycle)"] - 1)}
            for comp in COMPONENTS:
                row[f"Contribution of {comp} [g CO2e/pkm]"] = c[comp] - base[comp]
            if ac != "A350-900":
                row["P(benefit smaller than half the A350 benefit)"] = float(np.mean(rel > 0.5 * sub_mc[fuel]))
            rows.append(row)
    return pd.DataFrame(rows)


def altitude_sweep():
    """Move each supersonic cruise band up/down and recompute the metric.

    Fuel burn hardly changes (the ISA is isothermal between 11 and 20 km, so the
    speed and Breguet range factor stay the same); the emission indices and,
    above all, the water-vapour sensitivity change with altitude.
    """
    rows = []
    for ac0 in (overture_like(), concorde()):
        span = ac0.h_cruise_end_ft - ac0.h_cruise_start_ft
        for z_km in np.arange(13.5, 19.01, 0.5):
            mid_ft = z_km * 1e3 / FT
            ac = replace(ac0, h_cruise_start_ft=mid_ft - span / 2, h_cruise_end_ft=mid_ft + span / 2)
            ms = {f.name: fly_mission(ac, f, ROUTE_KM, LOAD_FACTOR) for f in (JETA1, SAF100)}
            for label, q in (("low", 0.0), ("central", None), ("high", 1.0)):
                model = ClimateModel(ClimateParameters(q_h2o_sst=q))
                tot = {f: g_co2e(model.energy_forcing(m)["Total"]) for f, m in ms.items()}
                rows.append({"Aircraft": ac0.name, "Mean cruise altitude [km]": z_km,
                             "H2O sensitivity": label,
                             "Total Jet A-1 [g CO2e/pkm]": tot["Jet A-1"],
                             "Total SAF100 [g CO2e/pkm]": tot["SAF100"],
                             "SAF100 change [%]": 100 * (tot["SAF100"] / tot["Jet A-1"] - 1),
                             "Fuel [g/pkm]": 1e3 * ms["Jet A-1"].per_pkm("fuel")})
    return pd.DataFrame(rows)


def lca_sweep(missions):
    rows = []
    for r in np.linspace(0.10, 0.94, 43):
        model = ClimateModel(ClimateParameters(lca_reduction=r))
        for ac in AIRCRAFT:
            jet = g_co2e(model.energy_forcing(missions[(ac, "Jet A-1")])["Total"])
            saf = g_co2e(model.energy_forcing(missions[(ac, "SAF100")])["Total"])
            rows.append({"lca_reduction": r, "Aircraft": ac, "change_pct": 100 * (saf / jet - 1)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def plot_components(cen, mc, saf_tab):
    import matplotlib.pyplot as plt

    fig, (ax, axw) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})
    width, gap = 0.2, 0.06
    positions, ticklabels = [], []
    for i, ac in enumerate(AIRCRAFT):
        for j, fuel in enumerate(FUELS):
            x = i * 1.15 + (j - 1) * (width + gap)
            positions.append(x)
            ticklabels.append(fuel.replace("Jet A-1", "Jet"))
            c = cen[(ac, fuel)]
            pos_base, neg_base = 0.0, 0.0
            for comp in COMPONENTS:
                v = c[comp]
                if abs(v) < 1e-9:
                    continue
                if v >= 0:
                    ax.bar(x, v, width, bottom=pos_base, color=COMPONENT_COLOURS[comp],
                           edgecolor=ax.get_facecolor(), linewidth=1.0)
                    pos_base += v
                else:
                    ax.bar(x, v, width, bottom=neg_base, color=COMPONENT_COLOURS[comp],
                           edgecolor=ax.get_facecolor(), linewidth=1.0)
                    neg_base += v
            lo, hi = pct(mc[(ac, fuel)], 5), pct(mc[(ac, fuel)], 95)
            ax.plot([x, x], [lo, hi], color=INK, linewidth=1.2, zorder=5)
            ax.scatter([x], [c["Total"]], s=34, color=INK, edgecolor="white", linewidth=1.5, zorder=6)
            if ac != "A350-900":
                ax.scatter([x], [c["Total"] + c["Contrail upper bound"]], s=34, marker="D",
                           facecolor="white", edgecolor=INK, linewidth=1.1, zorder=6)
        ax.text(i * 1.15, -0.13, ac, transform=ax.get_xaxis_transform(), ha="center",
                color=INK, fontsize=9)
    for k, ac in enumerate(AIRCRAFT):   # label only the Jet A-1 totals, above their whisker
        jet = cen[(ac, "Jet A-1")]["Total"]
        top = pct(mc[(ac, "Jet A-1")], 95)
        ax.text(k * 1.15 - (width + gap), top + 18, f"{jet:.0f}", ha="center",
                va="bottom", color=INK_2, fontsize=7.5)
    ax.set_xticks(positions, ticklabels, fontsize=7.5)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_ylabel("g CO2-equivalent per passenger-km (100-yr)")
    ax.set_title("a  Climate impact by component")
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=COMPONENT_COLOURS[c], label=c) for c in COMPONENTS]
    handles += [Line2D([], [], color=INK, marker="o", markersize=5, label="Total, 5-95 % range"),
                Line2D([], [], color=INK, marker="D", markersize=5, markerfacecolor="white",
                       linestyle="none", label="+ contrail upper bound (SSTs)")]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5)

    # b: what SAF100 changes, component by component
    sub = saf_tab[saf_tab.Fuel == "SAF100"].set_index("Aircraft")
    y = np.arange(len(AIRCRAFT))
    h = 0.15
    for n, comp in enumerate(COMPONENTS):
        vals = [sub.loc[ac, f"Contribution of {comp} [g CO2e/pkm]"] for ac in AIRCRAFT]
        axw.barh(y + (n - 2) * (h + 0.01), vals, height=h, color=COMPONENT_COLOURS[comp], label=comp)
    ylabels = [f"{ac}\nnet {sub.loc[ac, 'Change in total, central [%]']:+.0f} %" for ac in AIRCRAFT]
    axw.axvline(0, color="#c3c2b7", linewidth=1)
    axw.set_yticks(y, ylabels)
    axw.invert_yaxis()
    axw.set_xlabel("Change with SAF100 vs Jet A-1 [g CO2e/pkm]")
    axw.set_title("b  What SAF100 changes")
    axw.grid(axis="y", visible=False)
    fig.suptitle(f"{ROUTE_NAME}, {int(100 * LOAD_FACTOR)} % load factor; SAF life-cycle CO2 cut 70 % "
                 "(central)", x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout()
    return save(fig, "fig6_climate_components.png")


def plot_sensitivity(lca, alt):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ac in AIRCRAFT:
        s = lca[lca.Aircraft == ac]
        ax1.plot(100 * s.lca_reduction, s.change_pct, color=AIRCRAFT_COLOURS[ac], label=ac)
        end = s.iloc[-1]
        ax1.text(100 * end.lca_reduction + 1, end.change_pct, f"{end.change_pct:.0f} %",
                 va="center", color=INK_2, fontsize=8)
    ax1.axvline(70, color="#c3c2b7", linewidth=1)
    ax1.text(71, -2, "central\nassumption", color=MUTED, fontsize=7.5, va="top")
    ax1.set_xlabel("SAF life-cycle CO2 reduction [%]")
    ax1.set_ylabel("Change in total climate metric with SAF100 [%]")
    ax1.set_title("a  SAF benefit vs SAF production pathway")
    ax1.set_xlim(8, 102)
    ax1.legend(loc="lower left")

    for ac in ("Overture-like", "Concorde"):
        s = alt[alt.Aircraft == ac]
        c = s[s["H2O sensitivity"] == "central"]
        lo = s[s["H2O sensitivity"] == "low"]
        hi = s[s["H2O sensitivity"] == "high"]
        ax2.fill_between(c["Mean cruise altitude [km]"], lo["SAF100 change [%]"],
                         hi["SAF100 change [%]"], color=AIRCRAFT_COLOURS[ac], alpha=0.12, linewidth=0)
        ax2.plot(c["Mean cruise altitude [km]"], c["SAF100 change [%]"], color=AIRCRAFT_COLOURS[ac],
                 label=f"{ac} (band: H2O sensitivity range)")
    a350_change = float(lca[(lca.Aircraft == "A350-900")].iloc[(lca[lca.Aircraft == "A350-900"]
                                                                .lca_reduction - 0.70).abs().argsort().iloc[0]].change_pct)
    ax2.axhline(a350_change, color=AIRCRAFT_COLOURS["A350-900"], linewidth=1.5)
    ax2.text(13.6, a350_change + 1.5, f"A350-900 at 11 km: {a350_change:.0f} %", color=INK_2, fontsize=8)
    for ac0, va in ((overture_like(), "top"), (concorde(), "bottom")):
        zmid = 0.5 * (ac0.h_cruise_start_ft + ac0.h_cruise_end_ft) * FT / 1e3
        ax2.axvline(zmid, color="#c3c2b7", linewidth=1)
        ax2.text(zmid + 0.06, 9 if va == "top" else 3, f"{ac0.name}\ndesign", color=MUTED,
                 fontsize=7.5, va="top")
    ax2.set_xlabel("Mean cruise altitude [km]")
    ax2.set_title("b  SAF benefit vs supersonic cruise altitude")
    ax2.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig7_saf_benefit_sensitivity.png")


def plot_literature():
    import matplotlib.pyplot as plt

    lit = load_sensitivities()
    model = ClimateModel()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    z = np.linspace(13.1, 19.5, 100)
    to_fuel = EI_H2O_IN_STUDIES / EF_PER_SENSITIVITY_UNIT   # J/kg H2O -> mW m-2 per Tg fuel/yr
    lo = model.h2o_sst_per_kg_h2o(z, q=0.0) * to_fuel
    hi = model.h2o_sst_per_kg_h2o(z, q=1.0) * to_fuel
    ce = model.h2o_sst_per_kg_h2o(z) * to_fuel
    ax.fill_between(z, lo, hi, color=SERIES[0], alpha=0.12, linewidth=0)
    ax.plot(z, ce, color=SERIES[0], label="used in this study (interpolated)")
    for key, label in (("h2o_sst_low_alt", "van 't Hoff 2025, Mach 1.6 fleet"),
                       ("h2o_sst_high_alt", "van 't Hoff 2025, Mach 2.0 fleet")):
        r = lit.loc[key]
        ax.errorbar([r.altitude_km], [r.central], yerr=[[r.central - r.low], [r.high - r.central]],
                    fmt="o", color=SERIES[0], ecolor=SERIES[0], capsize=3, markersize=6,
                    markeredgecolor="white")
        dy = -14 if "1.6" in label else 0
        ax.annotate(label, (r.altitude_km, r.central), xytext=(8, dy), textcoords="offset points",
                    va="center", color=INK_2, fontsize=8)
    for key, label, dx in (("h2o_zhang2023", "Zhang 2023 (1 model)", 8),
                           ("h2o_zhang2021", "Zhang 2021 (1 model)", -8)):
        r = lit.loc[key]
        ax.scatter([r.altitude_km], [r.central], marker="s", s=36, color=SERIES[1],
                   edgecolor="white", linewidth=1.5, zorder=4)
        ax.annotate(label, (r.altitude_km, r.central), xytext=(dx, 0), textcoords="offset points",
                    va="center", ha="left" if dx > 0 else "right", color=INK_2, fontsize=8)
    sub = lit.loc["h2o_sub"]
    ax.errorbar([11.0], [sub.central * EI_H2O_IN_STUDIES],
                yerr=[[(sub.central - sub.low) * EI_H2O_IN_STUDIES], [(sub.high - sub.central) * EI_H2O_IN_STUDIES]],
                fmt="o", color=SERIES[2], ecolor=SERIES[2], capsize=3, markersize=6, markeredgecolor="white")
    ax.annotate("subsonic aviation (Lee 2021)", (11.0, sub.central * EI_H2O_IN_STUDIES), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK_2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlim(10.5, 23)
    ax.set_xlabel("Cruise altitude [km]")
    ax.set_ylabel("H2O forcing [mW m-2 per Tg fuel/yr]")
    ax.set_title("Water vapour forcing grows ~50x from 11 km to 18 km")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, "fig8_h2o_sensitivity_literature.png")


def key_metrics(missions, cen, saf_tab, mc):
    jet = {ac: missions[(ac, "Jet A-1")] for ac in AIRCRAFT}
    s = saf_tab.set_index(["Aircraft", "Fuel"])
    ratio_jet = mc[("Overture-like", "Jet A-1")] / mc[("A350-900", "Jet A-1")]
    ratio_saf = mc[("Overture-like", "SAF100")] / mc[("A350-900", "SAF100")]
    lines = [
        "# Key metrics",
        "",
        f"Route {ROUTE_NAME} ({ROUTE_KM:.0f} km), {int(100 * LOAD_FACTOR)} % load factor. "
        "Generated by `scripts/step4_climate.py` - do not edit by hand.",
        "",
        "| Metric | Concorde | Overture-like | A350-900 |",
        "|---|---|---|---|",
        "| Fuel per passenger-km [g] | " + " | ".join(f"{1e3 * jet[a].per_pkm('fuel'):.1f}" for a in AIRCRAFT) + " |",
        "| Fuel per pkm relative to A350-900 | " + " | ".join(
            f"{jet[a].per_pkm('fuel') / jet['A350-900'].per_pkm('fuel'):.1f}x" for a in AIRCRAFT) + " |",
        "| Cruise thrust setting [%] | " + " | ".join(
            f"{100 * jet[a].cruise_mean('thrust_fraction'):.0f}" for a in AIRCRAFT) + " |",
        "| Soot (nvPM number) cut at cruise with SAF100 [%] | " + " | ".join(
            f"{missions[(a, 'SAF100')].cruise_mean('nvpm_change_pct'):.0f}" for a in AIRCRAFT) + " |",
        "| Water vapour at cruise with SAF100 [%] | " + " | ".join(
            f"{100 * (missions[(a, 'SAF100')].per_pkm('h2o', 'cruise') / jet[a].per_pkm('h2o', 'cruise') - 1):+.1f}"
            for a in AIRCRAFT) + " |",
        "| Cruise EINOx, Jet A-1 [g/kg] | " + " | ".join(f"{jet[a].cruise_mean('ei_nox'):.1f}" for a in AIRCRAFT) + " |",
        "| Climate metric, Jet A-1 [g CO2e/pkm, 100-yr] | " + " | ".join(
            f"{cen[(a, 'Jet A-1')]['Total']:.0f} ({pct(mc[(a, 'Jet A-1')], 5):.0f}-{pct(mc[(a, 'Jet A-1')], 95):.0f})"
            for a in AIRCRAFT) + " |",
        "| Climate metric, SAF100 [g CO2e/pkm] | " + " | ".join(
            f"{cen[(a, 'SAF100')]['Total']:.0f} ({pct(mc[(a, 'SAF100')], 5):.0f}-{pct(mc[(a, 'SAF100')], 95):.0f})"
            for a in AIRCRAFT) + " |",
        "| Change with SAF100, central (5-95 %) | " + " | ".join(
            f"{s.loc[(a, 'SAF100'), 'Change in total, central [%]']:.0f} % "
            f"({s.loc[(a, 'SAF100'), 'Change in total, MC 5 % [%]']:.0f} to "
            f"{s.loc[(a, 'SAF100'), 'Change in total, MC 95 % [%]']:.0f} %)" for a in AIRCRAFT) + " |",
        "| Change in CO2 alone with SAF100 [%] | " + " | ".join(
            f"{s.loc[(a, 'SAF100'), 'Change in CO2 alone [%]']:.0f}" for a in AIRCRAFT) + " |",
        "",
        f"* Overture-like vs A350-900, total climate metric per pkm: {np.median(ratio_jet):.1f}x with Jet A-1 "
        f"(5-95 %: {pct(ratio_jet, 5):.1f}-{pct(ratio_jet, 95):.1f}x), {np.median(ratio_saf):.1f}x when both use SAF100 "
        f"({pct(ratio_saf, 5):.1f}-{pct(ratio_saf, 95):.1f}x).",
        f"* Probability that SAF100 helps the Overture-like aircraft less than half as much as the A350-900: "
        f"{100 * s.loc[('Overture-like', 'SAF100'), 'P(benefit smaller than half the A350 benefit)']:.0f} %.",
        "* Contrails: in standard-atmosphere air, the Schmidt-Appleman threshold is met at 35,000-45,000 ft "
        "but not at 50,000-60,000 ft; in the extratropical reference atmospheres the air at supersonic cruise "
        "would have to cool 24-34 K to reach ice saturation (6-19 K at subsonic cruise).",
        "",
        "Ranges are 5th-95th percentiles of a Monte Carlo over the climate sensitivities and the SAF "
        "life-cycle CO2 cut (10-94 %), not over aircraft performance.",
    ]
    path = ROOT / "results" / "key_metrics.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    apply_style()
    missions = run_missions()
    cen = central(missions)
    mc = monte_carlo(missions)
    t5 = component_table(cen, mc)
    t6 = saf_effect_table(cen, mc)
    alt = altitude_sweep()
    lca = lca_sweep(missions)
    write_table(t5, "table5_climate_components")
    write_table(t6, "table6_saf_effect")
    write_table(alt, "table7_altitude_sweep")
    plot_components(cen, mc, t6)
    plot_sensitivity(lca, alt)
    plot_literature()
    km = key_metrics(missions, cen, t6, mc)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("STEP 4 - climate metric [g CO2e per pkm, GWP100-equivalent]")
    print(t5[["Aircraft", "Fuel"] + [f"{c} [g CO2e/pkm]" for c in COMPONENTS]
             + ["Total, central [g CO2e/pkm]", "Total, MC 5 % [g CO2e/pkm]",
                "Total, MC 95 % [g CO2e/pkm]"]].round(1).to_string(index=False))
    print("\nEffect of SAF:")
    print(t6[["Aircraft", "Fuel", "Change in total, central [%]", "Change in total, MC 5 % [%]",
              "Change in total, MC 95 % [%]", "Change in CO2 alone [%]"]].round(1).to_string(index=False))
    print("\n" + km.read_text())


if __name__ == "__main__":
    main()
