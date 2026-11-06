# Supersonic flight with SAF: emissions and contrails

Course project, *Aviation and Environment* (ISAE-SUPAERO, 3rd semester).

**Question.** Would sustainable aviation fuel (SAF) cut the climate impact of a supersonic
airliner as much as it cuts that of a subsonic one? And can contrails form and persist at
supersonic cruise altitudes (50,000-60,000 ft)?

**Approach.** Two papers are re-implemented and chained together, then combined with
published climate sensitivities:

| Step | What | Based on |
|---|---|---|
| 1 | In-flight emission indices of a supersonic engine (S2F2), validated on Concorde | Fusaro, Viola & Galassini (2021), *Aerospace* 8, 331 |
| 2 | Mission fuel burn and emissions for Concorde, an Overture-like Mach 1.7 jet and an A350-900, each with Jet A-1, SAF50 and SAF100; SAF soot reduction | Fusaro et al. (2021); Teoh et al. (2022), *ES&T* 56, 17246 |
| 3 | Contrail formation (Schmidt-Appleman criterion) and persistence (ice supersaturation) at each cruise altitude | Teoh et al. (2022); Schumann (1996); AFGL reference atmospheres |
| 4 | One climate metric (100-yr energy forcing, as g CO2-eq per passenger-km): CO2, stratospheric water vapour, NOx, sulphate, contrails + Monte Carlo | Teoh et al. (2022) Eq. 3; van 't Hoff et al. (2025); Zhang et al. (2021, 2023); Lee et al. (2021) |

**The step-by-step explanation of the code is in [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md).**

## Headline results (Paris-New York, 80 % of seats filled)

| | Concorde | Overture-like | A350-900 |
|---|---|---|---|
| Fuel per passenger-km | 140 g (4.9x) | 82.5 g (2.9x) | 28.7 g |
| Soot cut at cruise with SAF100 | -8 % | -37 % | -57 % |
| Water vapour at cruise with SAF100 | +7.5 % | +7.6 % | +7.9 % |
| Climate metric, Jet A-1 [g CO2e/pkm] | 889 | 455 | 226 |
| **Change with SAF100** (5-95 % range) | **-18 %** (-25 to +3 %) | **-22 %** (-29 to +3 %) | **-52 %** (-57 to -33 %) |

* The same 71 % life-cycle CO2 cut becomes only a ~20 % climate benefit for supersonic aircraft,
  against ~52 % for the subsonic aircraft. Three reasons: (1) at 15-18 km the water vapour and NOx
  effects, which SAF does not reduce, are as large as the CO2 effect; (2) SAF contains no sulphur,
  so it removes the cooling from sulphate aerosol, which is large in the stratosphere; (3) contrails
  barely persist up there, so SAF's main non-CO2 benefit for subsonic jets (fewer soot particles
  means shorter-lived contrails) has little to act on, and at the near-maximum thrust of supersonic
  cruise SAF cuts soot only weakly anyway.
* Contrails: in standard-atmosphere air the Schmidt-Appleman threshold is met at 35,000-45,000 ft
  but not at 50,000-60,000 ft. In the extratropical reference atmospheres, the air at supersonic
  cruise would have to cool by 24-34 K to reach ice saturation (6-19 K at subsonic cruise), so
  persistent contrails are rare. The exception is the cold tropical tropopause.
* Flying lower helps. At a 13.5 km mean cruise altitude the SAF100 benefit grows to about -37 %
  (Figure 7b).

All numbers are in [`results/key_metrics.md`](results/key_metrics.md) and `results/tables/`.

## How to run

```bash
cd supersonic_project
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python scripts/run_all.py        # Steps 1-4, ~15 s, regenerates results/
python -m pytest -q              # 37 tests: paper values, cross-checks, physics identities
```

Each step can also be run alone, e.g. `python scripts/step3_contrails.py`.

**Optional, ERA5 statistics.** The reference atmospheres are climatological means. To count how
often contrails form and persist, download ERA5 (free Copernicus account; see the docstring):
`pip install cdsapi xarray netCDF4`, then `python scripts/download_era5.py`, then re-run Step 3.

## Repository map

```
supersaf/            the model (a small Python package)
  constants.py         physical constants, units
  atmosphere.py        ISA + AFGL reference atmospheres
  fuels.py             Jet A-1 / SAF blend properties (Teoh Table 1), life-cycle CO2
  engines.py           LTO certification data, EI-vs-fuel-flow correlation
  emissions.py         Boeing FFM2, S2F2 (Fusaro Eqs. 2-17), SAF soot model (Teoh Eq. 1)
  aircraft.py          Concorde, Overture-like, A350-900
  mission.py           mission fuel burn (segment fractions + Breguet cruise-climb) and emissions
  contrails.py         Schmidt-Appleman criterion, ice supersaturation, frost point
  climate.py           energy-forcing metric, literature sensitivities, Monte Carlo
  era5.py              optional ERA5 contrail statistics
  plotting.py          figure style
scripts/             step1 ... step4, run_all.py, download_era5.py
data/                engines_lto.csv, afgl_reference_atmospheres.csv, literature_sensitivities.csv
results/             figures/, tables/ (CSV + Markdown), key_metrics.md
tests/               pytest suite
tools/               how data/*.csv were extracted from their sources
docs/WALKTHROUGH.md  step-by-step explanation
```

## Main limitations

* S2F2 is validated against one Concorde data point. The Overture-like aircraft (aerodynamics,
  engine, weights) is an assumption, and its engine emissions use a scaled CFM56-7B27E as a proxy.
* The non-CO2 climate sensitivities come from 1-4 global models and differ by up to 5x. They are
  sampled in the Monte Carlo but dominate the uncertainty.
* Contrails at supersonic altitudes are bounded, not simulated (CoCiP is not validated in the
  stratosphere). The persistence analysis uses mean atmospheres unless you add ERA5.
* The NOx term for subsonic aviation includes the methane response (Lee et al. 2021). The
  supersonic one is ozone-only, as in the fleet studies used.

## References

* Fusaro, R., Viola, N., Galassini, D. (2021). Sustainable Supersonic Fuel Flow Method. *Aerospace* 8, 331.
* Teoh, R. et al. (2022). Targeted Use of Sustainable Aviation Fuel to Maximize Climate Benefits. *ES&T* 56, 17246-17255.
* van 't Hoff, J. A. et al. (2025). Multi-model assessment of the atmospheric and radiative effects of supersonic transport aircraft. *ACP* 25, 2515-2550.
* Zhang, J. et al. (2021). Stratospheric ozone and climate forcing sensitivity to cruise altitudes for fleets of potential supersonic transport aircraft. *JGR Atmospheres* 126, e2021JD034971.
* Zhang, J. et al. (2023). Potential impacts on ozone and climate from a proposed fleet of supersonic aircraft. *Earth's Future* 11, e2022EF003409.
* Lee, D. S. et al. (2021). The contribution of global aviation to anthropogenic climate forcing for 2000 to 2018. *Atmos. Environ.* 244, 117834.
* Schumann, U. (1996). On conditions for contrail formation from aircraft exhausts. *Meteorol. Z.* 5, 4-23.
* Gierens, K. et al. (1999). A distribution law for relative humidity in the upper troposphere and lower stratosphere derived from three years of MOZAIC measurements. *Ann. Geophys.* 17, 1218-1226.
* Anderson, G. P. et al. (1986). AFGL atmospheric constituent profiles (0-120 km). AFGL-TR-86-0110.
* ICAO Aircraft Engine Emissions Databank, v32 (via pycontrails).
* Roskam, J. Airplane Design, Part I (mission fuel fractions).
