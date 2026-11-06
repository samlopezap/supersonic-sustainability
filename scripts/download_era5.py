"""Download an ERA5 sample for the contrail statistics of Step 3 (optional).

Requirements
  1. A free account at https://cds.climate.copernicus.eu and the ERA5
     licence accepted on the "ERA5 hourly data on pressure levels" page.
  2. Your API key in ~/.cdsapirc (instructions on the CDS website).
  3. pip install cdsapi xarray netCDF4

Default request (~300 MB): North Atlantic corridor 30-70 N, 80 W-0, year 2019
(the year studied by Teoh et al.), days 1, 8, 15 and 22 of every month at
00 and 12 UTC, pressure levels 250-70 hPa (subsonic and supersonic cruise).

Run:  python scripts/download_era5.py --year 2019
Then: python scripts/step3_contrails.py        (it finds data/era5/*.nc)
"""
import argparse
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "era5"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--year", default="2019")
    parser.add_argument("--days", default="01,08,15,22", help="comma-separated days of month")
    parser.add_argument("--times", default="00:00,12:00")
    parser.add_argument("--levels", default="70,100,125,150,175,200,225,250")
    parser.add_argument("--area", default="70,-80,30,0", help="North,West,South,East")
    args = parser.parse_args()

    import cdsapi  # imported here so the rest of the project works without it

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / f"era5_pl_{args.year}_north_atlantic.nc"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["temperature", "specific_humidity"],
        "pressure_level": args.levels.split(","),
        "year": [args.year],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": args.days.split(","),
        "time": args.times.split(","),
        "area": [float(v) for v in args.area.split(",")],
        "data_format": "netcdf",
    }
    print("Requesting ERA5 from the Copernicus CDS (this can queue for a while)...")
    cdsapi.Client().retrieve("reanalysis-era5-pressure-levels", request, str(target))
    print("saved", target)


if __name__ == "__main__":
    main()
