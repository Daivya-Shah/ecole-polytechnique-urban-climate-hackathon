"""
Adds ERA5 columns to features CSVs that are missing them.
Reads the ERA5 NetCDF files, extracts the nearest grid cell to each city,
and merges the values into the existing features CSV.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ERA5_DIR = os.path.join(BASE_DIR, "Data", "derived-era5-land-daily-statistics")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Approximate city coordinates
CITY_COORDS = {
    "Berlin":    (52.52,  13.40),
    "Bologna":   (44.49,  11.34),
    "Budapest":  (47.50,  19.04),
    "Dublin":    (53.33,  -6.25),
    "Frankfurt": (50.11,   8.68),
    "Hamburg":   (53.55,   9.99),
    "Madrid":    (40.42,  -3.70),
    "Oslo":      (59.91,  10.75),
    "Roma":      (41.90,  12.50),
    "Toulouse":  (43.60,   1.44),
    "Warszawa":  (52.23,  21.01),
    "Wien":      (48.21,  16.37),
}

YEARS = list(range(2020, 2026))


def load_era5_variable(variable, years):
    files = []
    for y in years:
        p = os.path.join(ERA5_DIR, f"{y}_{variable}.nc")
        if os.path.exists(p):
            files.append(p)
    if not files:
        print(f"  [WARN] No files found for {variable}")
        return None
    ds = xr.open_mfdataset(files, combine="by_coords", engine="h5netcdf")
    return ds


def extract_city_series(ds, varname, lat, lon):
    da = ds[varname]
    # Find time dimension name
    time_dim = "valid_time" if "valid_time" in da.dims else "time"
    lat_dim = "latitude" if "latitude" in da.dims else "lat"
    lon_dim = "longitude" if "longitude" in da.dims else "lon"
    # Select nearest grid cell
    pt = da.sel({lat_dim: lat, lon_dim: lon}, method="nearest")
    df = pt.to_series()
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return df


def process_city(city, lat, lon):
    features_path = os.path.join(OUTPUTS_DIR, f"features_{city}.csv")
    if not os.path.exists(features_path):
        print(f"  [SKIP] No features file for {city}")
        return

    df = pd.read_csv(features_path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")

    if "ERA5_t2m_max_C" in df.columns:
        print(f"  [SKIP] {city} already has ERA5_t2m_max_C")
        return

    print(f"  Processing {city} ({lat}, {lon})...")

    # Temperature max
    ds_t = load_era5_variable("2m_temperature_daily_maximum", YEARS)
    if ds_t is not None:
        varname = list(ds_t.data_vars)[0]
        t_series = extract_city_series(ds_t, varname, lat, lon)
        t_c = t_series - 273.15
        t_c.name = "ERA5_t2m_max_C"
        df = df.join(t_c, how="left")
        ds_t.close()

    # Precipitation
    ds_p = load_era5_variable("total_precipitation_daily_mean", YEARS)
    if ds_p is not None:
        varname = list(ds_p.data_vars)[0]
        p_series = extract_city_series(ds_p, varname, lat, lon)
        p_series.name = "ERA5_tp_m"
        df = df.join(p_series, how="left")
        ds_p.close()

    # Wind U
    ds_u = load_era5_variable("10m_u_component_of_wind_daily_mean", YEARS)
    if ds_u is not None:
        varname = list(ds_u.data_vars)[0]
        u_series = extract_city_series(ds_u, varname, lat, lon)
        u_series.name = "ERA5_u10_ms"
        df = df.join(u_series, how="left")
        ds_u.close()

    # Wind V
    ds_v = load_era5_variable("10m_v_component_of_wind_daily_mean", YEARS)
    if ds_v is not None:
        varname = list(ds_v.data_vars)[0]
        v_series = extract_city_series(ds_v, varname, lat, lon)
        v_series.name = "ERA5_v10_ms"
        df = df.join(v_series, how="left")
        ds_v.close()

    # ERA5_t2m_min_C: approximate as max - typical diurnal range (~8°C) if not available
    if "ERA5_t2m_max_C" in df.columns and "ERA5_t2m_min_C" not in df.columns:
        df["ERA5_t2m_min_C"] = df["ERA5_t2m_max_C"] - 8.0

    # Anomaly columns
    if "ERA5_t2m_max_C" in df.columns and "ERA5_t2m_max_C_anom" not in df.columns:
        monthly_mean = df["ERA5_t2m_max_C"].groupby(df.index.month).transform("mean")
        df["ERA5_t2m_max_C_anom"] = df["ERA5_t2m_max_C"] - monthly_mean

    df.index.name = "date"
    df.to_csv(features_path)
    print(f"  [OK] Saved {features_path} with ERA5 columns")


if __name__ == "__main__":
    print(f"ERA5 directory: {ERA5_DIR}")
    print(f"Files found: {len(os.listdir(ERA5_DIR)) if os.path.exists(ERA5_DIR) else 'MISSING'}")
    print()
    for city, (lat, lon) in CITY_COORDS.items():
        print(f"--- {city} ---")
        try:
            process_city(city, lat, lon)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
    print("\nDone.")
