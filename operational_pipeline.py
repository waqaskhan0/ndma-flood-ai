"""
Daily operational flood-alert pipeline.

What it does:
1. Fetches current-year daily rainfall for every district from NASA POWER.
2. Aggregates rainfall into the same district/year style features used by the model.
3. Loads models/flood_model_best.pkl + scaler.pkl + model_info.pkl.
4. Writes latest_predictions.csv and active_alerts.csv for dashboard monitoring.

Example:
    python operational_pipeline.py

For an offline test using the historical rainfall CSV:
    python operational_pipeline.py --no-fetch --as-of 2022-08-31
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from all_districts_data import ALL_DISTRICTS


BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "data", "raw")
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
LIVE_DIR = os.path.join(PROCESSED_DIR, "live")
MODEL_DIR = os.path.join(BASE, "models")

os.makedirs(LIVE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

CURRENT_RAIN_CSV = os.path.join(LIVE_DIR, "pakistan_rainfall_current_year.csv")
LATEST_PREDICTIONS_CSV = os.path.join(PROCESSED_DIR, "latest_predictions.csv")
ACTIVE_ALERTS_CSV = os.path.join(PROCESSED_DIR, "active_alerts.csv")
ACTIVE_ALERTS_JSON = os.path.join(PROCESSED_DIR, "active_alerts.json")

TERRAIN_MAP = {"plains": 1, "coastal": 2, "hills": 3, "mountains": 4}
RISK_MAP = {"high": 3, "medium": 2, "low": 1}
PROVINCE_MAP = {"Punjab": 1, "Sindh": 2, "KPK": 3, "Balochistan": 4, "AJK": 5, "GB": 6}

BASE_FEATURES = [
    "annual_total_mm",
    "annual_max_daily",
    "monsoon_total_mm",
    "monsoon_avg_daily",
    "monsoon_max_daily",
    "monsoon_rainy_days",
    "premonsoon_total_mm",
    "elevation_m",
    "river_proximity",
    "terrain_code",
    "geo_risk_code",
    "province_code",
    "flood_intensity_score",
    "lat",
    "lon",
    "year",
]

YEAR_CONTEXT_COLS = [
    "annual_total_mm",
    "monsoon_total_mm",
    "monsoon_max_daily",
    "flood_intensity_score",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the daily flood-alert pipeline.")
    parser.add_argument("--as-of", default=None, help="Date to score, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--no-fetch", action="store_true", help="Use local rainfall CSV instead of calling NASA POWER.")
    parser.add_argument(
        "--rainfall-csv",
        default=os.path.join(RAW_DIR, "pakistan_rainfall_2010_2022.csv"),
        help="Fallback rainfall CSV used with --no-fetch.",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        help="Override model optimal_threshold.",
    )
    parser.add_argument(
        "--max-districts",
        type=int,
        default=None,
        help="Debug option: fetch only the first N districts.",
    )
    return parser.parse_args()


def districts_frame():
    df = pd.DataFrame(ALL_DISTRICTS).drop_duplicates(subset=["district", "province"]).copy()
    elevation_path = os.path.join(RAW_DIR, "district_elevation.csv")
    if os.path.exists(elevation_path):
        elev = pd.read_csv(elevation_path)
        keep_cols = [
            "district",
            "province",
            "lat",
            "lon",
            "elevation_m",
            "terrain_type",
            "river_proximity",
            "flood_risk_geo",
        ]
        df = elev[[c for c in keep_cols if c in elev.columns]].drop_duplicates(subset=["district", "province"])
    if "flood_risk_geo" not in df.columns:
        df["flood_risk_geo"] = np.where(
            (df["river_proximity"] == 1) | (df["elevation_m"] < 100),
            "high",
            "medium",
        )
    return df.reset_index(drop=True)


def nasa_power_url(lat, lon, start, end):
    return (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=PRECTOTCORR"
        "&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start}"
        f"&end={end}"
        "&format=JSON"
        "&time-standard=UTC"
    )


def fetch_nasa_current_year(districts, as_of, retries=3):
    start = f"{as_of.year}0101"
    end = as_of.strftime("%Y%m%d")
    rows = []

    print(f"[1/4] Fetching NASA POWER rainfall ({start} to {end})...")
    for idx, row in districts.iterrows():
        district = row["district"]
        lat = row["lat"]
        lon = row["lon"]
        url = nasa_power_url(lat, lon, start, end)

        print(f"      {idx + 1:>3}/{len(districts)} {district:<24}", end="", flush=True)
        ok = False
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    payload = response.json()
                    rain_data = payload["properties"]["parameter"]["PRECTOTCORR"]
                    for date_str, rain_mm in rain_data.items():
                        if rain_mm == -999:
                            rain_mm = 0.0
                        rows.append(
                            {
                                "district": district,
                                "lat": lat,
                                "lon": lon,
                                "date": date_str,
                                "year": int(date_str[:4]),
                                "month": int(date_str[4:6]),
                                "day": int(date_str[6:8]),
                                "rainfall_mm": round(max(0.0, float(rain_mm)), 3),
                            }
                        )
                    ok = True
                    print(" OK", flush=True)
                    break
                if response.status_code == 429:
                    time.sleep(30)
                else:
                    time.sleep(5)
            except Exception:
                time.sleep(5 * attempt)
        if not ok:
            print(" FAILED", flush=True)
        time.sleep(0.5)

    if not rows:
        raise RuntimeError("No rainfall data was fetched. Check internet/API access.")

    rainfall = pd.DataFrame(rows)
    rainfall.to_csv(CURRENT_RAIN_CSV, index=False)
    return rainfall


def load_local_rainfall(path, as_of):
    print(f"[1/4] Loading local rainfall from {path}...")
    rainfall = pd.read_csv(path)
    rainfall["date"] = pd.to_datetime(rainfall["date"].astype(str), format="%Y%m%d", errors="coerce")
    rainfall = rainfall[rainfall["date"].dt.year == as_of.year].copy()
    rainfall = rainfall[rainfall["date"] <= pd.Timestamp(as_of)].copy()
    rainfall["year"] = rainfall["date"].dt.year
    rainfall["month"] = rainfall["date"].dt.month
    rainfall["day"] = rainfall["date"].dt.day
    rainfall["date"] = rainfall["date"].dt.strftime("%Y%m%d")
    if rainfall.empty:
        raise RuntimeError(f"No local rainfall rows found for {as_of.year} through {as_of}.")
    return rainfall


def aggregate_features(rainfall, districts, as_of):
    print("[2/4] Preprocessing rainfall into model features...")
    rainfall = rainfall.copy()
    rainfall["rainfall_mm"] = pd.to_numeric(rainfall["rainfall_mm"], errors="coerce").fillna(0)

    monsoon = (
        rainfall[rainfall["month"].isin([6, 7, 8, 9])]
        .groupby(["district", "year"])["rainfall_mm"]
        .agg(
            monsoon_total_mm="sum",
            monsoon_avg_daily="mean",
            monsoon_max_daily="max",
            monsoon_rainy_days=lambda x: (x > 5).sum(),
        )
        .reset_index()
    )

    annual = (
        rainfall.groupby(["district", "year"])["rainfall_mm"]
        .agg(annual_total_mm="sum", annual_max_daily="max")
        .reset_index()
    )

    premonsoon = (
        rainfall[rainfall["month"].isin([4, 5])]
        .groupby(["district", "year"])["rainfall_mm"]
        .sum()
        .reset_index()
        .rename(columns={"rainfall_mm": "premonsoon_total_mm"})
    )

    base = districts[["district", "province", "lat", "lon", "elevation_m", "terrain_type", "river_proximity", "flood_risk_geo"]].copy()
    base["year"] = as_of.year
    df = base.merge(annual, on=["district", "year"], how="left")
    df = df.merge(monsoon, on=["district", "year"], how="left")
    df = df.merge(premonsoon, on=["district", "year"], how="left")

    df["terrain_code"] = df["terrain_type"].map(TERRAIN_MAP)
    df["geo_risk_code"] = df["flood_risk_geo"].map(RISK_MAP)
    df["province_code"] = df["province"].map(PROVINCE_MAP)
    df.fillna(0, inplace=True)

    df["flood_intensity_score"] = (
        df["monsoon_total_mm"] * (df["river_proximity"] + 1) / (df["elevation_m"] + 1)
    ).round(4)
    return df


def clean_series(series, default=0.0):
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def safe_div(numerator, denominator, default=0.0):
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def map_from_series(values, mapping, default):
    return values.astype(str).map(mapping).fillna(default).astype(float)


def build_model_features(data, info):
    features = info["features"]
    metadata = info.get("feature_metadata", {})
    base_features = info.get("base_features", BASE_FEATURES)

    X = pd.DataFrame(index=data.index)
    for col in base_features:
        X[col] = clean_series(data[col]) if col in data.columns else 0.0

    if metadata:
        X["rain_x_river"] = X["monsoon_total_mm"] * (X["river_proximity"] + 1)
        X["rain_x_geo_risk"] = X["monsoon_total_mm"] * X["geo_risk_code"]
        X["intensity_x_river"] = X["flood_intensity_score"] * (X["river_proximity"] + 1)
        X["elev_inverse"] = safe_div(pd.Series(1.0, index=X.index), X["elevation_m"] + 1)
        X["monsoon_intensity"] = safe_div(X["monsoon_total_mm"], X["monsoon_rainy_days"] + 1)
        X["rainfall_concentration"] = safe_div(X["monsoon_max_daily"], X["monsoon_avg_daily"] + 0.1)
        X["premonsoon_ratio"] = safe_div(X["premonsoon_total_mm"], X["annual_total_mm"] + 1)
        X["low_elevation_high_rain"] = (X["elevation_m"] < 100).astype(int) * X["monsoon_total_mm"]
        X["river_proximity_squared"] = X["river_proximity"] ** 2
        X["terrain_elevation_risk"] = X["terrain_code"] * X["elev_inverse"]
        X["saturation_index"] = safe_div(
            X["premonsoon_total_mm"] * X["monsoon_total_mm"],
            X["elevation_m"] + 1,
        )
        X["annual_to_monsoon_ratio"] = safe_div(X["monsoon_total_mm"], X["annual_total_mm"] + 1)
        X["extreme_daily_share"] = safe_div(X["monsoon_max_daily"], X["monsoon_total_mm"] + 1)

        q = metadata.get("rainfall_quantiles", {})
        X["extreme_rain_indicator"] = (
            X["monsoon_max_daily"] > q.get("monsoon_max_daily_q75", X["monsoon_max_daily"].quantile(0.75))
        ).astype(int)
        X["extreme_monsoon_indicator"] = (
            X["monsoon_total_mm"] > q.get("monsoon_total_mm_q75", X["monsoon_total_mm"].quantile(0.75))
        ).astype(int)
        X["extreme_intensity_indicator"] = (
            X["flood_intensity_score"] > q.get("flood_intensity_score_q75", X["flood_intensity_score"].quantile(0.75))
        ).astype(int)

        districts = data["district"] if "district" in data.columns else pd.Series("", index=data.index)
        provinces = data["province"] if "province" in data.columns else pd.Series("", index=data.index)
        global_rate = metadata.get("global_flood_rate", 0.2)
        X["district_flood_prior"] = map_from_series(
            districts, metadata.get("district_flood_prior", {}), global_rate
        )
        X["province_flood_prior"] = map_from_series(
            provinces, metadata.get("province_flood_prior", {}), global_rate
        )

        for col, stats in metadata.get("district_climatology", {}).items():
            district_mean = map_from_series(districts, stats.get("mean", {}), stats.get("global_mean", 0.0))
            district_std = map_from_series(districts, stats.get("std", {}), stats.get("global_std", 1.0))
            district_std = district_std.replace(0, stats.get("global_std", 1.0) or 1.0)
            X[f"{col}_district_z"] = safe_div(X[col] - district_mean, district_std)

        grouped = data.groupby("year") if "year" in data.columns else None
        for col in YEAR_CONTEXT_COLS:
            if grouped is not None and col in data.columns:
                year_mean = grouped[col].transform("mean")
                year_max = grouped[col].transform("max")
            else:
                year_mean = pd.Series(X[col].mean(), index=data.index)
                year_max = pd.Series(X[col].max(), index=data.index)

            baseline = metadata.get("year_baseline", {}).get(col, {"mean": 0.0, "std": 1.0})
            X[f"{col}_year_mean"] = clean_series(year_mean)
            X[f"{col}_year_max"] = clean_series(year_max)
            X[f"{col}_year_pressure"] = safe_div(
                clean_series(year_mean) - baseline.get("mean", 0.0),
                pd.Series(baseline.get("std", 1.0) or 1.0, index=data.index),
            )
            X[f"{col}_district_vs_year"] = safe_div(X[col], clean_series(year_mean) + 1)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
    return X[features]


def alert_level(score, threshold):
    if score >= max(0.66, threshold + 0.25):
        return "High"
    if score >= threshold:
        return "Watch"
    return "Normal"


def run_predictions(feature_df, as_of, threshold_override=None):
    print("[3/4] Running model predictions...")
    model = joblib.load(os.path.join(MODEL_DIR, "flood_model_best.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    info = joblib.load(os.path.join(MODEL_DIR, "model_info.pkl"))

    threshold = float(threshold_override if threshold_override is not None else info.get("optimal_threshold", 0.5))
    X = build_model_features(feature_df, info)
    scores = model.predict_proba(scaler.transform(X))[:, 1]

    output = feature_df.copy()
    output.insert(0, "run_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    output.insert(1, "as_of_date", as_of.strftime("%Y-%m-%d"))
    output["risk_score"] = scores.round(4)
    output["alert_threshold"] = threshold
    output["alert_level"] = output["risk_score"].apply(lambda score: alert_level(score, threshold))
    output["recommendation"] = np.where(
        output["alert_level"] == "High",
        "Immediate review: verify PMD/FFD bulletin, river gauges, and local reports.",
        np.where(
            output["alert_level"] == "Watch",
            "Monitor closely: check next rainfall update and local conditions.",
            "No model alert.",
        ),
    )
    return output.sort_values("risk_score", ascending=False), threshold


def save_alerts(predictions, threshold):
    print("[4/4] Saving predictions and alerts...")
    predictions.to_csv(LATEST_PREDICTIONS_CSV, index=False)
    alerts = predictions[predictions["risk_score"] >= threshold].copy()
    alerts.to_csv(ACTIVE_ALERTS_CSV, index=False)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "alert_threshold": threshold,
        "alert_count": int(len(alerts)),
        "alerts": alerts[
            [
                "district",
                "province",
                "risk_score",
                "alert_level",
                "monsoon_total_mm",
                "monsoon_max_daily",
                "recommendation",
            ]
        ].head(50).to_dict("records"),
    }
    with open(ACTIVE_ALERTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return alerts


def main():
    args = parse_args()
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else date.today()
    districts = districts_frame()
    if args.max_districts:
        districts = districts.head(args.max_districts).copy()

    if args.no_fetch:
        rainfall = load_local_rainfall(args.rainfall_csv, as_of)
    else:
        rainfall = fetch_nasa_current_year(districts, as_of)

    features = aggregate_features(rainfall, districts, as_of)
    predictions, threshold = run_predictions(features, as_of, args.alert_threshold)
    alerts = save_alerts(predictions, threshold)

    print()
    print("=" * 70)
    print("DAILY FLOOD ALERT PIPELINE COMPLETE")
    print("=" * 70)
    print(f"As of date       : {as_of}")
    print(f"Districts scored : {len(predictions)}")
    print(f"Alert threshold  : {threshold:.2f}")
    print(f"Active alerts    : {len(alerts)}")
    print(f"Predictions CSV  : {LATEST_PREDICTIONS_CSV}")
    print(f"Alerts CSV       : {ACTIVE_ALERTS_CSV}")
    if len(alerts):
        print("\nTop alerts:")
        cols = ["district", "province", "risk_score", "alert_level", "monsoon_total_mm", "monsoon_max_daily"]
        print(alerts[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
