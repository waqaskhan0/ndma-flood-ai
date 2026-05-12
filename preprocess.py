"""
Step 2: Data Preprocessing & Feature Engineering
=================================================
Run this AFTER you have all 3 files in data/raw/:
  - flood_labels.csv
  - district_elevation.csv
  - pakistan_rainfall_2010_2022.csv  (or it auto-generates sample data)

Run:
    python preprocess_data.py

Output:
    data/processed/flood_features.csv   <- ready for ML training
"""

import pandas as pd
import numpy as np
import os
import calendar

os.makedirs("data/processed", exist_ok=True)

print("=" * 55)
print("  STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 55)

# ─────────────────────────────────────────────────
# 1. LOAD FLOOD LABELS
# ─────────────────────────────────────────────────
print("\n[1/6] Loading flood labels...")
labels = pd.read_csv("data/raw/flood_labels.csv")
print(f"      {len(labels)} rows — {labels['district'].nunique()} districts x {labels['year'].nunique()} years")

# ─────────────────────────────────────────────────
# 2. LOAD ELEVATION DATA
# ─────────────────────────────────────────────────
print("\n[2/6] Loading elevation data...")
elevation = pd.read_csv("data/raw/district_elevation.csv")
print(f"      {len(elevation)} districts loaded")

# ─────────────────────────────────────────────────
# 3. LOAD OR GENERATE RAINFALL DATA
# ─────────────────────────────────────────────────
print("\n[3/6] Loading rainfall data...")
rainfall_path = "data/raw/pakistan_rainfall_2010_2022.csv"

if os.path.exists(rainfall_path):
    rainfall_raw = pd.read_csv(rainfall_path)
    rainfall_raw["date"] = pd.to_datetime(rainfall_raw["date"], format="%Y%m%d", errors="coerce")
    rainfall_raw["year"]  = rainfall_raw["date"].dt.year
    rainfall_raw["month"] = rainfall_raw["date"].dt.month
    print(f"      Real rainfall loaded: {len(rainfall_raw)} rows")
else:
    print("      Rainfall CSV not found — generating realistic sample data")
    print("      (Run download_rainfall.py later for real NASA data)")

    districts = labels["district"].unique()
    monthly_weights = {
        1:0.5, 2:0.4, 3:0.6, 4:0.5, 5:0.4, 6:1.2,
        7:4.0, 8:4.5, 9:2.5, 10:0.8, 11:0.3, 12:0.4
    }
    flood_year_mult = {
        2010:2.2, 2011:1.8, 2012:1.3, 2013:0.8,
        2014:1.4, 2015:1.1, 2016:1.0, 2017:1.2,
        2018:0.9, 2019:1.1, 2020:1.5, 2021:1.0, 2022:2.5
    }
    rows = []
    np.random.seed(42)
    for year in range(2010, 2023):
        for month in range(1, 13):
            days = calendar.monthrange(year, month)[1]
            for district in districts:
                base = monthly_weights[month] * flood_year_mult[year]
                for day in range(1, days + 1):
                    rain = max(0, np.random.exponential(base))
                    rows.append({
                        "district": district, "year": year,
                        "month": month, "rainfall_mm": round(rain, 2)
                    })
    rainfall_raw = pd.DataFrame(rows)
    print(f"      Generated {len(rainfall_raw)} sample rainfall rows")

# ─────────────────────────────────────────────────
# 4. ENGINEER RAINFALL FEATURES
# ─────────────────────────────────────────────────
print("\n[4/6] Engineering rainfall features...")

if "year" not in rainfall_raw.columns:
    rainfall_raw["date"]  = pd.to_datetime(rainfall_raw["date"])
    rainfall_raw["year"]  = rainfall_raw["date"].dt.year
    rainfall_raw["month"] = rainfall_raw["date"].dt.month

# Monsoon season (Jun–Sep) features
monsoon = (
    rainfall_raw[rainfall_raw["month"].isin([6,7,8,9])]
    .groupby(["district","year"])["rainfall_mm"]
    .agg(
        monsoon_total_mm  = "sum",
        monsoon_avg_daily = "mean",
        monsoon_max_daily = "max",
        monsoon_rainy_days= lambda x: (x > 5).sum()
    )
    .reset_index()
)

# Annual totals
annual = (
    rainfall_raw
    .groupby(["district","year"])["rainfall_mm"]
    .agg(annual_total_mm="sum", annual_max_daily="max")
    .reset_index()
)

# Pre-monsoon (Apr–May) — soil saturation indicator
premonsoon = (
    rainfall_raw[rainfall_raw["month"].isin([4,5])]
    .groupby(["district","year"])["rainfall_mm"]
    .sum()
    .reset_index()
    .rename(columns={"rainfall_mm":"premonsoon_total_mm"})
)

print(f"      Monsoon features  : {monsoon.shape}")
print(f"      Annual features   : {annual.shape}")
print(f"      Pre-monsoon       : {premonsoon.shape}")

# ─────────────────────────────────────────────────
# 5. MERGE ALL SOURCES
# ─────────────────────────────────────────────────
print("\n[5/6] Merging all data sources...")

df = labels.copy()
df = df.merge(annual,     on=["district","year"], how="left")
df = df.merge(monsoon,    on=["district","year"], how="left")
df = df.merge(premonsoon, on=["district","year"], how="left")

elev_cols = ["district","province","lat","lon",
             "elevation_m","terrain_type","river_proximity","flood_risk_geo"]
df = df.merge(elevation[elev_cols], on="district", how="left")

# ─────────────────────────────────────────────────
# 6. DERIVED FEATURES + ENCODING
# ─────────────────────────────────────────────────
print("\n[6/6] Adding derived features...")

terrain_map  = {"plains":1, "coastal":2, "hills":3, "mountains":4}
risk_map     = {"high":3, "medium":2, "low":1}
province_map = {"Punjab":1,"Sindh":2,"KPK":3,"Balochistan":4,"AJK":5,"GB":6}

df["terrain_code"]  = df["terrain_type"].map(terrain_map)
df["geo_risk_code"] = df["flood_risk_geo"].map(risk_map)
df["province_code"] = df["province"].map(province_map)

# Key composite feature: rain intensity + proximity / elevation
df["flood_intensity_score"] = (
    (
        df["monsoon_total_mm"] * 0.6
        + df["annual_max_daily"] * 2
        + df["monsoon_rainy_days"] * 1.5
    )
    *
    (df["river_proximity"] + 1)
    *
    df["geo_risk_code"]
    /
    np.sqrt(df["elevation_m"] + 1)
).round(4)

df.fillna(0, inplace=True)
float_cols = df.select_dtypes(include="float64").columns
df[float_cols] = df[float_cols].round(4)

# ─────────────────────────────────────────────────
# FLOOD INTENSITY CATEGORY
# ─────────────────────────────────────────────────

def classify_intensity(score):

    if score < 15:
        return "very_low"

    elif score < 50:
        return "low"

    elif score < 100:
        return "moderate"

    elif score < 250:
        return "high"

    elif score < 500:
        return "very_high"

    else:
        return "extreme"


# Create category column
df["flood_intensity_category"] = (
    df["flood_intensity_score"]
    .apply(classify_intensity)
)

# Optional numeric encoding
intensity_map = {
    "very_low": 1,
    "low": 2,
    "moderate": 3,
    "high": 4,
    "very_high": 5,
    "extreme": 6
}

df["flood_intensity_code"] = (
    df["flood_intensity_category"]
    .map(intensity_map)
)

# Show distribution
print("\nFlood Intensity Distribution:")
print(df["flood_intensity_category"].value_counts())

df.to_csv("data/processed/flood_features.csv", index=False)

# ─────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────
print()
print("=" * 55)
print("  PREPROCESSING COMPLETE!")
print("=" * 55)
print(f"\n  Saved to : data/processed/flood_features.csv")
print(f"  Shape    : {df.shape[0]} rows x {df.shape[1]} columns")

print(f"\n  All features:")
skip = {"district","year","flooded","province","terrain_type","flood_risk_geo"}
for col in df.columns:
    if col not in skip:
        print(f"    - {col}")

flooded     = int(df["flooded"].sum())
not_flooded = int((df["flooded"]==0).sum())
total       = len(df)
print(f"\n  Class balance:")
print(f"    Flooded     : {flooded}  ({flooded/total*100:.1f}%)")
print(f"    Not flooded : {not_flooded} ({not_flooded/total*100:.1f}%)")

if flooded/total < 0.35:
    print("    NOTE: Imbalanced — train_model.py handles this automatically")

print(f"\n  Preview:")
print(df[["district","year","monsoon_total_mm","elevation_m",
          "river_proximity","flood_intensity_score","flooded"]].head(8).to_string(index=False))

print("\n  Next step → run:   python train_model.py")