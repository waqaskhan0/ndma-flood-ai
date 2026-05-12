"""
Step 1A: Create elevation data for ALL 160+ Pakistan districts
==============================================================
Run:
    python create_elevation_data_full.py

Output:
    data/raw/district_elevation.csv
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from all_districts_data import ALL_DISTRICTS

os.makedirs("data/raw", exist_ok=True)

df = pd.DataFrame(ALL_DISTRICTS)

# Remove duplicates (some districts listed twice)
df = df.drop_duplicates(subset=["district", "province"]).reset_index(drop=True)

# Compute flood_risk_geo
def compute_risk(row):
    if row["elevation_m"] < 100 and row["river_proximity"] == 1:
        return "high"
    elif row["elevation_m"] < 300 and row["river_proximity"] == 1:
        return "medium"
    elif row["elevation_m"] < 300:
        return "medium"
    else:
        return "low"

df["flood_risk_geo"] = df.apply(compute_risk, axis=1)

df.to_csv("data/raw/district_elevation.csv", index=False)

print("=" * 55)
print("  district_elevation.csv — ALL DISTRICTS")
print("=" * 55)
print(f"\n  Total districts : {len(df)}")
print(f"\n  By province:")
for prov, grp in df.groupby("province"):
    print(f"    {prov:<20} : {len(grp)} districts")
print(f"\n  Terrain breakdown:")
for t, c in df["terrain_type"].value_counts().items():
    print(f"    {t:<12} : {c}")
print(f"\n  Flood risk:")
for r, c in df["flood_risk_geo"].value_counts().items():
    print(f"    {r:<8} : {c}")
print(f"\n  Elevation: {df['elevation_m'].min()}m (min) — {df['elevation_m'].max()}m (max)")
print(f"\n  Saved to: data/raw/district_elevation.csv")
print(f"\n  Next → run: python create_flood_labels_full.py")