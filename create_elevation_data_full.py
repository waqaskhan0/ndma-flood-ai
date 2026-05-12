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
    elevation = row["elevation_m"]
    river = row["river_proximity"]
    province = row["province"]
    terrain = row["terrain_type"]

    # Sindh: very flat Indus floodplain / coastal areas
    if province == "Sindh" and elevation < 80 and river == 1:
        return "high"

    # South Punjab / Indus belt: river flood risk
    elif province == "Punjab" and elevation < 180 and river == 1:
        return "high"

    # Balochistan: hill torrents / flash floods in low-medium areas
    elif province == "Balochistan" and elevation < 500 and river == 1:
        return "medium"

    # KPK / AJK / GB: flash flood risk in hilly and mountain valleys
    elif province in ["KPK", "AJK", "GB"] and terrain in ["hills", "mountains"] and river == 1:
        return "medium"

    # Coastal districts: urban/coastal flooding
    elif terrain == "coastal" and elevation < 50:
        return "high"

    # General Pakistan rule: low elevation near river
    elif elevation < 150 and river == 1:
        return "high"

    # General Pakistan rule: medium elevation near river
    elif elevation < 400 and river == 1:
        return "medium"

    # Flat lowland areas even if not directly near river
    elif elevation < 250 and terrain in ["plains", "coastal"]:
        return "medium"

    # Otherwise lower flood susceptibility
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