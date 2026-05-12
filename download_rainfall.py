"""
Step 1C: Download REAL Rainfall Data from NASA POWER API
=========================================================
Downloads daily precipitation for ALL 160 districts (2010–2022).
No API key needed. Completely free.

Run:
    python download_rainfall.py

Time estimate:
    ~25–40 minutes total (160 districts × ~10 seconds each)
    The script saves progress so if it stops you can resume.

Output:
    data/raw/pakistan_rainfall_2010_2022.csv
"""

import requests
import pandas as pd
import os
import time
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from all_districts_data import ALL_DISTRICTS

os.makedirs("data/raw",       exist_ok=True)
os.makedirs("data/raw/cache", exist_ok=True)

# Remove duplicate districts
import pandas as _pd
_df = _pd.DataFrame(ALL_DISTRICTS).drop_duplicates(subset=["district","province"])
DISTRICTS = _df[["district","lat","lon"]].to_dict("records")

OUTPUT_CSV     = "data/raw/pakistan_rainfall_2010_2022.csv"
PROGRESS_FILE  = "data/raw/cache/download_progress.json"

# ─────────────────────────────────────────────────────────────
# Load progress (so we can resume if interrupted)
# ─────────────────────────────────────────────────────────────
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
    completed = set(progress.get("completed", []))
    print(f"  Resuming — {len(completed)} districts already downloaded")
else:
    completed = set()
    progress  = {"completed": []}

# Load existing data if any
if os.path.exists(OUTPUT_CSV) and len(completed) > 0:
    all_data = pd.read_csv(OUTPUT_CSV).to_dict("records")
    print(f"  Loaded {len(all_data)} existing rows")
else:
    all_data = []

# ─────────────────────────────────────────────────────────────
# Download function
# ─────────────────────────────────────────────────────────────
def download_district(district, lat, lon, retries=3):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=PRECTOTCORR"
        "&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        "&start=20100101"
        "&end=20221231"
        "&format=JSON"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                data = r.json()
                rain_data = data["properties"]["parameter"]["PRECTOTCORR"]
                rows = []
                for date_str, rain_mm in rain_data.items():
                    if rain_mm == -999:   # NASA uses -999 for missing
                        rain_mm = 0.0
                    rows.append({
                        "district":    district,
                        "lat":         lat,
                        "lon":         lon,
                        "date":        date_str,
                        "year":        int(date_str[:4]),
                        "month":       int(date_str[4:6]),
                        "day":         int(date_str[6:8]),
                        "rainfall_mm": round(max(0.0, float(rain_mm)), 3),
                    })
                return rows
            elif r.status_code == 429:
                print(f"        Rate limited — waiting 30s...")
                time.sleep(30)
            else:
                print(f"        HTTP {r.status_code} — retrying ({attempt+1}/{retries})")
                time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"        Timeout — retrying ({attempt+1}/{retries})")
            time.sleep(10)
        except Exception as e:
            print(f"        Error: {e} — retrying ({attempt+1}/{retries})")
            time.sleep(5)
    return None

# ─────────────────────────────────────────────────────────────
# Main download loop
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  NASA POWER RAINFALL DOWNLOADER — ALL DISTRICTS")
print("=" * 60)
print(f"\n  Total districts : {len(DISTRICTS)}")
print(f"  Period          : 2010-01-01 to 2022-12-31")
print(f"  Already done    : {len(completed)}")
print(f"  Remaining       : {len(DISTRICTS) - len(completed)}")
print(f"\n  Starting download... (Ctrl+C to pause, re-run to resume)\n")

todo = [d for d in DISTRICTS if d["district"] not in completed]

for i, dist in enumerate(todo, 1):
    name = dist["district"]
    lat  = dist["lat"]
    lon  = dist["lon"]

    print(f"  [{i:>3}/{len(todo)}] Downloading {name} ({lat:.2f}, {lon:.2f})...", end=" ", flush=True)

    rows = download_district(name, lat, lon)

    if rows:
        all_data.extend(rows)
        completed.add(name)
        progress["completed"] = list(completed)

        # Save progress every 5 districts
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)

        # Save CSV every 10 districts
        if i % 10 == 0 or i == len(todo):
            df_save = pd.DataFrame(all_data)
            df_save.to_csv(OUTPUT_CSV, index=False)
            print(f"OK ({len(rows)} days) — progress saved", flush=True)
        else:
            print(f"OK ({len(rows)} days)", flush=True)
    else:
        print(f"FAILED — skipping {name}", flush=True)

    # Polite delay between requests
    time.sleep(1.5)

# ─────────────────────────────────────────────────────────────
# Final save
# ─────────────────────────────────────────────────────────────
if all_data:
    df_final = pd.DataFrame(all_data)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print()
    print("=" * 60)
    print("  DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\n  Rows downloaded  : {len(df_final):,}")
    print(f"  Districts        : {df_final['district'].nunique()}")
    print(f"  Date range       : {df_final['date'].min()} → {df_final['date'].max()}")
    print(f"  Avg daily rain   : {df_final['rainfall_mm'].mean():.2f}mm")
    print(f"  Max single day   : {df_final['rainfall_mm'].max():.1f}mm")
    print(f"\n  File saved to    : {OUTPUT_CSV}")
    print(f"  File size        : {os.path.getsize(OUTPUT_CSV)/1024/1024:.1f} MB")
    print(f"\n  Next → run: python preprocess_data.py")
else:
    print("\n  No data downloaded. Check your internet connection.")