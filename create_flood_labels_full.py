"""
Step 1B: Create flood labels for ALL 160+ Pakistan districts
=============================================================
Run:
    python create_flood_labels_full.py

Output:
    data/raw/flood_labels.csv
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from all_districts_data import ALL_DISTRICTS

os.makedirs("data/raw", exist_ok=True)

# All district names
import pandas as _pd
_df = _pd.DataFrame(ALL_DISTRICTS).drop_duplicates(subset=["district","province"])
ALL_DISTRICT_NAMES = _df["district"].tolist()

# ─────────────────────────────────────────────────────────────
# FLOOD EVENTS — based on NDMA annual reports + ReliefWeb
# ─────────────────────────────────────────────────────────────
FLOOD_EVENTS = {
    2010: [  # Worst floods in history — 1/5th of Pakistan submerged
        # Punjab
        "Multan","Muzaffargarh","DG Khan","Dera Ghazi Khan","Rahim Yar Khan","Bahawalpur",
        "Jhang","Bhakkar","Layyah","Lodhran","Khanewal","Vehari",
        # Sindh — entire province affected
        "Sukkur","Larkana","Jacobabad","Kashmore","Kamber","Shikarpur","Dadu","Thatta",
        "Hyderabad","Mirpur Khas","Sanghar","Nawabshah","Naushahro Feroze","Khairpur",
        "Ghotki","Qambar Shahdadkot","Jamshoro","Matiari",
        # KPK — flash floods
        "Swat","Nowshera","Charsadda","Dir Upper","Dir Lower","Kohistan","Chitral",
        "Malakand","Shangla","Buner","Battagram","Mansehra","Kolai Pallas","Torghar",
        # Balochistan
        "Naseerabad","Jaffarabad","Dera Bugti","Jhal Magsi","Sibi","Lehri",
        # AJK / GB
        "Muzaffarabad","Neelum","Hattian Bala",
        "Diamer","Roundu","Tangir","Darel",
    ],
    2011: [  # Sindh floods — second consecutive year
        "Sukkur","Larkana","Jacobabad","Kashmore","Kamber","Shikarpur","Dadu","Thatta",
        "Hyderabad","Mirpur Khas","Sanghar","Nawabshah","Naushahro Feroze","Khairpur",
        "Ghotki","Badin","Tando Allahyar","Tando Muhammad Khan","Qambar Shahdadkot",
        "Matiari","Jamshoro","Sujawal",
        "Naseerabad","Jaffarabad","Jhal Magsi","Dera Bugti",
        "Rahim Yar Khan","Bahawalpur","Lodhran",
        "Karachi Central","Karachi West","Malir",
    ],
    2012: [  # Moderate — mainly Sindh and south Punjab
        "Sukkur","Larkana","Dadu","Kashmore","Kamber","Nawabshah","Sanghar","Khairpur",
        "Rahim Yar Khan","Bahawalpur","DG Khan","Dera Ghazi Khan","Muzaffargarh",
        "Naseerabad","Jaffarabad","Jhal Magsi",
    ],
    2013: [  # KPK and GB flash floods
        "Swat","Dir Upper","Dir Lower","Chitral","Kohistan","Shangla","Battagram",
        "Muzaffarabad","Neelum",
        "Gilgit","Diamer","Astore","Ghizer",
    ],
    2014: [  # Punjab and KPK
        "Lahore","Gujranwala","Sialkot","Gujrat","Sheikhupura","Narowal",
        "Multan","Muzaffargarh","DG Khan","Dera Ghazi Khan","Layyah","Bhakkar",
        "Nowshera","Charsadda","Peshawar","Mardan","Swabi",
        "Muzaffarabad","Bagh","Hattian Bala","Jhelum Valley","Mirpur","Kotli",
    ],
    2015: [  # KPK and GB flash floods
        "Swat","Dir Upper","Dir Lower","Kohistan","Chitral","Shangla","Malakand",
        "Battagram","Mansehra","Torghar","Kolai Pallas",
        "Gilgit","Hunza","Nagar","Diamer","Astore","Ghizer","Gupis Yasin",
        "Muzaffarabad","Neelum",
    ],
    2016: [  # Moderate — localized
        "Multan","Bahawalpur","Rahim Yar Khan","Lodhran","Muzaffargarh",
        "Sukkur","Larkana","Nawabshah","Khairpur",
        "Naseerabad","Jaffarabad",
    ],
    2017: [  # Multiple provinces
        "Lahore","Gujranwala","Sheikhupura","Hafizabad",
        "DG Khan","Dera Ghazi Khan","Muzaffargarh","Layyah",
        "Nowshera","Charsadda","Swat","Mardan","Swabi",
        "Naseerabad","Jaffarabad","Dera Bugti","Jhal Magsi",
        "Sukkur","Kashmore","Ghotki",
    ],
    2018: [  # KPK and Balochistan
        "Swat","Dir Upper","Dir Lower","Kohistan","Chitral","Shangla","Battagram",
        "Mansehra","Torghar",
        "Naseerabad","Jaffarabad","Lasbela","Kalat","Khuzdar","Awaran",
        "Gwadar","Turbat","Panjgur",
        "Gilgit","Hunza","Diamer",
    ],
    2019: [  # Sindh and Balochistan
        "Sukkur","Jacobabad","Kashmore","Dadu","Nawabshah","Khairpur",
        "Naseerabad","Jaffarabad","Lasbela","Jhal Magsi","Dera Bugti","Lehri",
        "Tank","Lakki Marwat","Dera Ismail Khan",
    ],
    2020: [  # Severe monsoon — Sindh, Balochistan, KPK
        "Sukkur","Larkana","Jacobabad","Kashmore","Kamber","Dadu","Thatta","Badin",
        "Nawabshah","Sanghar","Khairpur","Ghotki","Qambar Shahdadkot","Hyderabad",
        "Tando Allahyar","Mirpur Khas","Matiari","Jamshoro","Sujawal",
        "Naseerabad","Jaffarabad","Lasbela","Kalat","Khuzdar","Jhal Magsi",
        "Dera Bugti","Awaran","Kech",
        "Swat","Nowshera","Tank","Dera Ismail Khan","Lakki Marwat",
        "DG Khan","Dera Ghazi Khan","Rahim Yar Khan","Lodhran","Muzaffargarh",
    ],
    2021: [  # KPK and AJK / GB
        "Swat","Dir Upper","Dir Lower","Kohistan","Chitral","Shangla","Malakand",
        "Battagram","Mansehra","Torghar","Kolai Pallas",
        "Muzaffarabad","Neelum","Bagh","Hattian Bala","Jhelum Valley",
        "Gilgit","Hunza","Diamer","Astore","Skardu","Ghanche",
    ],
    2022: [  # CATASTROPHIC — 1/3 of Pakistan underwater, 33M affected
        # All of Sindh
        "Karachi Central","Karachi East","Karachi West","Karachi South","Malir",
        "Hyderabad","Sukkur","Larkana","Thatta","Dadu","Jacobabad","Kashmore",
        "Kamber","Shikarpur","Mirpur Khas","Sanghar","Nawabshah","Naushahro Feroze",
        "Khairpur","Ghotki","Badin","Tharparkar","Umerkot","Matiari","Jamshoro",
        "Sujawal","Tando Allahyar","Tando Muhammad Khan","Qambar Shahdadkot",
        # South & central Punjab
        "Multan","Muzaffargarh","DG Khan","Dera Ghazi Khan","Rahim Yar Khan","Bahawalpur",
        "Bahawalnagar","Lodhran","Vehari","Layyah","Bhakkar","Jhang","Khanewal",
        # KPK
        "Swat","Nowshera","Charsadda","Dir Upper","Dir Lower","Kohistan","Tank",
        "Chitral","Shangla","Malakand","Battagram","Mansehra","Dera Ismail Khan",
        "Lakki Marwat","Bannu","Torghar","Kolai Pallas","Mohmand","Bajaur",
        # Balochistan
        "Naseerabad","Jaffarabad","Lasbela","Kalat","Khuzdar","Jhal Magsi",
        "Dera Bugti","Sibi","Awaran","Kech","Turbat","Panjgur","Gwadar","Harnai",
        "Quetta","Mastung","Pishin","Loralai","Barkhan","Musakhel","Lehri",
        # AJK / GB
        "Muzaffarabad","Neelum","Hattian Bala","Bagh","Jhelum Valley",
        "Gilgit","Hunza","Diamer","Astore","Roundu",
    ],
}

# Build CSV
rows = []
for year in range(2010, 2023):
    flooded_this_year = FLOOD_EVENTS.get(year, [])
    for district in ALL_DISTRICT_NAMES:
        rows.append({
            "district": district,
            "year":     year,
            "flooded":  1 if district in flooded_this_year else 0,
        })

df = pd.DataFrame(rows)
df.to_csv("data/raw/flood_labels.csv", index=False)

print("=" * 55)
print("  flood_labels.csv — ALL DISTRICTS")
print("=" * 55)
print(f"\n  Total rows      : {len(df)}")
print(f"  Districts       : {df['district'].nunique()}")
print(f"  Years           : 2010–2022")
print(f"  Flooded records : {df['flooded'].sum()}")
print(f"  Not flooded     : {(df['flooded']==0).sum()}")
print(f"\n  Floods per year:")
for year, grp in df[df["flooded"]==1].groupby("year"):
    bar = "█" * len(grp)
    print(f"    {year}: {bar} ({len(grp)})")
print(f"\n  Saved to: data/raw/flood_labels.csv")
print(f"\n  Next → run: python download_rainfall.py")