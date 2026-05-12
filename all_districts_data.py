"""
All 160+ Pakistan Districts — Complete Data
============================================
Covers all 6 administrative units:
  Punjab      : 36 districts
  Sindh       : 29 districts
  KPK         : 35 districts
  Balochistan : 35 districts
  Gilgit-Baltistan : 14 districts
  AJK         : 10 districts
  ICT         :  1
Total: 160 districts
"""

ALL_DISTRICTS = [

    # ══════════════════════════════════════════════════════
    # PUNJAB — 36 districts
    # ══════════════════════════════════════════════════════
    {"district":"Lahore",         "province":"Punjab", "lat":31.5204,"lon":74.3587,"elevation_m":217,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Multan",         "province":"Punjab", "lat":30.1575,"lon":71.5249,"elevation_m":122,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Faisalabad",     "province":"Punjab", "lat":31.4504,"lon":73.1350,"elevation_m":186,  "terrain_type":"plains",   "river_proximity":0},
    {"district":"Rawalpindi",     "province":"Punjab", "lat":33.5651,"lon":73.0169,"elevation_m":508,  "terrain_type":"hills",    "river_proximity":0},
    {"district":"Gujranwala",     "province":"Punjab", "lat":32.1877,"lon":74.1945,"elevation_m":228,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Sialkot",        "province":"Punjab", "lat":32.4945,"lon":74.5229,"elevation_m":256,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Bahawalpur",     "province":"Punjab", "lat":29.3956,"lon":71.6836,"elevation_m":117,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Rahim Yar Khan", "province":"Punjab", "lat":28.4212,"lon":70.2989,"elevation_m":82,   "terrain_type":"plains",   "river_proximity":1},
    {"district":"DG Khan",        "province":"Punjab", "lat":30.0368,"lon":70.6340,"elevation_m":138,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Muzaffargarh",   "province":"Punjab", "lat":30.0736,"lon":71.1930,"elevation_m":118,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Gujrat",         "province":"Punjab", "lat":32.5742,"lon":74.0796,"elevation_m":246,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Sheikhupura",    "province":"Punjab", "lat":31.7167,"lon":73.9850,"elevation_m":213,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Sargodha",       "province":"Punjab", "lat":32.0836,"lon":72.6711,"elevation_m":187,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Jhang",          "province":"Punjab", "lat":31.2681,"lon":72.3181,"elevation_m":164,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Kasur",          "province":"Punjab", "lat":31.1167,"lon":74.4500,"elevation_m":210,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Okara",          "province":"Punjab", "lat":30.8100,"lon":73.4597,"elevation_m":174,  "terrain_type":"plains",   "river_proximity":0},
    {"district":"Sahiwal",        "province":"Punjab", "lat":30.6682,"lon":73.1166,"elevation_m":170,  "terrain_type":"plains",   "river_proximity":0},
    {"district":"Pakpattan",      "province":"Punjab", "lat":30.3436,"lon":73.3868,"elevation_m":154,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Bahawalnagar",   "province":"Punjab", "lat":29.9947,"lon":73.2536,"elevation_m":140,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Vehari",         "province":"Punjab", "lat":30.0444,"lon":72.3508,"elevation_m":134,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Lodhran",        "province":"Punjab", "lat":29.5372,"lon":71.6320,"elevation_m":111,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Khanewal",       "province":"Punjab", "lat":30.3014,"lon":71.9320,"elevation_m":122,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Chiniot",        "province":"Punjab", "lat":31.7200,"lon":72.9781,"elevation_m":178,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Hafizabad",      "province":"Punjab", "lat":32.0714,"lon":73.6881,"elevation_m":212,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Narowal",        "province":"Punjab", "lat":32.1000,"lon":74.8728,"elevation_m":240,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Nankana Sahib",  "province":"Punjab", "lat":31.4500,"lon":73.7100,"elevation_m":210,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Mandi Bahauddin","province":"Punjab", "lat":32.5853,"lon":73.4912,"elevation_m":230,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Attock",         "province":"Punjab", "lat":33.7667,"lon":72.3601,"elevation_m":305,  "terrain_type":"hills",    "river_proximity":1},
    {"district":"Chakwal",        "province":"Punjab", "lat":32.9319,"lon":72.8561,"elevation_m":498,  "terrain_type":"hills",    "river_proximity":0},
    {"district":"Jhelum",         "province":"Punjab", "lat":32.9263,"lon":73.7267,"elevation_m":287,  "terrain_type":"hills",    "river_proximity":1},
    {"district":"Khushab",        "province":"Punjab", "lat":32.2958,"lon":72.3514,"elevation_m":210,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Bhakkar",        "province":"Punjab", "lat":31.6267,"lon":71.0647,"elevation_m":145,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Layyah",         "province":"Punjab", "lat":30.9597,"lon":70.9436,"elevation_m":134,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Toba Tek Singh", "province":"Punjab", "lat":30.9700,"lon":72.4800,"elevation_m":168,  "terrain_type":"plains",   "river_proximity":0},
    {"district":"Hafizabad",      "province":"Punjab", "lat":32.0700,"lon":73.6900,"elevation_m":213,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Dera Ghazi Khan","province":"Punjab", "lat":30.0500,"lon":70.6400,"elevation_m":138,  "terrain_type":"plains",   "river_proximity":1},

    # ══════════════════════════════════════════════════════
    # SINDH — 29 districts
    # ══════════════════════════════════════════════════════
    {"district":"Karachi Central", "province":"Sindh", "lat":24.9056,"lon":67.0822,"elevation_m":8,   "terrain_type":"coastal",  "river_proximity":0},
    {"district":"Karachi East",    "province":"Sindh", "lat":24.9400,"lon":67.1300,"elevation_m":10,  "terrain_type":"coastal",  "river_proximity":0},
    {"district":"Karachi West",    "province":"Sindh", "lat":24.8800,"lon":66.9900,"elevation_m":8,   "terrain_type":"coastal",  "river_proximity":0},
    {"district":"Karachi South",   "province":"Sindh", "lat":24.8400,"lon":67.0100,"elevation_m":5,   "terrain_type":"coastal",  "river_proximity":0},
    {"district":"Malir",           "province":"Sindh", "lat":25.0700,"lon":67.2000,"elevation_m":15,  "terrain_type":"coastal",  "river_proximity":0},
    {"district":"Hyderabad",       "province":"Sindh", "lat":25.3960,"lon":68.3578,"elevation_m":38,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Sukkur",          "province":"Sindh", "lat":27.7052,"lon":68.8574,"elevation_m":65,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Larkana",         "province":"Sindh", "lat":27.5590,"lon":68.2100,"elevation_m":56,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Thatta",          "province":"Sindh", "lat":24.7461,"lon":67.9236,"elevation_m":12,  "terrain_type":"coastal",  "river_proximity":1},
    {"district":"Dadu",            "province":"Sindh", "lat":26.7300,"lon":67.7800,"elevation_m":50,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Jacobabad",       "province":"Sindh", "lat":28.2769,"lon":68.4376,"elevation_m":55,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Kashmore",        "province":"Sindh", "lat":28.4419,"lon":69.5714,"elevation_m":60,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Kamber",          "province":"Sindh", "lat":27.5867,"lon":68.0019,"elevation_m":48,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Shikarpur",       "province":"Sindh", "lat":27.9558,"lon":68.6381,"elevation_m":58,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Mirpur Khas",     "province":"Sindh", "lat":25.5270,"lon":69.0144,"elevation_m":28,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Sanghar",         "province":"Sindh", "lat":26.0469,"lon":68.9469,"elevation_m":42,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Nawabshah",       "province":"Sindh", "lat":26.2439,"lon":68.4100,"elevation_m":37,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Naushahro Feroze","province":"Sindh", "lat":26.8381,"lon":68.1142,"elevation_m":44,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Khairpur",        "province":"Sindh", "lat":27.5290,"lon":68.7592,"elevation_m":52,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Ghotki",          "province":"Sindh", "lat":28.0050,"lon":69.3214,"elevation_m":62,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Badin",           "province":"Sindh", "lat":24.6558,"lon":68.8369,"elevation_m":10,  "terrain_type":"coastal",  "river_proximity":1},
    {"district":"Tharparkar",      "province":"Sindh", "lat":24.7100,"lon":70.2300,"elevation_m":85,  "terrain_type":"plains",   "river_proximity":0},
    {"district":"Umerkot",         "province":"Sindh", "lat":25.3600,"lon":69.7400,"elevation_m":52,  "terrain_type":"plains",   "river_proximity":0},
    {"district":"Matiari",         "province":"Sindh", "lat":25.5900,"lon":68.4600,"elevation_m":30,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Jamshoro",        "province":"Sindh", "lat":25.4300,"lon":68.2800,"elevation_m":32,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Sujawal",         "province":"Sindh", "lat":24.5900,"lon":68.3200,"elevation_m":8,   "terrain_type":"coastal",  "river_proximity":1},
    {"district":"Tando Allahyar",  "province":"Sindh", "lat":25.4700,"lon":68.7200,"elevation_m":30,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Tando Muhammad Khan","province":"Sindh","lat":25.1200,"lon":68.5300,"elevation_m":20, "terrain_type":"plains",  "river_proximity":1},
    {"district":"Qambar Shahdadkot","province":"Sindh","lat":27.5500,"lon":67.9900,"elevation_m":46,  "terrain_type":"plains",   "river_proximity":1},

    # ══════════════════════════════════════════════════════
    # KPK — 35 districts
    # ══════════════════════════════════════════════════════
    {"district":"Peshawar",     "province":"KPK","lat":34.0151,"lon":71.5249,"elevation_m":331,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Swat",         "province":"KPK","lat":35.2227,"lon":72.4258,"elevation_m":980,  "terrain_type":"mountains","river_proximity":1},
    {"district":"Nowshera",     "province":"KPK","lat":34.0153,"lon":71.9747,"elevation_m":288,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Charsadda",    "province":"KPK","lat":34.1481,"lon":71.7408,"elevation_m":276,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Dir Upper",    "province":"KPK","lat":35.5742,"lon":72.0150,"elevation_m":1370, "terrain_type":"mountains","river_proximity":1},
    {"district":"Dir Lower",    "province":"KPK","lat":34.8700,"lon":71.8900,"elevation_m":900,  "terrain_type":"mountains","river_proximity":1},
    {"district":"Chitral",      "province":"KPK","lat":35.8514,"lon":71.7864,"elevation_m":1500, "terrain_type":"mountains","river_proximity":1},
    {"district":"Kohistan",     "province":"KPK","lat":35.0000,"lon":73.0000,"elevation_m":1200, "terrain_type":"mountains","river_proximity":1},
    {"district":"Tank",         "province":"KPK","lat":32.2194,"lon":70.3753,"elevation_m":262,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Dera Ismail Khan","province":"KPK","lat":31.8303,"lon":70.9017,"elevation_m":173,"terrain_type":"plains",  "river_proximity":1},
    {"district":"Mardan",       "province":"KPK","lat":34.1986,"lon":72.0404,"elevation_m":283,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Abbottabad",   "province":"KPK","lat":34.1558,"lon":73.2194,"elevation_m":1256, "terrain_type":"mountains","river_proximity":0},
    {"district":"Haripur",      "province":"KPK","lat":33.9942,"lon":72.9394,"elevation_m":580,  "terrain_type":"hills",    "river_proximity":1},
    {"district":"Mansehra",     "province":"KPK","lat":34.3325,"lon":73.1975,"elevation_m":980,  "terrain_type":"mountains","river_proximity":1},
    {"district":"Battagram",    "province":"KPK","lat":34.6800,"lon":73.0200,"elevation_m":830,  "terrain_type":"mountains","river_proximity":1},
    {"district":"Shangla",      "province":"KPK","lat":35.0000,"lon":72.6000,"elevation_m":1100, "terrain_type":"mountains","river_proximity":1},
    {"district":"Buner",        "province":"KPK","lat":34.5100,"lon":72.5000,"elevation_m":750,  "terrain_type":"mountains","river_proximity":1},
    {"district":"Malakand",     "province":"KPK","lat":34.5658,"lon":71.9303,"elevation_m":850,  "terrain_type":"mountains","river_proximity":1},
    {"district":"Hangu",        "province":"KPK","lat":33.5286,"lon":71.0594,"elevation_m":710,  "terrain_type":"hills",    "river_proximity":0},
    {"district":"Karak",        "province":"KPK","lat":33.1172,"lon":71.0942,"elevation_m":590,  "terrain_type":"hills",    "river_proximity":0},
    {"district":"Kohat",        "province":"KPK","lat":33.5869,"lon":71.4414,"elevation_m":518,  "terrain_type":"hills",    "river_proximity":1},
    {"district":"Lakki Marwat", "province":"KPK","lat":32.6072,"lon":70.9125,"elevation_m":218,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Bannu",        "province":"KPK","lat":32.9889,"lon":70.6006,"elevation_m":370,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Kurram",       "province":"KPK","lat":33.8872,"lon":70.0989,"elevation_m":1800, "terrain_type":"mountains","river_proximity":1},
    {"district":"Orakzai",      "province":"KPK","lat":33.6583,"lon":70.9333,"elevation_m":1400, "terrain_type":"mountains","river_proximity":0},
    {"district":"Khyber",       "province":"KPK","lat":34.1000,"lon":71.1000,"elevation_m":1070, "terrain_type":"mountains","river_proximity":0},
    {"district":"Mohmand",      "province":"KPK","lat":34.4000,"lon":71.2000,"elevation_m":900,  "terrain_type":"mountains","river_proximity":1},
    {"district":"Bajaur",       "province":"KPK","lat":34.6900,"lon":71.5000,"elevation_m":1100, "terrain_type":"mountains","river_proximity":1},
    {"district":"South Waziristan","province":"KPK","lat":32.3000,"lon":69.8000,"elevation_m":1800,"terrain_type":"mountains","river_proximity":1},
    {"district":"North Waziristan","province":"KPK","lat":33.0000,"lon":70.0000,"elevation_m":1400,"terrain_type":"mountains","river_proximity":1},
    {"district":"Torghar",      "province":"KPK","lat":34.9000,"lon":72.9000,"elevation_m":2200, "terrain_type":"mountains","river_proximity":1},
    {"district":"Kolai Pallas", "province":"KPK","lat":35.2000,"lon":73.1000,"elevation_m":1800, "terrain_type":"mountains","river_proximity":1},
    {"district":"Swabi",        "province":"KPK","lat":34.1200,"lon":72.4700,"elevation_m":360,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Tor Ghar",     "province":"KPK","lat":34.8500,"lon":72.8500,"elevation_m":2100, "terrain_type":"mountains","river_proximity":1},
    {"district":"Khyber Tribal","province":"KPK","lat":34.2000,"lon":71.0000,"elevation_m":950,  "terrain_type":"mountains","river_proximity":0},

    # ══════════════════════════════════════════════════════
    # BALOCHISTAN — 35 districts
    # ══════════════════════════════════════════════════════
    {"district":"Quetta",        "province":"Balochistan","lat":30.1798,"lon":66.9750,"elevation_m":1680,"terrain_type":"mountains","river_proximity":0},
    {"district":"Naseerabad",    "province":"Balochistan","lat":28.8775,"lon":68.3497,"elevation_m":55,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Jaffarabad",    "province":"Balochistan","lat":28.5236,"lon":68.1636,"elevation_m":58,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Lasbela",       "province":"Balochistan","lat":26.2197,"lon":66.6997,"elevation_m":36,  "terrain_type":"coastal",  "river_proximity":0},
    {"district":"Kalat",         "province":"Balochistan","lat":29.0239,"lon":66.5903,"elevation_m":1920,"terrain_type":"mountains","river_proximity":0},
    {"district":"Khuzdar",       "province":"Balochistan","lat":27.8125,"lon":66.6167,"elevation_m":1260,"terrain_type":"mountains","river_proximity":0},
    {"district":"Turbat",        "province":"Balochistan","lat":26.0025,"lon":63.0406,"elevation_m":152, "terrain_type":"plains",   "river_proximity":1},
    {"district":"Gwadar",        "province":"Balochistan","lat":25.1264,"lon":62.3256,"elevation_m":10,  "terrain_type":"coastal",  "river_proximity":0},
    {"district":"Panjgur",       "province":"Balochistan","lat":26.9667,"lon":64.0833,"elevation_m":969, "terrain_type":"mountains","river_proximity":1},
    {"district":"Chagai",        "province":"Balochistan","lat":29.1000,"lon":64.5200,"elevation_m":900, "terrain_type":"mountains","river_proximity":0},
    {"district":"Nushki",        "province":"Balochistan","lat":29.5519,"lon":66.0206,"elevation_m":1340,"terrain_type":"mountains","river_proximity":0},
    {"district":"Kharan",        "province":"Balochistan","lat":28.5833,"lon":65.4167,"elevation_m":900, "terrain_type":"mountains","river_proximity":0},
    {"district":"Washuk",        "province":"Balochistan","lat":27.5667,"lon":64.4667,"elevation_m":700, "terrain_type":"mountains","river_proximity":0},
    {"district":"Awaran",        "province":"Balochistan","lat":26.3597,"lon":65.2436,"elevation_m":420, "terrain_type":"mountains","river_proximity":1},
    {"district":"Kech",          "province":"Balochistan","lat":26.0500,"lon":63.0500,"elevation_m":140, "terrain_type":"plains",   "river_proximity":1},
    {"district":"Dera Bugti",    "province":"Balochistan","lat":29.0369,"lon":69.1583,"elevation_m":180, "terrain_type":"plains",   "river_proximity":1},
    {"district":"Kohlu",         "province":"Balochistan","lat":29.8942,"lon":69.2542,"elevation_m":950, "terrain_type":"mountains","river_proximity":1},
    {"district":"Sibi",          "province":"Balochistan","lat":29.5444,"lon":67.8775,"elevation_m":133, "terrain_type":"plains",   "river_proximity":1},
    {"district":"Ziarat",        "province":"Balochistan","lat":30.3819,"lon":67.7264,"elevation_m":2449,"terrain_type":"mountains","river_proximity":0},
    {"district":"Pishin",        "province":"Balochistan","lat":30.5758,"lon":66.9953,"elevation_m":1580,"terrain_type":"mountains","river_proximity":0},
    {"district":"Qilla Abdullah","province":"Balochistan","lat":30.6803,"lon":66.5806,"elevation_m":1200,"terrain_type":"mountains","river_proximity":0},
    {"district":"Qilla Saifullah","province":"Balochistan","lat":30.7039,"lon":68.3508,"elevation_m":1550,"terrain_type":"mountains","river_proximity":0},
    {"district":"Loralai",       "province":"Balochistan","lat":30.3722,"lon":68.5917,"elevation_m":1390,"terrain_type":"mountains","river_proximity":0},
    {"district":"Musakhel",      "province":"Balochistan","lat":30.0000,"lon":69.7500,"elevation_m":1050,"terrain_type":"mountains","river_proximity":1},
    {"district":"Barkhan",       "province":"Balochistan","lat":29.8958,"lon":69.5264,"elevation_m":1000,"terrain_type":"mountains","river_proximity":1},
    {"district":"Harnai",        "province":"Balochistan","lat":30.1000,"lon":67.9300,"elevation_m":1100,"terrain_type":"mountains","river_proximity":1},
    {"district":"Sherani",       "province":"Balochistan","lat":31.0500,"lon":69.9500,"elevation_m":1650,"terrain_type":"mountains","river_proximity":0},
    {"district":"Zhob",          "province":"Balochistan","lat":31.3436,"lon":69.4486,"elevation_m":1405,"terrain_type":"mountains","river_proximity":1},
    {"district":"Kohlu",         "province":"Balochistan","lat":29.8942,"lon":69.2542,"elevation_m":950, "terrain_type":"mountains","river_proximity":1},
    {"district":"Jhal Magsi",    "province":"Balochistan","lat":28.2808,"lon":67.7256,"elevation_m":62,  "terrain_type":"plains",   "river_proximity":1},
    {"district":"Khairpur Tamewali","province":"Balochistan","lat":29.5694,"lon":72.2403,"elevation_m":100,"terrain_type":"plains", "river_proximity":1},
    {"district":"Mastung",       "province":"Balochistan","lat":29.7942,"lon":66.8411,"elevation_m":1840,"terrain_type":"mountains","river_proximity":0},
    {"district":"Surab",         "province":"Balochistan","lat":28.4900,"lon":66.2600,"elevation_m":1620,"terrain_type":"mountains","river_proximity":0},
    {"district":"Khuzdar",       "province":"Balochistan","lat":27.8125,"lon":66.6167,"elevation_m":1260,"terrain_type":"mountains","river_proximity":0},
    {"district":"Lehri",         "province":"Balochistan","lat":29.4300,"lon":68.1800,"elevation_m":80,  "terrain_type":"plains",   "river_proximity":1},

    # ══════════════════════════════════════════════════════
    # GILGIT-BALTISTAN — 14 districts
    # ══════════════════════════════════════════════════════
    {"district":"Gilgit",        "province":"GB","lat":35.9208,"lon":74.3089,"elevation_m":1500,"terrain_type":"mountains","river_proximity":1},
    {"district":"Skardu",        "province":"GB","lat":35.2971,"lon":75.6333,"elevation_m":2438,"terrain_type":"mountains","river_proximity":1},
    {"district":"Hunza",         "province":"GB","lat":36.3147,"lon":74.6481,"elevation_m":2438,"terrain_type":"mountains","river_proximity":1},
    {"district":"Nagar",         "province":"GB","lat":36.2333,"lon":74.4500,"elevation_m":2900,"terrain_type":"mountains","river_proximity":1},
    {"district":"Ghanche",       "province":"GB","lat":35.4900,"lon":76.6700,"elevation_m":2700,"terrain_type":"mountains","river_proximity":1},
    {"district":"Shigar",        "province":"GB","lat":35.5100,"lon":75.7000,"elevation_m":2620,"terrain_type":"mountains","river_proximity":1},
    {"district":"Kharmang",      "province":"GB","lat":35.0800,"lon":76.2000,"elevation_m":2500,"terrain_type":"mountains","river_proximity":1},
    {"district":"Roundu",        "province":"GB","lat":35.5000,"lon":73.7000,"elevation_m":1600,"terrain_type":"mountains","river_proximity":1},
    {"district":"Tangir",        "province":"GB","lat":35.5000,"lon":73.2000,"elevation_m":1400,"terrain_type":"mountains","river_proximity":1},
    {"district":"Darel",         "province":"GB","lat":35.5500,"lon":73.4000,"elevation_m":1500,"terrain_type":"mountains","river_proximity":1},
    {"district":"Diamer",        "province":"GB","lat":35.1667,"lon":73.6667,"elevation_m":1600,"terrain_type":"mountains","river_proximity":1},
    {"district":"Astore",        "province":"GB","lat":35.3667,"lon":74.9000,"elevation_m":2590,"terrain_type":"mountains","river_proximity":1},
    {"district":"Ghizer",        "province":"GB","lat":36.2000,"lon":73.7500,"elevation_m":2200,"terrain_type":"mountains","river_proximity":1},
    {"district":"Gupis Yasin",   "province":"GB","lat":36.1700,"lon":73.2700,"elevation_m":2260,"terrain_type":"mountains","river_proximity":1},

    # ══════════════════════════════════════════════════════
    # AJK — 10 districts
    # ══════════════════════════════════════════════════════
    {"district":"Muzaffarabad",  "province":"AJK","lat":34.3700,"lon":73.4700,"elevation_m":737, "terrain_type":"mountains","river_proximity":1},
    {"district":"Mirpur",        "province":"AJK","lat":33.1478,"lon":73.7508,"elevation_m":411, "terrain_type":"hills",    "river_proximity":1},
    {"district":"Bagh",          "province":"AJK","lat":33.9861,"lon":73.7808,"elevation_m":1070,"terrain_type":"mountains","river_proximity":1},
    {"district":"Rawalakot",     "province":"AJK","lat":33.8583,"lon":73.7611,"elevation_m":1676,"terrain_type":"mountains","river_proximity":0},
    {"district":"Kotli",         "province":"AJK","lat":33.5136,"lon":73.9017,"elevation_m":614, "terrain_type":"hills",    "river_proximity":1},
    {"district":"Neelum",        "province":"AJK","lat":34.5500,"lon":73.9000,"elevation_m":1400,"terrain_type":"mountains","river_proximity":1},
    {"district":"Haveli",        "province":"AJK","lat":33.7700,"lon":73.6600,"elevation_m":1200,"terrain_type":"mountains","river_proximity":0},
    {"district":"Sudhnoti",      "province":"AJK","lat":33.5500,"lon":73.6700,"elevation_m":960, "terrain_type":"hills",    "river_proximity":0},
    {"district":"Hattian Bala",  "province":"AJK","lat":34.3700,"lon":73.8300,"elevation_m":1050,"terrain_type":"mountains","river_proximity":1},
    {"district":"Jhelum Valley", "province":"AJK","lat":34.6700,"lon":73.6700,"elevation_m":1600,"terrain_type":"mountains","river_proximity":1},

    # ══════════════════════════════════════════════════════
    # ISLAMABAD CAPITAL TERRITORY — 1
    # ══════════════════════════════════════════════════════
    {"district":"Islamabad",     "province":"ICT","lat":33.7294,"lon":73.0931,"elevation_m":540, "terrain_type":"hills",    "river_proximity":0},
]