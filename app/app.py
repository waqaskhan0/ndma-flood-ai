"""
NDMA Flood Risk Early Warning Dashboard
========================================
Run from inside C:\\ndma-flood-ai\\:

    streamlit run app/app.py

Then open browser at:  http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NDMA Flood Risk Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
      font-size: 2rem; font-weight: 700; color: #0C447C;
      border-bottom: 3px solid #0C447C; padding-bottom: 8px; margin-bottom: 0;
  }
  .sub-title { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
  .metric-card {
      background: #f8f9fa; border-radius: 10px; padding: 16px 20px;
      border-left: 5px solid #0C447C; margin-bottom: 10px;
  }
  .alert-high {
      background: #FFEBEE; border-left: 5px solid #D32F2F;
      padding: 14px 18px; border-radius: 8px; margin-bottom: 8px;
  }
  .alert-med {
      background: #FFF3E0; border-left: 5px solid #F57C00;
      padding: 14px 18px; border-radius: 8px; margin-bottom: 8px;
  }
  .alert-low {
      background: #E8F5E9; border-left: 5px solid #388E3C;
      padding: 14px 18px; border-radius: 8px; margin-bottom: 8px;
  }
  .stMetric { background: #f8f9fa; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "data", "processed", "flood_features.csv"))
    return df

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE, "models", "flood_model_best.pkl")
    scaler_path = os.path.join(BASE, "models", "scaler.pkl")
    info_path   = os.path.join(BASE, "models", "model_info.pkl")
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    info   = joblib.load(info_path)
    return model, scaler, info

@st.cache_data
def load_operational_alerts():
    alerts_path = os.path.join(BASE, "data", "processed", "active_alerts.csv")
    if os.path.exists(alerts_path):
        return pd.read_csv(alerts_path)
    return pd.DataFrame()

df            = load_data()
model, scaler, info = load_model()
operational_alerts = load_operational_alerts()

FEATURES = info["features"]
FEATURE_METADATA = info.get("feature_metadata", {})
BASE_FEATURES = info.get("base_features", [
    "annual_total_mm", "annual_max_daily", "monsoon_total_mm",
    "monsoon_avg_daily", "monsoon_max_daily", "monsoon_rainy_days",
    "premonsoon_total_mm", "elevation_m", "river_proximity",
    "terrain_code", "geo_risk_code", "province_code",
    "flood_intensity_score", "lat", "lon", "year",
])
DISTRICT_CONTEXT_COLS = [
    "annual_total_mm", "monsoon_total_mm", "monsoon_max_daily",
    "flood_intensity_score", "monsoon_rainy_days",
]
YEAR_CONTEXT_COLS = [
    "annual_total_mm", "monsoon_total_mm",
    "monsoon_max_daily", "flood_intensity_score",
]

# ─────────────────────────────────────────────────────────────
# COMPUTE RISK SCORES for all districts / years
# ─────────────────────────────────────────────────────────────
def _clean_series(series, default=0.0):
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )

def _safe_div(numerator, denominator, default=0.0):
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)

def _map_from_series(values, mapping, default):
    return values.astype(str).map(mapping).fillna(default).astype(float)

def build_model_features(data):
    X = pd.DataFrame(index=data.index)
    for col in BASE_FEATURES:
        X[col] = _clean_series(data[col]) if col in data.columns else 0.0

    metadata = FEATURE_METADATA
    if metadata:
        X["rain_x_river"] = X["monsoon_total_mm"] * (X["river_proximity"] + 1)
        X["rain_x_geo_risk"] = X["monsoon_total_mm"] * X["geo_risk_code"]
        X["intensity_x_river"] = X["flood_intensity_score"] * (X["river_proximity"] + 1)
        X["elev_inverse"] = _safe_div(pd.Series(1.0, index=X.index), X["elevation_m"] + 1)
        X["monsoon_intensity"] = _safe_div(X["monsoon_total_mm"], X["monsoon_rainy_days"] + 1)
        X["rainfall_concentration"] = _safe_div(X["monsoon_max_daily"], X["monsoon_avg_daily"] + 0.1)
        X["premonsoon_ratio"] = _safe_div(X["premonsoon_total_mm"], X["annual_total_mm"] + 1)
        X["low_elevation_high_rain"] = (X["elevation_m"] < 100).astype(int) * X["monsoon_total_mm"]
        X["river_proximity_squared"] = X["river_proximity"] ** 2
        X["terrain_elevation_risk"] = X["terrain_code"] * X["elev_inverse"]
        X["saturation_index"] = _safe_div(
            X["premonsoon_total_mm"] * X["monsoon_total_mm"],
            X["elevation_m"] + 1,
        )
        X["annual_to_monsoon_ratio"] = _safe_div(X["monsoon_total_mm"], X["annual_total_mm"] + 1)
        X["extreme_daily_share"] = _safe_div(X["monsoon_max_daily"], X["monsoon_total_mm"] + 1)

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
        X["district_flood_prior"] = _map_from_series(
            districts, metadata.get("district_flood_prior", {}), global_rate
        )
        X["province_flood_prior"] = _map_from_series(
            provinces, metadata.get("province_flood_prior", {}), global_rate
        )

        for col, stats in metadata.get("district_climatology", {}).items():
            district_mean = _map_from_series(districts, stats.get("mean", {}), stats.get("global_mean", 0.0))
            district_std = _map_from_series(districts, stats.get("std", {}), stats.get("global_std", 1.0))
            district_std = district_std.replace(0, stats.get("global_std", 1.0) or 1.0)
            X[f"{col}_district_z"] = _safe_div(X[col] - district_mean, district_std)

        grouped = data.groupby("year") if "year" in data.columns else None
        for col in YEAR_CONTEXT_COLS:
            if grouped is not None and col in data.columns:
                year_mean = grouped[col].transform("mean")
                year_max = grouped[col].transform("max")
            else:
                year_mean = pd.Series(X[col].mean(), index=data.index)
                year_max = pd.Series(X[col].max(), index=data.index)

            baseline = metadata.get("year_baseline", {}).get(col, {"mean": 0.0, "std": 1.0})
            X[f"{col}_year_mean"] = _clean_series(year_mean)
            X[f"{col}_year_max"] = _clean_series(year_max)
            X[f"{col}_year_pressure"] = _safe_div(
                _clean_series(year_mean) - baseline.get("mean", 0.0),
                pd.Series(baseline.get("std", 1.0) or 1.0, index=data.index),
            )
            X[f"{col}_district_vs_year"] = _safe_div(X[col], _clean_series(year_mean) + 1)
    else:
        X["rain_x_river"] = X["monsoon_total_mm"] * X["river_proximity"]
        X["rain_x_geo_risk"] = X["monsoon_total_mm"] * X["geo_risk_code"]
        X["intensity_x_river"] = X["flood_intensity_score"] * X["river_proximity"]
        X["elev_inverse"] = _safe_div(pd.Series(1.0, index=X.index), X["elevation_m"] + 1)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in FEATURES:
        if col not in X.columns:
            X[col] = 0.0
    return X[FEATURES]

def get_risk_scores(data):
    X = build_model_features(data)
    X_sc = scaler.transform(X)
    proba = model.predict_proba(X_sc)[:, 1]
    return proba

df["risk_score"] = get_risk_scores(df)
df["risk_level"] = pd.cut(df["risk_score"],
                           bins=[0, 0.33, 0.66, 1.0],
                           labels=["Low", "Medium", "High"])

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/32/Flag_of_Pakistan.svg",
             width=80)
    st.markdown("## 🌊 NDMA Flood AI")
    st.markdown("*Early Warning Decision Support System*")
    st.divider()

    # Year filter
    years = sorted(df["year"].unique())
    selected_year = st.selectbox("📅 Select Year", years, index=len(years)-1)

    # Province filter
    provinces = ["All Provinces"] + sorted(df["province"].unique().tolist())
    selected_province = st.selectbox("🗺 Filter by Province", provinces)

    # Risk threshold
    default_alert_threshold = float(np.clip(info.get("optimal_threshold", 0.6), 0.05, 0.95))
    risk_threshold = st.slider(
        "⚠️ Alert Threshold (Risk Score)",
        min_value=0.05, max_value=0.95, value=round(default_alert_threshold, 2), step=0.01,
        help="Districts above this score trigger an early warning. Default comes from the trained model."
    )

    st.divider()
    st.markdown(f"**Model:** {info['name']}")
    st.markdown(f"**F1 Score:** {info['f1']:.3f}")
    st.markdown(f"**AUC-ROC:** {info['auc']:.3f}")
    st.markdown(f"**Data:** 2010–2022 | 36 Districts")

# ─────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────
df_year = df[df["year"] == selected_year].copy()
if selected_province != "All Provinces":
    df_year = df_year[df_year["province"] == selected_province]

df_all_years = df.copy()
if selected_province != "All Provinces":
    df_all_years = df_all_years[df_all_years["province"] == selected_province]

# High risk districts
high_risk = df_year[df_year["risk_score"] >= risk_threshold].sort_values("risk_score", ascending=False)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌊 NDMA Flood Risk Early Warning Dashboard</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">AI-powered flood risk prediction for Pakistan districts | Showing: <b>{selected_year}</b> | Province: <b>{selected_province}</b></p>', unsafe_allow_html=True)

if not operational_alerts.empty:
    latest_as_of = operational_alerts.get("as_of_date", pd.Series(["latest"])).iloc[0]
    top_live = operational_alerts.sort_values("risk_score", ascending=False).head(10)
    st.markdown(f"""
    <div class="alert-high">
        <b>Operational Daily Alerts</b> — {len(operational_alerts)} active district alert(s), as of {latest_as_of}<br>
        <b>Top Alerts:</b> {", ".join(top_live["district"].astype(str).tolist())}
    </div>
    """, unsafe_allow_html=True)
    live_cols = [
        "district", "province", "risk_score", "alert_level",
        "monsoon_total_mm", "monsoon_max_daily", "recommendation",
    ]
    live_cols = [col for col in live_cols if col in operational_alerts.columns]
    with st.expander("View operational alert table", expanded=False):
        st.dataframe(
            operational_alerts[live_cols].sort_values("risk_score", ascending=False),
            use_container_width=True,
            height=280,
        )

# ─────────────────────────────────────────────────────────────
# EARLY WARNING ALERT BANNER
# ─────────────────────────────────────────────────────────────
if len(high_risk) > 0:
    st.markdown(f"""
    <div class="alert-high">
        🚨 <b>EARLY WARNING ACTIVE</b> — {len(high_risk)} district(s) above risk threshold ({risk_threshold:.0%})<br>
        <b>High Risk Districts:</b> {", ".join(high_risk["district"].tolist())}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-low">
        ✅ <b>No Active Warnings</b> — All districts below risk threshold ({risk_threshold:.0%}) for {selected_year}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Districts Monitored",  len(df_year))
with c2:
    st.metric("🔴 High Risk",
              len(df_year[df_year["risk_level"] == "High"]),
              delta=None)
with c3:
    st.metric("🟠 Medium Risk",
              len(df_year[df_year["risk_level"] == "Medium"]))
with c4:
    st.metric("🟢 Low Risk",
              len(df_year[df_year["risk_level"] == "Low"]))
with c5:
    avg_risk = df_year["risk_score"].mean()
    st.metric("Avg Risk Score", f"{avg_risk:.2f}",
              delta=f"{'⚠ High' if avg_risk > 0.5 else 'Normal'}")

st.divider()

# ─────────────────────────────────────────────────────────────
# ROW 1: MAP + RISK TABLE
# ─────────────────────────────────────────────────────────────
col_map, col_table = st.columns([3, 2])

with col_map:
    st.markdown("#### 🗺 Interactive Risk Map")

    # Build Folium map
    m = folium.Map(location=[30.5, 69.5], zoom_start=6,
                   tiles="CartoDB positron")

    def risk_color(score):
        if score >= 0.66:  return "#D32F2F"
        elif score >= 0.33: return "#F57C00"
        else:               return "#388E3C"

    def risk_label(score):
        if score >= 0.66:  return "HIGH"
        elif score >= 0.33: return "MEDIUM"
        else:               return "LOW"

    for _, row in df_year.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue
        color = risk_color(row["risk_score"])
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10 + row["risk_score"] * 12,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(f"""
                <b>{row['district']}</b><br>
                Province: {row['province']}<br>
                Risk Score: {row['risk_score']:.2f}<br>
                Risk Level: <b>{risk_label(row['risk_score'])}</b><br>
                Elevation: {row['elevation_m']}m<br>
                Monsoon Rain: {row['monsoon_total_mm']:.0f}mm<br>
                Actual Flood: {'Yes ✓' if row['flooded'] == 1 else 'No'}
            """, max_width=200),
            tooltip=f"{row['district']}: {risk_label(row['risk_score'])} ({row['risk_score']:.2f})"
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px;border-radius:8px;
                border:1px solid #ccc;font-size:13px'>
        <b>Risk Level</b><br>
        🔴 High (&gt;0.66)<br>
        🟠 Medium (0.33–0.66)<br>
        🟢 Low (&lt;0.33)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, height=420, use_container_width=True)

with col_table:
    st.markdown("#### 📋 District Risk Rankings")
    display_df = df_year[["district","province","risk_score","risk_level",
                           "elevation_m","river_proximity","flooded"]]\
                 .sort_values("risk_score", ascending=False).reset_index(drop=True)

    def color_risk(val):
        if val == "High":   return "background-color: #FFEBEE; color: #D32F2F; font-weight: bold"
        elif val == "Medium": return "background-color: #FFF3E0; color: #E65100"
        else:               return "background-color: #E8F5E9; color: #2E7D32"

    styled = display_df.rename(columns={
        "district":"District","province":"Province",
        "risk_score":"Score","risk_level":"Level",
        "elevation_m":"Elev(m)","river_proximity":"River",
        "flooded":"Flooded"
    }).style\
      .map(color_risk, subset=["Level"])\
      .format({"Score": "{:.2f}"})

    st.dataframe(styled, height=400, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# ROW 2: TIME SERIES + BAR CHART
# ─────────────────────────────────────────────────────────────
col_ts, col_bar = st.columns([3, 2])

with col_ts:
    st.markdown("#### 📈 Risk Score Over Time by District")
    top_districts = df_year.sort_values("risk_score", ascending=False)["district"].head(8).tolist()
    selected_districts = st.multiselect(
        "Select districts to compare:",
        options=sorted(df["district"].unique()),
        default=top_districts[:4]
    )
    if selected_districts:
        ts_data = df[df["district"].isin(selected_districts)].copy()
        fig_ts = px.line(
            ts_data, x="year", y="risk_score", color="district",
            markers=True,
            labels={"risk_score": "Flood Risk Score", "year": "Year", "district": "District"},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_ts.add_hline(y=risk_threshold, line_dash="dash",
                         line_color="red", annotation_text="Alert Threshold")
        fig_ts.add_hline(y=0.66, line_dash="dot",
                         line_color="orange", annotation_text="High Risk")
        fig_ts.update_layout(height=320, margin=dict(t=20,b=20),
                             legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_ts, use_container_width=True)

with col_bar:
    st.markdown("#### 🏙 Risk by Province")
    prov_summary = df_year.groupby("province").agg(
        avg_risk=("risk_score","mean"),
        high_districts=("risk_level", lambda x: (x=="High").sum()),
        total=("district","count")
    ).reset_index().sort_values("avg_risk", ascending=False)

    fig_bar = px.bar(
        prov_summary, x="avg_risk", y="province", orientation="h",
        color="avg_risk",
        color_continuous_scale=["#388E3C","#F57C00","#D32F2F"],
        range_color=[0,1],
        labels={"avg_risk":"Avg Risk Score","province":"Province"},
        text="avg_risk"
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(height=320, margin=dict(t=20,b=20),
                           coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# ROW 3: RAINFALL TREND + FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
col_rain, col_feat = st.columns(2)

with col_rain:
    st.markdown("#### 🌧 Rainfall Trend (Monsoon Season)")
    district_sel = st.selectbox("Select District", sorted(df["district"].unique()), key="rain_district")
    rain_data = df[df["district"] == district_sel][["year","monsoon_total_mm","flooded"]].copy()

    fig_rain = go.Figure()
    fig_rain.add_trace(go.Bar(
        x=rain_data["year"], y=rain_data["monsoon_total_mm"],
        name="Monsoon Rainfall (mm)",
        marker_color=["#D32F2F" if f==1 else "#1565C0" for f in rain_data["flooded"]],
        opacity=0.8
    ))
    fig_rain.update_layout(
        height=300, margin=dict(t=20,b=20),
        yaxis_title="Rainfall (mm)", xaxis_title="Year",
        legend=dict(orientation="h"),
        annotations=[dict(x=0.01,y=1.05,xref="paper",yref="paper",
                         text="🔴 Red = Flood Year | 🔵 Blue = No Flood",
                         showarrow=False, font=dict(size=11))]
    )
    st.plotly_chart(fig_rain, use_container_width=True)

with col_feat:
    st.markdown("#### 🔍 What Drives Flood Risk?")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names  = FEATURES
    else:
        try:
            for _, est in model.estimators:
                if hasattr(est, "feature_importances_"):
                    importances = est.feature_importances_
                    feat_names  = FEATURES
                    break
        except:
            importances = np.ones(len(FEATURES)) / len(FEATURES)
            feat_names  = FEATURES

    feat_df = pd.DataFrame({"Feature": feat_names,
                             "Importance": importances})\
              .sort_values("Importance", ascending=True).tail(10)

    fig_feat = px.bar(
        feat_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=["#1565C0","#F57C00","#D32F2F"],
        labels={"Importance":"Importance Score"}
    )
    fig_feat.update_layout(height=300, margin=dict(t=20,b=20),
                            coloraxis_showscale=False)
    st.plotly_chart(fig_feat, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# ROW 4: DISTRICT DEEP DIVE
# ─────────────────────────────────────────────────────────────
st.markdown("#### 🔬 District Deep Dive")
col_d1, col_d2 = st.columns([1, 2])

with col_d1:
    district_deep = st.selectbox("Select District for Analysis",
                                  sorted(df["district"].unique()), key="deep_district")
    d_data = df[df["district"] == district_deep].copy()
    latest = d_data[d_data["year"] == selected_year].iloc[0]

    risk_score = latest["risk_score"]
    if risk_score >= 0.66:
        level_html = f'<div class="alert-high">🔴 <b>HIGH RISK</b> — Score: {risk_score:.2f}</div>'
    elif risk_score >= 0.33:
        level_html = f'<div class="alert-med">🟠 <b>MEDIUM RISK</b> — Score: {risk_score:.2f}</div>'
    else:
        level_html = f'<div class="alert-low">🟢 <b>LOW RISK</b> — Score: {risk_score:.2f}</div>'

    st.markdown(level_html, unsafe_allow_html=True)
    st.markdown(f"**Province:** {latest['province']}")
    st.markdown(f"**Elevation:** {latest['elevation_m']}m")
    st.markdown(f"**River Nearby:** {'Yes' if latest['river_proximity']==1 else 'No'}")
    st.markdown(f"**Monsoon Rain:** {latest['monsoon_total_mm']:.0f}mm")
    st.markdown(f"**Actual Flood {selected_year}:** {'✅ Yes' if latest['flooded']==1 else '❌ No'}")

with col_d2:
    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        title={"text": f"Flood Risk Score — {district_deep} ({selected_year})"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#1565C0"},
            "steps": [
                {"range": [0,  33], "color": "#C8E6C9"},
                {"range": [33, 66], "color": "#FFE0B2"},
                {"range": [66,100], "color": "#FFCDD2"},
            ],
            "threshold": {
                "line":  {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": risk_threshold * 100
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40,b=20,l=30,r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#888;font-size:13px'>
    NDMA Flood Risk Early Warning Dashboard &nbsp;|&nbsp;
    Built with Python · Scikit-learn · XGBoost · Streamlit &nbsp;|&nbsp;
    Data: NASA POWER · NDMA Annual Reports · SRTM DEM &nbsp;|&nbsp;
    For NDMA Assistant Manager (AI) Interview Project
</div>
""", unsafe_allow_html=True)
