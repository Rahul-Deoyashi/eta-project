"""
app.py
------
Streamlit demo app (Dark Mode Edition): High-contrast UI, custom dark-mode hovercards,
and side-by-side static vs AI ETA metric breakdowns.

Run with:
    streamlit run src/app.py
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import plotly.graph_objects as go

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

STOPS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "train_stops.csv")
SIM_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "simulated_runs.csv")
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "station_level_features.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "xgb_station_model.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "outputs", "station_model_metrics.json")

st.set_page_config(
    page_title="Dynamic ETA Forecast - Indian Railways",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Styling Overrides: Fixes Dropdown Contrast, Table Contrast, and Sidebar Toggle
# ---------------------------------------------------------------------------
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #f8fafc;
}

.stApp {
    background-color: #090d16;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Sidebar Toggle Restore */
button[kind="header"], 
button[data-testid="baseButton-header"],
[data-testid="stSidebarNav"] button {
    color: #f8fafc !important;
    visibility: visible !important;
}

/* Force Dropdown Input Boxes to High Contrast */
div[data-baseweb="select"] {
    background-color: #1e293b !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] * {
    color: #ffffff !important;
    fill: #ffffff !important;
    background-color: transparent !important;
}

/* Dropdown Menu Container */
div[data-baseweb="popover"] div {
    background-color: #0f172a !important;
}

/* Dropdown List Items */
ul[role="listbox"] {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
}

ul[role="listbox"] li {
    background-color: #0f172a !important;
    color: #f8fafc !important;
}

ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
}

/* Titles */
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    font-weight: 500;
}

/* Metric Cards */
.card-grid {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.metric-card {
    flex: 1;
    background: #131c2e;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
.metric-card.baseline { border-top: 4px solid #f59e0b; }
.metric-card.model { border-top: 4px solid #10b981; }
.metric-card.info { border-top: 4px solid #38bdf8; }

.metric-card .label {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 4px 0;
}
.metric-card .sub-badge {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 700;
    color: #34d399;
    background: rgba(16, 185, 129, 0.15);
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Inspector Highlight Box */
.inspector-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 20px;
}
.inspector-title {
    font-size: 0.85rem;
    text-transform: uppercase;
    color: #818cf8;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.explainer-box {
    background: #0f172a;
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.6;
    margin-top: 15px;
    border: 1px solid #1e293b;
}

/* Sidebar Dark Theme Overrides */
section[data-testid="stSidebar"] {
    background-color: #0d1322;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
</style>
"""

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_data():
    stops = pd.read_csv(STOPS_CSV, low_memory=False)
    sim = pd.read_csv(SIM_CSV, low_memory=False)
    feat_raw = pd.read_csv(FEATURES_CSV, low_memory=False)
    return stops, sim, feat_raw

@st.cache_data
def load_aggregate_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

def get_commercial_stops(stops_df, train_number):
    sub = stops_df[stops_df["train_number"].astype(str) == str(train_number)].sort_values("stop_seq")
    max_seq = sub["stop_seq"].max()
    return sub[(sub["halt_minutes"] > 0) | (sub["stop_seq"] == 0) | (sub["stop_seq"] == max_seq)].reset_index(drop=True)

def build_comparison(stops_df, sim_df, feat_raw, model, feat_cols, train_number, sim_date):
    commercial_stops = get_commercial_stops(stops_df, train_number)

    sim_journey = sim_df[
        (sim_df["train_number"].astype(str) == str(train_number)) & (sim_df["sim_date"] == sim_date)
    ].sort_values("stop_seq").reset_index(drop=True)

    if sim_journey.empty:
        return None, commercial_stops

    encoded = pd.get_dummies(feat_raw, columns=["train_type", "train_zone", "next_station_zone"], dummy_na=False)
    for c in feat_cols:
        if c not in encoded.columns:
            encoded[c] = 0

    mask = (feat_raw["train_number"].astype(str) == str(train_number)) & (feat_raw["sim_date"] == sim_date)
    journey_features_raw = feat_raw[mask].sort_values("cur_stop_seq").reset_index(drop=True)
    journey_features_enc = encoded[mask].sort_values("cur_stop_seq").reset_index(drop=True)

    results = []
    origin = commercial_stops.iloc[0]
    sim_origin = sim_journey[sim_journey["stop_seq"] == origin["stop_seq"]].iloc[0]
    results.append({
        "Station": origin["station_name"],
        "Scheduled (min)": round(sim_origin["scheduled_arrival_min"], 1),
        "Current Method (min)": round(sim_origin["scheduled_arrival_min"], 1),
        "AI Model (min)": round(sim_origin["scheduled_arrival_min"], 1),
        "What Actually Happened (min)": round(sim_origin["actual_arrival_min"], 1),
        "Static Error (min)": 0.0,
        "AI Error (min)": 0.0,
    })

    for idx in range(1, len(commercial_stops)):
        stop = commercial_stops.iloc[idx]
        seq = stop["stop_seq"]
        sim_row = sim_journey[sim_journey["stop_seq"] == seq]
        if sim_row.empty:
            continue
        sim_row = sim_row.iloc[0]

        row_mask = journey_features_raw["next_station_name"] == stop["station_name"]
        if not row_mask.any():
            continue
        raw_row = journey_features_raw[row_mask].iloc[0]
        enc_row = journey_features_enc[row_mask].iloc[0]

        scheduled_min = raw_row["next_scheduled_min"]
        actual_min = sim_row["actual_arrival_min"]
        naive_delay = raw_row["cur_delay_minutes"]

        X = enc_row[feat_cols].to_frame().T.astype(float)
        ml_delay = max(0.0, float(model.predict(X)[0]))

        naive_eta = scheduled_min + naive_delay
        ml_eta = scheduled_min + ml_delay

        results.append({
            "Station": stop["station_name"],
            "Scheduled (min)": round(scheduled_min, 1),
            "Current Method (min)": round(naive_eta, 1),
            "AI Model (min)": round(ml_eta, 1),
            "What Actually Happened (min)": round(actual_min, 1),
            "Static Error (min)": round(abs(naive_eta - actual_min), 1),
            "AI Error (min)": round(abs(ml_eta - actual_min), 1),
        })

    return pd.DataFrame(results), commercial_stops

def render_sidebar(sim_df):
    st.sidebar.markdown("### 🎛️ Dynamic Controls")
    
    sampled_trains = sorted(sim_df["train_number"].astype(str).unique())
    train_names = (
        sim_df.drop_duplicates("train_number")
        .assign(train_number_str=lambda d: d["train_number"].astype(str))
        .set_index("train_number_str")["train_name"]
        .to_dict()
    )
    options = [f"{t} - {train_names.get(t, '')}" for t in sampled_trains]
    default_idx = next((i for i, o in enumerate(options) if o.startswith("12502")), 0)

    choice = st.sidebar.selectbox("Select Train", options, index=default_idx)
    train_number = choice.split(" - ")[0]

    test_dates = sorted(sim_df[sim_df["sim_date"] >= "2026-01-24"]["sim_date"].unique())
    default_date_idx = min(2, len(test_dates) - 1)
    sim_date = st.sidebar.selectbox("Select Date", test_dates, index=default_date_idx)

    st.sidebar.markdown("---")
    st.sidebar.caption("🚀 Select any train & date above to test real-time inference vs NTES baselines.")

    return train_number, sim_date

def render_dark_plotly_chart(result_df):
    fig = go.Figure()

    # Unified Hover Card Template
    hover_template = (
        "<b>Station: %{x}</b><br><br>"
        "📅 Static Scheduled: <b>%{customdata[0]:.1f} min</b><br>"
        "⚠️ Static NTES ETA: <b>%{customdata[1]:.1f} min</b><br>"
        "⚡ AI Dynamic Model: <b>%{customdata[2]:.1f} min</b><br>"
        "🎯 Actual Arrival: <b>%{customdata[3]:.1f} min</b><br><br>"
        "❌ Static NTES Error: <span style='color:#f59e0b;'><b>%{customdata[4]:.1f} min</b></span><br>"
        "✅ AI Model Error: <span style='color:#10b981;'><b>%{customdata[5]:.1f} min</b></span>"
        "<extra></extra>"
    )

    customdata = result_df[[
        "Scheduled (min)", 
        "Current Method (min)", 
        "AI Model (min)", 
        "What Actually Happened (min)", 
        "Static Error (min)", 
        "AI Error (min)"
    ]].values

    # Scheduled
    fig.add_trace(go.Scatter(
        x=result_df["Station"], y=result_df["Scheduled (min)"],
        mode='lines+markers', name='Scheduled Timetable',
        line=dict(color='#64748b', width=2, dash='dash'),
        customdata=customdata,
        hovertemplate=hover_template
    ))

    # Static (NTES)
    fig.add_trace(go.Scatter(
        x=result_df["Station"], y=result_df["Current Method (min)"],
        mode='lines+markers', name='Current Method (NTES)',
        line=dict(color='#f59e0b', width=3),
        customdata=customdata,
        hovertemplate=hover_template
    ))

    # AI Model
    fig.add_trace(go.Scatter(
        x=result_df["Station"], y=result_df["AI Model (min)"],
        mode='lines+markers', name='AI Dynamic Model',
        line=dict(color='#10b981', width=3.5),
        customdata=customdata,
        hovertemplate=hover_template
    ))

    # Actual Arrival
    fig.add_trace(go.Scatter(
        x=result_df["Station"], y=result_df["What Actually Happened (min)"],
        mode='lines+markers', name='Actual Arrival',
        line=dict(color='#ef4444', width=3.5),
        customdata=customdata,
        hovertemplate=hover_template
    ))

    # Peak Discrepancy Callout Flag
    worst_idx = result_df["Static Error (min)"].idxmax()
    worst_val = result_df.loc[worst_idx, "Static Error (min)"]
    
    if worst_val > 10:
        worst_st_name = result_df.loc[worst_idx, "Station"]
        worst_y = max(
            result_df.loc[worst_idx, "Current Method (min)"],
            result_df.loc[worst_idx, "What Actually Happened (min)"]
        )
        
        fig.add_annotation(
            x=worst_st_name,
            y=worst_y,
            text=f"⚠️ Static Method off by {worst_val:.0f} mins at {worst_st_name}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#f59e0b",
            ax=0,
            ay=-50,
            bordercolor="#f59e0b",
            borderwidth=1.5,
            borderpad=6,
            bgcolor="#1e1b4b",
            font=dict(size=12, color="#fef08a", family="Plus Jakarta Sans")
        )

    fig.update_layout(
        xaxis_title="Stations Along Route",
        yaxis_title="Minutes since departure",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_size=13,
            font_color="#f8fafc",
            font_family="Plus Jakarta Sans",
            bordercolor="#334155"
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0")),
        plot_bgcolor="#090d16",
        paper_bgcolor="#090d16",
        font=dict(color="#cbd5e1"),
        height=500,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1e293b')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1e293b')

    st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    st.markdown('<div class="hero-title">🚆 Dynamic ETA Forecast for Indian Railways</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">SIH #26028 — Real-time AI delay forecasting vs NTES static baselines</div>', unsafe_allow_html=True)

    stops_df, sim_df, feat_raw = load_data()
    model = load_model()
    feat_cols = model.get_booster().feature_names
    agg_metrics = load_aggregate_metrics()

    train_number, sim_date = render_sidebar(sim_df)

    if agg_metrics:
        imp = agg_metrics.get("mae_improvement_pct", 0)
        naive_m = agg_metrics.get("naive_baseline", {}).get("mae", 0)
        xgb_m = agg_metrics.get("xgboost_model", {}).get("mae", 0)
        
        st.markdown(f"""
        <div class="card-grid">
            <div class="metric-card baseline">
                <div class="label">Dataset Static Baseline Error</div>
                <div class="value">{naive_m:.1f} <span style="font-size:1.1rem;font-weight:500;">min</span></div>
                <div style="font-size:0.8rem;color:#94a3b8;">NTES Static Logic</div>
            </div>
            <div class="metric-card model">
                <div class="label">AI Model Delay Error</div>
                <div class="value">{xgb_m:.1f} <span style="font-size:1.1rem;font-weight:500;">min</span></div>
                <div class="sub-badge">⚡ {imp:.1f}% Better Accuracy</div>
            </div>
            <div class="metric-card info">
                <div class="label">Live Signals Processed</div>
                <div class="value" style="font-size:1.3rem;margin-top:10px;color:#38bdf8;">TSR, Weather, Congestion</div>
                <div style="font-size:0.8rem;color:#94a3b8;">Dynamic Route Predictions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    result_df, commercial_stops = build_comparison(
        stops_df, sim_df, feat_raw, model, feat_cols, train_number, sim_date
    )

    if result_df is None or len(result_df) < 2:
        st.warning("No simulated data available for this train/date combination. Try another choice.")
        return

    # Peak Discrepancy Highlight Box
    worst_idx = result_df["Static Error (min)"].idxmax()
    st_data = result_df.loc[worst_idx]

    st.markdown(f"""
    <div class="inspector-card">
        <div class="inspector-title">🚨 Peak Discrepancy Highlight: {st_data['Station']}</div>
        <div style="display:flex; gap:30px; margin-top:10px; align-items:center;">
            <div>
                <span style="color:#94a3b8; font-size:0.85rem;">NTES Static Error:</span>
                <div style="font-size:1.4rem; font-weight:700; color:#f59e0b;">{st_data['Static Error (min)']} min</div>
            </div>
            <div>
                <span style="color:#94a3b8; font-size:0.85rem;">AI Model Error:</span>
                <div style="font-size:1.4rem; font-weight:700; color:#34d399;">{st_data['AI Error (min)']} min</div>
            </div>
            <div>
                <span style="color:#94a3b8; font-size:0.85rem;">Actual Delay from Departure:</span>
                <div style="font-size:1.4rem; font-weight:700; color:#f87171;">{round(st_data['What Actually Happened (min)'] - st_data['Scheduled (min)'], 1)} min</div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <span style="color:#818cf8; font-size:0.85rem; font-weight:700;">AI Precision Boost:</span>
                <div style="font-size:1.4rem; font-weight:800; color:#38bdf8;">{round(st_data['Static Error (min)'] - st_data['AI Error (min)'], 1)} min closer</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plotly Chart with Unified Hover Box
    render_dark_plotly_chart(result_df)

    st.markdown("##### 📋 Detailed Station Breakdown")
    
    # HTML Table Representation
    st.markdown(
        result_df.to_html(index=False, classes="custom-table")
        .replace('<table border="1" class="dataframe custom-table">', '<table style="width:100%; border-collapse:collapse; background-color:#131c2e; color:#f8fafc; border-radius:10px; overflow:hidden;">')
        .replace('<th>', '<th style="padding:12px; background-color:#1e293b; color:#94a3b8; text-align:left; border-bottom:1px solid #334155;">')
        .replace('<td>', '<td style="padding:12px; border-bottom:1px solid #1e293b; color:#f8fafc;">'),
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="explainer-box">
        <b>💡 Why AI out-predicts NTES static baselines:</b><br>
        NTES static logic assumes a delayed train recovers schedule according to a fixed speed timetable. Our XGBoost engine continuously evaluates upcoming temporary speed restrictions (TSR), severe weather alerts, bottleneck section occupancy, and train priority rules to forecast real-world delays.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()