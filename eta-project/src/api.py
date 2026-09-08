import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STOPS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "train_stops.csv")
SIM_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "simulated_runs.csv")
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "station_level_features.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "xgb_station_model.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "outputs", "station_model_metrics.json")

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
feat_cols = model.get_booster().feature_names

stops_df = pd.read_csv(STOPS_CSV, low_memory=False)
sim_df = pd.read_csv(SIM_CSV, low_memory=False)
feat_raw = pd.read_csv(FEATURES_CSV, low_memory=False)

def generate_heuristic_explanation(ml_delay, naive_delay, raw_row):
    diff = ml_delay - naive_delay
    if diff <= 1: return "Normal historical running buffers applied."
   
    reasons = []
    tsr = raw_row.get("upcoming_tsr_frac", 0)
    cong = raw_row.get("upcoming_congestion_mean", 0)
    fog = raw_row.get("upcoming_foggy_frac", 0)
   
    total_weight = tsr + cong + fog + 0.001
   
    if tsr > 0.05: reasons.append(f"+{round(diff * (tsr/total_weight), 1)} min (Track Restrictions)")
    if cong > 0.2: reasons.append(f"+{round(diff * (cong/total_weight), 1)} min (Route Congestion)")
    if fog > 0.1:  reasons.append(f"+{round(diff * (fog/total_weight), 1)} min (Weather/Fog)")
       
    if not reasons: reasons.append(f"+{round(diff, 1)} min (Historical Delay)")
    return " | ".join(reasons)

@app.get("/api/controls")
def get_controls():
    sampled_trains = sorted(sim_df["train_number"].astype(str).unique())
    train_names = sim_df.drop_duplicates("train_number").assign(train_number_str=lambda d: d["train_number"].astype(str)).set_index("train_number_str")["train_name"].to_dict()
    trains = [{"id": t, "label": f"{t} - {train_names.get(t, '')}"} for t in sampled_trains]
    dates = sorted(sim_df[sim_df["sim_date"] >= "2026-01-24"]["sim_date"].unique())
    return {"trains": trains, "dates": dates}

@app.get("/api/journey")
def get_journey(train_number: str, sim_date: str):
    sub = stops_df[stops_df["train_number"].astype(str) == train_number].sort_values("stop_seq")
    max_seq = sub["stop_seq"].max()
    commercial_stops = sub[(sub["halt_minutes"] > 0) | (sub["stop_seq"] == 0) | (sub["stop_seq"] == max_seq)].reset_index(drop=True)

    sim_journey = sim_df[(sim_df["train_number"].astype(str) == train_number) & (sim_df["sim_date"] == sim_date)].sort_values("stop_seq").reset_index(drop=True)
    if sim_journey.empty: return {"error": "No data"}

    encoded = pd.get_dummies(feat_raw, columns=["train_type", "train_zone", "next_station_zone"], dummy_na=False)
    for c in feat_cols:
        if c not in encoded.columns: encoded[c] = 0

    mask = (feat_raw["train_number"].astype(str) == train_number) & (feat_raw["sim_date"] == sim_date)
    journey_features_raw = feat_raw[mask].sort_values("cur_stop_seq").reset_index(drop=True)
    journey_features_enc = encoded[mask].sort_values("cur_stop_seq").reset_index(drop=True)

    results = []
    origin = commercial_stops.iloc[0]
    sim_origin = sim_journey[sim_journey["stop_seq"] == origin["stop_seq"]].iloc[0]
   
    results.append({
        "Station": origin["station_name"], "Lat": float(origin["lat"]), "Lng": float(origin["lon"]),
        "Scheduled": round(sim_origin["scheduled_arrival_min"], 1),
        "Static_ETA": round(sim_origin["scheduled_arrival_min"], 1),
        "AI_ETA": round(sim_origin["scheduled_arrival_min"], 1),
        "Actual": round(sim_origin["actual_arrival_min"], 1),
        "Static_Error": 0.0, "AI_Error": 0.0, "Explanation": "Origin Departure",
        "Has_TSR": False, "Has_Fog": False, "Has_Congestion": False, "Primary_Reason": "Normal"
    })

    for idx in range(1, len(commercial_stops)):
        stop = commercial_stops.iloc[idx]
        seq = stop["stop_seq"]
        sim_row = sim_journey[sim_journey["stop_seq"] == seq]
        if sim_row.empty: continue
       
        row_mask = journey_features_raw["next_station_name"] == stop["station_name"]
        if not row_mask.any(): continue
       
        raw_row = journey_features_raw[row_mask].iloc[0]
        enc_row = journey_features_enc[row_mask].iloc[0]

        X = enc_row[feat_cols].to_frame().T.astype(float)
        ml_delay = max(0.0, float(model.predict(X)[0]))

        naive_delay = raw_row["cur_delay_minutes"]
        naive_eta = raw_row["next_scheduled_min"] + naive_delay
        ml_eta = raw_row["next_scheduled_min"] + ml_delay
        actual = sim_row.iloc[0]["actual_arrival_min"]
       
        tsr_val = raw_row.get("upcoming_tsr_frac", 0)
        fog_val = raw_row.get("upcoming_foggy_frac", 0)
        cong_val = raw_row.get("upcoming_congestion_mean", 0)
       
        # Calculate Dominant Reason for the Delay Segments Map
        primary_reason = "Normal"
        if ml_delay > 5:
            max_val = max(tsr_val, fog_val, cong_val)
            if max_val == cong_val and cong_val > 0.1: primary_reason = "Congestion"
            elif max_val == tsr_val and tsr_val > 0.05: primary_reason = "TSR"
            elif max_val == fog_val and fog_val > 0.1: primary_reason = "Weather"
            else: primary_reason = "Historical"

        results.append({
            "Station": stop["station_name"], "Lat": float(stop["lat"]), "Lng": float(stop["lon"]),
            "Scheduled": round(raw_row["next_scheduled_min"], 1),
            "Static_ETA": round(naive_eta, 1),
            "AI_ETA": round(ml_eta, 1),
            "Actual": round(actual, 1),
            "Static_Error": round(abs(naive_eta - actual), 1),
            "AI_Error": round(abs(ml_eta - actual), 1),
            "Explanation": generate_heuristic_explanation(ml_delay, naive_delay, raw_row),
            "Has_TSR": bool(tsr_val > 0.05),
            "Has_Fog": bool(fog_val > 0.1),
            "Has_Congestion": bool(cong_val > 0.2),
            "Primary_Reason": primary_reason
        })

    origin_dep = str(origin["departure"])
    if pd.isna(origin["departure"]):
        origin_dep = "13:00:00"

    return {
        "data": results,
        "origin_departure": origin_dep
    }