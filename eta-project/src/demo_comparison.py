"""
demo_comparison.py
-------------------
Generates a clean, presentation-ready comparison table for a SINGLE chosen
train + simulated day: Scheduled ETA vs Static/Naive ETA (mirrors what
current systems like NTES / Where-is-my-Train effectively do: assume the
current delay persists unchanged) vs our ML Dynamic ETA vs the Actual outcome.

Only real COMMERCIAL stops (halt_minutes > 0, or the very first/last stop)
are shown -- the raw dataset includes every trackside waypoint, which is
correct for model training (more granularity) but clutters a demo table.

Usage: edit TRAIN_NUMBER and SIM_DATE below, then run:
    python src/demo_comparison.py

Outputs a CSV to outputs/demo_comparison_<train>_<date>.csv, printable /
loadable directly into the Streamlit app (app.py).
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

STOPS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "train_stops.csv")
SIM_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "simulated_runs.csv")
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "station_level_features.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "xgb_station_model.json")

import sys

TRAIN_NUMBER = sys.argv[1] if len(sys.argv) > 1 else "12301"
SIM_DATE = sys.argv[2] if len(sys.argv) > 2 else "2026-01-25"


def get_commercial_stops(train_number):
    df = pd.read_csv(STOPS_CSV, low_memory=False)
    sub = df[df["train_number"].astype(str) == str(train_number)].sort_values("stop_seq")
    max_seq = sub["stop_seq"].max()
    real_stops = sub[(sub["halt_minutes"] > 0) | (sub["stop_seq"] == 0) | (sub["stop_seq"] == max_seq)]
    return real_stops.reset_index(drop=True)


def main():
    print(f"Building demo comparison for train {TRAIN_NUMBER} on {SIM_DATE}...")

    commercial_stops = get_commercial_stops(TRAIN_NUMBER)
    print(f"Found {len(commercial_stops)} commercial stops: "
          f"{', '.join(commercial_stops['station_name'])}")

    sim = pd.read_csv(SIM_CSV, low_memory=False)
    sim_journey = sim[
        (sim["train_number"].astype(str) == str(TRAIN_NUMBER)) &
        (sim["sim_date"] == SIM_DATE)
    ].sort_values("stop_seq").reset_index(drop=True)

    if sim_journey.empty:
        raise ValueError(f"No simulated data for train {TRAIN_NUMBER} on {SIM_DATE}. "
                          f"Check the train was in the 300-train sample and the date "
                          f"is within the simulated 30-day range.")

    print("Loading trained model...")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    feat_cols = model.get_booster().feature_names

    features_all_raw = pd.read_csv(FEATURES_CSV, low_memory=False)
    features_all = pd.get_dummies(
        features_all_raw, columns=["train_type", "train_zone", "next_station_zone"], dummy_na=False
    )
    # align to exactly the columns the model was trained on (fills any
    # missing dummy columns with 0, e.g. train types not seen for this train)
    for col in feat_cols:
        if col not in features_all.columns:
            features_all[col] = 0
    journey_features = features_all[
        (features_all["train_number"].astype(str) == str(TRAIN_NUMBER)) &
        (features_all["sim_date"] == SIM_DATE)
    ].sort_values("cur_stop_seq").reset_index(drop=True)

    results = []
    # first commercial stop is the origin -- no prediction needed for it
    origin = commercial_stops.iloc[0]
    sim_origin = sim_journey[sim_journey["stop_seq"] == origin["stop_seq"]].iloc[0]
    results.append({
        "Station": origin["station_name"],
        "Scheduled (min from origin)": round(sim_origin["scheduled_arrival_min"], 1),
        "Static/Naive ETA (min)": round(sim_origin["scheduled_arrival_min"], 1),
        "ML Dynamic ETA (min)": round(sim_origin["scheduled_arrival_min"], 1),
        "Actual (min)": round(sim_origin["actual_arrival_min"], 1),
        "Naive Error (min)": 0.0,
        "ML Error (min)": 0.0,
    })

    for idx in range(1, len(commercial_stops)):
        stop = commercial_stops.iloc[idx]
        seq = stop["stop_seq"]
        sim_row = sim_journey[sim_journey["stop_seq"] == seq]
        if sim_row.empty:
            continue
        sim_row = sim_row.iloc[0]

        # the feature row whose "next_station_name" matches this stop's name
        # (built by build_station_level_model.py, one row per station-to-
        # station hop) gives us the naive + ML prediction made FROM the
        # previous commercial stop.
        feat_row = journey_features[journey_features["next_station_name"] == stop["station_name"]]
        if feat_row.empty:
            continue
        feat_row = feat_row.iloc[0]

        scheduled_min = feat_row["next_scheduled_min"]
        actual_min = sim_row["actual_arrival_min"]

        naive_delay = feat_row["cur_delay_minutes"]
        X = feat_row[feat_cols].to_frame().T.astype(float)
        ml_delay = max(0.0, float(model.predict(X)[0]))

        naive_eta_min = scheduled_min + naive_delay
        ml_eta_min = scheduled_min + ml_delay

        results.append({
            "Station": stop["station_name"],
            "Scheduled (min from origin)": round(scheduled_min, 1),
            "Static/Naive ETA (min)": round(naive_eta_min, 1),
            "ML Dynamic ETA (min)": round(ml_eta_min, 1),
            "Actual (min)": round(actual_min, 1),
            "Naive Error (min)": round(abs(naive_eta_min - actual_min), 1),
            "ML Error (min)": round(abs(ml_eta_min - actual_min), 1),
        })

    result_df = pd.DataFrame(results)
    print()
    print(result_df.to_string(index=False))

    print()
    print(f"Mean Naive Error: {result_df['Naive Error (min)'].mean():.2f} min")
    print(f"Mean ML Error:    {result_df['ML Error (min)'].mean():.2f} min")

    out_path = os.path.join(
        PROJECT_ROOT, "outputs", f"demo_comparison_{TRAIN_NUMBER}_{SIM_DATE}.csv"
    )
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
