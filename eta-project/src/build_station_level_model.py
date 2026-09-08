"""
build_station_level_model.py
------------------------------
Rebuilds the feature set and retrains the model at COMMERCIAL-STOP
granularity (station to station), instead of raw trackside-waypoint
granularity. This is what a real passenger-facing ETA product actually
needs: "what time will the train reach the next real station" -- not
"what time will it reach some untitled trackside point 8 waypoints away".

A commercial stop = a station with halt_minutes > 0, or the first/last stop
of the journey (matches what real ETA apps display).

Much faster than the original build_features.py since there are only ~8-15
commercial stops per journey instead of ~100-220 trackside waypoints.

Outputs:
  data/processed/station_level_features.csv
  outputs/xgb_station_model.json
  outputs/station_model_metrics.json
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

STOPS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "train_stops.csv")
SIM_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "simulated_runs.csv")
FEAT_OUT = os.path.join(PROJECT_ROOT, "data", "processed", "station_level_features.csv")
MODEL_OUT = os.path.join(PROJECT_ROOT, "outputs", "xgb_station_model.json")
METRICS_OUT = os.path.join(PROJECT_ROOT, "outputs", "station_model_metrics.json")

TRAIN_DAYS_CUTOFF = "2026-01-24"


def get_commercial_stop_seqs_per_train(stops_df):
    """Returns {train_number: set(stop_seq values that are commercial stops)}"""
    result = {}
    for tn, g in stops_df.groupby("train_number"):
        max_seq = g["stop_seq"].max()
        commercial = g[(g["halt_minutes"] > 0) | (g["stop_seq"] == 0) | (g["stop_seq"] == max_seq)]
        result[tn] = set(commercial["stop_seq"].tolist())
    return result


def main():
    print("Loading static stops and simulated runs...")
    stops_df = pd.read_csv(STOPS_CSV, low_memory=False)
    sim_df = pd.read_csv(SIM_CSV, low_memory=False)
    sim_df["sim_date"] = pd.to_datetime(sim_df["sim_date"])

    sampled_trains = sim_df["train_number"].unique()
    stops_df = stops_df[stops_df["train_number"].isin(sampled_trains)]

    print("Identifying commercial stops per train...")
    commercial_map = get_commercial_stop_seqs_per_train(stops_df)

    sim_df["is_train_period"] = sim_df["sim_date"] < TRAIN_DAYS_CUTOFF

    journey_total_dist = (
        sim_df.groupby(["train_number", "sim_date"])["cum_distance_km"]
        .max()
        .rename("journey_total_distance_km")
    )
    sim_df = sim_df.join(journey_total_dist, on=["train_number", "sim_date"])

    print("Building station-to-station feature pairs...")
    rows = []
    grouped = sim_df.groupby(["train_number", "sim_date"], sort=False)
    n_journeys = len(grouped)

    for gi, ((train_number, sim_date), g) in enumerate(grouped):
        if gi % 1000 == 0:
            print(f"  processing journey {gi}/{n_journeys}...")

        commercial_seqs = commercial_map.get(train_number)
        if not commercial_seqs:
            continue

        g = g.sort_values("stop_seq").reset_index(drop=True)
        g = g[g["stop_seq"].isin(commercial_seqs)].reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue

        delay_arr = g["delay_minutes"].to_numpy()
        cum_dist_arr = g["cum_distance_km"].to_numpy()
        dist_prev_arr = g["dist_from_prev_km"].fillna(0).to_numpy()
        tsr_arr = g["tsr_active"].to_numpy()
        congestion_arr = g["congestion_index"].to_numpy()
        foggy_arr = (g["weather_state"] == "foggy").to_numpy()
        stop_seq_arr = g["stop_seq"].to_numpy()
        station_name_arr = g["station_name"].to_numpy()
        station_zone_arr = g["station_zone"].to_numpy()
        sched_arr = g["scheduled_arrival_min"].to_numpy()
        journey_total_dist = g["journey_total_distance_km"].iloc[0]

        cur_train_type = g["train_type"].iloc[0]
        cur_train_zone = g["train_zone"].iloc[0]
        cur_priority = g["priority_rank"].iloc[0]
        cur_is_train_period = g["is_train_period"].iloc[0]

        for i in range(n - 1):
            j = i + 1  # next COMMERCIAL stop (station-to-station, no fixed horizon needed)

            upcoming_dist_km = float(dist_prev_arr[j])
            upcoming_tsr_frac = float(tsr_arr[j])
            upcoming_congestion_mean = float(congestion_arr[j])
            upcoming_foggy_frac = float(foggy_arr[j])

            rows.append((
                train_number, sim_date, cur_train_type, cur_train_zone,
                cur_priority, cur_is_train_period,
                station_name_arr[i], station_name_arr[j],
                stop_seq_arr[i], delay_arr[i], cum_dist_arr[i],
                journey_total_dist - cum_dist_arr[i],
                sched_arr[i], sched_arr[j],
                upcoming_dist_km, upcoming_tsr_frac, upcoming_congestion_mean,
                upcoming_foggy_frac, station_zone_arr[j],
                delay_arr[j],
            ))

    col_names = [
        "train_number", "sim_date", "train_type", "train_zone",
        "priority_rank", "is_train_period",
        "cur_station_name", "next_station_name",
        "cur_stop_seq", "cur_delay_minutes", "cur_cum_distance_km",
        "distance_remaining_km", "cur_scheduled_min", "next_scheduled_min",
        "upcoming_dist_km", "upcoming_tsr_frac", "upcoming_congestion_mean",
        "upcoming_foggy_frac", "next_station_zone", "target_delay_minutes",
    ]
    feat_df = pd.DataFrame(rows, columns=col_names)
    print(f"Built {len(feat_df)} station-to-station pairs.")

    print("Computing historical averages (train-period only)...")
    train_only = feat_df[feat_df["is_train_period"]]
    hist_by_train = train_only.groupby("train_number")["target_delay_minutes"].mean().rename("hist_avg_delay_by_train")
    hist_by_type = train_only.groupby("train_type")["target_delay_minutes"].mean().rename("hist_avg_delay_by_type")
    overall_hist_avg = train_only["target_delay_minutes"].mean()

    feat_df = feat_df.join(hist_by_train, on="train_number")
    feat_df = feat_df.join(hist_by_type, on="train_type")
    feat_df["hist_avg_delay_by_train"] = feat_df["hist_avg_delay_by_train"].fillna(overall_hist_avg)
    feat_df["hist_avg_delay_by_type"] = feat_df["hist_avg_delay_by_type"].fillna(overall_hist_avg)

    feat_df.to_csv(FEAT_OUT, index=False)
    print(f"Saved features to {FEAT_OUT}")

    # --- Train model ---
    print("\nTraining station-level XGBoost model...")
    model_df = pd.get_dummies(
        feat_df, columns=["train_type", "train_zone", "next_station_zone"], dummy_na=False
    )

    non_feature_cols = [
        "train_number", "sim_date", "is_train_period", "target_delay_minutes",
        "cur_station_name", "next_station_name",
    ]
    feature_cols = [c for c in model_df.columns if c not in non_feature_cols]

    train_rows = model_df[model_df["is_train_period"]]
    test_rows = model_df[~model_df["is_train_period"]]
    print(f"Train rows: {len(train_rows)}, Test rows: {len(test_rows)}")

    X_train, y_train = train_rows[feature_cols], train_rows["target_delay_minutes"]
    X_test, y_test = test_rows[feature_cols], test_rows["target_delay_minutes"]

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 0, None)
    naive_preds = test_rows["cur_delay_minutes"].values

    def report(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"  {name}: MAE={mae:.2f} min, RMSE={rmse:.2f} min, R2={r2:.3f}")
        return {"mae": mae, "rmse": rmse, "r2": r2}

    print("\n=== Station-to-station results (test period) ===")
    naive_metrics = report("Naive (assume delay unchanged from last station)", y_test, naive_preds)
    model_metrics = report("XGBoost station-level model", y_test, preds)
    improvement = 100 * (naive_metrics["mae"] - model_metrics["mae"]) / naive_metrics["mae"]
    print(f"\nMAE improvement over naive baseline: {improvement:.1f}%")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save_model(MODEL_OUT)

    with open(METRICS_OUT, "w") as f:
        json.dump({
            "naive_baseline": naive_metrics,
            "xgboost_model": model_metrics,
            "mae_improvement_pct": improvement,
            "n_train_rows": len(train_rows),
            "n_test_rows": len(test_rows),
        }, f, indent=2)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")


if __name__ == "__main__":
    main()
