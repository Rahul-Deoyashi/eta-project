"""
build_features.py
------------------
Turns simulated_runs.csv into an ML-ready table for predicting delay at the
NEXT stop, given information available at the CURRENT stop plus real-time
conditions on the segment ahead (TSR, weather, congestion -- these represent
live/forecastable signals a real control-room system would have, not future
"cheating" information).

PREDICTION FRAMING (important, read this before changing anything):
  Given a train currently at stop i (with its delay-so-far, distance
  covered, historical performance, etc.) and known real-time conditions on
  the segment from stop i to stop i+1 (TSR active?, weather?, congestion
  level?), predict delay_minutes AT STOP i+1.

  This mirrors a real deployed system: every time a train reports its
  position, the system recomputes ETA for the next station using live
  conditions on the segment ahead. Chaining this prediction forward
  (recursively, feeding each predicted delay back in as "current delay")
  is how you'd extend it to predict ETA at stations further down the line.

LEAKAGE PREVENTION:
  - Historical averages (per train, per train_type) are computed using ONLY
    the TRAIN-period simulated days (first 24 of 30), then applied to both
    train and test rows. Test-period rows never contribute to these stats.
  - The train/test split itself is by DATE (first 24 days train, last 6 days
    test) rather than random row sampling, since random splitting would let
    the model see other stops of the SAME journey in the training set for
    the same day, which does not reflect the future.
  - The target (delay_minutes at i+1) is NEVER used to compute any feature
    for row i.

Output: data/processed/model_features.csv
"""

import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IN_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "simulated_runs.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "model_features.csv")

TRAIN_DAYS_CUTOFF = "2026-01-24"  # days < this = train period, >= this = test period
HORIZON = 8  # predict delay this many stops ahead (not just the immediate next stop)


def main():
    print("Loading simulated_runs.csv...")
    df = pd.read_csv(IN_CSV, low_memory=False)
    df["sim_date"] = pd.to_datetime(df["sim_date"])
    df = df.sort_values(["train_number", "sim_date", "stop_seq"]).reset_index(drop=True)

    print(f"Loaded {len(df)} rows across {df['train_number'].nunique()} trains, "
          f"{df['sim_date'].nunique()} days.")

    # Total route distance per (train_number, sim_date) journey -- needed for
    # "distance remaining" feature. Since the static route is the same every
    # day, this is really per-train, but computing per (train,date) is safe
    # and simple.
    print("Computing per-journey total distance...")
    journey_total_dist = (
        df.groupby(["train_number", "sim_date"])["cum_distance_km"]
        .max()
        .rename("journey_total_distance_km")
    )
    df = df.join(journey_total_dist, on=["train_number", "sim_date"])

    # --- Build "current stop" -> "next stop" pairs within each journey ---
    print("Building current-stop -> next-stop pairs...")
    df["is_train_period"] = df["sim_date"] < TRAIN_DAYS_CUTOFF

    grouped = df.groupby(["train_number", "sim_date"], sort=False)

    rows = []
    n_journeys = len(grouped)
    for gi, ((train_number, sim_date), g) in enumerate(grouped):
        if gi % 1000 == 0:
            print(f"  processing journey {gi}/{n_journeys}...")
        g = g.sort_values("stop_seq").reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue

        # Pull everything into plain numpy arrays once per journey -- avoids
        # slow repeated pandas .iloc calls inside the inner loop.
        delay_arr = g["delay_minutes"].to_numpy()
        cum_dist_arr = g["cum_distance_km"].to_numpy()
        dist_prev_arr = g["dist_from_prev_km"].fillna(0).to_numpy()
        tsr_arr = g["tsr_active"].to_numpy()
        congestion_arr = g["congestion_index"].to_numpy()
        foggy_arr = (g["weather_state"] == "foggy").to_numpy()
        stop_seq_arr = g["stop_seq"].to_numpy()
        station_zone_arr = g["station_zone"].to_numpy()
        journey_total_dist = g["journey_total_distance_km"].iloc[0]

        cur_train_type = g["train_type"].iloc[0]
        cur_train_zone = g["train_zone"].iloc[0]
        cur_priority = g["priority_rank"].iloc[0]
        cur_is_train_period = g["is_train_period"].iloc[0]

        for i in range(n - 1):
            target_idx = min(i + HORIZON, n - 1)
            if target_idx == i:
                continue

            start = max(0, i - 2)
            avg_delay_last3 = float(np.mean(delay_arr[start:i + 1]))

            # aggregated real-time/forecastable conditions over the upcoming
            # stretch (i+1 .. target_idx)
            upcoming_slice = slice(i + 1, target_idx + 1)
            upcoming_dist_km = float(np.sum(dist_prev_arr[upcoming_slice]))
            upcoming_tsr_frac = float(np.mean(tsr_arr[upcoming_slice]))
            upcoming_congestion_mean = float(np.mean(congestion_arr[upcoming_slice]))
            upcoming_foggy_frac = float(np.mean(foggy_arr[upcoming_slice]))

            rows.append((
                train_number, sim_date, cur_train_type, cur_train_zone,
                cur_priority, cur_is_train_period,
                stop_seq_arr[i], delay_arr[i], cum_dist_arr[i], avg_delay_last3,
                journey_total_dist - cum_dist_arr[i], target_idx - i,
                upcoming_dist_km, upcoming_tsr_frac, upcoming_congestion_mean,
                upcoming_foggy_frac, station_zone_arr[target_idx],
                delay_arr[target_idx],
            ))

    col_names = [
        "train_number", "sim_date", "train_type", "train_zone",
        "priority_rank", "is_train_period",
        "cur_stop_seq", "cur_delay_minutes", "cur_cum_distance_km",
        "avg_delay_last3_stops", "distance_remaining_km", "stops_ahead",
        "upcoming_dist_km", "upcoming_tsr_frac", "upcoming_congestion_mean",
        "upcoming_foggy_frac", "next_station_zone", "target_delay_minutes",
    ]
    feat_df = pd.DataFrame(rows, columns=col_names)
    print(f"Built {len(feat_df)} (current->next) training pairs.")

    # --- Historical average features, computed ONLY from train-period rows ---
    print("Computing historical averages (train-period only, to avoid leakage)...")
    train_only = feat_df[feat_df["is_train_period"]]

    hist_by_train = (
        train_only.groupby("train_number")["target_delay_minutes"]
        .mean()
        .rename("hist_avg_delay_by_train")
    )
    hist_by_type = (
        train_only.groupby("train_type")["target_delay_minutes"]
        .mean()
        .rename("hist_avg_delay_by_type")
    )
    overall_hist_avg = train_only["target_delay_minutes"].mean()

    feat_df = feat_df.join(hist_by_train, on="train_number")
    feat_df = feat_df.join(hist_by_type, on="train_type")
    feat_df["hist_avg_delay_by_train"] = feat_df["hist_avg_delay_by_train"].fillna(overall_hist_avg)
    feat_df["hist_avg_delay_by_type"] = feat_df["hist_avg_delay_by_type"].fillna(overall_hist_avg)

    # --- One-hot encode categoricals the model will use ---
    print("One-hot encoding categorical features...")
    feat_df = pd.get_dummies(
        feat_df,
        columns=["train_type", "train_zone", "next_station_zone"],
        dummy_na=False,
    )

    feat_df.to_csv(OUT_CSV, index=False)
    print(f"Done. Wrote {len(feat_df)} rows, {feat_df.shape[1]} columns to {OUT_CSV}")
    print(f"Train-period rows: {feat_df['is_train_period'].sum()}, "
          f"Test-period rows: {(~feat_df['is_train_period']).sum()}")


if __name__ == "__main__":
    main()
