"""
train_model.py
--------------
Trains an XGBoost regression model to predict delay at the next stop, using
the feature table built by build_features.py. Evaluates it against a NAIVE
baseline (the current static-ETA approach: "assume delay stays the same as
right now") -- beating this naive baseline is the core proof-of-value for
the hackathon pitch.

Train/test split: by DATE (already encoded in is_train_period from the
feature-building step), NOT randomly by row, since random splitting would
leak information between stops of the same journey/day.
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IN_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "model_features.csv")
MODEL_OUT = os.path.join(PROJECT_ROOT, "outputs", "xgb_model.json")
METRICS_OUT = os.path.join(PROJECT_ROOT, "outputs", "metrics.json")
IMPORTANCE_OUT = os.path.join(PROJECT_ROOT, "outputs", "feature_importance.csv")

# Columns that are identifiers/labels/leakage risks -- never fed to the model
NON_FEATURE_COLS = [
    "train_number", "sim_date", "is_train_period", "target_delay_minutes",
]


def main():
    print("Loading feature table...")
    df = pd.read_csv(IN_CSV, low_memory=False)

    train_df = df[df["is_train_period"]].copy()
    test_df = df[~df["is_train_period"]].copy()
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Using {len(feature_cols)} features.")

    X_train, y_train = train_df[feature_cols], train_df["target_delay_minutes"]
    X_test, y_test = test_df[feature_cols], test_df["target_delay_minutes"]

    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("Predicting on test set...")
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # delay can't be negative

    # --- Naive baseline: "next-stop delay = current delay" (this is
    # essentially what static-schedule + current-delay systems assume,
    # since they don't model segment-specific real-time conditions) ---
    naive_preds = test_df["cur_delay_minutes"].values

    def report(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"  {name}: MAE={mae:.2f} min, RMSE={rmse:.2f} min, R2={r2:.3f}")
        return {"mae": mae, "rmse": rmse, "r2": r2}

    print("\n=== Results (test period, last 6 simulated days) ===")
    naive_metrics = report("Naive baseline (assume delay unchanged)", y_test, naive_preds)
    model_metrics = report("XGBoost model", y_test, preds)

    improvement_pct = 100 * (naive_metrics["mae"] - model_metrics["mae"]) / naive_metrics["mae"]
    print(f"\nMAE improvement over naive baseline: {improvement_pct:.1f}%")

    # --- Feature importance ---
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print("\nTop 10 most important features:")
    print(importance.head(10))

    # --- Save everything ---
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save_model(MODEL_OUT)

    metrics = {
        "naive_baseline": naive_metrics,
        "xgboost_model": model_metrics,
        "mae_improvement_pct": improvement_pct,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_features": len(feature_cols),
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    importance.to_csv(IMPORTANCE_OUT, header=["importance"])

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")
    print(f"Saved feature importance to {IMPORTANCE_OUT}")


if __name__ == "__main__":
    main()
