import os
import pandas as pd

# File paths
input_csv = os.path.join("data", "processed", "simulated_runs.csv")
output_csv = os.path.join("data", "processed", "simulated_runs.csv")

if not os.path.exists(input_csv):
    print(f"Error: Could not find {input_csv}. Ensure you are running this from your project root.")
else:
    print("Reading original simulated_runs.csv...")
    df = pd.read_csv(input_csv, low_memory=False)
   
    orig_size_mb = os.path.getsize(input_csv) / (1024 * 1024)
    print(f"Original File Size: {orig_size_mb:.2f} MB")
    print(f"Original Row Count: {len(df):,} rows")
   
    # Get unique trains
    unique_trains = sorted(df['train_number'].astype(str).unique())
    print(f"Total Trains Found: {len(unique_trains)}")
   
    # Keep train 12502 (our primary demo train) plus 15 other trains
    target_train_prefix = "12502"
    demo_trains = [t for t in unique_trains if t.startswith(target_train_prefix)]
   
    # Fill remaining slots up to 20 trains
    for t in unique_trains:
        if t not in demo_trains:
            demo_trains.append(t)
        if len(demo_trains) >= 20:
            break
           
    print(f"Filtering dataset down to {len(demo_trains)} key trains...")
   
    # Filter dataset
    filtered_df = df[df['train_number'].astype(str).isin(demo_trains)].copy()
   
    # Overwrite the CSV with the lightweight version
    filtered_df.to_csv(output_csv, index=False)
   
    new_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print(f"New File Size: {new_size_mb:.2f} MB")
    print(f"New Row Count: {len(filtered_df):,} rows")
    print("Successfully shortened simulated_runs.csv for Render deployment!")