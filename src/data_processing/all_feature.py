import pandas as pd
import os
import datetime

# Đường dẫn tương đối an toàn dựa trên vị trí file script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
folder_path = os.path.join(root_dir, "data", "processed")

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

file_list = [
    f for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')
]

log(f"Found {len(file_list)} CSV file(s) to clean and fix.")

for file in file_list:
    file_path = os.path.join(folder_path, file)

    log(f"--- Fixing file: {file} ---")

    df = pd.read_csv(file_path)

    log(f"Features in {file}: {list(df.columns)}, number: {len(df.columns)}")
    log(f"Number of rows in {file}: {len(df)}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        log(f"Start date in {file}: {df['time'].min()}")
        log(f"End date in {file}: {df['time'].max()}")