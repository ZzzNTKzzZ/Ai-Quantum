import os
import pandas as pd
import datetime

# Đường dẫn thư mục xử lý dữ liệu tương đối an toàn dựa trên vị trí file script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
folder_path = os.path.join(root_dir, "data", "processed")

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def main():
    m1_path = os.path.join(folder_path, "m1_vn46.csv")
    if not os.path.exists(m1_path):
        log(f"Error: {m1_path} not found.")
        return

    # 1. Load m1 to get the master trading date grid
    log("Loading master date grid from m1_vn46.csv...")
    df_m1 = pd.read_csv(m1_path)
    df_m1["time"] = pd.to_datetime(df_m1["time"])
    if df_m1["time"].dt.tz is not None:
        df_m1["time"] = df_m1["time"].dt.tz_localize(None)
    
    # Sort and get unique dates
    master_dates = sorted(df_m1["time"].dt.normalize().unique())
    log(f"Master trading dates: {len(master_dates)} days (from {master_dates[0].strftime('%Y-%m-%d')} to {master_dates[-1].strftime('%Y-%m-%d')})")
    
    # List of daily files to align
    daily_prefixes = [
        "c1_", "c2_", "e1_", "e2_", "e3_", "e4_", "e5_",
        "g1_", "g3_", "g4_", "m2_", "m3_", "m4_", "m5_",
        "s1_", "s2_", "s4_"
    ]
    
    file_list = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')
    ]
    
    for file in file_list:
        # Check if it is a daily file (based on prefix)
        is_daily = any(file.startswith(prefix) for prefix in daily_prefixes)
        if not is_daily:
            continue
            
        file_path = os.path.join(folder_path, file)
        log(f"Aligning daily file: {file}...")
        
        df = pd.read_csv(file_path)
        df["time"] = pd.to_datetime(df["time"])
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_localize(None)
            
        df["time"] = df["time"].dt.normalize()
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        
        # Reindex to master dates
        df = df.set_index("time")
        df = df.reindex(master_dates)
        
        # Forward-fill first for missing days. We DO NOT backward-fill (bfill) to avoid look-ahead bias.
        df = df.ffill()
        
        # Reset index to make 'time' a column again
        df = df.reset_index()
        df["time"] = df["time"].dt.strftime('%Y-%m-%d')
        
        df.to_csv(file_path, index=False)
        log(f"Saved aligned file: {file_path} (rows: {len(df)})")

if __name__ == "__main__":
    main()
