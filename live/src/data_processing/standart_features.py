import pandas as pd
import os
import datetime

# Đường dẫn thư mục tương đối an toàn dựa trên vị trí file script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
folder_path = os.path.join(root_dir, "data", "processed")

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Get list of CSV files
file_list = [
    f for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')
]

log(f"Found {len(file_list)} CSV file(s) to clean and fix.")

for file in file_list:
    file_path = os.path.join(folder_path, file)
    log(f"--- Fixing file: {file} ---")
    
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # 1. Lowercase all headers first to handle any mixed case issues
        df.columns = df.columns.str.lower()
        
        # 2. Find the best candidate for the real date/time column
        # We prioritize 'time', then 'date', then anything starting with 'time.' or 'date.'
        all_cols = list(df.columns)
        target_col = None
        
        # Priority order to find the true source of truth for time
        for candidate in ["time", "date", "observation_date", "day"]:
            if candidate in all_cols:
                target_col = candidate
                break
        
        # If explicit names aren't found, look for duplicated clones like 'time.1'
        if not target_col:
            clone_cols = [c for c in all_cols if c.startswith("time.") or c.startswith("date.")]
            if clone_cols:
                target_col = clone_cols[0] # Pick the first clone as the source
                log(f"No base time column found. Using clone '{target_col}' as source.")

        if target_col:
            log(f"Extracting valid timeline from '{target_col}'...")
            # Convert the chosen column to datetime standard
            df["valid_time"] = pd.to_datetime(df[target_col], errors="coerce")
            
            # 3. Clean-up: Drop ALL columns that match time/date keywords or their clones
            cols_to_drop = [
                c for c in df.columns 
                if c in ["date", "time", "observation_date", "day"] or c.startswith("time.") or c.startswith("date.") or c.startswith("unnamed")
            ]
            df = df.drop(columns=cols_to_drop, errors="ignore")
            
            # 4. Rename our clean timeline to the final 'time'
            df = df.rename(columns={"valid_time": "time"})
            
            # Move 'time' column to the front or back (Optional, but keeps it organized)
            # Let's push 'time' to be the first column for better readability
            remaining_cols = [c for c in df.columns if c != "time"]
            df = df[["time"] + remaining_cols]
            
        else:
            raise ValueError("Could not find any source or clone for date/time in this file.")

        # 5. Save the perfectly cleaned DataFrame
        df.to_csv(file_path, index=False)
        log(f"SUCCESS: Cleaned and unified {file} successfully.")
        
    except Exception as e:
        log(f"ERROR: Failed to clean {file}. Details: {e}")

log("Data rescue and pipeline execution finished.")