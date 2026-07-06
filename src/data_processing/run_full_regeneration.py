import os
import shutil
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
    if res.returncode != 0:
        print(f"FAILED: {cmd}")
        print(res.stderr)
        raise RuntimeError(f"Command {cmd} failed.")
    else:
        print(res.stdout)

# Step 1: Copy test/ files to raw/
print("Step 1: Copying new crawled data from data/test to data/raw...")
test_dir = os.path.join(root_dir, "data", "test")
raw_dir = os.path.join(root_dir, "data", "raw")
for f in os.listdir(test_dir):
    if f.endswith('.csv'):
        shutil.copy(os.path.join(test_dir, f), os.path.join(raw_dir, f))
print("All files copied successfully!")

# Step 2: Rebuild m1_vn46.csv from raw stocks
print("\nStep 2: Rebuilding m1_vn46.csv...")
run_cmd("python src/data_processing/rebuild_m1.py")

# Step 3: Run restore_raw_data.py
print("\nStep 3: Restoring raw data to processed...")
run_cmd("python src/data_collection/restore_raw_data.py")

# Step 4: Run derived_variable.py
print("\nStep 4: Running derived_variable.py...")
run_cmd("python src/data_processing/derived_variable.py")

# Step 5: Run align_daily_features.py
print("\nStep 5: Running align_daily_features.py...")
run_cmd("python src/data_processing/align_daily_features.py")

# Step 6: Run slit_date.py
print("\nStep 6: Running slit_date.py...")
run_cmd("python src/data_processing/slit_date.py")

# Step 7: Run process_pipeline.py
print("\nStep 7: Running process_pipeline.py...")
run_cmd("python src/data_processing/process_pipeline.py")

print("\nALL STEPS COMPLETED SUCCESSFULLY!")
