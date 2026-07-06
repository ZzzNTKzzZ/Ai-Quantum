import pandas as pd
import os

parquet_path = r'c:\Users\ADMIN\Desktop\Kaggle\output\hmm_v3_op1_extended\master_drl_ready_full.parquet'
xlsx_path = r'c:\Users\ADMIN\Desktop\Kaggle\output\hmm_v3_op1_extended\master_drl_ready_full.xlsx'

print(f"Reading from {parquet_path}...")
try:
    df = pd.read_parquet(parquet_path)
    # Convert timezone-aware datetime to timezone-naive for Excel compatibility
    for col in df.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns, Asia/Ho_Chi_Minh]']).columns:
        df[col] = df[col].dt.tz_localize(None)
    print(f"Writing to {xlsx_path} (this may take a moment)...")
    df.to_excel(xlsx_path, index=False)
    print("Conversion to XLSX successful!")
except Exception as e:
    print(f"Error: {e}")
