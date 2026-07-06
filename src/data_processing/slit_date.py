import pandas as pd
import os

# Đường dẫn tương đối an toàn dựa trên vị trí file script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
folder_path = os.path.join(root_dir, "data", "processed")

file_list = [
    f for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')
]
for file in file_list:
    full_file_path = os.path.join(folder_path, file)
    df = pd.read_csv(full_file_path)

    # Chuyển sang datetime
    if "Date" in df.columns:
        df["time"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.drop("Date", axis=1)

    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.drop("date", axis=1)
    elif "observation_date" in df.columns:
        df["time"] = pd.to_datetime(df["observation_date"], errors="coerce")
        df = df.drop("observation_date", axis=1)
    else:
        raise ValueError("Không tìm thấy cột ngày tháng")
    # Phân loại daily vs monthly dựa trên tiền tố
    monthly_prefixes = ["e6_", "e7_", "e8_", "s3_", "g5_", "g2_"]
    is_monthly = any(file.startswith(p) for p in monthly_prefixes)

    if is_monthly:
        df_f = df[df["time"] >= "2016-10-01"].copy()
    else:
        df_f = df[df["time"] >= "2016-10-05"].copy()

    # Ghi đè lại file gốc
    df_f.to_csv(full_file_path, index=False)

    print(f"success {file}")