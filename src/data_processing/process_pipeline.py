import pandas as pd
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
folder_path = os.path.join(root_dir, "data", "processed")

final_df = pd.DataFrame()
daily_freq = ["c2_", "c1_", "m3_", "m5_"]
monthly_freq = ["e8_", "e7_", "s3_"]

all_files = os.listdir(folder_path)

# Lọc file daily
daily_files = [
    f for f in all_files
    if os.path.isfile(os.path.join(folder_path, f))
    and f.startswith(tuple(daily_freq))
]

# Lọc file monthly
monthly_files = [
    f for f in all_files
    if os.path.isfile(os.path.join(folder_path, f))
    and f.startswith(tuple(monthly_freq))
]

print("Daily files:", daily_files)
print("Monthly files:", monthly_files)

daily_df = pd.DataFrame()

def z_score(s):
    # Làm mịn dữ liệu: kẹp biên ở phân vị 1% và 99% để loại bỏ outliers và giảm kurtosis cực đoan
    q_low = s.quantile(0.01)
    q_high = s.quantile(0.99)
    s_clipped = s.clip(q_low, q_high)
    # Chuẩn hóa Z-score
    return (s_clipped - s_clipped.mean()) / (s_clipped.std() + 1e-9)

def combine_daily_features():
    for f in daily_files: 
        file_path = os.path.join(folder_path, f)
        if "c1" in f:
            c1_csv = pd.read_csv(file_path)
            daily_df["amihud_diff_normalized"] = c1_csv["amihud_diff_normalized"]
        elif "c2" in f:
            c2_csv = pd.read_csv(file_path)
            daily_df["ret_disp"] = c2_csv["ret_disp"]
        elif "m3" in f: 
            m3_csv = pd.read_csv(file_path)
            daily_df["rolling_vol_5"] = m3_csv["rolling_vol_5"]
        elif "m5" in f:
            m5_csv = pd.read_csv(file_path)
            daily_df["volume_ratio"] = m5_csv["volume_ratio"]
            
monthly_df = pd.DataFrame()

def combine_monthly_feature():
    for f in monthly_files: 
        file_path = os.path.join(folder_path, f)
        if "e8" in f:
            e8_csv = pd.read_csv(file_path)
            # Chuyển đổi CPI từ dạng lũy kế năm (YoY) sang tốc độ tăng hằng tháng (MoM inflation rate)
            monthly_df["cpi_mom"] = e8_csv["cpi_index"].pct_change() * 100
            # Điền NaN đầu tiên bằng giá trị kế tiếp hoặc 0
            monthly_df["cpi_mom"] = monthly_df["cpi_mom"].bfill().fillna(0.0)
        elif "e7" in f:
            e7_csv = pd.read_csv(file_path)
            # Làm mịn tăng trưởng tín dụng thành mức thay đổi theo tháng (MoM difference)
            monthly_df["credit_growth_mom"] = e7_csv["credit_growth_yoy"].diff()
            monthly_df["credit_growth_mom"] = monthly_df["credit_growth_mom"].bfill().fillna(0.0)
        elif "s3" in f: 
            s3_csv = pd.read_csv(file_path)
            monthly_df["pmi_vn"] = s3_csv["pmi_vn"]


combine_daily_features()
combine_monthly_feature()

daily_df["time"] = pd.read_csv(os.path.join(folder_path, "m2_vix.csv"))["time"]
monthly_df["time"] = pd.read_csv(os.path.join(folder_path, "s3_pmi_vietnam.csv"))["time"]

print("\n--- Daily Features Head ---")
print(daily_df.head(5))
print("\n--- Monthly Features Head ---")
print(monthly_df.head(5))

def async_freq_daily_monthly(daily, monthly):
    # Sao chép để tránh thay đổi DataFrame gốc
    daily_sorted = daily.copy()
    monthly_sorted = monthly.copy()
    
    # Chuyển cột time sang datetime
    daily_sorted["time"] = pd.to_datetime(daily_sorted["time"])
    monthly_sorted["time"] = pd.to_datetime(monthly_sorted["time"])
    
    # Sắp xếp tăng dần theo thời gian (bắt buộc đối với pd.merge_asof)
    daily_sorted = daily_sorted.sort_values("time")
    monthly_sorted = monthly_sorted.sort_values("time")
    
    # Ghép dữ liệu daily với dữ liệu monthly gần nhất trước đó
    result = pd.merge_asof(
        daily_sorted,
        monthly_sorted,
        on="time",
        direction="backward"
    )
    
    # Đưa cột 'time' lên vị trí đầu tiên
    cols = ["time"] + [col for col in result.columns if col != "time"]
    result = result[cols]
    
    return result


# Gọi hàm để kiểm tra và in kết quả
final_df = async_freq_daily_monthly(daily_df, monthly_df)

# Áp dụng làm mịn (winsorize) và chuẩn hóa Z-score cho tất cả các biến đặc trưng
feature_cols = [col for col in final_df.columns if col != 'time']
for col in feature_cols:
    final_df[col] = z_score(final_df[col])

print("\n--- Merged & Normalized DataFrame (Head 5) ---")
print(final_df.head(5))
print(f"Total rows: {len(final_df)}")

# Tạo thư mục output nếu chưa tồn tại
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Lưu cả hai định dạng tệp để tương thích tốt với các notebook
final_df.to_csv(os.path.join(output_dir, "TFT_HMM_BL.csv"), index=False)
final_df.to_csv(os.path.join(output_dir, "hmm_data.csv"), index=False)
print(f"Save DataFrame to {os.path.join(output_dir, 'TFT_HMM_BL.csv')} and {os.path.join(output_dir, 'hmm_data.csv')}")
