import os
import sys
import datetime
import numpy as np
import pandas as pd

# Configure stdout for UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Đường dẫn thư mục xử lý dữ liệu tương đối an toàn dựa trên vị trí file script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
folder_path = os.path.join(root_dir, "data", "processed")

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def read_sort_csv(file):
    df = pd.read_csv(file)
    # Chuyển tất cả tên cột thành chữ thường để thống nhất
    df.columns = df.columns.str.lower()
    # Tìm cột ngày tháng và đổi tên thành 'time'
    date_cols = ["time", "date", "quarter", "observation_date"]
    found_col = None
    for col in date_cols:
        if col in df.columns:
            found_col = col
            break
    if found_col is not None:
        df["time"] = pd.to_datetime(df[found_col])
        if found_col != "time":
            df = df.drop(columns=[found_col])
    else:
        raise KeyError(f"No date column found in {file}. Columns: {df.columns.tolist()}")
    df = df.sort_values("time").reset_index(drop=True)
    return df

def m1(file):
    log("Calculating indicators for M1: log_return, rolling_vol_20d, return_5d, return_20d, volume_ratio...")
    df = read_sort_csv(file)
    required_cols = ["log_return", "rolling_vol_20d", "return_5d", "return_20d", "volume_ratio"]
    if not all(col in df.columns for col in required_cols):
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["rolling_vol_20d"] = df["log_return"].rolling(20).std() * np.sqrt(252)
        df["return_5d"] = df["close"] / df["close"].shift(5) - 1
        df["return_20d"] = df["close"] / df["close"].shift(20) - 1
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def m2(file):
    log("Calculating indicators for M2: Log Vix")
    df = read_sort_csv(file)
    required_cols = ["close_vix", "high_vix", "low_vix", "open_vix"]
    if not all(col in df.columns for col in required_cols):
        df["close_vix"] = np.log(df["close"] / df["close"].shift(1))
        df["high_vix"] = np.log(df["high"] / df["high"].shift(1))
        df["low_vix"] = np.log(df["low"] / df["low"].shift(1))
        df["open_vix"] = np.log(df["open"] / df["open"].shift(1))
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def m3(file):
    log("Calculating indicators for M3: Log return, rolling 5-day")
    df = read_sort_csv(file)
    required_cols = ["log_return", "rolling_vol_5"]
    if not all(col in df.columns for col in required_cols):
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["rolling_vol_5"] = df["log_return"].rolling(5).std() * np.sqrt(252)
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def m4(file):
    log("Calculating indicators for M4: Fnb_ratio")
    df = read_sort_csv(file)
    required_cols = ["fnb_ratio"]
    if not all(col in df.columns for col in required_cols):
        df["fnb_ratio"] = (df["buy"] - df["sale"]) / (df["buy"] + df["sale"])
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def m5(file):
    log("Calculating indicators for M5: volume_ratio, volume_z20...")
    df = read_sort_csv(file)
    required_cols = ["volume_ratio", "volume_z20"]
    if not all(col in df.columns for col in required_cols):
        roll_mean = df["volume"].rolling(20).mean()
        roll_std = df["volume"].rolling(20).std()
        df["volume_ratio"] = df["volume"] / roll_mean
        df["volume_z20"] = (df["volume"] - roll_mean) / roll_std
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def e1(file):
    log("Calculating indicators for E1: fx_log_ret...")
    df = read_sort_csv(file)
    required_cols = ["fx_log_ret"]
    if not all(col in df.columns for col in required_cols):
        df["fx_log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def e2(file):
    log("Calculating indicators for E2: interbank_on_diff (winsorized)...")
    df = read_sort_csv(file)
    required_cols = ["interbank_on_diff"]
    if not all(col in df.columns for col in required_cols):
        base_col = "fir_pa" if "fir_pa" in df.columns else "fid_pa"
        diff_series = df[base_col].diff()
        q_low = diff_series.quantile(0.005)
        q_high = diff_series.quantile(0.995)
        df["interbank_on_diff"] = diff_series.clip(lower=q_low, upper=q_high)
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def e3(file):
    log("Calculating indicators for E3: vn5y_yield_diff...")
    df = read_sort_csv(file)
    required_cols = ["vn5y_yield_diff"]
    if not all(col in df.columns for col in required_cols):
        base_col = "vn5y_yield" if "vn5y_yield" in df.columns else "close"
        df["vn5y_yield_diff"] = df[base_col].diff()
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def e4(file):
    log("Calculating indicators for E4: epu_log...")
    df = read_sort_csv(file)
    required_cols = ["epu_log"]
    if not all(col in df.columns for col in required_cols):
        base_col = "usepuindxd" if "usepuindxd" in df.columns else df.columns[1]
        df["epu_log"] = np.log(df[base_col].replace(0, np.nan))
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def e5(file):
    log("Calculating indicators for E5: gpr_log...")
    df = read_sort_csv(file)
    required_cols = ["gpr_log"]
    if not all(col in df.columns for col in required_cols):
        base_col = "gprd_act" if "gprd_act" in df.columns else ("gprd" if "gprd" in df.columns else df.columns[1])
        df["gpr_log"] = np.log(df[base_col].replace(0, np.nan))
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def e6(file):
    log("Calculating indicators for E6: m2_growth_yoy_diff...")
    df = read_sort_csv(file)
    required_cols = ["m2_growth_yoy_diff"]
    if not all(col in df.columns for col in required_cols):
        df["m2_growth_yoy_diff"] = df["m2_growth_yoy"].diff()
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def e7(file):
    log("Calculating indicators for E7: credit_growth_yoy_diff...")
    df = read_sort_csv(file)
    required_cols = ["credit_growth_yoy_diff"]
    if not all(col in df.columns for col in required_cols):
        df["credit_growth_yoy_diff"] = df["credit_growth_yoy"].diff()
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def e8(file):
    log("Calculating indicators for E8: cpi_yoy, cpi_mom_diff...")
    df = read_sort_csv(file)
    required_cols = ["cpi_yoy", "cpi_mom_diff"]
    if not all(col in df.columns for col in required_cols):
        df["cpi_yoy"] = df["cpi_index"] / df["cpi_index"].shift(12) - 1
        df["cpi_mom_diff"] = df["cpi_index"].diff()
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def s1(file):
    log("Calculating indicators for S1: oil_ret_5d...")
    df = read_sort_csv(file)
    required_cols = ["oil_ret_5d"]
    if not all(col in df.columns for col in required_cols):
        df["oil_ret_5d"] = np.log(df["close"] / df["close"].shift(5))
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def s2(file):
    log("Calculating indicators for S2: gold_ret...")
    df = read_sort_csv(file)
    required_cols = ["gold_ret"]
    if not all(col in df.columns for col in required_cols):
        df["gold_ret"] = np.log(df["close"] / df["close"].shift(1))
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def s3(file):
    log("Calculating indicators for S3: pmi_vn_above50...")
    df = read_sort_csv(file)
    required_cols = ["pmi_vn_above50"]
    if not all(col in df.columns for col in required_cols):
        df["pmi_vn_above50"] = (df["pmi_vn"] > 50).astype(int)
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def s4(file):
    log("Calculating indicators for S4: copper_ret_5d...")
    df = read_sort_csv(file)
    required_cols = ["copper_ret_5d"]
    if not all(col in df.columns for col in required_cols):
        df["copper_ret_5d"] = np.log(df["close"] / df["close"].shift(5))
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def c1(file):
    log("Calculating indicators for C1: amihud_diff...")
    df = read_sort_csv(file)
    required_cols = ["amihud_diff"]
    if not all(col in df.columns for col in required_cols):
        df["amihud_diff"] = df["amihud_diff_normalized"] if "amihud_diff_normalized" in df.columns else np.log(df["illiq_roll20"]).diff() * 1e9
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def c2(file):
    log("Calculating indicators for C2: ret_disp...")
    df = read_sort_csv(file)
    required_cols = ["ret_disp"]
    if not all(col in df.columns for col in required_cols):
        df["ret_disp"] = df["ret_disp_smooth"]
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def g1(file):
    log("Calculating indicators for G1: dxy_ret...")
    df = read_sort_csv(file)
    required_cols = ["dxy_ret"]
    if not all(col in df.columns for col in required_cols):
        df["dxy_ret"] = np.log(df["close"] / df["close"].shift(1))
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def g2(file):
    log("Calculating indicators for G2: fed_rate_diff...")
    df = read_sort_csv(file)
    required_cols = ["fed_rate_diff"]
    if not all(col in df.columns for col in required_cols):
        df["fed_rate_diff"] = df["fedfunds"].diff()
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def g3(file):
    log("Calculating indicators for G3: us10y_diff...")
    df = read_sort_csv(file)
    required_cols = ["us10y_diff"]
    if not all(col in df.columns for col in required_cols):
        df["us10y_diff"] = df["close"].diff()
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def g4(file):
    log("Calculating indicators for G4: china_ret_5d...")
    df = read_sort_csv(file)
    required_cols = ["china_ret_5d"]
    if not all(col in df.columns for col in required_cols):
        df["china_ret_5d"] = np.log(df["close"] / df["close"].shift(5))
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def g5(file):
    log("Calculating indicators for G5: fdi_realized_yoy...")
    df = read_sort_csv(file)
    required_cols = ["fdi_realized_yoy"]
    if not all(col in df.columns for col in required_cols):
        # Detect frequency by looking at the diff of time
        time_diffs = df["time"].diff().dt.days.dropna()
        if not time_diffs.empty and time_diffs.median() > 60:
            # Quarterly
            shift_periods = 4
        else:
            # Monthly
            shift_periods = 12
        df["fdi_realized_yoy"] = df["fdi_usd_million"] / df["fdi_usd_million"].shift(shift_periods) - 1
    df.to_csv(file, index=False)
    log(f"Saved {file}")

def main():
    log("Starting derived variables calculation...")
    if not os.path.exists(folder_path):
        log(f"Error: folder {folder_path} not found.")
        return
        
    file_list = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')
    ]
    log(f"Found {len(file_list)} CSV files in {folder_path}.")
    
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        
        # Match file to corresponding function
        if file.startswith("m1_"):
            m1(file_path)
        elif file.startswith("m2_"):
            m2(file_path)
        elif file.startswith("m3_"):
            m3(file_path)
        elif file.startswith("m4_"):
            m4(file_path)
        elif file.startswith("m5_"):
            m5(file_path)
        elif file.startswith("e1_"):
            e1(file_path)
        elif file.startswith("e2_"):
            e2(file_path)
        elif file.startswith("e3_"):
            e3(file_path)
        elif file.startswith("e4_"):
            e4(file_path)
        elif file.startswith("e5_"):
            e5(file_path)
        elif file.startswith("e6_"):
            e6(file_path)
        elif file.startswith("e7_"):
            e7(file_path)
        elif file.startswith("e8_"):
            e8(file_path)
        elif file.startswith("s1_"):
            s1(file_path)
        elif file.startswith("s2_"):
            s2(file_path)
        elif file.startswith("s3_"):
            s3(file_path)
        elif file.startswith("s4_"):
            s4(file_path)
        elif file.startswith("c1_"):
            c1(file_path)
        elif file.startswith("c2_"):
            c2(file_path)
        elif file.startswith("g1_"):
            g1(file_path)
        elif file.startswith("g2_"):
            g2(file_path)
        elif file.startswith("g3_"):
            g3(file_path)
        elif file.startswith("g4_"):
            g4(file_path)
        elif file.startswith("g5_"):
            g5(file_path)

if __name__ == "__main__":
    main()