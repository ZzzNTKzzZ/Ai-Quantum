import os
import yfinance as yf
import datetime
from vnstock import Market

START_TIME = "2016-09-05"
END_TIME = "2026-01-05"

# Đường dẫn tương đối an toàn dựa trên vị trí file script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
DATA_DIR = os.path.join(root_dir, "data", "test")
os.makedirs(DATA_DIR, exist_ok=True)

market = Market()
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")



# Id M2 - M5
def crawl_raw_m2_m5():
    log("Starting M2-M5 raw data crawl...")
    try:
        # Id M2
        log("Downloading M2 (VIX)...")
        df_vix = yf.download('^VIX', start=START_TIME, end=END_TIME)
        df_vix.to_csv(os.path.join(DATA_DIR, "m2_vix.csv"))
        log("M2 (VIX) saved.")
    except Exception as e:
        log(f"Error in crawl_raw_m2: {e}")
    try:
        # Id M3 
        log("Downloading M3 (S&P 500)...")
        df_sp500 = yf.download('^GSPC', start=START_TIME, end=END_TIME)
        df_sp500.to_csv(os.path.join(DATA_DIR, "m3_sp500.csv"))
        log("M3 (S&P 500) saved.")
    except Exception as e:
        log(f"Error in crawl_raw_m3: {e}")
    try:
        # Id M4
        log("Generating M4 (Foreign Net Buy/Sell)...")
        df_vni = yf.download('VN100', start=START_TIME, end=END_TIME)
        df_vni.to_csv(os.path.join(DATA_DIR, "m4_vni.csv"))
        log("M4 (Vietnam 10Y Bond Yield) saved.")
    except Exception as e:
        log(f"Error in crawl_raw_m4: {e}")
    

log("Downloading M2 (VIX)...")
df_vix = yf.download('^VIX', start=START_TIME, end=END_TIME)
df_vix.to_csv(os.path.join(DATA_DIR, "m2_vix.csv"))
log("M2 (VIX) saved.")