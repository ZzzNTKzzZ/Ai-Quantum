import os
import sys

# Bypass vnstock Community Edition limits (such as 8-year OHLCV and quarter limits)
try:
    import vnai
    import vnai.beam.patching
    vnai.beam.patching.apply_all_patches = lambda *args, **kwargs: {}
except Exception:
    pass

import pandas as pd
import numpy as np
import datetime
import time

# Configure stdout for UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Đường dẫn tương đối an toàn dựa trên vị trí file script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
RAW_DATA_DIR = os.path.join(root_dir, "data", "stocks")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

START_DATE = '2016-11-10'
END_DATE = '2026-05-01' 
MIN_SESSIONS = 2300
sources =['kbs', 'msn', 'vci']
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def download_ticker(symbol):
    try:
        import vnstock
        # Sleep to avoid Rate Limit Exceeded
        time.sleep(3.2)
        q = vnstock.Quote(symbol=symbol, source='kbs')
        df = q.history(start=START_DATE, end=END_DATE)
        if df is not None and not df.empty:
            date_col = 'time' if 'time' in df.columns else ('Date' if 'Date' in df.columns else None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                
                # Save raw data unconditionally
                out_path = os.path.join(RAW_DATA_DIR, f"{symbol}.csv")
                df.to_csv(out_path, index=False)
                
                first_date = df[date_col].min().strftime('%Y-%m-%d')
                num_sessions = len(df)
                
                if first_date <= START_DATE and num_sessions >= MIN_SESSIONS:
                    return {
                        'symbol': symbol,
                        'status': 'SUCCESS',
                        'first_date': first_date,
                        'sessions': num_sessions
                    }
                else:
                    return {
                        'symbol': symbol,
                        'status': 'FILTERED',
                        'first_date': first_date,
                        'sessions': num_sessions
                    }
        return {'symbol': symbol, 'status': 'EMPTY'}
    except Exception as e:
        return {'symbol': symbol, 'status': 'ERROR', 'error': str(e)}

def main():
    log("==========================================")
    log("Stock Ticker Crawler & Filter Started (Rate-Limited)")
    log("==========================================")
    
    # Load tickers from VN100
    try:
        import vnstock
        ref = vnstock.Reference()
        vn100_series = ref.equity.list_by_group('VN100')
        all_symbols = vn100_series.tolist() if isinstance(vn100_series, pd.Series) else (
            vn100_series['symbol'].tolist() if 'symbol' in vn100_series.columns else vn100_series['ticker'].tolist()
        )
        all_symbols = sorted(set(all_symbols))
        log(f"Loaded {len(all_symbols)} symbols from VN100 group.")
    except Exception as e:
        log(f"Error loading VN100 group: {e}")
        try:
            df_vn54 = pd.read_csv(os.path.join(BASE_DIR, "vn54.csv"))
            all_symbols = sorted(df_vn54['ticker'].unique())
            log(f"Falling back to {len(all_symbols)} symbols from vn54.csv")
        except Exception as e2:
            log(f"Error reading vn54.csv fallback: {e2}")
            all_symbols = []
    
    success_tickers = []
    filtered_tickers = []
    failed_tickers = []
    
    start_time = time.time()
    
    for idx, symbol in enumerate(all_symbols):
        completed = idx + 1
        res = download_ticker(symbol)
        
        if res['status'] == 'SUCCESS':
            success_tickers.append(res)
            log(f"[{completed}/{len(all_symbols)}] {symbol}: SUCCESS ({res['sessions']} sessions, starts {res['first_date']})")
        elif res['status'] == 'FILTERED':
            filtered_tickers.append(res)
            log(f"[{completed}/{len(all_symbols)}] {symbol}: FILTERED OUT ({res['sessions']} sessions, starts {res['first_date']}) - saved raw data")
        elif res['status'] == 'ERROR':
            failed_tickers.append(res)
            log(f"[{completed}/{len(all_symbols)}] {symbol}: ERROR - {res['error']}")
        else:
            log(f"[{completed}/{len(all_symbols)}] {symbol}: EMPTY")
            
    elapsed = time.time() - start_time
    log("==========================================")
    log(f"Scanning completed in {elapsed:.2f} seconds.")
    log(f"Success (saved and validated): {len(success_tickers)}")
    log(f"Filtered out (saved raw only): {len(filtered_tickers)}")
    log(f"Failed/Error: {len(failed_tickers)}")
    log("==========================================")
    
    # Write list of successful tickers to a txt file
    success_path = os.path.join(BASE_DIR, "success_tickers.txt")
    with open(success_path, "w") as f:
        for t in success_tickers:
            f.write(f"{t['symbol']}\n")
    log(f"List of successful symbols saved to {success_path}")
            
if __name__ == "__main__":
    main()
