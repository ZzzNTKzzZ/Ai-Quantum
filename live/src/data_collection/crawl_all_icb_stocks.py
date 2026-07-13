import os
import sys
import pandas as pd
import numpy as np
import datetime
import time

try:
    import vnai
    import vnai.beam.patching
    vnai.beam.patching.apply_all_patches = lambda *args, **kwargs: {}
except Exception:
    pass

sys.stdout.reconfigure(encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
RAW_DATA_DIR = os.path.join(root_dir, "data", "stocks")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

START_DATE = '2016-11-10'
END_DATE = '2026-05-01'
MIN_SESSIONS = 2300

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def download_ticker(symbol):
    try:
        import vnstock
        out_path = os.path.join(RAW_DATA_DIR, f"{symbol}.csv")
        if os.path.exists(out_path):
            return {'symbol': symbol, 'status': 'SKIPPED', 'sessions': 0, 'first_date': ''}
            
        time.sleep(3.2) # Avoid Rate Limit
        q = vnstock.Quote(symbol=symbol, source='kbs')
        df = q.history(start=START_DATE, end=END_DATE)
        if df is not None and not df.empty:
            date_col = 'time' if 'time' in df.columns else ('Date' if 'Date' in df.columns else None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                df.to_csv(out_path, index=False)
                
                first_date = df[date_col].min().strftime('%Y-%m-%d')
                num_sessions = len(df)
                return {
                    'symbol': symbol,
                    'status': 'SUCCESS',
                    'first_date': first_date,
                    'sessions': num_sessions
                }
        return {'symbol': symbol, 'status': 'EMPTY'}
    except Exception as e:
        return {'symbol': symbol, 'status': 'ERROR', 'error': str(e)}

def main():
    log("==========================================")
    log("ICB Stocks Crawler Started (Rate-Limited)")
    log("==========================================")
    
    csv_path = os.path.join(script_dir, "industries.csv")
    if not os.path.exists(csv_path):
        log(f"Error: {csv_path} not found. Please run get_industries.py first.")
        return
        
    df_icb = pd.read_csv(csv_path)
    all_symbols = sorted(df_icb['symbol'].unique().tolist())
    log(f"Loaded {len(all_symbols)} unique symbols from industries.csv")
    
    success_tickers = []
    skipped_tickers = []
    failed_tickers = []
    
    start_time = time.time()
    
    for idx, symbol in enumerate(all_symbols):
        completed = idx + 1
        res = download_ticker(symbol)
        
        if res['status'] == 'SUCCESS':
            success_tickers.append(res)
            log(f"[{completed}/{len(all_symbols)}] {symbol}: SUCCESS ({res['sessions']} sessions)")
        elif res['status'] == 'SKIPPED':
            skipped_tickers.append(res)
            # log(f"[{completed}/{len(all_symbols)}] {symbol}: SKIPPED (already exists)")
        elif res['status'] == 'ERROR':
            failed_tickers.append(res)
            log(f"[{completed}/{len(all_symbols)}] {symbol}: ERROR - {res.get('error', '')}")
        else:
            log(f"[{completed}/{len(all_symbols)}] {symbol}: EMPTY")
            
    elapsed = time.time() - start_time
    log("==========================================")
    log(f"Scanning completed in {elapsed:.2f} seconds.")
    log(f"Success: {len(success_tickers)}")
    log(f"Skipped (already exist): {len(skipped_tickers)}")
    log(f"Failed/Error: {len(failed_tickers)}")
    log("==========================================")

if __name__ == "__main__":
    main()
