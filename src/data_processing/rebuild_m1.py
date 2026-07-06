import os
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

m1_tickers = ['BID', 'MWG', 'NKG', 'NLG', 'NT2', 'PDR', 'PHR', 'PNJ', 'PVD', 'PVT', 'REE', 'BMP',
 'SBT', 'SJS', 'SSI', 'STB', 'TCH', 'VCB', 'VHC', 'VIC', 'MSN', 'VNM', 'BVH', 'KDH',
 'CTG', 'DCM', 'DGW', 'DIG', 'DPM', 'DXG', 'EIB', 'FPT', 'GAS', 'CII', 'GMD', 'HAG',
 'HCM', 'HDC', 'HPG', 'HSG', 'HT1', 'KBC', 'KDC', 'MBB', 'VSC', 'CTD']

print("Starting to rebuild m1_vn46.csv from raw stock data...")
dfs = []
for t in m1_tickers:
    p = os.path.join(root_dir, 'data', 'stocks', f'{t}.csv')
    if not os.path.exists(p):
        print(f"Warning: {t}.csv not found at {p}")
        continue
    df = pd.read_csv(p)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df['ticker'] = t
    
    # Calculate indicators per ticker correctly
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['rolling_vol_20d'] = df['log_return'].rolling(20).std() * np.sqrt(252)
    df['return_5d'] = df['close'] / df['close'].shift(5) - 1
    df['return_20d'] = df['close'] / df['close'].shift(20) - 1
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['og_return'] = df['close'].pct_change()
    
    dfs.append(df)

if dfs:
    m1_df = pd.concat(dfs, ignore_index=True)
    m1_df = m1_df.sort_values('time').reset_index(drop=True)
    
    # Filter to start from 2016-10-05 onwards (it already does, but let's be sure)
    m1_df = m1_df[m1_df['time'] >= '2016-10-05'].copy()
    
    output_path = os.path.join(root_dir, 'data', 'processed', 'm1_vn46.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m1_df.to_csv(output_path, index=False)
    print(f"Successfully rebuilt m1_vn46.csv! Saved to {output_path} (rows: {len(m1_df)})")
else:
    print("Error: No stock files were loaded.")
