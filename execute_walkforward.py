import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

parquet_path = r'c:\Users\ADMIN\Desktop\Kaggle\output\hmm_v3_op1_extended\master_drl_ready_full.parquet'
xlsx_path = r'c:\Users\ADMIN\Desktop\Kaggle\output\hmm_v3_op1_extended\master_drl_ready_full.xlsx'

print("Reading data...")
df = pd.read_parquet(parquet_path)

# 1. Tạo target
df['target_return_1d'] = df.groupby('ticker')['close'].pct_change(1).shift(-1)
df['target_bin'] = (df['target_return_1d'] > 0).astype(int)

# 2. Định nghĩa Feature Cols
semantic_sector_probs = [col for col in df.columns if col.startswith('prob_sector_')]
feature_cols = [col for col in df.columns if col.startswith('prob_market_')] + semantic_sector_probs + ['rolling_vol_20d', 'return_5d', 'volume_ratio']

# Keep only columns that actually exist
feature_cols = [c for c in feature_cols if c in df.columns]

# 3. Chạy Walk-Forward
df_backtest = df.dropna(subset=['target_return_1d']).reset_index(drop=True)
start_test_date = pd.Timestamp('2022-01-01')
test_dates = sorted(df_backtest[df_backtest['time'] >= start_test_date]['time'].unique())

df_backtest['final_meta_pred_prob'] = np.nan

print(f"Running Meta-Classifier Walk-Forward for {len(test_dates)} days...")
for current_date in tqdm(test_dates):
    train_mask = df_backtest['time'] < current_date
    X_train = df_backtest.loc[train_mask, feature_cols]
    y_train = df_backtest.loc[train_mask, 'target_bin']
    
    test_mask = df_backtest['time'] == current_date
    X_test = df_backtest.loc[test_mask, feature_cols]
    
    if len(X_train) < 1000 or len(X_test) == 0: continue
        
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    df_backtest.loc[test_mask, 'final_meta_pred_prob'] = probs

# Merge back
print("Merging results...")
df = df.merge(df_backtest[['time', 'ticker', 'final_meta_pred_prob']], on=['time', 'ticker'], how='left')

# Save Parquet
print("Saving Parquet...")
df.to_parquet(parquet_path, index=False)

# Save Excel
print("Saving XLSX...")
for col in df.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns, Asia/Ho_Chi_Minh]']).columns:
    df[col] = df[col].dt.tz_localize(None)
df.to_excel(xlsx_path, index=False)
print("DONE! XLSX now has final_meta_pred_prob.")
