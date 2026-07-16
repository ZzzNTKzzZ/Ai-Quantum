import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys

# Đảm bảo in tiếng Việt ra console không bị lỗi
sys.stdout.reconfigure(encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "output", "master_drl_ready_ticker.parquet")

if not os.path.exists(data_path):
    print(f"Lỗi: Không tìm thấy file {data_path}. Vui lòng chạy HMM Pipeline trước.")
    sys.exit(1)

print("Đang nạp dữ liệu...")
master_ticker = pd.read_parquet(data_path)

# 1. Chuẩn bị Target (Dự báo ngày T+1, T+2, T+3)
master_ticker['target_return_1d'] = master_ticker.groupby('ticker')['close'].pct_change(1).shift(-1)
master_ticker['target_bin_1d'] = (master_ticker['target_return_1d'] > 0).astype(int)

master_ticker['target_return_2d'] = master_ticker.groupby('ticker')['close'].pct_change(2).shift(-2)
master_ticker['target_bin_2d'] = (master_ticker['target_return_2d'] > 0).astype(int)

master_ticker['target_return_3d'] = master_ticker.groupby('ticker')['close'].pct_change(3).shift(-3)
master_ticker['target_bin_3d'] = (master_ticker['target_return_3d'] > 0).astype(int)

# 2. Định nghĩa các cột Features
macro_probs = [col for col in master_ticker.columns if col.startswith('Macro_Prob_')]
market_probs = [col for col in master_ticker.columns if col.startswith('Market_Prob_')]
sector_probs = [col for col in master_ticker.columns if col.startswith('prob_sector_')]
ticker_probs = [col for col in master_ticker.columns if col.startswith('prob_ticker_')]
feature_cols = ticker_probs + sector_probs + market_probs + macro_probs + ['rolling_vol_20d', 'return_5d', 'return_20d', 'rolling_vol_5', 'mom_1M', 'dist_MA50', 'volume_ratio']

# Kiểm tra xem dữ liệu có đủ feature không
missing_features = [f for f in feature_cols if f not in master_ticker.columns]
if missing_features:
    print(f"Cảnh báo: Thiếu các features: {missing_features}")
    feature_cols = [f for f in feature_cols if f in master_ticker.columns]

# 3. Phân chia Train/Test cho Live Trading
latest_date = master_ticker['time'].max()
print(f"\n=== CHẾ ĐỘ LIVE TRADING ===")
print(f"Ngày giao dịch mới nhất (T): {latest_date.strftime('%Y-%m-%d')}")

# Test: Chỉ lấy đúng ngày hiện tại (T)
test_mask = master_ticker['time'] == latest_date
X_test_live = master_ticker.loc[test_mask, feature_cols]
live_results = master_ticker.loc[test_mask, ['time', 'ticker', 'close']].copy()

# 4. Huấn luyện Meta-Classifier và Dự báo cho T+1, T+2, T+3
for horizon in [1, 2, 3]:
    target_col = f'target_return_{horizon}d'
    bin_col = f'target_bin_{horizon}d'
    
    # Train: Bỏ qua các ngày cuối (vì target bị NaN) và lấy tất cả lịch sử trước đó
    train_mask = (master_ticker['time'] < latest_date) & (master_ticker[target_col].notna())
    X_train_live = master_ticker.loc[train_mask, feature_cols]
    y_train_live = master_ticker.loc[train_mask, bin_col]
    
    print(f"Đang huấn luyện mô hình LightGBM cho T+{horizon} ({len(X_train_live)} mẫu dữ liệu)...")
    clf_live = lgb.LGBMClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        random_state=42, 
        verbose=-1, 
        n_jobs=-1, 
        class_weight='balanced'
    )
    clf_live.fit(X_train_live, y_train_live)
    
    # Dự báo
    probs_live = clf_live.predict_proba(X_test_live)[:, 1]
    live_results[f'XS Tăng T+{horizon}'] = np.round(probs_live, 4)

# 5. Tổng hợp bảng tín hiệu
# Trung bình xác suất của 3 ngày để làm tín hiệu chính thức
live_results['Xác Suất Tăng'] = live_results[['XS Tăng T+1', 'XS Tăng T+2', 'XS Tăng T+3']].mean(axis=1)
live_results['Tín Hiệu'] = live_results['Xác Suất Tăng'].apply(lambda x: 'Tăng (Khuyên Mua)' if x > 0.5 else 'Giảm (Cảnh Báo)')
live_results = live_results.sort_values('Xác Suất Tăng', ascending=False).reset_index(drop=True)

print("\n🏆 TOP 15 MÃ CỔ PHIẾU TIỀM NĂNG NHẤT (Trung bình XS T+1, T+2, T+3):")
# In format đẹp hơn
pd.options.display.float_format = '{:.4f}'.format
print(live_results.head(15).to_string(index=False))

# Lưu tín hiệu ra CSV
output_csv = os.path.join(script_dir, "output", f"live_trading_signals_{latest_date.strftime('%Y%m%d')}.csv")
live_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n[OK] Đã xuất toàn bộ bảng tín hiệu ra: {output_csv}")
