# HYBRID DUAL-FREQUENCY TICKER-SPECIFIC HMM - CHI TIET TOAN BO QUY TRINH

**Ngon ngu: Vietnamese (Tieng Viet)**

---

## TONG QUAN TOAN BO QUY TRINH

- **Tong cong so phan**: 30 phan
  - Markdown (giai thich): 15 phan
  - Code (thuc hien): 15 phan

**Chuc nang chinh:**
- Ket hop danh gia VI MAO (Monthly) va THI TRUONG CHUNG (Daily)
- Dinh vi pha bien dong va trang thai th? truong
- Su dung HMM multi-layer: Macro -> Market -> Sector -> Ticker
- Dieu luyen tren du lieu tich su va dang ky hoat dong giao dich

---

## PHAN 1: MARKDOWN (Giai Thich)
**Noi dung:**

```
# Hybrid Dual-Frequency Ticker-Specific HMM

Quy trình kết hợp đánh giá Vĩ mô (Monthly) và Thị trường chung (Daily) để định vị pha biến động, sau đó ép cấu trúc thị trường chung lên từng mã cổ phiếu (...
```

## PHAN 2: CODE (Thuc Hien)
**Dong so**: 41 dong
**Noi dung chinh (15 dong dau):**

```python
def get_hmm_filtered_inference(model, Z):
    import numpy as np
    N = len(Z)
    K = model.n_components
    filtered_probs = np.zeros((N, K))
    filtered_regimes = np.zeros(N, dtype=int)
    for t in range(1, N + 1):
        Z_slice = Z[:t]
        try:
            probs_slice = model.predict_proba(Z_slice)
            filtered_probs[t-1] = probs_slice[-1]
            regimes_slice = model.predict(Z_slice)
            filtered_regimes[t-1] = regimes_slice[-1]
        except Exception as e:
            filtered_probs[t-1] = np.ones(K) / K

# ... (26 tong lines con lai) ...
```

## PHAN 3: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 1. Tải Dữ Liệu & Tạo Chỉ Báo Thị Trường Đại Diện (VN-Index Proxy)
```

## PHAN 4: CODE (Thuc Hien)
**Dong so**: 34 dong
**Noi dung chinh (15 dong dau):**

```python
print("Đang tải dữ liệu hmm_data.csv và m1_vn46.csv...")
df_daily_base = pd.read_csv('../output/hmm_data.csv')
df_daily_base['time'] = pd.to_datetime(df_daily_base['time'])

df_m1 = pd.read_csv('../data/processed/m1_vn46.csv')
df_m1['time'] = pd.to_datetime(df_m1['time']).dt.normalize()

# Tạo Market Proxy từ rổ VN46
market_ret = df_m1.groupby('time')['log_return'].mean().reset_index()
market_ret.columns = ['time', 'vnindex_log_ret']
market_close = df_m1.groupby('time')['close'].mean().reset_index()
market_close.columns = ['time', 'vnindex_close']

df_market = df_daily_base.merge(market_ret, on='time', how='left')
df_market = df_market.merge(market_close, on='time', how='left')

# ... (19 tong lines con lai) ...
```

## PHAN 5: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 2. Bộ Lọc Kiểm Định Kỹ Thuật (Stationarity & Kurtosis)

Mục đích của bước này là đánh giá tính chất toán học của các đặc trưng để chọn lọc đầu vào ổn định cho HMM:
1. **Kiểm định tính dừng (Station...
```

## PHAN 6: CODE (Thuc Hien)
**Dong so**: 27 dong
**Noi dung chinh (15 dong dau):**

```python
def check_stationarity(s):
    s = s.dropna()
    if len(s) < 30: return False, np.nan, np.nan, np.nan, np.nan
    p_adf = adfuller(s, autolag='AIC')[1]
    p_kpss = kpss(s, regression='c', nlags='auto')[1] 
    kurt = kurtosis(s)
    skw = skew(s)
    is_stat = (p_adf < 0.05) and (p_kpss >= 0.05) and (abs(kurt) < 10)
    return is_stat, p_adf, p_kpss, kurt, skw

daily_pool = [c for c in df_market.columns if c not in ['time', 'vnindex_log_ret', 'vnindex_close', 'vnindex_vol20']]
stat_results = []
for c in daily_pool:
    is_stat, p_a, p_k, kurt, skw = check_stationarity(df_market[c])
    stat_results.append({'feature': c, 'is_stationary': is_stat, 'p_adf': p_a, 'p_kpss': p_k, 'kurtosis': kurt, 'skewness': skw})

# ... (12 tong lines con lai) ...
```

## PHAN 7: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 3. Thiết Lập Tập Dữ Liệu Train/OOS & Hàm Hỗ Trợ HMM sử dụng NQT + Rank
```

## PHAN 8: CODE (Thuc Hien)
**Dong so**: 20 dong
**Noi dung chinh (15 dong dau):**

```python
HMM_TRAIN_END = pd.Timestamp('2019-12-31')

def make_Z(df, features, window=252):
    fd = df[['time'] + features].dropna().reset_index(drop=True)
    nqt_df = pd.DataFrame(index=fd.index)
    for col in features:
        rolling_rank = fd[col].rolling(window=window, min_periods=1).rank()
        rolling_count = fd[col].rolling(window=window, min_periods=1).count()
        pct = (rolling_rank - 0.5) / rolling_count
        nqt_values = norm.ppf(pct)
        nqt_df[col] = np.clip(nqt_values, -3.0, 3.0)
    Z_all = nqt_df.values
    train_mask = fd['time'] <= HMM_TRAIN_END
    Z_tr = Z_all[train_mask]
    return fd, Z_tr, Z_all

# ... (5 tong lines con lai) ...
```

## PHAN 9: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 4. Điểm Thông Tin Tương Hỗ (Mutual Information - MI) & Lựa Chọn Đặc Trưng Tham Lam & Kiểm Soát VIF

Điểm MI đo lường mức độ phụ thuộc thông tin (kể cả phi tuyến) giữa đặc trưng đầu vào và trị tuyệt...
```

## PHAN 10: CODE (Thuc Hien)
**Dong so**: 73 dong
**Noi dung chinh (15 dong dau):**

```python
# Tạo Y_proxy rule-based cho Market
# Normalize duplicated merge columns if present
for std_col in ['vnindex_log_ret', 'vnindex_vol20']:
    if std_col not in df_market.columns:
        alt = next((c for c in [f'{std_col}_y', f'{std_col}_x'] if c in df_market.columns), None)
        if alt is None:
            raise KeyError(f"{std_col} column not found in df_market")
        df_market = df_market.rename(columns={alt: std_col})

df_market = df_market.loc[:, ~df_market.columns.duplicated()]
vol_median = df_market['vnindex_vol20'].median()
def label_proxy(row):
    ret = row['vnindex_log_ret']
    vol = row['vnindex_vol20']
    if ret > 0 and vol < vol_median: return 0 # Bull / Low Vol

# ... (58 tong lines con lai) ...
```

## PHAN 11: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 5. Chạy Grid Search Tìm Cấu Hình HMM Tốt Nhất (Kiến Trúc Tách Lớp)

Hệ thống giờ đây được đánh giá theo 2 tầng:
1. **Tầng Vĩ Mô (Macro HMM):** Chạy trên khung thời gian Tháng (Monthly) sử dụng các ...
```

## PHAN 12: CODE (Thuc Hien)
**Dong so**: 128 dong
**Noi dung chinh (15 dong dau):**

```python
import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
import warnings

def n_params(K, D, M=2):
    return (K - 1) + K * (K - 1) + K * (M - 1) + K * M * D + K * M * D * (D + 1) // 2

# =====================================================================
# 1. PREPARE & EVALUATE MACRO HMM (MONTHLY)
# =====================================================================
df_market['year_month'] = df_market['time'].dt.to_period('M')
df_monthly = df_market.groupby('year_month').last().reset_index()

train_mask_macro = df_monthly['time'] <= HMM_TRAIN_END

# ... (113 tong lines con lai) ...
```

## PHAN 13: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 6. Tự Động Ánh Xạ & Gán Nhãn Trạng Thái (K-agnostic Labeling)

Ánh xạ nhãn trạng thái tự động cho cả vĩ mô tháng và thị trường ngày bám sát theo kỳ vọng lợi nhuận kì vọng và biến động tài sản.

###...
```

## PHAN 14: CODE (Thuc Hien)
**Dong so**: 47 dong
**Noi dung chinh (15 dong dau):**

```python
df_market_res = df_market[['time']].copy()
df_market_res['market_regime'] = global_market_regimes_filtered
stats_market = []
for k in range(K_market):
    mask = df_market_res['market_regime'] == k
    ret_k = df_market.loc[mask, 'vnindex_log_ret'].mean() * 100 if mask.sum() > 0 else 0
    vol_k = df_market.loc[mask, 'vnindex_vol20'].mean() * 100 if mask.sum() > 0 else 0
    stats_market.append({'state': k, 'mean_ret_%': ret_k, 'vol_%': vol_k})
rs_market = pd.DataFrame(stats_market)
display(rs_market)

def auto_label(rs, K):
    ret = rs['mean_ret_%'].values
    vol = rs['vol_%'].values
    if K == 2:

# ... (32 tong lines con lai) ...
```

## PHAN 15: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 7. Chuẩn bị Dữ liệu Ngành & Huấn luyện Sector HMM (Grid Search K)
Huấn luyện HMM cho từng nhóm ngành độc lập. Kết quả sẽ được dùng làm feature cho Ticker HMM.
```

## PHAN 16: CODE (Thuc Hien)
**Dong so**: 149 dong
**Noi dung chinh (15 dong dau):**

```python
import pandas as pd
import numpy as np
from hmmlearn.hmm import GMMHMM

# Ánh xạ Ngành và tạo Dữ liệu Sector
_ind_df = pd.read_csv('../src/data_collection/industries.csv')
_ind_df = _ind_df[_ind_df['icb_level'] == 1]
industry_mapping = dict(zip(_ind_df['symbol'], _ind_df['icb_name']))

print("Đang tạo đặc trưng nhóm ngành (Sector Features)...")
df_m1['industry'] = df_m1['ticker'].map(industry_mapping)
sector_df = df_m1.groupby(['industry', 'time']).agg(
    sector_log_ret=('log_return', 'mean'),
    sector_volume=('volume', 'sum')
).reset_index()

# ... (134 tong lines con lai) ...
```

## PHAN 17: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 8. Huấn Luyện Ticker HMM Kết Hợp Vĩ Mô & Ngành
Sử dụng đặc trưng của Ticker kết hợp với xác suất (prob) của Market HMM và Sector HMM để huấn luyện mô hình riêng cho từng mã. Để tối ưu thời gian, K ...
```

## PHAN 18: CODE (Thuc Hien)
**Dong so**: 113 dong
**Noi dung chinh (15 dong dau):**

```python
def make_Z_ticker(df_source, features, window=252):
    fd = df_source[['time'] + features].dropna().reset_index(drop=True)
    nqt_df = pd.DataFrame(index=fd.index)
    for col in features:
        rolling_rank = fd[col].rolling(window=window, min_periods=1).rank()
        rolling_count = fd[col].rolling(window=window, min_periods=1).count()
        pct = (rolling_rank - 0.5) / rolling_count
        nqt_values = norm.ppf(pct)
        nqt_df[col] = np.clip(nqt_values, -3.0, 3.0)
    Z_all = nqt_df.values
    return fd, Z_all


tickers = df_m1['ticker'].unique()


# ... (98 tong lines con lai) ...
```

## PHAN 19: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 9. Meta-Classifier (LightGBM) Dự Báo Cổ Phiếu T+1
```

## PHAN 20: MARKDOWN (Giai Thich)
**Noi dung:**

```
### 9.1 Chế độ Backtest (Walk-Forward Validation)
**Mục đích:** Đánh giá hiệu suất của thuật toán trong quá khứ một cách trung thực nhất, loại bỏ hoàn toàn Look-ahead Bias.
**Cách hoạt động:** Dùng vò...
```

## PHAN 21: CODE (Thuc Hien)
**Dong so**: 62 dong
**Noi dung chinh (15 dong dau):**

```python

# 1. Tạo nhãn dự báo (Target): Dự báo return_1d (T+1)
master_ticker['target_return_1d'] = master_ticker.groupby('ticker')['close'].pct_change(1).shift(-1)
master_ticker['target_bin'] = (master_ticker['target_return_1d'] > 0).astype(int)

# Bỏ đi những dòng không có target (ngày cuối cùng của dữ liệu chưa biết tương lai)
# LƯU Ý: Với Live Trading, ta VẪN CẦN giữ lại ngày cuối cùng dù target là NaN để dự báo. 
# Nên ta sẽ copy ra một df riêng cho backtest.
df_backtest = master_ticker.dropna(subset=['target_return_1d']).reset_index(drop=True)

# Lọc các features

semantic_sector_probs = [col for col in df_backtest.columns if col.startswith('prob_sector_')]
market_probs = [col for col in df_backtest.columns if col.startswith('Market_Prob_')]
ticker_probs = [col for col in df_backtest.columns if col.startswith('prob_ticker_')]

# ... (47 tong lines con lai) ...
```

## PHAN 22: MARKDOWN (Giai Thich)
**Noi dung:**

```
### 9.2 Chế độ Live Trading (Dự báo ngày T+1)
**Mục đích:** Dùng để ra quyết định mua/bán hằng ngày.
**Cách hoạt động:** Chỉ lấy ngày giao dịch cuối cùng trong file dữ liệu làm tập Test, toàn bộ phần ...
```

## PHAN 23: CODE (Thuc Hien)
**Dong so**: 35 dong
**Noi dung chinh (15 dong dau):**

```python

import lightgbm as lgb

print(f"=== CHẾ ĐỘ LIVE TRADING ===")
# Xác định ngày cuối cùng có trong dữ liệu
latest_date = master_ticker['time'].max()
print(f"Ngày giao dịch mới nhất (T): {latest_date.strftime('%Y-%m-%d')}")

# Tập Train: Tất cả dữ liệu trước ngày T (phải loại bỏ các dòng NaN ở target)
train_mask = (master_ticker['time'] < latest_date) & (master_ticker['target_return_1d'].notna())
X_train_live = master_ticker.loc[train_mask, feature_cols]
y_train_live = master_ticker.loc[train_mask, 'target_bin']

# Tập Test: Duy nhất dữ liệu của ngày T
test_mask = master_ticker['time'] == latest_date

# ... (20 tong lines con lai) ...
```

## PHAN 24: MARKDOWN (Giai Thich)
**Noi dung:**

```
### 9.3 Thống kê Hiệu suất Tài chính (Financial Backtest)
**Mục đích:** Mô phỏng giao dịch thực tế trên tập Walk-Forward OOS. So sánh Lợi nhuận tích lũy (Equity Curve) và Sharpe Ratio của Chiến lược s...
```

## PHAN 25: CODE (Thuc Hien)
**Dong so**: 86 dong
**Noi dung chinh (15 dong dau):**

```python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if 'df_backtest' in locals() and 'final_meta_pred_prob' in df_backtest.columns:
    df_bt = df_backtest.dropna(subset=['final_meta_pred_prob']).copy()
    N_stocks = df_bt['ticker'].nunique()
    
    # 1. Chiến lược 1: Mua rải đều (AI Equal Weight)
    # Lệnh mua ăn target_return_1d, lệnh đứng ngoài ăn 0%
    df_bt['signal_eq'] = (df_bt['final_meta_pred_prob'] > 0.4).astype(int)
    port_eq = df_bt.groupby('time').apply(
        lambda x: (x['target_return_1d'] * x['signal_eq']).sum() / N_stocks
    ).reset_index(name='ret_eq')

# ... (71 tong lines con lai) ...
```

## PHAN 26: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 10. Lưu Kết Quả & Trực Quan Hóa Tương Tác
```

## PHAN 27: CODE (Thuc Hien)
**Dong so**: 142 dong
**Noi dung chinh (15 dong dau):**

```python
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Lưu Market Regimes
if 'global_vars' in globals():
    global_vars.to_parquet(OUTPUT_DIR / 'market_regimes.parquet', index=False)
    global_vars.to_csv(OUTPUT_DIR / 'market_regimes.csv', index=False)
    print(f"Đã lưu: market_regimes (.parquet & .csv) {global_vars.shape}")

# 2. Lưu Sector Regimes
if 'df_sector_hmm' in globals():
    df_sector_hmm.to_parquet(OUTPUT_DIR / 'sector_regimes.parquet', index=False)
    df_sector_hmm.to_csv(OUTPUT_DIR / 'sector_regimes.csv', index=False)
    print(f"Đã lưu: sector_regimes (.parquet & .csv){df_sector_hmm.shape}")

# 3. Lưu Ticker Regimes (Master File)

# ... (127 tong lines con lai) ...
```

## PHAN 28: MARKDOWN (Giai Thich)
**Noi dung:**

```
## 11. Hiển Thị Biểu Đồ Tĩnh (Mã BID)
```

## PHAN 29: CODE (Thuc Hien)
**Dong so**: 9 dong
**Noi dung chinh (15 dong dau):**

```python



ticker_to_plot = 'BID'
date_range = (df_plot['time'].min(), df_plot['time'].max())

print(f"Đang vẽ biểu đồ tĩnh cho {ticker_to_plot}...")
plot_ticker_regimes(ticker_to_plot, date_range)

```

## PHAN 30: CODE (Thuc Hien)
**Dong so**: 25 dong
**Noi dung chinh (15 dong dau):**

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test_all, probs_all)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# ... (10 tong lines con lai) ...
```
