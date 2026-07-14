# Hybrid Dual-Frequency Ticker-Specific HMM - Chi Tiet Toan Vo Quy Trinh

## Tong Quan Chung

- Tong so phan (Cells): 30
  - Markdown sections: 15
  - Code sections: 15

## Chi Tiet Tung Phan

### Phan 1: Markdown Section

```markdown
# Hybrid Dual-Frequency Ticker-Specific HMM

Quy trình kết hợp đánh giá Vĩ mô (Monthly) và Thị trường chung (Daily) để định vị pha biến động, sau đó ép cấu trúc thị trường chung lên từng mã cổ phiếu (Ticker-Specific) và dùng Meta-Classifier dự đoán lợi suất ngày tới.
```

### Phan 2: Code Section

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
... (41 tong lines)
```

### Phan 3: Markdown Section

```markdown
## 1. Tải Dữ Liệu & Tạo Chỉ Báo Thị Trường Đại Diện (VN-Index Proxy)
```

### Phan 4: Code Section

```python
print("Đang tải dữ liệu hmm_data.csv và m1_vn46.csv...")
df_daily_base = pd.read_csv('../output/hmm_data.csv')
df_daily_base['time'] = pd.to_datetime(df_daily_base['time'])

df_m1 = pd.read_csv('../data/processed/m1_vn46.csv')
df_m1['time'] = pd.to_datetime(df_m1['time']).dt.normalize()

# Tạo Market Proxy từ rổ VN46
market_ret = df_m1.groupby('time')['log_return'].mean().reset_index()
market_ret.columns = ['time', 'vnindex_log_ret']
... (34 tong lines)
```

### Phan 5: Markdown Section

```markdown
## 2. Bộ Lọc Kiểm Định Kỹ Thuật (Stationarity & Kurtosis)

Mục đích của bước này là đánh giá tính chất toán học của các đặc trưng để chọn lọc đầu vào ổn định cho HMM:
1. **Kiểm định tính dừng (Stationarity):** Sử dụng cả ADF (yêu cầu bác bỏ giả thuyết không, $p < 0.05$) và KPSS (yêu cầu chấp nhận gi...
```

### Phan 6: Code Section

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

... (27 tong lines)
```

### Phan 7: Markdown Section

```markdown
## 3. Thiết Lập Tập Dữ Liệu Train/OOS & Hàm Hỗ Trợ HMM sử dụng NQT + Rank
```

### Phan 8: Code Section

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
... (20 tong lines)
```

### Phan 9: Markdown Section

```markdown
## 4. Điểm Thông Tin Tương Hỗ (Mutual Information - MI) & Lựa Chọn Đặc Trưng Tham Lam & Kiểm Soát VIF

Điểm MI đo lường mức độ phụ thuộc thông tin (kể cả phi tuyến) giữa đặc trưng đầu vào và trị tuyệt đối lợi suất thị trường `|vnindex_log_ret|` (đại diện cho trạng thái biến động). Điểm số MI cao chỉ...
```

### Phan 10: Code Section

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
... (73 tong lines)
```

### Phan 11: Markdown Section

```markdown
## 5. Chạy Grid Search Tìm Cấu Hình HMM Tốt Nhất (Kiến Trúc Tách Lớp)

Hệ thống giờ đây được đánh giá theo 2 tầng:
1. **Tầng Vĩ Mô (Macro HMM):** Chạy trên khung thời gian Tháng (Monthly) sử dụng các biến Vĩ mô để xác định bối cảnh tổng thể. Dữ liệu được tịnh tiến (shift 1 tháng) để xử lý hoàn toàn ...
```

### Phan 12: Code Section

```python
import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
import warnings

def n_params(K, D, M=2):
    return (K - 1) + K * (K - 1) + K * (M - 1) + K * M * D + K * M * D * (D + 1) // 2

# =====================================================================
# 1. PREPARE & EVALUATE MACRO HMM (MONTHLY)
... (128 tong lines)
```

### Phan 13: Markdown Section

```markdown
## 6. Tự Động Ánh Xạ & Gán Nhãn Trạng Thái (K-agnostic Labeling)

Ánh xạ nhãn trạng thái tự động cho cả vĩ mô tháng và thị trường ngày bám sát theo kỳ vọng lợi nhuận kì vọng và biến động tài sản.

### Labling Monthly
| Trạng thái (Status) | Ý nghĩa (Meaning) | Ngưỡng chọn / Ràng buộc (Constraint) |
...
```

### Phan 14: Code Section

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
... (47 tong lines)
```

### Phan 15: Markdown Section

```markdown
## 7. Chuẩn bị Dữ liệu Ngành & Huấn luyện Sector HMM (Grid Search K)
Huấn luyện HMM cho từng nhóm ngành độc lập. Kết quả sẽ được dùng làm feature cho Ticker HMM.
```

### Phan 16: Code Section

```python
import pandas as pd
import numpy as np
from hmmlearn.hmm import GMMHMM

# Ánh xạ Ngành và tạo Dữ liệu Sector
_ind_df = pd.read_csv('../src/data_collection/industries.csv')
_ind_df = _ind_df[_ind_df['icb_level'] == 1]
industry_mapping = dict(zip(_ind_df['symbol'], _ind_df['icb_name']))

print("Đang tạo đặc trưng nhóm ngành (Sector Features)...")
... (149 tong lines)
```

### Phan 17: Markdown Section

```markdown
## 8. Huấn Luyện Ticker HMM Kết Hợp Vĩ Mô & Ngành
Sử dụng đặc trưng của Ticker kết hợp với xác suất (prob) của Market HMM và Sector HMM để huấn luyện mô hình riêng cho từng mã. Để tối ưu thời gian, K được cố định = 3.
```

### Phan 18: Code Section

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
... (113 tong lines)
```

### Phan 19: Markdown Section

```markdown
## 9. Meta-Classifier (LightGBM) Dự Báo Cổ Phiếu T+1
```

### Phan 20: Markdown Section

```markdown
### 9.1 Chế độ Backtest (Walk-Forward Validation)
**Mục đích:** Đánh giá hiệu suất của thuật toán trong quá khứ một cách trung thực nhất, loại bỏ hoàn toàn Look-ahead Bias.
**Cách hoạt động:** Dùng vòng lặp quét qua từng ngày, train bằng quá khứ và dự đoán ngày hiện tại. Tốn thời gian chạy (chỉ nên ...
```

### Phan 21: Code Section

```python

# 1. Tạo nhãn dự báo (Target): Dự báo return_1d (T+1)
master_ticker['target_return_1d'] = master_ticker.groupby('ticker')['close'].pct_change(1).shift(-1)
master_ticker['target_bin'] = (master_ticker['target_return_1d'] > 0).astype(int)

# Bỏ đi những dòng không có target (ngày cuối cùng của dữ liệu chưa biết tương lai)
# LƯU Ý: Với Live Trading, ta VẪN CẦN giữ lại ngày cuối cùng dù target là NaN để dự báo. 
# Nên ta sẽ copy ra một df riêng cho backtest.
df_backtest = master_ticker.dropna(subset=['target_return_1d']).reset_index(drop=True)

... (62 tong lines)
```

### Phan 22: Markdown Section

```markdown
### 9.2 Chế độ Live Trading (Dự báo ngày T+1)
**Mục đích:** Dùng để ra quyết định mua/bán hằng ngày.
**Cách hoạt động:** Chỉ lấy ngày giao dịch cuối cùng trong file dữ liệu làm tập Test, toàn bộ phần lịch sử làm Train. Chạy cực nhanh (1 giây). Bỏ qua hoàn toàn dữ liệu mục tiêu (target) của ngày cuối...
```

### Phan 23: Code Section

```python

import lightgbm as lgb

print(f"=== CHẾ ĐỘ LIVE TRADING ===")
# Xác định ngày cuối cùng có trong dữ liệu
latest_date = master_ticker['time'].max()
print(f"Ngày giao dịch mới nhất (T): {latest_date.strftime('%Y-%m-%d')}")

# Tập Train: Tất cả dữ liệu trước ngày T (phải loại bỏ các dòng NaN ở target)
train_mask = (master_ticker['time'] < latest_date) & (master_ticker['target_return_1d'].notna())
... (35 tong lines)
```

### Phan 24: Markdown Section

```markdown
### 9.3 Thống kê Hiệu suất Tài chính (Financial Backtest)
**Mục đích:** Mô phỏng giao dịch thực tế trên tập Walk-Forward OOS. So sánh Lợi nhuận tích lũy (Equity Curve) và Sharpe Ratio của Chiến lược so với việc Mua & Nắm giữ toàn bộ rổ cổ phiếu.
**Quy tắc giả lập:** 
- Vốn được chia đều cho tất cả c...
```

### Phan 25: Code Section

```python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if 'df_backtest' in locals() and 'final_meta_pred_prob' in df_backtest.columns:
    df_bt = df_backtest.dropna(subset=['final_meta_pred_prob']).copy()
    N_stocks = df_bt['ticker'].nunique()
    
    # 1. Chiến lược 1: Mua rải đều (AI Equal Weight)
... (86 tong lines)
```

### Phan 26: Markdown Section

```markdown
## 10. Lưu Kết Quả & Trực Quan Hóa Tương Tác
```

### Phan 27: Code Section

```python
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Lưu Market Regimes
if 'global_vars' in globals():
    global_vars.to_parquet(OUTPUT_DIR / 'market_regimes.parquet', index=False)
    global_vars.to_csv(OUTPUT_DIR / 'market_regimes.csv', index=False)
    print(f"Đã lưu: market_regimes (.parquet & .csv) {global_vars.shape}")

# 2. Lưu Sector Regimes
if 'df_sector_hmm' in globals():
... (142 tong lines)
```

### Phan 28: Markdown Section

```markdown
## 11. Hiển Thị Biểu Đồ Tĩnh (Mã BID)
```

### Phan 29: Code Section

```python



ticker_to_plot = 'BID'
date_range = (df_plot['time'].min(), df_plot['time'].max())

print(f"Đang vẽ biểu đồ tĩnh cho {ticker_to_plot}...")
plot_ticker_regimes(ticker_to_plot, date_range)

```

### Phan 30: Code Section

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test_all, probs_all)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
... (25 tong lines)
```
