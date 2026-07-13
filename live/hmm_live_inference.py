import pickle
import os
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
#!/usr/bin/env python
# coding: utf-8

# # Mô Hình Hóa Trạng Thái Thị Trường Đa Tần Số Phân Cấp Cho Từng Mã (Ticker-Specific Hierarchical Dual-Frequency HMM)
# 
# Notebook này thiết lập quy trình **HMM kép phân cấp** ở cấp độ **Từng mã cổ phiếu riêng biệt (Ticker-Specific)** bằng phương pháp **Ticker-Specific Inference từ Global HMM** 
# 1. **Tầng vĩ mô dài hạn (Monthly Macro HMM):** Huấn luyện trên lưới thời gian Tháng thực tế (116 mẫu) từ các chỉ số kinh tế lớn (`cpi_mom`, `credit_growth_mom`, `pmi_vn`).
# 2. **Tầng thị trường ngắn hạn (Daily Market HMM):** Huấn luyện trên Market Proxy (chỉ số đại diện thị trường) với các biến số tối ưu từ Grid Search.
# 3. **Suy luận trạng thái riêng cho từng mã (Ticker-Specific Inference):**
#    * Sử dụng các tham số tối ưu đã huấn luyện của Daily Market HMM.
#    * Đối với từng mã cổ phiếu, tính toán đặc trưng Z-score ngắn hạn dựa trên biến động của chính mã đó (`rolling_vol_5_ticker`) kết hợp các đặc trưng vĩ mô/thị trường.
#    * Suy luận chuỗi trạng thái ẩn (`market_regime`) riêng biệt cho mã đó.
# 4. **Tầng Hợp Nhất (State Fusion Layer):** Hợp nhất xác suất ẩn đa tần số thông qua phép ghép nối lùi `pd.merge_asof` không rò rỉ tương lai.
# 5. **Trực quan hóa tương tác (Interactive Plotting):** Sử dụng các thanh trượt thời gian (SelectionRangeSlider) để đo đạc biến động giá và khối lượng của từng mã tương ứng với các trạng thái ẩn HMM.
# 

# In[14]:


import os
import numpy as np
import pandas as pd
import warnings
import pickle
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import skew, kurtosis, norm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from hmmlearn.hmm import GMMHMM, GaussianHMM

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Thư mục đầu ra được thiết lập tại: {OUTPUT_DIR.resolve()}")


# ## 1. Tải Dữ Liệu & Tạo Chỉ Báo Thị Trường Đại Diện (VN-Index Proxy)
# 
# Chúng ta nạp bộ dữ liệu cơ sở `hmm_data.csv` và thực hiện tích hợp các đặc trưng vi mô của thị trường chứng khoán Việt Nam (được tổng hợp từ lưới mã VN46 trong `m1_vn46.csv`).

# In[15]:


print("Đang tải dữ liệu hmm_data.csv...")
df_daily_base = pd.read_csv('output/hmm_data.csv')
df_daily_base['time'] = pd.to_datetime(df_daily_base['time'])

print("Đang tích hợp chỉ số VN-Index, tỷ giá và dòng tiền khối ngoại...")
df_m1 = pd.read_csv('data/processed/m1_vn46.csv')
df_m1['time'] = pd.to_datetime(df_m1['time']).dt.normalize()
market_ret = df_m1.groupby('time')['log_return'].mean().reset_index()
market_ret.columns = ['time', 'vnindex_log_ret']
market_close = df_m1.groupby('time')['close'].mean().reset_index()
market_close.columns = ['time', 'vnindex_close']

df = df_daily_base.merge(market_ret, on='time', how='left')
df = df.merge(market_close, on='time', how='left')
df = df.dropna(subset=['vnindex_log_ret', 'vnindex_close']).reset_index(drop=True)
df['vnindex_vol20'] = df['vnindex_log_ret'].rolling(20).std() * np.sqrt(252)

df_fnb = pd.read_csv('data/processed/m4_foreign_net_buy_sell.csv')
df_fnb['time'] = pd.to_datetime(df_fnb['time'])
df = df.merge(df_fnb[['time', 'fnb_ratio']], on='time', how='left')

df_fx = pd.read_csv('data/processed/e1_usdvnd.csv')
df_fx['time'] = pd.to_datetime(df_fx['time'])
df = df.merge(df_fx[['time', 'fx_log_ret']], on='time', how='left')
df = df.dropna().reset_index(drop=True)
print(f"Kích thước bảng dữ liệu gốc: {df.shape}")
df.head(5)


print("Đang trích xuất lưới dữ liệu vĩ mô tần suất Tháng...")
df_monthly = df[['time', 'cpi_mom', 'credit_growth_mom', 'pmi_vn']].copy()
df_monthly['year_month'] = df_monthly['time'].dt.to_period('M')
df_monthly = df_monthly.groupby('year_month').first().reset_index(drop=True)

macro_features = ['cpi_mom', 'credit_growth_mom', 'pmi_vn']
df_monthly[macro_features] = df_monthly[macro_features].shift(1)
df_monthly = df_monthly.dropna().reset_index(drop=True)

# --- KẾT QUẢ GRID SEARCH ĐƯỢC HARDCODE ĐỂ CHẠY LIVE ---
HMM_TRAIN_END = pd.Timestamp('2019-12-31')
K_MACRO_BEST = 2
K_DAILY_BEST = 4
DAILY_FEATURES_BEST = ['rolling_vol_5', 'fx_log_ret', 'ret_disp', 'amihud_diff_normalized']

def make_Z(features, window=252):
    fd = df[['time'] + features].dropna().reset_index(drop=True)
    nqt_df = pd.DataFrame(index=fd.index)
    for col in features:
        rolling_rank = fd[col].rolling(window=window, min_periods=1).rank()
        rolling_count = fd[col].rolling(window=window, min_periods=1).count()
        pct = (rolling_rank - 0.5) / rolling_count
        nqt_values = norm.ppf(pct)
        nqt_df[col] = np.clip(nqt_values, -3.0, 3.0)
    Z_all = nqt_df.values
    return fd, Z_all, Z_all # Trả về Z_all 2 lần để tương thích biến Z_tr, Z_all = Z_all, Z_all

# ## 8. Huấn Luyện Lại Mô Hình HMM Tối Ưu Cuối Cùng (Refit Final Model)
# 
# Refit lại mô hình Monthly và Daily chung của thị trường.

# In[22]:


model_macro = GaussianHMM(n_components=K_MACRO_BEST, covariance_type='full', random_state=RANDOM_STATE, n_iter=200)
# Only train on train set, then predict on all (V2)
Z_macro_all = df_monthly[macro_features].values
macro_train_mask = df_monthly['time'] <= HMM_TRAIN_END
Z_macro_train = Z_macro_all[macro_train_mask]
macro_model_path = os.path.join(MODEL_DIR, 'macro_model.pkl')
if os.path.exists(macro_model_path):
    print("Loading pre-trained Macro HMM...")
    with open(macro_model_path, 'rb') as f:
        model_macro = pickle.load(f)
else:
    model_macro.fit(Z_macro_train)
    with open(macro_model_path, 'wb') as f:
        pickle.dump(model_macro, f)
macro_states = model_macro.predict(Z_macro_all)
macro_probs = model_macro.predict_proba(Z_macro_all)
print(f"Monthly Macro HMM hội tụ: {model_macro.monitor_.converged}")

fd_z, Z_tr, Z_all = make_Z(DAILY_FEATURES_BEST, window=252)
daily_model_path = os.path.join(MODEL_DIR, 'daily_market_model.pkl')
if os.path.exists(daily_model_path):
    print("Loading pre-trained Daily Market HMM...")
    with open(daily_model_path, 'rb') as f:
        model_daily = pickle.load(f)
else:
    model_daily, ll_daily, _ = fit_hmm(Z_tr, Z_all[len(Z_tr):], K_DAILY_BEST, n_seeds=10)
    with open(daily_model_path, 'wb') as f:
        pickle.dump(model_daily, f)
daily_states = model_daily.predict(Z_all)
daily_probs = model_daily.predict_proba(Z_all)
print(f"Daily Market HMM hội tụ: {model_daily.monitor_.converged}")


# ## 9. Tự Động Ánh Xạ & Gán Nhãn Trạng Thái (K-agnostic Labeling)
# 
# Ánh xạ nhãn trạng thái tự động cho cả vĩ mô tháng và thị trường ngày bám sát theo kỳ vọng lợi nhuận kì vọng và biến động tài sản.
# 
# ### Labling Monthly
# | Trạng thái (Status) | Ý nghĩa (Meaning) | Ngưỡng chọn / Ràng buộc (Constraint) |
# |---------------------|-------------------|--------------------------------------|
# | **Macro_Stagnant** (K=2,3) | Giai đoạn vĩ mô trì trệ, sản xuất suy giảm hoặc tăng trưởng chậm. | `pmi_vn` thấp nhất |
# | **Macro_Stable** (K=3) | Giai đoạn vĩ mô ổn định, tăng trưởng sản xuất vừa phải. | `pmi_vn` ở mức trung vị / trung bình |
# | **Macro_Expansion** (K=2,3) | Giai đoạn vĩ mô mở rộng, sản xuất tăng trưởng mạnh mẽ. | `pmi_vn` cao nhất |
# 
# ### Labling Daily
# K = 3
# | Trạng thái (Status) | Ý nghĩa (Meaning) | Ngưỡng chọn / Ràng buộc (Constraint) |
# |---------------------|-------------------|--------------------------------------|
# | **Bull** | Thị trường tăng trưởng, xu hướng đi lên và ít biến động (rủi ro thấp). | `ret` không thấp nhất và `vol` thấp nhất trong 2 trạng thái còn lại |
# | **Sideways** | Thị trường đi ngang, dao động tích lũy trong biên độ. | Trạng thái còn lại sau khi đã xác định Bull và Bear |
# | **Bear** | Thị trường suy thoái, xu hướng giảm mạnh và rủi ro cao. | `ret` thấp nhất (argmin) |
# 
# K = 4
# | Trạng thái (Status) | Ý nghĩa (Meaning) | Ngưỡng chọn / Ràng buộc (Constraint) |
# |---------------------|-------------------|--------------------------------------|
# | **Crisis** | Thị trường suy thoái, biến động mạnh, rủi ro cao. | `ret < 0` và `vol >= median` |
# | **CalmBull** | Pha tăng trưởng bền vững, ổn định, ít rủi ro. | `ret > 0` và `vol < median` |
# | **Euphoria** | Thị trường hưng phấn, tăng mạnh kèm dao động lớn. | `ret >> 0` và `vol >= median` |
# | **Tranquil** | Thị trường "lặng sóng", đi ngang, ảm đạm. | `ret ≈ 0` và `vol < median` |
# 

# In[23]:


stats_macro = []
for k in range(K_MACRO_BEST):
    mask = macro_states == k
    stats_macro.append({
        'state': k, 'n_months': int(mask.sum()),
        'mean_pmi': df_monthly.loc[mask, 'pmi_vn'].mean(),
        'mean_cpi': df_monthly.loc[mask, 'cpi_mom'].mean()
    })
df_sm = pd.DataFrame(stats_macro)

def auto_label_macro(rs_macro, K):
    pmi = rs_macro['mean_pmi'].values
    order = np.argsort(pmi)
    if K == 2:
        return {int(order[0]): 'Macro_Stagnant', int(order[1]): 'Macro_Expansion'}
    elif K == 3:
        return {int(order[0]): 'Macro_Stagnant', int(order[1]): 'Macro_Stable', int(order[2]): 'Macro_Expansion'}
    return {int(order[i]): f'Macro_Tier{i+1}' for i in range(K)}

STATE_TO_LABEL_MACRO = auto_label_macro(df_sm, K_MACRO_BEST)
print("--- Bản đồ ánh xạ Trạng thái Vĩ mô -> Nhãn ---")
for s, l in sorted(STATE_TO_LABEL_MACRO.items()):
    r = df_sm.loc[s]
    print(f"state {s} -> {l:<16} PMI={r['mean_pmi']:.3f}, CPI={r['mean_cpi']:.3f}")

fd_z['regime'] = daily_states
stats_daily = []
for k in range(K_DAILY_BEST):
    mask = fd_z['regime'] == k
    stats_daily.append({
        'state': k, 'n_days': int(mask.sum()),
        'mean_ret_%': df.loc[mask, 'vnindex_log_ret'].mean()*100,
        'vol_%': df.loc[mask, 'vnindex_vol20'].mean()*100
    })
rs_daily = pd.DataFrame(stats_daily)

def auto_label(rs, K):
    ret = rs['mean_ret_%'].values
    vol = rs['vol_%'].values
    if K == 2:
        return {int(np.argmin(ret)): 'Bear', int(np.argmax(ret)): 'Bull'}
    elif K == 3:
        order = np.argsort(ret)
        return {int(order[0]): 'Bear', int(order[1]): 'Sideways', int(order[2]): 'Bull'}
    elif K == 4:
        vol_med = np.median(vol)
        labels = {}
        for k in range(K):
            up = ret[k] >= 0
            calm = vol[k] < vol_med
            if up and calm: labels[k] = 'CalmBull'
            elif up and not calm: labels[k] = 'Euphoria'
            elif (not up) and calm: labels[k] = 'TranquilBull'
            else: labels[k] = 'Crisis'

        from collections import Counter
        cnt = Counter(labels.values())
        if any(v > 1 for v in cnt.values()):
            for lbl_dup in [l for l,c in cnt.items() if c > 1]:
                dup_states = [k for k,l in labels.items() if l == lbl_dup]
                dup_states_sorted = sorted(dup_states, key=lambda s: vol[s])
                for s in dup_states_sorted[1:]:
                    labels[s] = 'Tranquil' if lbl_dup != 'CalmBull' else 'TranquilBull'
        return labels
    return {i: f'State_{i}' for i in range(K)}

STATE_TO_LABEL_DAILY = auto_label(rs_daily, K_DAILY_BEST)
print("\n--- Bản đồ ánh xạ Trạng thái Ngày -> Nhãn ---")
for s, l in sorted(STATE_TO_LABEL_DAILY.items()):
    r = rs_daily.loc[s]
    print(f"state {s} -> {l:<12} ret={r['mean_ret_%']:+.3f}%/d, vol={r['vol_%']:5.2f}%")


# ## 10. Suy Luận Trạng Thái Ẩn Cho Từng Mã Cổ Phiếu (Ticker-Specific Inference)
# 
# Tiến hành áp dụng mô hình Daily HMM chung đã được huấn luyện lên chuỗi dữ liệu đặc trưng riêng của từng mã cổ phiếu để tạo ra regimes riêng biệt cho mã đó.

# In[24]:


def make_Z_ticker(df_source, features, window=252):
    # Bước 2: Chuẩn hóa theo "Ký ức của riêng Ticker" (Expanding NQT)
    fd = df_source[['time'] + features].dropna().reset_index(drop=True)
    nqt_df = pd.DataFrame(index=fd.index)
    for col in features:
        # Dùng expanding().rank() để xếp hạng giá trị hiện tại so với TẤT CẢ lịch sử trước đó (từ 1 đến t)
        # Khắc phục triệt để lỗi mất trí nhớ dài hạn (Absolute Scale Loss) của Rolling Window
        rolling_rank = fd[col].expanding(min_periods=1).rank()
        rolling_count = fd[col].expanding(min_periods=1).count()
        pct = (rolling_rank - 0.5) / rolling_count
        # Ánh xạ bách phân vị thành Z-score của phân phối chuẩn
        nqt_values = norm.ppf(pct)
        nqt_df[col] = np.clip(nqt_values, -3.0, 3.0)
    Z_all = nqt_df.values
    return fd, Z_all

tickers = df_m1['ticker'].unique()
print(f'Bắt đầu xử lý suy luận cho {len(tickers)} mã...')

global_vars = df[['time', 'fx_log_ret', 'ret_disp', 'amihud_diff_normalized', 'fnb_ratio',
                  'vnindex_log_ret', 'vnindex_close', 'vnindex_vol20',
                  'credit_growth_mom', 'cpi_mom', 'pmi_vn']].copy()

df_macro_res = pd.DataFrame({
    'time': df_monthly['time'].values,
    'macro_regime': macro_states,
    'macro_regime_label': pd.Series(macro_states).map(STATE_TO_LABEL_MACRO).values,
})
for k in range(K_MACRO_BEST):
    df_macro_res[f'prob_macro_{k}'] = macro_probs[:, k]
df_macro_res = df_macro_res.sort_values('time').reset_index(drop=True)

ticker_dfs = []
for i, ticker in enumerate(tickers):
    # Bước 1: Cách ly dữ liệu và Trích xuất Động lượng (Momentum) cho Cổ phiếu này
    df_tick = df_m1[df_m1['ticker'] == ticker].copy().sort_values('time').reset_index(drop=True)
    df_tick['rolling_vol_5'] = df_tick['log_return'].rolling(5).std() * np.sqrt(252)

    # Tính lợi suất 20 ngày (Động lượng ngắn hạn)
    df_tick['mom_1M'] = df_tick['close'].pct_change(20)
    # Tính khoảng cách đến MA50 (Động lượng trung hạn)
    df_tick['dist_MA50'] = df_tick['close'] / df_tick['close'].rolling(50).mean() - 1

    ticker_aligned = global_vars.merge(
        df_tick[['time', 'open', 'high', 'low', 'close', 'volume', 'log_return', 
                 'rolling_vol_20d', 'return_5d', 'return_20d', 'volume_ratio', 'rolling_vol_5', 'mom_1M', 'dist_MA50']],
        on='time', how='left'
    )

    # Ép HMM phải nhận diện xu hướng bằng cách truyền mom_1M và dist_MA50
    DAILY_FEATURES_V3 = ['rolling_vol_5', 'fx_log_ret', 'amihud_diff_normalized', 'mom_1M', 'dist_MA50']
    fd_z_tick, Z_all_tick = make_Z_ticker(ticker_aligned, DAILY_FEATURES_V3, window=252)

    # --- V3 UPGRADE: Bước 3 - Huấn luyện HMM Độc lập (Private HMM Training) ---
    # Dùng covariance_type='diag' (Ma trận đường chéo) để ngăn lỗi sụp đổ ma trận do dữ liệu của 1 mã quá mỏng
    model_ticker = GMMHMM(n_components=K_DAILY_BEST, n_mix=2, covariance_type='diag', min_covar=0.01, random_state=RANDOM_STATE, n_iter=200)
    ticker_train_mask = fd_z_tick['time'] <= HMM_TRAIN_END
    Z_train_tick = Z_all_tick[ticker_train_mask]

    # Tránh lỗi những mã lên sàn quá muộn không đủ dữ liệu train
    ticker_model_path = os.path.join(MODEL_DIR, f'ticker_{ticker}.pkl')
    if os.path.exists(ticker_model_path):
        with open(ticker_model_path, 'rb') as f:
            model_ticker = pickle.load(f)
    else:
        if len(Z_train_tick) > 100:
            model_ticker.fit(Z_train_tick)
            with open(ticker_model_path, 'wb') as f:
                pickle.dump(model_ticker, f)
        else:
            model_ticker = model_daily # Fallback về model thị trường chung

    ticker_daily_states = model_ticker.predict(Z_all_tick)
    ticker_daily_probs = model_ticker.predict_proba(Z_all_tick)

    # Bước 4: Tự động Gắn nhãn Không Cảm Tính (Dynamic Auto-Labeling)
    df_temp_label = pd.merge(fd_z_tick[['time']], ticker_aligned[['time', 'log_return']].drop_duplicates(subset=['time']), on='time', how='left')
    df_temp_label['market_regime'] = ticker_daily_states
    df_temp_label = df_temp_label.dropna()

    stats_daily_ticker = []
    for k in range(K_DAILY_BEST):
        mask = df_temp_label['market_regime'] == k
        if mask.sum() > 0:
            ret_k = df_temp_label.loc[mask, 'log_return'].mean() * 100
            vol_k = df_temp_label.loc[mask, 'log_return'].std() * np.sqrt(252) * 100
        else:
            ret_k, vol_k = 0, 0
        stats_daily_ticker.append({'state': k, 'mean_ret_%': ret_k, 'vol_%': vol_k})
    rs_daily_ticker = pd.DataFrame(stats_daily_ticker)
    STATE_TO_LABEL_DAILY_TICKER = auto_label(rs_daily_ticker, K_DAILY_BEST)

    # V3_OP1: KHÔNG DÙNG BỘ LỌC - Suy luận 100% theo trạng thái từng ngày
    ticker_daily_states_filtered = ticker_daily_states.copy()

    ticker_daily_labels_filtered = pd.Series(ticker_daily_states_filtered).map(STATE_TO_LABEL_DAILY_TICKER).values

    df_tick_daily_res = pd.DataFrame({
        'time': fd_z_tick['time'].values,
        'market_regime': ticker_daily_states_filtered,
        'market_regime_label': ticker_daily_labels_filtered,
    })
    for k in range(K_DAILY_BEST):
        df_tick_daily_res[f'prob_market_{k}'] = ticker_daily_probs[:, k]
    df_tick_daily_res = df_tick_daily_res.sort_values('time').reset_index(drop=True)

    # Ghép HMM ngắn hạn (Ngày) với HMM dài hạn (Tháng)
    merged_states_tick = pd.merge_asof(df_tick_daily_res, df_macro_res, on='time', direction='backward')
    merged_states_tick = merged_states_tick.ffill().bfill().fillna(0.0)

    state_cols = ['market_regime', 'market_regime_label', 'macro_regime', 'macro_regime_label'] + [f'prob_market_{k}' for k in range(K_DAILY_BEST)] + [f'prob_macro_{k}' for k in range(K_MACRO_BEST)]
    ticker_master = ticker_aligned.merge(merged_states_tick[['time'] + state_cols], on='time', how='left')
    ticker_master['ticker'] = ticker

    # Tạo nhãn ghép (Ví dụ: Macro_Expansion_Bull) làm Context cho DRL
    ticker_master['joint_regime_label'] = ticker_master['macro_regime_label'] + '_' + ticker_master['market_regime_label']
    ticker_dfs.append(ticker_master)

master_ticker = pd.concat(ticker_dfs, ignore_index=True)
master_ticker = master_ticker.dropna(subset=['close']).reset_index(drop=True)
state_cols = state_cols + ['joint_regime_label']
cols_reordered = ['time', 'ticker'] + [col for col in master_ticker.columns if col not in ['time', 'ticker']]
master_ticker = master_ticker[cols_reordered]
print(f'Hoàn thành suy luận. Kích thước master_ticker: {master_ticker.shape}')


# ## 11. Lưu Kết Quả Đầu Ra & Chia Tập Dữ Liệu Splits
# 
# Lưu master_drl_ready_ticker.parquet và chia các tập Train/Val/Test.

# In[25]:


master_ticker = master_ticker.drop_duplicates(subset=['time', 'ticker'], keep='last')
master_ticker.to_parquet(OUTPUT_DIR / 'master_drl_ready_ticker.parquet', index=False)
hmm_regimes_merged_ticker = master_ticker[['time', 'ticker'] + state_cols].copy()
hmm_regimes_merged_ticker.to_csv(OUTPUT_DIR / 'hmm_regimes_merged_ticker.csv', index=False)
print("Đã lưu các file master của từng mã.")

split_dir = OUTPUT_DIR / 'splits_ticker'
split_dir.mkdir(parents=True, exist_ok=True)

df_sorted = master_ticker.sort_values('time').reset_index(drop=True)
train_end = '2019-12-31'
val_end = '2022-12-31'

df_train = df_sorted[df_sorted['time'] <= train_end].reset_index(drop=True)
df_val = df_sorted[(df_sorted['time'] > train_end) & (df_sorted['time'] <= val_end)].reset_index(drop=True)
df_test = df_sorted[df_sorted['time'] > val_end].reset_index(drop=True)

df_train.to_parquet(split_dir / 'train_set.parquet', index=False)
df_val.to_parquet(split_dir / 'val_set.parquet', index=False)
df_test.to_parquet(split_dir / 'test_set.parquet', index=False)
print("Đã phân chia các tập dữ liệu Splits cho từng mã thành công!")


# ## 12. Trực Quan Hóa Tương Tác Regimes Từng Mã (Interactive Ticker Regime Visualization)
# 
# Sử dụng widget để chọn mã cổ phiếu và thanh khoảng (Range Slider) để đo đạc biến động giá cũng như các trạng thái ẩn HMM tương ứng.

# In[26]:


df_ticker = pd.read_parquet(OUTPUT_DIR / 'master_drl_ready_ticker.parquet')
df_ticker['time'] = pd.to_datetime(df_ticker['time'])

# Map unique labels to colors dynamically based on actual values in data (V2)
unique_labels = sorted(df_ticker['market_regime_label'].dropna().unique())
color_pool = ['#bbdefb', '#c8e6c9', '#fff9c4', '#ffcdd2', '#e0e0e0', '#d1c4e9', '#ffe0b2']
predefined = {
    'Bull': '#c8e6c9',          # Light Green
    'Bear': '#ffcdd2',          # Light Red
    'Sideways': '#e0e0e0',      # Light Grey
    'CalmBull': '#c8e6c9',      # Light Green
    'Euphoria': '#fff9c4',      # Light Yellow
    'Crisis': '#ffcdd2',        # Light Red
    'Tranquil': '#bbdefb',      # Light Grey
    'TranquilBull': '#bbdefb',      # Light Grey
    'Daily_Tier1': '#bbdefb',   # Light Blue
    'Daily_Tier2': '#c8e6c9',   # Light Green
    'Daily_Tier3': '#fff9c4',   # Light Yellow
    'Daily_Tier4': '#ffcdd2',   # Light Red
    'Daily_Tier5': '#e0e0e0',   # Light Grey
}



