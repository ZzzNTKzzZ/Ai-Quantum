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

OUTPUT_DIR = Path('../output/hmm_v3_op1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Thư mục đầu ra được thiết lập tại: {OUTPUT_DIR.resolve()}")


# ## 1. Tải Dữ Liệu & Tạo Chỉ Báo Thị Trường Đại Diện (VN-Index Proxy)
# 
# Chúng ta nạp bộ dữ liệu cơ sở `hmm_data.csv` và thực hiện tích hợp các đặc trưng vi mô của thị trường chứng khoán Việt Nam (được tổng hợp từ lưới mã VN46 trong `m1_vn46.csv`).

# In[15]:


print("Đang tải dữ liệu hmm_data.csv...")
df_daily_base = pd.read_csv('../output/hmm_data.csv')
df_daily_base['time'] = pd.to_datetime(df_daily_base['time'])

print("Đang tích hợp chỉ số VN-Index, tỷ giá và dòng tiền khối ngoại...")
df_m1 = pd.read_csv('../data/processed/m1_vn46.csv')
df_m1['time'] = pd.to_datetime(df_m1['time']).dt.normalize()
market_ret = df_m1.groupby('time')['log_return'].mean().reset_index()
market_ret.columns = ['time', 'vnindex_log_ret']
market_close = df_m1.groupby('time')['close'].mean().reset_index()
market_close.columns = ['time', 'vnindex_close']

df = df_daily_base.merge(market_ret, on='time', how='left')
df = df.merge(market_close, on='time', how='left')
df = df.dropna(subset=['vnindex_log_ret', 'vnindex_close']).reset_index(drop=True)
df['vnindex_vol20'] = df['vnindex_log_ret'].rolling(20).std() * np.sqrt(252)

df_fnb = pd.read_csv('../data/processed/m4_foreign_net_buy_sell.csv')
df_fnb['time'] = pd.to_datetime(df_fnb['time'])
df = df.merge(df_fnb[['time', 'fnb_ratio']], on='time', how='left')

df_fx = pd.read_csv('../data/processed/e1_usdvnd.csv')
df_fx['time'] = pd.to_datetime(df_fx['time'])
df = df.merge(df_fx[['time', 'fx_log_ret']], on='time', how='left')
df = df.dropna().reset_index(drop=True)
print(f"Kích thước bảng dữ liệu gốc: {df.shape}")
df.head(5)


# ## 2. Bộ Lọc Kiểm Định Kỹ Thuật (Stationarity & Kurtosis)
# 
# Mục đích của bước này là đánh giá tính chất toán học của các đặc trưng để chọn lọc đầu vào ổn định cho HMM:
# 1. **Kiểm định tính dừng (Stationarity):** Sử dụng cả ADF (yêu cầu bác bỏ giả thuyết không, $p < 0.05$) và KPSS (yêu cầu chấp nhận giả thuyết không, $p \ge 0.05$) để kiểm tra chuỗi dừng thực tế ở dạng $I(0)$. Chuỗi dừng đảm bảo các thuộc tính thống kê không thay đổi theo thời gian, tránh lỗi ước lượng suy biến.
# 2. **Hệ số nhọn (Excess Kurtosis):** Kiểm tra xem phân phối của biến có đuôi quá dày hay không ($|kurt| < 10$). Phân phối lệch chuẩn quá mức sẽ vi phạm giả định phân phối Gauss ẩn của HMM.
# 
# ### Bảng giải thích các chỉ số kiểm định:
# 
# | Chỉ số (Metric) | Ý nghĩa (Meaning) | Ngưỡng chấp nhận (Threshold) | Tác dụng trong HMM (Purpose in HMM) |
# | :--- | :--- | :--- | :--- |
# | **ADF (p-value)** | Kiểm định giả thuyết nghiệm đơn vị (Unit Root). Giả thuyết $H_0$: Chuỗi không dừng. | $p < 0.05$ (Bác bỏ $H_0$, chuỗi dừng) | HMM yêu cầu các đặc trưng đầu vào có tính phân phối ổn định qua thời gian để tránh sai số ước lượng. |
# | **KPSS (p-value)** | Kiểm định tính dừng xung quanh xu thế. Giả thuyết $H_0$: Chuỗi dừng. | $p \ge 0.05$ (Không bác bỏ $H_0$, chuỗi dừng) | Bổ trợ cho ADF để xác nhận chắc chắn chuỗi dừng (tránh lỗi ngụy tạo từ ADF). |
# | **Kurtosis (Hệ số nhọn)** | Đo lường mức độ tập trung của phân phối quanh giá trị trung bình và độ dày của đuôi. | $\vert \text{Kurt} \vert < 10$ | Tránh hiện tượng đuôi quá béo (fat-tails) gây nhiễu cho ước lượng GMM trong HMM. |
# | **Skewness (Hệ số lệch)** | Đo lường tính bất đối xứng của phân phối dữ liệu quanh giá trị trung bình. | Ghi nhận mô tả | Đánh giá xu hướng lệch của biến trước khi nạp vào phân phối chuẩn hỗn hợp. |

# In[16]:


print("Đang trích xuất lưới dữ liệu vĩ mô tần suất Tháng...")
df_monthly = df[['time', 'cpi_mom', 'credit_growth_mom', 'pmi_vn']].copy()
df_monthly['year_month'] = df_monthly['time'].dt.to_period('M')
df_monthly = df_monthly.groupby('year_month').first().reset_index(drop=True)

# Lag macro features by 1 month to avoid publication look-ahead bias (V2)
macro_features = ['cpi_mom', 'credit_growth_mom', 'pmi_vn']
df_monthly[macro_features] = df_monthly[macro_features].shift(1)
df_monthly = df_monthly.dropna().reset_index(drop=True)

daily_pool = ['rolling_vol_5', 'volume_ratio', 'ret_disp', 'amihud_diff_normalized', 'fnb_ratio', 'fx_log_ret']

def check_stationarity_and_distribution(df_source, feature_list):
    results = []
    for c in feature_list:
        s = df_source[c].dropna()
        if len(s) < 30:
            is_stat, p_adf, p_kpss = False, np.nan, np.nan
        else:
            p_adf = adfuller(s, autolag='AIC')[1]
            p_kpss = kpss(s, regression='c', nlags='auto')[1]
            is_stat = (p_adf < 0.05) and (p_kpss >= 0.05)
        k = kurtosis(s)
        sk = skew(s)
        results.append({
            'feature': c,
            'kpss': p_kpss,
            'adf': p_adf,
            'stationary': is_stat,
            'kurt': k,
            'skew': sk,
            'pass_kurt': abs(k) < 10,
            'keep': is_stat and (abs(k) < 10)
        })
    return pd.DataFrame(results)

print("--- Kết quả kiểm định vĩ mô Tháng ---")
display(check_stationarity_and_distribution(df_monthly, macro_features).round(3))

print("\n--- Kết quả kiểm định thị trường Ngày ---")
display(check_stationarity_and_distribution(df, daily_pool).round(3))


# ## 3. Điểm Thông Tin Tương Hỗ (Mutual Information - MI)
# 
# Điểm MI đo lường mức độ phụ thuộc thông tin (kể cả phi tuyến) giữa đặc trưng đầu vào và trị tuyệt đối lợi suất thị trường `|vnindex_log_ret|` (đại diện cho trạng thái biến động). Điểm số MI cao chỉ ra biến đó có khả năng giải thích cao cho sự thay đổi biến động.
# 
# ### Bảng giải thích các chỉ số đo lường lượng tin:
# 
# | Chỉ số (Metric) | Ý nghĩa (Meaning) | Ngưỡng chấp nhận (Threshold) | Tác dụng trong HMM (Purpose in HMM) |
# | :--- | :--- | :--- | :--- |
# | **Mutual Information (MI)** | Đo lường lượng thông tin chung thu được về biến mục tiêu thông qua đặc trưng đầu vào (kể cả quan hệ phi tuyến). | Càng cao càng tốt ($\text{MI} > 0.01$ khuyên dùng) | Xác định đặc trưng nào chứa nhiều thông tin giải thích nhất cho trạng thái biến động của thị trường. |
# 

# In[17]:


y_daily = df['vnindex_log_ret'].abs().values
X_daily = df[daily_pool].values
mi_scores = mutual_info_regression(X_daily, y_daily, random_state=RANDOM_STATE)
df_mi = pd.DataFrame({'feature': daily_pool, 'mi': mi_scores}).sort_values('mi', ascending=False)
print("--- Bảng xếp hạng MI Scores ---")
display(df_mi.round(4))


# ## 4. Lựa Chọn Đặc Trưng Tham Lam & Kiểm Soát VIF
# 
# Hàm lọc biến đảm bảo tính đa dạng và hạn chế đa cộng tuyến tuyến tính sử dụng điểm số MI kết hợp hệ số phóng đại phương sai VIF.
# 
# ### Bảng giải thích chỉ số kiểm soát đa cộng tuyến và đa dạng:
# 
# | Chỉ số (Metric) | Ý nghĩa (Meaning) | Ngưỡng chấp nhận (Threshold) | Tác dụng trong HMM (Purpose in HMM) |
# | :--- | :--- | :--- | :--- |
# | **Variance Inflation Factor (VIF)** | Đo lường mức độ ảnh hưởng của hiện tượng đa cộng tuyến (multicollinearity) giữa một đặc trưng với các đặc trưng khác. | $\text{VIF} < 5.0$ (Tốt); $\text{VIF} \ge 10.0$ (Đa cộng tuyến nghiêm trọng) | Ngăn ngừa ma trận hiệp phương sai của HMM bị suy biến (null eigenvalue) khi huấn luyện mô hình Gauss hỗn hợp. |
# | **Block Diversity** | Đảm bảo tính đa dạng thông tin bằng cách lấy ít nhất một biến đại diện từ mỗi khâu (`Market`, `Economy`, `Credit`). | Bắt buộc chọn từ các nhóm khác nhau | Giúp HMM quan sát thị trường từ nhiều khía cạnh bổ trợ thay vì chỉ tập trung vào một nguồn tin. |

# In[18]:


BLOCKS = {
    'M': ['rolling_vol_5', 'volume_ratio', 'fnb_ratio'],
    'E': ['fx_log_ret'],
    'C': ['amihud_diff_normalized', 'ret_disp']
}

def get_block(feature):
    for b, features in BLOCKS.items():
        if feature in features: return b
    return None

def select_top_features(n_feats):
    selected = []
    for block in ['M', 'E', 'C']:
        block_features = [f for f in df_mi['feature'].values if get_block(f) == block]
        if block_features: selected.append(block_features[0])
    remaining = [f for f in df_mi['feature'].values if f not in selected]
    for f in remaining:
        if len(selected) >= n_feats: break
        trial = selected + [f]
        X_sub = df[trial].values
        X_sub = np.column_stack([np.ones(len(X_sub)), X_sub])
        vif = variance_inflation_factor(X_sub, len(trial))
        if vif < 5.0: selected.append(f)
    return selected

print("Đặc trưng Daily được lọc theo số lượng biến:")
for n in [4, 5, 6]:
    print(f" n_features={n} : {select_top_features(n)}")


# ## 5. Thiết Lập Tập Dữ Liệu Train/OOS & Hàm Hỗ Trợ HMM
# 
# Tính toán **Rolling Z-score (window=252 ngày)** phi rò rỉ và định nghĩa hàm fit mô hình.

# In[19]:


HMM_TRAIN_END = pd.Timestamp('2019-12-31')

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
    train_mask = fd['time'] <= HMM_TRAIN_END
    Z_tr = Z_all[train_mask]
    return fd, Z_tr, Z_all

def n_params(K, D):
    M = 2
    return (K - 1) + K * (K - 1) + K * (M - 1) + K * M * D + K * M * D * (D + 1) // 2

def fit_hmm(Z_train, Z_oos, K, n_seeds=5):
    best_ll, m_best = -np.inf, None
    oos_lls = []
    for seed in range(n_seeds):
        try:
            m = GMMHMM(n_components=K, n_mix=2, covariance_type='full',
                       min_covar=0.01, n_iter=200, tol=1e-4, random_state=seed*7+1)
            m.fit(Z_train)
            ll = m.score(Z_train)
            if len(Z_oos) > 0:
                oos_lls.append(m.score(Z_oos))
            if ll > best_ll:
                best_ll, m_best = ll, m
        except:
            continue
    avg_ll_oos = np.mean(oos_lls) if oos_lls else np.nan
    return m_best, best_ll, avg_ll_oos

def evaluate(model, Z_train, Z_oos):
    K        = model.n_components
    persist  = np.diag(model.transmat_)
    duration = 1.0 / (1.0 - persist + 1e-9)
    states   = model.predict(Z_train)
    counts   = np.bincount(states, minlength=K) / len(states)
    return {
        'll_oos':    model.score(Z_oos) if len(Z_oos) > 0 else np.nan,
        'min_dur':   float(duration.min()),
        'min_share': float(counts.min()),
        'max_share': float(counts.max()),
    }


# ## 6. Chạy Grid Search Tìm Cấu HÌnh HMM Tốt Nhất
# 
# Tối ưu hóa đồng thời số lượng đặc trưng ($n_{features}$) và số lượng trạng thái ẩn ($K$) cho mô hình ngày, và số trạng thái ẩn vĩ mô tháng. Các cấu hình được đánh giá qua các chỉ số chất lượng để chọn ra mô hình cân bằng nhất.
# 
# ### Bảng giải thích các chỉ số đánh giá Grid Search HMM:
# 
# | Chỉ số (Metric) | Ý nghĩa (Meaning) | Ngưỡng chọn/Ràng buộc (Constraint) | Tác dụng trong tối ưu hóa HMM (Purpose in HMM) |
# | :--- | :--- | :--- | :--- |
# | **D** | Số lượng chiều (dimensions) của dữ liệu. | Tối ưu theo mô hình | Quyết định xem mỗi bước được quyết định bởi bao nhiêu biến số |
# | **ll_in** | Log-Likelihood trong mẫu (In-sample). Đo lường độ khớp của mô hình với tập Train. | Càng cao càng tốt | Đánh giá khả năng giải thích dữ liệu huấn luyện của mô hình. |
# | **ll_oos** | Log-Likelihood ngoài mẫu (Out-of-sample). Đo lường độ khớp trên tập kiểm thử chưa thấy ứng với seed tốt nhất. | Càng cao càng tốt | Đánh giá khả năng tổng quát hóa của mô hình tốt nhất chọn được. |
# | **avg_ll_oos** | Trung bình Log-Likelihood ngoài mẫu (Average OOS LL) tính trên tất cả các seed hội tụ. | Càng cao càng tốt | Đo lường độ ổn định tổng quát hóa của thuật toán, tránh sự may rủi của một seed khởi tạo duy nhất. |
# | **bic** | Chỉ số thông tin Bayesian (Bayesian Information Criterion). Phạt mô hình có quá nhiều tham số. | Càng thấp càng tốt | Lựa chọn số lượng đặc trưng và số trạng thái ẩn tối ưu nhất, ngăn chặn quá khớp (Overfitting). |
# | **min_dur** | Thời gian lưu trú tối thiểu ở một trạng thái (Minimum State Duration). | $\ge 3$ phiên giao dịch/tháng | Đảm bảo các trạng thái ẩn có tính bền vững nhất định, tránh việc chuyển đổi trạng thái liên tục gây nhiễu (chattering). |
# | **min_share / max_share** | Tỷ lệ số phiên tối thiểu/tối đa thuộc một trạng thái trong tập Train. | $\text{min\_share} \ge 0.05$, $\text{max\_share} \le 0.75$ | Đảm bảo sự phân bổ cân bằng giữa các trạng thái ẩn, tránh trường hợp một trạng thái quá loãng hoặc chiếm hết dữ liệu. |
# 

# In[20]:


# Only fit on train set to prevent look-ahead bias (V2)
Z_macro_all = df_monthly[macro_features].values
macro_train_mask = df_monthly['time'] <= HMM_TRAIN_END
Z_macro_train = Z_macro_all[macro_train_mask]
results_macro = []
for K in [2, 3, 4]:
    best_ll, best_m = -np.inf, None
    for seed in range(5):
        try:
            m = GMMHMM(n_components=K, covariance_type='full', random_state=seed*7+1, n_iter=200)
            m.fit(Z_macro_train)
            ll = m.score(Z_macro_train)
            if ll > best_ll: best_ll, best_m = ll, m
        except: continue
    if best_m is not None:
        p = K * (len(macro_features) + len(macro_features)*(len(macro_features)+1)/2) + K*(K-1) + (K-1)
        bic = -2 * best_ll + p * np.log(len(Z_macro_train))
        persist = np.diag(best_m.transmat_)
        duration = 1.0 / (1.0 - persist + 1e-9)
        results_macro.append({'K': K, 'll': best_ll, 'bic': bic, 'min_dur': float(duration.min())})
df_res_macro = pd.DataFrame(results_macro)
print("--- Kết quả Grid Search Monthly HMM ---")
display(df_res_macro.round(2))

results_daily = []
for n_feats in [4, 5, 6]:
    selected_feats = select_top_features(n_feats)
    fd_z, Z_tr, Z_all = make_Z(selected_feats, window=252)
    Z_oos = Z_all[len(Z_tr):]
    D, n_tr = Z_tr.shape[1], Z_tr.shape[0]
    for K in [3, 4]:
        m, ll, avg_ll_oos = fit_hmm(Z_tr, Z_oos, K, n_seeds=5)
        if m is None: continue
        p = n_params(K, D)
        bic = -2 * ll + p * np.log(n_tr)
        ev = evaluate(m, Z_tr, Z_oos)
        results_daily.append({
            'n_features': n_feats, 'K': K, 'D': D, 'll_in': ll, 'bic': bic,
            'll_oos': ev['ll_oos'], 'avg_ll_oos': avg_ll_oos, 'min_dur': ev['min_dur'],
            'min_share': ev['min_share'], 'max_share': ev['max_share'], 'features': selected_feats
        })
df_res_daily = pd.DataFrame(results_daily)
print("\n--- Kết quả Grid Search Daily HMM ---")
display(df_res_daily.drop(columns='features').round(2))


# ## 7. Lựa Chọn Cấu Hình Bằng Điểm Xếp Hạng Tổng Hợp (Composite Score)
# 
# Đánh giá và lựa chọn cấu hình HMM tối ưu nhất cho cả hai mô hình dựa trên tiêu chí điểm phạt BIC và khả năng tổng quát hóa dữ liệu OOS.
# $$Composite = 0.3 \cdot Rank_{bic} + 0.5 \cdot Rank_{oos} + 0.2 \cdot Rank_{min\_dur}$$
# 
# Nếu trong trường hợp có 2 cấu hình có cùng mức điểm đánh giá thì chọn cấu hình có **thời gian** nhỏ hơn
# >Công thức đánh giá dựa trên ý kiến chủ quan để thể hiện kết quả của quá trình pipeline ưu tiên hiệu suất dự đoán của mô hình với thời gian ngắn

# In[21]:


best_macro_row = df_res_macro.sort_values('bic').iloc[0]
K_MACRO_BEST = int(best_macro_row['K'])
print(f"Optimal Monthly HMM: K={K_MACRO_BEST}")

valid_daily = df_res_daily[
    (df_res_daily['min_dur']   >= 3.0) &
    (df_res_daily['min_share'] >= 0.05) &
    (df_res_daily['max_share'] <= 0.75)
].copy()
if len(valid_daily) == 0: valid_daily = df_res_daily.copy()

valid_daily['rank_bic'] = valid_daily['bic'].rank(ascending=True)
valid_daily['rank_oos'] = valid_daily['avg_ll_oos'].rank(ascending=False)
valid_daily['rank_min_dur'] = valid_daily['min_dur'].rank(ascending=True)
valid_daily['composite'] = 0.3 * valid_daily['rank_bic'] + 0.5 * valid_daily['rank_oos'] + 0.2 * valid_daily['rank_min_dur']
valid_daily = valid_daily.sort_values(['composite', 'rank_min_dur']).reset_index(drop=True)

print("\n--- Top các cấu hình Daily tốt nhất ---")
display(valid_daily.drop(columns='features').round(2).head(5))

best_daily = valid_daily.iloc[0]
K_DAILY_BEST = 4 # int(best_daily['K'])
DAILY_FEATURES_BEST = best_daily['features']
print(f"\n>>> Lựa chọn tối ưu Daily HMM: n_features={len(DAILY_FEATURES_BEST)}, K={K_DAILY_BEST}")
print(f">>> Bộ đặc trưng tối ưu Daily: {DAILY_FEATURES_BEST}")


# ## 8. Huấn Luyện Lại Mô Hình HMM Tối Ưu Cuối Cùng (Refit Final Model)
# 
# Refit lại mô hình Monthly và Daily chung của thị trường.

# In[22]:


model_macro = GaussianHMM(n_components=K_MACRO_BEST, covariance_type='full', random_state=RANDOM_STATE, n_iter=200)
# Only train on train set, then predict on all (V2)
Z_macro_all = df_monthly[macro_features].values
macro_train_mask = df_monthly['time'] <= HMM_TRAIN_END
Z_macro_train = Z_macro_all[macro_train_mask]
model_macro.fit(Z_macro_train)
macro_states = model_macro.predict(Z_macro_all)
macro_probs = model_macro.predict_proba(Z_macro_all)
print(f"Monthly Macro HMM hội tụ: {model_macro.monitor_.converged}")

fd_z, Z_tr, Z_all = make_Z(DAILY_FEATURES_BEST, window=252)
model_daily, ll_daily, _ = fit_hmm(Z_tr, Z_all[len(Z_tr):], K_DAILY_BEST, n_seeds=10)
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
    if len(Z_train_tick) > 100:
        model_ticker.fit(Z_train_tick)
    else:
        model_ticker = model_daily # Fallback về model thị trường chung

    ticker_daily_states = model_ticker.predict(Z_all_tick)
    ticker_daily_probs = model_ticker.predict_proba(Z_all_tick)

    # Bước 4: Tự động Gắn nhãn Không Cảm Tính (Dynamic Auto-Labeling)
    df_temp_label = pd.merge(fd_z_tick[['time']], ticker_aligned[['time', 'log_return']], on='time')
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


import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive_output

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
REGIME_COLORS = {}
for i, label in enumerate(unique_labels):
    if label in predefined:
        REGIME_COLORS[label] = predefined[label]
    else:
        REGIME_COLORS[label] = color_pool[i % len(color_pool)]

def plot_ticker_regimes(ticker, date_range):
    sub_df = df_ticker[
        (df_ticker['ticker'] == ticker) &
        (df_ticker['time'] >= date_range[0]) &
        (df_ticker['time'] <= date_range[1])
    ].copy().sort_values('time').reset_index(drop=True)

    if len(sub_df) == 0:
        print("Không có dữ liệu cho khoảng thời gian này.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Đồ thị giá đóng cửa
    ax1.plot(sub_df['time'], sub_df['close'], color='black', linewidth=1.5, label=f"{ticker} Close")
    ax1.set_title(f"Biểu Đồ Trạng Thái Ẩn HMM - Mã: {ticker}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Giá Đóng Cửa (VND)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Tô màu nền theo trạng thái HMM
    regime_series = sub_df['market_regime_label']
    times = sub_df['time']

    for i in range(len(sub_df) - 1):
        reg = regime_series.iloc[i]
        color = REGIME_COLORS.get(reg, '#ffffff')
        ax1.axvspan(times.iloc[i], times.iloc[i+1], color=color, alpha=0.6)
        ax2.axvspan(times.iloc[i], times.iloc[i+1], color=color, alpha=0.6)

    # Đồ thị khối lượng giao dịch
    ax2.bar(sub_df['time'], sub_df['volume'], color='grey', alpha=0.6, label="Volume")
    ax2.set_ylabel("Khối Lượng", fontsize=12)
    ax2.set_xlabel("Thời Gian", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Tạo legend tùy chỉnh cho regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, edgecolor='none', label=label) 
        for label, color in REGIME_COLORS.items()
    ]
    ax1.legend(handles=legend_elements + [ax1.get_lines()[0]], loc='upper left')

    plt.tight_layout()
    plt.show()

# Thiết lập Widgets cho Jupyter
ticker_dropdown = widgets.Dropdown(
    options=sorted(df_ticker['ticker'].unique()),
    value='HPG',
    description='Mã Ticker:',
)

dates = sorted(df_ticker['time'].unique())
date_slider = widgets.SelectionRangeSlider(
    options=dates,
    index=(0, len(dates)-1),
    description='Khoảng Đo:',
    orientation='horizontal',
    layout={'width': '80%'}
)

ui = widgets.VBox([ticker_dropdown, date_slider])
out = interactive_output(plot_ticker_regimes, {'ticker': ticker_dropdown, 'date_range': date_slider})

display(ui, out)

