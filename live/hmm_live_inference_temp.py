#!/usr/bin/env python
# coding: utf-8

# # Hybrid Dual-Frequency Ticker-Specific HMM
# 
# Quy trÃ¬nh káº¿t há»£p Ä‘Ã¡nh giÃ¡ VÄ© mÃ´ (Monthly) vÃ  Thá»‹ trÆ°á»ng chung (Daily) Ä‘á»ƒ Ä‘á»‹nh vá»‹ pha biáº¿n Ä‘á»™ng, sau Ä‘Ã³ Ã©p cáº¥u trÃºc thá»‹ trÆ°á»ng chung lÃªn tá»«ng mÃ£ cá»• phiáº¿u (Ticker-Specific) vÃ  dÃ¹ng Meta-Classifier dá»± Ä‘oÃ¡n lá»£i suáº¥t ngÃ y tá»›i.

# In[10]:


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
            filtered_regimes[t-1] = 0
    return filtered_regimes, filtered_probs


import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import skew, kurtosis, norm
from sklearn.feature_selection import mutual_info_regression
from hmmlearn.hmm import GMMHMM, GaussianHMM
import lightgbm as lgb
import shap
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = Path('../output/hmm_v3_op1_extended')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"ThÆ° má»¥c Ä‘áº§u ra Ä‘Æ°á»£c thiáº¿t láº­p táº¡i: {OUTPUT_DIR.resolve()}")


# ## 1. Táº£i Dá»¯ Liá»‡u & Táº¡o Chá»‰ BÃ¡o Thá»‹ TrÆ°á»ng Äáº¡i Diá»‡n (VN-Index Proxy)

# In[11]:


print("Äang táº£i dá»¯ liá»‡u hmm_data.csv vÃ  m1_vn46.csv...")
df_daily_base = pd.read_csv('../output/hmm_data.csv')
df_daily_base['time'] = pd.to_datetime(df_daily_base['time'])

df_m1 = pd.read_csv('../data/processed/m1_vn46.csv')
df_m1['time'] = pd.to_datetime(df_m1['time']).dt.normalize()

# Táº¡o Market Proxy tá»« rá»• VN46
market_ret = df_m1.groupby('time')['log_return'].mean().reset_index()
market_ret.columns = ['time', 'vnindex_log_ret']
market_close = df_m1.groupby('time')['close'].mean().reset_index()
market_close.columns = ['time', 'vnindex_close']

df_market = df_daily_base.merge(market_ret, on='time', how='left')
df_market = df_market.merge(market_close, on='time', how='left')
df_market = df_market.dropna(subset=['vnindex_log_ret', 'vnindex_close']).reset_index(drop=True)
df_market['vnindex_vol20'] = df_market['vnindex_log_ret'].rolling(20).std() * np.sqrt(252)

try:
    df_fnb = pd.read_csv('../data/processed/m4_foreign_net_buy_sell.csv')
    df_fnb['time'] = pd.to_datetime(df_fnb['time'])
    df_market = df_market.merge(df_fnb[['time', 'fnb_ratio']], on='time', how='left')
except:
    pass

try:
    df_fx = pd.read_csv('../data/processed/e1_usdvnd.csv')
    df_fx['time'] = pd.to_datetime(df_fx['time'])
    df_market = df_market.merge(df_fx[['time', 'fx_log_ret']], on='time', how='left')
except:
    pass

df_market = df_market.dropna().reset_index(drop=True)
print(f"KÃ­ch thÆ°á»›c báº£ng dá»¯ liá»‡u Market gá»‘c: {df_market.shape}")


# ## 2. Bá»™ Lá»c Kiá»ƒm Äá»‹nh Ká»¹ Thuáº­t (Stationarity & Kurtosis)
# 
# Má»¥c Ä‘Ã­ch cá»§a bÆ°á»›c nÃ y lÃ  Ä‘Ã¡nh giÃ¡ tÃ­nh cháº¥t toÃ¡n há»c cá»§a cÃ¡c Ä‘áº·c trÆ°ng Ä‘á»ƒ chá»n lá»c Ä‘áº§u vÃ o á»•n Ä‘á»‹nh cho HMM:
# 1. **Kiá»ƒm Ä‘á»‹nh tÃ­nh dá»«ng (Stationarity):** Sá»­ dá»¥ng cáº£ ADF (yÃªu cáº§u bÃ¡c bá» giáº£ thuyáº¿t khÃ´ng, $p < 0.05$) vÃ  KPSS (yÃªu cáº§u cháº¥p nháº­n giáº£ thuyáº¿t khÃ´ng, $p \ge 0.05$) Ä‘á»ƒ kiá»ƒm tra chuá»—i dá»«ng thá»±c táº¿ á»Ÿ dáº¡ng $I(0)$. Chuá»—i dá»«ng Ä‘áº£m báº£o cÃ¡c thuá»™c tÃ­nh thá»‘ng kÃª khÃ´ng thay Ä‘á»•i theo thá»i gian, trÃ¡nh lá»—i Æ°á»›c lÆ°á»£ng suy biáº¿n.
# 2. **Há»‡ sá»‘ nhá»n (Excess Kurtosis):** Kiá»ƒm tra xem phÃ¢n phá»‘i cá»§a biáº¿n cÃ³ Ä‘uÃ´i quÃ¡ dÃ y hay khÃ´ng ($|kurt| < 10$). PhÃ¢n phá»‘i lá»‡ch chuáº©n quÃ¡ má»©c sáº½ vi pháº¡m giáº£ Ä‘á»‹nh phÃ¢n phá»‘i Gauss áº©n cá»§a HMM.
# 
# ### Báº£ng giáº£i thÃ­ch cÃ¡c chá»‰ sá»‘ kiá»ƒm Ä‘á»‹nh:
# 
# | Chá»‰ sá»‘ (Metric) | Ã nghÄ©a (Meaning) | NgÆ°á»¡ng cháº¥p nháº­n (Threshold) | TÃ¡c dá»¥ng trong HMM (Purpose in HMM) |
# | :--- | :--- | :--- | :--- |
# | **ADF (p-value)** | Kiá»ƒm Ä‘á»‹nh giáº£ thuyáº¿t nghiá»‡m Ä‘Æ¡n vá»‹ (Unit Root). Giáº£ thuyáº¿t $H_0$: Chuá»—i khÃ´ng dá»«ng. | $p < 0.05$ (BÃ¡c bá» $H_0$, chuá»—i dá»«ng) | HMM yÃªu cáº§u cÃ¡c Ä‘áº·c trÆ°ng Ä‘áº§u vÃ o cÃ³ tÃ­nh phÃ¢n phá»‘i á»•n Ä‘á»‹nh qua thá»i gian Ä‘á»ƒ trÃ¡nh sai sá»‘ Æ°á»›c lÆ°á»£ng. |
# | **KPSS (p-value)** | Kiá»ƒm Ä‘á»‹nh tÃ­nh dá»«ng xung quanh xu tháº¿. Giáº£ thuyáº¿t $H_0$: Chuá»—i dá»«ng. | $p \ge 0.05$ (KhÃ´ng bÃ¡c bá» $H_0$, chuá»—i dá»«ng) | Bá»• trá»£ cho ADF Ä‘á»ƒ xÃ¡c nháº­n cháº¯c cháº¯n chuá»—i dá»«ng (trÃ¡nh lá»—i ngá»¥y táº¡o tá»« ADF). |
# | **Kurtosis (Há»‡ sá»‘ nhá»n)** | Äo lÆ°á»ng má»©c Ä‘á»™ táº­p trung cá»§a phÃ¢n phá»‘i quanh giÃ¡ trá»‹ trung bÃ¬nh vÃ  Ä‘á»™ dÃ y cá»§a Ä‘uÃ´i. | $\vert \text{Kurt} \vert < 10$ | TrÃ¡nh hiá»‡n tÆ°á»£ng Ä‘uÃ´i quÃ¡ bÃ©o (fat-tails) gÃ¢y nhiá»…u cho Æ°á»›c lÆ°á»£ng GMM trong HMM. |
# | **Skewness (Há»‡ sá»‘ lá»‡ch)** | Äo lÆ°á»ng tÃ­nh báº¥t Ä‘á»‘i xá»©ng cá»§a phÃ¢n phá»‘i dá»¯ liá»‡u quanh giÃ¡ trá»‹ trung bÃ¬nh. | Ghi nháº­n mÃ´ táº£ | ÄÃ¡nh giÃ¡ xu hÆ°á»›ng lá»‡ch cá»§a biáº¿n trÆ°á»›c khi náº¡p vÃ o phÃ¢n phá»‘i chuáº©n há»—n há»£p. |

# In[12]:


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
stat_df = pd.DataFrame(stat_results)
display(stat_df)

selected_raw_features = stat_df[stat_df['is_stationary']]['feature'].tolist()
if not selected_raw_features:
    selected_raw_features = daily_pool # Fallback

print(f'\n[KEEP] CÃ¡c Ä‘áº·c trÆ°ng ÄÆ¯á»¢C GIá»® Láº I ({len(selected_raw_features)} biáº¿n): {selected_raw_features}')
dropped = [c for c in daily_pool if c not in selected_raw_features]
print(f'[DROP] CÃ¡c Ä‘áº·c trÆ°ng Bá»Š LOáº I Bá»Ž ({len(dropped)} biáº¿n): {dropped}')



# ## 3. Thiáº¿t Láº­p Táº­p Dá»¯ Liá»‡u Train/OOS & HÃ m Há»— Trá»£ HMM sá»­ dá»¥ng NQT + Rank

# In[13]:


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

fd_market, Z_tr_market, Z_all_market = make_Z(df_market, selected_raw_features)
df_market_Z = pd.DataFrame(Z_all_market, columns=[c + '_Z' for c in selected_raw_features])
fd_market = fd_market.merge(df_market[['time', 'vnindex_log_ret', 'vnindex_close', 'vnindex_vol20']], on='time', how='left')
df_market = pd.concat([fd_market, df_market_Z], axis=1)


# ## 4. Äiá»ƒm ThÃ´ng Tin TÆ°Æ¡ng Há»— (Mutual Information - MI) & Lá»±a Chá»n Äáº·c TrÆ°ng Tham Lam & Kiá»ƒm SoÃ¡t VIF
# 
# Äiá»ƒm MI Ä‘o lÆ°á»ng má»©c Ä‘á»™ phá»¥ thuá»™c thÃ´ng tin (ká»ƒ cáº£ phi tuyáº¿n) giá»¯a Ä‘áº·c trÆ°ng Ä‘áº§u vÃ o vÃ  trá»‹ tuyá»‡t Ä‘á»‘i lá»£i suáº¥t thá»‹ trÆ°á»ng `|vnindex_log_ret|` (Ä‘áº¡i diá»‡n cho tráº¡ng thÃ¡i biáº¿n Ä‘á»™ng). Äiá»ƒm sá»‘ MI cao chá»‰ ra biáº¿n Ä‘Ã³ cÃ³ kháº£ nÄƒng giáº£i thÃ­ch cao cho sá»± thay Ä‘á»•i biáº¿n Ä‘á»™ng.
# 
# ### Báº£ng giáº£i thÃ­ch cÃ¡c chá»‰ sá»‘ Ä‘o lÆ°á»ng lÆ°á»£ng tin:
# 
# | Chá»‰ sá»‘ (Metric) | Ã nghÄ©a (Meaning) | NgÆ°á»¡ng cháº¥p nháº­n (Threshold) | TÃ¡c dá»¥ng trong HMM (Purpose in HMM) |
# | :--- | :--- | :--- | :--- |
# | **Mutual Information (MI)** | Äo lÆ°á»ng lÆ°á»£ng thÃ´ng tin chung thu Ä‘Æ°á»£c vá» biáº¿n má»¥c tiÃªu thÃ´ng qua Ä‘áº·c trÆ°ng Ä‘áº§u vÃ o (ká»ƒ cáº£ quan há»‡ phi tuyáº¿n). | CÃ ng cao cÃ ng tá»‘t ($\text{MI} > 0.01$ khuyÃªn dÃ¹ng) | XÃ¡c Ä‘á»‹nh Ä‘áº·c trÆ°ng nÃ o chá»©a nhiá»u thÃ´ng tin giáº£i thÃ­ch nháº¥t cho tráº¡ng thÃ¡i biáº¿n Ä‘á»™ng cá»§a thá»‹ trÆ°á»ng. |
# 
# HÃ m lá»c biáº¿n Ä‘áº£m báº£o tÃ­nh Ä‘a dáº¡ng vÃ  háº¡n cháº¿ Ä‘a cá»™ng tuyáº¿n tuyáº¿n tÃ­nh sá»­ dá»¥ng Ä‘iá»ƒm sá»‘ MI káº¿t há»£p há»‡ sá»‘ phÃ³ng Ä‘áº¡i phÆ°Æ¡ng sai VIF.
# 
# ### Báº£ng giáº£i thÃ­ch chá»‰ sá»‘ kiá»ƒm soÃ¡t Ä‘a cá»™ng tuyáº¿n vÃ  Ä‘a dáº¡ng:
# 
# | Chá»‰ sá»‘ (Metric) | Ã nghÄ©a (Meaning) | NgÆ°á»¡ng cháº¥p nháº­n (Threshold) | TÃ¡c dá»¥ng trong HMM (Purpose in HMM) |
# | :--- | :--- | :--- | :--- |
# | **Variance Inflation Factor (VIF)** | Äo lÆ°á»ng má»©c Ä‘á»™ áº£nh hÆ°á»Ÿng cá»§a hiá»‡n tÆ°á»£ng Ä‘a cá»™ng tuyáº¿n (multicollinearity) giá»¯a má»™t Ä‘áº·c trÆ°ng vá»›i cÃ¡c Ä‘áº·c trÆ°ng khÃ¡c. | $\text{VIF} < 5.0$ (Tá»‘t); $\text{VIF} \ge 10.0$ (Äa cá»™ng tuyáº¿n nghiÃªm trá»ng) | NgÄƒn ngá»«a ma tráº­n hiá»‡p phÆ°Æ¡ng sai cá»§a HMM bá»‹ suy biáº¿n (null eigenvalue) khi huáº¥n luyá»‡n mÃ´ hÃ¬nh Gauss há»—n há»£p. |
# | **Block Diversity** | Äáº£m báº£o tÃ­nh Ä‘a dáº¡ng thÃ´ng tin báº±ng cÃ¡ch láº¥y Ã­t nháº¥t má»™t biáº¿n Ä‘áº¡i diá»‡n tá»« má»—i khÃ¢u (`Market`, `Economy`, `Credit`). | Báº¯t buá»™c chá»n tá»« cÃ¡c nhÃ³m khÃ¡c nhau | GiÃºp HMM quan sÃ¡t thá»‹ trÆ°á»ng tá»« nhiá»u khÃ­a cáº¡nh bá»• trá»£ thay vÃ¬ chá»‰ táº­p trung vÃ o má»™t nguá»“n tin. |

# In[14]:


# Táº¡o Y_proxy rule-based cho Market
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
    elif ret < 0 and vol > vol_median: return 1 # Bear / High Vol
    else: return 2 # Sideways

df_market['Y_proxy'] = df_market.apply(label_proxy, axis=1)

Z_features = [c + '_Z' for c in selected_raw_features]
X_train = df_market[Z_features].dropna()
y_train = df_market.loc[X_train.index, 'Y_proxy']

# TÃ­nh SHAP
clf = lgb.LGBMClassifier(n_estimators=50, random_state=RANDOM_STATE, verbose=-1)
clf.fit(X_train, y_train)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train)
if isinstance(shap_values, list):
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    vals = shap_values.values if hasattr(shap_values, 'values') else np.array(shap_values)
    if len(vals.shape) == 3:
        if vals.shape[1] == len(Z_features):
            mean_shap = np.abs(vals).mean(axis=0).mean(axis=1)
        else:
            mean_shap = np.abs(vals).mean(axis=0).mean(axis=0)
    else:
        mean_shap = np.abs(vals).mean(axis=0)
shap_df = pd.DataFrame({'feature': Z_features, 'shap_importance': mean_shap})

# TÃ­nh MI vá»›i |vnindex_log_ret|
target_mi = np.abs(df_market.loc[X_train.index, 'vnindex_log_ret'])
mi_scores = mutual_info_regression(X_train, target_mi, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({'feature': Z_features, 'mi_score': mi_scores})

# Gá»™p Ä‘iá»ƒm & Lá»c VIF tham lam
feature_scores = shap_df.merge(mi_df, on='feature')
feature_scores['total_score'] = feature_scores['shap_importance'] * feature_scores['mi_score']
feature_scores = feature_scores.sort_values('total_score', ascending=False)

def filter_vif_greedy(df, features, threshold=5.0):
    selected = []
    for f in features:
        trial = selected + [f]
        X_sub = df[trial].values
        X_sub = np.column_stack([np.ones(len(X_sub)), X_sub])
        vif = variance_inflation_factor(X_sub, len(trial))
        if vif < threshold:
            selected.append(f)
    return selected

final_features = filter_vif_greedy(X_train, feature_scores['feature'].tolist())
print("Top features (SHAP+MI) qua bá»™ lá»c VIF:")
macro_pool = [c for c in ['cpi_mom_Z', 'credit_growth_mom_Z', 'fnb_ratio_Z', 'pmi_vn_Z', 'fx_log_ret_Z'] if c in X_train.columns]
final_features = set(final_features).union(set(macro_pool))
macro_features = [f for f in final_features if f in macro_pool]
market_features = [f for f in final_features if f not in macro_pool]
print('Macro Features:', macro_features)
print('Market Features:', market_features)
display(feature_scores.head(10))


# ## 5. Cháº¡y Grid Search TÃ¬m Cáº¥u HÃ¬nh HMM Tá»‘t Nháº¥t (Kiáº¿n TrÃºc TÃ¡ch Lá»›p)
# 
# Há»‡ thá»‘ng giá» Ä‘Ã¢y Ä‘Æ°á»£c Ä‘Ã¡nh giÃ¡ theo 2 táº§ng:
# 1. **Táº§ng VÄ© MÃ´ (Macro HMM):** Cháº¡y trÃªn khung thá»i gian ThÃ¡ng (Monthly) sá»­ dá»¥ng cÃ¡c biáº¿n VÄ© mÃ´ Ä‘á»ƒ xÃ¡c Ä‘á»‹nh bá»‘i cáº£nh tá»•ng thá»ƒ. Dá»¯ liá»‡u Ä‘Æ°á»£c tá»‹nh tiáº¿n (shift 1 thÃ¡ng) Ä‘á»ƒ xá»­ lÃ½ hoÃ n toÃ n váº¥n Ä‘á» Look-ahead Bias do Ä‘á»™ trá»… cÃ´ng bá»‘ (Publication Lag).
# 2. **Táº§ng Thá»‹ TrÆ°á»ng (Market HMM):** Cháº¡y trÃªn khung thá»i gian NgÃ y (Daily). Háº¥p thá»¥ xÃ¡c suáº¥t VÄ© MÃ´ (`Macro_Prob`) káº¿t há»£p cÃ¹ng cÃ¡c biáº¿n thá»‹ trÆ°á»ng chung Ä‘á»ƒ xÃ¡c Ä‘á»‹nh tráº¡ng thÃ¡i rá»§i ro hÃ ng ngÃ y má»™t cÃ¡ch toÃ n diá»‡n.
# 
# ### Báº£ng giáº£i thÃ­ch cÃ¡c chá»‰ sá»‘ Ä‘Ã¡nh giÃ¡ Grid Search HMM:
# 
# | Chá»‰ sá»‘ (Metric) | Ã nghÄ©a (Meaning) | NgÆ°á»¡ng chá»n/RÃ ng buá»™c (Constraint) | TÃ¡c dá»¥ng trong tá»‘i Æ°u hÃ³a HMM (Purpose in HMM) |
# | :--- | :--- | :--- | :--- |
# | **D** | Sá»‘ lÆ°á»£ng chiá»u (dimensions) cá»§a dá»¯ liá»‡u. | Tá»‘i Æ°u theo mÃ´ hÃ¬nh | Quyáº¿t Ä‘á»‹nh xem má»—i bÆ°á»›c Ä‘Æ°á»£c quyáº¿t Ä‘á»‹nh bá»Ÿi bao nhiÃªu biáº¿n sá»‘ |
# | **ll_in** | Log-Likelihood trong máº«u (In-sample). Äo lÆ°á»ng Ä‘á»™ khá»›p cá»§a mÃ´ hÃ¬nh vá»›i táº­p Train. | CÃ ng cao cÃ ng tá»‘t | ÄÃ¡nh giÃ¡ kháº£ nÄƒng giáº£i thÃ­ch dá»¯ liá»‡u huáº¥n luyá»‡n cá»§a mÃ´ hÃ¬nh. |
# | **ll_oos** | Log-Likelihood ngoÃ i máº«u (Out-of-sample). Äo lÆ°á»ng Ä‘á»™ khá»›p trÃªn táº­p kiá»ƒm thá»­ chÆ°a tháº¥y á»©ng vá»›i seed tá»‘t nháº¥t. | CÃ ng cao cÃ ng tá»‘t | ÄÃ¡nh giÃ¡ kháº£ nÄƒng tá»•ng quÃ¡t hÃ³a cá»§a mÃ´ hÃ¬nh tá»‘t nháº¥t chá»n Ä‘Æ°á»£c. |
# | **bic** | Chá»‰ sá»‘ thÃ´ng tin Bayesian (Bayesian Information Criterion). Pháº¡t mÃ´ hÃ¬nh cÃ³ quÃ¡ nhiá»u tham sá»‘. | CÃ ng tháº¥p cÃ ng tá»‘t | Lá»±a chá»n sá»‘ lÆ°á»£ng Ä‘áº·c trÆ°ng vÃ  sá»‘ tráº¡ng thÃ¡i áº©n tá»‘i Æ°u nháº¥t, ngÄƒn cháº·n quÃ¡ khá»›p (Overfitting). |
# | **min_dur** | Thá»i gian lÆ°u trÃº tá»‘i thiá»ƒu á»Ÿ má»™t tráº¡ng thÃ¡i (Minimum State Duration). | $\ge 2$ phiÃªn | Äáº£m báº£o cÃ¡c tráº¡ng thÃ¡i áº©n cÃ³ tÃ­nh bá»n vá»¯ng nháº¥t Ä‘á»‹nh, trÃ¡nh chattering. |
# | **min_share / max_share** | Tá»· lá»‡ sá»‘ phiÃªn tá»‘i thiá»ƒu/tá»‘i Ä‘a thuá»™c má»™t tráº¡ng thÃ¡i trong táº­p Train. | $\text{min\_share} \ge 0.05$, $\text{max\_share} \le 0.85$ | Äáº£m báº£o sá»± phÃ¢n bá»• cÃ¢n báº±ng giá»¯a cÃ¡c tráº¡ng thÃ¡i áº©n. |
# 
# $$Composite = 0.3 \cdot Rank_{bic} + 0.5 \cdot Rank_{oos} + 0.2 \cdot Rank_{min\_dur}$$
# 
# 

# In[15]:


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
Z_data_macro = df_monthly[macro_features].fillna(0).values
Z_train_macro = Z_data_macro[train_mask_macro]
Z_oos_macro = Z_data_macro[~train_mask_macro]

def evaluate_hmm(K, Z_train, Z_oos, seeds=3):
    Z_train = Z_train + np.random.normal(0, 1e-4, Z_train.shape)
    if len(Z_oos) > 0: Z_oos = Z_oos + np.random.normal(0, 1e-4, Z_oos.shape)
    best_bic, best_ll_oos, best_model = np.inf, -np.inf, None
    best_min_dur, best_min_share, best_max_share = 0, 0, 1

    for seed in range(seeds):
        try:
            m = GMMHMM(n_components=K, n_mix=2, covariance_type='diag', min_covar=0.01, n_iter=200, random_state=seed*7)
            m.fit(Z_train)
            ll_train = m.score(Z_train)

            p = n_params(K, Z_train.shape[1])
            bic = -2 * ll_train + p * np.log(len(Z_train))
            ll_oos = m.score(Z_oos) if len(Z_oos)>0 else np.nan

            persist = np.diag(m.transmat_)
            min_dur = float((1.0 / (1.0 - persist + 1e-9)).min())

            preds = m.predict(Z_train)
            counts = np.bincount(preds, minlength=K) / len(Z_train)
            min_share = float(counts.min())
            max_share = float(counts.max())

            if bic < best_bic and min_dur >= 2.0 and min_share >= 0.05 and max_share <= 0.85:
                best_bic, best_ll_oos, best_model = bic, ll_oos, m
                best_min_dur, best_min_share, best_max_share = min_dur, min_share, max_share
        except Exception as e:
            continue
    return best_model, best_bic, best_ll_oos, best_min_dur, best_min_share, best_max_share

print("==> EVALUATING MACRO HMM (MONTHLY TIMEFRAME)...")
results_macro, models_macro = [], {}
for K in [2, 3]:
    m, bic, ll_oos, min_dur, min_share, max_share = evaluate_hmm(K, Z_train_macro, Z_oos_macro, seeds=5)
    if m:
        results_macro.append({'K': K, 'BIC': bic, 'll_oos': ll_oos, 'min_dur': min_dur, 'min_share': min_share, 'max_share': max_share})
        models_macro[K] = m
res_df_macro = pd.DataFrame(results_macro)

prob_cols = []
if len(res_df_macro) > 0:
    res_df_macro['Rank_bic'] = res_df_macro['BIC'].rank(ascending=True)
    res_df_macro['Rank_oos'] = res_df_macro['ll_oos'].rank(ascending=False)
    res_df_macro['Composite'] = 0.5 * res_df_macro['Rank_bic'] + 0.5 * res_df_macro['Rank_oos']
    res_df_macro = res_df_macro.sort_values('Composite')
    K_macro = int(res_df_macro.iloc[0]['K'])
    best_macro_hmm = models_macro[K_macro]
    display(res_df_macro)
    print(f"--> Quyáº¿t Ä‘á»‹nh K tá»‘i Æ°u Macro: K = {K_macro}")

    global_macro_regimes, macro_probs = get_hmm_filtered_inference(best_macro_hmm, Z_data_macro)

    for i in range(K_macro):
        col = f'Macro_Prob_{i}'
        df_monthly[col] = macro_probs[:, i]
        prob_cols.append(col)

    # SHIFT 1 THÃNG Äá»‚ Xá»¬ LÃ Äá»˜ TRá»„ CÃ”NG Bá» (PUBLICATION LAG)
    df_monthly_shifted = df_monthly.copy()
    df_monthly_shifted['year_month'] = df_monthly_shifted['year_month'] + 1

    df_market = df_market.merge(df_monthly_shifted[['year_month'] + prob_cols], on='year_month', how='left')
    df_market[prob_cols] = df_market[prob_cols].ffill().fillna(0) # Äiá»n 0 cho nhá»¯ng ngÃ y Ä‘áº§u tiÃªn chÆ°a cÃ³ vÄ© mÃ´
else:
    print("KhÃ´ng tÃ¬m tháº¥y cáº¥u hÃ¬nh Macro HMM há»™i tá»¥!")


# =====================================================================
# 2. PREPARE & EVALUATE MARKET HMM (DAILY + MACRO PROBS)
# =====================================================================
print("\n==> EVALUATING MARKET HMM (DAILY TIMEFRAME WITH MACRO AWARENESS)...")
train_mask_market = df_market['time'] <= HMM_TRAIN_END

# Market HMM giá» Ä‘Ã¢y láº¥y Ä‘áº§u vÃ o lÃ  cáº£ biáº¿n thá»‹ trÆ°á»ng VÃ€ xÃ¡c suáº¥t VÄ© mÃ´!
combined_market_features = market_features + prob_cols[:-1] if prob_cols else market_features
Z_data_market = df_market[combined_market_features].fillna(0).values

Z_train_market = Z_data_market[train_mask_market]
Z_oos_market = Z_data_market[~train_mask_market]

results_market, models_market = [], {}
for K in [2, 3, 4]:
    m, bic, ll_oos, min_dur, min_share, max_share = evaluate_hmm(K, Z_train_market, Z_oos_market)
    if m:
        results_market.append({'K': K, 'BIC': bic, 'll_oos': ll_oos, 'min_dur': min_dur, 'min_share': min_share, 'max_share': max_share})
        models_market[K] = m
res_df_market = pd.DataFrame(results_market)

if len(res_df_market) > 0:
    res_df_market['Rank_bic'] = res_df_market['BIC'].rank(ascending=True)
    res_df_market['Rank_oos'] = res_df_market['ll_oos'].rank(ascending=False)
    res_df_market['Rank_min_dur'] = res_df_market['min_dur'].rank(ascending=False)
    res_df_market['Composite'] = 0.3 * res_df_market['Rank_bic'] + 0.5 * res_df_market['Rank_oos'] + 0.2 * res_df_market['Rank_min_dur']
    res_df_market = res_df_market.sort_values('Composite')
    K_market = 0 # ThÆ°á»ng cá»‘ Ä‘á»‹nh K=3
    if K_market not in models_market:
        K_market = int(res_df_market.iloc[0]['K'])
    best_market_hmm = models_market[K_market]
    display(res_df_market)
    print(f"--> Quyáº¿t Ä‘á»‹nh K tá»‘i Æ°u Market: K = {K_market}")

    global_market_regimes_filtered, market_probs = get_hmm_filtered_inference(best_market_hmm, Z_data_market)
    for i in range(K_market):
        df_market[f'Market_Prob_{i}'] = market_probs[:, i]
else:
    print("KhÃ´ng tÃ¬m tháº¥y cáº¥u hÃ¬nh Market HMM há»™i tá»¥!")



# ## 6. Tá»± Äá»™ng Ãnh Xáº¡ & GÃ¡n NhÃ£n Tráº¡ng ThÃ¡i (K-agnostic Labeling)
# 
# Ãnh xáº¡ nhÃ£n tráº¡ng thÃ¡i tá»± Ä‘á»™ng cho cáº£ vÄ© mÃ´ thÃ¡ng vÃ  thá»‹ trÆ°á»ng ngÃ y bÃ¡m sÃ¡t theo ká»³ vá»ng lá»£i nhuáº­n kÃ¬ vá»ng vÃ  biáº¿n Ä‘á»™ng tÃ i sáº£n.
# 
# ### Labling Monthly
# | Tráº¡ng thÃ¡i (Status) | Ã nghÄ©a (Meaning) | NgÆ°á»¡ng chá»n / RÃ ng buá»™c (Constraint) |
# |---------------------|-------------------|--------------------------------------|
# | **Macro_Stagnant** (K=2,3) | Giai Ä‘oáº¡n vÄ© mÃ´ trÃ¬ trá»‡, sáº£n xuáº¥t suy giáº£m hoáº·c tÄƒng trÆ°á»Ÿng cháº­m. | `pmi_vn` tháº¥p nháº¥t |
# | **Macro_Stable** (K=3) | Giai Ä‘oáº¡n vÄ© mÃ´ á»•n Ä‘á»‹nh, tÄƒng trÆ°á»Ÿng sáº£n xuáº¥t vá»«a pháº£i. | `pmi_vn` á»Ÿ má»©c trung vá»‹ / trung bÃ¬nh |
# | **Macro_Expansion** (K=2,3) | Giai Ä‘oáº¡n vÄ© mÃ´ má»Ÿ rá»™ng, sáº£n xuáº¥t tÄƒng trÆ°á»Ÿng máº¡nh máº½. | `pmi_vn` cao nháº¥t |
# 
# ### Labling Daily
# K = 3
# | Tráº¡ng thÃ¡i (Status) | Ã nghÄ©a (Meaning) | NgÆ°á»¡ng chá»n / RÃ ng buá»™c (Constraint) |
# |---------------------|-------------------|--------------------------------------|
# | **Bull** | Thá»‹ trÆ°á»ng tÄƒng trÆ°á»Ÿng, xu hÆ°á»›ng Ä‘i lÃªn vÃ  Ã­t biáº¿n Ä‘á»™ng (rá»§i ro tháº¥p). | `ret` khÃ´ng tháº¥p nháº¥t vÃ  `vol` tháº¥p nháº¥t trong 2 tráº¡ng thÃ¡i cÃ²n láº¡i |
# | **Sideways** | Thá»‹ trÆ°á»ng Ä‘i ngang, dao Ä‘á»™ng tÃ­ch lÅ©y trong biÃªn Ä‘á»™. | Tráº¡ng thÃ¡i cÃ²n láº¡i sau khi Ä‘Ã£ xÃ¡c Ä‘á»‹nh Bull vÃ  Bear |
# | **Bear** | Thá»‹ trÆ°á»ng suy thoÃ¡i, xu hÆ°á»›ng giáº£m máº¡nh vÃ  rá»§i ro cao. | `ret` tháº¥p nháº¥t (argmin) |
# 
# K = 4
# | Tráº¡ng thÃ¡i (Status) | Ã nghÄ©a (Meaning) | NgÆ°á»¡ng chá»n / RÃ ng buá»™c (Constraint) |
# |---------------------|-------------------|--------------------------------------|
# | **Crisis** | Thá»‹ trÆ°á»ng suy thoÃ¡i, biáº¿n Ä‘á»™ng máº¡nh, rá»§i ro cao. | `ret < 0` vÃ  `vol >= median` |
# | **CalmBull** | Pha tÄƒng trÆ°á»Ÿng bá»n vá»¯ng, á»•n Ä‘á»‹nh, Ã­t rá»§i ro. | `ret > 0` vÃ  `vol < median` |
# | **Euphoria** | Thá»‹ trÆ°á»ng hÆ°ng pháº¥n, tÄƒng máº¡nh kÃ¨m dao Ä‘á»™ng lá»›n. | `ret > 0` vÃ  `vol >= median` |
# | **Tranquil** | Thá»‹ trÆ°á»ng "láº·ng sÃ³ng", Ä‘i ngang, áº£m Ä‘áº¡m. | `ret â‰ˆ 0` vÃ  `vol < median` |

# In[16]:


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
        return {int(np.argmin(ret)): 'Bear', int(np.argmax(ret)): 'Bull'}
    elif K == 3:
        sharpe = ret / (vol + 1e-9)
        order = np.argsort(sharpe)
        return {int(order[0]): 'Bear', int(order[1]): 'Sideways', int(order[2]): 'Bull'}
    elif K >= 4:
        from scipy.optimize import linear_sum_assignment
        ret_Z = (ret - np.mean(ret)) / (np.std(ret) + 1e-9)
        vol_Z = (vol - np.mean(vol)) / (np.std(vol) + 1e-9)

        scores = np.zeros((K, 4))
        for i in range(K):
            scores[i, 0] = -ret_Z[i] + vol_Z[i] # Crisis
            scores[i, 1] = -ret_Z[i] - vol_Z[i] # Tranquil
            scores[i, 2] =  ret_Z[i] - vol_Z[i] # CalmBull
            scores[i, 3] =  ret_Z[i] + vol_Z[i] # Euphoria

        row_ind, col_ind = linear_sum_assignment(-scores)
        label_names = ['Crisis', 'Tranquil', 'CalmBull', 'Euphoria']
        labels = {r: label_names[c] for r, c in zip(row_ind, col_ind)}

        unassigned = set(range(K)) - set(row_ind)
        for i, st in enumerate(unassigned):
            labels[st] = f'Daily_Tier{i+5}'

        return labels
    return {i: f'State_{i}' for i in range(K)}

STATE_TO_LABEL_MARKET = auto_label(rs_market, K_market)
df_market_res['market_regime_label'] = df_market_res['market_regime'].map(STATE_TO_LABEL_MARKET)
for k in range(K_market):
    df_market_res[f'prob_market_{k}'] = df_market[f'Market_Prob_{k}']


# ## 7. Chuáº©n bá»‹ Dá»¯ liá»‡u NgÃ nh & Huáº¥n luyá»‡n Sector HMM (Grid Search K)
# Huáº¥n luyá»‡n HMM cho tá»«ng nhÃ³m ngÃ nh Ä‘á»™c láº­p. Káº¿t quáº£ sáº½ Ä‘Æ°á»£c dÃ¹ng lÃ m feature cho Ticker HMM.

# In[18]:


import pandas as pd
import numpy as np
from hmmlearn.hmm import GMMHMM

# Ãnh xáº¡ NgÃ nh vÃ  táº¡o Dá»¯ liá»‡u Sector
_ind_df = pd.read_csv('../src/data_collection/industries.csv')
_ind_df = _ind_df[_ind_df['icb_level'] == 1]
industry_mapping = dict(zip(_ind_df['symbol'], _ind_df['icb_name']))

print("Äang táº¡o Ä‘áº·c trÆ°ng nhÃ³m ngÃ nh (Sector Features)...")
df_m1['industry'] = df_m1['ticker'].map(industry_mapping)
sector_df = df_m1.groupby(['industry', 'time']).agg(
    sector_log_ret=('log_return', 'mean'),
    sector_volume=('volume', 'sum')
).reset_index()

from scipy.stats import norm
def make_nqt(series, window=252):
    rolling_rank = series.rolling(window=window, min_periods=1).rank()
    rolling_count = series.rolling(window=window, min_periods=1).count()
    pct = (rolling_rank - 0.5) / rolling_count
    return np.clip(norm.ppf(pct), -3.0, 3.0)

sector_dfs = []
for ind, group in sector_df.groupby('industry'):
    group = group.sort_values('time').reset_index(drop=True)
    group['sector_vol20'] = group['sector_log_ret'].rolling(20).std() * np.sqrt(252)
    group['sector_vol5'] = group['sector_log_ret'].rolling(5).std() * np.sqrt(252)
    group['sector_volume_ratio'] = group['sector_volume'] / group['sector_volume'].rolling(20).mean()

    group['sector_log_ret_Z'] = make_nqt(group['sector_log_ret'])
    group['sector_vol20_Z'] = make_nqt(group['sector_vol20'])
    group['sector_vol5_Z'] = make_nqt(group['sector_vol5'])
    group['sector_volume_ratio_Z'] = make_nqt(group['sector_volume_ratio'])
    sector_dfs.append(group)

df_sector_final = pd.concat(sector_dfs, ignore_index=True).dropna().reset_index(drop=True)

Z_sector_cols = ['sector_log_ret_Z', 'sector_vol20_Z', 'sector_vol5_Z', 'sector_volume_ratio_Z']
def n_params(K, D, M=2):
    return (K - 1) + K * (K - 1) + K * (M - 1) + K * M * D + K * M * D * (D + 1) // 2

def evaluate_hmm_sector(K, Z_train, Z_oos, seeds=3):
    Z_train = Z_train + np.random.normal(0, 1e-4, Z_train.shape)
    if len(Z_oos) > 0: Z_oos = Z_oos + np.random.normal(0, 1e-4, Z_oos.shape)
    best_bic, best_ll_oos, best_model = np.inf, -np.inf, None
    best_min_dur = 0
    for seed in range(seeds):
        try:
            m = GMMHMM(n_components=K, n_mix=2, covariance_type='diag', min_covar=0.01, n_iter=100, random_state=seed*7)
            m.fit(Z_train)
            ll_train = m.score(Z_train)
            p = n_params(K, Z_train.shape[1])
            bic = -2 * ll_train + p * np.log(len(Z_train))
            ll_oos = m.score(Z_oos) if len(Z_oos)>0 else np.nan
            persist = np.diag(m.transmat_)
            min_dur = float((1.0 / (1.0 - persist + 1e-9)).min())
            if bic < best_bic and min_dur >= 3.0:
                best_bic, best_ll_oos, best_model, best_min_dur = bic, ll_oos, m, min_dur
        except: continue
    return best_model, best_bic, best_ll_oos, best_min_dur

def auto_label_sector(rs, K):
    ret = rs['mean_ret'].values; vol = rs['mean_vol'].values
    if K == 2: return {int(np.argmin(ret)): 'Bear', int(np.argmax(ret)): 'Bull'}
    elif K == 3:
        sharpe = ret / (vol + 1e-9)
        order = np.argsort(sharpe)
        return {int(order[0]): 'Bear', int(order[1]): 'Sideways', int(order[2]): 'Bull'}
    elif K >= 4:
        from scipy.optimize import linear_sum_assignment
        ret_Z = (ret - np.mean(ret)) / (np.std(ret) + 1e-9)
        vol_Z = (vol - np.mean(vol)) / (np.std(vol) + 1e-9)
        scores = np.zeros((K, 4))
        for i in range(K):
            scores[i, 0] = -ret_Z[i] + vol_Z[i]
            scores[i, 1] = -ret_Z[i] - vol_Z[i]
            scores[i, 2] =  ret_Z[i] - vol_Z[i]
            scores[i, 3] =  ret_Z[i] + vol_Z[i]
        row_ind, col_ind = linear_sum_assignment(-scores)
        label_names = ['Crisis', 'Sideways', 'Bull', 'Euphoria']
        labels = {r: label_names[c] for r, c in zip(row_ind, col_ind)}
        unassigned = set(range(K)) - set(row_ind)
        for i, st in enumerate(unassigned): labels[st] = f'Tier{i}'
        return labels
    return {i: f'State_{i}' for i in range(K)}

print("Huáº¥n luyá»‡n Sector HMM (Tá»± Ä‘á»™ng Grid Search chá»n K tá»‘t nháº¥t cho tá»«ng ngÃ nh)...")
sector_results = []
all_semantic_labels = set()
HMM_TRAIN_END = pd.Timestamp('2019-12-31')

for industry, group in df_sector_final.groupby('industry'):
    group = group.sort_values('time').reset_index(drop=True)
    Z_sec = group[Z_sector_cols].fillna(0).values
    if len(Z_sec) < 100: continue

    train_mask = group['time'] <= HMM_TRAIN_END
    Z_train = Z_sec[train_mask]
    Z_oos = Z_sec[~train_mask]
    if len(Z_train) < 50: 
        Z_train = Z_sec; Z_oos = np.array([])

    results, models = [], {}
    for K in [2, 3, 4]:
        m, bic, ll_oos, min_dur = evaluate_hmm_sector(K, Z_train, Z_oos)
        if m:
            results.append({'K': K, 'BIC': bic, 'll_oos': ll_oos, 'min_dur': min_dur})
            models[K] = m

    res_df = pd.DataFrame(results)
    if len(res_df) > 0:
        res_df['Rank_bic'] = res_df['BIC'].rank(ascending=True)
        res_df['Rank_oos'] = res_df['ll_oos'].rank(ascending=False)
        res_df['Rank_min_dur'] = res_df['min_dur'].rank(ascending=False)
        res_df['Composite'] = 0.3 * res_df['Rank_bic'] + 0.5 * res_df['Rank_oos'] + 0.2 * res_df['Rank_min_dur']
        res_df = res_df.sort_values('Composite')
        best_K = int(res_df.iloc[0]['K'])
        best_model = models[best_K]
    else:
        print(f"[-] Bá» qua {industry}: KhÃ´ng model nÃ o há»™i tá»¥.")
        continue 

    group['sector_regime'], probs = get_hmm_filtered_inference(best_model, Z_sec)

    stats = []
    for k in range(best_K):
        mask = group['sector_regime'] == k
        r = group.loc[mask, 'sector_log_ret'].mean() if mask.sum() > 0 else 0.0
        v = group.loc[mask, 'sector_vol20'].mean() if mask.sum() > 0 else 0.0
        stats.append({'state': k, 'mean_ret': r, 'mean_vol': v})

    labels = auto_label_sector(pd.DataFrame(stats), best_K)
    group['sector_regime_label'] = group['sector_regime'].map(labels)
    group['sector_best_K'] = best_K

    for k in range(best_K):
        semantic = labels[k]
        group[f'prob_sector_{semantic}'] = probs[:, k]
        all_semantic_labels.add(f'prob_sector_{semantic}')

    sector_results.append(group)
    print(f"[+] HoÃ n thÃ nh Sector HMM: {industry} (Tá»‘i Æ°u K={best_K})")

df_sector_hmm = pd.concat(sector_results, ignore_index=True)
for col in all_semantic_labels:
    if col not in df_sector_hmm.columns: df_sector_hmm[col] = 0.0
    else: df_sector_hmm[col] = df_sector_hmm[col].fillna(0.0)


# ## 8. Huáº¥n Luyá»‡n Ticker HMM Káº¿t Há»£p VÄ© MÃ´ & NgÃ nh
# Sá»­ dá»¥ng Ä‘áº·c trÆ°ng cá»§a Ticker káº¿t há»£p vá»›i xÃ¡c suáº¥t (prob) cá»§a Market HMM vÃ  Sector HMM Ä‘á»ƒ huáº¥n luyá»‡n mÃ´ hÃ¬nh riÃªng cho tá»«ng mÃ£. Äá»ƒ tá»‘i Æ°u thá»i gian, K Ä‘Æ°á»£c cá»‘ Ä‘á»‹nh = 3.

# In[ ]:


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

market_cols = [c for c in df_market.columns if 'Z' not in c and c not in ['time', 'Y_proxy']]
global_vars = df_market[['time'] + market_cols].copy()
global_vars['market_regime_label'] = df_market_res['market_regime_label']

print(f'Báº¯t Ä‘áº§u huáº¥n luyá»‡n vÃ  suy luáº­n cho {len(tickers)} mÃ£ Ticker...')
# Gá»™p xÃ¡c suáº¥t Market vÃ o global_vars Ä‘á»ƒ dÃ¹ng lÃ m feature cho Ticker
for k in range(K_market):
    prob_col = f'Market_Prob_{k}'
    if prob_col in df_market.columns:
        global_vars[prob_col] = df_market[prob_col]

ticker_dfs = []
from tqdm.auto import tqdm
for i, ticker in enumerate(tqdm(tickers, desc='Processing Tickers')):
    df_tick = df_m1[df_m1['ticker'] == ticker].copy().sort_values('time').reset_index(drop=True)
    df_tick['rolling_vol_5'] = df_tick['log_return'].rolling(5).std() * np.sqrt(252)
    df_tick['mom_1M'] = df_tick['close'].pct_change(20)
    df_tick['dist_MA50'] = df_tick['close'] / df_tick['close'].rolling(50).mean() - 1

    for col in ['volume_ratio', 'rolling_vol_20d', 'return_5d', 'return_20d']:
        if col not in df_tick.columns:
            if col == 'volume_ratio': df_tick['volume_ratio'] = df_tick['volume'] / df_tick['volume'].rolling(20).mean()
            elif col == 'rolling_vol_20d': df_tick['rolling_vol_20d'] = df_tick['log_return'].rolling(20).std() * np.sqrt(252)
            elif col == 'return_5d': df_tick['return_5d'] = df_tick['close'].pct_change(5)
            elif col == 'return_20d': df_tick['return_20d'] = df_tick['close'].pct_change(20)

    ticker_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'log_return', 'industry', 
                   'rolling_vol_20d', 'return_5d', 'return_20d', 'volume_ratio', 'rolling_vol_5', 'mom_1M', 'dist_MA50']

    cols_to_drop = [c for c in ticker_cols if c != 'time' and c in global_vars.columns]
    global_vars_clean = global_vars.drop(columns=cols_to_drop)
    ticker_aligned = global_vars_clean.merge(df_tick[ticker_cols], on='time', how='inner')

    # GhÃ©p xÃ¡c suáº¥t Sector vÃ o Ticker
    sector_cols_to_merge = ['time', 'industry', 'sector_regime', 'sector_regime_label'] + list(all_semantic_labels)
    ticker_aligned = ticker_aligned.merge(df_sector_hmm[sector_cols_to_merge], on=['time', 'industry'], how='left')
    ticker_aligned[list(all_semantic_labels)] = ticker_aligned[list(all_semantic_labels)].fillna(0)

    # Chá»n features káº¿t há»£p: GiÃ¡ trá»‹ Ticker + Market Prob + Sector Prob
    tick_specific_features = ['log_return', 'rolling_vol_20d', 'volume_ratio']
    market_prob_features = sorted([col for col in ticker_aligned.columns if col.startswith('Market_Prob_')])[:-1]
    macro_prob_features = sorted([col for col in ticker_aligned.columns if col.startswith('Macro_Prob_')])[:-1]
    sector_prob_features = sorted([c for c in all_semantic_labels if c in ticker_aligned.columns and ticker_aligned[c].std() > 1e-6])[:-1]

    hybrid_features = tick_specific_features + macro_prob_features + market_prob_features + sector_prob_features

    fd_z_tick, Z_all_tick = make_Z_ticker(ticker_aligned, hybrid_features, window=252)

    if len(Z_all_tick) < 100:
        print(f"[-] Bá» qua {ticker}: Dá»¯ liá»‡u quÃ¡ ngáº¯n")
        continue

    # Huáº¥n luyá»‡n Ticker HMM riÃªng biá»‡t (K=3)
    K_tick = 3
    ticker_hmm = GMMHMM(n_components=K_tick, n_mix=2, covariance_type='diag', min_covar=0.01, n_iter=100, random_state=42)

    try:
        ticker_hmm.fit(Z_all_tick)
        ticker_daily_states, ticker_daily_probs = get_hmm_filtered_inference(ticker_hmm, Z_all_tick)

        # Tá»± Ä‘á»™ng gÃ¡n nhÃ£n Ticker Regime (Bull/Bear/Sideways) dá»±a trÃªn Return & Vol cá»§a Ticker
        stats = []
        for k in range(K_tick):
            mask = ticker_daily_states == k
            r = ticker_aligned.loc[mask, 'log_return'].mean()
            v = ticker_aligned.loc[mask, 'rolling_vol_20d'].mean()
            stats.append({'state': k, 'mean_ret': r, 'mean_vol': v})

        ticker_labels_map = auto_label_sector(pd.DataFrame(stats), K_tick)
        ticker_daily_labels = pd.Series(ticker_daily_states).map(ticker_labels_map).values

    except Exception as e:
        print(f"Lá»—i khi train/predict mÃ£ {ticker}: {e}")
        ticker_daily_states = np.zeros(len(Z_all_tick), dtype=int)
        ticker_daily_probs = np.zeros((len(Z_all_tick), K_tick))
        ticker_daily_probs[:, 0] = 1.0
        ticker_daily_labels = np.array(['Unknown'] * len(Z_all_tick))

    df_tick_daily_res = pd.DataFrame({
        'time': fd_z_tick['time'].values,
        'ticker_regime': ticker_daily_states,
        'ticker_regime_label': ticker_daily_labels,
    })
    for k in range(K_tick):
        df_tick_daily_res[f'prob_ticker_{k}'] = ticker_daily_probs[:, k]

    state_cols = ['ticker_regime', 'ticker_regime_label'] + [f'prob_ticker_{k}' for k in range(K_tick)]
    ticker_master = ticker_aligned.merge(df_tick_daily_res[['time'] + state_cols], on='time', how='inner')
    ticker_master['ticker'] = ticker

    ticker_dfs.append(ticker_master)

master_ticker = pd.concat(ticker_dfs, ignore_index=True)
master_ticker = master_ticker.dropna(subset=['close']).reset_index(drop=True)
cols_reordered = ['time', 'ticker'] + [col for col in master_ticker.columns if col not in ['time', 'ticker']]
master_ticker = master_ticker[cols_reordered]
print(f'HoÃ n thÃ nh huáº¥n luyá»‡n Ticker HMM. KÃ­ch thÆ°á»›c master_ticker: {master_ticker.shape}')


# ## 9. Meta-Classifier (LightGBM) Dá»± BÃ¡o Cá»• Phiáº¿u T+1

# ### 9.1 Cháº¿ Ä‘á»™ Backtest (Walk-Forward Validation)
# **Má»¥c Ä‘Ã­ch:** ÄÃ¡nh giÃ¡ hiá»‡u suáº¥t cá»§a thuáº­t toÃ¡n trong quÃ¡ khá»© má»™t cÃ¡ch trung thá»±c nháº¥t, loáº¡i bá» hoÃ n toÃ n Look-ahead Bias.
# **CÃ¡ch hoáº¡t Ä‘á»™ng:** DÃ¹ng vÃ²ng láº·p quÃ©t qua tá»«ng ngÃ y, train báº±ng quÃ¡ khá»© vÃ  dá»± Ä‘oÃ¡n ngÃ y hiá»‡n táº¡i. Tá»‘n thá»i gian cháº¡y (chá»‰ nÃªn cháº¡y 1 láº§n khi cáº§n láº¥y report).

# In[ ]:


# 1. Táº¡o nhÃ£n dá»± bÃ¡o (Target): Dá»± bÃ¡o return_1d (T+1)
master_ticker['target_return_1d'] = master_ticker.groupby('ticker')['close'].pct_change(1).shift(-1)
master_ticker['target_bin'] = (master_ticker['target_return_1d'] > 0).astype(int)

# Bá» Ä‘i nhá»¯ng dÃ²ng khÃ´ng cÃ³ target (ngÃ y cuá»‘i cÃ¹ng cá»§a dá»¯ liá»‡u chÆ°a biáº¿t tÆ°Æ¡ng lai)
# LÆ¯U Ã: Vá»›i Live Trading, ta VáºªN Cáº¦N giá»¯ láº¡i ngÃ y cuá»‘i cÃ¹ng dÃ¹ target lÃ  NaN Ä‘á»ƒ dá»± bÃ¡o. 
# NÃªn ta sáº½ copy ra má»™t df riÃªng cho backtest.
df_backtest = master_ticker.dropna(subset=['target_return_1d']).reset_index(drop=True)

# Lá»c cÃ¡c features

semantic_sector_probs = [col for col in df_backtest.columns if col.startswith('prob_sector_')]
market_probs = [col for col in df_backtest.columns if col.startswith('Market_Prob_')]
ticker_probs = [col for col in df_backtest.columns if col.startswith('prob_ticker_')]
feature_cols = market_probs + semantic_sector_probs + ticker_probs + ['rolling_vol_20d', 'return_5d', 'volume_ratio']

# 2. Walk-Forward
start_test_date = pd.Timestamp('2022-01-01')
test_dates = sorted(df_backtest[df_backtest['time'] >= start_test_date]['time'].unique())

print(f"Báº¯t Ä‘áº§u Walk-Forward Training cho {len(test_dates)} ngÃ y giao dá»‹ch...")
df_backtest['final_meta_pred_prob'] = np.nan

from tqdm.notebook import tqdm
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

for i, current_date in enumerate(tqdm(test_dates, desc="Walk-Forward Daily Train")):
    train_mask = df_backtest['time'] < current_date
    X_train = df_backtest.loc[train_mask, feature_cols]
    y_train = df_backtest.loc[train_mask, 'target_bin']

    test_mask = df_backtest['time'] == current_date
    X_test = df_backtest.loc[test_mask, feature_cols]

    if len(X_train) < 1000 or len(X_test) == 0: continue

    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    df_backtest.loc[test_mask, 'final_meta_pred_prob'] = probs

# 3. ÄÃ¡nh giÃ¡
test_mask_all = df_backtest['time'] >= start_test_date
y_test_all = df_backtest.loc[test_mask_all, 'target_bin']
probs_all = df_backtest.loc[test_mask_all, 'final_meta_pred_prob']

valid_idx = probs_all.notna()
y_test_all = y_test_all[valid_idx]
probs_all = probs_all[valid_idx]
preds_all = (probs_all > 0.5).astype(int)

from sklearn.metrics import classification_report, roc_auc_score
print("\n--- BÃO CÃO PHÃ‚N LOáº I WALK-FORWARD (RETURN_1D) ---")
print(classification_report(y_test_all, preds_all))
print(f"ROC-AUC Score (Daily Walk-Forward): {roc_auc_score(y_test_all, probs_all):.4f}")

# Cáº­p nháº­t káº¿t quáº£ backtest vÃ o master_ticker
master_ticker = master_ticker.merge(df_backtest[['time', 'ticker', 'final_meta_pred_prob']], on=['time', 'ticker'], how='left')


# ### 9.2 Cháº¿ Ä‘á»™ Live Trading (Dá»± bÃ¡o ngÃ y T+1)
# **Má»¥c Ä‘Ã­ch:** DÃ¹ng Ä‘á»ƒ ra quyáº¿t Ä‘á»‹nh mua/bÃ¡n háº±ng ngÃ y.
# **CÃ¡ch hoáº¡t Ä‘á»™ng:** Chá»‰ láº¥y ngÃ y giao dá»‹ch cuá»‘i cÃ¹ng trong file dá»¯ liá»‡u lÃ m táº­p Test, toÃ n bá»™ pháº§n lá»‹ch sá»­ lÃ m Train. Cháº¡y cá»±c nhanh (1 giÃ¢y). Bá» qua hoÃ n toÃ n dá»¯ liá»‡u má»¥c tiÃªu (target) cá»§a ngÃ y cuá»‘i vÃ¬ nÃ³ chÆ°a xáº£y ra.

# In[ ]:


import lightgbm as lgb

print(f"=== CHáº¾ Äá»˜ LIVE TRADING ===")
# XÃ¡c Ä‘á»‹nh ngÃ y cuá»‘i cÃ¹ng cÃ³ trong dá»¯ liá»‡u
latest_date = master_ticker['time'].max()
print(f"NgÃ y giao dá»‹ch má»›i nháº¥t (T): {latest_date.strftime('%Y-%m-%d')}")

# Táº­p Train: Táº¥t cáº£ dá»¯ liá»‡u trÆ°á»›c ngÃ y T (pháº£i loáº¡i bá» cÃ¡c dÃ²ng NaN á»Ÿ target)
train_mask = (master_ticker['time'] < latest_date) & (master_ticker['target_return_1d'].notna())
X_train_live = master_ticker.loc[train_mask, feature_cols]
y_train_live = master_ticker.loc[train_mask, 'target_bin']

# Táº­p Test: Duy nháº¥t dá»¯ liá»‡u cá»§a ngÃ y T
test_mask = master_ticker['time'] == latest_date
X_test_live = master_ticker.loc[test_mask, feature_cols]

print(f"Äang huáº¥n luyá»‡n mÃ´ hÃ¬nh Live trÃªn {len(X_train_live)} Ä‘iá»ƒm dá»¯ liá»‡u lá»‹ch sá»­...")
clf_live = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1, class_weight='balanced')
clf_live.fit(X_train_live, y_train_live)

print(f"Äang dá»± bÃ¡o xÃ¡c suáº¥t tÄƒng giÃ¡ cho phiÃªn ngÃ y mai (T+1)...")
probs_live = clf_live.predict_proba(X_test_live)[:, 1]

# GÃ¡n káº¿t quáº£ vÃ o master_ticker
master_ticker.loc[test_mask, 'final_meta_pred_prob'] = probs_live

# Báº£ng xáº¿p háº¡ng tÃ­n hiá»‡u
live_results = master_ticker.loc[test_mask, ['time', 'ticker', 'industry', 'close', 'final_meta_pred_prob']].copy()
live_results['TÃ­n Hiá»‡u'] = live_results['final_meta_pred_prob'].apply(lambda x: 'TÄƒng (KhuyÃªn Mua)' if x > 0.5 else 'Giáº£m (Cáº£nh BÃ¡o)')
live_results = live_results.sort_values('final_meta_pred_prob', ascending=False).reset_index(drop=True)

print("\nðŸ† TOP MÃƒ Cá»” PHIáº¾U TIá»€M NÄ‚NG NHáº¤T CHO T+1:")
display(live_results.head(10))


# ### 9.3 Thá»‘ng kÃª Hiá»‡u suáº¥t TÃ i chÃ­nh (Financial Backtest)
# **Má»¥c Ä‘Ã­ch:** MÃ´ phá»ng giao dá»‹ch thá»±c táº¿ trÃªn táº­p Walk-Forward OOS. So sÃ¡nh Lá»£i nhuáº­n tÃ­ch lÅ©y (Equity Curve) vÃ  Sharpe Ratio cá»§a Chiáº¿n lÆ°á»£c so vá»›i viá»‡c Mua & Náº¯m giá»¯ toÃ n bá»™ rá»• cá»• phiáº¿u.
# **Quy táº¯c giáº£ láº­p:** 
# - Vá»‘n Ä‘Æ°á»£c chia Ä‘á»u cho táº¥t cáº£ cÃ¡c mÃ£ (VÃ­ dá»¥ rá»• cÃ³ 46 mÃ£, má»—i mÃ£ chiáº¿m 1/46 vá»‘n).
# - NgÃ y T: Náº¿u Meta-Classifier dá»± bÃ¡o `prob > 0.5`, ta mua/náº¯m giá»¯ mÃ£ Ä‘Ã³ á»Ÿ ngÃ y T+1. Náº¿u `prob <= 0.5`, bÃ¡n ra giá»¯ tiá»n máº·t (lá»£i suáº¥t = 0).

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if 'df_backtest' in locals() and 'final_meta_pred_prob' in df_backtest.columns:
    df_bt = df_backtest.dropna(subset=['final_meta_pred_prob']).copy()
    N_stocks = df_bt['ticker'].nunique()

    # 1. Chiáº¿n lÆ°á»£c 1: Mua ráº£i Ä‘á»u (AI Equal Weight)
    # Lá»‡nh mua Äƒn target_return_1d, lá»‡nh Ä‘á»©ng ngoÃ i Äƒn 0%
    df_bt['signal_eq'] = (df_bt['final_meta_pred_prob'] > 0.4).astype(int)
    port_eq = df_bt.groupby('time').apply(
        lambda x: (x['target_return_1d'] * x['signal_eq']).sum() / N_stocks
    ).reset_index(name='ret_eq')

    # 2. Chiáº¿n lÆ°á»£c 2: Táº­p trung Top 5 (AI Top 5 Conviction)
    # Má»—i ngÃ y chá»n ra 5 mÃ£ cÃ³ xÃ¡c suáº¥t cao nháº¥t. Náº¿u xÃ¡c suáº¥t > 0.5 thÃ¬ mua (dÃ nh 20% vá»‘n).
    def top_k_return(day_data, k=5):
        # Sáº¯p xáº¿p tá»« cao xuá»‘ng tháº¥p
        top_k = day_data.sort_values('final_meta_pred_prob', ascending=False).head(k)
        # Chá»‰ mua nhá»¯ng mÃ£ > 0.5
        top_k_buy = top_k[top_k['final_meta_pred_prob'] > 0.4]
        if len(top_k_buy) == 0:
            return 0.0
        # TÃ­nh trung bÃ¬nh lá»£i nhuáº­n cá»§a k mÃ£ (vá»‘n chia Ä‘á»u k pháº§n)
        return top_k_buy['target_return_1d'].sum() / k

    port_top5 = df_bt.groupby('time').apply(lambda x: top_k_return(x, k=5)).reset_index(name='ret_top5')

    # 3. Chiáº¿n lÆ°á»£c Benchmark (Buy & Hold toÃ n bá»™ rá»•)
    bench = df_bt.groupby('time').apply(
        lambda x: x['target_return_1d'].sum() / N_stocks
    ).reset_index(name='ret_bench')

    # Gá»™p káº¿t quáº£
    df_perf = pd.merge(port_eq, port_top5, on='time')
    df_perf = pd.merge(df_perf, bench, on='time')

    df_perf['cum_eq'] = (1 + df_perf['ret_eq']).cumprod()
    df_perf['cum_top5'] = (1 + df_perf['ret_top5']).cumprod()
    df_perf['cum_bench'] = (1 + df_perf['ret_bench']).cumprod()

    # HÃ m tÃ­nh Metrics
    def calc_metrics(returns):
        cum = (1 + returns).cumprod()
        ann_ret = (cum.iloc[-1] ** (252 / len(returns))) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        roll_max = cum.cummax()
        drawdown = cum / roll_max - 1
        max_dd = drawdown.min()
        win_rate = (returns > 0).mean()
        return ann_ret, ann_vol, sharpe, max_dd, win_rate

    metrics_eq = calc_metrics(df_perf['ret_eq'])
    metrics_top5 = calc_metrics(df_perf['ret_top5'])
    metrics_bench = calc_metrics(df_perf['ret_bench'])

    # Hiá»ƒn thá»‹ Báº£ng Thá»‘ng KÃª
    perf_table = pd.DataFrame({
        'AI Táº­p Trung (Top 5)': [f"{metrics_top5[0]*100:.1f}%", f"{metrics_top5[1]*100:.1f}%", f"{metrics_top5[2]:.2f}", f"{metrics_top5[3]*100:.1f}%", f"{metrics_top5[4]*100:.1f}%"],
        'AI Ráº£i Äá»u (Equal)': [f"{metrics_eq[0]*100:.1f}%", f"{metrics_eq[1]*100:.1f}%", f"{metrics_eq[2]:.2f}", f"{metrics_eq[3]*100:.1f}%", f"{metrics_eq[4]*100:.1f}%"],
        'Buy & Hold Benchmark': [f"{metrics_bench[0]*100:.1f}%", f"{metrics_bench[1]*100:.1f}%", f"{metrics_bench[2]:.2f}", f"{metrics_bench[3]*100:.1f}%", f"{metrics_bench[4]*100:.1f}%"]
    }, index=['Lá»£i nhuáº­n nÄƒm (Ann. Ret)', 'Biáº¿n Ä‘á»™ng nÄƒm (Ann. Vol)', 'Sharpe Ratio', 'Max Drawdown', 'Tá»· lá»‡ ngÃ y lÃ£i (Win Rate)'])

    print("\nðŸ“Š Báº¢NG THá»NG KÃŠ HIá»†U SUáº¤T GIAO Dá»ŠCH Vá»šI NHIá»€U CHIáº¾N THUáº¬T (OOS)")
    display(perf_table)

    # Váº½ Equity Curve
    plt.figure(figsize=(14, 7))
    plt.plot(df_perf['time'], df_perf['cum_top5'], label=f"AI Táº­p Trung Top 5 (Sharpe: {metrics_top5[2]:.2f})", color='#f39c12', linewidth=2.5)
    plt.plot(df_perf['time'], df_perf['cum_eq'], label=f"AI Ráº£i Äá»u (Sharpe: {metrics_eq[2]:.2f})", color='#e74c3c', linewidth=2)
    plt.plot(df_perf['time'], df_perf['cum_bench'], label=f"Thá»‹ trÆ°á»ng (Sharpe: {metrics_bench[2]:.2f})", color='#95a5a6', linewidth=1.5, alpha=0.8)

    plt.title("ðŸ“ˆ ÄÆ°á»ng cong Lá»£i nhuáº­n (Equity Curve) - Äa Chiáº¿n Thuáº­t", fontsize=16, fontweight='bold')
    plt.ylabel("TÃ i khoáº£n (Tá»‰ lá»‡)", fontsize=12)
    plt.xlabel("Thá»i gian", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("Vui lÃ²ng cháº¡y Ã” 9.1 (Walk-Forward Backtest) trÆ°á»›c Ä‘á»ƒ cÃ³ dá»¯ liá»‡u váº½ biá»ƒu Ä‘á»“!")


# ## 10. LÆ°u Káº¿t Quáº£ & Trá»±c Quan HÃ³a TÆ°Æ¡ng TÃ¡c

# In[ ]:


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. LÆ°u Market Regimes
if 'global_vars' in globals():
    global_vars.to_parquet(OUTPUT_DIR / 'market_regimes.parquet', index=False)
    global_vars.to_csv(OUTPUT_DIR / 'market_regimes.csv', index=False)
    print(f"ÄÃ£ lÆ°u: market_regimes (.parquet & .csv) {global_vars.shape}")

# 2. LÆ°u Sector Regimes
if 'df_sector_hmm' in globals():
    df_sector_hmm.to_parquet(OUTPUT_DIR / 'sector_regimes.parquet', index=False)
    df_sector_hmm.to_csv(OUTPUT_DIR / 'sector_regimes.csv', index=False)
    print(f"ÄÃ£ lÆ°u: sector_regimes (.parquet & .csv){df_sector_hmm.shape}")

# 3. LÆ°u Ticker Regimes (Master File)
master_ticker.to_parquet(OUTPUT_DIR / 'master_drl_ready_full.parquet', index=False)
master_ticker.to_csv(OUTPUT_DIR / 'master_drl_ready_full.csv', index=False)
print("ÄÃ£ lÆ°u: master_drl_ready_full (.parquet & .csv)")
print("\nHoÃ n táº¥t quÃ¡ trÃ¬nh lÆ°u káº¿t quáº£!")

import ipywidgets as widgets
from ipywidgets import interactive_output
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df_plot = master_ticker.copy()
if 'df_market_res' in globals() and 'market_regime_label' in df_market_res.columns:
    df_market_true = df_market_res[['time', 'market_regime_label']].rename(columns={'market_regime_label': 'true_market_regime'})
    df_plot = df_plot.merge(df_market_true, on='time', how='left')
else:
    df_plot['true_market_regime'] = df_plot.get('market_regime_label', df_plot.get('true_market_regime', 'Unknown'))

if 'market_regime_label' in df_plot.columns:
    df_plot.rename(columns={'market_regime_label': 'ticker_regime'}, inplace=True)

SEMANTIC_COLORS = {
    'Bull': '#2ecc71',
    'CalmBull': '#2ecc71',
    'Bear': '#e74c3c',
    'Crisis': '#e74c3c',
    'Euphoria': '#f39c12',
    'Sideways': '#95a5a6',
    'Tranquil': '#95a5a6',
    'State_0': '#2ecc71',
    'State_1': '#e74c3c',
    'State_2': '#95a5a6',
    'Tier0': '#e74c3c',
    'Tier1': '#95a5a6',
    'Tier2': '#2ecc71',
    'Unknown': '#e0e0e0'
}

def get_color(label):
    return SEMANTIC_COLORS.get(label, '#e0e0e0')

def plot_ticker_regimes(ticker, date_range):
    sub_df = df_plot[
        (df_plot['ticker'] == ticker) &
        (df_plot['time'] >= date_range[0]) &
        (df_plot['time'] <= date_range[1])
    ].copy().sort_values('time').reset_index(drop=True)

    if len(sub_df) == 0:
        print("KhÃ´ng cÃ³ dá»¯ liá»‡u cho khoáº£ng thá»i gian nÃ y.")
        return

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 18), sharex=True, gridspec_kw={'height_ratios': [3, 3, 3, 1.5, 1.5]})
    times = sub_df['time']

    # 1. Market Regime vs VN-Index Close
    ax1.set_title(f"MÃ£: {ticker} | NgÃ nh: {sub_df['industry'].iloc[0]}", fontsize=16, fontweight='bold')
    ax1.set_ylabel("VN-Index (Market)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    m_regimes = sub_df['true_market_regime']
    for i in range(1, len(sub_df)):
        if pd.notna(m_regimes.iloc[i]):
            c = get_color(m_regimes.iloc[i])
            ax1.plot([times.iloc[i-1], times.iloc[i]], [sub_df['vnindex_close'].iloc[i-1], sub_df['vnindex_close'].iloc[i]], color=c, linewidth=1.5)

    present_market = sorted(m_regimes.dropna().unique())
    legend_elements_1 = [Patch(facecolor=get_color(reg), edgecolor='none', label=f'Market: {reg}') for reg in present_market]
    ax1.legend(handles=legend_elements_1, loc='upper left', bbox_to_anchor=(1.01, 1))

    # 2. Sector Regime vs Ticker Close
    ax2.set_ylabel("GiÃ¡ NgÃ nh (Sector)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    s_regimes = sub_df['sector_regime_label']
    for i in range(1, len(sub_df)):
        if pd.notna(s_regimes.iloc[i]):
            c = get_color(s_regimes.iloc[i])
            ax2.plot([times.iloc[i-1], times.iloc[i]], [sub_df['close'].iloc[i-1], sub_df['close'].iloc[i]], color=c, linewidth=1.5)

    present_sector = sorted(s_regimes.dropna().unique())
    legend_elements_2 = [Patch(facecolor=get_color(reg), edgecolor='none', label=f'Sector: {reg}') for reg in present_sector]
    ax2.legend(handles=legend_elements_2, loc='upper left', bbox_to_anchor=(1.01, 1))

    # 3. Ticker Regime vs Ticker Close
    ax3.set_ylabel("GiÃ¡ MÃ£ (Ticker)", fontsize=12)
    ax3.grid(True, alpha=0.3)

    t_regimes = sub_df['ticker_regime_label']
    for i in range(1, len(sub_df)):
        if pd.notna(t_regimes.iloc[i]):
            c = get_color(t_regimes.iloc[i])
            ax3.plot([times.iloc[i-1], times.iloc[i]], [sub_df['close'].iloc[i-1], sub_df['close'].iloc[i]], color=c, linewidth=1.5)

    present_ticker = sorted(t_regimes.dropna().unique())
    legend_elements_3 = [Patch(facecolor=get_color(reg), edgecolor='none', label=f'Ticker: {reg}') for reg in present_ticker]
    ax3.legend(handles=legend_elements_3, loc='upper left', bbox_to_anchor=(1.01, 1))

    # 4. Volume
    ax4.bar(times, sub_df['volume'], color='grey', alpha=0.6, label="Volume")
    ax4.set_ylabel("Khá»‘i LÆ°á»£ng", fontsize=12)
    ax4.grid(True, alpha=0.3)

    # 5. Meta-Classifier Prob
    if 'final_meta_pred_prob' in sub_df.columns:
        ax5.plot(times, sub_df['final_meta_pred_prob'], color='blue', linewidth=1.5, label="XÃ¡c suáº¥t tÄƒng T+1 (Meta)")
        ax5.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax5.fill_between(times, sub_df['final_meta_pred_prob'], 0.5, where=(sub_df['final_meta_pred_prob'] > 0.5), color='#2ecc71', alpha=0.3)
        ax5.fill_between(times, sub_df['final_meta_pred_prob'], 0.5, where=(sub_df['final_meta_pred_prob'] <= 0.5), color='#e74c3c', alpha=0.3)
        ax5.set_ylabel("XÃ¡c suáº¥t TÄƒng", fontsize=12)
        ax5.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax5.set_ylim(0, 1)

    ax5.set_xlabel("Thá»i Gian", fontsize=12)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

ticker_dropdown = widgets.Dropdown(options=sorted(df_plot['ticker'].unique()), value=df_plot['ticker'].iloc[0], description='MÃ£ Ticker:')
dates = sorted(df_plot['time'].unique())
date_slider = widgets.SelectionRangeSlider(options=dates, index=(0, len(dates)-1), description='Khoáº£ng Äo:', orientation='horizontal', layout={'width': '80%'})

ui = widgets.VBox([ticker_dropdown, date_slider])
out = interactive_output(plot_ticker_regimes, {'ticker': ticker_dropdown, 'date_range': date_slider})

display(ui, out)


# ## 11. Hiá»ƒn Thá»‹ Biá»ƒu Äá»“ TÄ©nh (MÃ£ BID)

# In[ ]:


ticker_to_plot = 'BID'
date_range = (df_plot['time'].min(), df_plot['time'].max())

print(f"Äang váº½ biá»ƒu Ä‘á»“ tÄ©nh cho {ticker_to_plot}...")
plot_ticker_regimes(ticker_to_plot, date_range)


# In[ ]:


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

artifact_dir = r'C:\Users\ADMIN\.gemini\antigravity-cli\brain\12658c65-0507-48c7-a2d2-82b401dc1a40'
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)

img_path = os.path.join(artifact_dir, 'roc_curve.png')
plt.savefig(img_path, dpi=150, bbox_inches='tight')
print(f"ROC curve saved to {img_path}")
plt.show()

