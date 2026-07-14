import os
import numpy as np
import pandas as pd
import warnings
import pickle
from pathlib import Path
from scipy.stats import norm
from hmmlearn.hmm import GMMHMM, GaussianHMM

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- THIẾT LẬP THƯ MỤC ---
# Giả sử chạy từ thư mục Kaggle (hoặc thư mục chứa thư mục data/output)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Thư mục 'live'
PROJECT_DIR = os.path.dirname(BASE_DIR) # Thư mục 'Kaggle'
OUTPUT_DIR = Path(os.path.join(PROJECT_DIR, 'live', 'output'))
MODEL_DIR = OUTPUT_DIR / 'models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
print(f"Thư mục đầu ra được thiết lập tại: {OUTPUT_DIR.resolve()}")

# --- CẤU HÌNH ---
HMM_TRAIN_END = pd.Timestamp('2019-12-31')
K_MACRO_BEST = 2
K_MARKET_BEST = 3  # Thường cố định K=3 hoặc 4 cho thị trường chung
K_SECTOR_BEST = 3  # Cố định K=3 cho ngành để chạy live nhanh
K_TICKER_BEST = 3  # Cố định K=3 cho cổ phiếu

# Cấu hình biến đặc trưng
MACRO_FEATURES = ['cpi_mom', 'credit_growth_mom', 'pmi_vn']
MARKET_FEATURES = ['rolling_vol_5', 'volume_ratio'] 
Z_SECTOR_COLS = ['sector_log_ret_Z', 'sector_vol20_Z', 'sector_vol5_Z', 'sector_volume_ratio_Z']

def get_hmm_filtered_inference(model, Z):
    """
    Trong live inference, ta dùng mô hình đã train (pkl) để predict trực tiếp trên dữ liệu mới.
    Không cần dùng vòng lặp walk-forward t=1..N như khi backtest.
    """
    try:
        filtered_probs = model.predict_proba(Z)
        filtered_regimes = model.predict(Z)
    except Exception:
        K = model.n_components
        filtered_probs = np.ones((len(Z), K)) / K
        filtered_regimes = np.zeros(len(Z), dtype=int)
    return filtered_regimes, filtered_probs

def make_nqt(series, window=252):
    rolling_rank = series.rolling(window=window, min_periods=1).rank()
    rolling_count = series.rolling(window=window, min_periods=1).count()
    pct = (rolling_rank - 0.5) / rolling_count
    return np.clip(norm.ppf(pct), -3.0, 3.0)

def make_Z_ticker(df_source, features, window=252):
    fd = df_source[['time'] + features].dropna().reset_index(drop=True)
    nqt_df = pd.DataFrame(index=fd.index)
    for col in features:
        rolling_rank = fd[col].expanding(min_periods=1).rank()
        rolling_count = fd[col].expanding(min_periods=1).count()
        pct = (rolling_rank - 0.5) / rolling_count
        nqt_values = norm.ppf(pct)
        nqt_df[col] = np.clip(nqt_values, -3.0, 3.0)
    Z_all = nqt_df.values
    return fd, Z_all

# Hàm tự động gán nhãn
def auto_label_macro(rs_macro, K):
    pmi = rs_macro['mean_pmi'].values
    order = np.argsort(pmi)
    if K == 2:
        return {int(order[0]): 'Macro_Stagnant', int(order[1]): 'Macro_Expansion'}
    return {int(order[i]): f'Macro_Tier{i+1}' for i in range(K)}

def auto_label_market(rs, K):
    ret = rs['mean_ret_%'].values
    vol = rs['vol_%'].values
    if K == 3:
        sharpe = ret / (vol + 1e-9)
        order = np.argsort(sharpe)
        return {int(order[0]): 'Bear', int(order[1]): 'Sideways', int(order[2]): 'Bull'}
    return {i: f'State_{i}' for i in range(K)}

def auto_label_sector(rs, K):
    ret = rs['mean_ret'].values; vol = rs['mean_vol'].values
    if K == 3:
        sharpe = ret / (vol + 1e-9)
        order = np.argsort(sharpe)
        return {int(order[0]): 'Bear', int(order[1]): 'Sideways', int(order[2]): 'Bull'}
    return {i: f'State_{i}' for i in range(K)}

def main():
    # ---------------------------------------------------------
    # 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
    # ---------------------------------------------------------
    print("Đang tải dữ liệu...")
    df_daily_base = pd.read_csv(os.path.join(PROJECT_DIR, 'output', 'hmm_data.csv'))
    df_daily_base['time'] = pd.to_datetime(df_daily_base['time'])

    df_m1 = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'm1_vn46.csv'))
    df_m1['time'] = pd.to_datetime(df_m1['time']).dt.normalize()

    # Thêm thông tin ngành
    _ind_df = pd.read_csv(os.path.join(PROJECT_DIR, 'src', 'data_collection', 'industries.csv'))
    _ind_df = _ind_df[_ind_df['icb_level'] == 1]
    industry_mapping = dict(zip(_ind_df['symbol'], _ind_df['icb_name']))
    df_m1['industry'] = df_m1['ticker'].map(industry_mapping)

    # Market Proxy
    market_ret = df_m1.groupby('time')['log_return'].mean().reset_index()
    market_ret.columns = ['time', 'vnindex_log_ret']
    market_close = df_m1.groupby('time')['close'].mean().reset_index()
    market_close.columns = ['time', 'vnindex_close']
    market_vol = df_m1.groupby('time')['volume'].sum().reset_index()
    market_vol.columns = ['time', 'vnindex_volume']

    df_market = df_daily_base.merge(market_ret, on='time', how='left')
    df_market = df_market.merge(market_close, on='time', how='left')
    df_market = df_market.merge(market_vol, on='time', how='left')
    df_market = df_market.dropna(subset=['vnindex_log_ret', 'vnindex_close']).reset_index(drop=True)
    df_market['vnindex_vol20'] = df_market['vnindex_log_ret'].rolling(20).std() * np.sqrt(252)
    df_market['rolling_vol_5'] = df_market['vnindex_log_ret'].rolling(5).std() * np.sqrt(252)
    df_market['volume_ratio'] = df_market['vnindex_volume'] / df_market['vnindex_volume'].rolling(20).mean()

    # Lấy các biến vĩ mô
    try:
        df_fnb = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'm4_foreign_net_buy_sell.csv'))
        df_fnb['time'] = pd.to_datetime(df_fnb['time'])
        df_market = df_market.merge(df_fnb[['time', 'fnb_ratio']], on='time', how='left')
    except: pass

    df_market = df_market.dropna().reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 2. MACRO HMM (BƯỚC 1 - MONTHLY)
    # ---------------------------------------------------------
    print("--- CHẠY MACRO HMM ---")
    df_market['year_month'] = df_market['time'].dt.to_period('M')
    df_monthly = df_market.groupby('year_month').last().reset_index()
    df_monthly[MACRO_FEATURES] = df_monthly[MACRO_FEATURES].shift(1) # Chống Lookahead bias
    df_monthly = df_monthly.dropna().reset_index(drop=True)
    
    Z_macro_all = df_monthly[MACRO_FEATURES].values
    macro_train_mask = df_monthly['time'] <= HMM_TRAIN_END
    Z_macro_train = Z_macro_all[macro_train_mask]

    model_macro = GaussianHMM(n_components=K_MACRO_BEST, covariance_type='full', random_state=RANDOM_STATE, n_iter=200)
    macro_model_path = MODEL_DIR / 'macro_model.pkl'
    if macro_model_path.exists():
        with open(macro_model_path, 'rb') as f:
            model_macro = pickle.load(f)
    else:
        model_macro.fit(Z_macro_train)
        with open(macro_model_path, 'wb') as f:
            pickle.dump(model_macro, f)

    macro_states, macro_probs = get_hmm_filtered_inference(model_macro, Z_macro_all)
    
    stats_macro = []
    for k in range(K_MACRO_BEST):
        mask = macro_states == k
        stats_macro.append({
            'state': k, 'n_months': int(mask.sum()),
            'mean_pmi': df_monthly.loc[mask, 'pmi_vn'].mean(),
            'mean_cpi': df_monthly.loc[mask, 'cpi_mom'].mean()
        })
    df_sm = pd.DataFrame(stats_macro)
    STATE_TO_LABEL_MACRO = auto_label_macro(df_sm, K_MACRO_BEST)

    for i in range(K_MACRO_BEST):
        df_monthly[f'Macro_Prob_{i}'] = macro_probs[:, i]

    df_monthly_shifted = df_monthly.copy()
    df_monthly_shifted['year_month'] = df_monthly_shifted['year_month'] + 1
    prob_cols = [f'Macro_Prob_{i}' for i in range(K_MACRO_BEST)]
    df_market = df_market.merge(df_monthly_shifted[['year_month'] + prob_cols], on='year_month', how='left')
    df_market[prob_cols] = df_market[prob_cols].ffill().fillna(0)

    # ---------------------------------------------------------
    # 3. MARKET HMM (BƯỚC 2 - DAILY)
    # ---------------------------------------------------------
    print("--- CHẠY MARKET HMM ---")
    # Normalize Z cho Market
    for f in MARKET_FEATURES:
        df_market[f'{f}_Z'] = make_nqt(df_market[f])
    
    market_z_features = [f'{f}_Z' for f in MARKET_FEATURES] + prob_cols[:-1]
    Z_market_all = df_market[market_z_features].fillna(0).values
    market_train_mask = df_market['time'] <= HMM_TRAIN_END
    Z_market_train = Z_market_all[market_train_mask]

    model_market = GMMHMM(n_components=K_MARKET_BEST, n_mix=2, covariance_type='diag', min_covar=0.01, n_iter=200, random_state=RANDOM_STATE)
    market_model_path = MODEL_DIR / 'daily_market_model.pkl'
    if market_model_path.exists():
        with open(market_model_path, 'rb') as f:
            model_market = pickle.load(f)
    else:
        model_market.fit(Z_market_train)
        with open(market_model_path, 'wb') as f:
            pickle.dump(model_market, f)

    market_states, market_probs = get_hmm_filtered_inference(model_market, Z_market_all)
    
    df_market_res = df_market[['time']].copy()
    df_market_res['market_regime'] = market_states
    stats_market = []
    for k in range(K_MARKET_BEST):
        mask = df_market_res['market_regime'] == k
        ret_k = df_market.loc[mask, 'vnindex_log_ret'].mean() * 100 if mask.sum() > 0 else 0
        vol_k = df_market.loc[mask, 'vnindex_vol20'].mean() * 100 if mask.sum() > 0 else 0
        stats_market.append({'state': k, 'mean_ret_%': ret_k, 'vol_%': vol_k})
    STATE_TO_LABEL_MARKET = auto_label_market(pd.DataFrame(stats_market), K_MARKET_BEST)
    
    for i in range(K_MARKET_BEST):
        df_market[f'Market_Prob_{i}'] = market_probs[:, i]

    # ---------------------------------------------------------
    # 4. SECTOR HMM (BƯỚC 3 - NGÀNH)
    # ---------------------------------------------------------
    print("--- CHẠY SECTOR HMM ---")
    sector_df = df_m1.groupby(['industry', 'time']).agg(
        sector_log_ret=('log_return', 'mean'),
        sector_volume=('volume', 'sum')
    ).reset_index()

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
    
    sector_results = []
    all_semantic_labels = set()
    for industry, group in df_sector_final.groupby('industry'):
        group = group.sort_values('time').reset_index(drop=True)
        Z_sec = group[Z_SECTOR_COLS].fillna(0).values
        if len(Z_sec) < 100: continue
        
        sec_train_mask = group['time'] <= HMM_TRAIN_END
        Z_sec_train = Z_sec[sec_train_mask]
        if len(Z_sec_train) < 50: Z_sec_train = Z_sec

        model_sector = GMMHMM(n_components=K_SECTOR_BEST, n_mix=2, covariance_type='diag', min_covar=0.01, n_iter=100, random_state=RANDOM_STATE)
        ind_safe_name = industry.replace(' ', '_').replace('/', '_')
        sector_model_path = MODEL_DIR / f'sector_{ind_safe_name}.pkl'
        
        if sector_model_path.exists():
            with open(sector_model_path, 'rb') as f:
                model_sector = pickle.load(f)
        else:
            try:
                model_sector.fit(Z_sec_train)
                with open(sector_model_path, 'wb') as f:
                    pickle.dump(model_sector, f)
            except Exception as e:
                print(f"Bỏ qua ngành {industry} do lỗi hội tụ: {e}")
                continue
                
        group['sector_regime'], probs = get_hmm_filtered_inference(model_sector, Z_sec)
        
        stats = []
        for k in range(K_SECTOR_BEST):
            mask = group['sector_regime'] == k
            r = group.loc[mask, 'sector_log_ret'].mean() if mask.sum() > 0 else 0.0
            v = group.loc[mask, 'sector_vol20'].mean() if mask.sum() > 0 else 0.0
            stats.append({'state': k, 'mean_ret': r, 'mean_vol': v})

        labels = auto_label_sector(pd.DataFrame(stats), K_SECTOR_BEST)
        group['sector_regime_label'] = group['sector_regime'].map(labels)
        group['sector_best_K'] = K_SECTOR_BEST

        for k in range(K_SECTOR_BEST):
            semantic = labels[k]
            group[f'prob_sector_{semantic}'] = probs[:, k]
            all_semantic_labels.add(f'prob_sector_{semantic}')

        sector_results.append(group)
        print(f"[+] Sector HMM: {industry} OK")

    df_sector_hmm = pd.concat(sector_results, ignore_index=True)
    for col in all_semantic_labels:
        if col not in df_sector_hmm.columns: df_sector_hmm[col] = 0.0
        else: df_sector_hmm[col] = df_sector_hmm[col].fillna(0.0)

    # ---------------------------------------------------------
    # 5. TICKER HMM (BƯỚC CUỐI - GỘP TẤT CẢ)
    # ---------------------------------------------------------
    print("--- CHẠY TICKER HMM ---")
    tickers = df_m1['ticker'].unique()
    global_vars = df_market[['time'] + [f'Market_Prob_{k}' for k in range(K_MARKET_BEST)] + prob_cols].copy()
    
    ticker_dfs = []
    for i, ticker in enumerate(tickers):
        df_tick = df_m1[df_m1['ticker'] == ticker].copy().sort_values('time').reset_index(drop=True)
        df_tick['rolling_vol_20d'] = df_tick['log_return'].rolling(20).std() * np.sqrt(252)
        df_tick['volume_ratio'] = df_tick['volume'] / df_tick['volume'].rolling(20).mean()
        df_tick['return_5d'] = df_tick['close'].pct_change(5)
        df_tick['return_20d'] = df_tick['close'].pct_change(20)
        df_tick['rolling_vol_5'] = df_tick['log_return'].rolling(5).std() * np.sqrt(252)
        df_tick['mom_1M'] = df_tick['close'].pct_change(20)
        df_tick['dist_MA50'] = df_tick['close'] / df_tick['close'].rolling(50).mean() - 1

        ticker_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'log_return', 'industry', 'rolling_vol_20d', 'volume_ratio', 'return_5d', 'return_20d', 'rolling_vol_5', 'mom_1M', 'dist_MA50']
        ticker_aligned = global_vars.merge(df_tick[ticker_cols], on='time', how='inner')

        # Gộp Sector
        sector_cols_to_merge = ['time', 'industry'] + list(all_semantic_labels)
        ticker_aligned = ticker_aligned.merge(df_sector_hmm[sector_cols_to_merge], on=['time', 'industry'], how='left')
        ticker_aligned[list(all_semantic_labels)] = ticker_aligned[list(all_semantic_labels)].fillna(0)

        # Chọn features lai (Hybrid Features)
        tick_specific_features = ['log_return', 'rolling_vol_20d', 'volume_ratio']
        market_prob_features = sorted([col for col in ticker_aligned.columns if col.startswith('Market_Prob_')])[:-1]
        macro_prob_features = sorted([col for col in ticker_aligned.columns if col.startswith('Macro_Prob_')])[:-1]
        sector_prob_features = sorted([c for c in all_semantic_labels if c in ticker_aligned.columns and ticker_aligned[c].std() > 1e-6])[:-1]

        hybrid_features = tick_specific_features + macro_prob_features + market_prob_features + sector_prob_features

        fd_z_tick, Z_all_tick = make_Z_ticker(ticker_aligned, hybrid_features, window=252)

        if len(Z_all_tick) < 100: continue

        model_ticker = GMMHMM(n_components=K_TICKER_BEST, n_mix=2, covariance_type='diag', min_covar=0.01, random_state=RANDOM_STATE, n_iter=100)
        ticker_model_path = MODEL_DIR / f'ticker_{ticker}.pkl'
        
        tick_train_mask = fd_z_tick['time'] <= HMM_TRAIN_END
        Z_train_tick = Z_all_tick[tick_train_mask]
        if len(Z_train_tick) < 50: Z_train_tick = Z_all_tick

        if ticker_model_path.exists():
            with open(ticker_model_path, 'rb') as f:
                model_ticker = pickle.load(f)
        else:
            try:
                model_ticker.fit(Z_train_tick)
                with open(ticker_model_path, 'wb') as f:
                    pickle.dump(model_ticker, f)
            except:
                continue

        ticker_daily_states, ticker_daily_probs = get_hmm_filtered_inference(model_ticker, Z_all_tick)

        stats = []
        for k in range(K_TICKER_BEST):
            mask = ticker_daily_states == k
            r = ticker_aligned.loc[mask, 'log_return'].mean()
            v = ticker_aligned.loc[mask, 'rolling_vol_20d'].mean()
            stats.append({'state': k, 'mean_ret': r, 'mean_vol': v})

        ticker_labels_map = auto_label_sector(pd.DataFrame(stats), K_TICKER_BEST)
        ticker_daily_labels = pd.Series(ticker_daily_states).map(ticker_labels_map).values

        df_tick_daily_res = pd.DataFrame({
            'time': fd_z_tick['time'].values,
            'ticker_regime': ticker_daily_states,
            'ticker_regime_label': ticker_daily_labels,
        })
        for k in range(K_TICKER_BEST):
            df_tick_daily_res[f'prob_ticker_{k}'] = ticker_daily_probs[:, k]

        state_cols = ['ticker_regime', 'ticker_regime_label'] + [f'prob_ticker_{k}' for k in range(K_TICKER_BEST)]
        ticker_master = ticker_aligned.merge(df_tick_daily_res[['time'] + state_cols], on='time', how='inner')
        ticker_master['ticker'] = ticker

        ticker_dfs.append(ticker_master)
        
    master_ticker = pd.concat(ticker_dfs, ignore_index=True)
    master_ticker = master_ticker.dropna(subset=['close']).reset_index(drop=True)
    cols_reordered = ['time', 'ticker'] + [col for col in master_ticker.columns if col not in ['time', 'ticker']]
    master_ticker = master_ticker[cols_reordered]
    
    # ---------------------------------------------------------
    # 6. LƯU KẾT QUẢ VÀO FILE PARQUET/CSV
    # ---------------------------------------------------------
    master_ticker.to_parquet(OUTPUT_DIR / 'master_drl_ready_ticker.parquet', index=False)
    print(f"HOÀN TẤT. File master_drl_ready_ticker.parquet đã được lưu tại {OUTPUT_DIR}. (Size: {master_ticker.shape})")

if __name__ == '__main__':
    main()
