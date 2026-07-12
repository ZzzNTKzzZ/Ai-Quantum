
# --------------------------------------------------import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import warnings
warnings.filterwarnings('ignore') # Tắt các cảnh báo không cần thiết để Log sạch đẹp hơn

# mà không cần can thiệp sâu vào bên trong mã nguồn.
class CONFIG:
    TRAINING_MODE = 'fast_split' # 'fast_split' or 'rolling_window'
    # 1. PHÍ GIAO DỊCH (Transaction Cost): 
    # Tỷ lệ phần trăm CTCK thu cho mỗi lần bạn mua hoặc bán cổ phiếu. (VD: 0.2% = 0.002)
    COST_RATE = 0.0003

    TURNOVER_PENALTY_RATE = 0.1
    DRAWDOWN_PENALTY_RATE = 2.0
    LIQUIDITY_BONUS_RATE = 0.02
    TRAIN_WINDOW = 252
    TEST_WINDOW = 21
    SAVE_MODEL_PATH = "AI_Brain_Current.zip"

    # Các giá trị tính điểm (Reward/Penalty System)
    REWARD_WIN_MULT = 100
    REWARD_LOSS_MULT = 200
    REWARD_ALPHA_MULT = 50
    REWARD_CASH_CRASH_MULT = 100
    PENALTY_OVER_DIVERSIFICATION = 0.01
    
        
    # 7. CHẾ ĐỘ CHỜ HÀNG VỀ T+2 (Rule 7: Settlement Lock):
    # Đặc sản của Chứng khoán VN. Số ngày cổ phiếu bị "nhốt" chưa về tài khoản. 
    # Trong những ngày này (VD: 2 ngày đầu tiên), AI tuyệt đối không thể bán cắt lỗ/chốt lời dù thị trường sập.
    T_PLUS_SETTLEMENT = 3
    
    # 8. SỐ BƯỚC ĐÀO TẠO (Training Timesteps):
    # Số vòng lặp để AI cập nhật bộ não. Càng cao (50k - 100k) AI càng thông minh nhưng thời gian train càng lâu.
    TRAINING_TIMESTEPS = 100000
    
    # 9. ĐỘ PHÂN BỔ VỐN (Entropy Coefficient):
    # Quyết định AI sẽ All-in hay Rải rác.
    # Giá trị nhỏ (0.001) -> Ưu tiên All-in vài mã mạnh nhất.
    # Giá trị lớn (0.05) -> Ưu tiên chia đều tiền ra mua nhiều mã để phân tán rủi ro.
    ENT_COEF = 0.01

    # Neuron Network

def load_data():
    # 1. Nạp dữ liệu thô
    raw_df = pd.read_parquet(r"C:\Users\ADMIN\Desktop\Kaggle\output\hmm_v3_op1_extended\master_drl_ready_full.parquet")
    raw_df['time'] = pd.to_datetime(raw_df['time'])
    
    # Lấy các DataFrame cơ sở
    returns_df = raw_df.pivot(index='time', columns='ticker', values='log_return').fillna(0)
    weights_dim = returns_df.shape[1] # Số lượng mã cổ phiếu
    
    vol_df = raw_df.pivot(index='time', columns='ticker', values='volume').fillna(0)
    close_df = raw_df.pivot(index='time', columns='ticker', values='close').ffill().fillna(0)
    open_df = raw_df.pivot(index='time', columns='ticker', values='open').ffill().fillna(0)
    low_df = raw_df.pivot(index='time', columns='ticker', values='low').ffill().fillna(0)
    high_df = raw_df.pivot(index='time', columns='ticker', values='high').ffill().fillna(0)
    
    # -------------------------------------
    # CÁC CHỈ BÁO VĨ MÔ & KỸ THUẬT (DÀNH CHO AI HỌC)
    # -------------------------------------
    market_return_5d = returns_df.rolling(5).sum().mean(axis=1).fillna(0) 
    market_return_20d = returns_df.rolling(20).sum().mean(axis=1).fillna(0) 
    market_vol_20d = returns_df.rolling(20).std().mean(axis=1).fillna(0) 
    
    price_df = np.exp(returns_df.cumsum())
    ma20_df = price_df.rolling(20).mean()
    dist_ma20_df = ((price_df - ma20_df) / ma20_df).fillna(0)

    # -------------------------------------
    # CÁC CHỈ BÁO KỊCH BẢN LUẬT (KHÔNG CHO AI HỌC)
    # -------------------------------------
    ema_df_20 = close_df.ewm(span=20, adjust=False).mean()
    ema_df_50 = close_df.ewm(span=50, adjust=False).mean()
    ema_df_200 = close_df.ewm(span=200, adjust=False).mean()

    # SMA Vol
    def calc_sma(df, n=5):
        return df.rolling(n).mean()
    sma_vol_5_df = calc_sma(vol_df)
    sma_vol_20_df = calc_sma(vol_df, n=20)
    # RSI (Chuẩn Wilder's Smoothing)
    change_df = close_df - close_df.shift(1)
    gain_df = change_df.clip(lower=0)
    loss_df = change_df.clip(upper=0).abs()
    ema_gain = gain_df.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    ema_loss = loss_df.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    rs_df = ema_gain / ema_loss
    rsi_df = 100 - (100 / (1 + rs_df))
    rsi_df = rsi_df.fillna(100)

    # MACD
    macd_df = close_df.ewm(span=12, adjust=False).mean() - close_df.ewm(span=26, adjust=False).mean()
    signal_df = macd_df.ewm(span=9, adjust=False).mean()
    hist_df = macd_df - signal_df
    
    # Độ dốc xu hướng (Slope)
    def calc_slope(df, n=3):
        return (df - df.shift(n)) / n
    slope_ema20_3_df = calc_slope(ema_df_20, n=3)  
    slope_sma5_3_df = calc_sma(sma_vol_5_df, n=3)
    # Candlestick Patterns (Đã sửa lỗi Boolean Error)
    body_df = (close_df - open_df).abs()
    tail_df = (np.minimum(close_df, open_df) - low_df)
    head_df = high_df - np.maximum(close_df, open_df)
    
    is_hammer_df = (tail_df >= 2 * body_df) & (head_df <= 0.1 * body_df) & (body_df > 0)
    is_bull_engulf_df = (close_df.shift(1) < open_df.shift(1)) & (close_df > open_df) & (close_df >= open_df.shift(1)) & (open_df <= close_df.shift(1))

    # LowestLow(5) và HighestHigh(10)
    lowest_low_5_df = low_df.rolling(5).min()
    highest_high_5_df = high_df.rolling(5).max()
    highest_high_10_df = high_df.rolling(10).max()
    highest_high_20_df = high_df.rolling(20).max()

    # Đáy và đỉnh
    # Ngày T-2 là Đỉnh nếu nó cao hơn T-4, T-3 (Quá khứ) VÀ cao hơn T-1, T (Hiện tại)
    is_peak = (high_df.shift(2) > high_df.shift(4)) & \
                (high_df.shift(2) > high_df.shift(3)) & \
                (high_df.shift(2) > high_df.shift(1)) & \
                (high_df.shift(2) > high_df)

    # Ngày T-2 là Đáy nếu nó thấp hơn T-4, T-3 (Quá khứ) VÀ thấp hơn T-1, T (Hiện tại)
    is_valley = (low_df.shift(2) < low_df.shift(4)) & \
                (low_df.shift(2) < low_df.shift(3)) & \
                (low_df.shift(2) < low_df.shift(1)) & \
                (low_df.shift(2) < low_df)
    
    peak_values = high_df.shift(2).where(is_peak, np.nan)
    valley_values = low_df.shift(2).where(is_valley, np.nan)

    last_peak_df = peak_values.ffill()
    last_valley_df = valley_values.ffill()
        
    # FIBONACCI
    wave_amplitude_df = last_peak_df - last_valley_df
    wave_state_df = pd.DataFrame(np.nan, index=high_df.index, columns=high_df.columns)
    
    wave_state_df[is_peak] = 1
    wave_state_df[is_valley] = -1
    
    last_swing_type_df = wave_state_df.ffill()
    
    # SUP() RES()
    n = 20 # Số phiên để xác nhận Đỉnh/Đáy (Bạn có thể tinh chỉnh 10 hoặc 20)
    window = 2 * n + 1
    
    # 1. Tìm mức cao nhất/thấp nhất trong 41 ngày gần nhất (20 ngày trước + 1 ngày giữa + 20 ngày sau)
    rolling_max = high_df.rolling(window).max()
    rolling_min = low_df.rolling(window).min()
    
    # 2. XÁC NHẬN ĐỈNH/ĐÁY KHÔNG DÙNG CENTER=TRUE
    # Nếu giá cao nhất của 20 ngày trước (shift(20)) BẰNG với mức Max của toàn bộ 41 ngày
    # -> Chứng tỏ 20 ngày trước là một ĐỈNH CỤC BỘ (Peak)
    is_peak = (high_df.shift(n) == rolling_max)
    is_valley = (low_df.shift(n) == rolling_min)
    
    # 3. Ghi nhớ mức giá tại Đỉnh/Đáy đó
    peak_values = high_df.shift(n).where(is_peak, np.nan)
    valley_values = low_df.shift(n).where(is_valley, np.nan)
    
    # 4. Kéo dài giá trị ra để lấy Hỗ trợ (Sup) và Kháng cự (Res) gần nhất
    res_df = peak_values.ffill() # Đỉnh gần nhất (Res)
    sup_df = valley_values.ffill() # Đáy gần nhất (Sup)


    # SÓNG TĂNG CHÍNH = Mốc cuối cùng được xác nhận là 1 cái Đỉnh
    # Trả về giá trị True/False cho từng ngày
    is_upward_wave_df = (last_swing_type_df == 1)
    fib_382_df = last_peak_df - (0.382 * wave_amplitude_df)
    fib_500_df = last_peak_df - (0.500 * wave_amplitude_df)
    fib_618_df = last_peak_df - (0.618 * wave_amplitude_df)
    fib_ext_127_df = last_peak_df + (0.272 * wave_amplitude_df)
    fib_ext_161_df = last_peak_df + (0.618 * wave_amplitude_df)

     # Bollinger Bands 
    bb_middle_df = close_df.rolling(20).mean()
    bb_std_df = close_df.rolling(20).std()
    bb_upper_df = bb_middle_df + (2 * bb_std_df)
    bb_lower_df = bb_middle_df - (2 * bb_std_df)
    
    # 2. TÍNH ĐỘ RỘNG DẢI BĂNG (BB WIDTH)
    bb_width_df = (bb_upper_df - bb_lower_df) / bb_middle_df
    
    # 3. TÍN HIỆU BIẾN ĐỘNG MẠNH (DẢI BĂNG MỞ RỘNG)
    # So sánh độ rộng hôm nay với hôm qua VÀ với Trung bình 20 ngày
    bb_expanding_df = (bb_width_df > bb_width_df.shift(1)) & \
                        (bb_width_df > bb_width_df.rolling(20).mean())
    
    # ACCUMULATION
    highest_high_21_df = high_df.rolling(21).max().shift(1)
    lowest_low_21_df = low_df.rolling(21).min().shift(1)
        
    # Biên độ dao động của cái hộp nhỏ hơn 15%
    is_accumulation_df = (highest_high_21_df - lowest_low_21_df) / lowest_low_21_df <= 0.15
    # Bóc tách dữ liệu HMM từ raw_df
    prob0_df = raw_df.pivot(index='time', columns='ticker', values='prob_ticker_0').fillna(0) 
    prob1_df = raw_df.pivot(index='time', columns='ticker', values='prob_ticker_1').fillna(0) 
    prob2_df = raw_df.pivot(index='time', columns='ticker', values='prob_ticker_2').fillna(0) 
    vol_20d_df = raw_df.pivot(index='time', columns='ticker', values='rolling_vol_20d').fillna(0)
    ret_20d_df = raw_df.pivot(index='time', columns='ticker', values='return_20d').fillna(0)
    vol_ratio_df = raw_df.pivot(index='time', columns='ticker', values='volume_ratio').fillna(0)

    # Broadcast vĩ mô
    mkt_ret5_df = pd.DataFrame(np.tile(market_return_5d.values.reshape(-1, 1), (1, weights_dim)), index=returns_df.index, columns=returns_df.columns)
    mkt_ret20_df = pd.DataFrame(np.tile(market_return_20d.values.reshape(-1, 1), (1, weights_dim)), index=returns_df.index, columns=returns_df.columns)
    mkt_vol_df = pd.DataFrame(np.tile(market_vol_20d.values.reshape(-1, 1), (1, weights_dim)), index=returns_df.index, columns=returns_df.columns)
    
    # -------------------------------------------------------------------------
    # GOM THÀNH 2 BẢNG BIỆT LẬP (1 CHO AI, 1 CHO LUẬT KỊCH BẢN)
    # -------------------------------------------------------------------------
    from scipy.stats import norm
    def apply_nqt(df):
        ranked = df.rank(axis=1, method='average')
        uniform = (ranked - 0.5) / df.shape[1]
        nqt = pd.DataFrame(norm.ppf(uniform), index=df.index, columns=df.columns)
        return nqt.replace([np.inf, -np.inf], 0).fillna(0)

    # BẢNG 1: Dành cho Mạng Nơ-ron (Đã qua NQT Cross-Sectional)
    core_list = [prob0_df, prob1_df, prob2_df, vol_20d_df, ret_20d_df, vol_ratio_df, mkt_ret5_df, mkt_ret20_df, mkt_vol_df, dist_ma20_df]
    ai_features_df = pd.concat([apply_nqt(df) for df in core_list], axis=1).fillna(0)
                                
    # BẢNG 2: Kho vũ khí cho kịch bản của con người (AI không được thấy)

    list_strategies = [vol_df ,low_df ,close_df, 
                       ema_df_20, ema_df_50, ema_df_200, 
                       bb_lower_df, bb_middle_df ,bb_upper_df,
                       rsi_df, hist_df, macd_df,
                       sma_vol_20_df,
                       slope_ema20_3_df, slope_sma5_3_df, 
                       is_hammer_df.astype(float), is_bull_engulf_df.astype(float),
                       lowest_low_5_df, highest_high_5_df ,highest_high_10_df, highest_high_20_df,
                       fib_382_df, fib_500_df, fib_618_df, is_upward_wave_df, fib_ext_127_df, fib_ext_161_df,
                       sup_df, res_df,
                       bb_expanding_df,
                       is_accumulation_df,
                       ]

    strategies_features_df = pd.concat(list_strategies, axis=1).fillna(0)
    
    num_strategies_features = 31
    
    # HỢP NHẤT TOÀN BỘ CHỈ BÁO VÀO AI FEATURES (Áp dụng NQT cho mảng chiến lược đưa vào AI)
    ai_strategies_df = pd.concat([apply_nqt(df) for df in list_strategies], axis=1).fillna(0)
    ai_features_df = pd.concat([ai_features_df, ai_strategies_df], axis=1)

    # Đồng bộ độ dài
    returns_df, ai_features_df = returns_df.align(ai_features_df, axis=0, join='inner')
    returns_df, strategies_features_df = returns_df.align(strategies_features_df, axis=0, join='inner')
    tickers = returns_df.columns.tolist()
    
    dates = returns_df.index.strftime('%d/%m/%Y').tolist()
    return returns_df, ai_features_df, strategies_features_df, weights_dim, tickers, num_strategies_features, dates

class AdvancedPortfolioEnv(gym.Env):
        def __init__(self, returns_df, ai_features_df, script_features_df, weights_dim, num_script_features, tickers=None, step_size=1, cost_rate=CONFIG.COST_RATE, is_test=False, dates=None):

            super().__init__()
            self.returns_arr = np.exp(returns_df.values) - 1.0 
            
            # Lưu 2 mảng tách biệt
            self.features_arr = ai_features_df.values
            self.script_arr = script_features_df.values
            
            self.weights_dim = weights_dim
            self.num_script_features = num_script_features
            self.n_steps = len(self.returns_arr)
            
            self.step_size = step_size
            self.cost_rate = cost_rate
            self.is_test = is_test 
            self.tickers = tickers
            self.dates = dates
            self.prev_weights = np.zeros(self.weights_dim)
            self.high_watermark = 1.0
            self.current_portfolio_value = 1.0
            
            self.action_space = spaces.Box(low=0, high=1, shape=(self.weights_dim,), dtype=np.float32)
            # Báo cho AI biết là nó chỉ cần học 10 tính năng (vì ai_features_df có 10 cột)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.weights_dim, 41), dtype=np.float32)
            
            self.current_step = 0
            self.weights = np.zeros(self.weights_dim)
            self.entry_prices = np.zeros(self.weights_dim)
            self.entry_scenarios = np.zeros(self.weights_dim)
            self.entry_stages = np.zeros(self.weights_dim)
        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.weights = np.zeros(self.weights_dim)
            return self._get_obs(), {}
            
        def _get_obs(self):
            idx = min(self.current_step, self.n_steps - 1)
            obs_1d = self.features_arr[idx]
            obs_2d = obs_1d.reshape(41, self.weights_dim).T
            return obs_2d.astype(np.float32)
            
        def step(self, action):
            current_obs = self._get_obs()
            idx = min(self.current_step, self.n_steps - 1)
            
            current_scripts = self.script_arr[idx].reshape(self.num_script_features, self.weights_dim).T
            prev_idx = max(0, idx - 1)
            prev_scripts = self.script_arr[prev_idx].reshape(self.num_script_features, self.weights_dim).T
            
            scenario_bonus = 0.0
            
            for i in range(self.weights_dim):
                current_regime = np.argmax([current_obs[i][0], current_obs[i][1], current_obs[i][2]])
                
                volume          = current_scripts[i][0]
                low             = current_scripts[i][1]
                close           = current_scripts[i][2]
                ema20           = current_scripts[i][3]
                ema50           = current_scripts[i][4]
                ema200          = current_scripts[i][5]
                bb_lower        = current_scripts[i][6]
                bb_middle       = current_scripts[i][7]
                bb_upper        = current_scripts[i][8]  
                rsi             = current_scripts[i][9]
                macd_hist       = current_scripts[i][10]
                macd_line       = current_scripts[i][11]
                sma_vol_20      = current_scripts[i][12]
                slope_ema20_3   = current_scripts[i][13]
                slope_sma5_3    = current_scripts[i][14]
                is_hammer       = current_scripts[i][15]
                is_bull_engulf  = current_scripts[i][16]
                lowest_low_5    = current_scripts[i][17]
                highest_high_5  = current_scripts[i][18]
                highest_high_10 = current_scripts[i][19]
                highest_high_20 = current_scripts[i][20]
                fib_382         = current_scripts[i][21]
                fib_500         = current_scripts[i][22]
                fib_618         = current_scripts[i][23]
                is_upward_wave  = current_scripts[i][24]
                fib_ext_127     = current_scripts[i][25]
                fib_ext_161     = current_scripts[i][26]
                sup             = current_scripts[i][27]
                res             = current_scripts[i][28]
                bb_expanding    = current_scripts[i][29]
                is_accumulation = current_scripts[i][30]

                prev_volume     = prev_scripts[i][0]
                prev_close      = prev_scripts[i][1]
                prev_ema20      = prev_scripts[i][3]
                prev_ema50      = prev_scripts[i][4]
                prev_macd_hist  = prev_scripts[i][10]
                prev_macd_line  = prev_scripts[i][11]

                is_downtrend_aligned = (ema20 < ema50) and (ema50 < ema200)
                is_vol_dropping = (slope_sma5_3 < 0) 
                is_macd_bullish_divergence = (low < sup) and (macd_hist > 0 and prev_macd_hist <= 0)
                
                def calc_bonus(act, target, max_points):
                    return max_points * (1.0 - abs(act - target))
                
                bonus_i = 0.0

                # [KỊCH BẢN 1] Cash Position (Override)
                if is_downtrend_aligned and (macd_line < 0) and (rsi < 50) and is_vol_dropping and (current_regime == 0):
                    action[i] = 0.0
                    self.entry_scenarios[i] = 0
                    self.entry_stages[i] = 0
                    self.entry_prices[i] = 0.0
                    continue

                if self.weights[i] > 0: 
                    scenario_id = self.entry_scenarios[i]
                    entry_price = self.entry_prices[i]
                    stage = self.entry_stages[i]
                    
                    if scenario_id == 2:
                        # [KỊCH BẢN 2] EMA Pullback
                        risk = entry_price - ema50 if entry_price > ema50 else (entry_price * 0.05)
                        if stage == 1:
                            is_green_confirm = (close > prev_close) and (close > ema20)
                            is_broken_with_vol = (close < ema50) and (volume > sma_vol_20)
                            if is_broken_with_vol: 
                                bonus_i += calc_bonus(action[i], 0.0, 100.0)
                                self.entry_scenarios[i] = 0
                            elif is_green_confirm: 
                                bonus_i += calc_bonus(action[i], 1.0, 70.0)
                                self.entry_stages[i] = 2
                        elif stage == 2:
                            is_macd_dead_cross = (macd_hist < 0) and (prev_macd_hist >= 0)
                            is_ema_dead_cross = (ema20 < ema50) and (prev_ema20 >= prev_ema50)
                            if close < ema50 or close < (lowest_low_5 * 0.98) or close >= highest_high_10 or close >= (entry_price + 2 * risk) or is_ema_dead_cross: 
                                bonus_i += calc_bonus(action[i], 0.0, 100.0)
                                self.entry_scenarios[i] = 0
                                
                    elif scenario_id == 3:
                        # [KỊCH BẢN 3] Fibonacci Pullback
                        is_green_candle = close > current_scripts[i][2]
                        if stage == 1:
                            if close < fib_618: 
                                bonus_i += calc_bonus(action[i], 0.0, 100.0)
                                self.entry_scenarios[i] = 0
                            elif is_green_candle and close > entry_price: 
                                bonus_i += calc_bonus(action[i], 1.0, 70.0)
                                self.entry_stages[i] = 2
                        elif stage == 2:
                            if close < fib_618 or close >= fib_ext_161 or close >= highest_high_10: 
                                bonus_i += calc_bonus(action[i], 0.0, 100.0)
                                self.entry_scenarios[i] = 0
                            
                    elif scenario_id == 4:
                        # [KỊCH BẢN 4] Support Bounce
                        if close < sup * 0.98 or close >= highest_high_10 or close >= ema50: 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 5:
                        # [KỊCH BẢN 5] Range Trading
                        if close < sup * 0.98 or close >= res: 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 6:
                        # [KỊCH BẢN 6] RSI Oversold in BULL
                        if close < ema50 or close >= highest_high_20: 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 7 or scenario_id == 8:
                        # [KỊCH BẢN 7] Bollinger Trend Ride / [KỊCH BẢN 8] Breakout Resistance
                        if stage == 1:
                            is_retest_success = (low <= res) and (close > res) and (volume < sma_vol_20)
                            is_breakout_runaway = (close > res * 1.05)
                            if is_retest_success: 
                                bonus_i += calc_bonus(action[i], 1.0, 60.0)
                                self.entry_stages[i] = 2
                            if close < res * 0.97: 
                                bonus_i += calc_bonus(action[i], 0.0, 100.0)
                                self.entry_scenarios[i] = 0
                        elif stage == 2:
                            if close < res * 0.97 or close >= fib_ext_127: 
                                bonus_i += calc_bonus(action[i], 0.0, 100.0)
                                self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 9:
                        # [KỊCH BẢN 9] Bollinger Mean Reversion
                        if (bb_expanding == 1.0 and close < bb_lower) or (close < bb_lower * 0.98): 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        elif stage == 1:
                            if close > bb_lower: 
                                bonus_i += calc_bonus(action[i], 1.0, 50.0)
                                self.entry_stages[i] = 2
                        elif stage == 2:
                            if close >= bb_upper: 
                                bonus_i += calc_bonus(action[i], 0.0, 100.0)
                                self.entry_scenarios[i] = 0
                            elif close >= bb_middle: 
                                bonus_i += calc_bonus(action[i], 0.5, 50.0)
                                self.entry_stages[i] = 3
                        elif stage == 3:
                            if close >= bb_upper: 
                                bonus_i += calc_bonus(action[i], 0.0, 100.0)
                                self.entry_scenarios[i] = 0
                            
                    elif scenario_id == 10:
                        # [KỊCH BẢN 10] RSI Swing
                        if close < sup * 0.98 or rsi > 70: 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 11:
                        # [KỊCH BẢN 11] Bollinger + MACD Crossover
                        if close < bb_lower * 0.98 or close >= bb_upper: 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 12:
                        # [KỊCH BẢN 12] MACD Zero Line Bounce
                        if (macd_hist < 0 and prev_macd_line >= 0) or (ema20 < ema50 and prev_ema20 >= prev_ema50): 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 13:
                        # [KỊCH BẢN 13] Golden Cross
                        if (macd_hist < 0 and prev_macd_line >= 0) or (ema20 < ema50 and prev_ema20 >= prev_ema50): 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 14:
                        # [KỊCH BẢN 14] Oversold Reversal
                        if close < entry_price * 0.96 or close >= ema20: 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 15:
                        # [KỊCH BẢN 15] Falling Wedge Breakout
                        target_price = entry_price + (res - sup)
                        if close < sup or close >= target_price: 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0
                        
                    elif scenario_id == 16:
                        # [KỊCH BẢN 16] Dead Cat Bounce
                        if close < entry_price * 0.97 or close >= entry_price * 1.07: 
                            bonus_i += calc_bonus(action[i], 0.0, 100.0)
                            self.entry_scenarios[i] = 0

                else:
                    if action[i] > 0.0:
                        # [KỊCH BẢN 2] EMA Pullback
                        if current_regime == 2 and (ema20 > ema50) and (close > ema50) and (slope_ema20_3 > 0) and (low <= ema20) and (40 <= rsi <= 55) and (is_bull_engulf == 1.0 or is_hammer == 1.0) and (volume > prev_volume):
                            self.entry_prices[i], self.entry_scenarios[i], self.entry_stages[i] = close, 2, 1
                            bonus_i += calc_bonus(action[i], 0.3, 30.0)
                        
                        # [KỊCH BẢN 3] Fibonacci Pullback
                        elif current_regime == 2 and is_upward_wave == 1.0 and (fib_618 <= close <= fib_382) and (is_hammer == 1.0 or is_bull_engulf == 1.0):
                            self.entry_prices[i], self.entry_scenarios[i], self.entry_stages[i] = close, 3, 1
                            bonus_i += calc_bonus(action[i], 0.3, 30.0)
                        
                        # [KỊCH BẢN 4] Support Bounce
                        elif current_regime == 2 and low <= sup and volume < sma_vol_20 and is_bull_engulf == 1.0:
                            self.entry_prices[i], self.entry_scenarios[i] = close, 4
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)
                        
                        # [KỊCH BẢN 5] Range Trading
                        elif current_regime == 1 and is_accumulation == 1.0 and low <= sup and volume < sma_vol_20:
                            self.entry_prices[i], self.entry_scenarios[i] = close, 5
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)
                        
                        # [KỊCH BẢN 6] RSI Oversold in BULL
                        elif current_regime == 2 and ema20 > ema50 and (30 <= rsi <= 40) and low <= ema50 and is_hammer == 1.0:
                            self.entry_prices[i], self.entry_scenarios[i] = close, 6
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)
                        
                        # [KỊCH BẢN 7] Bollinger Trend Ride
                        elif current_regime == 2 and bb_expanding == 1.0 and close >= (bb_upper * 0.99) and slope_ema20_3 > 0 and volume > prev_volume:
                            self.entry_prices[i], self.entry_scenarios[i], self.entry_stages[i] = close, 7, 1
                            bonus_i += calc_bonus(action[i], 0.4, 40.0)
                        
                        # [KỊCH BẢN 8] Breakout Resistance
                        elif current_regime == 2 and is_accumulation == 1.0 and close >= res and volume > (1.5 * sma_vol_20) and macd_hist > 0:
                            self.entry_prices[i], self.entry_scenarios[i], self.entry_stages[i] = close, 8, 1
                            bonus_i += calc_bonus(action[i], 0.4, 40.0)
                        
                        # [KỊCH BẢN 9] Bollinger Mean Reversion
                        elif current_regime == 1 and bb_expanding == 0.0 and low <= bb_lower and rsi < 30:
                            self.entry_prices[i], self.entry_scenarios[i], self.entry_stages[i] = close, 9, 1
                            bonus_i += calc_bonus(action[i], 0.5, 50.0)
                        
                        # [KỊCH BẢN 10] RSI Swing
                        elif current_regime == 1 and is_accumulation == 1.0 and rsi < 30 and low <= sup:
                            self.entry_prices[i], self.entry_scenarios[i] = close, 10
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)
                        
                        # [KỊCH BẢN 11] Bollinger + MACD Crossover
                        elif current_regime == 1 and bb_expanding == 0.0 and is_accumulation == 1.0 and low <= bb_lower and (macd_hist > 0 and prev_macd_hist <= 0):
                            self.entry_prices[i], self.entry_scenarios[i] = close, 11
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)
                        
                        # [KỊCH BẢN 12] MACD Zero Line Bounce
                        elif current_regime == 2 and (-0.1 <= macd_line <= 0.1) and macd_hist > 0 and prev_macd_line <= 0:
                            self.entry_prices[i], self.entry_scenarios[i] = close, 12
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)
                        
                        # [KỊCH BẢN 13] Golden Cross
                        elif current_regime == 2 and close > ema200 and ema20 > ema50 and prev_ema20 <= prev_ema50 and macd_line > 0:
                            self.entry_prices[i], self.entry_scenarios[i] = close, 13
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)
                        
                        # [KỊCH BẢN 14] Oversold Reversal
                        elif current_regime == 0 and slope_sma5_3 < 0 and rsi < 25 and is_macd_bullish_divergence:
                            self.entry_prices[i], self.entry_scenarios[i] = close, 14
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)
                        
                        # [KỊCH BẢN 15] Falling Wedge Breakout
                        elif current_regime == 0 and (close > res) and (volume > 1.5 * sma_vol_20) and (close != low):
                            self.entry_prices[i], self.entry_scenarios[i] = close, 15
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)
                        
                        # [KỊCH BẢN 16] Dead Cat Bounce
                        elif current_regime == 0 and rsi < 20 and is_hammer == 1.0 and volume > sma_vol_20 and (close != low):
                            self.entry_prices[i], self.entry_scenarios[i] = close, 16
                            bonus_i += calc_bonus(action[i], 1.0, 100.0)

                scenario_bonus += bonus_i

            action_sum = np.sum(action)
            if action_sum > 0:
                action = action / action_sum
            else:
                action = np.zeros_like(action)
            
            turnover = np.sum(np.abs(action - self.weights))
            cost = self.cost_rate * turnover
            
            self.prev_weights = self.weights.copy()
            self.weights = action
            
            market_ret = np.mean(self.returns_arr[self.current_step])
            daily_ret = np.sum(self.weights * self.returns_arr[self.current_step]) - cost
            
            self.current_portfolio_value = self.current_portfolio_value * (1 + daily_ret)
            
            if daily_ret < 0:
                reward = daily_ret * getattr(CONFIG, 'REWARD_LOSS_MULT', 200)
            else:
                reward = daily_ret * getattr(CONFIG, 'REWARD_WIN_MULT', 100)
                
            alpha = daily_ret - market_ret
            if alpha > 0:
                reward += alpha * getattr(CONFIG, 'REWARD_ALPHA_MULT', 50)
                
            if action_sum == 0.0 and market_ret < 0:
                reward += abs(market_ret) * getattr(CONFIG, 'REWARD_CASH_CRASH_MULT', 100)
                
            turnover_penalty = turnover * getattr(CONFIG, 'TURNOVER_PENALTY_RATE', 0.1)
            reward -= turnover_penalty
            
            vol_ratio = current_scripts[:, 0] / (current_scripts[:, 12] + 1e-9)
            liquidity_bonus = np.sum(self.weights * vol_ratio) * getattr(CONFIG, 'LIQUIDITY_BONUS_RATE', 0.02)
            reward += liquidity_bonus
            
            num_active_positions = np.sum(self.weights > 0.05)
            reward -= (num_active_positions * getattr(CONFIG, 'PENALTY_OVER_DIVERSIFICATION', 0.01))
            
            if self.current_portfolio_value > self.high_watermark:
                self.high_watermark = self.current_portfolio_value
            else:
                drawdown = (self.high_watermark - self.current_portfolio_value) / self.high_watermark
                reward -= drawdown * getattr(CONFIG, 'DRAWDOWN_PENALTY_RATE', 2.0)
                
            reward += scenario_bonus / 1000.0
            
            if getattr(self, 'is_test', False):
                date_str = self.dates[self.current_step] if getattr(self, 'dates', None) else f'Step {self.current_step}'
                if action_sum > 0.0 or daily_ret != 0.0:
                    allocations = [f"{self.tickers[k]}: {self.weights[k]*100:.0f}%" for k in range(self.weights_dim) if self.weights[k] > 0.01]
                    alloc_str = ", ".join(allocations) if allocations else "100% Cash"
                    print(f"[{date_str}] 📊 TỔNG KẾT: Danh mục [{alloc_str}] | Lãi: {daily_ret*100:+.2f}% | Bonus: {scenario_bonus:+.1f} | Reward: {reward:+.1f}")
                
            self.current_step += 1
            done = self.current_step >= self.n_steps - 1 
            
            return self._get_obs(), float(reward), done, False, {'net_return': daily_ret}

class AdvancedTickerExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            num_tickers = observation_space.shape[0] # 46 mã
            num_features_per_ticker = observation_space.shape[1] # 11 tính năng
            
            # Tầng 1 (Local): Học phân tích RIÊNG LẺ
            self.ticker_net = nn.Sequential(
                nn.Linear(num_features_per_ticker, 32),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU()
            )
            
            # --- CẢI TIẾN: Tầng 1.5 (Attention) ---
            # Cho phép các cổ phiếu giao tiếp và so sánh sức mạnh với nhau
            self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
            self.layer_norm = nn.LayerNorm(16)
            # --------------------------------------
    
            # Tầng 2 (Global): Tổng hợp ra quyết định
            self.global_net = nn.Sequential(
                nn.Linear(num_tickers * 16, features_dim),
                nn.ReLU()
            )
    
        def forward(self, observations: th.Tensor) -> th.Tensor:
            batch_size, num_tickers, num_features = observations.shape
            
            # 1. Trích xuất Local
            obs_reshaped = observations.view(batch_size * num_tickers, num_features)
            ticker_features = self.ticker_net(obs_reshaped)
            
            # Đưa về dạng 3D: (Batch, Số_Mã, Đặc_trưng) để chạy Attention
            ticker_features = ticker_features.view(batch_size, num_tickers, 16) 
            
            # 2. --- CẢI TIẾN: Cross-Ticker Attention ---
            # Các mã cổ phiếu "nhìn" nhau để tìm ra con mạnh nhất (Leader)
            attn_out, _ = self.attention(ticker_features, ticker_features, ticker_features)
            
            # Residual connection + LayerNorm (Bắt buộc phải có để đạo hàm không bị nổ))
            ticker_features = self.layer_norm(ticker_features + attn_out)
            # --------------------------------------------
            
            # 3. Đưa Tầng 1.5 vào Tầng 22
            flattened_features = ticker_features.view(batch_size, num_tickers * 16)
            global_features = self.global_net(flattened_features)
            
            return global_features

returns_df, ai_features_df, strategies_features_df, weights_dim, tickers, num_strategies_features, dates = load_data()

from stable_baselines3.common.vec_env import VecNormalize
print("\n[LUẬT 4] KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN...")

TRAINING_MODE = getattr(CONFIG, 'TRAINING_MODE', 'fast_split') # 'fast_split' or 'rolling_window'

total_days = len(returns_df)

model = None

policy_kwargs = dict(
    features_extractor_class=AdvancedTickerExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

all_test_returns = []

if TRAINING_MODE == 'fast_split':
    print(f"\n=======================================================")
    print(f"--- CHẾ ĐỘ HỌC NHANH (FAST SPLIT: 70% Train, 15% Val, 15% Test) ---")
    
    train_ratio = 0.7
    val_ratio = 0.15
    
    train_end = int(total_days * train_ratio)
    val_end = int(total_days * (train_ratio + val_ratio))
    
    returns_train = returns_df.iloc[:train_end]
    ai_train = ai_features_df.iloc[:train_end]
    strategies_train = strategies_features_df.iloc[:train_end]
    dates_train = dates[:train_end]
    
    returns_val = returns_df.iloc[train_end:val_end]
    ai_val = ai_features_df.iloc[train_end:val_end]
    strategies_val = strategies_features_df.iloc[train_end:val_end]
    dates_val = dates[train_end:val_end]
    
    returns_test = returns_df.iloc[val_end:]
    ai_test = ai_features_df.iloc[val_end:]
    strategies_test = strategies_features_df.iloc[val_end:]
    dates_test = dates[val_end:]
    
    train_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(
        returns_train, ai_train, strategies_train, weights_dim, num_strategies_features, tickers=tickers, dates=dates_train
    )])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    print(f"Khởi tạo Mạng Nơ-ron AI (Train: {len(dates_train)} days, Val: {len(dates_val)} days, Test: {len(dates_test)} days)...")
    model = PPO("MlpPolicy", train_env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                n_steps=1024,
                learning_rate=0.00005,
                ent_coef=CONFIG.ENT_COEF,
                batch_size=90)
                
    print(f"Đang huấn luyện mô hình ({CONFIG.TRAINING_TIMESTEPS} steps)...")
    model.learn(total_timesteps=CONFIG.TRAINING_TIMESTEPS)
    
    model.save(getattr(CONFIG, 'SAVE_MODEL_PATH', "AI_Brain_Current.zip"))
    train_env.save("vec_normalize.pkl")
    
    print(f"\nĐang chạy Backtest trên Test Set (CÓ TÍNH PHÍ GIAO DỊCH {CONFIG.COST_RATE * 100}%)...")
    print("-------------------------------------------------------")
    
    test_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(
        returns_test, ai_test, strategies_test, weights_dim, num_strategies_features, tickers=tickers, dates=dates_test, is_test=True
    )])
    test_env = VecNormalize.load("vec_normalize.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False
    obs = test_env.reset()
    
    done = [False]
    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        if 'net_return' in info[0]:
            all_test_returns.append(info[0]['net_return'])
            
    total_profit = (np.prod(1 + np.array(all_test_returns)) - 1) * 100
    print(f"\n🏆 TỔNG LỢI NHUẬN TEST SET: {total_profit:+.2f}%")
    
elif TRAINING_MODE == 'rolling_window':
    train_window = getattr(CONFIG, 'TRAIN_WINDOW', 252)
    test_window = getattr(CONFIG, 'TEST_WINDOW', 21)
    
    current_start = 0
    fold_idx = 1
    
    while current_start + train_window < total_days:
        train_start = current_start
        train_end = current_start + train_window
        test_start = train_end
        test_end = min(test_start + test_window, total_days)
        
        if test_end - test_start < 2:
            break
            
        print(f"\n=======================================================")
        print(f"--- CHU KỲ {fold_idx}: HỌC {dates[train_start]} -> {dates[train_end-1]} | THỰC CHIẾN: {dates[test_start]} -> {dates[test_end-1]} ---")
        
        returns_train = returns_df.iloc[train_start:train_end]
        ai_train = ai_features_df.iloc[train_start:train_end]
        strategies_train = strategies_features_df.iloc[train_start:train_end]
        dates_train = dates[train_start:train_end]
        
        returns_test = returns_df.iloc[test_start:test_end]
        ai_test = ai_features_df.iloc[test_start:test_end]
        strategies_test = strategies_features_df.iloc[test_start:test_end]
        dates_test = dates[test_start:test_end]
        
        train_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(
            returns_train, ai_train, strategies_train, weights_dim, num_strategies_features, tickers=tickers, dates=dates_train
        )])
        
        if model is None:
            print("Khởi tạo Mạng Nơ-ron AI hoàn toàn mới...")
            train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
            model = PPO("MlpPolicy", train_env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                n_steps=1024,
                learning_rate=0.00005,
                ent_coef=CONFIG.ENT_COEF,
                batch_size=90,
                use_sde=True,          # Bật SDE
                sde_sample_freq=4      # Đổi hướng khám phá 4 ngày 1 lần (khoảng 1 tuần giao dịch)
               )
        else:
            print("Nạp kiến thức cũ, chuyển Sàn môi trường để học tiếp (Continual Learning)...")
            train_env = VecNormalize.load("vec_normalize.pkl", train_env)
            train_env.training = True
            train_env.norm_reward = True
            model.set_env(train_env)
            
        print(f"Đang huấn luyện mô hình ({CONFIG.TRAINING_TIMESTEPS} steps)...")
        model.learn(total_timesteps=CONFIG.TRAINING_TIMESTEPS)
        
        model.save(getattr(CONFIG, 'SAVE_MODEL_PATH', "AI_Brain_Current.zip"))
        train_env.save("vec_normalize.pkl")
        
        print(f"Đang chạy Backtest thực chiến (CÓ TÍNH PHÍ GIAO DỊCH {CONFIG.COST_RATE * 100}%)...")
        print("-------------------------------------------------------")
        
        test_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(
            returns_test, ai_test, strategies_test, weights_dim, num_strategies_features, tickers=tickers, dates=dates_test, is_test=True
        )])
        obs = test_env.reset()
        
        done = [False]
        fold_returns = []
        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if 'net_return' in info[0]:
                all_test_returns.append(info[0]['net_return'])
                fold_returns.append(info[0]['net_return'])
                

        # Tính lợi nhuận lũy kế cho chu kỳ hiện tại
        fold_profit = (np.prod(1 + np.array(fold_returns)) - 1) * 100
        print(f"💰 KẾT QUẢ CHU KỲ {fold_idx}: Lợi nhuận thực chiến = {fold_profit:+.2f}%")
        print("="*55 + "\n")
        current_start += test_window
        fold_idx += 1
        

    total_profit = (np.prod(1 + np.array(all_test_returns)) - 1) * 100
    print(f"\n🏆 TỔNG LỢI NHUẬN BACKTEST TOÀN BỘ THỜI GIAN: {total_profit:+.2f}%")
    print(f"\nĐã lưu bộ não AI mới nhất vào {getattr(CONFIG, 'SAVE_MODEL_PATH', 'AI_Brain_Current.zip')}. Có thể dùng trực tiếp cho Live Trading ngày mai!")