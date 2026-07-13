import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# mà không cần can thiệp sâu vào bên trong mã nguồn.
class CONFIG:
    TRAINING_MODE = 'fast_split' # 'fast_split' or 'rolling_window'
    # 1. PHÍ GIAO DỊCH (Transaction Cost): 
    # Tỷ lệ phần trăm CTCK thu cho mỗi lần bạn mua hoặc bán cổ phiếu. (VD: 0.2% = 0.002)
    COST_RATE = 0.0001

    TURNOVER_PENALTY_RATE = 0.3
    DRAWDOWN_PENALTY_RATE = 5.0
    LIQUIDITY_BONUS_RATE = 0.02
    TRAIN_WINDOW = 252
    TEST_WINDOW = 21
    SAVE_MODEL_PATH = "v7_3/AI_Brain_v7_3.zip"

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
    TRAINING_TIMESTEPS = 1_500_000
    
    # 9. ĐỘ PHÂN BỔ VỐN (Entropy Coefficient):
    # Quyết định AI sẽ All-in hay Rải rác.
    # Giá trị nhỏ (0.001) -> Ưu tiên All-in vài mã mạnh nhất.
    # Giá trị lớn (0.05) -> Ưu tiên chia đều tiền ra mua nhiều mã để phân tán rủi ro.
    ENT_COEF = 0.005

    # 10. NEURAL NETWORK & PPO HYPERPARAMETERS
    FEATURES_DIM = 256
    NET_ARCH = [64, 64]
    N_STEPS = 1024
    LEARNING_RATE = 0.00005
    BATCH_SIZE = 90
    USE_SDE = False
    SDE_SAMPLE_FREQ = 5

    # Neuron Network

def load_data():
    # 1. Nạp dữ liệu thô
    raw_df = pd.read_parquet(r"output/master_drl_ready_ticker.parquet")
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
    momentum_3d_df = (close_df / close_df.shift(3) - 1).fillna(0) # T+3 Momentum

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
    slope_sma5_3_df = calc_sma(sma_vol_5_df, n=20)
    slope_sma5_50_df = calc_sma(sma_vol_5_df, n=50)
    slope_sma5_200_df = calc_sma(sma_vol_5_df, n=200)
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
    # V6 Fix: Tính đỉnh/đáy bằng trailing max/min (Bảo vệ tuyệt đối khỏi Lookahead Bias)
    rolling_max = high_df.rolling(window=2 * n + 1, min_periods=1).max()
    rolling_min = low_df.rolling(window=2 * n + 1, min_periods=1).min()

    is_peak = (high_df == rolling_max)
    is_valley = (low_df == rolling_min)
    
    # 3. Ghi nhớ mức giá tại Đỉnh/Đáy đó
    peak_values = high_df.where(is_peak, np.nan)
    valley_values = low_df.where(is_valley, np.nan)
    
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
    prob0_df = raw_df.pivot(index='time', columns='ticker', values='prob_market_0').fillna(0) 
    prob1_df = raw_df.pivot(index='time', columns='ticker', values='prob_market_1').fillna(0) 
    prob2_df = raw_df.pivot(index='time', columns='ticker', values='prob_market_2').fillna(0) 
    prob3_df = raw_df.pivot(index='time', columns='ticker', values='prob_market_3').fillna(0)
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
    core_list = [prob0_df, prob1_df, prob2_df, vol_20d_df, ret_20d_df, vol_ratio_df, mkt_ret5_df, mkt_ret20_df, mkt_vol_df, dist_ma20_df, momentum_3d_df]
    ai_features_df = pd.concat([apply_nqt(df) for df in core_list], axis=1).fillna(0)
                                
    # BẢNG 2: Kho vũ khí cho kịch bản của con người (AI không được thấy)

    list_strategies = [vol_df ,low_df ,close_df, 
                       ema_df_20, ema_df_50, ema_df_200, 
                       bb_lower_df, bb_middle_df ,bb_upper_df,
                       rsi_df, hist_df, macd_df,
                       sma_vol_20_df,
                       slope_ema20_3_df, slope_sma5_3_df, 
                       slope_sma5_50_df, slope_sma5_200_df,
                       is_hammer_df.astype(float), is_bull_engulf_df.astype(float),
                       lowest_low_5_df, highest_high_5_df ,highest_high_10_df, highest_high_20_df,
                       fib_382_df, fib_500_df, fib_618_df, is_upward_wave_df, fib_ext_127_df, fib_ext_161_df,
                       sup_df, res_df,
                       bb_expanding_df,
                       is_accumulation_df,
                       ]

    strategies_features_df = pd.concat(list_strategies, axis=1).fillna(0)
    
    num_strategies_features = 33
    
    # HỢP NHẤT TOÀN BỘ CHỈ BÁO VÀO AI FEATURES (Áp dụng NQT cho mảng chiến lược đưa vào AI)
    ai_strategies_df = pd.concat([apply_nqt(df) for df in list_strategies], axis=1).fillna(0)
    ai_features_df = pd.concat([ai_features_df, ai_strategies_df], axis=1)
    num_ai_features = 10
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
            self.num_features = self.features_arr.shape[1] // self.weights_dim
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.weights_dim, self.num_features), dtype=np.float32)
            
            self.current_step = 0
            self.weights = np.zeros(self.weights_dim)
            self.entry_prices = np.zeros(self.weights_dim)
            self.entry_scenarios = np.zeros(self.weights_dim)
            self.entry_stages = np.zeros(self.weights_dim)
            self.weight_unlocked = np.zeros(self.weights_dim)
            self.locked_weights = np.zeros((3, self.weights_dim)) # FIFO Queue
            self.avg_entry_prices = np.zeros(self.weights_dim)
            self.initial_capital = 100_000_000 # Giả lập vốn 100 triệu
            self.current_capital = self.initial_capital
            self.peak_nav = self.initial_capital
            self.prev_drawdown = 0.0
        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.weights = np.zeros(self.weights_dim)
            self.weight_unlocked = np.zeros(self.weights_dim)
            self.locked_weights = np.zeros((3, self.weights_dim))
            self.avg_entry_prices = np.zeros(self.weights_dim)
            self.current_capital = self.initial_capital
            self.peak_nav = self.initial_capital
            self.prev_drawdown = 0.0
            return self._get_obs(), {}
            
        def _get_obs(self):
            idx = min(self.current_step, self.n_steps - 1)
            obs_1d = self.features_arr[idx]
            obs_2d = obs_1d.reshape(self.num_features, self.weights_dim).T
            return obs_2d.astype(np.float32)
            
        def step(self, action):
            current_obs = self._get_obs()
            idx = min(self.current_step, self.n_steps - 1)
            
            current_scripts = self.script_arr[idx].reshape(self.num_script_features, self.weights_dim).T
            prev_idx = max(0, idx - 1)
            prev_scripts = self.script_arr[prev_idx].reshape(self.num_script_features, self.weights_dim).T
            
            # V6 Fix: Xóa Kịch bản Hard-code
            # V6 Fix: Tuyến tính hóa thay vì Softmax
            # Normalize action và sửa lỗi logic Zero-out của bản cũ
            action = np.clip(action, 0, 1)
            action_sum = np.sum(action)
            if action_sum > 1.0:
                action = action / action_sum
                
            # [HÀNG ĐỢI FIFO] Cập nhật hàng về tài khoản
            self.weight_unlocked += self.locked_weights[0]
            self.locked_weights[0] = self.locked_weights[1]
            self.locked_weights[1] = self.locked_weights[2]
            self.locked_weights[2] = 0.0
            
            # [FIFO] Áp dụng luật khoá: Chỉ được bán tối đa bằng số hàng đã về (unlocked)
            max_sell = self.weight_unlocked
            action = np.maximum(action, self.weights - max_sell)
            
            # Nếu vì ép bán ít đi mà tổng action > 1.0, ta phải giảm bớt lượng mua (buy)
            action_sum = np.sum(action)
            if action_sum > 1.0:
                buys = np.maximum(0, action - self.weights)
                sells = np.maximum(0, self.weights - action)
                excess = action_sum - 1.0
                total_buys = np.sum(buys)
                if total_buys > 0:
                    reduction_factor = 1.0 - (excess / total_buys)
                    buys = buys * reduction_factor
                action = self.weights - sells + buys
                
            # [FIFO] Cập nhật lại kho sau khi mua/bán
            delta = action - self.weights
            sells = np.maximum(0, -delta)
            buys = np.maximum(0, delta)
            
            self.weight_unlocked -= sells
            
            T_SETTLE = getattr(CONFIG, 'T_PLUS_SETTLEMENT', 3)
            if T_SETTLE == 0:
                self.weight_unlocked += buys
            else:
                idx = min(T_SETTLE - 1, 2)
                self.locked_weights[idx] += buys
                
            turnover = np.sum(np.abs(action - self.weights))
            cost = self.cost_rate * turnover
            
            self.prev_weights = self.weights.copy()
            self.weights = action
            
            market_ret = np.mean(self.returns_arr[self.current_step])
            daily_ret = np.sum(self.weights * self.returns_arr[self.current_step]) - cost
            
            self.current_portfolio_value = self.current_portfolio_value * (1 + daily_ret)
            
            # V7.2 Fix: Hàm Reward mượt mà, tránh sợ hãi cực độ
            
            base_reward = daily_ret * 100 
            if daily_ret < 0:
                base_reward *= 2.0  # Phạt gấp 2 lần nếu lỗ
            
            reward = base_reward
            
            if getattr(self, 'is_test', False):

            
                date_str = self.dates[self.current_step] if getattr(self, 'dates', None) else f'Step {self.current_step}'

            
                

            
                # Tracker for detailed logs

            
                trade_logs = []

            
                current_prices = current_scripts[:, 2] # close price is index 2

            
                

            
                for i in range(self.weights_dim):

            
                    old_w = self.prev_weights[i]

            
                    new_w = self.weights[i]

            
                    

            
                    if new_w > old_w + 0.001: # Mua thêm

            
                        added_w = new_w - old_w

            
                        if old_w < 0.001:

            
                            self.avg_entry_prices[i] = current_prices[i]

            
                        else:

            
                            self.avg_entry_prices[i] = (old_w * self.avg_entry_prices[i] + added_w * current_prices[i]) / new_w

            
                        trade_logs.append(f"🟢 MUA {self.tickers[i]}: Tỷ trọng {old_w*100:.1f}% -> {new_w*100:.1f}% | Giá khớp: {current_prices[i]:.2f}")

            
                        

            
                    elif new_w < old_w - 0.001: # Bán

            
                        sold_w = old_w - new_w

            
                        if self.avg_entry_prices[i] > 0:

            
                            profit_pct = (current_prices[i] - self.avg_entry_prices[i]) / self.avg_entry_prices[i]

            
                            profit_pct -= self.cost_rate # Tính luôn thuế phí

            
                            profit_money = (sold_w * self.current_capital) * profit_pct

            
                            

            
                            icon = "🔥 CHỐT LỜI" if profit_pct > 0 else "🩸 CẮT LỖ"

            
                            trade_logs.append(f"{icon} {self.tickers[i]}: Tỷ trọng {old_w*100:.1f}% -> {new_w*100:.1f}% | Giá bán: {current_prices[i]:.2f} | Giá vốn: {self.avg_entry_prices[i]:.2f} | Lãi/Lỗ: {profit_pct*100:+.2f}% ({profit_money:+,.0f} đ)")

            
                        if new_w < 0.001:

            
                            self.avg_entry_prices[i] = 0

            
                

            
                # Cập nhật vốn hiện tại (NAV)

            
                prev_capital = self.current_capital

            
                self.current_capital = self.current_capital * (1 + daily_ret)

            
                daily_capital_change = self.current_capital - prev_capital

            
                

            
                weight_sum = np.sum(self.weights)

            
                if len(trade_logs) > 0 or weight_sum > 0.01:

            
                    cash_weight = max(0.0, 1.0 - weight_sum)

            
                    allocations = [f"{self.tickers[k]}: {self.weights[k]*100:.1f}%" for k in range(self.weights_dim) if self.weights[k] > 0.01]

            
                    alloc_str = ", ".join(allocations) if allocations else "Trống"

            
                    

            
                    print(f"\n[{date_str}] 📅 BÁO CÁO GIAO DỊCH:")

            
                    for log in trade_logs:

            
                        print(f"  > {log}")

            
                    print(f"  💵 Tỷ trọng: Tiền mặt {cash_weight*100:.1f}% | Cổ phiếu: [{alloc_str}]")

            
                    print(f"  📈 Tài sản: {daily_capital_change:+,.0f} đ | Hiệu suất ngày: {daily_ret*100:+.2f}% | Tổng NAV: {self.current_capital:,.0f} đ")

            
                    

            
            # ------------ HỆ THỐNG PHẠT DRAWDOWN & GAME OVER ------------
            self.peak_nav = max(self.peak_nav, getattr(self, 'current_capital', 100_000_000))
            drawdown = (self.peak_nav - getattr(self, 'current_capital', 100_000_000)) / self.peak_nav
            
            prev_dd = getattr(self, 'prev_drawdown', 0.0)
            if drawdown > prev_dd:
                dd_increase = drawdown - prev_dd
                reward -= (dd_increase * 1000)
            
            self.prev_drawdown = drawdown
            
            force_terminate = False
            if drawdown > 0.30:
                reward -= 100
                force_terminate = True
                if getattr(self, 'is_test', False):
                    print(f"💀 GAME OVER: Cháy tài khoản! Drawdown {drawdown*100:.1f}% tại Step {self.current_step}")
            # -----------------------------------------------------------
            
            self.current_step += 1
            done = (self.current_step >= self.n_steps - 1) or force_terminate
            
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

def main():
    print('=====================================================')
    print('CHẾ ĐỘ LIVE TRADING: DRL PPO META-AGENT')
    print('=====================================================')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "model", "v7_3", "AI_Brain_v7_Seed321_Profit_50.80.zip")
    env_path = os.path.join(script_dir, "..", "model", "v7_3", "vec_normalize.pkl")
    
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy model tại {model_path}")
        return
        
    print("1. Nạp dữ liệu thị trường và trích xuất Features...")
    returns_df, ai_features_df, strategies_features_df, weights_dim, tickers, num_strategies_features, dates = load_data()
    
    print("2. Khởi tạo Môi trường Giao dịch (Trading Environment) với 252 ngày gần nhất...")
    # Lấy 252 ngày cuối cùng để Agent làm nóng (warmup) trạng thái danh mục
    warmup_days = min(252, len(returns_df))
    returns_test = returns_df.iloc[-warmup_days:]
    ai_test = ai_features_df.iloc[-warmup_days:]
    strategies_test = strategies_features_df.iloc[-warmup_days:]
    dates_test = dates[-warmup_days:]
    
    test_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(
            returns_test, ai_test, strategies_test, weights_dim, num_strategies_features, tickers=tickers, dates=dates_test, is_test=True
    )])
    
    try:
        test_env = VecNormalize.load(env_path, test_env)
        test_env.training = False
        test_env.norm_reward = False
    except Exception as e:
        print(f"Cảnh báo: Không thể nạp vec_normalize.pkl: {e}")
    
    print(f"3. Tải bộ não AI (PPO) từ: {model_path}")
    model = PPO.load(model_path)
    
    print("4. Mô phỏng danh mục từ quá khứ đến hiện tại để đồng bộ trạng thái (Warmup)...")
    obs = test_env.reset()
    done = [False]
    
    last_action = None
    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        last_action = action[0] # Lấy action của phần tử đầu tiên trong VecEnv
        
    print("\\nHoàn tất mô phỏng! Trích xuất tín hiệu giao dịch cho T+1...")
    
    # 5. Xuất kết quả
    latest_date = pd.to_datetime(dates_test[-1])
    
    results = pd.DataFrame({
        'ticker': tickers,
        'target_weight_%': np.round(last_action * 100, 2)
    })
    
    # Chỉ lọc ra các mã mà AI muốn phân bổ vốn > 0.01%
    results = results[results['target_weight_%'] > 0.01].sort_values('target_weight_%', ascending=False).reset_index(drop=True)
    
    print(f"\\n🏆 TỶ TRỌNG DANH MỤC KHUYẾN NGHỊ CHO NGÀY MAI ({latest_date.strftime('%Y-%m-%d')}):")
    print(results.to_string(index=False))
    
    output_csv = os.path.join(script_dir, "output", f"drl_target_weights_{latest_date.strftime('%Y%m%d')}.csv")
    results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\\n[OK] Đã xuất toàn bộ bảng tỷ trọng mục tiêu ra: {output_csv}")

if __name__ == "__main__":
    main()
