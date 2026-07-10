#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
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


# ## [CẤU HÌNH CHIẾN LƯỢC ĐẦU TƯ - AI TRADING]

# ## Khối cấu hình trung tâm (Control Panel) giúp bạn tinh chỉnh "tính cách" của AI 

# In[33]:


# mà không cần can thiệp sâu vào bên trong mã nguồn.
class CONFIG:
    # 1. PHÍ GIAO DỊCH (Transaction Cost): 
    # Tỷ lệ phần trăm CTCK thu cho mỗi lần bạn mua hoặc bán cổ phiếu. (VD: 0.2% = 0.002)
    COST_RATE = 0.0003

    # 2. CHU KỲ REBALANCE (Rule 1: Holding Period):
    # Thời gian (tính bằng ngày) AI bị ép phải "ôm chặt" danh mục trước khi được phép mua bán lại.
    # Mục đích: Giảm thiểu Over-trading (giao dịch quá nhiều) để không bị bào mòn tài khoản bởi phí giao dịch.
    HOLDING_PERIOD = 1

    # 3. NGƯỠNG TỰ TIN VÀO CỔ PHIẾU (Rule 5: Cash Conviction Threshold):
    # Mức độ tự tin tối thiểu (từ 0.0 đến 1.0) AI cần có để quyết định giải ngân.
    # Nếu mức tự tin cao nhất của AI thấp hơn ngưỡng này (thị trường đang quá rủi ro/khó đoán), 
    # nó sẽ tự động chừa lại phần vốn đó dưới dạng Tiền mặt để phòng thủ.
    CASH_CONVICTION_THRESHOLD = 0.2

    # 4. HÌNH PHẠT RỦI RO (Rule 3: Sharpe Penalty):
    # Mức điểm âm sẽ phạt AI nếu danh mục của nó chọn bị Lỗ sau 1 chu kỳ Holding Period.
    # Con số càng âm (VD: -2.0 hoặc -5.0) sẽ rèn luyện cho AI tính cẩn thận, chỉ mua mã an toàn.
    PENALTY_FOR_LOSS = -1.0

    # 5. CẮT LỖ ĐỘNG (Rule 6: Stop-Loss):
    # Mức rớt giá tối đa cho phép. Nếu đang trong thời gian "ôm chặt" mà danh mục sập chạm mốc này,
    # lệnh ngắt mạch sẽ kích hoạt: Bán tháo toàn bộ danh mục ngay lập tức để rút về Tiền mặt bảo toàn tính mạng.
    STOP_LOSS_THRESHOLD = -0.07

    # 6. CHỐT LỜI SỚM (Rule 6: Take-Profit):
    # Mức lãi kỳ vọng. Nếu danh mục sinh lời nhanh và đạt mốc này trước khi hết hạn ôm,
    # kích hoạt ngắt mạch: Chốt lời toàn bộ, bỏ túi tiền mặt và đứng ngoài thị trường chờ chu kỳ mới.
    TAKE_PROFIT_THRESHOLD = 0.15

    # 7. CHẾ ĐỘ CHỜ HÀNG VỀ T+2 (Rule 7: Settlement Lock):
    # Đặc sản của Chứng khoán VN. Số ngày cổ phiếu bị "nhốt" chưa về tài khoản. 
    # Trong những ngày này (VD: 2 ngày đầu tiên), AI tuyệt đối không thể bán cắt lỗ/chốt lời dù thị trường sập.
    T_PLUS_SETTLEMENT = 3

    # 8. CHIA KHÚC DỮ LIỆU ĐỂ HỌC VÀ THI (Rule 4: Walk-Forward Folds):
    # Kỹ thuật chống "Học Vẹt" (Overfitting). Dữ liệu sẽ được chia làm N phần. 
    # AI sẽ học ở quá khứ và bị quăng vào tương lai xa lạ để thi đấu.
    WALK_FORWARD_SPLITS = 3

    # 8. SỐ BƯỚC ĐÀO TẠO (Training Timesteps):
    # Số vòng lặp để AI cập nhật bộ não. Càng cao (50k - 100k) AI càng thông minh nhưng thời gian train càng lâu.
    TRAINING_TIMESTEPS = 15000

    # 9. ĐỘ PHÂN BỔ VỐN (Entropy Coefficient):
    # Quyết định AI sẽ All-in hay Rải rác.
    # Giá trị nhỏ (0.001) -> Ưu tiên All-in vài mã mạnh nhất.
    # Giá trị lớn (0.05) -> Ưu tiên chia đều tiền ra mua nhiều mã để phân tán rủi ro.
    ENT_COEF = 0.001

    # Neuron Network


# ## LUẬT 2: MARKET CONTEXT (TỔNG HỢP VĨ MÔ & VI MÔ)

# ## Hàm này nạp dữ liệu từ file Parquet (đã chạy HMM) và chế biến thêm các chỉ báo để tạo thành 

# In[34]:


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

    # Bollinger Bands
    bb_mid_df = close_df.rolling(20).mean()
    std_df = close_df.rolling(20).std(ddof=0)
    bb_lower_df = bb_mid_df - 2 * std_df

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
    highest_high_10_df = high_df.rolling(10).max()
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
    # BẢNG 1: Dành cho Mạng Nơ-ron (Chỉ 10 tính năng cốt lõi)
    ai_features_df = pd.concat([
        prob0_df, prob1_df, prob2_df, vol_20d_df, ret_20d_df, vol_ratio_df, 
        mkt_ret5_df, mkt_ret20_df, mkt_vol_df, dist_ma20_df
    ], axis=1).fillna(0)

    # BẢNG 2: Kho vũ khí cho kịch bản của con người (AI không được thấy)

    list_strategies = [vol_df ,low_df ,close_df, 
                       ema_df_20, ema_df_50, ema_df_200, 
                       bb_lower_df, rsi_df, hist_df, 
                       slope_ema20_3_df, slope_sma5_3_df, 
                       is_hammer_df.astype(float), is_bull_engulf_df.astype(float),
                       lowest_low_5_df, highest_high_10_df]

    strategies_features_df = pd.concat(list_strategies, axis=1).fillna(0)

    num_strategies_features = 15

    # Đồng bộ độ dài
    returns_df, ai_features_df = returns_df.align(ai_features_df, axis=0, join='inner')
    returns_df, strategies_features_df = returns_df.align(strategies_features_df, axis=0, join='inner')
    tickers = returns_df.columns.tolist()

    return returns_df, ai_features_df, strategies_features_df, weights_dim, tickers, num_strategies_features


# ## LUẬT 1, 3, 5, 6: MÔI TRƯỜNG ĐẦU TƯ (GYM ENVIRONMENT)

# ## Lớp này mô phỏng lại Sàn chứng khoán. Nơi AI sẽ thử nghiệm các lệnh Mua/Bán và nhận Phạt/Thưởng (Reward).

# In[36]:


class AdvancedPortfolioEnv(gym.Env):
        def __init__(self, returns_df, ai_features_df, script_features_df, weights_dim, num_script_features, tickers=None, step_size=CONFIG.HOLDING_PERIOD, cost_rate=CONFIG.
  COST_RATE, is_test=False):
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

            self.action_space = spaces.Box(low=0, high=1, shape=(self.weights_dim,), dtype=np.float32)
            # Báo cho AI biết là nó chỉ cần học 10 tính năng (vì ai_features_df có 10 cột)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.weights_dim, 10), dtype=np.float32)

            self.current_step = 0
            self.weights = np.zeros(self.weights_dim)
            self.entry_prices = np.zeros(self.weights_dim)
        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.weights = np.zeros(self.weights_dim)
            return self._get_obs(), {}

        def _get_obs(self):
            idx = min(self.current_step, self.n_steps - 1)
            obs_1d = self.features_arr[idx]
            obs_2d = obs_1d.reshape(10, self.weights_dim).T
            return obs_2d.astype(np.float32)

        def step(self, action):
            current_obs = self._get_obs()
            idx = min(self.current_step, self.n_steps - 1)

            # Bóc tách kho vũ khí kịch bản tại ngày hôm nay
            current_scripts = self.script_arr[idx].reshape(self.num_script_features, self.weights_dim).T

            # Dữ liệu hôm qua (t - 1)
            prev_idx = max(0, idx - 1)
            prev_scripts = self.script_arr[prev_idx].reshape(self.num_script_features, self.weights_dim).T
            # ==============================================================
            # KIỂM DUYỆT VÀ GHI ĐÈ LỆNH CỦA AI BẰNG KỊCH BẢN CỦA CON NGƯỜI
            # ==============================================================
            for i in range(self.weights_dim):
                current_regime = np.argmax([current_obs[i][0], current_obs[i][1], current_obs[i][2]]) # 0: Bear, 1: SideWay, 2: Bull

                # 13 chỉ báo kịch bản 
                volume          = current_scripts[i][0]
                low             = current_scripts[i][1]
                close           = current_scripts[i][2]
                ema20           = current_scripts[i][3]
                ema50           = current_scripts[i][4]
                ema200          = current_scripts[i][5]
                bb_lower        = current_scripts[i][6]
                rsi             = current_scripts[i][7]
                macd_hist       = current_scripts[i][8]
                slope_ema20_3     = current_scripts[i][9]
                slope_sma5_3      = current_scripts[i][10]
                is_hammer       = current_scripts[i][11]
                is_bull_engulf  = current_scripts[i][12]
                lowest_low_5    = current_scripts[i][13]
                highest_high_10 = current_scripts[i][14]
                # chỉ báo hôm qua
                prev_volume = prev_scripts[i][0]

                # ---- KỊCH BẢN ĐẦU TƯ TRỰC TIẾP ----
                # -----------------------------------
                # Regime Ticker
                # -----------------------------------
                # =======================================================
                # CỖ MÁY TRẠNG THÁI (STATE MACHINE) CỦA MỘT TRADER ĐÍCH THỰC
                # Luôn phải kiểm tra VỊ THẾ đầu tiên, KHÔNG ĐƯỢC nhét vào trong kịch bản
                # =======================================================

                if self.weights[i] > 0: 
                    # ---------------------------------------------------
                    # TRẠNG THÁI 1: ĐANG GIỮ HÀNG -> CHỈ QUAN TÂM ĐẾN EXIT
                    # ---------------------------------------------------
                    entry_price = self.entry_prices[i] 

                    # Tính mức Risk để chốt lời TP2
                    stop_loss_level = max(ema50, lowest_low_5 * 0.98)
                    risk = entry_price - stop_loss_level if entry_price > stop_loss_level else (entry_price * 0.05)

                    # [SL] Cắt Lỗ
                    if close < ema50 or close < (lowest_low_5 * 0.98): 
                        action[i] = 0.0
                        self.entry_prices[i] = 0.0

                    # [TP1] Chốt đỉnh cũ
                    elif close >= highest_high_10: 
                        action[i] = 0.0
                        self.entry_prices[i] = 0.0

                    # [TP2] Chốt R:R 1:2
                    elif close >= (entry_price + 2 * risk): 
                        action[i] = 0.0
                        self.entry_prices[i] = 0.0

                    # [TP3] Trailing Stop thủng EMA20
                    elif close < ema20: 
                        action[i] = 0.0
                        self.entry_prices[i] = 0.0

                    # Chưa thoả mãn gì thì ép gồng lãi tiếp
                    else:
                        action[i] = max(action[i], self.weights[i])

                else:
                    # ---------------------------------------------------
                    # TRẠNG THÁI 2: CHƯA CÓ HÀNG -> TÌM ĐIỂM MUA VÀ LỌC LỆNH AI
                    # ---------------------------------------------------
                    is_buy_signal = False

                    # QUÉT CÁC KỊCH BẢN KỸ THUẬT
                    if current_regime == 2: # Chỉ săn mồi khi đang Bull
                        if (ema20 > ema50) and (close > ema50) and (slope_ema20_3 > 0):
                            if (low <= ema20) and (40 <= rsi <= 55) and (is_bull_engulf == 1.0 or is_hammer == 1.0) and (volume > prev_volume):
                                is_buy_signal = True 
                                print(f"[{self.tickers[i]}] EMA Pullback Triggered!")

                    # BỘ LỌC CẮT LỚP AI (Tôn trọng tuyệt đối quyết định tỷ trọng của AI)
                    if is_buy_signal == True:
                        if action[i] > 0.0:
                            # AI MỞ LỆNH: Cứ để nguyên biến action[i] của AI, không gán = 0.2 nữa. 
                            # Bạn chỉ cần thực hiện ghi chép sổ sách:
                            self.entry_prices[i] = close 
                        else:
                            # AI TỪ CHỐI MỞ LỆNH: Kịch bản đẹp nhưng AI chê.
                            # Vì bạn nói "không cần bổ sung vào", ta sẽ Tôn trọng AI tuyệt đối -> Bỏ qua mã này!
                            action[i] = 0.0
                    else:
                        # KỊCH BẢN KHÔNG THỎA MÃN: Bất chấp AI có thèm mua, Tước quyền giải ngân!
                        action[i] = 0.0

            conviction = np.max(action)
            investment_ratio = conviction / CONFIG.CASH_CONVICTION_THRESHOLD if conviction < CONFIG.CASH_CONVICTION_THRESHOLD else 1.0 

            action = action / (np.sum(action) + 1e-9)
            action = action * investment_ratio 

            cost = self.cost_rate * np.sum(np.abs(action - self.weights))
            self.weights = action

            end_step = min(self.current_step + self.step_size, self.n_steps)
            days_held = end_step - self.current_step

            if days_held == 0:
                return self._get_obs(), 0, True, False, {}

            daily_portfolio_returns = []
            cum_ret_since_rebalance = 0.0
            is_cash_mode = False

            for t in range(self.current_step, end_step):
                if is_cash_mode:
                    daily_portfolio_returns.append(0.0)
                    continue

                daily_ret = np.sum(self.weights * self.returns_arr[t])
                if t == self.current_step:
                    daily_ret -= cost

                daily_portfolio_returns.append(daily_ret)
                cum_ret_since_rebalance = (1 + cum_ret_since_rebalance) * (1 + daily_ret) - 1
                days_since_buy = t - self.current_step

                # Ngắt mạch cắt lỗ chốt lời theo T+2
                if days_since_buy >= CONFIG.T_PLUS_SETTLEMENT:
                    if cum_ret_since_rebalance <= CONFIG.STOP_LOSS_THRESHOLD or cum_ret_since_rebalance >= CONFIG.TAKE_PROFIT_THRESHOLD:
                        exit_cost = self.cost_rate * np.sum(self.weights)
                        daily_portfolio_returns[-1] -= exit_cost 
                        self.weights = np.zeros(self.weights_dim) 
                        is_cash_mode = True 

            daily_portfolio_returns = np.array(daily_portfolio_returns)
            cum_return = np.prod(1 + daily_portfolio_returns) - 1

            mean_ret = np.mean(daily_portfolio_returns)
            std_ret = np.std(daily_portfolio_returns) + 1e-9
            sharpe = (mean_ret / std_ret) * np.sqrt(252) 

            reward = sharpe + CONFIG.PENALTY_FOR_LOSS if cum_return < 0 else sharpe

            self.current_step = end_step
            done = self.current_step >= self.n_steps - 1 

            return self._get_obs(), float(reward), done, False, {'net_return': cum_return}


# ## MẠNG NƠ-RON TRÍ TUỆ NHÂN TẠO (NEURAL NETWORK STRUCTURE)

# ## Đây là "Bộ Não" thực sự của AI. Mạng Nơ-ron này chịu trách nhiệm Đọc dữ liệu (Extractor).

# In[37]:


class AdvancedTickerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        num_tickers = observation_space.shape[0] # 46 mã
        num_features_per_ticker = observation_space.shape[1] # 11 tính năng

        # Tầng 1 (Local): Học phân tích RIÊNG LẺ từng mã cổ phiếu (Ticker-level)
        self.ticker_net = nn.Sequential(
            nn.Linear(num_features_per_ticker, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Tầng 2 (Global): Tổng hợp toàn bộ 46 mã lại để ra quyết định Toàn cục
        self.global_net = nn.Sequential(
            nn.Linear(num_tickers * 16, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size, num_tickers, num_features = observations.shape
        # Nhồi 11 tính năng vào Tầng 1
        obs_reshaped = observations.view(batch_size * num_tickers, num_features)
        ticker_features = self.ticker_net(obs_reshaped)

        # Đưa Tầng 1 vào Tầng 2
        ticker_features = ticker_features.view(batch_size, num_tickers * 16)
        global_features = self.global_net(ticker_features)
        return global_features


# In[ ]:


def walk_forward_train_test(returns_df, ai_features_df, strategies_features_df, weights_dim, tickers, num_strategies_features):
    print("\n[LUẬT 4] KHỞI ĐỘNG WALK-FORWARD VALIDATION (CHỐNG HỌC VẸT)...")

    total_days = len(returns_df)
    chunk_size = total_days // CONFIG.WALK_FORWARD_SPLITS

    # Chia dữ liệu: Trong Fold 1, Học từ Khúc 0 đến Khúc 2, Thi đấu Khúc 2 đến Cuối cùng
    folds = [
        (0, chunk_size * 2, chunk_size * 2, total_days) 
    ]

    # Gắn Não (Neural Network) vào Trái Tim (PPO Algorithm)
    policy_kwargs = dict(
        features_extractor_class=AdvancedTickerExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"\n=======================================================")
        print(f"--- FOLD {fold_idx + 1}: HỌC TỪ NGÀY {train_start}->{train_end}, THI ĐẤU NGÀY {test_start}->{test_end} ---")

        # Dữ liệu Train
        returns_train = returns_df.iloc[train_start:train_end]
        features_train = features_df.iloc[train_start:train_end]

        # Dữ liệu Test (Tương lai chưa biết)
        returns_test = returns_df.iloc[test_start:test_end]
        features_test = features_df.iloc[test_start:test_end]

        # Tạo Sàn Chứng Khoán Giả lập cho quá trình Học
        train_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(returns_train, features_train, weights_dim)])

        # Khởi tạo Mô hình Trí tuệ Nhân tạo - Sử dụng Giải thuật PPO (Thuật toán số 1 hiện nay cho Trading)
        model = PPO("MlpPolicy", train_env, 
                    policy_kwargs=policy_kwargs, 
                    verbose=1, # Bật hiển thị Log lúc Học để theo dõi sức khỏe AI (loss, entropy...)
                    n_steps=1024,
                    learning_rate=0.0003,
                    ent_coef=CONFIG.ENT_COEF,
                    batch_size=64)

        print(f"Đang huấn luyện mô hình ({CONFIG.TRAINING_TIMESTEPS} steps)...")
        # Kích hoạt quá trình Học tập
        model.learn(total_timesteps=CONFIG.TRAINING_TIMESTEPS)

        print(f"Đang chạy Backtest thực chiến (CÓ TÍNH PHÍ GIAO DỊCH {CONFIG.COST_RATE * 100}%)...")
        print("-------------------------------------------------------")

        # Tạo Sàn Chứng Khoán Thật cho quá trình Thi Đấu (is_test=True để in Log)
        test_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(returns_test, features_test, weights_dim, tickers=tickers, is_test=True)])
        obs = test_env.reset()

        portfolio_returns = []
        done = False

        # Vòng lặp Thi đấu: Từ Ngày đầu tiên của Tương lai đến Ngày cuối cùng
        while not done:
            # AI tự nhìn Biểu đồ và Quyết định Xuống tiền
            action, _ = model.predict(obs, deterministic=True)
            # Hệ thống ghi nhận Quyết định, trừ Tiền Phí và Phản hồi Kết Quả (Lãi/Lỗ)
            obs, reward, done, info = test_env.step(action)

            # Ghi chép Lợi nhuận
            if 'net_return' in info[0]:
                portfolio_returns.append(info[0]['net_return'])

        # Tổng kết Kì thi
        test_len = len(returns_test)
        cum_ret = (np.prod(1 + np.array(portfolio_returns)) - 1) * 100
        print(f"=> [KẾT QUẢ] LỢI NHUẬN TÍCH LŨY FOLD {fold_idx + 1} (SAU {test_len} NGÀY THỰC CHIẾN): {cum_ret:.2f}%")

if __name__ == "__main__":
    returns_df, ai_features_df, strategies_features_df, weights_dim, tickers, num_strategies_features = load_data()
    walk_forward_train_test(returns_df, ai_features_df, strategies_features_df, weights_dim, tickers, num_strategies_features)


# ## LUẬT 4: WALK-FORWARD VALIDATION (CHỐNG HỌC VẸT)

# ## Đạo diễn toàn bộ quá trình: Chia dữ liệu -> Gọi AI vào Học (Train) -> Bắt AI đi Thi (Test) -> Báo cáo Lợi nhuận
