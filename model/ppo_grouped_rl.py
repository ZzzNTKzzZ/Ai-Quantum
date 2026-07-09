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
warnings.filterwarnings('ignore')

# ==========================================
# [CẤU HÌNH CHIẾN LƯỢC ĐẦU TƯ - AI TRADING]
# ==========================================
class CONFIG:
    # 1. Phí giao dịch của CTCK cho mỗi lần mua/bán (VD: 0.2% = 0.002)
    COST_RATE = 0.002
    
    # 2. Luật 1: Chu kỳ Rebalance. AI sẽ bị ép "ôm chặt" danh mục trong bao nhiêu ngày trước khi mua bán tiếp?
    HOLDING_PERIOD = 30
    
    # 3. Luật 5: Ngưỡng tự tin tối thiểu (Cash Conviction Threshold). 
    CASH_CONVICTION_THRESHOLD = 0.2
    
    # 4. Luật 3: Mức phạt nếu AI chọn danh mục mà sau đó bị lỗ (Sharpe Penalty).
    PENALTY_FOR_LOSS = -2.0
    
    # 5. LUẬT 6 (MỚI): BỎ CHẠY KHI BẤT THƯỜNG (STOP-LOSS)
    # Nếu danh mục đang ôm bị lỗ chạm mức này (VD: -7% = -0.07), ngắt mạch lập tức bán sạch thành Tiền Mặt.
    STOP_LOSS_THRESHOLD = -0.07
    
    # 6. LUẬT 6 (MỚI): CHỐT LỜI SỚM (TAKE-PROFIT)
    # Nếu danh mục sinh lời đạt mức này trước thời hạn (VD: +15% = 0.15), chốt lời ngay, chuyển thành Tiền mặt.
    TAKE_PROFIT_THRESHOLD = 0.15
    
    # 7. Luật 4: Walk-Forward Folds (Số khúc dữ liệu để test).
    WALK_FORWARD_SPLITS = 3
    
    # 8. Thời gian đào tạo AI (Số bước học).
    TRAINING_TIMESTEPS = 15000
    
    # 9. Độ tập trung vốn (Entropy Coefficient). 
    ENT_COEF = 0.001

# ==========================================
# LUẬT 2: MARKET CONTEXT (NHÌN VĨ MÔ THỊ TRƯỜNG)
# ==========================================
def load_data():
    raw_df = pd.read_parquet(r"C:\Users\ADMIN\Desktop\Kaggle\output\hmm_v3_op1_extended\master_drl_ready_full.parquet")
    raw_df['time'] = pd.to_datetime(raw_df['time'])
    
    returns_df = raw_df.pivot(index='time', columns='ticker', values='log_return').fillna(0)
    weights_dim = returns_df.shape[1]
    
    market_return_5d = returns_df.rolling(5).sum().mean(axis=1).fillna(0)
    market_return_20d = returns_df.rolling(20).sum().mean(axis=1).fillna(0)
    market_vol_20d = returns_df.rolling(20).std().mean(axis=1).fillna(0)
    
    # --- LUẬT MỚI: CHỈ SỐ MA20 VÀ GIÁ ĐÓNG CỬA ---
    # Phục hồi giá giả định từ log_return
    price_df = np.exp(returns_df.cumsum())
    # Tính đường Trung bình động 20 ngày (MA20)
    ma20_df = price_df.rolling(20).mean()
    # Feature: Khoảng cách từ Giá đóng cửa đến MA20 (tính bằng %)
    dist_ma20_df = ((price_df - ma20_df) / ma20_df).fillna(0)

    prob0_df = raw_df.pivot(index='time', columns='ticker', values='prob_ticker_0').fillna(0)
    prob1_df = raw_df.pivot(index='time', columns='ticker', values='prob_ticker_1').fillna(0)
    prob2_df = raw_df.pivot(index='time', columns='ticker', values='prob_ticker_2').fillna(0)
    vol_20d_df = raw_df.pivot(index='time', columns='ticker', values='rolling_vol_20d').fillna(0)
    ret_5d_df = raw_df.pivot(index='time', columns='ticker', values='return_5d').fillna(0)
    ret_20d_df = raw_df.pivot(index='time', columns='ticker', values='return_20d').fillna(0)
    vol_ratio_df = raw_df.pivot(index='time', columns='ticker', values='volume_ratio').fillna(0)

    for df_macro, name in zip([market_return_5d, market_return_20d, market_vol_20d], ['MKT_RET5', 'MKT_RET20', 'MKT_VOL']):
        macro_expanded = pd.DataFrame(np.tile(df_macro.values.reshape(-1, 1), (1, weights_dim)), 
                                      index=returns_df.index, columns=returns_df.columns)
        if name == 'MKT_RET5': mkt_ret5_df = macro_expanded
        elif name == 'MKT_RET20': mkt_ret20_df = macro_expanded
        elif name == 'MKT_VOL': mkt_vol_df = macro_expanded

    ticker_features_df = pd.concat([prob0_df, prob1_df, prob2_df, vol_20d_df, ret_5d_df, ret_20d_df, vol_ratio_df, 
                                    mkt_ret5_df, mkt_ret20_df, mkt_vol_df, dist_ma20_df], axis=1)
    
    returns_df, ticker_features_df = returns_df.align(ticker_features_df, axis=0, join='inner')
    return returns_df, ticker_features_df, weights_dim

# ==========================================
# LUẬT 1, 3, 5, 6: MÔI TRƯỜNG ĐẦU TƯ
# ==========================================
class AdvancedPortfolioEnv(gym.Env):
    def __init__(self, returns_df, features_df, weights_dim, step_size=CONFIG.HOLDING_PERIOD, cost_rate=CONFIG.COST_RATE, is_test=False):
        super().__init__()
        self.returns_arr = np.exp(returns_df.values) - 1.0 
        self.features_arr = features_df.values
        self.weights_dim = weights_dim
        self.n_steps = len(self.returns_arr)
        
        self.step_size = step_size
        self.cost_rate = cost_rate
        self.is_test = is_test
        
        self.action_space = spaces.Box(low=0, high=1, shape=(self.weights_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.weights_dim, 11), dtype=np.float32)
        
        self.current_step = 0
        self.weights = np.zeros(self.weights_dim)
        
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.weights = np.zeros(self.weights_dim)
        return self._get_obs(), {}
        
    def _get_obs(self):
        idx = min(self.current_step, self.n_steps - 1)
        obs_1d = self.features_arr[idx]
        obs_2d = obs_1d.reshape(11, self.weights_dim).T
        return obs_2d.astype(np.float32)
        
    def step(self, action):
        conviction = np.max(action)
        if conviction < CONFIG.CASH_CONVICTION_THRESHOLD:
            investment_ratio = conviction / CONFIG.CASH_CONVICTION_THRESHOLD
        else:
            investment_ratio = 1.0
            
        action = action / (np.sum(action) + 1e-9)
        action = action * investment_ratio 
        
        if self.is_test:
            cash_pct = (1.0 - investment_ratio) * 100
            print(f"| Ngày {self.current_step:04d} | ĐẢO HÀNG: Giải ngân {investment_ratio*100:.1f}% vốn, Tiền mặt {cash_pct:.1f}%")
        
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
            
            # Cập nhật lợi nhuận tích lũy tạm thời
            cum_ret_since_rebalance = (1 + cum_ret_since_rebalance) * (1 + daily_ret) - 1
            
            # LUẬT 6: Kiểm tra Bỏ chạy (Stop-Loss) hoặc Chốt Lời (Take-Profit)
            if cum_ret_since_rebalance <= CONFIG.STOP_LOSS_THRESHOLD or cum_ret_since_rebalance >= CONFIG.TAKE_PROFIT_THRESHOLD:
                if self.is_test:
                    if cum_ret_since_rebalance <= CONFIG.STOP_LOSS_THRESHOLD:
                        print(f"   -> [CẮT LỖ] Ngày {t:04d}: Lỗ {cum_ret_since_rebalance*100:.1f}% -> Bán tháo toàn bộ!")
                    else:
                        print(f"   -> [CHỐT LỜI] Ngày {t:04d}: Lãi {cum_ret_since_rebalance*100:.1f}% -> Cầm tiền mặt chờ tháng sau!")
                        
                # Phải mất phí giao dịch để bán tháo ra Tiền Mặt
                exit_cost = self.cost_rate * np.sum(self.weights)
                daily_portfolio_returns[-1] -= exit_cost
                self.weights = np.zeros(self.weights_dim) # Bán sạch, cầm 100% Tiền Mặt
                is_cash_mode = True
        
        daily_portfolio_returns = np.array(daily_portfolio_returns)
        cum_return = np.prod(1 + daily_portfolio_returns) - 1
        
        mean_ret = np.mean(daily_portfolio_returns)
        std_ret = np.std(daily_portfolio_returns) + 1e-9
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        
        if cum_return < 0:
            reward = sharpe + CONFIG.PENALTY_FOR_LOSS
        else:
            reward = sharpe
            
        self.current_step = end_step
        done = self.current_step >= self.n_steps - 1
        
        return self._get_obs(), float(reward), done, False, {'net_return': cum_return}

# ==========================================
# MẠNG NƠ-RON TRÍ TUỆ NHÂN TẠO
# ==========================================
class AdvancedTickerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        num_tickers = observation_space.shape[0]
        num_features_per_ticker = observation_space.shape[1]
        
        self.ticker_net = nn.Sequential(
            nn.Linear(num_features_per_ticker, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.global_net = nn.Sequential(
            nn.Linear(num_tickers * 16, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size, num_tickers, num_features = observations.shape
        obs_reshaped = observations.view(batch_size * num_tickers, num_features)
        ticker_features = self.ticker_net(obs_reshaped)
        ticker_features = ticker_features.view(batch_size, num_tickers * 16)
        global_features = self.global_net(ticker_features)
        return global_features

# ==========================================
# LUẬT 4: WALK-FORWARD VALIDATION
# ==========================================
def walk_forward_train_test(returns_df, features_df, weights_dim):
    print("\n[LUẬT 4] KHỞI ĐỘNG WALK-FORWARD VALIDATION (CHỐNG HỌC VẸT)...")
    
    total_days = len(returns_df)
    chunk_size = total_days // CONFIG.WALK_FORWARD_SPLITS
    
    folds = [
        (0, chunk_size * 2, chunk_size * 2, total_days) 
    ]
    
    policy_kwargs = dict(
        features_extractor_class=AdvancedTickerExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"\n=======================================================")
        print(f"--- FOLD {fold_idx + 1}: HỌC TỪ NGÀY {train_start}->{train_end}, THI ĐẤU NGÀY {test_start}->{test_end} ---")
        
        returns_train = returns_df.iloc[train_start:train_end]
        features_train = features_df.iloc[train_start:train_end]
        
        returns_test = returns_df.iloc[test_start:test_end]
        features_test = features_df.iloc[test_start:test_end]
        
        train_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(returns_train, features_train, weights_dim)])
        
        model = PPO("MlpPolicy", train_env, 
                    policy_kwargs=policy_kwargs, 
                    verbose=1, 
                    n_steps=1024,
                    learning_rate=0.0003,
                    ent_coef=CONFIG.ENT_COEF,
                    batch_size=64)
                    
        print(f"Đang huấn luyện mô hình ({CONFIG.TRAINING_TIMESTEPS} steps)...")
        model.learn(total_timesteps=CONFIG.TRAINING_TIMESTEPS)
        
        print(f"Đang chạy Backtest thực chiến (CÓ TÍNH PHÍ GIAO DỊCH {CONFIG.COST_RATE * 100}%)...")
        print("-------------------------------------------------------")
        test_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(returns_test, features_test, weights_dim, is_test=True)])
        obs = test_env.reset()
        
        portfolio_returns = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if 'net_return' in info[0]:
                portfolio_returns.append(info[0]['net_return'])
            
        test_len = len(returns_test)
        cum_ret = (np.prod(1 + np.array(portfolio_returns)) - 1) * 100
        print(f"=> [KẾT QUẢ] LỢI NHUẬN TÍCH LŨY FOLD {fold_idx + 1} (SAU {test_len} NGÀY THỰC CHIẾN): {cum_ret:.2f}%")

if __name__ == "__main__":
    returns_df, ticker_features_df, weights_dim = load_data()
    walk_forward_train_test(returns_df, ticker_features_df, weights_dim)
