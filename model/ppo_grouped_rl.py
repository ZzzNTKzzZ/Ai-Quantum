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

# ==========================================
# [CẤU HÌNH CHIẾN LƯỢC ĐẦU TƯ - AI TRADING]
# ==========================================
# Khối cấu hình trung tâm (Control Panel) giúp bạn tinh chỉnh "tính cách" của AI 
# mà không cần can thiệp sâu vào bên trong mã nguồn.
class CONFIG:
    # 1. PHÍ GIAO DỊCH (Transaction Cost): 
    # Tỷ lệ phần trăm CTCK thu cho mỗi lần bạn mua hoặc bán cổ phiếu. (VD: 0.2% = 0.002)
    COST_RATE = 0.002
    
    # 2. CHU KỲ REBALANCE (Rule 1: Holding Period):
    # Thời gian (tính bằng ngày) AI bị ép phải "ôm chặt" danh mục trước khi được phép mua bán lại.
    # Mục đích: Giảm thiểu Over-trading (giao dịch quá nhiều) để không bị bào mòn tài khoản bởi phí giao dịch.
    HOLDING_PERIOD = 30
    
    # 3. NGƯỠNG TỰ TIN VÀO CỔ PHIẾU (Rule 5: Cash Conviction Threshold):
    # Mức độ tự tin tối thiểu (từ 0.0 đến 1.0) AI cần có để quyết định giải ngân.
    # Nếu mức tự tin cao nhất của AI thấp hơn ngưỡng này (thị trường đang quá rủi ro/khó đoán), 
    # nó sẽ tự động chừa lại phần vốn đó dưới dạng Tiền mặt để phòng thủ.
    CASH_CONVICTION_THRESHOLD = 0.2
    
    # 4. HÌNH PHẠT RỦI RO (Rule 3: Sharpe Penalty):
    # Mức điểm âm sẽ phạt AI nếu danh mục của nó chọn bị Lỗ sau 1 chu kỳ Holding Period.
    # Con số càng âm (VD: -2.0 hoặc -5.0) sẽ rèn luyện cho AI tính cẩn thận, chỉ mua mã an toàn.
    PENALTY_FOR_LOSS = -2.0
    
    # 5. CẮT LỖ ĐỘNG (Rule 6: Stop-Loss):
    # Mức rớt giá tối đa cho phép. Nếu đang trong thời gian "ôm chặt" mà danh mục sập chạm mốc này,
    # lệnh ngắt mạch sẽ kích hoạt: Bán tháo toàn bộ danh mục ngay lập tức để rút về Tiền mặt bảo toàn tính mạng.
    STOP_LOSS_THRESHOLD = -0.07
    
    # 6. CHỐT LỜI SỚM (Rule 6: Take-Profit):
    # Mức lãi kỳ vọng. Nếu danh mục sinh lời nhanh và đạt mốc này trước khi hết hạn ôm,
    # kích hoạt ngắt mạch: Chốt lời toàn bộ, bỏ túi tiền mặt và đứng ngoài thị trường chờ chu kỳ mới.
    TAKE_PROFIT_THRESHOLD = 0.15
    
    # 7. CHIA KHÚC DỮ LIỆU ĐỂ HỌC VÀ THI (Rule 4: Walk-Forward Folds):
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

# ==========================================
# LUẬT 2: MARKET CONTEXT (TỔNG HỢP VĨ MÔ & VI MÔ)
# ==========================================
# Hàm này nạp dữ liệu từ file Parquet (đã chạy HMM) và chế biến thêm các chỉ báo để tạo thành 
# một bộ "Hồ sơ năng lực" (Features) 11 chiều cho mỗi cổ phiếu mỗi ngày.
def load_data():
    # Nạp dữ liệu thô
    raw_df = pd.read_parquet(r"C:\Users\ADMIN\Desktop\Kaggle\output\hmm_v3_op1_extended\master_drl_ready_full.parquet")
    raw_df['time'] = pd.to_datetime(raw_df['time'])
    
    # Lấy Log Return (Tỷ suất sinh lời logarit)
    returns_df = raw_df.pivot(index='time', columns='ticker', values='log_return').fillna(0)
    weights_dim = returns_df.shape[1] # Số lượng mã cổ phiếu
    
    # -------------------------------------
    # CÁC CHỈ BÁO VĨ MÔ (CỦA TOÀN THỊ TRƯỜNG VN-INDEX)
    # -------------------------------------
    # Lấy trung bình toàn thị trường để AI biết đang Uptrend hay Downtrend tổng thể
    market_return_5d = returns_df.rolling(5).sum().mean(axis=1).fillna(0) # Tỷ suất sinh lời VN-Index 5 ngày
    market_return_20d = returns_df.rolling(20).sum().mean(axis=1).fillna(0) # Tỷ suất sinh lời VN-Index 20 ngày
    market_vol_20d = returns_df.rolling(20).std().mean(axis=1).fillna(0) # Độ biến động (Rủi ro) VN-Index 20 ngày
    
    # -------------------------------------
    # CÁC CHỈ BÁO KỸ THUẬT (MA20 VÀ PRICE)
    # -------------------------------------
    # Khôi phục lại Giá (Normalized Price) từ Log Return
    price_df = np.exp(returns_df.cumsum())
    
    # Tính đường Trung bình động 20 ngày (MA20)
    ma20_df = price_df.rolling(20).mean()
    
    # Tuyệt chiêu Quantitative: Khỏang cách từ Giá đến đường MA20 (Distance to MA20)
    # Tỷ lệ dương: Đang Uptrend ngắn hạn (Nằm trên MA20). Tỷ lệ âm: Đang Downtrend (Nằm dưới MA20).
    dist_ma20_df = ((price_df - ma20_df) / ma20_df).fillna(0)

    # -------------------------------------
    # CÁC CHỈ BÁO VI MÔ (CỦA TỪNG CỔ PHIẾU TỪ HMM)
    # -------------------------------------
    prob0_df = raw_df.pivot(index='time', columns='ticker', values='prob_ticker_0').fillna(0) # Xác suất Bear Market
    prob1_df = raw_df.pivot(index='time', columns='ticker', values='prob_ticker_1').fillna(0) # Xác suất Sideway
    prob2_df = raw_df.pivot(index='time', columns='ticker', values='prob_ticker_2').fillna(0) # Xác suất Bull Market
    vol_20d_df = raw_df.pivot(index='time', columns='ticker', values='rolling_vol_20d').fillna(0)
    ret_5d_df = raw_df.pivot(index='time', columns='ticker', values='return_5d').fillna(0)
    ret_20d_df = raw_df.pivot(index='time', columns='ticker', values='return_20d').fillna(0)
    vol_ratio_df = raw_df.pivot(index='time', columns='ticker', values='volume_ratio').fillna(0)

    # Nhân bản các chỉ số Vĩ mô để ghép chung vào bảng Vi mô (Vì Vĩ mô ngày nào cũng như nhau cho mọi mã)
    for df_macro, name in zip([market_return_5d, market_return_20d, market_vol_20d], ['MKT_RET5', 'MKT_RET20', 'MKT_VOL']):
        macro_expanded = pd.DataFrame(np.tile(df_macro.values.reshape(-1, 1), (1, weights_dim)), 
                                      index=returns_df.index, columns=returns_df.columns)
        if name == 'MKT_RET5': mkt_ret5_df = macro_expanded
        elif name == 'MKT_RET20': mkt_ret20_df = macro_expanded
        elif name == 'MKT_VOL': mkt_vol_df = macro_expanded

    # Ghép 11 tính năng lại thành 1 siêu bảng duy nhất để "Mớm" cho AI
    ticker_features_df = pd.concat([prob0_df, prob1_df, prob2_df, vol_20d_df, ret_5d_df, ret_20d_df, vol_ratio_df, 
                                    mkt_ret5_df, mkt_ret20_df, mkt_vol_df, dist_ma20_df], axis=1)
    
    # Đồng bộ ngày tháng giữa 2 bảng để không bị lệch dữ liệu
    returns_df, ticker_features_df = returns_df.align(ticker_features_df, axis=0, join='inner')
    return returns_df, ticker_features_df, weights_dim

# ==========================================
# LUẬT 1, 3, 5, 6: MÔI TRƯỜNG ĐẦU TƯ (GYM ENVIRONMENT)
# ==========================================
# Lớp này mô phỏng lại Sàn chứng khoán. Nơi AI sẽ thử nghiệm các lệnh Mua/Bán và nhận Phạt/Thưởng (Reward).
class AdvancedPortfolioEnv(gym.Env):
    def __init__(self, returns_df, features_df, weights_dim, step_size=CONFIG.HOLDING_PERIOD, cost_rate=CONFIG.COST_RATE, is_test=False):
        super().__init__()
        self.returns_arr = np.exp(returns_df.values) - 1.0 # Chuyển log return về Return bình thường
        self.features_arr = features_df.values
        self.weights_dim = weights_dim
        self.n_steps = len(self.returns_arr)
        
        self.step_size = step_size
        self.cost_rate = cost_rate
        self.is_test = is_test # Cờ kiểm tra: Đang Train (Học) hay đang Test (Thi đấu thực tế)
        
        # Không gian Hành động (Action Space): Một mảng [46] giá trị từ 0 đến 1, đại diện cho Tỷ trọng mua từng mã
        self.action_space = spaces.Box(low=0, high=1, shape=(self.weights_dim,), dtype=np.float32)
        
        # Không gian Quan sát (Observation Space): Một mảng [46, 11] chứa 11 thông số (Features) của 46 mã
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.weights_dim, 11), dtype=np.float32)
        
        self.current_step = 0
        self.weights = np.zeros(self.weights_dim) # Khởi tạo danh mục 100% Tiền mặt
        
    def reset(self, seed=None, options=None):
        # Đưa Sàn chứng khoán về ngày 0, reset mọi thứ về Tiền mặt
        self.current_step = 0
        self.weights = np.zeros(self.weights_dim)
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Trích xuất 11 thông số của ngày hiện tại để AI "Mở mắt" nhìn thị trường
        idx = min(self.current_step, self.n_steps - 1)
        obs_1d = self.features_arr[idx]
        obs_2d = obs_1d.reshape(11, self.weights_dim).T
        return obs_2d.astype(np.float32)
        
    def step(self, action):
        # 1. Cơ chế Tiền mặt dự phòng (Rule 5: Cash Buffer)
        conviction = np.max(action)
        if conviction < CONFIG.CASH_CONVICTION_THRESHOLD:
            investment_ratio = conviction / CONFIG.CASH_CONVICTION_THRESHOLD
        else:
            investment_ratio = 1.0 # Tự tin cao -> All-in 100% vốn
            
        # Chuẩn hóa tỷ trọng cho tổng bằng 1 (100%), sau đó nhân với tỷ lệ giải ngân
        action = action / (np.sum(action) + 1e-9)
        action = action * investment_ratio 
        
        # Báo cáo Log trong quá trình Thực chiến (Test)
        if self.is_test:
            cash_pct = (1.0 - investment_ratio) * 100
            print(f"| Ngày {self.current_step:04d} | ĐẢO HÀNG: Giải ngân {investment_ratio*100:.1f}% vốn, Tiền mặt {cash_pct:.1f}%")
        
        # 2. Tính Phí Giao Dịch
        # Chỉ lấy phí trên sự "Thay đổi" của danh mục. Nếu giữ nguyên mã cũ thì không tốn phí.
        cost = self.cost_rate * np.sum(np.abs(action - self.weights))
        self.weights = action
        
        # 3. Chu kỳ Ôm hàng (Rule 1: Holding Period)
        end_step = min(self.current_step + self.step_size, self.n_steps)
        days_held = end_step - self.current_step
        
        if days_held == 0:
            return self._get_obs(), 0, True, False, {}
            
        daily_portfolio_returns = []
        cum_ret_since_rebalance = 0.0
        is_cash_mode = False
        
        # Bước lùi thời gian: AI nhắm mắt ngồi xem lợi nhuận chạy mỗi ngày trong suốt 30 ngày Holding
        for t in range(self.current_step, end_step):
            # Nếu đã cắt lỗ/chốt lời và rút về Tiền mặt, lợi nhuận các ngày sau = 0%
            if is_cash_mode:
                daily_portfolio_returns.append(0.0)
                continue
                
            daily_ret = np.sum(self.weights * self.returns_arr[t])
            
            # Trừ thẳng phí giao dịch vào ngày đầu tiên mua/bán
            if t == self.current_step:
                daily_ret -= cost
                
            daily_portfolio_returns.append(daily_ret)
            
            # Tính lợi nhuận tích lũy tạm thời từ đầu chu kỳ
            cum_ret_since_rebalance = (1 + cum_ret_since_rebalance) * (1 + daily_ret) - 1
            
            # 4. Cơ chế Ngắt mạch (Rule 6: Dynamic Stop-Loss & Take-Profit)
            if cum_ret_since_rebalance <= CONFIG.STOP_LOSS_THRESHOLD or cum_ret_since_rebalance >= CONFIG.TAKE_PROFIT_THRESHOLD:
                if self.is_test:
                    if cum_ret_since_rebalance <= CONFIG.STOP_LOSS_THRESHOLD:
                        print(f"   -> [CẮT LỖ] Ngày {t:04d}: Lỗ {cum_ret_since_rebalance*100:.1f}% -> Bán tháo toàn bộ!")
                    else:
                        print(f"   -> [CHỐT LỜI] Ngày {t:04d}: Lãi {cum_ret_since_rebalance*100:.1f}% -> Cầm tiền mặt chờ tháng sau!")
                        
                # Phải mất phí giao dịch (Market Order) để bán tháo mọi thứ ra Tiền Mặt
                exit_cost = self.cost_rate * np.sum(self.weights)
                daily_portfolio_returns[-1] -= exit_cost # Trừ ngay vào lợi nhuận ngày hôm đó
                self.weights = np.zeros(self.weights_dim) # Danh mục trắng trơn: Cầm 100% Tiền Mặt
                is_cash_mode = True # Kích hoạt chế độ nghỉ phép
        
        # Tính toán kết quả Báo cáo cuối chu kỳ
        daily_portfolio_returns = np.array(daily_portfolio_returns)
        cum_return = np.prod(1 + daily_portfolio_returns) - 1
        
        # Tính chỉ số Sharpe (Đo lường Lợi nhuận trên Rủi ro)
        mean_ret = np.mean(daily_portfolio_returns)
        std_ret = np.std(daily_portfolio_returns) + 1e-9
        sharpe = (mean_ret / std_ret) * np.sqrt(252) # Annualized Sharpe
        
        # 5. Hình Phạt Kỷ Luật (Rule 3: Sharpe Penalty)
        # Nếu lỗ, AI sẽ bị trừ thêm điểm kỷ luật để nó chừa thói quen mua mã rủi ro
        if cum_return < 0:
            reward = sharpe + CONFIG.PENALTY_FOR_LOSS
        else:
            reward = sharpe
            
        # Chốt sổ và Nhảy sang ngày Rebalance tiếp theo
        self.current_step = end_step
        done = self.current_step >= self.n_steps - 1 # Báo hiệu hết dữ liệu
        
        return self._get_obs(), float(reward), done, False, {'net_return': cum_return}

# ==========================================
# MẠNG NƠ-RON TRÍ TUỆ NHÂN TẠO (NEURAL NETWORK STRUCTURE)
# ==========================================
# Đây là "Bộ Não" thực sự của AI. Mạng Nơ-ron này chịu trách nhiệm Đọc dữ liệu (Extractor).
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

# ==========================================
# LUẬT 4: WALK-FORWARD VALIDATION (CHỐNG HỌC VẸT)
# ==========================================
# Đạo diễn toàn bộ quá trình: Chia dữ liệu -> Gọi AI vào Học (Train) -> Bắt AI đi Thi (Test) -> Báo cáo Lợi nhuận
def walk_forward_train_test(returns_df, features_df, weights_dim):
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
        test_env = DummyVecEnv([lambda: AdvancedPortfolioEnv(returns_test, features_test, weights_dim, is_test=True)])
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
    returns_df, ticker_features_df, weights_dim = load_data()
    walk_forward_train_test(returns_df, ticker_features_df, weights_dim)
