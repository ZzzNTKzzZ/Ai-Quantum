# 📊 LIVE TRADING PIPELINE - QUY TRÌNH CHI TIẾT

## 🎯 Tổng Quan Pipeline

Quy trình Live Trading được chia thành **5 giai đoạn chính**, chạy tuần tự để sinh ra tín hiệu giao dịch cho ngày T+1 dựa trên dữ liệu ngày T:

```
Data Crawl → HMM Inference → LightGBM Prediction → DRL Portfolio → Trading Signals
```

---

## 📋 CHI TIẾT TỪNG GIAI ĐOẠN

### **GIAI ĐOẠN 1: TẢI DỮ LIỆU CHỨNG KHOÁN (Data Crawling)**

**File:** `crawl_live_46.py` & `auto_update_daily.py`

#### Chức năng chính:
- Tải dữ liệu giao dịch hàng ngày cho **46 mã chứng khoán VN**
- Cập nhật liên tục từ ngày cuối cùng trong dữ liệu cũ

#### Quy trình chi tiết:

1. **Định nghĩa 46 mã cổ phiếu:**
   - `BID, BMP, BVH, CII, CTD, CTG, DCM, DGW, DIG, DPM, DXG, EIB, FPT, GAS, GMD, HAG, HCM, HDC, HPG, HSG, HT1, KBC, KDC, KDH, MBB, MSN, MWG, NKG, NLG, NT2, PDR, PHR, PNJ, PVD, PVT, REE, SBT, SJS, SSI, STB, TCH, VCB, VHC, VIC, VNM, VSC`

2. **Lấy dữ liệu từ API:**
   - Kiểm tra file cũ → Lấy ngày cuối cùng
   - Tính `start_date = last_date + 1 ngày`
   - Thử các nguồn khác nhau (kbs, tcbs, vci) để đảm bảo lấy được dữ liệu

3. **Xử lý dữ liệu:**
   - Ghép dữ liệu mới vào dữ liệu cũ
   - Loại bỏ trùng lặp
   - Lưu vào `live/data/stocks/{symbol}.csv`

4. **Kiểm tra Macro Data (Vĩ mô):**
   - Nếu dữ liệu vĩ mô (CPI, Credit Growth, PMI) cũ > 3 ngày → Crawl lại
   - Ghép vào `data/raw/` và `data/test/`

#### Output:
```
live/data/stocks/{symbol}.csv  (e.g., VCB.csv, MBB.csv, ...)
data/raw/usdvnd.csv            (Macro data)
```

---

### **GIAI ĐOẠN 2: HMM INFERENCE (Học máy - Regime Detection)**

**File:** `hmm_live_inference.py`

#### Chức năng chính:
- Phát hiện **4 trạng thái thị trường** (Regime) khác nhau:
  1. **Macro Level** (2 states): Đóng cửng vs Mở rộng
  2. **Market Level** (3 states): Giảm (Bear) / Ngang (Sideways) / Tăng (Bull)
  3. **Sector Level** (3 states): Giảm / Ngang / Tăng cho mỗi ngành
  4. **Ticker Level** (3 states): Giảm / Ngang / Tăng cho từng mã

#### Quy trình chi tiết:

**BẢN CHẠY TRỊ CHI SỐ HMM (Hidden Markov Model):**

1. **BƯỚC 1: MACRO HMM (Hàng tháng)**
   ```
   Input Features: CPI MoM, Credit Growth MoM, PMI VN
   K (States): 2
   Train Mask: data <= 2019-12-31
   
   Output: Macro_Prob_0, Macro_Prob_1 (xác suất từng state)
   ```
   - Lấy dữ liệu hàng tháng
   - Huấn luyện GaussianHMM
   - Tự động gán nhãn: "Macro_Stagnant" vs "Macro_Expansion"
   - **Forward-fill** xác suất xuống dữ liệu hàng ngày

2. **BƯỚC 2: MARKET HMM (Hàng ngày)**
   ```
   Input Features: 
   - rolling_vol_5 (biến động 5 ngày)
   - volume_ratio (so với trung bình 20 ngày)
   - Macro_Prob_0 (từ Bước 1)
   
   K (States): 3
   Train Mask: data <= 2019-12-31
   Model: GMMHMM (Gaussian Mixture + HMM)
   
   Output: Market_Prob_0, Market_Prob_1, Market_Prob_2
   ```
   - Quy chuẩn hóa features bằng **NQT** (Normal Quantile Transform)
   - Gán nhãn tự động: "Bear", "Sideways", "Bull"
   - Tính xác suất cho từng state mỗi ngày

3. **BƯỚC 3: SECTOR HMM (Theo ngành, hàng ngày)**
   ```
   Input Features (cho mỗi ngành):
   - sector_log_ret_Z (return quy chuẩn)
   - sector_vol20_Z (biến động 20 ngày)
   - sector_vol5_Z (biến động 5 ngày)
   - sector_volume_ratio_Z (volume ratio)
   
   K (States): 3
   Model: GMMHMM cho từng ngành
   
   Output: prob_sector_Bear, prob_sector_Sideways, prob_sector_Bull
   ```
   - Tính state của từng ngành riêng
   - Loại bỏ ngành có dữ liệu < 100 ngày
   - Lưu model vào `output/models/sector_{industry_name}.pkl`

4. **BƯỚC 4: TICKER HMM (Mỗi mã, hàng ngày)**
   ```
   Input Features (Hybrid - lai):
   - Ticker-specific: log_return, rolling_vol_20d, volume_ratio
   - Macro Features: Macro_Prob_0, ...
   - Market Features: Market_Prob_0, ...
   - Sector Features: prob_sector_Bear, ...
   
   K (States): 3
   Model: GMMHMM cho từng mã
   
   Output: prob_ticker_0, prob_ticker_1, prob_ticker_2
   ```
   - Xây dựng hybrid features từ tất cả levels trên
   - HMM cấp ticker học cách kết hợp thông tin vĩ mô + ngành + cá nhân
   - Lưu model vào `output/models/ticker_{ticker}.pkl`

#### Output:
```
output/master_drl_ready_ticker.parquet

Columns:
- time, ticker
- OHLCV (open, high, low, close, volume)
- HMM Probabilities:
  - Macro_Prob_0, Macro_Prob_1
  - Market_Prob_0, Market_Prob_1, Market_Prob_2
  - prob_sector_Bear, prob_sector_Sideways, prob_sector_Bull
  - prob_ticker_0, prob_ticker_1, prob_ticker_2
- Technical Indicators:
  - rolling_vol_20d, volume_ratio, return_5d, return_20d, ...
  - rolling_vol_5, mom_1M, dist_MA50, ...
```

---

### **GIAI ĐOẠN 3: LIGHTGBM PREDICTION (Dự báo Tăng/Giảm)**

**File:** `live_trading.py`

#### Chức năng chính:
- Dự báo xác suất tăng giá **T+1** (ngày mai) cho mỗi mã
- Tính toán **tín hiệu giao dịch** (Khuyên Mua / Cảnh Báo)

#### Quy trình chi tiết:

1. **Chuẩn bị Target (Y):**
   ```python
   target_return_1d = close(T+1) / close(T) - 1  (Luôn được shift -1)
   target_bin = (target_return_1d > 0).astype(int)  # 1 = UP, 0 = DOWN
   ```
   - Chỉ dùng dữ liệu T-1 để dự báo T+1

2. **Định nghĩa Features (X):**
   ```
   Các features từ HMM + Technical:
   - prob_ticker_0, prob_ticker_1, prob_ticker_2
   - prob_sector_*
   - Market_Prob_*
   - Macro_Prob_*
   - rolling_vol_20d, return_5d, return_20d
   - rolling_vol_5, mom_1M, dist_MA50, volume_ratio
   ```

3. **Chia dữ liệu (Time Series Split):**
   ```
   Train: Tất cả ngày trước ngày mới nhất T (không gồm ngày T)
           Chỉ những hàng có target_return_1d != NaN
   
   Test: Chỉ ngày T (hiện tại)
   ```

4. **Huấn luyện LightGBM:**
   ```python
   model = LGBMClassifier(
       n_estimators=100,
       learning_rate=0.05,
       class_weight='balanced',  # Xử lý mất cân bằng dữ liệu
       n_jobs=-1
   )
   model.fit(X_train, y_train)
   ```

5. **Dự báo ngày T+1:**
   ```python
   probs = model.predict_proba(X_test)[:, 1]  # P(UP)
   signal = "Tăng (Khuyên Mua)" if probs > 0.5 else "Giảm (Cảnh Báo)"
   ```

#### Output:
```
live/output/live_trading_signals_YYYYMMDD.csv

Columns:
- time: Ngày giao dịch
- ticker: Mã chứng khoán
- close: Giá đóng cửa ngày T
- Xác Suất Tăng: 0.0 - 1.0
- Tín Hiệu: "Tăng (Khuyên Mua)" hoặc "Giảm (Cảnh Báo)"

Sắp xếp theo xác suất tăng (giảm dần)
TOP 15 MÃ TIỀM NĂNG NHẤT
```

---

### **GIAI ĐOẠN 4: DRL PORTFOLIO OPTIMIZATION (Tối ưu Tỷ trọng - Deep RL)**

**File:** `drl_live_trading.py`

#### Chức năng chính:
- Sử dụng **Deep Reinforcement Learning (PPO Agent)** để tìm **tỷ trọng tối ưu** cho từng mã
- AI học cách phân bổ 100% vốn giữa 46 mã để **tối đa hóa lợi nhuận** với **kiểm soát rủi ro**

#### Kiến trúc môi trường (Environment):

1. **Action Space (Đầu ra của AI):**
   ```
   Weights = [w1, w2, ..., w46]
   Constraints:
   - sum(weights) = 1.0 (Tất cả vốn)
   - 0 <= wi <= 1 (Mỗi mã từ 0% đến 100%)
   - Không được short selling (bán khống)
   ```

2. **State Space (Đầu vào cho AI):**
   ```
   Feature Stacks:
   - BẢNG 1 (AI Features): 
     * HMM Probabilities (prob_ticker_0, 1, 2)
     * Technical Normalized: rolling_vol_20d, return_20d, volume_ratio
     * Market Features: Market_Prob, Macro_Prob
     * Distance to MA20, Momentum 3D
   
   - BẢNG 2 (Rule-Based Strategies - AI không nhìn thấy):
     * Price Action: EMA20/50/200
     * Bollinger Bands: upper, middle, lower, width
     * RSI, MACD, Histogram
     * Candlestick Patterns: Hammer, Bull Engulfing
     * Support/Resistance Levels
     * Fibonacci Levels
     * Accumulation Zones
   
   - Portfolio State:
     * Current weights (tỷ trọng hiện tại)
     * Cash position
     * Daily returns
   ```

3. **Reward System (Hàm thưởng):**
   ```
   r_total = r_profit + r_loss + r_alpha + r_penalties
   
   1. Reward Lợi Nhuận:
      if portfolio_return > 0:
          r_profit = portfolio_return * REWARD_WIN_MULT * 100
      else:
          r_loss = portfolio_return * REWARD_LOSS_MULT * 100  (âm)
   
   2. Reward Alpha (Vượt Market):
      r_alpha = (portfolio_return - market_return) * REWARD_ALPHA_MULT * 100
   
   3. Penalty Chi phí Giao dịch:
      cost = turnover * COST_RATE
      r_cost = -cost * 100
   
   4. Penalty Turnover (Thay đổi tỷ trọng):
      turnover = sum(|new_weight - old_weight|) / 2
      r_turnover = -turnover * TURNOVER_PENALTY_RATE
   
   5. Penalty Drawdown (Mốt rơi):
      if portfolio_value < high_watermark:
          drawdown = (high_watermark - portfolio_value) / high_watermark
          r_drawdown = -drawdown * DRAWDOWN_PENALTY_RATE
   
   6. Bonus Thanh khoản (Khối lượng giao dịch):
      r_liquidity = volume_ratio * LIQUIDITY_BONUS_RATE
   
   7. Penalty Tiền mặt rơi:
      if cash < 0:  # Không có đủ tiền
          r_cash_crash = -1 * REWARD_CASH_CRASH_MULT
   
   8. Penalty Over-Diversification:
      entropy = -sum(weight * log(weight + 1e-8))
      if entropy > threshold:
          r_entropy = -PENALTY_OVER_DIVERSIFICATION
   ```

4. **Chi phí Giao dịch & T+3 Settlement:**
   ```
   Mỗi lần mua:
   - Phí = 0.01% (Chi phí CTCK)
   - Lock T+3: Cổ phiếu mua hôm nay không thể bán được
             trong 3 ngày tới (Quy định Chứng khoán VN)
   
   AI học: "Không nên trade liên tục liên tục, cần chọn lọc"
   ```

5. **Huấn luyện PPO (Proximal Policy Optimization):**
   ```
   Model Config:
   - Algorithm: PPO (Stable Baselines3)
   - Network Architecture: [64, 64] hidden layers
   - Features Dimension: 256 (Custom Feature Extractor)
   - Timesteps: 1,500,000 (1.5M steps)
   - Learning Rate: 0.00005
   - Batch Size: 90
   - Entropy Coefficient: 0.005 (Khuyến khích diversification)
   - N Steps: 1024
   
   Training Modes:
   - fast_split: 252 ngày train, 21 ngày test
   - rolling_window: Walk-forward validation
   ```

6. **Custom Feature Extractor:**
   - Xử lý 2 bảng feature khác nhau
   - NQT normalization (Normal Quantile Transform)
   - Concatenate AI Features + Strategies Features

#### Output:
```
output/v7_3/AI_Brain_v7_3.zip  (Trained PPO Model)

Live Inference cho ngày T+1:
- Input: State của market, portfolio ở ngày T
- Output: weights = [w_VCB, w_MBB, ..., w_VSC]
  * Mỗi wi biểu thị % vốn nên đầu tư vào mã i
  * Tự động chuẩn hóa: sum(w) = 1.0
```

---

### **GIAI ĐOẠN 5: TRADING SIGNALS & VERIFICATION**

**File:** `live_trading.py` (Output) & `verify_data.py` (Backtesting)

#### Output Trading Signals:

1. **File CSV Tín hiệu:**
   ```
   live/output/live_trading_signals_YYYYMMDD.csv
   
   Columns:
   - time: Ngày giao dịch (ngày T)
   - ticker: Mã cổ phiếu
   - close: Giá đóng cửa ngày T
   - Xác Suất Tăng: Xác suất tăng T+1 (0.0-1.0)
   - Tín Hiệu: "Tăng (Khuyên Mua)" / "Giảm (Cảnh Báo)"
   ```

2. **Top 15 Mã Tiềm Năng:**
   - Sắp xếp theo xác suất tăng (giảm dần)
   - In ra console để theo dõi

#### Verification (Backtesting):

**File:** `verify_data.py`

1. **Lấy giá thực tế T+1, T+2:**
   - API VNDirect hoặc Yahoo Finance
   - Lấy 3 ngày giao dịch tiếp theo

2. **Tính lợi nhuận:**
   ```
   pct_change_T2 = (price_T2 - price_T0) / price_T0 * 100
   ```

3. **Đánh giá tín hiệu:**
   ```
   - Tín hiệu MUA → Kiểm tra: price_T2 > price_T0? (Lợi nhuận dương)
   - Tín hiệu BÁN → Kiểm tra: price_T2 < price_T0? (Tránh được lỗ)
   ```

4. **Báo cáo Win Rate:**
   ```
   Win Rate = (Số dự báo ĐÚNG / Tổng dự báo) * 100%
   ```

---

## 🔄 QUY TRÌNH TỔNG QUÁT (auto_update_daily.py)

```
┌─────────────────────────────────────────┐
│ 1. CHECK & UPDATE MACRO DATA            │
│    (CPI, Credit Growth, PMI)            │
│    Nếu > 3 ngày → Crawl lại             │
└─────────────────────┬───────────────────┘
                      ↓
┌─────────────────────────────────────────┐
│ 2. CRAWL 46 MÃ CHỨNG KHOÁN (crawl_live_46.py)     │
│    Lấy dữ liệu từ ngày cuối cùng       │
│    → Lưu vào live/data/stocks/         │
└─────────────────────┬───────────────────┘
                      ↓
┌─────────────────────────────────────────┐
│ 3. DATA PROCESSING (run_full_regeneration.py)   │
│    Tạo m1_vn46.csv, căn chỉnh dữ liệu │
└─────────────────────┬───────────────────┘
                      ↓
┌─────────────────────────────────────────┐
│ 4. HMM LIVE INFERENCE (hmm_live_inference.py)   │
│    Load models, sinh xác suất state   │
│    → master_drl_ready_ticker.parquet   │
└─────────────────────┬───────────────────┘
                      ↓
┌─────────────────────────────────────────┐
│ 5. LIGHTGBM PREDICTION (live_trading.py)      │
│    Dự báo Tăng/Giảm T+1                │
│    → live_trading_signals_YYYYMMDD.csv │
└─────────────────────┬───────────────────┘
                      ↓
┌─────────────────────────────────────────┐
│ 6. DRL PORTFOLIO (drl_live_trading.py)         │
│    Tối ưu tỷ trọng bằng PPO            │
│    → Weights cho mỗi mã                │
└─────────────────────┬───────────────────┘
                      ↓
         🎯 READY FOR LIVE TRADING 🎯
```

---

## 📊 DATA FLOW DIAGRAM

```
Live Data (Ngày T)
    ↓
├─→ OHLCV, Volume
├─→ Macro: CPI, Credit Growth, PMI, FNB Ratio
├─→ Market Proxy: VNINDEX (Log Return, Vol, Volume)
└─→ Industry Info
    ↓
HMM Pipeline (4 Levels)
    ├─→ Macro HMM: 2 States
    ├─→ Market HMM: 3 States
    ├─→ Sector HMM: 3 States (per industry)
    └─→ Ticker HMM: 3 States (per ticker)
    ↓
    master_drl_ready_ticker.parquet
    (Tất cả HMM Probs + Technical Indicators)
    ↓
LightGBM Classifier
    ├─→ Train: Historical data (T < Last)
    ├─→ Predict: Today (T)
    └─→ Output: P(UP) for T+1
    ↓
live_trading_signals_YYYYMMDD.csv
    ├─→ Ticker, Close Price, Signal, Probability
    └─→ Sorted by probability (descending)
    ↓
DRL Portfolio (PPO)
    ├─→ Optimize Weights: [w1, w2, ..., w46]
    ├─→ Maximize: Returns - Cost - Risk
    └─→ Output: Portfolio Allocation
    ↓
🚀 LIVE TRADING EXECUTION FOR T+1
```

---

## ⚙️ KEY CONFIGURATIONS & PARAMETERS

### HMM Configurations:
```
HMM_TRAIN_END = 2019-12-31        (Training cutoff)
K_MACRO_BEST = 2                  (Macro states)
K_MARKET_BEST = 3                 (Market states)
K_SECTOR_BEST = 3                 (Sector states)
K_TICKER_BEST = 3                 (Ticker states)
```

### LightGBM Parameters:
```
n_estimators = 100                (Trees)
learning_rate = 0.05              (Learning speed)
class_weight = 'balanced'          (Handle imbalance)
```

### DRL/PPO Parameters:
```
COST_RATE = 0.0001                (Phí giao dịch 0.01%)
T_PLUS_SETTLEMENT = 3             (T+2 settlement lock)
TRAINING_TIMESTEPS = 1,500,000    (1.5M steps)
ENT_COEF = 0.005                  (Entropy - diversification bonus)
LEARNING_RATE = 0.00005           (PPO learning rate)
BATCH_SIZE = 90                   (Mini-batch size)
```

### Technical Indicators:
```
EMA: 20, 50, 200 ngày
RSI: 20-period Wilder's Smoothing
MACD: 12/26/9
Bollinger Bands: 20-period, 2 std
Fibonacci: 38.2%, 50%, 61.8%, extensions
Support/Resistance: 20-period highs/lows
```

---

## 🎯 REAL-TIME WORKFLOW (Mỗi ngày)

```
Morning (Trước giờ mở cửa thị trường):
  1. Chạy auto_update_daily.py
  2. System tự động:
     - Crawl dữ liệu 46 mã
     - Chạy HMM inference (load models)
     - Dự báo bằng LightGBM
     - Sinh tín hiệu giao dịch
     - Tối ưu tỷ trọng bằng DRL
  3. Output: Bảng tín hiệu & tỷ trọng

Morning (Giờ mở cửa):
  - Đọc tín hiệu từ live_trading_signals_YYYYMMDD.csv
  - Theo dõi TOP 15 mã tiềm năng
  - Thực hiện giao dịch theo đề xuất

Afternoon (Sau giờ đóng cửa):
  - (Optional) Chạy verify_data.py để kiểm tra độ chính xác T+2
```

---

## 📈 OUTPUT FOLDER STRUCTURE

```
live/
├── output/
│   ├── live_trading_signals_20260623.csv    (Daily signals)
│   ├── master_drl_ready_ticker.parquet      (All features & HMM)
│   ├── models/
│   │   ├── macro_model.pkl
│   │   ├── daily_market_model.pkl
│   │   ├── sector_{industry}.pkl            (×7-8 files)
│   │   ├── ticker_{ticker}.pkl              (×46 files)
│   │   └── ...
│   └── v7_3/
│       └── AI_Brain_v7_3.zip                (Trained PPO model)
├── data/
│   ├── stocks/
│   │   ├── VCB.csv
│   │   ├── MBB.csv
│   │   └── ... (46 files)
│   └── ...
└── ...
```

---

## ✨ TÍNH NĂNG NỔNG BẬT

| Feature | Chi tiết |
|---------|---------|
| **Multi-Level HMM** | Macro → Market → Sector → Ticker (4 cấp độ) |
| **Hybrid AI** | LightGBM (prediction) + PPO (portfolio optimization) |
| **Live Mode** | Load models sẵn, không cần train lại |
| **Cost Modeling** | Chi phí giao dịch 0.01%, T+2 settlement lock |
| **Risk Control** | Drawdown penalty, turnover penalty, diversification |
| **Technical Analysis** | 30+ indicators (EMA, RSI, MACD, Bollinger, Fibonacci, ...) |
| **Fast Inference** | Chỉ cần ~1-2 phút cho toàn bộ 46 mã |
| **Modular Design** | Từng stage độc lập, dễ debug & update |

---

## 🔍 ĐIỂM MẠI KHÁC BIỆT

1. **HMM 4-Level**: Không phải tất cả hệ thống đều làm. Kết hợp vĩ mô → thị trường → ngành → cá nhân
2. **PPO+Technical**: AI học cách sử dụng technical indicators mà con người không thể nhìn thấy
3. **Settlement Lock**: Mô phỏng quy định thực tế VN (T+2 không bán được)
4. **Live Fast**: Load models pre-trained, không retrain hàng ngày
5. **Interpretability**: Output CSV rõ ràng: Tín hiệu + Xác suất + Top 15 mã

