# HYBRID DUAL-FREQUENCY TICKER-SPECIFIC HMM
## Quy Trinh Live Trading Chi Tiet Toan Bo

---

## TONG QUAN CHUNG (Overview)

### Tieu De Chinh:
```
Hybrid Dual-Frequency Ticker-Specific HMM

Quy trinh ket hop danh gia Vi mao (Monthly) va Thi truong chung (Daily)
de dinh vi pha bien dong, sau do ep cau truc thi truong chung len tung
ma co phieu (Ticker-Specific) va dung Meta-Classifier du doan loi suat ngay toi.
```

### Mo Ta Hang Nhat:
- **Hybrid**: Ket hop 2 tan suat - Vĩ mô (Monthly) và Thị trường (Daily)
- **Dual-Frequency**: Dùng dữ liệu ở nhiều khung thời gian
- **Ticker-Specific**: Mỗi mã cổ phiếu có HMM riêng
- **HMM**: Hidden Markov Model - mô hình xác suất cho biến động thị trường

---

## CHI TIET 30 PHAN CUA NOTEBOOK

### PHAN 1: KHOI TAO HE THONG & LOAD THU VIEN

**Tien Hanh (Steps):**
1. Dinh nghia ham `get_hmm_filtered_inference(model, Z)` de thuc hien Online filtering
   - Tinh toan xac suat trang thai (state probabilities) cho HMM
   - Duyet lan luot tung diem du lieu tu t=1 den N
   - Tinh toan Z_slice = Z[:t] de lay tat ca du lieu toi thoi diem hien tai
   - Tra ve: filtered_regimes (trang thai toi uu), filtered_probs (xac suat)

2. Import Thu Vien:
   - numpy, pandas: Xu ly du lieu
   - statsmodels (adfuller, kpss): Kiem dinh tinh dung (stationarity)
   - sklearn (variance_inflation_factor, mutual_info_regression): Chon bien
   - scipy (norm, skew, kurtosis): Thong ke
   - hmmlearn.hmm (GMMHMM, GaussianHMM): Mo hinh HMM
   - lightgbm: Machine learning
   - shap: Feature importance
   - joblib: Song song hoa

3. Thiet lap:
   - RANDOM_STATE = 42
   - OUTPUT_DIR = ../output/hmm_v3_op1_extended/

**Dau Ra (Output):**
- Toan bo thu vien san sang
- Duong dan output duoc tao
- Ham `get_hmm_filtered_inference()` san sang dung

---

### PHAN 2: TAI DU LIEU & TAO CHI BA THI TRUONG DAI DIEN

**Tien Hanh:**
1. Tải 2 file CSV:
   - `hmm_data.csv`: Dữ liệu market hàng ngày
   - `m1_vn46.csv`: Dữ liệu 46 mã cổ phiếu VN46 (minute/hourly level)

2. Chuẩn hóa thời gian: pd.to_datetime() và normalize() thành ngày

3. Tạo Market Proxy từ rổ VN46:
   - `market_log_ret` = mean của log_return của 46 mã
   - `market_close` = mean của close price của 46 mã
   - Merge lại với df_market

4. Tính toán chỉ số bổ sung:
   - `vnindex_vol20` = rolling volatility 20 ngày (annualized)
   - Thêm foreign net buy/sell ratio (nếu có)
   - Thêm USD/VND exchange rate changes (nếu có)

5. Loại bỏ NaN values

**Dau Ra:**
- `df_market`: DataFrame (2382 rows, 13 columns) với Market proxy
- Chuẩn bị sẵn cho bước kỳ tiếp

---

### PHAN 3: BO LOC KIEM DINH KY THUAT (Stationarity & Kurtosis Filter)

**Tien Hanh:**
1. **Kiểm định tính dừng (I(0) test):**
   - ADF test (Augmented Dickey-Fuller): p_adf < 0.05 → reject H0 (chuỗi dừng)
   - KPSS test: p_kpss >= 0.05 → accept H0 (chuỗi dừng)
   - Chuỗi dừng: cả ADF AND KPSS đều pass

2. **Kiểm định hệ số nhọn (Excess Kurtosis):**
   - |Kurtosis| < 10: OK (không quá béo)
   - Nếu > 10: có outlier quá nhiều

3. **Kiểm tra được giữ lại:**
   - `rolling_vol_5`: ✓ dừng
   - `volume_ratio`: ✓ dừng
   - `credit_growth_mom`: ✓ dừng
   - `cpi_mom`: ✓ dừng
   - `fnb_ratio`: ✓ dừng

4. **Loại bỏ:**
   - `amihud_diff_normalized`: ✗ non-stationary + kurtosis 13.7
   - `ret_disp`: ✗ non-stationary
   - `pmi_vn`: ✗ non-stationary
   - `fx_log_ret`: ✗ kurtosis 10.4 (quá béo)

**Dau Ra:**
- `selected_raw_features`: 5 biến đã lọc
- Chuẩn bị cho bước NQT transform

---

### PHAN 4: CHUAN HOA DU LIEU NQT (Normal Quantile Transform) & RANK

**Tien Hanh:**
1. **Tạo hàm `make_Z()`:**
   - Input: df, features, window=252 (1 năm giao dịch)
   - Tính rolling rank trong cửa sổ 252 ngày
   - Công thức: pct = (rank - 0.5) / count
   - NQT: Z = norm.ppf(pct) ~ N(0,1)
   - Clip: Z ∈ [-3, 3] để tránh extreme

2. **Tách Training & Out-of-Sample:**
   - Train: time <= 2019-12-31
   - OOS: time > 2019-12-31

3. **Output:**
   - `fd_market`: DataFrame gốc với features
   - `Z_tr_market`: Z-normalized training data
   - `Z_all_market`: Z-normalized toàn bộ
   - `df_market_Z`: DataFrame với cột `*_Z` suffix

**Dau Ra:**
- Dữ liệu chuẩn hóa sẵn sàng cho HMM
- Tranh lỗi look-ahead bias

---

### PHAN 5: MUTUAL INFORMATION (MI) & CHON BIEN THAM LAM + VIF FILTER

**Tien Hanh:**
1. **Tạo Target Proxy:**
   - Y_proxy = 3 lớp:
     - 0: Bull (ret > 0 AND vol < median)
     - 1: Bear (ret < 0 AND vol > median)
     - 2: Sideways (còn lại)

2. **Tính SHAP Importance:**
   - Train LightGBM classifier với 3 lớp
   - Dùng TreeExplainer để tính SHAP values
   - Lấy mean absolute SHAP

3. **Tính Mutual Information:**
   - Target MI = |vnindex_log_ret| (độ lớn biến động)
   - Dùng mutual_info_regression() từ sklearn
   - MI cao = biến có nhiều thông tin về biến động

4. **Lọc VIF Tham Lam:**
   - Sắp xếp theo total_score = SHAP * MI giảm dần
   - Chọn feature từng cái một
   - Kiểm tra VIF < 5.0 (tranh multicollinearity)
   - Dừng khi VIF vượt ngưỡng

5. **Tách Macro vs Market Features:**
   - Macro pool: cpi_mom_Z, credit_growth_mom_Z, fnb_ratio_Z
   - Market pool: rolling_vol_5_Z, volume_ratio_Z

**Dau Ra:**
- `macro_features`: 3 biến macro
- `market_features`: 2 biến market
- `final_features`: 5 biến tổng cộng

---

### PHAN 6: GRID SEARCH HMM TOI UU (Macro & Market Layer)

**Tien Hanh:**

**6a. Macro HMM (MONTHLY TIMEFRAME):**
1. Aggregate df_market thành tháng (year_month)
2. Grid search K ∈ [2, 3]:
   - Model: GMMHMM(n_components=K, n_mix=2, covariance_type='diag')
   - Chạy 5 seeds khác nhau
   - Tính BIC, LL OOS, min_duration, state share distribution

3. Ràng buộc chọn model:
   - min_duration >= 2.0 (phải ở trạng thái ít 2 phiên)
   - 0.05 <= min_share <= max_share <= 0.85 (cân bằng trạng thái)

4. Xếp hạng composite:
   - Rank_bic (BIC càng thấp càng tốt)
   - Rank_oos (LL OOS càng cao càng tốt)
   - Composite = 0.5 * Rank_bic + 0.5 * Rank_oos
   - **Quyết định: K_macro = 2 (tốt nhất)**

5. Chạy filtered inference trên Z toàn bộ (train + OOS)

6. **Xử lý Publication Lag:**
   - Shift Macro probabilities thêm 1 tháng
   - Merge lại với daily data
   - Forward fill NaN ở đầu

**6b. Market HMM (DAILY TIMEFRAME WITH MACRO AWARENESS):**
1. Input features = Market features + Macro_Prob columns
2. Grid search K ∈ [2, 3, 4]:
   - Tương tự Macro HMM
   - Nhưng composite = 0.3*Rank_bic + 0.5*Rank_oos + 0.2*Rank_min_dur

3. **Quyết định: K_market = 2**

4. Chạy filtered inference & lưu Market_Prob columns

**Dau Ra:**
- `best_macro_hmm`: Model HMM macro tốt nhất
- `best_market_hmm`: Model HMM market tốt nhất
- `df_market`: Thêm cột Macro_Prob_*, Market_Prob_*

---

### PHAN 7: TU DONG ANH XA & GAN NHAN TRANG THAI (K-agnostic Labeling)

**Tien Hanh:**
1. **Auto-label Market HMM (K=2):**
   - Tính mean return và volatility cho mỗi state
   - Sharpe ratio = ret / vol
   - State với return thấp nhất → **"Bear"**
   - State còn lại → **"Bull"**

2. **Auto-label Macro HMM (K=2):**
   - Tương tự: Low PMI → Stagnant, High PMI → Expansion

3. **K >= 4 (nếu có):**
   - Dùng linear_sum_assignment() để matching states với labels
   - Chuẩn hóa ret & vol thành z-scores
   - Gán: Crisis, Tranquil, CalmBull, Euphoria

4. **Output:**
   - `STATE_TO_LABEL_MARKET`: Mapping state → label
   - `market_regime_label`: Cột label cho mỗi row
   - `prob_market_k`: Cột xác suất mỗi state

**Dau Ra:**
- Market labeled regimes sẵn sàng
- Macro labeled regimes sẵn sàng
- Dữ liệu chuẩn bị cho Sector HMM

---

### PHAN 8: CHUAN BI DU LIEU NGANH & HUAN LUYEN SECTOR HMM

**Tien Hanh:**
1. **Load industries mapping:**
   - `industries.csv` → ticker-to-industry mapping
   - Lọc level 1 (broad sectors)

2. **Tạo Sector Features:**
   - Aggregate m1 data theo (industry, time)
   - Tính: sector_log_ret, sector_volume
   - Tính: sector_vol20, sector_vol5, sector_volume_ratio
   - NQT transform mỗi feature (window=252)

3. **Huấn luyện Sector HMM cho mỗi ngành:**
   - Grid search K ∈ [2, 3, 4] (cho từng ngành khác nhau)
   - Tương tự Market HMM
   - Chọn K tốt nhất cho từng ngành

4. **Auto-label theo Performance:**
   - Return cao + Vol thấp → Bull
   - Return thấp + Vol cao → Bear
   - Còn lại → Sideways / Intermediate states

5. **Lưu Sector Probabilities:**
   - `prob_sector_Bull`, `prob_sector_Bear`, v.v.

**Dau Ra:**
- `df_sector_hmm`: Toàn bộ sector data với regime & probabilities
- Mỗi ngành có K khác nhau (tối ưu riêng)

---

### PHAN 9: HUAN LUYEN TICKER HMM (K=3 fixed)

**Tien Hanh:**
1. **Cho mỗi ticker trong VN46:**
   - Tạo ticker features từ m1 data:
     - ticker_log_ret, ticker_vol20, ticker_vol5
     - ticker_volume_ratio
   - NQT transform

2. **Input cho Ticker HMM:**
   - Ticker features (4 cái)
   - Market probabilities (từ Phan 6b)
   - Sector probabilities (từ Phan 8)
   - K cố định = 3

3. **Huấn luyện:**
   - HMM với K=3 (cố định)
   - Dùng GMMHMM với jittered data

4. **Chạy filtered inference trên toàn bộ lịch sử**

5. **Output:**
   - `ticker_models` dictionary
   - `df_ticker_regimes` với regime labels
   - `ticker_probs` columns

**Dau Ra:**
- Mỗi ticker có riêng HMM model & probability predictions
- Chuẩn bị feature cho Meta-Classifier

---

### PHAN 9.1: CHUAN BI DATA BACKTEST (Walk-Forward Setup)

**Tien Hanh:**
1. **Tạo Master DataFrame:**
   - Merge tất cả m1 ticker data
   - Thêm Market_Prob columns
   - Thêm Sector_Prob columns
   - Thêm Ticker regimes
   - Tính target: 1D return tiếp theo (T+1)
   - Binarize target: return > 0 → 1 (UP), else 0 (DOWN)

2. **Chuẩn bị Feature Set:**
   - Market probabilities (Macro influence)
   - Sector probabilities (Industry influence)
   - Ticker probabilities (Ticker-specific regime)
   - Technical features: rolling_vol_20d, return_5d, volume_ratio

3. **Walk-Forward Loop (2022-01-01 → cuối):**
   - Ngày T: Train trên tất cả dữ liệu < T
   - Ngày T: Predict xác suất UP cho ngày T+1
   - Yêu cầu min 1000 training points
   - LGBMClassifier với n_estimators=100

**Dau Ra:**
- `df_backtest['final_meta_pred_prob']`: Xác suất tăng giá dự báo
- Performance metrics: ROC-AUC, Precision, Recall, F1

---

### PHAN 9.2: CHE DO LIVE TRADING (Prediction for T+1)

**Tien Hanh:**
1. **Training trên toàn bộ lịch sử:**
   - X_train = Tất cả dữ liệu lịch sử loại bỏ NaN target
   - Cả Market probs, Sector probs, Ticker probs

2. **Prediction cho ngày T (hôm nay):**
   - Lấy dữ liệu cuối cùng (latest_date)
   - X_test = Features của latest_date
   - Predict xác suất UP cho T+1

3. **Output:**
   - Top 10 tickets với xác suất tăng cao nhất
   - Tín hiệu: >= 0.5 → "BUY", < 0.5 → "SELL/HOLD"

**Dau Ra:**
- `live_results`: DataFrame với:
  - time, ticker, industry, close, final_meta_pred_prob
  - Trading signal (BUY/SELL)

---

### PHAN 9.3: THONG KE HIEU SUAT TAI CHINH (Financial Backtest)

**Tien Hanh:**
1. **Portfolio Strategy 1 - AI Focused (Top 5):**
   - Chỉ mua top 5 tickets với xác suất cao nhất
   - Nếu xác suất <= 0.5 → cash (return = 0%)

2. **Portfolio Strategy 2 - AI Equal Weight:**
   - Mua tất cả tickets với xác suất > 0.5
   - Equal weight cho mỗi cái

3. **Benchmark - Buy & Hold:**
   - Mua tất cả 46 tickets
   - Equal weight
   - Nắm giữ toàn bộ

4. **Mô phỏng (Backtesting):**
   - Lặp qua mỗi ngày trong OOS (2022+)
   - Giả lập giao dịch
   - Tính P&L

5. **Metrics:**
   - Annualized Return (%)
   - Annualized Volatility (%)
   - Sharpe Ratio
   - Max Drawdown (%)
   - Win Rate (% days profit)

**Dau Ra:**
- Comparison table: AI Focused vs AI Equal vs Buy&Hold
- Performance insights

---

## TONG KET LUONG TAI VA DAU RA TONG THE

### Tóm tắt Input:
| File | Nội dung | Kích thước |
|------|---------|-----------|
| hmm_data.csv | Market daily | 2382 rows |
| m1_vn46.csv | 46 tickers minute/hourly | ~109k rows |
| industries.csv | Mapping ticker-to-sector | 46 tickers |

### Tóm tắt Output (lưu tại OUTPUT_DIR):
| Loại | Nội dung |
|------|---------|
| Models | best_macro_hmm, best_market_hmm, ticker_models (46 cái) |
| Data | df_market (with regime), df_sector_hmm, df_ticker_regimes |
| Predictions | final_meta_pred_prob (xác suất T+1) |
| Backtest | ROC-AUC, Sharpe, Max Drawdown, Win Rate |

---

## CHI TIET THAM SO TUAN TRONG NOTEBOOK

### 1. Training Period:
- Cutoff: 2019-12-31
- Trước cutoff: Training
- Sau cutoff: Out-of-sample testing

### 2. HMM Configuration:
- **Macro**: K ∈ [2, 3] → chọn K=2
- **Market**: K ∈ [2, 3, 4] → chọn K=2
- **Sector**: K ∈ [2, 3, 4] → khác nhau từng ngành
- **Ticker**: K = 3 (cố định)

### 3. NQT Window:
- window = 252 (1 năm giao dịch)
- Clip range: [-3.0, 3.0]

### 4. Feature Selection:
- ADF test: p < 0.05
- KPSS test: p >= 0.05
- Kurtosis: |Kurt| < 10
- VIF: < 5.0

### 5. GMMHMM Configuration:
- n_mix = 2 (Gaussian mixtures)
- covariance_type = 'diag'
- min_covar = 0.01
- n_iter = 100-200 (Sector), 200 (Macro/Market)

### 6. LightGBM (Meta-Classifier):
- n_estimators = 100
- learning_rate = 0.05
- class_weight = 'balanced'
- n_jobs = -1 (parallel)

### 7. Random Seed:
- RANDOM_STATE = 42 (reproducibility)

---

## KET LUAN

### Quy Trình Toàn Bộ:
```
1. Load dữ liệu → 2. Filter stationarity → 3. NQT transform
   ↓
4. Tính MI & SHAP → 5. Lọc VIF → 6. Grid search HMM (Macro/Market)
   ↓
7. Auto-label states → 8. Sector HMM → 9. Ticker HMM
   ↓
10. Merge tất cả → 11. Walk-forward training → 12. Live prediction
   ↓
13. Backtest performance → 14. Output results
```

### Ưu Điểm Của Phương Pháp:
- ✓ Multi-layer HMM (Macro → Market → Sector → Ticker)
- ✓ Xử lý Publication Lag (shift 1 tháng)
- ✓ Feature selection khoa học (MI + SHAP + VIF)
- ✓ Online filtering (ngoại suy thời gian thực)
- ✓ Meta-classifier cuối cùng (tổng hợp tất cả signals)
- ✓ Walk-forward backtesting (tranh forward-bias)

---

**Ngôn Ngữ Viết: Vietnamese (Tiếng Việt)**
**Ngày Tạo: 2026-07-14**
**File Gốc: f.ipynb (30 cells, ~2400 lines)**
