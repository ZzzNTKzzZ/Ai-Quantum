# HYBRID DUAL-FREQUENCY TICKER-SPECIFIC HMM
## Toan Bo Quy Trinh Chi Tiet - Explained in Vietnamese

---

## 0. TONG QUAN CHUNG

### Tieu De Chinh (Main Title)
```
Hybrid Dual-Frequency Ticker-Specific HMM

Quy trinh ket hop danh gia Vi mao (Monthly) va Thi truong chung (Daily)
de dinh vi pha bien dong, sau do ep cau truc thi truong chung len
tung ma co phieu (Ticker-Specific) va dung Meta-Classifier du doan
loi suat ngay toi.
```

### Dac diem Kinh Dien (Key Features)
| Dac diem | Mo ta | Tac dung |
|----------|------|---------|
| **Dual-Frequency** | Ket hop du lieu Th?ng (Macro) va Ngay (Daily) | Nhan biet ca tren cum-hop tren dai han va bien dong cua ngay |
| **Multi-Layer HMM** | Xep tang: Macro -> Market -> Sector -> Ticker | Tao bo soan tin huu ich tu Tong the den Chi tiet |
| **Ticker-Specific** | Moi co phieu co HMM va xac suat rieng | Dieu luyen khong dung tren ng?nh chung
| **Meta-Classifier** | Su dung Sector & Market probs lam input | Tao niem tin du doan cao hon cho tung Ticker |
| **Publication Lag Handling** | Shift 1 thang du lieu Macro | Tranh Look-ahead Bias trong training |

---

## PHAN 1: Khoi tao He Thong & Load Thua Thao

### Mo Ta Chi Tiet (Detailed Description)
- Thiet lap duong dan output
- Import thu vien: numpy, pandas, statsmodels, sklearn, hmmlearn, lightgbm, shap, joblib
- Dat random seed = 42 de sinh ban tat dinh
- Dinh nghia ham get_hmm_filtered_inference() de thuc hien Online filtering cho HMM

### Dau Vao (Input)
- Phat hien: Thu vien

### Dau Ra (Output)
- Output directory: ../output/hmm_v3_op1_extended

---

## PHAN 2: Tai Du Lieu & Tao Chi Ba Thi Truong Dai Dien (VN-Index Proxy)

### Mo Ta Chi Tiet (Detailed Description)
- Doc 2 file CSV: hmm_data.csv (Daily) va m1_vn46.csv (Minute/Hourly)
- Tao VN-Index proxy tu trung binh danh gia cac cophieu trong ro VN46
- Tinh toan: log_return, close price, rolling vol 20 ngay
- Them du lieu phu: foreign net buy/sell, USD/VND exchange rate
- Loai bo hang co NaN value

### Dau Vao (Input)
- CSV files: ../output/hmm_data.csv, ../data/processed/m1_vn46.csv

### Dau Ra (Output)
- df_market: Dataframe co (2382 hang, 13 cot)

---

## PHAN 3: Bo Loc Kiem Dinh Ky Thuat (Stationarity & Kurtosis Filter)

### Mo Ta Chi Tiet (Detailed Description)
- Kiem dinh tinh dung (I(0)): Dung ADF & KPSS test (p < 0.05 & p >= 0.05)
- Kiem dinh he so nhon (Excess Kurtosis): |Kurt| < 10
- Loai bo bien khong on dinh (non-stationary) hoac co duoi qua day
- Giu lai 5 bien: rolling_vol_5, volume_ratio, credit_growth_mom, cpi_mom, fnb_ratio

### Dau Vao (Input)
- df_market voi 9 bien sang loc

### Dau Ra (Output)
- selected_raw_features: 5 bien duoc chon

---

## PHAN 4: Chuan hoa Du Lieu NQT (Normal Quantile Transform) & Rank

### Mo Ta Chi Tiet (Detailed Description)
- Dung Normal Quantile Transform (NQT) de chuyen du lieu thong thuong sang phan phoi chuan N(0,1)
- Rolling Rank tren cua so 252 ngay
- Clip gia tri toi [-3.0, 3.0] de tranh extreme outliers
- Tach training (truoc 2019-12-31) va out-of-sample (2020+) data

### Dau Vao (Input)
- selected_raw_features (5 bien) voi timestamp

### Dau Ra (Output)
- df_market_Z: Bien Z-normalized; fd_market, Z_tr_market, Z_all_market

---

## PHAN 5: MI & Chon Bien Tham Lam + Kiem Soat VIF

### Mo Ta Chi Tiet (Detailed Description)
- Tinh Mutual Information (MI) toi dich |vnindex_log_ret| (bieu dien bien dong)
- Tinh SHAP importance tu LightGBM classifier 3-class
- Lo chon bien: total_score = SHAP * MI
- Loc VIF tham lam: VIF < 5.0 (tranh da cong tuyen)
- Tach thanh: Macro Features (3) va Market Features (2)

### Dau Vao (Input)
- X_train: 5 bien Z-normalized

### Dau Ra (Output)
- final_features: [volume_ratio_Z, cpi_mom_Z, fnb_ratio_Z, credit_growth_mom_Z, rolling_vol_5_Z]

---

## PHAN 6: Grid Search HMM Toi Uu (Macro & Market Layer)

### Mo Ta Chi Tiet (Detailed Description)
- **Macro HMM (Monthly)**: Grid search K=[2,3], GMMHMM voi 2 mix components
- Tinh BIC, LL OOS, min_duration >= 2, state share [0.05, 0.85]
- Chon theo composite = 0.5*Rank_bic + 0.5*Rank_oos (K_macro=2)
- **Market HMM (Daily)**: Grid search K=[2,3,4], input = Market + Macro_Prob
- Composite = 0.3*Rank_bic + 0.5*Rank_oos + 0.2*Rank_mindur (K_market=2)

### Dau Vao (Input)
- Z_train_macro (monthly), Z_train_market (daily)

### Dau Ra (Output)
- best_macro_hmm (K=2), best_market_hmm (K=2), Macro_Prob & Market_Prob

---

## PHAN 7: Tu Dong Anh Xa & Gan Nhan Trang Thai (K-agnostic Labeling)

### Mo Ta Chi Tiet (Detailed Description)
- Auto-label Macro: Stagnant/Stable/Expansion theo pmi_vn
- Auto-label Market (K=2): Bear vs Bull theo return va volatility
- Su dung Sharpe ratio (ret/vol) hoac linear_sum_assignment cho K>=4
- Tao STATE_TO_LABEL_MARKET va STATE_TO_LABEL_MACRO mapping

### Dau Vao (Input)
- Market regimes voi K=2, Macro regimes voi K=2

### Dau Ra (Output)
- market_regime_label: 'Bear' hoac 'Bull'; macro_regime_label

---

## PHAN 8: Chuan Bi Du Lieu Nganh & Huan Luyen Sector HMM

### Mo Ta Chi Tiet (Detailed Description)
- Load industries.csv va anh xa ticker -> industry
- Tao sector features: sector_log_ret, sector_volume, sector_vol20, sector_vol5, sector_volume_ratio
- NQT transform cac sector features
- Grid search K=[2,3,4] cho moi industry rieng biet
- Output: df_sector_hmm voi sector_regime va prob_sector_* columns

### Dau Vao (Input)
- df_m1 (minute data), industries.csv

### Dau Ra (Output)
- df_sector_hmm: Moi industry co sector HMM, K = [2 den 4]

---

## PHAN 9: Huan Luyen Ticker HMM (K=3 fixed)

### Mo Ta Chi Tiet (Detailed Description)
- Lap tung ma co phieu trong VN46
- Tao ticker features tu m1 data: log_ret, vol20, vol5, volume_ratio
- NQT transform ticker features
- Input = Ticker features + Market_Prob + Sector_Prob (K=3 fixed)
- HMM filtered inference tren toan bo lich su
- Luu tru model, regime, va probability

### Dau Vao (Input)
- df_sector_hmm, m1_vn46.csv, market_probs, sector_probs

### Dau Ra (Output)
- ticker_models (dict), df_ticker_regimes, ticker_probs

---
