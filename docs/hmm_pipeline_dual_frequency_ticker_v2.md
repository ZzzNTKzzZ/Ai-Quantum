# HƯỚNG DẪN KỸ THUẬT: HMM PIPELINE ĐA TẦN SỐ PHÂN CẤP CHO TỪNG MÃ CỔ PHIẾU (BẢN V2)
*(Hierarchical Ticker-Specific Dual-Frequency HMM Pipeline - Version 2)*

Tài liệu này thuyết minh chi tiết cấu trúc kiến thức, luồng dữ liệu, logic xử lý, hệ thống chỉ báo, công thức toán học và quy trình thực thi mã nguồn của phiên bản cập nhật (V2) trong notebook [hmm_pipeline_dual_frequency_ticker.ipynb](file:///C:/Users/ADMIN/Desktop/Kaggle/notebooks/hmm_pipeline_dual_frequency_ticker.ipynb).

---

## 1. KIẾN THỨC NỀN TẢNG (KNOWLEDGE)

Mô hình **Ticker-Specific Hierarchical Dual-Frequency HMM** được thiết kế để giải quyết bài toán phi tuyến tính và nhiễu động cao của chuỗi thời gian tài chính thông qua việc cấu trúc trạng thái thị trường dưới dạng phân cấp đa tần số:

*   **Hidden Markov Model (HMM):** Giả định thị trường vận hành qua các trạng thái ẩn (regimes) không thể quan sát trực tiếp, quyết định xác suất phân phối của các biến số thực nghiệm (lợi suất, biến động, khối lượng).
*   **Gaussian Mixture Model HMM (GMMHMM):** Sử dụng hỗn hợp phân phối chuẩn để mô hình hóa hàm phát xạ. Giúp nắm bắt tính bất đối xứng, biến động vọt và phân phối đuôi dày của tài sản tài chính tốt hơn hẳn mô hình Gauss đơn lẻ.
*   **Cấu trúc phân cấp 2 tầng đa tần số (Hierarchical Dual-Frequency):**
    1.  *Tầng vĩ mô dài hạn (Monthly Macro HMM):* Hoạt động ở tần suất thấp (Tháng) nhằm xác định bối cảnh kinh tế lớn (Tăng trưởng mở rộng hoặc Suy thoái trì trệ) tác động đến xu thế dòng tiền dài hạn.
    2.  *Tầng thị trường ngắn hạn (Daily Market HMM):* Hoạt động ở tần suất cao (Ngày) trên các chỉ báo kỹ thuật của chỉ số đại diện để định vị các pha dao động ngắn hạn của thị trường.
*   **Suy luận Ticker-Specific từ Global Parameters:** Khắc phục hạn chế dữ liệu lịch sử ngắn của từng mã cổ phiếu bằng cách huấn luyện các tham số xác suất phát xạ và chuyển dịch trạng thái chung (Global Daily Market HMM) trên chỉ số thị trường đại diện (VN-Index Proxy). Sau đó, dùng chính bộ tham số này để giải mã chuỗi trạng thái ẩn cho riêng từng mã cổ phiếu dựa trên đặc trưng biến động nội tại của mã đó (`rolling_vol_5_ticker`).

---

## 2. LUỒNG DỮ LIỆU (DATAFLOW)

Quy trình dịch chuyển dữ liệu qua các pha xử lý và biến đổi để tạo ra kết quả sạch cuối cùng:

### 2.1. Nguồn dữ liệu đầu vào (Data Retrieval)
*   **Dữ liệu vĩ mô & đặc trưng kỹ thuật cơ sở:** [hmm_data.csv](file:///C:/Users/ADMIN/Desktop/Kaggle/output/hmm_data.csv) (PMI Việt Nam, CPI MoM, Tăng trưởng tín dụng MoM, Amihud Liquidity, Return Dispersion).
*   **Dữ liệu giá giao dịch của 46 mã cổ phiếu:** [m1_vn46.csv](file:///C:/Users/ADMIN/Desktop/Kaggle/data/processed/m1_vn46.csv) (Open, High, Low, Close, Volume, log_return, rolling_vol_20d).
*   **Dữ liệu dòng tiền khối ngoại:** [m4_foreign_net_buy_sell.csv](file:///C:/Users/ADMIN/Desktop/Kaggle/data/processed/m4_foreign_net_buy_sell.csv) (fnb_ratio).
*   **Dữ liệu tỷ giá ngoại tệ:** [e1_usdvnd.csv](file:///C:/Users/ADMIN/Desktop/Kaggle/data/processed/e1_usdvnd.csv) (fx_log_ret).

### 2.2. Các bước biến đổi dữ liệu (Data Transformation)
1.  **Đồng bộ lưới thời gian (Date Alignment):** Đồng nhất hóa lịch giao dịch của toàn bộ 46 mã cổ phiếu và các biến kinh tế/thanh khoản.
2.  **Khử rò rỉ thông tin vĩ mô (Macro Lag Shift):** Áp dụng độ trễ 1 tháng (`shift(1)`) cho toàn bộ biến tần suất tháng trước khi đồng bộ hóa sang lưới thời gian ngày. Điều này đảm bảo trạng thái vĩ mô được sử dụng tại ngày $t$ hoàn toàn đã được công bố chính thức.
3.  **Lọc tính chất chuỗi thời gian (Time Series Filters):** Thực hiện kiểm định ADF & KPSS để loại bỏ các đặc trưng không dừng ($I(1)$) và lọc bỏ các biến có độ nhọn vượt ngưỡng ($|Kurt| \ge 10$).
4.  **Xếp hạng lượng tin (Mutual Information Scoring):** Đo lường sự phụ thuận phi tuyến giữa biến đầu vào với $|vnindex\_log\_ret|$ để giữ lại các đặc trưng chứa nhiều thông tin nhất.
5.  **Kiểm soát đa cộng tuyến (VIF Control):** Loại bỏ các biến gây trùng lặp thông tin bằng cách duy trì Variance Inflation Factor ($VIF < 5.0$).
6.  **Nonparametric Quantile Transform (NQT):** Ánh xạ phân phối thực nghiệm của các đặc trưng được chọn trong cửa sổ lăn 252 ngày về phân phối chuẩn chuẩn tắc $\mathcal{N}(0, 1)$ nhằm thỏa mãn nghiêm ngặt giả định toán học của mô hình phát xạ HMM.

### 2.3. Kết quả đầu ra cuối cùng (Output Generation)
*   [master_drl_ready_ticker.parquet](file:///C:/Users/ADMIN/Desktop/Kaggle/output/hmm_dual/master_drl_ready_ticker.parquet): Bảng dữ liệu lớn tích hợp đầy đủ giá giao dịch, đặc trưng kỹ thuật, các trạng thái ẩn HMM đã gán nhãn (`market_regime_label`, `macro_regime_label`, `joint_regime_label`) và xác suất trạng thái tương ứng của toàn bộ 46 mã cổ phiếu.
*   [hmm_regimes_merged_ticker.csv](file:///C:/Users/ADMIN/Desktop/Kaggle/output/hmm_dual/hmm_regimes_merged_ticker.csv): File kết quả chỉ bao gồm thông tin định vị thời gian, mã cổ phiếu và các đặc trưng xác suất/trạng thái ẩn phục vụ tra cứu nhanh.
*   **Splits dữ liệu thời gian** lưu trữ trong thư mục [splits_ticker](file:///C:/Users/ADMIN/Desktop/Kaggle/output/hmm_dual/splits_ticker/):
    *   `train_set.parquet`: Dữ liệu huấn luyện ($\le$ `2019-12-31`).
    *   `val_set.parquet`: Dữ liệu hiệu chỉnh (`2020-01-01` đến `2022-12-31`).
    *   `test_set.parquet`: Dữ liệu kiểm thử độc lập ngoài mẫu ($\ge$ `2023-01-01`).

---

## 3. QUY TRÌNH THỰC THI MÃ NGUỒN (CODE EXECUTION FLOW)

Notebook thực thi tuần tự qua 13 bước khép kín:

```
[Khởi tạo môi trường] -> [Tải & Đồng bộ VN-Index Proxy] -> [Kiểm định dừng & nhọn]
                                                                     |
[Thiết lập NQT & HMM] <- [Lọc đa cộng tuyến VIF] <- [Đánh giá MI] <--+
          |
[Grid Search K & D] -> [Chọn cấu hình tối ưu] -> [Refit HMM cuối cùng]
                                                         |
[Lưu Parquet & Splits] <- [Ticker-Specific & Filter] <- [Gán nhãn trạng thái]
          |
[Widget trực quan hóa tương tác]
```

1.  **Thiết lập:** Cài đặt seed ngẫu nhiên (`42`) và tạo thư mục đầu ra `../output/hmm_dual`.
2.  **Đồng bộ chỉ số:** Tính toán biến động năm hóa của thị trường chung `vnindex_vol20` từ bình quân log return rổ VN46 và ghép nối với dòng tiền khối ngoại, tỷ giá.
3.  **Kiểm định thống kê:** Lọc các đặc trưng dừng và có độ nhọn đạt yêu cầu thống kê.
4.  **Tính điểm MI:** Tính toán và sắp xếp điểm thông tin tương hỗ đối với biến động tuyệt đối thị trường.
5.  **Lọc biến tham lam:** Đảm bảo tính đa dạng thông tin (lấy tối thiểu 1 biến từ mỗi nhóm Market, Economy, Credit) và kiểm soát $VIF < 5.0$.
6.  **Định nghĩa hàm:** Thiết lập hàm biến đổi NQT rolling 252 ngày, tính toán tham số tự do HMM và các tiêu chí đánh giá mô hình.
7.  **Grid Search:** Duyệt qua các số lượng trạng thái ẩn vĩ mô tháng $K \in [2, 3, 4]$ và trạng thái ngày $K \in [3, 4]$, số đặc trưng ngày $D \in [4, 5, 6]$.
8.  **Lựa chọn mô hình:** Đánh giá bằng điểm số Composite xếp hạng BIC, OOS Log-likelihood và thời gian lưu trú tối thiểu để tìm ra cấu hình ngày tối ưu nhất (`n_features=4`, `K=3`). Chọn cấu hình tháng tối ưu (`K=2`).
9.  **Huấn luyện lại:** Fit lại mô hình Monthly HMM và Daily Market GMMHMM trên tập Train ($\le$ 2019-12-31) để lưu tham số tối ưu.
10. **Gán nhãn tự động:** Ánh xạ trạng thái vĩ mô thành `Macro_Stagnant`, `Macro_Expansion` dựa trên PMI trung bình. Gán nhãn trạng thái ngày của mô hình K=3 thành `Bull`, `Bear`, `Sideways` (trong đó `Bear` có lợi suất trung bình thấp nhất, `Bull` có biến động thấp nhất trong hai trạng thái còn lại, và `Sideways` là trạng thái còn lại).
11. **Suy luận Ticker-Specific:** Đối với từng mã trong 46 mã, sử dụng tham số Global HMM giải mã trạng thái cục bộ. Áp dụng **Persistent Filter** (Majority vote 5 phiên + 80% Volatility breakout threshold) để lọc nhiễu chattering. Ghép nối với HMM vĩ mô lùi và tạo nhãn `joint_regime_label`.
12. **Ghi tệp tin:** Lưu Parquet master và phân chia 3 tập Train/Val/Test độc lập theo thời gian.
13. **Trực quan hóa:** Hiển thị widget tương tác Matplotlib tô màu nền theo trạng thái ẩn HMM của mã cổ phiếu được chọn.

---

## 4. CƠ SỞ TOÁN HỌC & CÔNG THỨC GIẢI NGHĨA (MATHEMATICS & FORMULAS)

### 4.1. Nonparametric Quantile Transform (NQT)
NQT biến đổi một biến ngẫu nhiên bất kỳ $X$ có hàm phân phối tích lũy (CDF) thực nghiệm $F_X(x)$ thành một biến ngẫu nhiên chuẩn tắc $Z \sim \mathcal{N}(0, 1)$:
$$pct_t = \frac{Rank(X_t) - 0.5}{N_{window}}$$
$$Z_t = \Phi^{-1}(pct_t)$$
*   $Rank(X_t)$ là thứ hạng của $X_t$ trong cửa sổ trượt $N_{window} = 252$ ngày.
*   $\Phi^{-1}$ là hàm phân phối chuẩn tích lũy ngược (Probit function).
*   *Ý nghĩa:* Triệt tiêu ảnh hưởng của phân phối đuôi dày hoặc bất đối xứng, đưa đặc trưng đầu vào về dạng chuẩn hoàn hảo, khớp tối ưu với cấu trúc Gaussian Emission của HMM.

### 4.2. Số lượng tham số tự do trong GMMHMM ($p$)
Để tính điểm BIC chính xác cho mô hình Gauss hỗn hợp với $K$ trạng thái ẩn, số chiều đặc trưng $D$, và $M=2$ hỗn hợp thành phần:
$$p = (K - 1) + K(K - 1) + K(M - 1) + KMD + KMD\frac{D(D+1)}{2}$$
*   $(K - 1)$: Tham số xác suất trạng thái khởi đầu.
*   $K(K - 1)$: Tham số ma trận xác suất chuyển trạng thái.
*   $K(M - 1)$: Tham số trọng số hỗn hợp (Mixture weights).
*   $KMD$: Tham số vector kỳ vọng (Means).
*   $KMD\frac{D(D+1)}{2}$: Tham số ma trận hiệp phương sai đầy đủ (Full Covariances).

### 4.3. Tiêu chuẩn thông tin Bayesian (BIC)
Dùng để phạt độ phức tạp của mô hình nhằm ngăn ngừa quá khớp (Overfitting):
$$BIC = -2 \ln(\hat{L}) + p \ln(N)$$
*   $\hat{L}$: Log-likelihood cực đại của mô hình trên tập Train.
*   $N$: Số lượng điểm dữ liệu huấn luyện.
*   *Ý nghĩa:* Mô hình có BIC càng thấp thì càng tối ưu (khớp tốt với lượng tham số tinh gọn nhất).

### 4.4. Thời gian lưu trú kỳ vọng ở một trạng thái (Expected State Duration)
$$\tau_i = \frac{1}{1 - a_{ii}}$$
*   $a_{ii}$ là xác suất nằm trên đường chéo chính của ma trận chuyển dịch trạng thái (xác suất ở lại trạng thái $i$).
*   *Ý nghĩa:* Đo lường độ bền vững theo thời gian của trạng thái ẩn. Pipeline yêu cầu $\tau_i \ge 3$ để tránh nhiễu chuyển đổi trạng thái liên tục.

### 4.5. Điểm Composite Score xếp hạng
$$Composite = 0.3 \cdot Rank_{bic} + 0.5 \cdot Rank_{oos} + 0.2 \cdot Rank_{min\_dur}$$
*   *Ý nghĩa:* Kết hợp đa mục tiêu (phạt phức tạp BIC, năng lực tổng quát hóa OOS Log-likelihood, và độ ổn định trạng thái) để tự động chọn cấu hình tối ưu.

---

## 5. ƯU NHƯỢC ĐIỂM & ĐÓNG GÓP CHO QUÁ TRÌNH HUẤN LUYỆN MODEL

### 5.1. Nhận xét Ưu điểm & Nhược điểm kỹ thuật
*   **Ưu điểm:**
    *   **Thực tế hóa quy trình giao dịch:** Việc loại bỏ hoàn toàn Look-ahead Bias bằng cách dịch pha biến vĩ mô và phân chia tập dữ liệu nghiêm ngặt theo thời gian giúp tránh ảo tưởng hiệu suất khi Backtest.
    *   **Giảm nhiễu hiệu quả:** Bộ lọc Persistent Filter giúp triệt tiêu nhiễu trạng thái ẩn cục bộ cực tốt. Các trạng thái vĩ mô và thị trường ổn định giúp Agent dễ dàng học được các chính sách phân bổ dài hạn.
    *   **Ổn định toán học:** Phép chuẩn hóa NQT giúp giải quyết triệt để lỗi phân phối dữ liệu tài chính không chuẩn, đảm bảo tính hội tụ của thuật toán tối ưu hóa EM.
*   **Nhược điểm:**
    *   **Độ trễ phản ứng:** Bộ lọc Persistent Filter sử dụng majority vote 5 phiên có thể tạo ra độ trễ khoảng 1-2 ngày khi thị trường chuyển trạng thái thực sự (nhưng được bù đắp bằng việc cho phép đổi trạng thái lập tức nếu biến động vượt ngưỡng 80% lịch sử).
    *   **Tính toán nặng:** Các phép xếp hạng rolling trượt liên tục trên toàn bộ dữ liệu 46 mã làm tăng thời gian tiền xử lý.

### 5.2. Vai trò đối với quá trình Train mô hình DRL (Deep Reinforcement Learning)
Trong bài toán tối ưu hóa danh mục đầu tư bằng DRL:
1.  **Xác định không gian trạng thái (State Space):** Trạng thái vĩ mô (`macro_regime_label`) và trạng thái thị trường ngày (`market_regime_label`) đóng vai trò là các biến Contextual cực kỳ quan trọng. Chúng cung cấp cho Agent "bản đồ thời tiết" hiện tại của thị trường để điều chỉnh mức độ chấp nhận rủi ro.
2.  **Đảm bảo tính chất Markov:** Chuỗi trạng thái ẩn HMM được làm mượt giúp biểu diễn thông tin cô đọng, thỏa mãn giả định Markov ($P(S_{t+1} | S_t, ..., S_0) = P(S_{t+1} | S_t)$). Nếu không có HMM, dữ liệu lợi suất nhiễu động cao sẽ phá vỡ tính chất này.
3.  **Tránh rò rỉ thông tin:** DRL Agent được huấn luyện trên `train_set.parquet`, hiệu chỉnh siêu tham số trên `val_set.parquet` và kiểm tra trên `test_set.parquet`. Sự phân chia phi rò rỉ bảo đảm Agent học được quy luật thực sự chứ không phải học thuộc lòng tương lai.
4.  **Tín hiệu xác suất mượt:** Các cột xác suất trạng thái liên tục (`prob_market_k`) cung cấp đầu vào dạng số thực phong phú, giúp các mạng deep Q-network hoặc Policy Gradient định lượng chính xác độ bất định của thị trường.
