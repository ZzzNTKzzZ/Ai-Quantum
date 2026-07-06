# HƯỚNG DẪN KỸ THUẬT: HMM PIPELINE ĐA TẦN SỐ PHÂN CẤP TÙY BIẾN CHO TỪNG MÃ CỔ PHIẾU (BẢN V3)
*(Hierarchical Ticker-Specific Independent Dual-Frequency HMM Pipeline - Version 3)*

Tài liệu này thuyết minh chi tiết cấu trúc kiến thức, luồng dữ liệu, logic xử lý, hệ thống chỉ báo, công thức toán học và quy trình thực thi mã nguồn của phiên bản tối ưu và đột phá nhất (V3) trong notebook [hmm_pipeline_dual_frequency_ticker_v3.ipynb](file:///C:/Users/ADMIN/Desktop/Kaggle/notebooks/hmm_pipeline_dual_frequency_ticker_v3.ipynb).

---

## 1. KIẾN THỨC NỀN TẢNG (KNOWLEDGE)

Mô hình **Ticker-Specific Hierarchical Dual-Frequency HMM V3** được thiết kế để giải quyết bài toán định vị trạng thái rủi ro của từng cổ phiếu trong môi trường tài chính cực kỳ nhiễu động, tập trung vào tính cá nhân hóa và định hướng xu hướng:

*   **Hidden Markov Model (HMM):** Giả định thị trường vận hành qua các trạng thái ẩn (regimes) quyết định xác suất phân phối của các biến số thực nghiệm.
*   **Gaussian Mixture Model HMM (GMMHMM):** Sử dụng hỗn hợp 2 phân phối chuẩn (`n_mix=2`) để mô hình hóa hàm phát xạ, xử lý các đuôi dày (fat tails) trong tài chính.
*   **Cấu trúc phân cấp đa tần số (Hierarchical Dual-Frequency):**
    1.  *Tầng vĩ mô dài hạn (Monthly Macro HMM):* Hoạt động ở tần suất thấp (Tháng) nhằm xác định bối cảnh kinh tế lớn (Tăng trưởng hoặc Suy thoái).
    2.  *Tầng cổ phiếu riêng biệt (Daily Ticker-Specific HMM):* Điểm khác biệt lớn nhất so với V2. Thay vì dùng chung một mô hình của thị trường (VN-Index) để áp đặt lên toàn bộ cổ phiếu, V3 **huấn luyện tự động một mô hình HMM độc lập cho từng mã cổ phiếu riêng biệt**. Điều này giúp mô hình hiểu được "tính cách rủi ro" khác nhau (ví dụ: biến động của FPT khác biệt hoàn toàn với biến động của DIG).
*   **Tích hợp Động lượng (Momentum Awareness):** Để khắc phục "căn bệnh mù hướng" của HMM truyền thống (dễ nhầm lẫn giữa Tích lũy và Phân phối vì cùng có biến động thấp), V3 cung cấp thêm các biến động lượng dài hạn và ngắn hạn cho HMM gom cụm.

---

## 2. LUỒNG DỮ LUỆU (DATAFLOW)

Quy trình dịch chuyển dữ liệu qua các pha xử lý và biến đổi:

### 2.1. Nguồn dữ liệu đầu vào (Data Retrieval)
*   **Dữ liệu vĩ mô & đặc trưng kỹ thuật cơ sở:** `hmm_data.csv` (PMI Việt Nam, CPI MoM, Tăng trưởng tín dụng MoM, Amihud Liquidity).
*   **Dữ liệu giá giao dịch của 46 mã cổ phiếu:** (Open, High, Low, Close, Volume, log_return).
*   **Dữ liệu dòng tiền khối ngoại & Tỷ giá:** `fnb_ratio` và `fx_log_ret`.

### 2.2. Các bước biến đổi dữ liệu (Data Transformation)
1.  **Trích xuất Momentum:** Tính toán lợi suất 20 phiên (`mom_1M`) và độ lệch giá so với trung bình trượt 50 phiên (`dist_MA50`).
2.  **Khử rò rỉ thông tin vĩ mô (Macro Lag Shift):** Áp dụng độ trễ 1 tháng (`shift(1)`) cho toàn bộ biến tần suất tháng.
3.  **Nonparametric Quantile Transform (NQT) dạng Mở rộng (Expanding Anchor):**
    Thay vì dùng cửa sổ trượt 252 ngày như V2 (làm mất quy mô tuyệt đối của các cú sập), V3 sử dụng cửa sổ **Expanding**. Nghĩa là dữ liệu mới tại ngày $t$ luôn được so sánh thứ hạng với *toàn bộ lịch sử giao dịch từ ngày đầu tiên lên sàn*. Điều này duy trì được "Ký ức dài hạn" của thị trường.

### 2.3. Kết quả đầu ra cuối cùng (Output Generation)
Toàn bộ kết xuất được đưa vào thư mục độc lập `output/hmm_v3/`:
*   [master_drl_ready_ticker.parquet](file:///C:/Users/ADMIN/Desktop/Kaggle/output/hmm_v3/master_drl_ready_ticker.parquet): Dữ liệu lớn tích hợp đầy đủ tính toán nội tại, trạng thái ẩn (`market_regime_label`, `macro_regime_label`, `joint_regime_label`).
*   [hmm_regimes_merged_ticker.csv](file:///C:/Users/ADMIN/Desktop/Kaggle/output/hmm_v3/hmm_regimes_merged_ticker.csv).
*   **Splits dữ liệu thời gian** `train_set.parquet`, `val_set.parquet`, `test_set.parquet` (Dành riêng cho DRL).

---

## 3. QUY TRÌNH THỰC THI MÃ NGUỒN (CODE EXECUTION FLOW)

V3 hoạt động thông qua cơ chế tự động hóa cực kỳ linh hoạt:

1.  **Đồng bộ & Chuẩn bị biến vĩ mô:** Chuẩn hóa NQT dữ liệu Vĩ mô và Fit mô hình Macro HMM (K=2).
2.  **Lựa chọn đặc trưng V3:** Cấu hình cố định `DAILY_FEATURES_V3 = ['rolling_vol_5', 'fx_log_ret', 'amihud_diff_normalized', 'mom_1M', 'dist_MA50']`.
3.  **Xử lý Vòng lặp từng Mã cổ phiếu (Ticker Loop):** Đối với từng mã trong 46 mã cổ phiếu:
    *   Tính toán Momentum và các biến cục bộ.
    *   Thực hiện Expanding NQT để khử nhiễu.
    *   **Fit HMM Độc Lập:** Khởi tạo `GMMHMM` với `covariance_type='diag'`. Fit riêng trên tập Train của cổ phiếu đó.
    *   **Tự động Gán Nhãn (Dynamic Auto-label):** Gọi hàm tính toán lại tỷ suất sinh lời thực tế của từng cụm 0, 1, 2 cho cổ phiếu đó, từ đó linh hoạt gán lại các nhãn `Bear`, `Bull`, `Sideways` tương ứng.
    *   Áp dụng **Persistent Filter** (Majority vote 5 phiên + 80% Volatility Breakout) để chống nhiễu ngắn hạn.
    *   Ghép nối nhãn ghép `joint_regime_label`.
4.  **Lưu kết quả phân chia (Splits):** Kết xuất trực tiếp ra cấu trúc sẵn sàng cho RL.

---

## 4. CƠ SỞ TOÁN HỌC & CÔNG THỨC GIẢI NGHĨA (MATHEMATICS & FORMULAS)

### 4.1. Nonparametric Quantile Transform (NQT) và Dạng Mở rộng (Expanding Anchor)

**a) Cơ bản: NQT là gì và Cách hoạt động?**
*   **Vấn đề:** Trong tài chính, dữ liệu lợi suất và biến động thường có hiện tượng "đuôi dày" (Fat Tails) và bất đối xứng – tức là các cú sập (Crash) xảy ra nhiều và khốc liệt hơn lý thuyết phân phối chuẩn. Tuy nhiên, mô hình HMM Gaussian lại yêu cầu đặc trưng đầu vào phải tuân theo phân phối chuẩn. Nếu đưa dữ liệu thô vào, HMM sẽ bị nhiễu và gom cụm sai lệch.
*   **Giải pháp NQT (Nonparametric Quantile Transform):** NQT ép một chuỗi dữ liệu bất kỳ về phân phối chuẩn hoàn hảo mà không cần quan tâm đến hình dáng ban đầu của nó (Phi tham số - Nonparametric).
    1. Lấy dữ liệu $X_t$ và xếp hạng (Rank) nó từ bé đến lớn.
    2. Chuyển thứ hạng này thành điểm phần trăm (Percentile) từ $0$ đến $1$.
    3. Dùng hàm Probit $\Phi^{-1}$ (nghịch đảo của phân phối chuẩn) để ánh xạ điểm phần trăm này thành điểm $Z$-score trên đường cong chuông $\mathcal{N}(0, 1)$ chuẩn mực.

**b) Dạng Mở rộng (Expanding Anchor) là gì? Nó giải quyết vấn đề gì?**
*   **Vấn đề của Cửa sổ trượt (Rolling NQT - Bản cũ):** Trước đây ta dùng cửa sổ trượt (ví dụ 252 ngày). Điều này có nghĩa là mức độ rủi ro hôm nay chỉ được so sánh với 1 năm qua. Nếu thị trường đi ngang biên độ hẹp cả năm, một phiên giảm nhẹ -2% cũng bị đẩy lên thành "Cú sập tồi tệ nhất năm" (Z-score chạm đáy). Rolling Window làm mất đi "Ký ức dài hạn" (Absolute Scale Loss).
*   **Giải pháp Expanding NQT (Bản V3):** Tại ngày giao dịch $t$, giá trị $X_t$ sẽ được xếp hạng so với **toàn bộ dữ liệu lịch sử tính từ ngày đầu tiên cổ phiếu lên sàn (từ $1 \rightarrow t$)**:
    $$pct_t = \frac{Rank_{1 \rightarrow t}(X_t) - 0.5}{t}$$
    $$Z_t = \Phi^{-1}(pct_t)$$
*   **Cách hoạt động & Ý nghĩa thực chiến:** Một cú sập kỷ lục vào năm 2022 sẽ được so sánh với toàn bộ 10 năm lịch sử trước đó, giữ nguyên mức độ nghiêm trọng rủi ro đuôi đen (Black Swan). Mô hình HMM sẽ phân định rạch ròi được đâu là "Nhiễu động nhẹ trong Downtrend" và đâu là "Khủng hoảng bán tháo" mà tuyệt đối không bị rò rỉ dữ liệu tương lai (vì tại ngày $t$, ta chỉ dùng dữ liệu đến $t$).

### 4.2. Hiệp phương sai Đường chéo (Diagonal Covariance Matrix)
Ở V3, ta sử dụng `covariance_type='diag'` thay vì `'full'`:
$$ \Sigma_k = \begin{bmatrix} \sigma_1^2 & 0 & \dots & 0 \\ 0 & \sigma_2^2 & \dots & 0 \\ \dots & \dots & \ddots & \dots \\ 0 & 0 & \dots & \sigma_D^2 \end{bmatrix} $$
*   *Ý nghĩa:* Giả định các đặc trưng (momentum, volatility, fx) độc lập có điều kiện nếu biết trước trạng thái ẩn. Sự đánh đổi này làm giảm mạnh số lượng tham số từ $\approx 120$ xuống còn $\approx 50$. Nhờ đó, ma trận hiệp phương sai không bị "sụp đổ" (Singular/Not Positive Definite) khi phải fit thuật toán trên tập dữ liệu nhỏ hẹp của một mã cổ phiếu.

---

## 5. ƯU NHƯỢC ĐIỂM & ĐÓNG GÓP CHO QUÁ TRÌNH HUẤN LUYỆN MODEL

### 5.1. Nhận xét Ưu điểm & Nhược điểm kỹ thuật
*   **Ưu điểm (So với V1 và V2):**
    *   **Giải quyết "Bệnh mù hướng":** Có Momentum, HMM tự học cách tách biệt biến động sinh ra bởi các đợt bán tháo (Bear) với biến động sinh ra bởi lực cầu đẩy giá lên (Bull).
    *   **Cá nhân hóa Rủi ro (Ticker-tailored):** Nhận diện được cấu trúc giá đặc thù của cổ phiếu lớn vs cổ phiếu nhỏ lẻ.
    *   **Ký ức dài hạn:** Khắc phục được nhược điểm "mất mốc tham chiếu" do chuẩn hóa Rolling window gây ra nhờ Expanding NQT.
*   **Nhược điểm:**
    *   **Bất ổn giai đoạn khởi thủy (Burn-in Instability):** Trong 1-2 năm đầu tiên khi một mã mới niêm yết lên sàn, hàm `expanding` có lượng mẫu số quá nhỏ, làm cho tính phân vị (percentile) thiếu chính xác và dao động cực mạnh.
    *   **Dễ lọt đáy (Sparsity of Features):** Một số mã cổ phiếu có tính thanh khoản ngắt quãng hoặc bị lỗi niêm yết sẽ khiến HMM cá nhân khó hội tụ hơn.
    *   Vẫn mang độ trễ 2-4 ngày nhất định khi xác nhận đảo chiều do cơ chế Lọc đa số (Majority Filter).

### 5.2. Sức mạnh đột phá đối với Model DRL (Deep Reinforcement Learning)
V3 cung cấp một **Môi trường Trạng thái (State Environment)** lý tưởng và chân thực nhất cho Reinforcement Learning:
1.  **Nhãn ghép sắc bén:** Việc phân nhánh `Macro_Stagnant_Bear` cho HPG khác hoàn toàn so với `Macro_Stagnant_Bear` của FPT. DRL sẽ biết được con mã nào thực sự đang "rơi tự do" để chốt lời/cắt lỗ.
2.  **Tín hiệu xác nhận Xu hướng:** Agent sẽ không còn hoang mang khi Volatility thấp (vì HMM đã tách Sideways và Bear rõ rệt bằng Momentum). Agent có thể tự tin tích lũy (Accumulate) mạnh mẽ vào các giai đoạn HMM dán nhãn `Sideways` kết hợp `Macro_Expansion`.
3.  **Tự bù trừ độ trễ:** DRL sử dụng chuỗi thời gian thông qua các mô hình Mạng nơ-ron hồi quy (LSTM/Transformer). Mọi độ trễ nhỏ sinh ra bởi lớp Persistent Filter của HMM đều sẽ được mạng Neural tự động tối ưu hóa bù trừ thông qua Action-Value Function (Q-learning).

---

## 6. GIẢI THÍCH CHI TIẾT TỪNG CELL CODE TRONG NOTEBOOK (V3 DEEP DIVE)

Dưới đây là mạch chảy logic của toàn bộ mã nguồn trong file `hmm_pipeline_dual_frequency_ticker_v3.ipynb`, giải nghĩa lý do tại sao chúng ta lại viết đoạn code đó:

### [Cell 1 - 2] Khởi tạo & Import thư viện
*   **Hành động:** Import thư viện lõi (`pandas`, `numpy`, `hmmlearn.hmm.GMMHMM`) và thiết lập thư mục đầu ra biệt lập `output/hmm_v3/`.
*   **Mục đích:** Đảm bảo toàn bộ kết quả của V3 (có bao gồm các biến Momentum) không ghi đè lên hoặc làm hỏng dữ liệu của bản V2 cũ, tạo vùng thử nghiệm an toàn (Sandbox) cho DRL.

### [Cell 3 - 5] Tiền xử lý & Kiểm định Vĩ mô (Macro Processing)
*   **Hành động:** Load file `hmm_data.csv`. Tính toán độ trễ (Lag) 1 tháng cho các biến kinh tế (CPI, Tín dụng, PMI) bằng lệnh `.shift(1)`. Chạy kiểm định rễ đơn vị (ADF, KPSS) và độ nhọn (Kurtosis).
*   **Mục đích:**
    *   Lệnh `shift(1)` là cốt lõi để **chống rò rỉ tương lai (Look-ahead bias)**. Ví dụ: Số liệu PMI tháng 5 chỉ được công bố vào tuần đầu tháng 6. Việc đẩy lùi số liệu giúp môi trường giao dịch giả lập không "nhìn lén" được báo cáo chưa phát hành.
    *   Các bài kiểm tra (ADF, KPSS) để loại bỏ các dữ liệu có tính xu hướng (Non-stationary). HMM chỉ hoạt động đúng trên dữ liệu dừng (Stationary).

### [Cell 6 - 9] Trích xuất Đặc trưng Bằng Học Máy (Feature Selection)
*   **Hành động:** Sử dụng Thuật toán Học máy **Mutual Information (MI)** để chấm điểm mức độ liên quan của các chỉ báo đối với mức biến động thị trường. Sau đó loại trừ các biến có hệ số đa cộng tuyến (VIF) > 5.0.
*   **Mục đích:** Rút gọn từ hàng chục chỉ báo xuống còn 4 chỉ báo sắc bén nhất (`rolling_vol_5`, `fx_log_ret`, `ret_disp`, `amihud_diff_normalized`). MI giỏi hơn hệ số tương quan Pearson ở chỗ nó bắt được các quy luật "Phi tuyến tính" (Non-linear) – một đặc tính nổi trội của thị trường tài chính.

### [Cell 10 - 15] Lưới tìm kiếm Tham số (Hyperparameter Grid Search)
*   **Hành động:** Định nghĩa hàm NQT. Chạy thử nghiệm HMM trên nhiều cấu hình (K=2, 3, 4) và chấm điểm bằng hệ thống **Composite Score** (Gồm điểm BIC phạt sự rườm rà, điểm Out-of-sample đánh giá dự phóng, và thời gian lưu trú State).
*   **Mục đích:** Máy tính tự động ra quyết định chọn K=2 cho Vĩ mô (Tăng trưởng/Suy thoái) và K=3 cho Cổ phiếu (Bull/Bear/Sideways) bằng toán học thuần túy, loại bỏ hoàn toàn cảm tính của con người.

### [Cell 16 - 19] Gắn nhãn Vĩ mô (Macro Labeling)
*   **Hành động:** Fit Macro HMM. Tính giá trị PMI trung bình của mỗi trạng thái. Đổi tên State 0, 1 thành `Macro_Stagnant` (Trì trệ) hoặc `Macro_Expansion` (Mở rộng).
*   **Mục đích:** Đưa ra "Biến bối cảnh" (Context Variable) cho mô hình DRL. Cổ phiếu trong pha *Macro Expansion* sẽ dễ tăng giá mạnh hơn là trong pha *Macro Stagnant*.

### [Cell 20 - 21] TRÁI TIM CỦA V3: Ticker-Specific Inference & Momentum
Đây là cụm code phức tạp và khác biệt nhất so với V1/V2:
*   **1. Tạo biến Động lượng (`mom_1M`, `dist_MA50`):** Giúp HMM mở mắt nhìn được xu hướng thay vì chỉ nhìn thấy độ biến động (Volatility).
*   **2. Áp dụng Expanding NQT:** Xếp hạng dữ liệu bằng `expanding().rank()` nhằm bảo toàn ký ức dài hạn (như đã giải thích ở Mục 4.1).
*   **3. GMMHMM `covariance_type='diag'`:** Thuật toán huấn luyện riêng một mô hình cho từng mã cổ phiếu (Ví dụ: FPT có HMM riêng, HPG có HMM riêng). Thiết lập `diag` để ngăn lỗi sụp đổ ma trận toán học khi dữ liệu của mã đó quá mỏng.
*   **4. Tự động Gắn nhãn (Dynamic Labeling):** Tính trung bình lợi suất sinh lời (`mean_ret`) của mã đó trong từng cụm HMM:
    * Cụm lợi suất thấp nhất $\rightarrow$ Dán nhãn `Bear`.
    * Trong 2 cụm còn lại, cụm có biến động thấp hơn $\rightarrow$ Dán nhãn `Sideways`.
    * Cụm cuối cùng $\rightarrow$ Dán nhãn `Bull`.
*   **5. Bộ lọc kiên định (Persistent Filter):** Chống nhiễu tín hiệu (chattering) bằng cách bắt buộc tín hiệu mới phải xuất hiện 5 phiên liên tiếp mới xác nhận, trừ phi có cú sốc biến động (Breakout) vượt mức lịch sử 80%.
*   **Mục đích tổng thể:** Trao cho DRL một bộ khung Rủi ro Tùy biến (Customized Risk Profile) sắc lẹm, không bị cào bằng như các bản cũ.

### [Cell 22 - 24] Đóng gói & Chia tách Dữ liệu (Splitting & Parquet Export)
*   **Hành động:** Lọc và tách dữ liệu thành Train (trước 2020), Validation (2020-2022) và Test (từ 2023). Lưu mảng dữ liệu vào định dạng `.parquet`.
*   **Mục đích:** `.parquet` đọc/ghi cực nhanh và lưu trữ nguyên vẹn cấu trúc dữ liệu cho Reinforcement Learning. Việc chia tách cứng theo mốc thời gian đảm bảo Agent khi được Backtest ở tập Test sẽ đối mặt với một tương lai hoàn toàn mù mịt (Chưa từng nhìn thấy), phản ánh đúng 100% năng lực giao dịch thực chiến.
