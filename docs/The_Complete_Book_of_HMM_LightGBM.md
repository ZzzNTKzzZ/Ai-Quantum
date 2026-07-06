# CUỐN SÁCH TOÀN TẬP: HỆ THỐNG GIAO DỊCH ĐỊNH LƯỢNG HMM & LightGBM
*(Mastering the Macro-Sector Hierarchical HMM & Meta-Classifier Pipeline - Dựa trên `d.ipynb`)*

Chào mừng bạn đến với tài liệu giải phẫu toàn diện nhất về Hệ thống Giao dịch Định lượng Đa tầng. Tài liệu này được thiết kế như một **cuốn sách giáo khoa thu nhỏ**, giúp bất kỳ nhà nghiên cứu hay nhà giao dịch nào cũng có thể hiểu sâu sắc triết lý, toán học và luồng code đằng sau hệ thống.

---

## CHƯƠNG 1: TRIẾT LÝ GIAO DỊCH - TOP-DOWN & BOTTOM-UP
Thay vì chỉ cắm mặt vào biểu đồ nến của một cổ phiếu để dự đoán, hệ thống này hoạt động theo triết lý của các Quỹ phòng hộ (Hedge Funds):
1. **Nhìn Trời (Market Regime):** Bão vĩ mô đang kéo tới hay trời đang quang mây tạnh? Nếu thị trường rớt thảm, mọi cổ phiếu đều sẽ chết.
2. **Nhìn Đất (Sector Regime):** Dòng tiền thông minh (Smart Money) đang trú ẩn ở nhóm Ngành nào? Bất động sản hay Ngân hàng?
3. **Nhìn Cây (Micro Ticker):** Cổ phiếu này có nội lực đủ mạnh để đón dòng tiền hay không? 

Hệ thống số hóa 3 triết lý trên thông qua Mô hình xác suất **Hidden Markov Model (HMM)** để đánh giá "Trời" và "Đất", sau đó dùng Trí tuệ nhân tạo **LightGBM** để soi "Cây".

---

## CHƯƠNG 2: CƠ SỞ TOÁN HỌC & ĐẶC TRƯNG DỮ LIỆU (FEATURES)

### 2.1. Biến Mục Tiêu (Target)
Hệ thống không dự đoán ngày mai cổ phiếu tăng cụ thể bao nhiêu % (rất dễ sai số), mà chuyển bài toán về Dự báo Phân loại (Classification):
*   `target_return_1d`: Lợi suất sinh lời của ngày T+1.
*   **`target_bin`**: Biến nhị phân. Nếu giá ngày mai TĂNG (`target_return_1d > 0`) -> `target_bin = 1`. Nếu giá Giảm/Đứng im -> `target_bin = 0`. AI sẽ học cách cược xem ngày mai ra 1 hay 0.

### 2.2. Đặc trưng Đầu vào (Input Features)
*   **Vĩ mô (Macro):**
    *   `credit_growth_mom`: Động lượng bơm tiền tín dụng. Dòng máu của chứng khoán.
    *   `cpi_mom`: Tốc độ Lạm phát.
    *   `fnb_ratio`: Tỷ lệ khối ngoại mua ròng / Biến động tỷ giá.
*   **Cổ phiếu (Micro):**
    *   `rolling_vol_5`, `rolling_vol_20d`: Độ giật (biến động) của giá trong 5 và 20 ngày qua.
    *   `mom_1M` (Momentum): Tốc độ tăng tốc của giá trong 1 tháng.
    *   `dist_MA50`: Mức độ chênh lệch của giá so với trung bình 50 ngày (Đo lường sự hưng phấn ngắn hạn).

### 2.3. Sàng lọc Dữ liệu
*   **Tính Dừng (Stationary - ADF / KPSS Tests):** Thuật toán chỉ dự đoán được những thứ có tính chu kỳ (dao động quanh 1 trục). ADF test kiểm tra xem dữ liệu có bị "trôi" vô định không. Nếu trôi, ta phải tính Lợi suất Log (Log-return) để kéo nó về trạng thái "Dừng".
*   **Kurtosis (Độ nhọn):** Cảnh báo cho thuật toán biết thị trường này hay có "Thiên nga đen" (Cú sập thảm khốc) để nó nới rộng vùng rủi ro.

### 2.4. Khử Nhiễu (NQT)
*   **NQT (Nonparametric Quantile Transform):** Ép mọi phân phối méo mó về lại Phân phối Chuẩn (Hình quả chuông).
*   **Cửa sổ trượt (Rolling Window):** Việc trượt trong 252 ngày (1 năm) giúp hệ thống "quên" đi quá khứ xa và thích nghi với thực tại mới. 

---

## CHƯƠNG 3: HMM - BẮT MẠCH CHU KỲ (MARKET & SECTOR)

HMM (Mô hình Markov Ẩn) giả định rằng thị trường bị chi phối bởi các "Thế lực vô hình" (Bò, Gấu). Ta không nhìn thấy thế lực đó, ta chỉ thấy Giá và Khối lượng nhảy múa. HMM sẽ dùng toán học để lật mặt thế lực này.

### 3.1. Lớp Vĩ Mô (Market Regime)
*   **Cách hoạt động:** HMM được train DUY NHẤT trên dữ liệu của VN-Index. Nó đẻ ra các xác suất Vĩ mô (VD: 80% là Bão, 20% là Nắng).
*   **Sự hiểu lầm phổ biến:** Nhiều người tưởng HMM lấy dữ liệu của mã FPT để dự đoán Vĩ mô. SAI! Nó dự đoán Vĩ mô bằng VN-Index, sau đó dùng lệnh "Broadcast" để áp đặt chung con số 80% Bão đó làm "Thời tiết chung" cho tất cả các cổ phiếu trong ngày hôm đó. Mọi mã đều phải chịu chung bầu trời Vĩ mô.

### 3.2. Lớp Nhóm Ngành (Sector Regime)
*   **Khác biệt:** Thay vì dùng VN-Index, ta tạo một "VN-Index thu nhỏ" cho Ngành Thép (bằng cách lấy trung bình giá HPG, HSG, NKG...). Sau đó chạy HMM trên Proxy này. Do đó, Vĩ mô có thể đang đi ngang, nhưng Ngành Thép lại đang "Bull" đón dòng tiền.

### 3.3. Tối ưu số lượng Trạng thái (Grid Search)
Hệ thống không cố định thị trường phải có 3 hay 4 trạng thái. Nó cho máy tính tự thi thố và chấm điểm bằng hàm `Composite`:
*   **BIC (Bayesian Information Criterion):** Phạt sự rườm rà. Chống mô hình "học vẹt" (Overfitting).
*   **ll_oos (Out-Of-Sample):** Trọng số cao nhất. Đo lường khả năng dự đoán đúng ở khoảng thời gian tương lai mà nó chưa từng được thấy.
*   **min_dur (Minimum Duration):** Trạng thái phải duy trì đủ lâu (VD > 10 ngày) để tránh bị nhiễu T+3 lướt sóng.

### 3.4. Dán nhãn (Semantic Labeling)
Toán học nhả ra các cục `State 0, State 1`. Hệ thống dùng thuật toán **Linear Sum Assignment** để chấm điểm: State nào Lợi suất Cao nhất + Biến động Thấp nhất -> Cấp cho cái tên `Bull` (Bò Tót). Ngược lại là `Bear` hoặc `Crisis`. Giúp ngôn ngữ hóa toán học.

---

## CHƯƠNG 4: TRÍ TUỆ NHÂN TẠO (LightGBM)

Phần hồn của hệ thống nằm ở HMM, phần trí tuệ thực thi nằm ở **Meta-Classifier (LightGBM)**. Nó tổng hợp Thời tiết Vĩ mô, Sóng Ngành, và Xung lực Cổ phiếu để cược xem `target_bin` ngày mai là 1 hay 0. Đẻ ra con số `final_meta_pred_prob` (Xác suất sinh lời).

### 4.1. Chế độ Backtest (Walk-Forward)
Sự tàn khốc của AI tài chính là "Nhìn trộm tương lai" (Look-ahead bias). Walk-forward chống lại điều đó:
*   Vòng lặp tịnh tiến: Cắt từ 01/01/2022. Tại ngày $T$, AI bị bịt mắt, chỉ được đọc sách giáo khoa từ ngày 1 đến $T-1$. Sau đó ép nó nhả cược cho ngày $T$.
*   Chạy xong, ngày hôm sau lại học lại từ đầu. Đảm bảo Báo cáo lợi nhuận in ra là sự thật 100%.

### 4.2. Chế độ Live Trading (Dự báo Thực chiến)
*   Đơn giản và uy lực: Gom toàn bộ dữ liệu lịch sử tính đến hôm qua làm tập Train. 
*   Lấy duy nhất thông tin đóng cửa chiều nay làm tập Test.
*   Nhả ra danh sách Xếp hạng các mã có xác suất tăng mạnh nhất cho sáng ngày mai.

---

## CHƯƠNG 5: HẬU KIỂM VÀ ĐỌC BIỂU ĐỒ

### 5.1. Thống kê Tài chính (Financial Metrics)
Báo cáo của Máy học (ROC-AUC) không mua được bánh mì. Ta phải dịch sang Tiền.
*   **Luật Giả lập:** Ngày T xác suất `prob > 0.5` -> Ngày T+1 xuống tiền mua và ăn trọn `target_return_1d`. Dưới 0.5 thì bán ôm tiền mặt.
*   **Sharpe Ratio:** Con số vĩ đại của Phố Wall. Đánh đổi 1 phần rủi ro lấy mấy phần lợi nhuận? (Lớn hơn 1 là xuất sắc).
*   **Max Drawdown:** Từ đỉnh cao nhất, tài khoản bị chia tài sản bao nhiêu phần trăm? Giúp Trader chuẩn bị tâm lý.

### 5.2. Cách đọc Biểu đồ 4 lớp
Bảng màu tuân theo chuẩn Tâm lý học Tài chính (Xanh-Tăng, Đỏ-Giảm, Tím-Khủng hoảng, Xám-Tích lũy):
*   **Biểu đồ 1 (Market):** Nền màu là Vĩ Mô HMM, đường vẽ là VN-Index. Trực quan hóa bão tố.
*   **Biểu đồ 2 (Sector):** Nền màu là Nhóm Ngành HMM, đường vẽ là Mã Cổ phiếu. Để xem con mã của bạn có đang lội ngược dòng ngành hay không.
*   **Biểu đồ 4 (Xác suất):** Đường Ranh giới 0.5. Nếu vệt màu cắm xuống Đỏ (dưới 0.5), hãy nhấc tay khỏi bàn phím và chốt lời.

---
*Tài liệu này được biên soạn như một cuốn Cẩm nang (Playbook) để bảo trì, nâng cấp, và đào tạo (bao gồm cả đào tạo cho tác nhân Học tăng cường - DRL) trên môi trường Notebook `d.ipynb` của dự án.*
