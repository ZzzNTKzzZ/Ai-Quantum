# GIẢI PHẪU CHI TIẾT & ĐÍNH CHÍNH QUY TRÌNH HOẠT ĐỘNG CỦA d.ipynb
*(Dựa trên góc nhìn của bạn và bổ sung Phân tích Chuyên sâu)*

Chào bạn, tư duy tổng thể của bạn về việc "Phân cấp rủi ro: Vĩ mô -> Ngành -> Cổ phiếu riêng lẻ" là **hoàn toàn chính xác và rất bén**. Đó chính là linh hồn của hệ thống. 

Tuy nhiên, trong các bước đi sâu vào thuật toán, có một số chỗ bạn đang bị nhầm lẫn giữa các kỹ thuật (ví dụ SHAP không hề tồn tại trong code này, hoặc sự nhầm lẫn về cách Market Regime áp lên Ticker). 

Dưới đây là một bản phân tích chi tiết, giải nghĩa công thức và đính chính lại toàn bộ 10 bước của bạn:

---

## BƯỚC 1: Nguồn Dữ liệu và Đặc trưng (Features)
Bạn cần biết các dữ liệu đó là gì? Đây là 3 nhóm cốt lõi:
1. **Macro (Kinh tế Vĩ mô):** `credit_growth_mom` (Tăng trưởng Tín dụng - đại diện cho thanh khoản nền kinh tế), `cpi_mom` (Lạm phát - kẻ thù của chứng khoán), `fnb_ratio` (Tỷ lệ mua ròng khối ngoại / Tỷ giá).
2. **Market (Vĩ mô Chứng khoán):** `vnindex_close` (Giá đóng cửa Index), `vnindex_vol20` (Độ biến động 20 ngày của thị trường chung).
3. **Micro (Cổ phiếu Ticker):** OHLCV, `log_return` (Lợi suất), `mom_1M` (Đà tăng trưởng giá 1 tháng), `dist_MA50` (Khoảng cách từ giá hiện tại đến đường trung bình 50 ngày).

---

## BƯỚC 2: Bộ lọc Kỹ thuật - Tính Dừng (Stationary)
Để HMM học được, dữ liệu không được trôi đi vô định (Trending) mà phải dao động quanh một trục (Stationary).
*   **ADF Test (Augmented Dickey-Fuller):** 
    *   *Ý nghĩa:* Kiểm tra xem dữ liệu có "Dừng" không. 
    *   *Công thức:* $\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \dots$ Nếu hệ số $\gamma < 0$ (p-value < 0.05) $\rightarrow$ Dữ liệu là Dừng.
*   **KPSS Test:** Ngược lại với ADF, kiểm tra xem dữ liệu có Xu hướng (Trend) ẩn không.
*   **Kurtosis (Độ nhọn):** 
    *   *Ý nghĩa:* Thị trường tài chính hay có "Thiên nga đen" (đuôi béo - fat tails). Kurtosis > 3 nghĩa là dữ liệu này chứa nhiều cú sập hầm bất ngờ hơn phân phối chuẩn. Nhận biết để HMM dùng cấu hình Gaussian phù hợp.

---

## BƯỚC 3: Chuẩn hóa NQT + Rolling Window 1 Năm
*   **Bạn nói:** Xếp hạng trong khung 1 năm để đánh giá mốc gần nhất nhưng vẫn nhớ trạng thái. Nó có lợi ích không và 1 năm có phù hợp không?
*   **Đính chính & Chuyên sâu:** Bạn hiểu **ĐÚNG** về cơ chế trượt (Rolling). Dùng 252 ngày (1 năm) có lợi ích là: Nó giúp mô hình thích nghi với "Bình thường mới". Ví dụ năm 2021 tăng nóng, thì sang 2022 nó sẽ quên đi sự nóng đó để chuẩn hóa lại.
*   **Tuy nhiên (Nhược điểm):** Rolling window làm mất "Thang đo tuyệt đối". Một cú sập nhẹ -2% trong một năm đi ngang yên bình sẽ bị đẩy thứ hạng (Rank) lên chót vót thành "Cú sốc thảm họa nhất năm", khiến HMM gán nhầm sang Crisis.
*   *Phù hợp không?* Với Ticker thì dùng 1 năm là tạm ổn để bắt sóng ngắn. Nhưng với Vĩ mô (Market), người ta chuộng dùng **Expanding Window** (Tính rank từ ngày đầu tiên lên sàn đến hiện tại) để mô hình không bao giờ quên những cuộc Đại Suy Thoái thực sự.

---

## BƯỚC 4: Mutual Information (MI) và VIF
*   **SỰ HIỂU LẦM (Đính chính):** Bạn nhắc đến SHAP và thiết lập sẵn 3 trạng thái Bull/Bear/Sideway. Ở bước này **KHÔNG CÓ SHAP** và **không thiết lập trước trạng thái**.
*   **Mutual Information (MI):**
    *   *Ý nghĩa:* Đo lường lượng thông tin (Entropy). Nó đo xem: Nếu tôi biết trước biến X (ví dụ Lạm phát), thì sự mơ hồ của tôi về biến Y (Lợi suất VN-Index) có giảm đi không?
    *   *Công thức:* $I(X; Y) = \sum \sum p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right)$
    *   *Lợi ích:* MI mạnh hơn Tương quan Pearson ở chỗ nó bắt được cả mối quan hệ đường cong, zích zắc (Phi tuyến tính).
*   **VIF (Variance Inflation Factor):** Đo độ đa cộng tuyến. VIF > 5 nghĩa là biến này có thể được suy ra từ biến khác $\rightarrow$ Bị trùng lặp $\rightarrow$ Loại bỏ để AI không bị tẩu hỏa nhập ma.

---

## BƯỚC 5: Grid Search - Tìm số Trạng thái (K) tối ưu
HMM không biết thị trường có mấy pha. Chúng ta bắt nó chạy thử K từ 2 đến 5 và chấm điểm bằng hàm:
`Composite = 0.3 * Rank_bic + 0.5 * Rank_oos + 0.2 * Rank_min_dur`
*   **BIC (Bayesian Information Criterion):** Chỉ số thống kê. Công thức: $BIC = -2 \ln(\hat{L}) + p \ln(N)$. Nó đo lường sự trùng khớp của mô hình với dữ liệu, NHƯNG sẽ phạt rất nặng nếu mô hình quá phức tạp (p lớn). Chống Overfitting.
*   **ll_oos (Out-Of-Sample Log-Likelihood):** Đúng như bạn nói, nó là khả năng dự báo. Ta giấu 1/3 dữ liệu cuối đi, train mô hình trên 2/3 đầu, rồi ốp sang 1/3 cuối. Mô hình nào có xác suất Log (Log-likelihood) cao nhất trên tệp bị giấu $\rightarrow$ Mô hình đó giỏi dự phóng tương lai nhất.
*   **min_dur (Minimum Duration):** Trạng thái kinh tế không thể nay Bull mai Bear. Min_dur tính số ngày trung bình thị trường lưu lại ở 1 State. Phải > 10 ngày mới được coi là ổn định.

---

## BƯỚC 6: Mapping Labeling (Dán nhãn Ngữ nghĩa)
*   **Tại sao phải labeling?** Vì toán học HMM chỉ nhả ra `State 0, State 1`. Nó câm điếc không biết đó là tăng hay giảm.
*   **Rule hoạt động:** Thuật toán (Linear Sum Assignment) tính trung bình **Lợi suất** và **Độ biến động** của từng State trong lịch sử.
    *   State nào Lợi suất Cao nhất + Biến động Thấp nhất $\rightarrow$ Định nghĩa là `Bull`.
    *   State nào Lợi suất Âm nặng + Biến động Cực đại $\rightarrow$ Định nghĩa là `Crisis`.
    *   State nằm giữa $\rightarrow$ `Sideways`.
*   *Ý nghĩa:* Chuyển ngôn ngữ máy thành ngôn ngữ của nhà giao dịch.

---

## BƯỚC 7: Suy luận trạng thái cho từng mã (NHẦM LẪN LỚN NHẤT)
*   **Bạn nói:** "Đoán regime market bằng cách áp chỉ số riêng của từng mã vào".
*   **Đính chính Khẩn cấp:** Điều này là sai kiến trúc. **Market Regime (Vĩ mô) KHÔNG BAO GIỜ bị ảnh hưởng bởi dữ liệu của 1 cổ phiếu.**
*   **Cách thức thực sự:** 
    1. VN-Index sinh ra các xác suất: `prob_market_Bear=0.8`, `prob_market_Bull=0.2`.
    2. Trong code có dòng lệnh: `master_ticker = master_ticker.merge(df_market_hmm, on='time')`.
    3. Đây là lệnh **Broadcast (Phân phối)**. Vào ngày 10/10, Vĩ mô đang là Bear, thì tự động FPT, HPG, BID vào ngày 10/10 đều bị áp đặt (nhận chung) con số `prob_market_Bear=0.8` này làm bối cảnh môi trường của chúng.

---

## BƯỚC 8: Lớp Nhóm Ngành (Sector Regime)
*   **Bạn nói đúng:** Lặp lại y hệt công nghệ của Vĩ mô.
*   **Sự khác biệt:**
    *   Thay vì VN-Index, ta tạo một "VN-Index thu nhỏ" cho từng ngành. Ví dụ: Lấy trung bình giá và biến động của VCB, BID, CTG để gộp thành **Proxy của Ngành Bank**.
    *   *Tại sao khác?* Ngành Thép biến động khủng khiếp thuật toán tự tìm ra K=4. Ngành Điện nước bình ổn thuật toán chốt K=2. Dòng tiền đánh xoay vòng, có thể Vĩ mô Sideways nhưng Ngành Bank lại đang Bull.

---

## BƯỚC 9: Áp dụng AI - Meta Classifier (LightGBM)
*   **LightGBM là gì?** Nền tảng của nó là Gradient Boosting (Cây quyết định tăng cường). Nó mọc ra hàng trăm cái cây, cây sau sẽ phân tích cái sai của cây trước để bù đắp vào. Nhận input là toàn bộ Prob_Market, Prob_Sector, Momentum. Đẻ ra Output là: Xác suất ngày mai tăng giá (0 -> 1).
*   **9.1 Backtest (Walk-forward):**
    Thuật toán tịnh tiến từng ngày. Để dự đoán ngày $T$, AI bị bịt mắt không cho nhìn gì ngoài dữ liệu từ $1 \rightarrow T-1$. Chạy xong lưu lại. Hôm sau Train lại từ đầu. Chống rò rỉ dữ liệu tương lai tuyệt đối.
*   **9.2 Live Trading:**
    Gom sạch sành sanh toàn bộ lịch sử loài người đến chiều hôm nay ném cho AI Train 1 phát duy nhất. Xong áp dữ liệu hôm nay vào để bắt nó nhả ra mã cược cho **sáng ngày mai**.

---

## BƯỚC 9.3: Thống kê Backtest (Financial Metrics)
*   **Backtest là gì:** Từ xác suất của 9.1, mô phỏng việc bạn cầm tiền thật đi mua bán: Nếu `prob > 0.5` thì All-in, `prob < 0.5` thì bán ôm tiền mặt.
*   **Benchmark:** So sánh với việc mua rổ cổ phiếu rồi vứt chìa khóa ngủ quên 3 năm. (Chiến lược Buy & Hold).
*   **Chỉ số:**
    *   *Sharpe Ratio:* $\frac{R_p - R_f}{\sigma_p}$. Đánh đổi 1 đồng rủi ro lấy mấy đồng lãi? Sharpe > 1 là Thần thánh.
    *   *Max Drawdown:* Mức chia tài khoản đau đớn nhất từ đỉnh. Bạn x2 tài khoản mà Drawdown 80% thì bạn đã nhảy lầu trước khi x2 rồi.
*   **Cách cải thiện:** Đưa thêm phí giao dịch (Commission), trượt giá (Slippage) và Thuật toán tối ưu hóa danh mục (Kelly, Markowitz) vào thay vì rải đều vốn.

---

## BƯỚC 10: Biểu đồ & Lưu Kết Quả
Bảng màu Tâm lý học Hành vi (Behavioral Finance):
*   **Biểu đồ 1 (Market):** Nền là Vĩ mô VN-Index, đường vẽ nét đen là VN-Index. Xem Vĩ mô đang ở pha nào.
*   **Biểu đồ 2 (Sector):** Nền là Dòng tiền Ngành, đường vẽ đen là Giá Cổ Phiếu Cụ Thể. (Trả lời câu hỏi: Cổ phiếu này đang đồng pha hay ngược pha với ngành?).
*   **Biểu đồ 4 (Xác suất):** Đường ranh giới sinh tử 0.5. Màu xanh là cược Mua, Màu Đỏ là xả hàng chạy trốn. Mức độ tô đậm tương ứng độ tự tin của LightGBM.
