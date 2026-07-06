# BẢN TỔNG HỢP CÁC CÔNG THỨC TOÁN HỌC TRONG CÁC MÔ HÌNH DỰ BÁO VÀ TỐI ƯU HÓA DANH MỤC

Tài liệu này tổng hợp toàn bộ các công thức toán học cốt lõi được sử dụng trong 6 mô hình: **FinBERT, GPT-2, XGBoost, Random Forest, PPO Actor-Critic, và TFT-HMM-BL**. Nội dung được giải thích chi tiết về ý nghĩa các phép toán, ký hiệu và quá trình biến đổi để người nghiên cứu dễ dàng nắm bắt.

---

## BẢNG GIẢI THÍCH KÝ HIỆU VÀ PHÉP TOÁN CƠ BẢN

| Ký hiệu / Phép toán | Tên gọi | Ý nghĩa toán học |
| :--- | :--- | :--- |
| $\odot$ | Hadamard Product (Tích Hadamard) | Phép nhân từng phần tử tương ứng của hai ma trận hoặc vectơ có cùng kích thước: $[A \odot B]_{ij} = A_{ij} \cdot B_{ij}$. |
| $\sigma(x)$ | Sigmoid Function (Hàm Sigmoid) | Hàm kích hoạt phi tuyến ánh xạ giá trị thực sang khoảng $(0, 1)$: $\sigma(x) = \frac{1}{1 + e^{-x}}$. Thường dùng làm bộ lọc thông tin (gate). |
| $\text{softmax}(x)$ | Softmax Function (Hàm Softmax) | Chuyển đổi một vectơ số thực thành một vectơ xác suất có tổng bằng 1: $\text{softmax}(x)_i = \frac{e^{x_i}}{\sum e^{x_j}}$. |
| $\mathbb{E}[x]$ hoặc $\hat{\mathbb{E}}[x]$ | Expectation (Kỳ vọng) | Giá trị trung bình kỳ vọng của biến ngẫu nhiên. Ký hiệu mũ ($\hat{}$) chỉ kỳ vọng thực nghiệm tính toán trên tập mẫu. |
| $\mathbb{I}(\text{điều kiện})$ | Indicator Function (Hàm chỉ thị) | Hàm trả về giá trị $1$ nếu điều kiện đúng, và $0$ nếu điều kiện sai. |
| $A^T$ hoặc $A^{-1}$ | Transpose & Inverse (Chuyển vị & Nghịch đảo) | - $A^T$: Đổi dòng thành cột và cột thành dòng của ma trận $A$.<br>- $A^{-1}$: Ma trận nghịch đảo thỏa mãn $A \cdot A^{-1} = I$ (ma trận đơn vị). |
| $\ln(x)$ hoặc $\log_2(x)$ | Natural & Binary Logarithm | Hàm logarit tự nhiên (cơ số $e$) hoặc cơ số $2$. |

---

## 1. TOÁN HỌC TRÊN CƠ CHẾ SỰ CHÚ Ý (ATTENTION & TRANSFORMER)
*Áp dụng trong: FinBERT, GPT-2, và Temporal Fusion Transformer (TFT)*

### 1.1. Cơ chế Tự chú ý (Self-Attention)
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
*   **Thành phần:** Ma trận Query ($Q \in \mathbb{R}^{L \times d_k}$), Key ($K \in \mathbb{R}^{L \times d_k}$), và Value ($V \in \mathbb{R}^{L \times d_v}$). Với $L$ là độ dài câu (hoặc chuỗi thời gian) và $d_k$ là kích thước vectơ đặc trưng.
*   **Ý nghĩa:** Tích vô hướng $Q K^T$ tính toán độ tương đồng (hoặc mức độ liên quan) giữa mọi từ (hoặc mọi bước thời gian) với nhau. Chia cho $\sqrt{d_k}$ để giữ phương sai của tích vô hướng bằng 1, tránh việc hàm Softmax bị bão hòa (khiến gradient tiến về 0).
*   **Sự biến đổi trong GPT-2 (Masked Attention):** Cộng thêm ma trận mặt nạ $M$ trước khi tính Softmax:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
    Trong đó $M_{ij} = 0$ nếu $i \ge j$ (quá khứ) và $-\infty$ nếu $i < j$ (tương lai). Vì $e^{-\infty} = 0$, mô hình sẽ hoàn toàn không chú ý đến các thông tin tương lai, đảm bảo tính tự hồi quy nhân quả.

### 1.2. Chuẩn hóa Lớp (Layer Normalization)
$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
*   **Ý nghĩa:** Tính trung bình $\mu$ và phương sai $\sigma^2$ trên toàn bộ các kênh đặc trưng của cùng một mẫu $x$ tại một thời điểm để chuẩn hóa dữ liệu về phân phối chuẩn có trung bình bằng 0 và phương sai bằng 1. Hệ số $\epsilon$ (rất nhỏ, cỡ $10^{-5}$) để tránh lỗi chia cho 0. $\gamma$ và $\beta$ là các tham số học được để khôi phục lại khả năng biểu diễn của mạng.

### 1.3. Gated Linear Unit (GLU) & Gated Residual Network (GRN)
*   **GLU:**
    $$\text{GLU}(\gamma) = \sigma(W_1 \gamma + b_1) \odot (W_2 \gamma + b_2)$$
    *Ý nghĩa:* Nhánh thứ nhất qua hàm Sigmoid $\sigma$ đóng vai trò là một "cánh cổng" (gate) kiểm soát tỷ lệ lượng thông tin của nhánh thứ hai được phép đi qua.
*   **GRN:**
    $$\text{GRN}(a, c) = \text{LayerNorm}(a + \text{GLU}(\eta_2))$$
    $$\eta_2 = W_3 \eta_1 + b_3, \quad \eta_1 = \text{ELU}(W_4 a + W_5 c + b_4)$$
    *Ý nghĩa:* Cho phép thông tin phi tuyến đi qua mạng một cách chọn lọc thông qua cơ chế residual connection ($a + \text{GLU}(\eta_2)$). Nếu mối quan hệ là đơn giản, cánh cổng GLU sẽ đóng lại (bằng 0), nơ-ron bỏ qua các biến đổi phi tuyến để truyền thẳng đầu vào $a$ qua LayerNorm.

---

## 2. TOÁN HỌC TRÊN CÂY QUYẾT ĐỊNH VÀ PHƯƠNG PHÁP ENSEMBLE
*Áp dụng trong: XGBoost và Random Forest*

### 2.1. Đo lường độ tinh khiết (Gini & Entropy)
*   **Gini Impurity:**
    $$\text{Gini}(D) = 1 - \sum_{i=1}^C p_i^2$$
*   **Entropy:**
    $$\text{Entropy}(D) = -\sum_{i=1}^C p_i \log_2(p_i)$$
    *Ý nghĩa:* Đánh giá mức độ hỗn loạn dữ liệu tại một nút cây quyết định. Giá trị càng nhỏ thể hiện dữ liệu phân chia càng đồng nhất về một lớp.

### 2.2. Hàm mục tiêu tối ưu hóa XGBoost
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
*   **Khai triển Taylor bậc hai:** Giúp xấp xỉ bất kỳ hàm mất mát $l(y_i, \hat{y}_i)$ nào để huấn luyện nhanh chóng.
    - $g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$ (đạo hàm bậc 1 - độ dốc sai số).
    - $h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$ (đạo hàm bậc 2 - độ cong sai số).
*   **Biến đổi tối ưu hóa lá:** Triệt tiêu đạo hàm của $\mathcal{L}^{(t)}$ theo trọng số lá $w_j$, ta thu được giá trị dự báo tối ưu tại lá thứ $j$:
    $$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

---

## 3. TOÁN HỌC TRONG HỌC TĂNG CƯỜNG (REINFORCEMENT LEARNING)
*Áp dụng trong: PPO Actor-Critic*

### 3.1. Clipped Surrogate Objective (Hàm mục tiêu xén chính sách)
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
*   **Tỷ lệ chính sách:** $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$.
*   **Ý nghĩa biến đổi:** Tránh việc cập nhật chính sách quá lớn gây đổ vỡ mô hình. Phép toán $\min$ và $\text{clip}$ hoạt động như sau:
    - Nếu lợi thế hành động dương ($\hat{A}_t > 0$), mô hình khuyến khích tăng xác suất hành động đó nhưng chặn trên tại $1+\epsilon$.
    - Nếu lợi thế hành động âm ($\hat{A}_t < 0$), mô hình giảm xác suất nhưng chặn dưới tại $1-\epsilon$.

### 3.2. Generalized Advantage Estimator (GAE)
$$\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
*   **Ý nghĩa:** Tính toán tổng lợi ích lũy kế của hành động tại thời điểm $t$ so với mức giá trị trung bình kỳ vọng được đánh giá bởi Critic $V_\phi$. Hệ số $\gamma \lambda$ chiết khấu các lợi ích ở tương lai xa để kiểm soát phương sai.

---

## 4. TOÁN HỌC TRONG MÔ HÌNH LAI GHÉP ĐỊNH LƯỢNG (TFT-HMM-BL)
*Áp dụng trong: TFT-HMM-BL*

### 4.1. Hidden Markov Model (HMM) với phân phối phát xạ Gaussian đa biến
Hàm phát xạ tại trạng thái $j$ của vectơ quan sát vĩ mô $O_t \in \mathbb{R}^d$:
$$b_j(O_t) = \frac{1}{(2\pi)^{d/2} |\Sigma_j|^{1/2}} \exp \left( -\frac{1}{2} (O_t - \mu_j)^T \Sigma_j^{-1} (O_t - \mu_j) \right)$$
*   **Ý nghĩa:** Tính toán xác suất xuất hiện của vectơ thị trường $O_t$ nếu thị trường đang ở chế độ $j$. Sử dụng ma trận nghịch đảo hiệp phương sai $\Sigma_j^{-1}$ để định tỷ lệ trọng số sai lệch của từng biến đặc trưng.

### 4.2. Xử lý tính phi dừng chuỗi thời gian (NLinear)
$$\tilde{X} = X - \mathbf{1} X_L \quad \rightarrow \quad \hat{Y} = W \tilde{X} + b \quad \rightarrow \quad \hat{Y}_{\text{NLinear}} = \hat{Y} + \mathbf{1} X_L$$
*   **Ý nghĩa biến đổi:** $\mathbf{1}$ là vectơ đơn vị. Phép toán trừ đi điểm dữ liệu cuối cùng $X_L$ chuyển đổi chuỗi thời gian thành dạng sai phân tương đối. Sau khi đi qua lớp tuyến tính để dự báo xu hướng tương lai, ta cộng ngược lại $X_L$ để trả dữ liệu về quy mô vật lý ban đầu. Phép biến đổi này triệt tiêu hiện tượng Mean Drift (trôi dạt giá trị trung bình) giữa tập Train và Test.

### 4.3. Công thức cập nhật Bayes Black-Litterman
*   **Lợi nhuận hậu nghiệm $E(R)$:**
    $$E(R) = \left[ (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1} \left[ (\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q \right]$$
*   **Hiệp phương sai hậu nghiệm $\hat{\Sigma}$:**
    $$\hat{\Sigma} = \Sigma + \left[ (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1}$$
*   **Quá trình biến đổi Bayes:**
    - Mô hình coi lợi nhuận tài sản thực tế là biến ngẫu nhiên có phân phối tiền nghiệm $R \sim \mathcal{N}(\Pi, \tau \Sigma)$.
    - Các quan điểm đầu tư là các quan sát có sai số $P \cdot R = Q + \epsilon$, với $\epsilon \sim \mathcal{N}(0, \Omega)$.
    - Áp dụng định lý Bayes để tìm phân phối xác suất hậu nghiệm của $R$ điều kiện theo các quan sát quan điểm. Kết quả là kỳ vọng hậu nghiệm $E(R)$ chính là trung bình trọng số của lợi nhuận tiền nghiệm $\Pi$ và quan điểm $Q$, được điều chỉnh theo ma trận bất định tương ứng $(\tau \Sigma)^{-1}$ và $\Omega^{-1}$.
    - Ma trận $\Omega$ được xác định động từ khoảng dự báo phân vị của TFT: $\Omega_{k,k} = c \cdot (\hat{y}^{(0.9)} - \hat{y}^{(0.1)})^2$. Khi khoảng phân vị dự báo rộng (độ bất định lớn), $\Omega^{-1}$ sẽ rất nhỏ, kéo trọng số niềm tin vào quan điểm $Q$ về gần 0, danh mục tự động quay lại trạng thái cân bằng tiền nghiệm $\Pi$.

### 4.4. Tối ưu hóa Mean-Variance (MVO) có ràng buộc
$$\max_{w} \quad w^T E(R) - \frac{\delta}{2} w^T \hat{\Sigma} w \quad \text{s.t.} \quad \mathbf{1}^T w = 1, \quad w \ge 0$$
*   **Ý nghĩa:** Tìm kiếm vectơ trọng số danh mục $w \in \mathbb{R}^N$ sao cho tối đa hóa lợi nhuận hậu nghiệm kỳ vọng $w^T E(R)$ đồng thời cực tiểu hóa rủi ro danh mục $w^T \hat{\Sigma} w$. Hệ số $\delta$ thể hiện mức độ né tránh rủi ro của nhà đầu tư. Ràng buộc $w \ge 0$ ngăn chặn hành vi bán khống (no short-selling).
