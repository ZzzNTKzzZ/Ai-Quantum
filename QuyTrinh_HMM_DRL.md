# TỔNG QUAN QUY TRÌNH HỆ THỐNG GIAO DỊCH (HMM + LightGBM + DRL)

Dưới đây là sơ đồ chi tiết toàn bộ quy trình từ lúc lấy dữ liệu ngày mới cho đến lúc ra quyết định mua/bán cổ phiếu. Quy trình này tự động chạy hằng ngày qua script `auto_update_daily.py`.

```mermaid
graph TD
    %% Định nghĩa các Style
    classDef crawl fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef hmm fill:#fff3e0,stroke:#f4511e,stroke-width:2px;
    classDef lgbm fill:#e8f5e9,stroke:#43a047,stroke-width:2px;
    classDef drl fill:#ffebee,stroke:#e53935,stroke-width:2px;
    classDef output fill:#eceff1,stroke:#546e7a,stroke-width:2px,stroke-dasharray: 5 5;

    %% 1. Tải Dữ Liệu
    subgraph Data_Pipeline [1. Data Pipeline - crawl_live_46.py & run_full_regeneration.py]
        A[Dữ liệu thô cuối ngày: <br>Giá chứng khoán, KLGD...] ::: crawl
        B[Dữ liệu Vĩ mô & Hàng hóa: <br>CPI, Tín dụng, PMI, SP500, Dầu...] ::: crawl
        C[Xử lý & Khớp dữ liệu: <br>Nội suy, Normalize, Ghép bảng] ::: process
        
        A --> C
        B --> C
    end

    %% 2. Mô hình HMM
    subgraph HMM_Pipeline [2. HMM Inference - hmm_live_inference.py]
        D[Bảng hmm_data.csv] ::: output
        C --> D
        
        D --> H1[Macro HMM <br> Nhận diện chu kỳ Kinh tế] ::: hmm
        D --> H2[Market HMM <br> Nhận diện pha VNINDEX] ::: hmm
        D --> H3[Sector HMM <br> Nhận diện dòng tiền Ngành] ::: hmm
        D --> H4[Ticker HMM <br> Nhận diện sóng Cổ phiếu] ::: hmm
        
        H1 --> E[master_drl_ready_ticker.parquet] ::: output
        H2 --> E
        H3 --> E
        H4 --> E
    end

    %% 3. LightGBM
    subgraph LightGBM_Pipeline [3. Meta Classifier - live_trading.py]
        E --> L1[Load các Xác suất HMM: <br>Macro, Market, Sector, Ticker] ::: lgbm
        L1 --> L2[Train LightGBM nhanh <br>trên dữ liệu lịch sử đến T-1] ::: lgbm
        L2 --> L3[Inference cho phiên T <br>Tính 'Xác Suất Tăng' cho T+1] ::: lgbm
        L3 --> F[live_trading_signals_YYYYMMDD.csv <br> Top Cổ Phiếu Khuyên Mua] ::: output
    end

    %% 4. PPO Agent
    subgraph DRL_Pipeline [4. Portfolio Allocation - drl_live_trading.py]
        E --> P1[Trích xuất Features <br>& Technical Indicators] ::: drl
        F -.->|Tham khảo| P1
        P1 --> P2[Môi trường Giao Dịch <br>AdvancedPortfolioEnv] ::: drl
        P2 --> P3[PPO Meta-Agent <br>Cross-Ticker Attention] ::: drl
        P3 --> G[drl_target_weights_YYYYMMDD.csv <br> Tỷ trọng % Khuyến nghị T+1] ::: output
    end
```

### Diễn giải Chi Tiết

**Bước 1: Data Pipeline (`crawl_live_46.py` & `run_full_regeneration.py`)**
1. **Lấy dữ liệu:** Cuối mỗi ngày, hệ thống gọi API để lấy dữ liệu giá/khối lượng của 46 mã cổ phiếu, dữ liệu hàng hóa toàn cầu (vàng, dầu, đồng), và dữ liệu kinh tế vĩ mô (CPI, tín dụng).
2. **Xử lý:** Ghép nối các dữ liệu khác khung thời gian (ngày, tháng) thông qua nội suy (forward fill) và tính toán các đặc trưng cơ sở (rolling_vol, volume_ratio).
3. **Đầu ra:** File `hmm_data.csv`.

**Bước 2: Hệ thống phân tích trạng thái thị trường (`hmm_live_inference.py`)**
Dùng file `hmm_data.csv` đẩy qua 4 tầng mô hình HMM độc lập:
1. **Macro HMM:** Tìm hiểu xem nền kinh tế đang ở chu kỳ nào (Phục hồi, Tăng trưởng, Suy thoái).
2. **Market HMM:** Xem thị trường chứng khoán (VNINDEX) đang Tăng, Giảm hay Đi ngang.
3. **Sector HMM:** Xác định dòng tiền đang chảy vào Nhóm ngành nào (Ngân hàng, Bất động sản, Bán lẻ,...).
4. **Ticker HMM:** Nhận diện đồ thị của từng cổ phiếu cụ thể.
*Tất cả xác suất dự đoán (Probabilities) của 4 lớp này sẽ được gộp lại thành một bảng dữ liệu siêu khổng lồ: `master_drl_ready_ticker.parquet`.*

**Bước 3: Màng lọc Machine Learning (`live_trading.py`)**
1. Nạp cái bảng khổng lồ kia vào mô hình **LightGBM**. 
2. Mô hình này sẽ phân tích các xác suất để trả lời câu hỏi: *"Với bối cảnh Vĩ mô này, Ngành này, Cổ phiếu này... thì ngày mai mã nào có khả năng tăng giá cao nhất?"*
3. **Đầu ra:** Bảng xếp hạng Top 15 cổ phiếu tiềm năng nhất (`live_trading_signals.csv`).

**Bước 4: Quyết định Phân bổ Vốn DRL (`drl_live_trading.py`)**
1. Mạng Neural Network **PPO Agent** (AI Reinforcement Learning) đọc toàn bộ thông số từ bảng khổng lồ.
2. Nó giả lập mô phỏng giao dịch trong vòng 252 ngày vừa qua (warm-up) để làm nóng bộ nhớ danh mục.
3. Dựa trên thuật toán *Cross-Ticker Attention* (so sánh các mã cổ phiếu với nhau), AI sẽ cân đo đong đếm rủi ro và ra quyết định **chia tỷ lệ tiền** để mua mã nào nhiều, mua mã nào ít.
4. **Đầu ra:** Danh sách tỷ trọng cụ thể cho ngày mai (`drl_target_weights.csv`). Ví dụ: HAG 26%, PDR 20%...
