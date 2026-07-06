# Hệ Thống Đầu Tư Định Lượng Lai Ghép (HMM - TFT - Black-Litterman)

Dự án này triển khai hệ thống đầu tư định lượng đa tài sản kết hợp Mô hình Markov ẩn (HMM), mạng nơ-ron Temporal Fusion Transformer (TFT) và tối ưu hóa danh mục đầu tư Bayesian Black-Litterman cho thị trường chứng khoán Việt Nam.

---

## 📁 Cấu Trúc Thư Mục Mới (Restructured Layout)

Thư mục dự án đã được sắp xếp lại ngăn nắp để phục vụ nghiên cứu và phát triển lâu dài:

```
Kaggle/
├── config/
│   └── SBV.INR_VNM.xml                     # Cấu hình, siêu dữ liệu và tệp XML gốc
│
├── data/
│   ├── raw/                                # Dữ liệu kinh tế vĩ mô & chỉ số thô chưa căn chỉnh
│   ├── stocks/                             # Dữ liệu giá lịch sử của 100 cổ phiếu VN100
│   ├── processed/                          # Dữ liệu đặc trưng hàng ngày đã căn chỉnh lưới thời gian
│   ├── reference/                          # Các bảng Excel tham chiếu và danh sách mã (vn100.csv)
│   └── test/                               # Dữ liệu kiểm thử vĩ mô gốc
│
├── src/                                    # Mã nguồn Python (Source Code)
│   ├── data_collection/                    # Các script cào/tải dữ liệu
│   │   ├── crawl_macro_data.py
│   │   ├── crawl_macro_raw_data.py
│   │   ├── crawl_raw_stocks.py
│   │   ├── convert_sbv.py
│   │   └── restore_raw_data.py
│   └── data_processing/                    # Tiền xử lý, tạo biến đặc trưng & ghép nối
│       ├── align_daily_features.py
│       ├── derived_variable.py
│       ├── standart_features.py
│       ├── slit_date.py
│       ├── all_feature.py
│       └── process_pipeline.py             # Script ghép nối & chuẩn hóa Z-score chính
│
├── notebooks/                              # Jupyter Notebooks nghiên cứu & huấn luyện
│   ├── eda_notebook.ipynb                  # Phân tích khám phá dữ liệu (EDA) & Lead-Lag
│   ├── hmm_pipeline.ipynb                  # Notebook thiết lập & grid search HMM chính
│   ├── hmm_pipeline_hmm_data.ipynb        # Huấn luyện HMM trên tập dữ liệu đồng bộ
│   └── hmm_pipeline_macro_features.ipynb  # Huấn luyện HMM trên các biến vĩ mô
│
├── docs/                                   # Báo cáo, tài liệu nghiên cứu và biểu đồ
│   ├── project_status_report.md            # Báo cáo tiến độ dự án hiện tại
│   ├── Bao_cao_TFT_HMM_Black_Litterman.md  # Báo cáo kỹ thuật chi tiết của hệ thống
│   ├── eda_summary_report.md               # Báo cáo chi tiết kết quả EDA
│   ├── data_pipeline_design.md             # Thiết kế nhãn và luồng dữ liệu
│   ├── final_data_features_summary.md      # Tổng quan các biến đặc trưng
│   ├── macromicro_vietnam.html
│   ├── plots/                              # Các biểu đồ kết quả đầu ra
│   └── model_notes/                        # Tài liệu nghiên cứu các mô hình liên quan (XGBoost, PPO...)
│
└── output/                                 # Kết quả đầu ra của các pipeline
    ├── TFT_HMM_BL.csv                      # Tập dữ liệu đồng bộ dùng cho TFT & BL
    ├── hmm_data.csv                        # Tập dữ liệu đồng bộ dùng cho HMM
    └── v2/                                 # Model HMM đã train & kết quả trạng thái
        ├── hmm_model.pkl
        ├── hmm_regimes.csv
        └── master_drl_ready.parquet
```

---

## 🚀 Hướng Dẫn Vận Hành Quy Trình Dữ Liệu (Data Pipeline)

Tất cả các script trong thư mục `src/` đã được cập nhật đường dẫn động tương đối (relative path) tự động phát hiện thư mục gốc của dự án. Bạn có thể chạy chúng từ bất kỳ thư mục nào:

1.  **Tính toán biến dẫn xuất (Derived Variables):**
    ```bash
    python src/data_processing/derived_variable.py
    ```
    *Mục đích:* Đọc dữ liệu thô trong `data/processed/` và tính toán log return, rolling volatility, volume ratio,... ghi đè lại cột mới vào chính các file đó.

2.  **Căn chỉnh lưới ngày giao dịch (Align Daily Grid):**
    ```bash
    python src/data_processing/align_daily_features.py
    ```
    *Mục đích:* Đồng bộ hóa toàn bộ 17 chuỗi dữ liệu ngày về chính xác 2,361 dòng (dựa trên mốc giao dịch thực tế của rổ cổ phiếu).

3.  **Ghép nối và chuẩn hóa Z-score đầu ra:**
    ```bash
    python src/data_processing/process_pipeline.py
    ```
    *Mục đích:* Sử dụng phương pháp ghép nối phi đồng bộ (asynchronous merge) để gộp dữ liệu ngày và vĩ mô tháng, sau đó làm mịn Winsorize & chuẩn hóa Z-score. Kết quả lưu tại `output/TFT_HMM_BL.csv` và `output/hmm_data.csv`.
