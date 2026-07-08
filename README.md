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
│   ├── e.ipynb                             # Notebook luồng HMM tích hợp & Meta-Model (Mới nhất)
│   ├── hmm_pipeline.ipynb                  # Notebook thiết lập & grid search HMM chính
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

---

## 🔧 Hướng Dẫn Cài Đặt Dữ Liệu Lõi Cho `e.ipynb` (HMM-LightGBM Meta-Model)

Thư mục dự án đã cấu hình `.gitignore` chặn tải lên các tệp dữ liệu lớn. Nếu bạn tải dự án này về và được cấp (hoặc giải nén) bộ dữ liệu mồi, hãy **chuyển 4 tệp dữ liệu đó vào đúng các vị trí sau** trước khi chạy Notebook:

1. Di chuyển tệp `hmm_data.csv` $\rightarrow$ Đặt vào thư mục: `output/`
2. Di chuyển tệp `m1_vn46.csv` $\rightarrow$ Đặt vào thư mục: `data/processed/`
3. Di chuyển tệp `m4_foreign_net_buy_sell.csv` $\rightarrow$ Đặt vào thư mục: `data/processed/`
4. Di chuyển tệp `e1_usdvnd.csv` $\rightarrow$ Đặt vào thư mục: `data/processed/`

Sau khi dán 4 tệp này vào đúng các thư mục như trên, bạn có thể mở `notebooks/e.ipynb` và chọn **Run All** một mạch từ trên xuống dưới mà không lo gặp lỗi!

---

## 🛠 Những Cập Nhật & Gỡ Lỗi Gần Đây (Recent Hotfixes)
- **Toàn vẹn Dữ liệu (Data Integrity):** Tự động loại bỏ các khoảng thời gian bị thiếu (NaN) ở 20 phiên giao dịch đầu tiên thay vì dùng `bfill` (làm giả dữ liệu) trong tập `m1_vn46.csv`.
- **Tối ưu Covariance HMM:** Cập nhật toàn bộ các mô hình (Market, Sector, Ticker) sang `covariance_type='diag'` để ngăn hiện tượng suy biến ma trận (Null Eigenvalue) do các vệt dữ liệu xác suất phẳng. Tối ưu I/O Bottleneck giúp thời gian train còn 10-12 phút.
- **Tích hợp Tqdm Progress Bar:** Bổ sung thanh tiến trình vào ô huấn luyện Ticker HMM để dễ dàng theo dõi thời gian và tốc độ đếm.
- **Fix Meta-Model (LightGBM):** Gỡ lỗi Key Error và nạp đầy đủ 100% các feature (Ticker Probs, Market Probs, Sector Probs) cho lớp mô hình Walk-Forward Validation.


## 🚀 Kiến trúc Mới Nhất: Hierarchical Dual-Frequency HMM (`f.ipynb`)
Hệ thống HMM đã được tái cấu trúc hoàn toàn trong `notebooks/f.ipynb` nhằm khắc phục lỗi Look-ahead Bias và nhiễu hàm bậc thang (Step-function noise) khi gộp dữ liệu Tháng và Ngày.
- **Tầng 1 (Macro HMM - Khung Tháng):** Xử lý độc lập dữ liệu vĩ mô theo từng tháng, xuất ra `Macro_Prob` và tịnh tiến (shift) lùi 1 tháng để áp dụng cho trading thực tế.
- **Tầng 2 (Market HMM - Khung Ngày):** Đánh giá thị trường hàng ngày dựa trên các chỉ báo kỹ thuật ngày kết hợp với `Macro_Prob` từ Tầng 1.
- **Tầng 3 (Ticker HMM):** Tích hợp cả Xác suất Vĩ mô, Xác suất Thị trường, Xác suất Dòng tiền Ngành và đặc trưng riêng của Ticker để dán nhãn trạng thái tối thượng.
